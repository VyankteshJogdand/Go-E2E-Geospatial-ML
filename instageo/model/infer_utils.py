# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""Utils for Running Inference."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import numpy as np
import rasterio
import torch
from codecarbon import EmissionsTracker
from pytorch_lightning import LightningModule
from rasterio.transform import from_bounds
from torch.utils.data import DataLoader
from tqdm import tqdm

from instageo.model.utils import get_carbon_info

logger = logging.getLogger(__name__)
JOINED_PREDICTION_FILENAME = "joined_prediction.tif"
# Overlap of 25% of chip size for seamless blending between tiles
OVERLAP_FACTOR = 0.25


def save_prediction(
    prediction: np.ndarray,
    output_folder: str,
    *,
    profile: Optional[Dict[str, Any]] = None,
    file_name: Optional[str] = None,
    bounds: Optional[tuple] = None,
    crs: Any = None,
    is_segmentation: Optional[bool] = None,
) -> Optional[str]:
    """Save prediction array to GeoTIFF.

    Per-chip: provide profile and file_name. Writes output_folder/<derived_basename>.
    Returns None.

    Joined (single mosaic): provide bounds (min_x, min_y, max_x, max_y), crs, and
    is_segmentation. Writes output_folder/joined_prediction.tif. Returns path.
    """
    if bounds is not None and crs is not None and is_segmentation is not None:
        # Joined mode: one file, transform from bounds
        min_x, min_y, max_x, max_y = bounds
        height, width = prediction.shape
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": rasterio.int8 if is_segmentation else rasterio.float32,
            "crs": crs,
            "transform": transform,
            "nodata": -1,
        }
        out_path = os.path.join(output_folder, JOINED_PREDICTION_FILENAME)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(prediction, 1)
        return out_path
    if profile is not None and file_name is not None:
        # Per-chip mode
        output_basename = os.path.basename(file_name).replace("chip", "prediction")
        output_file_path = os.path.join(output_folder, output_basename)
        with rasterio.open(output_file_path, "w", **profile) as dst:
            dst.write(prediction, 1)
        return None
    raise ValueError(
        "Either (bounds, crs, is_segmentation) or (profile, file_name) must be provided"
    )


def _start_tracker() -> Optional[EmissionsTracker]:
    """Initialize carbon emissions tracker."""
    try:
        tracker = EmissionsTracker(measure_power_secs=5, tracking_mode="machine", log_level="error")
        tracker.start()
        return tracker
    except Exception as e:
        logger.warning(
            "Carbon emissions tracker could not be started; CO2 metrics will be unavailable. "
            "Reason: %s: %s",
            type(e).__name__,
            e,
        )
        return None


def _stop_tracker(tracker: Optional[EmissionsTracker]) -> Dict:
    """Stop tracker and return carbon information."""
    if tracker is None:
        return {}
    try:
        tracker.stop()
        emissions_data = tracker._prepare_emissions_data()
        return get_carbon_info(emissions_data)
    except Exception as e:
        logger.warning(
            "Failed to collect carbon emissions data. Reason: %s: %s", type(e).__name__, e
        )
        return {}


def _collect_tiles(
    dataloader: DataLoader, device: str, model: LightningModule
) -> tuple[list[tuple[Any, ...]], list[Any], bool, int, int, Optional[torch.Tensor]]:
    """Collect input tiles and metadata from dataloader.

    Performs a probe forward pass on the first chip to determine task type
    (segmentation vs regression) and number of classes.

    Returns:
        Tuple of (all_inputs, all_masks, is_segmentation, chip_size, num_classes, probe_logits)
        probe_logits is the forward pass result from the first chip (for reuse in single-chip case)
    """
    all_inputs: list[tuple[Any, ...]] = []  # (input_np, file_name, profile, bounds, transform)
    all_masks: list[Any] = []
    is_segmentation: Optional[bool] = None
    chip_size: Optional[int] = None
    num_classes = 1
    probe_logits: Optional[torch.Tensor] = None

    with torch.no_grad():
        for (data, _), file_names, nan_mask in tqdm(dataloader, desc="Loading input tiles"):
            data = data.to(device)
            if is_segmentation is None:
                logits_one = model(data[:1])
                is_segmentation = logits_one.shape[1] > 1
                num_classes = int(logits_one.shape[1]) if logits_one.shape[1] > 1 else 1
                probe_logits = logits_one  # Save for potential reuse

            data_np = data.cpu().numpy()
            nan_mask = np.all(nan_mask, axis=1).astype(bool)

            for i, file_name in enumerate(file_names):
                with rasterio.open(file_name) as src:
                    if chip_size is None:
                        chip_size = src.width
                    profile = src.profile.copy()
                    bounds = src.bounds
                    transform = src.transform
                mask = np.asarray(nan_mask[i], dtype=bool)
                all_inputs.append((data_np[i], file_name, profile, bounds, transform))
                all_masks.append(mask)

    if not all_inputs:
        raise ValueError("No chips found in dataloader")

    if is_segmentation is None:
        raise RuntimeError(
            "Failed to determine is_segmentation from dataloader. "
            "This should not happen if chips were loaded successfully."
        )
    if chip_size is None:
        raise RuntimeError(
            "Failed to determine chip_size from dataloader. "
            "This should not happen if chips were loaded successfully."
        )

    return all_inputs, all_masks, is_segmentation, chip_size, num_classes, probe_logits


def _handle_single_chip(
    all_inputs: list,
    all_masks: list,
    is_segmentation: bool,
    model: LightningModule,
    device: str,
    output_folder: str,
    probe_logits: Optional[torch.Tensor] = None,
) -> None:
    """Process and save a single chip as joined prediction.

    Args:
        all_inputs: List of input tuples containing (data, filename, profile, bounds, transform)
        all_masks: List of NaN masks for each chip
        is_segmentation: Whether this is a segmentation task
        model: The model to use for inference
        device: Device to run inference on
        output_folder: Directory to save predictions
        probe_logits: Optional pre-computed logits from the probe pass to avoid redundant inference
    """
    (data_np, _, profile, bounds, _) = all_inputs[0]
    mask = all_masks[0]

    # Reuse probe logits if available (avoids redundant forward pass)
    if probe_logits is not None:
        logger.info("Reusing probe logits for single chip (skipping redundant forward pass)")
        logits = probe_logits
    else:
        data_t = torch.from_numpy(data_np).to(device)
        data_t = data_t.unsqueeze(0)
        with torch.no_grad():
            logits = model(data_t)

    if is_segmentation:
        pred = np.argmax(logits.cpu().numpy().squeeze(0), axis=0).astype(np.int8)
    else:
        pred = logits.cpu().numpy().squeeze().astype(np.float32)

    pred = np.where(mask, -1, pred)

    crs = profile["crs"]
    bounds_tuple = (bounds.left, bounds.bottom, bounds.right, bounds.top)
    save_prediction(
        pred, output_folder, bounds=bounds_tuple, crs=crs, is_segmentation=is_segmentation
    )
    logger.info("Single chip with stitching requested: saved as joined prediction (no blending)")


def _calculate_merged_extent(
    all_inputs: list, chip_size: int
) -> tuple[float, float, float, float, int, int, float, float]:
    """Calculate merged extent and dimensions.

    Returns:
        Tuple of (min_x, min_y, max_x, max_y, image_height, image_width, pixel_size_x, pixel_size_y)
    """
    min_x = min(info[3].left for info in all_inputs)
    min_y = min(info[3].bottom for info in all_inputs)
    max_x = max(info[3].right for info in all_inputs)
    max_y = max(info[3].top for info in all_inputs)

    ref_transform = all_inputs[0][4]
    pixel_size_x = abs(ref_transform[0])
    pixel_size_y = abs(ref_transform[4])

    chip_rows = []
    chip_cols = []
    for _, _, _, bounds, _ in all_inputs:
        row_start = int((max_y - bounds.top) / pixel_size_y)
        col_start = int((bounds.left - min_x) / pixel_size_x)
        chip_rows.append((row_start, row_start + chip_size))
        chip_cols.append((col_start, col_start + chip_size))

    image_height = max(r[1] for r in chip_rows) - min(r[0] for r in chip_rows)
    image_width = max(c[1] for c in chip_cols) - min(c[0] for c in chip_cols)

    return min_x, min_y, max_x, max_y, image_height, image_width, pixel_size_x, pixel_size_y


def _build_merged_input(
    all_inputs: list,
    all_masks: list,
    chip_size: int,
    image_height: int,
    image_width: int,
    max_y: float,
    min_x: float,
    pixel_size_x: float,
    pixel_size_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build merged input array from all chips.

    Returns:
        Tuple of (merged_input, merged_mask)
    """
    sample = all_inputs[0][0]
    merged_input = np.zeros(
        (image_height, image_width, sample.shape[0], sample.shape[1]), dtype=sample.dtype
    )
    merged_mask = np.zeros((image_height, image_width), dtype=bool)

    for (input_np, _, _, bounds, _), mask in zip(all_inputs, all_masks):
        row_start = int((max_y - bounds.top) / pixel_size_y)
        col_start = int((bounds.left - min_x) / pixel_size_x)
        row_start = max(0, min(row_start, image_height - chip_size))
        col_start = max(0, min(col_start, image_width - chip_size))
        row_end = row_start + chip_size
        col_end = col_start + chip_size
        merged_input[row_start:row_end, col_start:col_end, :, :] = np.transpose(
            input_np, (2, 3, 0, 1)
        )
        merged_mask[row_start:row_end, col_start:col_end] = mask

    return merged_input, merged_mask


def _run_seamless_blending(
    merged_input: np.ndarray,
    image_height: int,
    image_width: int,
    chip_size: int,
    is_segmentation: bool,
    num_classes: int,
    model: LightningModule,
    device: str,
    seamless_seg,
) -> np.ndarray:
    """Execute seamless-seg blending pipeline.

    Returns:
        Blended logits array
    """
    image_size = (image_height, image_width)
    tile_size = (chip_size, chip_size)
    # Even pixels are required for seamless blending between tiles
    overlap_size = int(max(2, (chip_size * OVERLAP_FACTOR) // 2 * 2))
    overlap = (overlap_size, overlap_size)

    plan, _ = seamless_seg.plan_regular_grid(image_size, tile_size, overlap)
    logger.info(
        "Seamless-seg plan: image_size=%s, tile_size=%s, overlap=%s",
        image_size,
        tile_size,
        overlap,
    )

    # Callback for seamless_seg.pytorch_outputs_generator
    def read_tile(geom):
        y_slc, x_slc = seamless_seg.shape_to_slices(geom)
        tile = merged_input[y_slc, x_slc]  # (H, W, C, T)
        # return (C, T, H, W) slice for the model.
        return np.transpose(tile, (2, 3, 0, 1))

    if is_segmentation:
        blended_logits = np.zeros((image_height, image_width, num_classes), dtype=np.float32)
    else:
        blended_logits = np.zeros((image_height, image_width), dtype=np.float32)

    in_tiles = seamless_seg.pytorch_outputs_generator(plan, model, read_tile, device=device)
    logger.info("Executing seamless-seg blending (run_plan)...")

    for _, out_geom, out_tile in seamless_seg.run_plan(plan, in_tiles):
        if not is_segmentation and out_tile.ndim == 3 and out_tile.shape[2] == 1:
            out_tile = out_tile.squeeze(axis=2)
        y_slc, x_slc = seamless_seg.shape_to_slices(out_geom)
        blended_logits[y_slc, x_slc] = out_tile

    return blended_logits


def _save_joined_prediction(
    blended_logits: np.ndarray,
    merged_mask: np.ndarray,
    is_segmentation: bool,
    all_inputs: list,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    output_folder: str,
) -> None:
    """Convert blended logits to predictions and save joined file."""
    if is_segmentation:
        predictions = np.argmax(blended_logits, axis=-1).astype(np.int8)
    else:
        predictions = blended_logits.astype(np.float32)

    predictions = np.where(merged_mask, -1, predictions)

    ref_profile = all_inputs[0][2]
    crs = ref_profile.get("crs")
    if crs is None:
        crs = "EPSG:4326"

    bounds_xy = (min_x, min_y, max_x, max_y)
    joined_path = save_prediction(
        predictions,
        output_folder,
        bounds=bounds_xy,
        crs=crs,
        is_segmentation=is_segmentation,
    )
    logger.info("Blending complete, saved joined prediction to %s", joined_path)


def _run_with_stitching(
    dataloader: DataLoader,
    model: LightningModule,
    device: str,
    output_folder: str,
) -> Dict:
    """Run inference with seamless-seg stitching."""
    try:
        import instageo.model.seamless_seg as seamless_seg
    except ImportError:
        raise ImportError(
            "seamless_seg module not found. seamless-seg is required when stitching=True"
        )

    # Collect tiles and metadata
    all_inputs, all_masks, is_segmentation, chip_size, num_classes, probe_logits = _collect_tiles(
        dataloader, device, model
    )
    num_chips = len(all_inputs)
    logger.info(
        "Input tiles collected: %d chips, is_segmentation=%s, chip_size=%s",
        num_chips,
        is_segmentation,
        chip_size,
    )

    # Handle single chip case
    if num_chips == 1:
        logger.info("Single chip detected; no stitching required. Saving as joined_prediction.tif.")
        _handle_single_chip(
            all_inputs, all_masks, is_segmentation, model, device, output_folder, probe_logits
        )
        return {"stitched": True}

    # Multi-chip stitching
    (
        min_x,
        min_y,
        max_x,
        max_y,
        image_height,
        image_width,
        pixel_size_x,
        pixel_size_y,
    ) = _calculate_merged_extent(all_inputs, chip_size)
    logger.info(
        "Merged extent: image_size=(%d, %d), num_chips=%d",
        image_height,
        image_width,
        num_chips,
    )

    merged_input, merged_mask = _build_merged_input(
        all_inputs,
        all_masks,
        chip_size,
        image_height,
        image_width,
        max_y,
        min_x,
        pixel_size_x,
        pixel_size_y,
    )

    blended_logits = _run_seamless_blending(
        merged_input,
        image_height,
        image_width,
        chip_size,
        is_segmentation,
        num_classes,
        model,
        device,
        seamless_seg,
    )

    _save_joined_prediction(
        blended_logits,
        merged_mask,
        is_segmentation,
        all_inputs,
        min_x,
        min_y,
        max_x,
        max_y,
        output_folder,
    )

    return {"stitched": True}


def _run_without_stitching(
    dataloader: DataLoader,
    model: LightningModule,
    device: str,
    output_folder: str,
    num_workers: int,
) -> Dict:
    """Run regular inference without stitching (per-chip predictions)."""
    logger.info("Running inference without stitching (per-chip predictions)")

    with torch.no_grad():
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for (data, _), file_names, nan_mask in tqdm(dataloader, desc="Running Inference"):
                data = data.to(device)
                prediction_batch = model(data)

                if prediction_batch.shape[1] == 1:  # Regression (single output channel)
                    prediction_batch = prediction_batch.cpu().numpy().squeeze(1)
                else:
                    prediction_batch = (
                        torch.argmax(prediction_batch, dim=1).cpu().numpy().astype(np.int8)
                    )

                # Mask out the predictions where the chip had no_data_value
                nan_mask = np.all(nan_mask, axis=1).astype(int)
                prediction_batch = np.where(nan_mask == 1, -1, prediction_batch)

                profiles = []
                for file_name in file_names:
                    with rasterio.open(file_name) as src:
                        profile = src.profile
                        profile.update(
                            count=1,
                            dtype=rasterio.int8
                            if prediction_batch.dtype == np.int8
                            else rasterio.float32,
                            nodata=-1,
                        )
                        profiles.append(profile)

                futures = [
                    executor.submit(
                        save_prediction,
                        prediction,
                        output_folder,
                        profile=profile,
                        file_name=file_name,
                    )
                    for prediction, file_name, profile in zip(
                        prediction_batch, file_names, profiles
                    )
                ]
                for future in futures:
                    future.result()

    return {"stitched": False}


def chip_inference(
    dataloader: DataLoader,
    output_folder: str,
    model: LightningModule,
    device: str = "gpu",
    num_workers: int = 4,
    stitching: bool = False,
) -> dict:
    """Chip Inference with optimizations.

    Performs inference on chips and saves predictions. When stitching=True,
    applies seamless-seg blending to logits and saves a single joined prediction file;
    otherwise saves per-chip files.

    Args:
        dataloader: Dataloader that yields input, label and input filenames.
        model: Trained model for inference.
        output_folder: Path to save predictions.
        device: Device used for inference.
        num_workers: Number of workers for concurrent file saving.
        stitching: If True, apply seamless-seg blending and save one joined prediction file.

    Returns:
        Dict containing:
            - "stitched" (bool): True if stitching was applied, False otherwise
            - Carbon tracking information (emissions, duration, etc.)
    """
    device = "cuda" if device == "gpu" else device
    model.eval()
    model.to(device)

    logger.info(
        "Starting inference: device=%s, stitching=%s, output_folder=%s",
        device,
        stitching,
        output_folder,
    )

    tracker = _start_tracker()

    if stitching:
        result = _run_with_stitching(dataloader, model, device, output_folder)
    else:
        result = _run_without_stitching(dataloader, model, device, output_folder, num_workers)

    carbon_info = _stop_tracker(tracker)
    return {**result, **carbon_info}
