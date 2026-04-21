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

"""InstaGeo Data pipeline Module."""

import os
import time
from abc import ABC
from typing import Any, Callable, Dict, List, Tuple

import backoff
import dask
import dask.distributed
import geopandas as gpd
import mgrs
import numpy as np
import pandas as pd
import rasterio  # noqa: F401
import ratelimit
import rioxarray  # noqa: F401
import xarray as xr
from absl import logging
from pyproj import Transformer
from pystac_client import Client
from tqdm import tqdm

from instageo.data.data_source_config import DataSourceConfig
from instageo.data.settings import DataPipelineSettings, NoDataValues

# Masks decoding positions
MASK_DECODING_POS: dict[str, dict] = {
    "HLS": {"cloud": 1, "near_cloud_or_shadow": 2, "cloud_shadow": 3, "water": 5},
    "S2": {"cloud": [8, 9], "water": [6]},
}

# No data values
NO_DATA_VALUES = NoDataValues()
DATA_PIPELINE_SETTINGS = DataPipelineSettings()

# Sleep delays (in seconds)
BATCH_DELAY_SECONDS = 5  # Delay between processing batches
ERROR_RETRY_DELAY_SECONDS = 120  # Delay before retrying after error (2 minutes)

# Microsoft Planetary Computer STAC API
MPC_STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def assert_in_dtype_range(value: int, dtype: np.dtype) -> None:
    """Raise AssertionError if value is not representable in dtype."""
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)
    assert info.min <= value <= info.max, (
        f"Value {value} is out of range for dtype {dtype} "
        f"(valid range: {info.min} to {info.max})"
    )


def mask_segmentation_map(
    chip: xr.DataArray,
    seg_map: xr.DataArray,
    chip_no_data_value: xr.DataArray,
    masking_strategy: str = "any",
) -> xr.DataArray:
    """Masks segmentation map.

    Checks for chip_no_data_value in the chip and masks the segmentation values
    that correspond to no data value in the chip (at least for one band).

    Args:
        seg_map (DataArray): Segmentation map to mask
        chip (DataArray): Chip that correspond to the segmentation map
        chip_no_data_value (int): Value to use for no data areas in the chips.
        masking_strategy (str): Masking strategy to apply ("each" for timestep-wise masking,
        and "any" to exclude pixels if the mask is present for at least one timestep. The
        behavior is the same if the chip is extracted for one timestep.)

    Returns:
        The segmentation map after masking
    """
    if masking_strategy == "each":
        valid_mask = (chip != chip_no_data_value).any(dim="band").astype(seg_map.dtype)
    elif masking_strategy == "any":
        valid_mask = (chip != chip_no_data_value).all(dim="band").astype(seg_map.dtype)
    else:
        raise ValueError(f"Invalid masking strategy: {masking_strategy}")

    seg_no_data_value = NO_DATA_VALUES.SEG_MAP
    assert_in_dtype_range(seg_no_data_value, seg_map.dtype)
    seg_map = seg_map.where(valid_mask, seg_no_data_value)
    return seg_map


def apply_mask(
    chip: xr.DataArray,
    mask: xr.DataArray,
    no_data_value: int,
    mask_decoder: Callable,
    data_source: str,
    masking_strategy: str = "each",
    mask_types: list[str] = list(MASK_DECODING_POS["HLS"].keys()),
) -> xr.DataArray:
    """Apply masking to a chip.

    Args:
        chip (xr.DataArray): Chip array containing the pixels to be masked out.
        mask (xr.DataArray): Array containing the masks.
        no_data_value (int): Value to be used for masked pixels.
        mask_decoder (Callable): Function to use to process/extract actual mask values
        data_source (str): Data source used to extract masking positions based on mask types
        masking_strategy (str): Masking strategy to apply ("each" for timestep-wise masking,
        and "any" to exclude pixels if the mask is present for at least one timestep. The
        behavior is the same if the chip is extracted for one timestep.)
        mask_types (list[str]): Mask types to apply.

    Returns:
        xr.DataArray: The masked data array.
    """
    for mask_type in mask_types:
        pos = MASK_DECODING_POS[data_source].get(mask_type, None)
        if pos:
            decoded_mask = mask_decoder(mask, pos)
            if masking_strategy == "each":
                # repeat across timesteps so that, each mask is applied to its
                # corresponding timestep
                decoded_mask = decoded_mask.values.repeat(chip.shape[0] // mask.shape[0], axis=0)
            elif masking_strategy == "any":
                # collapse the mask to exclude a pixel if its corresponding mask value
                # for at least one timestep is 1
                decoded_mask = decoded_mask.values.any(axis=0)
            chip = chip.where(decoded_mask == 0, other=no_data_value)
    return chip


def get_tile_info(
    data: pd.DataFrame,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 5,
    temporal_tolerance_minutes: int = 0,
) -> tuple[pd.DataFrame, list[tuple[str, list[str]]]]:
    """Get Tile Info.

    Retrieves a summary of all tiles required for a given dataset. The summary contains
    the desired start and end date for each tile. Also retrieves a list of queries
    that can be used to retrieve the tiles for each observation in `data`.

    Args:
        data (pd.DataFrame): A dataframe containing observation records.
        num_steps (int): Number of temporal time steps
        temporal_step (int): Size of each temporal step.
        temporal_tolerance (int): Number of days used as offset for the
        start and end dates to search for each tile.
        temporal_tolerance_minutes (int): Number of minutes to add to the temporal
            tolerance.

    Returns:
        A `tile_info` dataframe and a list of `tile_queries`
    """
    push_max_date_to_end_of_day = "time" not in data.columns
    data = data[["mgrs_tile_id", "input_features_date", "x", "y"]].reset_index(drop=True)
    tile_queries = []
    tile_info: Any = []
    for _, (tile_id, date, lon, lat) in data.iterrows():
        history = []
        for i in range(num_steps):
            curr_date = date - pd.Timedelta(days=temporal_step * i)
            history.append(curr_date.strftime("%Y-%m-%dT%H:%M:%S"))
            tile_info.append([tile_id, curr_date, lon, lat])
        tile_queries.append((tile_id, history))
    tile_info = (
        pd.DataFrame(tile_info, columns=["tile_id", "date", "lon", "lat"])
        .groupby("tile_id")
        .agg(
            min_date=("date", "min"),
            max_date=("date", "max"),
            lon_min=("lon", "min"),
            lon_max=("lon", "max"),
            lat_min=("lat", "min"),
            lat_max=("lat", "max"),
        )
    ).reset_index()

    total_temporal_tol = temporal_tolerance + (temporal_tolerance_minutes / (24 * 60))
    tile_info["min_date"] -= pd.Timedelta(days=total_temporal_tol)
    tile_info["max_date"] += pd.Timedelta(days=total_temporal_tol)
    tile_info["min_date"] = tile_info["min_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    if push_max_date_to_end_of_day:
        tile_info["max_date"] = tile_info["max_date"].dt.strftime("%Y-%m-%dT23:59:59")
    else:
        tile_info["max_date"] = tile_info["max_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return tile_info, tile_queries


def reproject_coordinates(df: pd.DataFrame, source_epsg: int = 4326) -> pd.DataFrame:
    """Reproject coordinates from the source EPSG to EPSG:4326.

    This function reprojects the geo coordinates found in df dataframe to the EPSG:4326

    Args:
        df (pd.DataFrame): DataFrame containing longitude and latitude columns.
        source_epsg (int): The EPSG code of the source CRS for invalid coordinates.

    Returns:
        pd.DataFrame: DataFrame with transformed and valid coordinates.
    """
    logging.info("Reprojecting coordinates to EPSG:4326...")
    transformer = Transformer.from_crs(f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True)

    # Reproject the invalid rows using Vectorized transformation
    x, y = transformer.transform(df["x"].values, df["y"].values)
    df[["x", "y"]] = np.column_stack((x, y))

    return df


def get_tiles(data: pd.DataFrame, src_crs: int = 4326, min_count: int = 100) -> pd.DataFrame:
    """Retrieve Tile IDs for Geospatial Observations from Satellite Data.

    This function associates each geospatial observation with a tile ID based on its
    geographic location, accommodating datasets with varying density across locations. By
    focusing on more densely populated areas, it enables more efficient resource usage and
    refined data analysis.

    The function assigns a tile ID to each observation, counts the occurrences within
    each tile, and retains only those tiles with a specified minimum count (`min_count`) of
    observations.

    Args:
        data: DataFrame containing geospatial observations with location coordinates.
        src_crs (int): CRS of points in `data`
        min_count: Minimum count of observations required per tile to retain.

    Returns:
        A subset of observations within tiles that meet or exceed the specified `min_count`.
    """
    if src_crs != 4326:
        data = reproject_coordinates(data, source_epsg=src_crs)
    if "mgrs_tile_id" not in data.columns:
        mgrs_object = mgrs.MGRS()
        get_mgrs_tile_id = lambda row: mgrs_object.toMGRS(row["y"], row["x"], MGRSPrecision=0)
        data["mgrs_tile_id"] = data.apply(get_mgrs_tile_id, axis=1)
    tile_counts = data.groupby("mgrs_tile_id").size().sort_values(ascending=False)
    data = pd.merge(data, tile_counts.reset_index(name="counts"), how="left", on="mgrs_tile_id")
    sub_data = data[data["counts"] >= min_count]
    assert not sub_data.empty, "No observation records left"
    return sub_data


def create_segmentation_map(
    chip: Any, df: pd.DataFrame, window_size: int, task_type: str = "seg"
) -> xr.DataArray:
    """Create a segmentation map for the chip using the DataFrame.

    Args:
        chip (Any): The chip (subset of the original data) for which the segmentation
            map is being created.
        df (pd.DataFrame): DataFrame containing the data to be used in the segmentation
            map.
        window_size (int): Window size to use around the observation pixel.
        task_type (str): Task type to use to adjust the data type of the segmentation maps.

    Returns:
         xr.DataArray: The created segmentation map as an xarray DataArray.
    """
    seg_map = xr.full_like(
        chip.isel(band=0),
        fill_value=NO_DATA_VALUES.SEG_MAP,
        dtype=np.int16 if task_type == "seg" else np.float32,
    )
    df = df[
        (chip["x"].min().item() <= df["geometry"].x)
        & (df["geometry"].x <= chip["x"].max().item())
        & (chip["y"].min().item() <= df["geometry"].y)
        & (df["geometry"].y <= chip["y"].max().item())
    ]
    cols, rows = np.floor(
        ~seg_map.rio.transform() * (df.geometry.x.values, df.geometry.y.values)
    ).astype(int)
    offsets = np.arange(-window_size, window_size + 1)
    offset_rows, offset_cols = np.meshgrid(offsets, offsets)
    window_rows = np.clip(rows[:, np.newaxis, np.newaxis] + offset_rows, 0, chip.sizes["x"] - 1)
    window_cols = np.clip(cols[:, np.newaxis, np.newaxis] + offset_cols, 0, chip.sizes["y"] - 1)
    window_labels = np.repeat(df.label.values, offset_rows.ravel().shape)
    seg_map.values[window_rows.ravel(), window_cols.ravel()] = window_labels
    return seg_map


def get_chip_coords(df: gpd.GeoDataFrame, tile: xr.DataArray, chip_size: int) -> np.array:
    """Get Chip Coordinates.

    Given a list of x,y coordinates tuples of a point and an xarray dataarray, this
    function returns the unique corresponding x,y indices of the grid where each point will fall
    when the DataArray is gridded such that each grid has size `chip_size`
    indices where it will fall.

    Args:
        gdf (gpd.GeoDataFrame): GeoPandas dataframe containing the point.
        tile (xr.DataArray): Tile DataArray.
        chip_size (int): Size of each chip.

    Returns:
        List of chip indices.
    """
    cols, rows = np.floor(
        ~tile.rio.transform() * (df.geometry.x.values, df.geometry.y.values)
    ).astype(int)
    return np.unique(np.stack((cols // chip_size, rows // chip_size), axis=-1), axis=0)


def get_pystac_client() -> Client:
    """Opens a pystac_client Client instance using MPC STAC API URL.

    Returns:
        Client : A client with an established connection to the STAC Catalog.
    """
    return Client.open(MPC_STAC_API_URL)


def adjust_dims(data: xr.DataArray) -> xr.DataArray:
    """Adjusts dimensions of a dataarray.

    This function stacks the "time" and "band" dims over a new "band" dim and reorders
    the dataarray dims into ("band","y","x").

    Args:
        data (xr.DataArray): A dataarray for which dimensions need to be adjusted.

    Returns:
        xr.DataArray: A 3D xarray DataArray without 'time' dimension.
    """
    num_bands = data["band"].size
    data = data.stack(time_band=("time", "band"))
    new_bands_indices = [
        f"{band}_{i // num_bands}" for i, (_, band) in enumerate(data.coords["time_band"].values)
    ]
    data = data.drop_vars(["time_band", "time", "band"])
    data.coords["time_band"] = new_bands_indices
    data = data.rename({"time_band": "band"}).transpose("band", "y", "x")
    return data


class BaseDataPipeline(ABC):
    """Base class for all data pipelines (Points and Raster).

    This class contains all common logic shared between point-based and raster-based
    data pipelines, including:
    - Common initialization parameters
    - Abstract hooks for data-source-specific behavior
    - Shared helper methods for mask application, validation, and saving
    """

    def __init__(
        self,
        config: DataSourceConfig,
        output_directory: str,
        chip_size: int,
        mask_types: List[str],
        masking_strategy: str,
        src_crs: int,
        spatial_resolution: float,
        qa_check: bool = True,
        task_type: str = "seg",
    ) -> None:
        """Initialize base pipeline with common parameters.

        Args:
            config: Data source configuration
            output_directory: Directory to save output files
            chip_size: Size of chips in pixels
            mask_types: Types of masks to apply
            masking_strategy: Strategy for masking
            src_crs: Source CRS EPSG code
            spatial_resolution: Spatial resolution in meters
            qa_check: Whether to perform quality checks
            task_type: Type of task (seg or reg)
        """
        self.config = config
        self.output_directory = output_directory
        self.chip_size = chip_size
        self.mask_types = mask_types
        self.masking_strategy = masking_strategy
        self.src_crs = src_crs
        self.spatial_resolution = spatial_resolution
        self.qa_check = qa_check
        self.task_type = task_type

    def setup(self) -> None:
        """Set up necessary configurations on Dask workers.

        Note: This is called on each Dask worker, not just the main process.
        Use this for per-worker initialization (e.g., authentication, GDAL options).
        """
        return self.config.setup_func()

    @ratelimit.limits(calls=DATA_PIPELINE_SETTINGS.COG_DOWNLOAD_RATELIMIT, period=60)
    @backoff.on_exception(
        backoff.expo,
        (rasterio.errors.RasterioIOError, Exception),
        max_tries=5,
        max_time=300,
        jitter=backoff.full_jitter,
    )
    def load_data(self, tile_dict: Dict[str, Any]) -> Tuple[xr.Dataset, xr.Dataset, str]:
        """Load data using data source configuration."""
        from instageo.data.stac_utils import open_stac_items

        return open_stac_items(
            tile_dict=tile_dict["granules"],
            epsg=self.src_crs,
            resolution=self.spatial_resolution,
            bands_asset=self.config.bands_asset,
            blocksize=self.config.blocksize,
            mask_band=self.config.mask_band,
            load_masks=True,
            fill_value=self.config.no_data_value,
            sign_func=self.config.sign_func,
        )

    def get_no_data_value(self) -> int:
        """Get no-data value from configuration."""
        return self.config.no_data_value

    def get_data_source_name(self) -> str:
        """Get data source name from configuration."""
        return self.config.data_source_name

    def get_mask_decoder(self) -> Any:
        """Get mask decoder from configuration."""
        return self.config.mask_decoder

    def _is_array_empty(self, array: xr.DataArray, no_data_value: int) -> bool:
        """Check if array contains only no-data values.

        Args:
            array: The array to check
            no_data_value: The no-data value to check against

        Returns:
            True if array contains only no-data values
        """
        return array.where(array != no_data_value).count().values == 0

    def _prepare_chip_for_save(
        self, chip: xr.DataArray, clip_range: Tuple[float, float] | None = None
    ) -> xr.DataArray:
        """Prepare chip for saving: clip values, handle NaN, set dtype.

        Args:
            chip: The chip to prepare
            clip_range: Optional (min, max) range for clipping

        Returns:
            Prepared chip ready for saving
        """
        # Clip if range provided
        if clip_range is not None:
            chip = chip.clip(min=clip_range[0], max=clip_range[1])

        # Replace NaN with no-data value
        no_data_value = self.get_no_data_value()
        chip = chip.where(~np.isnan(chip), no_data_value)

        return chip.astype(self.config.chip_dtype)

    def _prepare_label_for_save(self, seg_map: xr.DataArray) -> xr.DataArray:
        """Prepare segmentation map for saving: handle NaN, set dtype.

        Args:
            seg_map: The segmentation map to prepare

        Returns:
            Prepared label ready for saving
        """
        # Replace NaN with no-data value
        seg_map = seg_map.where(~np.isnan(seg_map), NO_DATA_VALUES.SEG_MAP)

        # Set dtype based on task type
        dtype = np.int8 if self.task_type == "seg" else np.float32
        return seg_map.astype(dtype)

    def get_clip_range(self) -> Tuple[float, float] | None:
        """Get the valid data range for clipping from configuration.

        Returns:
            Tuple of (min, max) values for clipping, or None for no clipping
        """
        return self.config.clip_range

    def _create_output_directories(self) -> None:
        """Create output directories for chips and segmentation maps."""
        os.makedirs(os.path.join(self.output_directory, "chips"), exist_ok=True)
        os.makedirs(os.path.join(self.output_directory, "seg_maps"), exist_ok=True)

    def _save_dataset_csv(
        self,
        chip_paths: List[str],
        label_paths: List[str] | None = None,
    ) -> None:
        """Save dataset CSV with chip and label paths.

        Args:
            chip_paths: List of chip file paths
            label_paths: Optional list of label file paths (None for bbox features)
        """
        logging.info("Saving dataframe of chips and segmentation maps.")
        output_filename = f"{self.get_data_source_name().lower()}_dataset.csv"

        if label_paths is None:
            chips_df = pd.DataFrame({"Input": chip_paths})
        else:
            chips_df = pd.DataFrame({"Input": chip_paths, "Label": label_paths})

        chips_df.to_csv(os.path.join(self.output_directory, output_filename))


class RasterDataPipeline(BaseDataPipeline):
    """Raster-based data pipeline with configurable data source.

    This class defines the structure for geospatial data processing pipelines that
    work with raster imagery and associated masks. It is designed to support large-scale,
    distributed processing using Dask, and to streamline the generation of data chips
    and segmentation labels for machine learning applications.
    """

    def __init__(
        self,
        config: DataSourceConfig,
        output_directory: str,
        chip_size: int,
        raster_path: str,
        mask_types: List[str],
        masking_strategy: str,
        src_crs: int,
        spatial_resolution: float,
        qa_check: bool = True,
        task_type: str = "seg",
        is_bbox_feature: bool = False,
    ) -> None:
        """Initialize raster pipeline with data source configuration.

        Args:
            config: Data source configuration
            output_directory: Directory to save output chips and segmentation maps
            chip_size: Size of the chip in pixels
            mask_types: List of mask types to apply
            masking_strategy: Strategy for masking (e.g., "and", "or")
            src_crs: Source coordinate reference system
            spatial_resolution: Spatial resolution in meters
            qa_check: Whether to perform quality checks
            task_type: Type of task (e.g., "seg", "cls")
            raster_path: Path to raster labels directory (raster-specific)
            is_bbox_feature: Whether this is a bbox feature task (raster-specific)
        """
        # Initialize common parameters via parent (including config)
        super().__init__(
            config=config,
            output_directory=output_directory,
            chip_size=chip_size,
            mask_types=mask_types,
            masking_strategy=masking_strategy,
            src_crs=src_crs,
            spatial_resolution=spatial_resolution,
            qa_check=qa_check,
            task_type=task_type,
        )
        # Raster-specific parameters
        self.raster_path = raster_path
        self.is_bbox_feature = is_bbox_feature

    def process_row(
        self,
        row_dict: Dict[str, Any],
        tile_dict: dict[str, Any],
    ) -> Tuple[str, str | None] | None:
        """Processes a single row of data (concrete implementation with data-source hooks).

        This method contains all the common logic for processing rows. Data-source-specific
        behavior is delegated to abstract methods that subclasses must implement.

        Arguments:
            row_dict: Dictionary created from a row in the observation records dataframe.
            tile_dict: Dictionary containing granules STAC items.

        Returns:
            Tuple of (chip_path, label_path) on success, or None if the tile produced no output
            (skipped due to cloud cover, size mismatch, empty label, or processing error).
        """
        from shapely.geometry import shape

        from instageo.data import geo_utils

        no_data_value = self.get_no_data_value()
        data_source = self.get_data_source_name()
        mask_decoder = self.get_mask_decoder()
        clip_range = self.get_clip_range()

        label_filename = (
            f"{os.path.splitext(row_dict['label_filename'])[0]}_{row_dict['mgrs_tile_id']}"
        )
        chip_filename = label_filename.replace("mask", "merged").replace("label", "chip")

        chip_path = os.path.join(self.output_directory, "chips", f"{chip_filename}.tif")
        label_path = os.path.join(self.output_directory, "seg_maps", f"{label_filename}.tif")

        if os.path.exists(chip_path) and os.path.exists(label_path):
            logging.info(f"Skipping {chip_path} because it's already created")
            return chip_path, label_path
        if os.path.exists(chip_path) and self.is_bbox_feature:
            logging.info(f"Skipping {chip_path} because it's already created")
            return chip_path, None

        try:
            dsb, dsm, _ = self.load_data(tile_dict)
            geometry = shape(row_dict["geometry"])

            # Process chip
            chip = geo_utils.slice_xr_dataset(dsb, geometry, chip_size=self.chip_size)
            if not self.is_bbox_feature:
                seg_map = xr.open_dataarray(
                    os.path.join(self.raster_path, row_dict["label_filename"])
                )
            else:
                seg_map = None

            if dsm is not None:
                chip_mask = geo_utils.slice_xr_dataset(dsm, geometry, chip_size=self.chip_size)
                chip = apply_mask(
                    chip=chip,
                    mask=chip_mask,
                    no_data_value=no_data_value,
                    mask_decoder=mask_decoder,
                    data_source=data_source,
                    mask_types=self.mask_types,
                    masking_strategy=self.masking_strategy,
                )

            if chip is not None and seg_map is None:
                chip = self._prepare_chip_for_save(chip, clip_range)
                chip.squeeze().rio.to_raster(chip_path)
                return chip_path, None
            elif (
                chip is not None
                and chip.sizes["x"] == seg_map.sizes["x"]
                and chip.sizes["y"] == seg_map.sizes["y"]
            ):
                # Overrides the chip coordinates to match the segmentation map.
                seg_map, chip = xr.align(seg_map, chip, join="override", exclude=["band"])

                if self.qa_check:
                    if self._is_array_empty(chip, no_data_value):
                        logging.warning(f"Skipping {chip_filename} due to cloud")
                        return None
                    seg_map = mask_segmentation_map(
                        chip, seg_map, no_data_value, self.masking_strategy
                    )
                    if self._is_array_empty(seg_map, NO_DATA_VALUES.SEG_MAP):
                        logging.warning(f"Skipping {label_filename} due to empty label")
                        return None
                seg_map = self._prepare_label_for_save(seg_map)
                chip = self._prepare_chip_for_save(chip, clip_range)

                seg_map.squeeze().rio.to_raster(label_path)
                chip.squeeze().rio.to_raster(chip_path)
                return chip_path, label_path
            else:
                logging.warning(
                    f"Skipping {chip_filename} due to chip or label size mismatch or None"
                )
                return None
        except Exception as e:
            logging.error(f"Error processing {chip_filename}: {str(e)}")
            return None

    def _process_batch(
        self,
        client: dask.distributed.Client,
        dataset: Dict[str, Any],
        batch_records: pd.DataFrame,
    ) -> List[Tuple[str, str]]:
        """Process a batch of records in parallel.

        Args:
            client: Dask client for parallel processing
            dataset: Dataset dictionary
            batch_records: Batch of records to process

        Returns:
            List of (chip_path, label_path) tuples for tiles that produced output.
        """
        futures = []
        for _, row in batch_records.iterrows():
            row_dict = row.to_dict()
            row_dict["geometry"] = row.geometry.__geo_interface__
            label_filename = (
                f"{os.path.splitext(row_dict['label_filename'])[0]}_" f"{row_dict['mgrs_tile_id']}"
            )
            chip_filename = label_filename.replace("mask", "merged").replace("label", "chip")

            chip_path = os.path.join(self.output_directory, "chips", f"{chip_filename}.tif")
            if os.path.exists(chip_path):
                logging.info(f"Skipping {chip_path} because it's already created")
                continue

            futures.append(
                client.submit(
                    self.process_row,
                    row_dict,
                    dataset[row_dict["stac_items_str"]],
                )
            )

        results = client.gather(futures)
        return [result for result in results if result is not None]

    def run(self, dataset: Dict[str, Any], obsv_records: pd.DataFrame) -> None:
        """Main method to run the pipeline and create all chips and corresponding labels.

        Arguments:
            dataset: A dataset mapping `key` to STAC Items.
            obsv_records: A dataframe containing a column that match each row to STAC items in
                `dataset` using `key`

        Returns:
            None.
        """
        self._create_output_directories()

        with dask.distributed.Client() as client:
            with dask.distributed.performance_report(
                filename=os.path.join(self.output_directory, "dask-report.html")
            ):
                client.run(self.setup)
                logging.info(f"View Dask Distributed Dashboard at {client.dashboard_link}.")

                chip_paths = []
                label_paths = []
                batch_size = DATA_PIPELINE_SETTINGS.BATCH_SIZE
                total_records = len(obsv_records)

                for i in tqdm(
                    range(0, total_records, batch_size),
                    desc="Processing batches",
                    total=total_records // batch_size,
                ):
                    batch_records = obsv_records.iloc[i : i + batch_size]
                    try:
                        results = self._process_batch(client, dataset, batch_records)
                        for chip_path, label_path in results:
                            chip_paths.append(chip_path)
                            label_paths.append(label_path)
                        time.sleep(BATCH_DELAY_SECONDS)
                    except Exception as e:
                        logging.error(f"Error processing batch {i // batch_size}: {str(e)}")
                        time.sleep(ERROR_RETRY_DELAY_SECONDS)
                        continue

                records_failed = total_records - len(chip_paths)
                if records_failed:
                    logging.warning(
                        f"Pipeline completed: {records_failed}/{total_records} record(s) "
                        "produced no output. Check logs above for per-record details."
                    )

        # Save results
        if self.is_bbox_feature:
            self._save_dataset_csv(chip_paths, label_paths=None)
        else:
            self._save_dataset_csv(chip_paths, label_paths)


class PointsDataPipeline(BaseDataPipeline):
    """Points-based data pipeline with configurable data source."""

    def __init__(
        self,
        config: DataSourceConfig,
        output_directory: str,
        chip_size: int,
        mask_types: List[str],
        masking_strategy: str,
        src_crs: int,
        spatial_resolution: float,
        qa_check: bool = True,
        window_size: int = 0,
        task_type: str = "seg",
    ) -> None:
        """Initialize points pipeline with data source configuration.

        Args:
            config: Data source configuration
            output_directory: Directory to save output chips and segmentation maps
            chip_size: Size of the chip in pixels
            mask_types: List of mask types to apply
            masking_strategy: Strategy for masking (e.g., "and", "or")
            src_crs: Source coordinate reference system
            spatial_resolution: Spatial resolution in meters
            qa_check: Whether to perform quality checks
            window_size: Window size for point-based sampling (points-specific)
            task_type: Type of task (e.g., "seg", "cls")
        """
        # Call parent constructor with common parameters (including config)
        super().__init__(
            config=config,
            output_directory=output_directory,
            chip_size=chip_size,
            mask_types=mask_types,
            masking_strategy=masking_strategy,
            src_crs=src_crs,
            spatial_resolution=spatial_resolution,
            qa_check=qa_check,
            task_type=task_type,
        )
        # Points-specific parameter
        self.window_size = window_size

    def get_tile_id(self, obsv_records: gpd.GeoDataFrame, tile_dict: Dict[str, Any]) -> str:
        """Extract tile ID using data source configuration."""
        return self.config.tile_id_extractor(obsv_records, tile_dict)

    def _is_stac_item_processed(
        self, stac_items_str: str, obsv_records: pd.DataFrame, existing_chips: set[str]
    ) -> bool:
        """Check if any chips and segmentation maps for a STAC item have been processed.

        Args:
            stac_items_str: The STAC items string identifier
            obsv_records: DataFrame containing observation records
            existing_chips: Set of already processed chip identifiers

        Returns:
            bool: True if any chips and segmentation maps exist, False otherwise
        """
        df = obsv_records[obsv_records["stac_items_str"] == stac_items_str]
        if df.empty:
            return False

        # Get the first record to construct the chip identifier
        first_record = df.iloc[0]
        date_id = first_record["date"].strftime("%Y%m%d")

        # Use mgrs_tile_id from the record instead of parsing from stac_items_str
        if "mgrs_tile_id" in first_record:
            tile_id = first_record["mgrs_tile_id"]
        else:
            # Fallback to parsing from stac_items_str (for HLS compatibility)
            tile_name_splits = stac_items_str.split("_")[0].split(".")
            if len(tile_name_splits) >= 4:
                tile_id = f"{tile_name_splits[1]}_{tile_name_splits[2]}_{tile_name_splits[3]}"
            else:
                tile_id = stac_items_str.split("_")[0]

        chip_base_id = f"{date_id}_{tile_id}"

        return chip_base_id in existing_chips

    def process_tile(
        self,
        obsv_records: gpd.GeoDataFrame,
        tile_dict: Dict[str, Any],
        batch_size: int,
    ) -> Tuple[list[str], list[str]]:
        """Processes a single tile.

        This method contains all the common logic for processing tiles. Data-source-specific
        behavior is delegated to abstract methods that subclasses must implement.

        Arguments:
            obsv_records: Observation records dataframe.
            tile_dict: Dictionary containing granules STAC items.
            batch_size: Number of records to process at a time.
        Returns: A tuple of chip and label filename lists.
        """
        import rasterio.errors

        tile_id = self.get_tile_id(obsv_records, tile_dict)
        stac_items_str = obsv_records.iloc[0]["stac_items_str"]
        chip_paths = []
        label_paths = []
        no_data_value = self.get_no_data_value()
        data_source = self.get_data_source_name()
        mask_decoder = self.get_mask_decoder()
        clip_range = self.get_clip_range()

        try:
            date_id = obsv_records.iloc[0]["date"].strftime("%Y%m%d")
            dsb, dsm, _ = self.load_data(tile_dict)
            n_chips_x = dsb.sizes["x"] // self.chip_size
            n_chips_y = dsb.sizes["y"] // self.chip_size
            chip_coords = get_chip_coords(obsv_records, dsb, self.chip_size)

            # Process chips in smaller batches to avoid overwhelming the API
            for i in range(0, len(chip_coords), batch_size):
                batch_coords = chip_coords[i : i + batch_size]
                chips, masks, seg_maps_temp_filenames, chips_temp_filenames = (
                    [],
                    [],
                    [],
                    [],
                )

                for x, y in batch_coords:
                    # TODO: handle potential partially out of bound chips
                    if (x >= n_chips_x) or (y >= n_chips_y):
                        continue

                    chip_id = f"{date_id}_{tile_id}_{x}_{y}"
                    chip_name = f"chip_{chip_id}.tif"
                    seg_map_name = f"seg_map_{chip_id}.tif"

                    chip_filename = os.path.join(self.output_directory, "chips", chip_name)
                    chips_temp_filenames.append(chip_filename)
                    seg_map_filename = os.path.join(self.output_directory, "seg_maps", seg_map_name)
                    seg_maps_temp_filenames.append(seg_map_filename)
                    if os.path.exists(chip_filename) or os.path.exists(seg_map_filename):
                        logging.info(f"Skipping {chip_filename} because it's already created")
                        continue

                    chip = dsb.isel(
                        x=slice(x * self.chip_size, (x + 1) * self.chip_size),
                        y=slice(y * self.chip_size, (y + 1) * self.chip_size),
                    )
                    chips.append(chip)

                    if dsm is not None:
                        chip_mask = dsm.isel(
                            x=slice(x * self.chip_size, (x + 1) * self.chip_size),
                            y=slice(y * self.chip_size, (y + 1) * self.chip_size),
                        )
                        masks.append(chip_mask)
                    else:
                        masks.append(None)

                # Process the batch
                try:
                    # Compute chips and masks locally before processing
                    chips = [chip.compute() for chip in chips]
                    masks = [mask.compute() for mask in masks] if dsm is not None else masks

                    for chip, mask, chip_filename, seg_map_filename in zip(
                        chips, masks, chips_temp_filenames, seg_maps_temp_filenames
                    ):
                        if mask is not None:
                            chip = apply_mask(
                                chip=chip,
                                mask=mask,
                                no_data_value=no_data_value,
                                mask_decoder=mask_decoder,
                                data_source=data_source,
                                mask_types=self.mask_types,
                                masking_strategy=self.masking_strategy,
                            )

                        if self._is_array_empty(chip, no_data_value):
                            logging.warning(f"Skipping {chip_filename} due to cloud")
                            continue

                        seg_map = create_segmentation_map(
                            chip, obsv_records, self.window_size, self.task_type
                        )
                        seg_map = mask_segmentation_map(
                            chip,
                            seg_map,
                            no_data_value,
                            self.masking_strategy,
                        )

                        if self._is_array_empty(seg_map, NO_DATA_VALUES.SEG_MAP):
                            logging.warning(f"Skipping {seg_map_filename} due to empty label")
                            continue

                        seg_map = self._prepare_label_for_save(seg_map)
                        chip = self._prepare_chip_for_save(chip, clip_range)

                        label_paths.append(seg_map_filename)
                        chip_paths.append(chip_filename)
                        seg_map.rio.to_raster(seg_map_filename)
                        chip.rio.to_raster(chip_filename)

                except Exception as e:
                    logging.error(f"Error processing batch: {str(e)}")
                    continue

                # Add a delay between batches to avoid rate limiting
                time.sleep(BATCH_DELAY_SECONDS)

        except rasterio.errors.RasterioIOError as e:
            logging.error(f"Error {e} when reading dataset containing: {stac_items_str}")
        except Exception as e:
            logging.error(f"Error {e} when processing {stac_items_str}")

        return chip_paths, label_paths

    def run(self, dataset: Dict[str, Any], obsv_records: pd.DataFrame) -> None:
        """Main method to run the pipeline and create all chips and corresponding labels.

        Arguments:
            dataset: A dataset mapping `key` to STAC Items.
            obsv_records: A dataframe containing a column that match each row to STAC items in
                `dataset` using `key`

        Returns:
            None.
        """
        self._create_output_directories()

        # Collect existing chip identifiers
        existing_chips = set()
        for filename in os.listdir(os.path.join(self.output_directory, "chips")):
            if filename.startswith("chip_") and filename.endswith(".tif"):
                # Extract the base identifier (date_tile_id) from the filename
                base_id = "_".join(
                    filename.split("_")[1:-2]
                )  # Remove 'chip_' prefix and x_y.tif suffix
                existing_chips.add(base_id)

        # Filter out already processed STAC items
        dataset = {
            stac_items_str: tile_dict
            for stac_items_str, tile_dict in dataset.items()
            if not self._is_stac_item_processed(stac_items_str, obsv_records, existing_chips)
        }

        if not dataset:
            logging.info("All STAC items have already been processed. Nothing to do.")
            return

        with dask.distributed.Client() as client:
            with dask.distributed.performance_report(
                filename=os.path.join(self.output_directory, "dask-report.html")
            ):
                client.run(self.setup)
                logging.info(f"View Dask Distributed Dashboard at {client.dashboard_link}.")

                chip_paths = []
                label_paths = []
                total_tiles = len(dataset)
                tiles_failed = 0
                for stac_items_str, tile_dict in tqdm(
                    dataset.items(), desc="Processing Dataset entries"
                ):
                    df = obsv_records[obsv_records["stac_items_str"] == stac_items_str]
                    future = client.submit(
                        self.process_tile,
                        df,
                        tile_dict,
                        batch_size=DATA_PIPELINE_SETTINGS.BATCH_SIZE,
                    )
                    tile_chips, tile_labels = future.result()
                    if tile_chips:
                        chip_paths.extend(tile_chips)
                        label_paths.extend(tile_labels)
                    else:
                        tiles_failed += 1

                if tiles_failed:
                    logging.warning(
                        f"Pipeline completed: {tiles_failed}/{total_tiles} tile(s) "
                        "produced no output. Check logs above for per-tile details."
                    )

        self._save_dataset_csv(chip_paths, label_paths)
