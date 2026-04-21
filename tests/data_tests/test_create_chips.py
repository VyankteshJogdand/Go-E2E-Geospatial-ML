import numpy as np
import pytest
import rioxarray
import xarray as xr

from instageo.data.data_pipeline import NO_DATA_VALUES, mask_segmentation_map


def test_segmentation_map_masking():
    chip_path = "tests/data/chip_178_022.tif"
    seg_map_path = "tests/data/chip_178_022.mask.tif"
    chip_no_data_value = -9999
    chip = rioxarray.open_rasterio(chip_path)
    seg_map = rioxarray.open_rasterio(seg_map_path).astype(chip.dtype)
    seg_map = seg_map.assign_coords(x=chip.x.values, y=chip.y.values)
    seg_map = mask_segmentation_map(chip, seg_map, chip_no_data_value)
    assert seg_map.where(seg_map != NO_DATA_VALUES.SEG_MAP).count().values == 0


def test_segmentation_map_masking_pass():
    chip = np.array([[1, 2, 3, 4], [1, 3, -9, 7], [6, 7, 3, 9]])
    chip = xr.DataArray(
        chip,
        dims=["band", "x"],
        coords={"band": np.arange(chip.shape[0]), "x": np.arange(chip.shape[1])},
    )
    seg_map = np.array([[1, -1, 1, 2]])
    seg_map = xr.DataArray(
        seg_map,
        dims=["band", "x"],
        coords={"band": np.arange(seg_map.shape[0]), "x": np.arange(seg_map.shape[1])},
    )
    seg_no_data_value = -1
    chip_no_data_value = -9

    # test each masking strategy
    seg_map = mask_segmentation_map(
        chip, seg_map, chip_no_data_value=chip_no_data_value, masking_strategy="each"
    )
    assert seg_map.where(seg_map != seg_no_data_value).count().values > 0
    np.testing.assert_array_equal(np.array([[1, -1, 1, 2]]), seg_map.values)

    # test any masking strategy
    seg_map = mask_segmentation_map(
        chip, seg_map, chip_no_data_value=chip_no_data_value, masking_strategy="any"
    )
    assert seg_map.where(seg_map != seg_no_data_value).count().values > 0
    np.testing.assert_array_equal(np.array([[1, -1, -1, 2]]), seg_map.values)


def test_segmentation_map_masking_fail():
    chip = np.array([[1, 2, 3, 4], [-9, -9, -9, -9], [6, 7, 3, 9]])
    chip = xr.DataArray(
        chip,
        dims=["band", "x"],
        coords={"band": np.arange(chip.shape[0]), "x": np.arange(chip.shape[1])},
    )
    seg_map = np.array([[1, -1, 1, 2]])
    seg_map = xr.DataArray(
        seg_map,
        dims=["band", "x"],
        coords={"band": np.arange(seg_map.shape[0]), "x": np.arange(seg_map.shape[1])},
    )
    seg_no_data_value = -1
    chip_no_data_value = -9
    seg_map = mask_segmentation_map(chip, seg_map, chip_no_data_value=chip_no_data_value)
    assert seg_map.where(seg_map != seg_no_data_value).count().values == 0
    np.testing.assert_array_equal(np.array([[-1, -1, -1, -1]]), seg_map.values)
