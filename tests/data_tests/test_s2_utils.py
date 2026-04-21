import numpy as np
import xarray as xr

from instageo.data.s2_utils import create_mask_from_scl


def test_create_mask_from_valid_scl_data():
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    coords = {"x": [0, 1], "y": [0, 1], "band": [1, 2]}
    sample_dataset = xr.Dataset({"bands": (["band", "y", "x"], data)}, coords=coords)
    sample_scl_data = xr.DataArray(np.array([[1, 2], [3, 4]]))

    class_ids = [2, 3]
    mask = create_mask_from_scl(sample_scl_data, class_ids)
    result = sample_dataset.where(mask.values == 0)
    expected_data = np.array(
        [
            [[1, np.nan], [np.nan, 4]],
            [[5, np.nan], [np.nan, 8]],
        ]
    )
    assert "bands" in result
    assert result["bands"].shape == sample_dataset["bands"].shape
    np.testing.assert_almost_equal(result["bands"].values, expected_data)


def test_create_mask_from_scl_data_no_classes():
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    coords = {"x": [0, 1], "y": [0, 1], "band": [1, 2]}
    sample_dataset = xr.Dataset({"bands": (["band", "y", "x"], data)}, coords=coords)
    sample_scl_data = xr.DataArray(np.array([[1, 2], [3, 4]]))

    class_ids = []
    mask = create_mask_from_scl(sample_scl_data, class_ids)
    result = sample_dataset.where(mask.values == 0)
    xr.testing.assert_equal(result, sample_dataset)
