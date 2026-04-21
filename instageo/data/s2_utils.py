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

"""Utility Functions for Reading and Processing Sentinel-2 Dataset."""

from typing import Any

import numpy as np
import xarray as xr


def create_mask_from_scl(
    scl_data: xr.Dataset | xr.DataArray, class_ids: list[int]
) -> xr.Dataset | xr.DataArray:
    """Creates masks based on SCL data .

    Arguments:
        scl_data: SCL input xarray Dataset or DataArray.
        class_ids: Class ids to use to produce the mask.

    Returns:
        Xarray dataset or dataarray containing the produced mask.
    """
    return scl_data.isin(class_ids).astype(np.int8)


def s2_setup() -> None:
    """Setup for S2 data source (no special setup needed)."""
    pass


def s2_extract_tile_id(obsv_records: Any, _tile_dict: dict[str, Any]) -> str:
    """Extract tile ID from MGRS tile ID field.

    Args:
        obsv_records: GeoDataFrame with 'mgrs_tile_id' column
        _tile_dict: Dictionary containing granule information (unused)

    Returns:
        MGRS tile ID string
    """
    return obsv_records.iloc[0]["mgrs_tile_id"]
