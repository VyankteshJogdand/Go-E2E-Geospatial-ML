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

"""Utility Functions for Reading and Processing Harmonized Landsat Sentinel-2 Dataset."""

from typing import Any

import earthaccess
import rasterio
import xarray as xr

from instageo.data.settings import GDALOptions


def decode_fmask_value(
    value: xr.Dataset | xr.DataArray, position: int
) -> xr.Dataset | xr.DataArray:
    """Decodes HLS v2.0 Fmask.

    Returns:
        Xarray dataset containing decoded bits.
    """
    quotient = value // (2**position)
    return quotient - ((quotient // 2) * 2)


def hls_setup() -> None:
    """Setup for HLS data source (authentication and GDAL environment)."""
    earthaccess.login(persist=True)
    rasterio.Env(**GDALOptions().model_dump()).__enter__()


def hls_extract_tile_id(_obsv_records: Any, tile_dict: dict[str, Any]) -> str:
    """Extract HLS tile ID from granule ID.

    Args:
        _obsv_records: Observation records (unused - tile ID in granule)
        tile_dict: Dictionary containing granule information

    Returns:
        Tile ID string (e.g., "S30_T10SEG_2023001T")
    """
    splits = tile_dict["granules"][0]["id"].split(".")
    return f"{splits[1]}_{splits[2]}_{splits[3]}"
