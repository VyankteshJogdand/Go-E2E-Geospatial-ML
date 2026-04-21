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

"""Utility Functions for generating Sentinel-1 chips."""

from typing import Any


def s1_setup() -> None:
    """Setup for S1 data source (no special setup needed)."""
    pass


def s1_extract_tile_id(obsv_records: Any, _tile_dict: dict[str, Any]) -> str:
    """Extract tile ID from MGRS tile ID field.

    Args:
        obsv_records: GeoDataFrame with 'mgrs_tile_id' column
        _tile_dict: Dictionary containing granule information (unused)

    Returns:
        MGRS tile ID string
    """
    return obsv_records.iloc[0]["mgrs_tile_id"]
