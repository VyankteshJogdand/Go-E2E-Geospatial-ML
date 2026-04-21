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

"""Data source configuration aggregating settings and custom logic."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from instageo.data.settings import (
    HLSAPISettings,
    HLSBandsSettings,
    HLSBlockSizes,
    NoDataValues,
    S1APISettings,
    S1BandsSettings,
    S1BlockSizes,
    S2APISettings,
    S2BandsSettings,
    S2BlockSizes,
)

NO_DATA_VALUES = NoDataValues()

# Aggregate existing settings per data source
SETTINGS_MAP: dict[str, dict[str, Any]] = {
    "HLS": {
        "api_url": HLSAPISettings().URL,
        "collections": HLSAPISettings().COLLECTIONS,
        "bands_asset": HLSBandsSettings().ASSET,
        "bands_nameplate": HLSBandsSettings().NAMEPLATE,
        "blocksize": (HLSBlockSizes().X, HLSBlockSizes().Y),
        "no_data_value": NO_DATA_VALUES.HLS,
    },
    "S2": {
        "api_url": S2APISettings().URL,
        "collections": S2APISettings().COLLECTIONS,
        "bands_asset": S2BandsSettings().ASSET,
        "bands_nameplate": S2BandsSettings().NAMEPLATE,
        "blocksize": (S2BlockSizes().X, S2BlockSizes().Y),
        "no_data_value": NO_DATA_VALUES.S2,
    },
    "S1": {
        "api_url": S1APISettings().URL,
        "collections": S1APISettings().COLLECTIONS,
        "bands_asset": S1BandsSettings().ASSET,
        "bands_nameplate": S1BandsSettings().NAMEPLATE,
        "blocksize": (S1BlockSizes().X, S1BlockSizes().Y),
        "no_data_value": NO_DATA_VALUES.S1,
    },
}


@dataclass(frozen=True)
class DataSourceConfig:
    """Configuration for data source-specific behavior.

    Aggregates settings from settings.py and adds custom functions/parameters
    that aren't in settings.

    Attributes:
        data_source_name: Name of the data source (e.g., "HLS", "S2", "S1")
        setup_func: Function to run for data source setup (e.g., authentication)
        mask_decoder: Function to decode quality masks (None for SAR data)
        tile_id_extractor: Function to extract tile ID from observation records/dict
        mask_band: Name of the mask band (e.g., "Fmask", "SCL", "")
        sign_func: Function to sign STAC URLs (None or planetary_computer.sign)
        clip_range: Optional range for clipping chip values (e.g., (0, 10000))
        field_prefix: Prefix for STAC item field names (e.g., "hls", "s2", "s1")
        supports_cloud_filtering: Whether data source supports cloud coverage filtering
        chip_dtype: NumPy dtype for saving chips (e.g., np.uint16 for optical, np.float32 for SAR)
    """

    data_source_name: str

    # Custom functions (not in settings)
    setup_func: Callable[[], None]
    mask_decoder: Callable | None
    tile_id_extractor: Callable[[Any, dict[str, Any]], str]

    # Parameters not in settings
    mask_band: str
    sign_func: Callable | None
    field_prefix: str
    clip_range: tuple[float, float] | None = None
    supports_cloud_filtering: bool = True  # False for SAR data like S1
    chip_dtype: type = np.uint16

    # Properties that retrieve from settings
    @property
    def api_url(self) -> str:
        """Get STAC API URL from settings."""
        return SETTINGS_MAP[self.data_source_name]["api_url"]

    @property
    def collections(self) -> list[str]:
        """Get STAC collections from settings."""
        return SETTINGS_MAP[self.data_source_name]["collections"]

    @property
    def bands_asset(self) -> list[str]:
        """Get band assets from settings."""
        return SETTINGS_MAP[self.data_source_name]["bands_asset"]

    @property
    def blocksize(self) -> tuple[int, int]:
        """Get block size from settings."""
        return SETTINGS_MAP[self.data_source_name]["blocksize"]

    @property
    def no_data_value(self) -> int:
        """Get no-data value from settings."""
        return SETTINGS_MAP[self.data_source_name]["no_data_value"]

    @property
    def bands_nameplate(self) -> dict[str, dict[str, str]]:
        """Get bands nameplate mapping from settings."""
        return SETTINGS_MAP[self.data_source_name]["bands_nameplate"]


def get_config(data_source: str) -> DataSourceConfig:
    """Get configuration for a data source.

    Imports are done here to avoid circular dependencies.
    Configs are built on-demand.

    Args:
        data_source: Data source name ("HLS", "S2", "S1", etc.)

    Returns:
        DataSourceConfig for the data source

    Raises:
        ValueError: If data_source is not supported
    """
    if data_source == "HLS":
        from instageo.data.hls_utils import (
            decode_fmask_value,
            hls_extract_tile_id,
            hls_setup,
        )

        return DataSourceConfig(
            data_source_name="HLS",
            setup_func=hls_setup,
            mask_decoder=decode_fmask_value,
            tile_id_extractor=hls_extract_tile_id,
            mask_band="Fmask",
            sign_func=None,
            clip_range=(0, 10000),
            field_prefix="hls",
        )

    elif data_source == "S2":
        from planetary_computer import sign

        from instageo.data.s2_utils import (
            create_mask_from_scl,
            s2_extract_tile_id,
            s2_setup,
        )

        return DataSourceConfig(
            data_source_name="S2",
            setup_func=s2_setup,
            mask_decoder=create_mask_from_scl,
            tile_id_extractor=s2_extract_tile_id,
            mask_band="SCL",
            sign_func=sign,
            clip_range=(0, 10000),
            field_prefix="s2",
        )

    elif data_source == "S1":
        from planetary_computer import sign

        from instageo.data.s1_utils import s1_extract_tile_id, s1_setup

        return DataSourceConfig(
            data_source_name="S1",
            setup_func=s1_setup,
            mask_decoder=None,
            tile_id_extractor=s1_extract_tile_id,
            mask_band="",
            sign_func=sign,
            clip_range=(0.0, 0.5),
            field_prefix="s1",
            supports_cloud_filtering=False,  # SAR data has no clouds
            chip_dtype=np.float32,
        )

    else:
        raise ValueError(f"Unknown data source: {data_source}. Supported: 'HLS', 'S2', 'S1'")


def get_supported_sources() -> list[str]:
    """Get list of supported data sources.

    Returns:
        List of supported data source names (e.g., ["HLS", "S2", "S1"])
    """
    return ["HLS", "S2", "S1"]


def get_supported_input_types() -> list[str]:
    """Get list of supported input types.

    Returns:
        List of supported input types (["point", "raster"])
    """
    return ["point", "raster"]
