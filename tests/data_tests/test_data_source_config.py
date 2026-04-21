import numpy as np
import pytest

from instageo.data.data_source_config import (
    get_config,
    get_supported_input_types,
    get_supported_sources,
)


def test_get_config_hls():
    config = get_config("HLS")
    assert config.data_source_name == "HLS"
    assert config.mask_band == "Fmask"
    assert config.field_prefix == "hls"
    assert config.sign_func is None
    assert config.clip_range == (0, 10000)
    assert config.supports_cloud_filtering is True
    assert config.chip_dtype == np.uint16


def test_get_config_s2():
    config = get_config("S2")
    assert config.data_source_name == "S2"
    assert config.mask_band == "SCL"
    assert config.field_prefix == "s2"
    assert config.clip_range == (0, 10000)
    assert config.supports_cloud_filtering is True
    assert config.chip_dtype == np.uint16


def test_get_config_s1():
    config = get_config("S1")
    assert config.data_source_name == "S1"
    assert config.mask_band == ""
    assert config.mask_decoder is None
    assert config.field_prefix == "s1"
    assert config.clip_range == (0.0, 0.5)
    assert config.supports_cloud_filtering is False
    assert config.chip_dtype == np.float32


def test_get_config_unknown():
    with pytest.raises(ValueError, match="Unknown data source"):
        get_config("UNKNOWN")


def test_get_supported_sources():
    assert set(get_supported_sources()) == {"HLS", "S2", "S1"}


def test_get_supported_input_types():
    assert set(get_supported_input_types()) == {"point", "raster"}


@pytest.mark.parametrize("data_source", ["HLS", "S2", "S1"])
def test_config_settings_properties(data_source):
    config = get_config(data_source)
    assert config.api_url is not None
    assert len(config.collections) > 0
    assert len(config.bands_asset) > 0
    assert config.blocksize[0] > 0 and config.blocksize[1] > 0
    assert config.no_data_value is not None
    assert isinstance(config.bands_nameplate, dict)


@pytest.mark.parametrize(
    "data_source, has_mask_decoder",
    [("HLS", True), ("S2", True), ("S1", False)],
)
def test_config_callable_fields(data_source, has_mask_decoder):
    config = get_config(data_source)
    assert callable(config.setup_func)
    assert callable(config.tile_id_extractor)
    if has_mask_decoder:
        assert callable(config.mask_decoder)
    else:
        assert config.mask_decoder is None
