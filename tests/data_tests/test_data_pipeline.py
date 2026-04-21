import os
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

from instageo.data.data_pipeline import (
    BaseDataPipeline,
    PointsDataPipeline,
    RasterDataPipeline,
    adjust_dims,
    assert_in_dtype_range,
    get_chip_coords,
    get_tile_info,
    get_tiles,
    reproject_coordinates,
)
from instageo.data.data_source_config import get_config


@pytest.fixture
def observation_data():
    data = pd.DataFrame(
        {
            "date": {
                0: "2022-06-08",
                1: "2022-06-08",
                2: "2022-06-08",
                3: "2022-06-08",
                4: "2022-06-09",
                5: "2022-06-09",
                6: "2022-06-09",
                7: "2022-06-08",
                8: "2022-06-09",
                9: "2022-06-09",
            },
            "x": {
                0: 44.48,
                1: 44.48865,
                2: 46.437787,
                3: 49.095545,
                4: -0.1305,
                5: 44.6216,
                6: 49.398908,
                7: 44.451435,
                8: 49.435228,
                9: 44.744167,
            },
            "y": {
                0: 15.115617,
                1: 15.099767,
                2: 14.714659,
                3: 16.066929,
                4: 28.028967,
                5: 16.16195,
                6: 16.139727,
                7: 15.209633,
                8: 16.151837,
                9: 15.287778,
            },
            "year": {
                0: 2022,
                1: 2022,
                2: 2022,
                3: 2022,
                4: 2022,
                5: 2022,
                6: 2022,
                7: 2022,
                8: 2022,
                9: 2022,
            },
        }
    )
    data["date"] = pd.to_datetime(data["date"])
    data["input_features_date"] = data["date"]
    return data


@pytest.fixture
def setup_and_teardown_output_dir():
    output_dir = "/tmp/test_hls"
    os.makedirs(output_dir, exist_ok=True)
    yield
    shutil.rmtree(output_dir)


def test_get_tiles(observation_data):
    hls_tiles = get_tiles(data=observation_data, min_count=1)
    assert list(hls_tiles["mgrs_tile_id"]) == [
        "38PMB",
        "38PMB",
        "38PPB",
        "39QTT",
        "30RYS",
        "38QMC",
        "39QUT",
        "38PMB",
        "39QUT",
        "38PMB",
    ]


def test_get_chip_coords():
    df = pd.read_csv("tests/data/sample_4326.csv")
    df = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.x, df.y)])
    df.set_crs(epsg=4326, inplace=True)
    df = df.to_crs(crs=32613)

    ds = xr.open_dataset("tests/data/HLS.S30.T38PMB.2022145T072619.v2.0.B02.tif")
    chip_coords = {tuple(coords) for coords in get_chip_coords(df, ds, 64)}
    assert chip_coords == {
        (2, 0),
        (0, 3),
        (2, 2),
        (0, 3),
        (2, 0),
        (3, 2),
        (2, 3),
        (0, 3),
        (2, 3),
        (1, 2),
    }


def test_get_tile_info(observation_data):
    hls_tiles = get_tiles(observation_data, min_count=3)
    tiles_info, tile_queries = get_tile_info(hls_tiles, num_steps=3, temporal_step=5)
    pd.testing.assert_frame_equal(
        tiles_info,
        pd.DataFrame(
            {
                "tile_id": ["38PMB"],
                "min_date": ["2022-05-24T00:00:00"],
                "max_date": ["2022-06-14T23:59:59"],
                "lon_min": [44.451435],
                "lon_max": [44.744167],
                "lat_min": [15.099767],
                "lat_max": [15.287778],
            }
        ),
        check_like=True,
    )
    assert tile_queries == [
        (
            "38PMB",
            ["2022-06-08T00:00:00", "2022-06-03T00:00:00", "2022-05-29T00:00:00"],
        ),
        (
            "38PMB",
            ["2022-06-08T00:00:00", "2022-06-03T00:00:00", "2022-05-29T00:00:00"],
        ),
        (
            "38PMB",
            ["2022-06-08T00:00:00", "2022-06-03T00:00:00", "2022-05-29T00:00:00"],
        ),
        (
            "38PMB",
            ["2022-06-09T00:00:00", "2022-06-04T00:00:00", "2022-05-30T00:00:00"],
        ),
    ]


def test_adjust_dims():
    """Tests dimensionality of a chip array."""

    data = np.random.rand(3, 6, 100, 100)
    dummy_chip = xr.DataArray(
        data,
        dims=("time", "band", "y", "x"),
        coords={
            "time": np.arange(3),
            "band": np.arange(6),
            "y": np.arange(100),
            "x": np.arange(100),
        },
    )
    assert dummy_chip.dims == ("time", "band", "y", "x")
    assert dummy_chip.shape == (3, 6, 100, 100)

    # Collapse time dimension
    dummy_chip = adjust_dims(dummy_chip)
    assert dummy_chip.dims == ("band", "y", "x")
    assert dummy_chip.shape == (18, 100, 100)


@pytest.mark.parametrize(
    "x, y, expected_x, expected_y, source_epsg",
    [
        (0, 0, 0.0, 0.0, 3857),
        (10_000_000, 0, 89.83, 0.0, 3857),
        (5000000, 5000000, 44.91, 40.91, 3857),
        (0, -10_000_000, 0.0, -66.44, 3857),
        (-1000000, -1000000, -8.98, -8.94, 3857),
        (1000000, -500000, 8.98, -4.48, 3857),
        (500000, 6200000, 15, 55.94, 32633),
    ],
)
def test_reproject_coordinates(x, y, expected_x, expected_y, source_epsg):
    df = pd.DataFrame({"x": [x], "y": [y]})
    result_df = reproject_coordinates(df, source_epsg)

    assert np.isclose(result_df["x"][0], expected_x, atol=0.01)
    assert np.isclose(result_df["y"][0], expected_y, atol=0.01)


# -- assert_in_dtype_range --


@pytest.mark.parametrize(
    "value, dtype",
    [(0, np.uint16), (255, np.uint8), (32767, np.int16), (1.5, np.float32)],
)
def test_assert_in_dtype_range_valid(value, dtype):
    assert_in_dtype_range(value, dtype)  # should not raise


@pytest.mark.parametrize(
    "value, dtype",
    [(-1, np.uint16), (256, np.uint8), (32768, np.int16)],
)
def test_assert_in_dtype_range_invalid(value, dtype):
    with pytest.raises(AssertionError):
        assert_in_dtype_range(value, dtype)


# -- Pipeline init --


def make_points_pipeline(config, tmp_path, **kwargs):
    defaults = dict(
        chip_size=256,
        mask_types=[],
        masking_strategy="any",
        src_crs=4326,
        spatial_resolution=30.0,
    )
    defaults.update(kwargs)
    return PointsDataPipeline(config=config, output_directory=str(tmp_path), **defaults)


def test_points_pipeline_init(tmp_path):
    config = get_config("HLS")
    pipeline = make_points_pipeline(config, tmp_path, window_size=2, task_type="seg")
    assert pipeline.config is config
    assert pipeline.chip_size == 256
    assert pipeline.window_size == 2
    assert pipeline.task_type == "seg"
    assert pipeline.output_directory == str(tmp_path)


def test_raster_pipeline_init_with_labels(tmp_path):
    config = get_config("HLS")
    pipeline = RasterDataPipeline(
        config=config,
        output_directory=str(tmp_path),
        chip_size=128,
        raster_path="/tmp/raster_labels",
        mask_types=[],
        masking_strategy="each",
        src_crs=4326,
        spatial_resolution=30.0,
        is_bbox_feature=False,
    )
    assert pipeline.config is config
    assert pipeline.chip_size == 128
    assert pipeline.raster_path == "/tmp/raster_labels"
    assert pipeline.is_bbox_feature is False


def test_raster_pipeline_init_bbox(tmp_path):
    # When is_bbox_feature=True, no raster label path is used
    config = get_config("S2")
    pipeline = RasterDataPipeline(
        config=config,
        output_directory=str(tmp_path),
        chip_size=256,
        raster_path="",
        mask_types=[],
        masking_strategy="any",
        src_crs=4326,
        spatial_resolution=10.0,
        is_bbox_feature=True,
    )
    assert pipeline.is_bbox_feature is True
    assert pipeline.get_data_source_name() == "S2"


# -- BaseDataPipeline helper methods (parametrized over all sources) --


@pytest.mark.parametrize("data_source", ["HLS", "S2", "S1"])
def test_get_no_data_value(data_source, tmp_path):
    config = get_config(data_source)
    pipeline = make_points_pipeline(config, tmp_path)
    assert pipeline.get_no_data_value() == config.no_data_value


@pytest.mark.parametrize("data_source", ["HLS", "S2", "S1"])
def test_get_data_source_name(data_source, tmp_path):
    config = get_config(data_source)
    pipeline = make_points_pipeline(config, tmp_path)
    assert pipeline.get_data_source_name() == data_source


@pytest.mark.parametrize(
    "data_source, has_decoder",
    [("HLS", True), ("S2", True), ("S1", False)],
)
def test_get_mask_decoder(data_source, has_decoder, tmp_path):
    config = get_config(data_source)
    pipeline = make_points_pipeline(config, tmp_path)
    decoder = pipeline.get_mask_decoder()
    if has_decoder:
        assert callable(decoder)
    else:
        assert decoder is None


@pytest.mark.parametrize(
    "data_source, expected_clip_range",
    [("HLS", (0, 10000)), ("S2", (0, 10000)), ("S1", (0.0, 0.5))],
)
def test_get_clip_range(data_source, expected_clip_range, tmp_path):
    config = get_config(data_source)
    pipeline = make_points_pipeline(config, tmp_path)
    assert pipeline.get_clip_range() == expected_clip_range


@pytest.mark.parametrize("data_source", ["HLS", "S2", "S1"])
def test_is_array_empty(data_source, tmp_path):
    config = get_config(data_source)
    pipeline = make_points_pipeline(config, tmp_path)
    no_data = config.no_data_value
    all_nodata = xr.DataArray(np.full((3, 3), no_data, dtype=np.float32))
    mixed = xr.DataArray(np.array([[no_data, 1], [2, no_data]], dtype=np.float32))
    assert pipeline._is_array_empty(all_nodata, no_data)
    assert not pipeline._is_array_empty(mixed, no_data)


# -- _is_stac_item_processed --


def test_is_stac_item_processed(tmp_path):
    config = get_config("HLS")
    pipeline = make_points_pipeline(config, tmp_path, window_size=0, task_type="seg")
    obsv_records = pd.DataFrame(
        {
            "stac_items_str": ["item_key"],
            "mgrs_tile_id": ["38PMB"],
            "date": [pd.Timestamp("2023-06-15")],
        }
    )
    assert pipeline._is_stac_item_processed("item_key", obsv_records, {"20230615_38PMB"}) is True
    assert pipeline._is_stac_item_processed("item_key", obsv_records, set()) is False


def test_is_stac_item_processed_fallback_parsing(tmp_path):
    """When mgrs_tile_id is absent, tile_id is parsed from stac_items_str."""
    config = get_config("HLS")
    pipeline = make_points_pipeline(config, tmp_path, window_size=0, task_type="seg")
    stac_items_str = "HLS.S30.T38PMB.2022145T072619_item"
    obsv_records = pd.DataFrame(
        {
            "stac_items_str": [stac_items_str],
            "date": [pd.Timestamp("2022-05-25")],
        }
    )
    chip_base_id = "20220525_S30_T38PMB_2022145T072619"
    assert pipeline._is_stac_item_processed(stac_items_str, obsv_records, {chip_base_id}) is True
