import json
import os
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from instageo.data.chip_creator import PointBasedChipCreator
from instageo.data.data_pipeline import PointsDataPipeline, RasterDataPipeline
from instageo.data.raster_chip_creator import RasterBasedChipCreator


def make_flags(data_source="HLS", **kwargs):
    flags = MagicMock()
    flags.data_source = data_source
    flags.output_directory = "/tmp/test_output"
    flags.chip_size = 256
    flags.masking_strategy = "any"
    flags.src_crs = 4326
    flags.spatial_resolution = 30.0
    flags.task_type = "seg"
    flags.mask_types = []
    flags.window_size = 0
    flags.raster_path = "/tmp/raster"
    flags.qa_check = True
    flags.is_bbox_feature = False
    flags.num_steps = 3
    flags.temporal_step = 30
    flags.temporal_tolerance = 5
    flags.cloud_coverage = 10
    flags.daytime_only = False
    for k, v in kwargs.items():
        setattr(flags, k, v)
    return flags


# -- Input type --


def test_point_based_chip_creator_input_type():
    creator = PointBasedChipCreator(make_flags())
    assert creator.get_input_type() == "point"


def test_raster_based_chip_creator_input_type():
    creator = RasterBasedChipCreator(make_flags())
    assert creator.get_input_type() == "raster"


# -- Dataset paths --


@pytest.mark.parametrize(
    "data_source, expected_json",
    [("HLS", "hls_dataset.json"), ("S2", "s2_dataset.json"), ("S1", "s1_dataset.json")],
)
def test_get_dataset_paths(data_source, expected_json):
    flags = make_flags(data_source, output_directory="/tmp/out")
    creator = PointBasedChipCreator(flags)
    dataset_file, records_file = creator.get_dataset_paths()
    assert dataset_file == f"/tmp/out/{expected_json}"
    assert records_file == "/tmp/out/filtered_obsv_records.gpkg"


# -- STAC fields --


@pytest.mark.parametrize(
    "data_source, expected_granules, expected_items",
    [
        ("HLS", "hls_granules", "hls_items"),
        ("S2", "s2_granules", "s2_items"),
        ("S1", "s1_granules", "s1_items"),
    ],
)
def test_get_stac_fields(data_source, expected_granules, expected_items):
    creator = PointBasedChipCreator(make_flags(data_source))
    granules_field, items_field = creator.get_stac_fields()
    assert granules_field == expected_granules
    assert items_field == expected_items


# -- STAC kwargs --


@pytest.mark.parametrize("data_source", ["HLS", "S2"])
def test_get_stac_kwargs_optical(data_source):
    creator = PointBasedChipCreator(make_flags(data_source))
    kwargs = creator._get_stac_kwargs()
    assert kwargs["num_steps"] == 3
    assert kwargs["temporal_step"] == 30
    assert kwargs["temporal_tolerance"] == 5
    assert kwargs["cloud_coverage"] == 10
    assert kwargs["daytime_only"] is False


def test_get_stac_kwargs_sar():
    creator = PointBasedChipCreator(make_flags("S1"))
    kwargs = creator._get_stac_kwargs()
    assert kwargs["cloud_coverage"] is None


# -- Setup output directory --


def test_setup_output_directory(tmp_path):
    flags = make_flags(output_directory=str(tmp_path / "chips_out"))
    creator = PointBasedChipCreator(flags)
    creator.setup_output_directory()
    assert os.path.exists(creator.output_directory)


# -- Instantiate pipeline --


def test_instantiate_pipeline_points():
    creator = PointBasedChipCreator(make_flags())
    pipeline = creator.instantiate_pipeline()
    assert isinstance(pipeline, PointsDataPipeline)


def test_instantiate_pipeline_raster():
    creator = RasterBasedChipCreator(make_flags())
    pipeline = creator.instantiate_pipeline()
    assert isinstance(pipeline, RasterDataPipeline)


def test_instantiate_pipeline_invalid_input_type():
    creator = PointBasedChipCreator(make_flags())
    creator.get_input_type = lambda: "invalid"
    with pytest.raises(ValueError, match="Unknown input_type"):
        creator.instantiate_pipeline()


# -- Dataset cache --


def test_get_or_create_dataset_from_cache(tmp_path):
    flags = make_flags(output_directory=str(tmp_path))
    creator = PointBasedChipCreator(flags)

    dataset = {"tile1": {"granules": []}}
    dataset_file, records_file = creator.get_dataset_paths()

    with open(dataset_file, "w") as f:
        json.dump(dataset, f)

    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    gdf.to_file(records_file, driver="GPKG")

    result_dataset, result_records = creator.get_or_create_dataset(gpd.GeoDataFrame())
    assert result_dataset == dataset
    assert isinstance(result_records, gpd.GeoDataFrame)


@patch("instageo.data.base_chip_creator.create_records_with_items")
@patch("instageo.data.base_chip_creator.add_stac_items")
def test_get_or_create_dataset_fresh_creation(mock_add_stac, mock_create_records, tmp_path):
    flags = make_flags(output_directory=str(tmp_path))
    creator = PointBasedChipCreator(flags)
    creator.get_stac_client = MagicMock()

    obsv_records = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    filtered_records = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    dataset = {"tile1": {"granules": []}}
    mock_add_stac.return_value = obsv_records
    mock_create_records.return_value = (filtered_records, dataset)

    result_dataset, _ = creator.get_or_create_dataset(obsv_records)

    assert result_dataset == dataset
    dataset_file, records_file = creator.get_dataset_paths()
    assert os.path.exists(dataset_file)
    assert os.path.exists(records_file)


@patch("instageo.data.base_chip_creator.create_records_with_items")
@patch("instageo.data.base_chip_creator.add_stac_items")
def test_get_or_create_dataset_raises_on_empty_stac(mock_add_stac, mock_create_records, tmp_path):
    flags = make_flags(output_directory=str(tmp_path))
    creator = PointBasedChipCreator(flags)
    creator.get_stac_client = MagicMock()

    obsv_records = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    mock_add_stac.return_value = obsv_records
    mock_create_records.return_value = (gpd.GeoDataFrame(), {})

    with pytest.raises(RuntimeError, match="STAC query returned no results"):
        creator.get_or_create_dataset(obsv_records)


# -- apply_date_offsets --


@pytest.mark.parametrize(
    "shift, forecast, expected_date, expected_features_date",
    [
        # shift=False, forecast=False → both dates unchanged
        (False, False, "2023-06-15", "2023-06-15"),
        # shift=False, forecast=True → input_features_date offset back by 30 days
        (False, True, "2023-06-15", "2023-05-16"),
        # shift=True, forecast=False → date shifted to month start, input_features_date == date
        (True, False, "2023-06-01", "2023-06-01"),
        # shift=True, forecast=True → date shifted then input_features_date offset back
        (True, True, "2023-06-01", "2023-05-02"),
    ],
)
def test_apply_date_offsets(shift, forecast, expected_date, expected_features_date):
    creator = PointBasedChipCreator(
        make_flags(shift_to_month_start=shift, is_forecasting_task=forecast, temporal_step=30)
    )
    data = gpd.GeoDataFrame({"date": ["2023-06-15"], "geometry": [Point(0, 0)]})
    result = creator.apply_date_offsets(data)
    assert result["date"].iloc[0] == pd.Timestamp(expected_date)
    assert result["input_features_date"].iloc[0] == pd.Timestamp(expected_features_date)


def test_apply_date_offsets_time_column_combined():
    """Time column is combined with date before any other offset is applied."""
    creator = PointBasedChipCreator(
        make_flags(shift_to_month_start=False, is_forecasting_task=False, temporal_step=30)
    )
    data = gpd.GeoDataFrame(
        {"date": ["2023-06-15"], "time": ["06:30:00"], "geometry": [Point(0, 0)]}
    )
    result = creator.apply_date_offsets(data)
    assert result["date"].iloc[0] == pd.Timestamp("2023-06-15 06:30:00")


# -- BaseChipCreator.run() orchestration --


def test_base_chip_creator_run_orchestration():
    creator = PointBasedChipCreator(make_flags())

    obsv_records = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    dataset = {"tile1": {"granules": []}}
    filtered_records = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    mock_pipeline = MagicMock()

    creator.get_observations = MagicMock(return_value=obsv_records)
    creator.get_or_create_dataset = MagicMock(return_value=(dataset, filtered_records))
    creator.instantiate_pipeline = MagicMock(return_value=mock_pipeline)

    creator.run()

    creator.get_observations.assert_called_once()
    creator.get_or_create_dataset.assert_called_once_with(obsv_records)
    creator.instantiate_pipeline.assert_called_once()
    mock_pipeline.run.assert_called_once_with(dataset, filtered_records)
