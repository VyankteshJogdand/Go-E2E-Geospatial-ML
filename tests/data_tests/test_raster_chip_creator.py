import json
import os
import pathlib
import shutil

import geopandas as gpd
import mgrs
import numpy as np
import pytest
import xarray as xr
from absl import flags
from shapely.geometry import box

from instageo.data.raster_chip_creator import main
from instageo.data.settings import DataPipelineSettings

FLAGS = flags.FLAGS

test_root = pathlib.Path(__file__).parent.resolve()
test_data_root = test_root.parent / "data"

_settings = DataPipelineSettings()
DATA_SOURCE_SPATIAL_RES = {
    "HLS": _settings.HLS_SPATIAL_RESOLUTION,
    "S2": _settings.S2_SPATIAL_RESOLUTION,
    "S1": _settings.S1_SPATIAL_RESOLUTION,
}

SEG_MAP_TIF = str(test_data_root / "label_1_18TWL.tif")
SEG_MAP_BOUNDS = (-74.0, 40.76683465713849, -73.97700312872654, 40.78983152841195)


def get_mgrs_tile(lon, lat):
    m = mgrs.MGRS()
    return m.toMGRS(lat, lon, MGRSPrecision=0)


@pytest.fixture
def output_dir():
    path = "/tmp/test_raster_chip_creator"
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def records_file(output_dir):
    """Create a records gpkg from the existing segmentation tif bounds."""
    left, bottom, right, top = SEG_MAP_BOUNDS
    cx, cy = (left + right) / 2, (bottom + top) / 2

    gdf = gpd.GeoDataFrame(
        {
            "date": ["2025-07-31"],
            "mgrs_tile_id": [get_mgrs_tile(cx, cy)],
            "label_filename": [os.path.basename(SEG_MAP_TIF)],
        },
        geometry=[box(left, bottom, right, top)],
        crs="EPSG:4326",
    )
    path = os.path.join(output_dir, "records.gpkg")
    gdf.to_file(path, driver="GPKG")
    return path


@pytest.fixture
def bbox_json(output_dir):
    """Create a bbox JSON file covering the same area."""
    bbox_features = [list(SEG_MAP_BOUNDS)]
    path = os.path.join(output_dir, "bbox_features.json")
    with open(path, "w") as f:
        json.dump(bbox_features, f)
    return path


def base_flags(output_dir, data_source="HLS"):
    return [
        "raster_chip_creator",
        "--output_directory",
        output_dir,
        "--data_source",
        data_source,
        "--chip_size",
        "256",
        "--masking_strategy",
        "any",
        "--num_steps",
        "1",
        "--temporal_step",
        "30",
        "--temporal_tolerance",
        "10",
        "--cloud_coverage",
        "50",
        "--nodaytime_only",
        "--qa_check",
    ]


@pytest.mark.auth
@pytest.mark.parametrize("data_source", ["HLS", "S2", "S1"])
def test_raster_chip_creator_standard(output_dir, records_file, data_source):
    FLAGS.unparse_flags()
    FLAGS(
        base_flags(output_dir, data_source)
        + [
            "--records_file",
            records_file,
            "--raster_path",
            str(test_data_root),
        ]
    )
    main(None)

    chips = os.listdir(os.path.join(output_dir, "chips"))
    seg_maps = os.listdir(os.path.join(output_dir, "seg_maps"))
    assert len(chips) > 0
    assert len(chips) == len(seg_maps)

    chip = xr.open_dataset(os.path.join(output_dir, "chips", chips[0]))
    seg_map = xr.open_dataset(os.path.join(output_dir, "seg_maps", seg_maps[0]))
    assert chip.band_data.shape[-2:] == (256, 256)
    assert np.unique(chip.band_data).size > 1
    assert seg_map.band_data.shape[-2:] == (256, 256)


@pytest.mark.auth
@pytest.mark.parametrize("data_source", ["HLS", "S2", "S1"])
def test_raster_chip_creator_bbox(output_dir, bbox_json, data_source):
    FLAGS.unparse_flags()
    FLAGS(
        base_flags(output_dir, data_source)
        + [
            "--is_bbox_feature",
            "--bbox_feature_path",
            bbox_json,
            "--date",
            "15-06-2023",
            "--spatial_resolution",
            str(DATA_SOURCE_SPATIAL_RES[data_source]),
        ]
    )
    main(None)

    chips = os.listdir(os.path.join(output_dir, "chips"))
    assert len(chips) > 0

    chip = xr.open_dataset(os.path.join(output_dir, "chips", chips[0]))
    assert chip.band_data.shape[-2:] == (256, 256)
    assert np.unique(chip.band_data).size > 1

    # No seg_maps for bbox feature mode
    assert (
        not os.path.exists(os.path.join(output_dir, "seg_maps"))
        or len(os.listdir(os.path.join(output_dir, "seg_maps"))) == 0
    )
