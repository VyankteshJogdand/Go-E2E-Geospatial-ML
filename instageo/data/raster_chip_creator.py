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
"""InstaGeo Chip Creator from Raster Module.

This module creates chips from raster-based label data or bounding boxes.
"""

import json
import logging as pylogging
from datetime import datetime
from typing import Any, Literal

import geopandas as gpd
from absl import app, flags, logging

from instageo.data import geo_utils
from instageo.data.base_chip_creator import BaseChipCreator
from instageo.data.flags import FLAGS

logging.set_verbosity(logging.INFO)
log = pylogging.getLogger(__name__)
log.setLevel(pylogging.WARNING)
pylogging.getLogger("botocore.credentials").setLevel(pylogging.WARNING)
pylogging.getLogger("earthaccess").setLevel(pylogging.WARNING)

# Define flags specific to raster-based chip creation
flags.DEFINE_string("records_file", None, "Path to input records file containing geometries.")
flags.DEFINE_string("raster_path", None, "Path to input raster file.")

flags.DEFINE_bool(
    "qa_check", True, "Whether to perform quality assurance check on chip and seg_map."
)

flags.DEFINE_bool(
    "is_bbox_feature",
    False,
    "Whether to use a bounding box feature file.",
)

flags.DEFINE_string(
    "bbox_feature_path",
    None,
    "Path to a JSON file containing a list of bounding boxes.",
)

flags.DEFINE_string("date", None, "Date of the observations.")


class RasterBasedChipCreator(BaseChipCreator):
    """Chip creator for raster-based label data.

    This class handles chip creation from:
    1. Existing raster files with labels
    2. Bounding box feature files (generates grid polygons)
    """

    def get_input_type(self) -> Literal["point", "raster"]:
        """Return input type for factory.

        Returns:
            "raster" to select raster-based pipelines
        """
        return "raster"

    def get_observations(self) -> gpd.GeoDataFrame:
        """Load observations from raster records or bounding box features.

        Returns:
            GeoDataFrame with observation records including geometry
        """
        if self.flags.is_bbox_feature:
            # Create grid polygons from bounding box features
            with open(self.flags.bbox_feature_path) as json_file:
                bb_feature = json.load(json_file)

            obsv_records = geo_utils.create_grid_polygons(
                bbox_list=bb_feature,
                date=self.flags.date
                if self.flags.date
                else datetime.strftime(datetime.now(), "%d-%m-%Y"),
                chip_size=self.flags.chip_size,
                spatial_resolution=self.flags.spatial_resolution,
                crs=self.flags.src_crs,
            )
        else:
            # Load from records file
            obsv_records = gpd.read_file(self.flags.records_file)
            obsv_records["geometry_4326"] = obsv_records["geometry"].to_crs("EPSG:4326")
            obsv_records = self.apply_date_offsets(obsv_records)

        return obsv_records


def main(argv: Any) -> None:
    """Raster Chip Creator.

    Given a raster file containing label information, the Raster Chip Creator
    creates small chips from larger satellite tiles which are suitable for
    training segmentation models.

    Supports all data sources: HLS, S2, and S1.
    """
    del argv

    # Create and run raster-based chip creator
    creator = RasterBasedChipCreator(FLAGS)
    creator.run()


if __name__ == "__main__":
    app.run(main)
