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

"""InstaGeo Chip Creator Module.

This module creates chips from point-based observation data (CSV/Parquet files).
"""

import ast
import logging as std_logging
import os
from typing import Any, List, Literal, Tuple

import geopandas as gpd
import pandas as pd
from absl import app, flags, logging
from dotenv import load_dotenv
from shapely.geometry import Point

from instageo.data.base_chip_creator import BaseChipCreator
from instageo.data.data_pipeline import get_tiles
from instageo.data.flags import FLAGS

load_dotenv(os.path.expanduser("~/.credentials"))
logging.set_verbosity(logging.INFO)
std_logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
std_logging.getLogger("earthaccess").setLevel(std_logging.WARNING)

# Define flags specific to point-based chip creation
flags.DEFINE_string("dataframe_path", None, "Path to the DataFrame file.")

flags.DEFINE_integer("min_count", 100, "Minimum observation counts per tile", lower_bound=1)
flags.DEFINE_enum(
    "data_format",
    "csv",
    ["csv", "parquet"],
    """Format of the original file containing the observations. The data must contain
    columns named 'date', 'x', 'y', 'label'.
    In case of a Parquet file, the partitions must be done by the 'year' and 'mgrs_tile_id'
    columns.
    """,
)
flags.DEFINE_string(
    "filters",
    None,
    """List of filters to use. Filters must be provided as tuples following the
    structure ('col_to_filter_on' ? 'operator' ? value). Applies only in case of
    Parquet files.
    - The columns on which to filter and the operators should be provided as strings.
    - The operators allowed are ['==', '=', '>', '>=', '<', '<=', '!=', 'in', 'not in']
    Example: "('year' ? '==' ? 2016); ('mgrs_tile_id' ? '!=' ? 'BAN')"
    "('year' ? 'in' ? [2016, 2020]); ('mgrs_tile_id' ? 'not in' ? ['13SCS', 'BAN'])"
    """,
)


def parse_tuple_list(flag_value: str) -> List[Tuple]:
    """Converts a string into a list of tuples.

    Args:
        flag_value (str): String containing values to parse as list of tuples.
            Each tuple is separated by ';'. Within each tuple values are separated
            by '?'.

    Returns:
        List[Tuple]: List of tuples extracted from the original string.
    """
    try:
        return [tuple(item.strip("()").split("?")) for item in flag_value.split(";")]
    except Exception as e:
        raise ValueError(f"Error parsing string {flag_value} to extract filters list: {e}")


_VALID_FILTER_OPERATORS = ["==", "=", ">", ">=", "<", "<=", "!=", "in", "not in"]


def parse_filters(flag_value: str) -> List[Tuple[str, str, Any]]:
    """Converts a list of tuples into valid PyArrow filters.

    Args:
        flag_value (str): String containing values to parse as list of tuples.
            Each tuple is separated by ';'. Within each tuple values are separated
            by '?'.

    Returns:
        List of filter tuples (column, operator, value) ready for PyArrow.
    """
    try:
        filters = parse_tuple_list(flag_value)
        parsed_filters = []
        for f in filters:
            col, op, val = f
            try:
                col = ast.literal_eval(col)
                op = ast.literal_eval(op)
                val = ast.literal_eval(val)
            except Exception as e:
                raise flags.ValidationError(f"Could not properly parse filter {f}: {e}")
            if not isinstance(col, str):
                raise flags.ValidationError("Provide the filter column as a string")
            if op not in _VALID_FILTER_OPERATORS:
                raise flags.ValidationError(
                    f"Operator '{op}' is not allowed. Must be one of {_VALID_FILTER_OPERATORS}"
                )
            parsed_filters.append((col, op, val))
    except flags.ValidationError:
        raise
    except Exception as e:
        raise flags.ValidationError(
            f"Filters must be provided as tuples e.g. \"('col' ? '==' ? value); ...\": {e}"
        )
    return parsed_filters


def check_required_flags() -> None:
    """Check if required flags are provided."""
    required_flags = ["dataframe_path", "output_directory"]
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise app.UsageError(f"Flag --{flag_name} is required.")


class PointBasedChipCreator(BaseChipCreator):
    """Chip creator for point-based observation data (CSV/Parquet).

    This class handles chip creation from files containing point observations
    with columns: date, x, y, label.
    """

    def get_input_type(self) -> Literal["point", "raster"]:
        """Return input type for factory.

        Returns:
            "point" to select point-based pipelines
        """
        return "point"

    def get_observations(self) -> gpd.GeoDataFrame:
        """Load observations from CSV or Parquet file.

        Returns:
            GeoDataFrame with observation records including geometry
        """
        # Load data based on format
        if self.flags.data_format == "parquet":
            try:
                filters = None
                if self.flags.filters:
                    filters = parse_filters(self.flags.filters)
                data = pd.read_parquet(
                    self.flags.dataframe_path,
                    engine="pyarrow",
                    filters=filters,
                )
            except Exception as e:
                raise ValueError(f"Error loading Parquet file: {e}")
        else:
            data = pd.read_csv(self.flags.dataframe_path)

        data = self.apply_date_offsets(data)

        # Get tiles (groups observations by MGRS tile)
        sub_data = get_tiles(data, src_crs=self.flags.src_crs, min_count=self.flags.min_count)

        # Convert to GeoDataFrame with Point geometries
        geometry = [Point(xy) for xy in zip(sub_data["x"], sub_data["y"])]
        gdf = gpd.GeoDataFrame(sub_data, geometry=geometry, crs=f"EPSG:{self.flags.src_crs}")
        gdf["geometry_4326"] = gdf["geometry"].to_crs("EPSG:4326")

        return gdf


def main(argv: Any) -> None:
    """CSV/Parquet Chip Creator.

    Given a file containing geo-located point observations and labels, the Chip
    Creator creates small chips from larger tiles which are suitable for training
    machine learning models.
    """
    del argv

    # Check required flags
    check_required_flags()

    # Create and run point-based chip creator
    creator = PointBasedChipCreator(FLAGS)
    creator.run()


if __name__ == "__main__":
    app.run(main)
