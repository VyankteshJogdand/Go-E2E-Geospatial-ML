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
"""Base Chip Creator Module.

This module provides a base class for chip creators that extracts common logic
for handling STAC workflows, caching, and pipeline instantiation.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Literal

import geopandas as gpd
from pystac_client import Client

from instageo.data.stac_utils import add_stac_items, create_records_with_items


class BaseChipCreator(ABC):
    """Base class for chip creators.

    This class handles common logic for:
    - Setting up output directories
    - Managing STAC workflow with caching
    - Instantiating the correct pipeline using the factory
    - Running the pipeline

    Subclasses must implement:
    - get_observations(): Return observation records
    - get_input_type(): Return "point" or "raster"
    """

    def __init__(self, flags: Any):
        """Initialize base chip creator.

        Args:
            flags: Command-line flags object (absl FLAGS)
        """
        from instageo.data.data_source_config import get_config

        self.flags = flags
        self.data_source = flags.data_source
        self.output_directory = flags.output_directory
        self.config = get_config(self.data_source)  # Create config once

    @abstractmethod
    def get_observations(self) -> gpd.GeoDataFrame:
        """Get observation records as a GeoDataFrame.

        This method is implemented differently by subclasses:
        - PointBasedChipCreator: Loads from CSV/Parquet
        - RasterBasedChipCreator: Loads from raster or creates from bbox

        Returns:
            GeoDataFrame with observation records
        """
        pass

    @abstractmethod
    def get_input_type(self) -> Literal["point", "raster"]:
        """Get the input type for this chip creator.

        Returns:
            Either "point" or "raster"
        """
        pass

    def apply_date_offsets(self, data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Apply date adjustments shared by all chip creators.

        Combines time column with date if present, optionally shifts to month
        start, then computes input_features_date based on whether this is a
        forecasting task.

        Args:
            data: GeoDataFrame with a "date" column (and optionally a "time" column)

        Returns:
            GeoDataFrame with updated "date" and new "input_features_date" column
        """
        import pandas as pd

        data["date"] = pd.to_datetime(data["date"])

        # Combine time column with date if present (expected format: HH:MM:SS)
        if "time" in data.columns:
            data["date"] = data["date"] + pd.to_timedelta(data["time"])

        if self.flags.shift_to_month_start:
            data["date"] = data["date"] - pd.offsets.MonthBegin(1)

        data["input_features_date"] = (
            data["date"] - pd.DateOffset(days=self.flags.temporal_step)
            if self.flags.is_forecasting_task
            else data["date"]
        )
        return data

    def setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_directory, exist_ok=True)
        logging.info(f"Output directory: {self.output_directory}")

    def get_dataset_paths(self) -> tuple[str, str]:
        """Get paths for dataset JSON and filtered records.

        Returns:
            Tuple of (dataset_json_path, records_gpkg_path)
        """
        dataset_file = os.path.join(
            self.output_directory, f"{self.data_source.lower()}_dataset.json"
        )
        records_file = os.path.join(self.output_directory, "filtered_obsv_records.gpkg")
        return dataset_file, records_file

    def get_stac_client(self) -> Client:
        """Get STAC client for the data source using configuration.

        Returns:
            STAC client instance
        """
        return Client.open(self.config.api_url)

    def get_stac_fields(self) -> tuple[str, str]:
        """Get the granules and items field names for the data source.

        Returns:
            Tuple of (granules_field, items_field)
        """
        prefix = self.config.field_prefix
        return (f"{prefix}_granules", f"{prefix}_items")

    def get_or_create_dataset(
        self, obsv_records: gpd.GeoDataFrame
    ) -> tuple[dict[str, Any], gpd.GeoDataFrame]:
        """Get existing dataset or create new one using STAC.

        This method handles caching: if dataset JSON already exists, it loads it.
        Otherwise, it queries STAC, creates the dataset, and saves it.

        Args:
            obsv_records: Observation records GeoDataFrame

        Returns:
            Tuple of (dataset dict, filtered observation records)
        """
        dataset_file, records_file = self.get_dataset_paths()

        # Check cache
        if os.path.exists(dataset_file) and os.path.exists(records_file):
            logging.info(f"{self.data_source} dataset JSON already exists, loading from cache")
            try:
                with open(dataset_file) as f:
                    dataset = json.load(f)
            except json.JSONDecodeError:
                logging.warning(
                    f"Cached dataset file {dataset_file} is invalid JSON — re-creating."
                )
                os.remove(dataset_file)
            else:
                filtered_records = gpd.read_file(records_file)
                return dataset, filtered_records

        # Create new dataset
        logging.info(f"Creating {self.data_source} dataset JSON")
        logging.info(f"Retrieving {self.data_source} tile IDs for observations")
        self.setup_output_directory()

        # Get STAC client and add items
        client = self.get_stac_client()

        # Build kwargs for add_stac_items function from flags
        stac_kwargs = self._get_stac_kwargs()
        obsv_records_with_items = add_stac_items(client, obsv_records, self.config, **stac_kwargs)

        # Create records with items
        granules_field, items_field = self.get_stac_fields()
        filtered_records, dataset = create_records_with_items(
            obsv_records_with_items, granules_field, items_field
        )

        if not dataset:
            raise RuntimeError(
                f"STAC query returned no results for data source '{self.data_source}'. "
                f"Check your date range, bounding box, and STAC endpoint. "
                f"Output directory: {self.output_directory}"
            )

        # Save to cache
        with open(dataset_file, "w") as f:
            json.dump(dataset, f, indent=4)
        filtered_records.to_file(records_file, driver="GPKG")

        logging.info(f"Saved {self.data_source} dataset to {dataset_file}")
        return dataset, filtered_records

    def _get_stac_kwargs(self) -> dict[str, Any]:
        """Build kwargs for add_stac_items functions from flags.

        Returns:
            Dictionary of keyword arguments
        """
        kwargs = {}

        # Common parameters across all data sources
        if hasattr(self.flags, "num_steps"):
            kwargs["num_steps"] = self.flags.num_steps
        if hasattr(self.flags, "temporal_step"):
            kwargs["temporal_step"] = self.flags.temporal_step
        if hasattr(self.flags, "temporal_tolerance"):
            kwargs["temporal_tolerance"] = self.flags.temporal_tolerance
        if hasattr(self.flags, "temporal_tolerance_minutes"):
            kwargs["temporal_tolerance_minutes"] = self.flags.temporal_tolerance_minutes

        # Cloud coverage filtering (only for optical sensors)
        if self.config.supports_cloud_filtering:
            if hasattr(self.flags, "cloud_coverage"):
                kwargs["cloud_coverage"] = self.flags.cloud_coverage
        else:
            # SAR data has no clouds
            kwargs["cloud_coverage"] = None

        # Daytime filtering (applicable to all data sources)
        if hasattr(self.flags, "daytime_only"):
            kwargs["daytime_only"] = self.flags.daytime_only

        return kwargs

    def instantiate_pipeline(self):
        """Instantiate the correct pipeline with data source configuration.

        Returns:
            Pipeline instance (PointsDataPipeline or RasterDataPipeline)
        """
        from instageo.data.data_pipeline import PointsDataPipeline, RasterDataPipeline

        input_type = self.get_input_type()

        # Build common pipeline parameters
        pipeline_kwargs = {
            "config": self.config,  # Reuse the config created in __init__
            "output_directory": self.flags.output_directory,
            "chip_size": self.flags.chip_size,
            "mask_types": getattr(self.flags, "mask_types", []),
            "masking_strategy": self.flags.masking_strategy,
            "src_crs": self.flags.src_crs,
            "spatial_resolution": getattr(self.flags, "spatial_resolution", None),
            "task_type": self.flags.task_type,
        }

        # Instantiate the appropriate pipeline based on input type
        if input_type == "point":
            pipeline_kwargs["window_size"] = self.flags.window_size
            pipeline = PointsDataPipeline(**pipeline_kwargs)
            logging.info("Instantiating PointsDataPipeline")
        elif input_type == "raster":
            pipeline_kwargs["raster_path"] = self.flags.raster_path
            pipeline_kwargs["qa_check"] = getattr(self.flags, "qa_check", True)
            pipeline_kwargs["is_bbox_feature"] = getattr(self.flags, "is_bbox_feature", False)
            pipeline = RasterDataPipeline(**pipeline_kwargs)
            logging.info("Instantiating RasterDataPipeline")
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        return pipeline

    def run(self) -> None:
        """Main entry point: orchestrates the entire chip creation workflow.

        This method:
        1. Gets observation records
        2. Creates or loads dataset with STAC items
        3. Instantiates the appropriate pipeline
        4. Runs the pipeline to create chips
        """
        logging.info(f"Starting {self.__class__.__name__}")
        logging.info(f"Data source: {self.data_source}")
        logging.info(f"Input type: {self.get_input_type()}")

        # Step 1: Get observations
        obsv_records = self.get_observations()
        logging.info(f"Loaded {len(obsv_records)} observation records")

        # Step 2: Get or create dataset
        dataset, filtered_records = self.get_or_create_dataset(obsv_records)
        logging.info(f"Dataset contains {len(dataset)} tiles")

        # Step 3: Instantiate pipeline
        pipeline = self.instantiate_pipeline()

        # Step 4: Run pipeline
        logging.info("Creating chips and segmentation maps")
        pipeline.run(dataset, filtered_records)

        logging.info("Chip creation complete!")
