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

"""Utility Functions for Reading and Processing Items from STAC Catalogues."""

import logging
import time
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import backoff
import geopandas as gpd

# Utility functions for STAC-based processing (shared between HLS and S2)
import pandas as pd
import ratelimit
import stackstac
import xarray as xr
from astral import LocationInfo
from astral.sun import sun
from pystac.item import Item
from pystac.item_collection import ItemCollection
from pystac_client import Client
from pystac_client.exceptions import APIError
from shapely.geometry import box
from shapely.ops import unary_union

from instageo.data import geo_utils
from instageo.data.data_pipeline import adjust_dims
from instageo.data.settings import DataPipelineSettings, S1BandsSettings

S1_BANDS = S1BandsSettings()

DATA_PIPELINE_SETTINGS = DataPipelineSettings()


def is_valid_dataset_entry(obsv: pd.Series, item_id_field: str) -> bool:
    """Checks STAC granules validity for a given observation.

    The granules will be added to the dataset if they are all non
    null and unique for all timesteps.

    Args:
        obsv (pandas.Series): Observation data for which to assess the validity

    Returns:
        True if the granules are unique and non null.
    """
    if any(granule is None for granule in obsv[item_id_field]) or (
        len(obsv[item_id_field]) != len(set(obsv[item_id_field]))
    ):
        return False
    return True


def is_daytime(item: Item) -> bool:
    """Check if it's daytime.

    Checks if it's daytime using `datetime` and `bbox` properties of a PySTAC item.
    It uses Astral to get the sunrise and sunset times for the item's centroid.

    Args:
        item (Item): PySTAC item

    Returns:
        bool: A boolean that is True if it's daytime
    """
    item_datetime = pd.to_datetime(item.properties.get("datetime"))
    if item_datetime is pd.NaT or item_datetime is None:
        return False
    centroid = box(*item.bbox).centroid
    city = LocationInfo("Unknown", "Unknown", "UTC", latitude=centroid.y, longitude=centroid.x)
    s = sun(city.observer, date=item_datetime)
    return s["sunrise"] <= item_datetime <= s["sunset"]


def rename_stac_items(
    item_collection: List[Item], nameplate: Dict[str, Dict[str, str]]
) -> List[Item]:
    """Rename STAC Assets.

    To make processing easier we map the asset names to a common naming convention
    defined in `nameplate`.

    Arguments:
        item_collection (List[Item]): A list of PySTAC items.
        nameplate (dict): A dictionary mapping collection_id to a dict of original_band:new_band.

    Returns:
        List of renamed PySTAC items.
    """
    for item in item_collection:
        if item.collection_id in nameplate:
            for original_band, new_band in nameplate[item.collection_id].items():
                if original_band in item.assets:
                    item.assets[new_band] = item.assets.pop(original_band)
    return item_collection


def dispatch_candidate_items(
    tile_observations: gpd.GeoDataFrame,
    tile_candidate_items: List[Item],
    item_id_field: str,
    candidate_items_field: str,
) -> pd.DataFrame | None:
    """Dispatches appropriate PySTAC items to each observation.

    A given observation will have a candidate item if it's geometry falls within the
    geometry of the granule.

    Args:
        tile_observations (pandas.DataFrame): DataFrame containing observations
            of the same tile.
        tile_candidate_items (List[Item]): List of candidate items for a given tile.
        item_id_field (str): Field name to store item IDs.
        candidate_items_field (str): Field name to store candidate items.

    Returns:
        A DataFrame with observations and their containing granules (items)
        when possible.
    """
    # Create GeoDataFrame of candidate items
    item_ids = [item.id for item in tile_candidate_items]
    items_gdf = gpd.GeoDataFrame.from_features(tile_candidate_items, crs=4326)
    items_gdf[item_id_field] = item_ids
    items_gdf = items_gdf[[item_id_field, "geometry"]]
    # Spatial join
    obs = tile_observations.set_geometry("geometry_4326")
    matches = gpd.sjoin(obs, items_gdf, predicate="within")
    if matches.empty:
        return None
    # Group matches
    obs[candidate_items_field] = (
        matches.groupby(matches.index)
        .agg({"index_right": lambda indices: [tile_candidate_items[i] for i in indices]})
        .reindex(obs.index, fill_value=[])
    )
    return obs


def find_closest_items(
    obsv: pd.Series,
    candidate_items_field: str,
    temporal_tolerance: int = 3,
    temporal_tolerance_minutes: int = 0,
    use_cloud_cover: bool = True,
) -> List[Item | None]:
    """Finds closest PySTAC items in time.

    Retrieves all PySTAC items within `temporal_tolerance` time window for a given observation
    and selects the best one based on the selection criteria.

    Selection behavior:
    - When use_cloud_cover=True (optical sensors like HLS, S2): Among all items within the
      temporal tolerance window, select the one with the least cloud coverage. This is
      appropriate for optical data where cloud coverage affects data quality.
    - When use_cloud_cover=False (SAR sensors like S1): Among all items within the temporal
      tolerance window, select the one that is temporally closest to the query date. This is
      appropriate for SAR data where cloud coverage is not applicable.

    Args:
        obsv (pandas.Series): Observation data containing the observation date
         and the candidate items from which to pick.
        candidate_items_field (str): Field name containing candidate items.
        temporal_tolerance (int): Number of days that can be tolerated for matching
            granules from the candidate items.
        temporal_tolerance_minutes (int): Number of minutes to add to the temporal
            tolerance.
        use_cloud_cover (bool): If True, select by minimum cloud coverage (optical sensors).
            If False, select by temporal proximity (SAR sensors). Defaults to True.

    Returns:
        List[Item | None]: A list containing items if found within the temporal
           tolerance or None values.
    """
    dates = obsv.tile_queries[1]
    items = obsv.get(candidate_items_field, [])
    if not items:
        return [None] * len(dates)
    closest_items: List[Item] = []
    for date in dates:
        query_date = pd.to_datetime(date, utc=True)
        candidates = [
            item
            for item in items
            if abs((item.datetime - query_date).total_seconds() / 60)
            <= (temporal_tolerance * 24 * 60 + temporal_tolerance_minutes)
        ]
        if not candidates:
            closest_items.append(None)
        else:
            if use_cloud_cover:
                # For optical data: select by minimum cloud coverage
                selected = min(
                    candidates,
                    key=lambda item: item.properties.get("eo:cloud_cover", 100),
                )
            else:
                # For SAR data: select by temporal proximity
                selected = min(
                    candidates,
                    key=lambda item: abs((item.datetime - query_date).total_seconds()),
                )
            closest_items.append(selected)
    return closest_items


def get_tile_info(
    data: gpd.GeoDataFrame,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 5,
    temporal_tolerance_minutes: int = 0,
) -> Tuple[pd.DataFrame, List[Tuple[str, List[str]]]]:
    """Get Tile Info.

    Retrieves a summary of all tiles required for a given dataset. The summary contains
    the desired start and end date for each tile. Also retrieves a list of queries
    that can be used to retrieve the tiles for each observation in `data`.

    Args:
        data (pd.DataFrame): A dataframe containing observation records.
        num_steps (int): Number of temporal time steps
        temporal_step (int): Size of each temporal step.
        temporal_tolerance (int): Number of days used as offset for the
        start and end dates to search for each tile.
        temporal_tolerance_minutes (int): Number of minutes to add to the temporal
            tolerance.

    Returns:
        A `tile_info` dataframe and a list of `tile_queries`
    """
    push_max_date_to_end_of_day = "time" not in data.columns
    df = data[["mgrs_tile_id", "input_features_date", "geometry_4326"]].reset_index(drop=True)
    tile_queries: List[Tuple[str, List[str]]] = []
    tile_info: List[Any] = []
    for _, (tile_id, date, geom) in df.iterrows():
        history: List[str] = []
        for i in range(num_steps):
            curr_date = pd.to_datetime(date) - pd.Timedelta(days=temporal_step * i)
            history.append(curr_date.strftime("%Y-%m-%dT%H:%M:%S"))
            tile_info.append([tile_id, curr_date, geom])
        tile_queries.append((tile_id, history))
    tile_info_df = (
        gpd.GeoDataFrame(
            tile_info,
            columns=["tile_id", "date", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
        .groupby("tile_id")
        .agg(
            {
                "geometry": lambda geoms: unary_union(geoms),
                "date": lambda dates: (dates.min(), dates.max()),
            }
        )
        .reset_index()
    )
    tile_info_df[["min_date", "max_date"]] = tile_info_df["date"].apply(pd.Series)
    tile_info_df[["lon_min", "lat_min", "lon_max", "lat_max"]] = tile_info_df["geometry"].apply(
        lambda geom: pd.Series(geom.bounds)
    )

    # Convert temporal tolerance to total days including minutes
    total_temporal_tol = temporal_tolerance + (temporal_tolerance_minutes / (24 * 60))
    tile_info_df["min_date"] -= pd.Timedelta(days=total_temporal_tol)
    tile_info_df["max_date"] += pd.Timedelta(days=total_temporal_tol)
    tile_info_df["min_date"] = tile_info_df["min_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    if push_max_date_to_end_of_day:
        tile_info_df["max_date"] = tile_info_df["max_date"].dt.strftime("%Y-%m-%dT23:59:59")
    else:
        tile_info_df["max_date"] = tile_info_df["max_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    tile_info_df = tile_info_df[
        ["tile_id", "min_date", "max_date", "lon_min", "lon_max", "lat_min", "lat_max"]
    ]
    return tile_info_df, tile_queries


def create_records_with_items(
    best_items: dict[str, pd.DataFrame],
    granules_field: str,
    items_field: str,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Creates the dataset from granules found.

    Args:
        best_items: A dictionary mapping each MGRS tile ID to
            a DataFrame containing the observations that fall within that tile with their
            associated PySTAC items representing granules.
        granules_field: Name of the field containing granule IDs (e.g. 'hls_granules' or
            's2_granules')
        items_field: Name of the field containing PySTAC items (e.g. 'hls_items' or 's2_items')

    Returns:
        A tuple containing a geopandas dataframe with `stac_items_str` column and a dictionary
        mapping `stac_items_str` to the PySTAC items representing the granules.
    """
    records_with_items = []
    dataset = {}
    for tile_id in best_items:
        obsvs = best_items[tile_id]
        obsvs[granules_field] = obsvs.apply(
            lambda obsv: [
                item.id if isinstance(item, Item) else None for item in obsv[items_field]
            ],
            axis=1,
        )
        obsvs = obsvs[
            obsvs.apply(partial(is_valid_dataset_entry, item_id_field=granules_field), axis=1)
        ]
        obsvs["stac_items_str"] = obsvs[granules_field].apply(lambda x: "_".join(x))
        for _, obsv in obsvs.drop_duplicates(subset=[granules_field]).iterrows():
            dataset[obsv["stac_items_str"]] = {
                "granules": [item.to_dict() for item in obsv[items_field]]
            }
        obsvs = obsvs.drop(["geometry_4326", items_field, "tile_queries", granules_field], axis=1)
        records_with_items.append(obsvs)
    filtered_records = pd.concat(records_with_items, ignore_index=True).set_geometry("geometry")
    return filtered_records, dataset


@ratelimit.limits(calls=DATA_PIPELINE_SETTINGS.METADATA_SEARCH_RATELIMIT, period=60)
@backoff.on_exception(
    backoff.expo,
    (APIError, RuntimeError),
    max_tries=5,
    max_time=300,  # 5 minutes max
    jitter=backoff.full_jitter,
)
def retrieve_stac_metadata(
    client: Client,
    tile_info_df: pd.DataFrame,
    collections: list[str],
    bands_nameplate: dict[str, dict[str, str]],
    cloud_coverage: int | None = 10,
    daytime_only: bool = False,
) -> dict[str, List[Item]]:
    """Retrieve STAC items Metadata.

    Given a tile_id, start_date and end_date, this function searches all the items
    available for this tile_id in this time window. A pystac_client Client is used
    to query a STAC API.

    Args:
        client (pystac_client.Client): pystac_client client to use to perform the search.
        tile_info_df (pd.DataFrame): A dataframe containing tile_id, start_date and
            end_date in each row.
        collections (list[str]): List of collection IDs to search in.
        cloud_coverage (int | None): Maximum percentage of cloud coverage to be
            tolerated for a granule.
        daytime_only (bool): Flag to determine whether to filter out night time granules.
        rate_limit_calls (int): Number of API calls allowed per period.
        rate_limit_period (int): Time period in seconds for rate limiting.

    Returns:
        A dictionary mapping tile_id to a list of available PySTAC items
        representing granules.
    """
    items_dict: Any = {}
    for _, (
        tile_id,
        start_date,
        end_date,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    ) in tile_info_df.iterrows():
        try:
            search = client.search(
                collections=collections,
                datetime=f"{start_date}/{end_date}",
                bbox=geo_utils.make_valid_bbox(lon_min, lat_min, lon_max, lat_max),
                sortby=[{"field": "datetime", "direction": "asc"}],
                query={} if cloud_coverage is None else {"eo:cloud_cover": {"lte": cloud_coverage}},
            )
            candidate_items = search.item_collection()
        except APIError as e:
            logging.warning(f"API Error for tile {tile_id}: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying
            continue

        if daytime_only:
            candidate_items = list(filter(is_daytime, candidate_items))
        if len(candidate_items) == 0:
            logging.warning(f"No items found for {tile_id}")
            continue
        candidate_items = rename_stac_items(candidate_items, bands_nameplate)
        items_dict[tile_id] = candidate_items
        time.sleep(1)  # Add a small delay between requests
    return items_dict


def find_best_items(
    data: pd.DataFrame,
    tiles_database: dict[str, List[Item]],
    item_id_field: str,
    candidate_items_field: str,
    items_field: str,
    temporal_tolerance: int = 12,
    temporal_tolerance_minutes: int = 0,
    use_cloud_cover: bool = True,
) -> dict[str, pd.DataFrame]:
    """Finds best PySTAC items for all observations when possible.

    For each observation, an attempt is made to retrieve the granules
    that actually contain the observation. The `dispatch_candidate_items` function is
    thus used to ensure that the best PySTAC items are not just
    the closest in terms of datetime.

    Args:
        data (pd.DataFrame): A dataframe containing observations that fall within a
            dense tile.
        tiles_database(dict[str, List[Item]]): A dictionary mapping a tile ID to a list
          of available PySTAC items.
        item_id_field (str): Field name to store item IDs.
        candidate_items_field (str): Field name to store candidate items.
        items_field (str): Field name to store the best items.
        temporal_tolerance (int): Tolerance (in days) for finding closest items.
        temporal_tolerance_minutes (int): Additional tolerance in minutes for finding
            closest items.
        use_cloud_cover (bool): If True, select items by minimum cloud coverage (optical sensors).
            If False, select items by temporal proximity (SAR sensors). Defaults to True.

    Returns:
        A dictionary mapping each MGRS tile ID to a DataFrame containing the observations
        that fall within that tile with their associated best PySTAC items representing
        the granules.
    """
    best_items = {}
    for tile_id in tiles_database:
        tile_obsvs = data[data["mgrs_tile_id"] == tile_id]
        # Before retrieving the temporally closest items, let's filter
        # the items by making sure only the ones with geometry in which
        # the observations fall are kept.
        tile_obsvs_with_items = dispatch_candidate_items(
            tile_obsvs,
            tiles_database[tile_id],
            item_id_field=item_id_field,
            candidate_items_field=candidate_items_field,
        )
        if tile_obsvs_with_items is None:
            continue
        tile_obsvs_with_items[items_field] = tile_obsvs_with_items.apply(
            lambda obsv: find_closest_items(
                obsv,
                candidate_items_field=candidate_items_field,
                temporal_tolerance=temporal_tolerance,
                temporal_tolerance_minutes=temporal_tolerance_minutes,
                use_cloud_cover=use_cloud_cover,
            ),
            axis=1,
        )
        best_items[tile_id] = tile_obsvs_with_items.drop(columns=[candidate_items_field])
    return best_items


def add_stac_items(
    client: Client,
    data: pd.DataFrame,
    config: Any,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 12,
    temporal_tolerance_minutes: int = 0,
    cloud_coverage: int | None = 10,
    daytime_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """Generic function to search and add STAC items for any data source.

    Data contains tile_id and a series of dates for which tiles are desired. This
    function finds the satellite granules closest to the desired dates with a
    tolerance of `temporal_tolerance`.

    Args:
        client: pystac_client Client to use for the search
        data: DataFrame containing observations that fall within tiles
        config: DataSourceConfig containing data-source-specific settings
        num_steps: Number of temporal steps into the past to fetch
        temporal_step: Step size (in days) for creating temporal steps
        temporal_tolerance: Tolerance (in days) for finding closest items
        temporal_tolerance_minutes: Additional tolerance in minutes for finding closest items
        cloud_coverage: Maximum percentage of cloud coverage (None for SAR data)
        daytime_only: Flag to filter out nighttime granules

    Returns:
        Dictionary mapping each MGRS tile ID to a DataFrame with associated STAC items
    """
    if "input_features_date" not in data.columns:
        data = data.rename(columns={"date": "input_features_date"})

    tiles_info, tile_queries = get_tile_info(
        data,
        num_steps=num_steps,
        temporal_step=temporal_step,
        temporal_tolerance=temporal_tolerance,
        temporal_tolerance_minutes=temporal_tolerance_minutes,
    )

    data["tile_queries"] = tile_queries
    tiles_database = retrieve_stac_metadata(
        client,
        tiles_info,
        collections=config.collections,
        bands_nameplate=config.bands_nameplate,
        cloud_coverage=cloud_coverage,
        daytime_only=daytime_only,
    )

    # Build field names using data source prefix
    prefix = config.field_prefix
    best_items = find_best_items(
        data,
        tiles_database,
        item_id_field=f"{prefix}_item_id",
        candidate_items_field=f"{prefix}_candidate_items",
        items_field=f"{prefix}_items",
        temporal_tolerance=temporal_tolerance,
        temporal_tolerance_minutes=temporal_tolerance_minutes,
        use_cloud_cover=config.supports_cloud_filtering,
    )
    return best_items


def open_stac_items(
    tile_dict: dict[str, Any],
    epsg: int,
    resolution: float,
    bands_asset: list[str],
    blocksize: tuple[int, int],
    mask_band: str,
    load_masks: bool = False,
    fill_value: int = 0,
    sign_func: Optional[Callable[[Item], Item]] = None,
) -> tuple[xr.DataArray, xr.DataArray | None, str]:
    """Opens multiple STAC Items as an xarray DataArray from given granules `tile_dict`.

    Args:
        tile_dict (dict[str, Any]): A dictionary containing granules IDs to retrieve
        for all timesteps of interest.
        epsg (int): CRS EPSG code.
        resolution (float): Spatial resolution in the specified CRS.
        bands_asset (list[str]): List of band names to load.
        blocksize (tuple[int, int]): Chunk size for x and y dimensions.
        mask_band (str): Name of the mask band (e.g. 'Fmask' or 'SCL').
        load_masks (bool): Whether or not to load the masks COGs.
        fill_value (int): Fill value for the data array.

    Returns:
        (xr.DataArray, xr.DataArray | None, str): A tuple of xarray DataArray combining
        data from all the COGs bands of interest, (optionally) the COGs masks and the
        CRS used.
    """
    # Load the bands for all timesteps and stack them in a data array
    assets_to_load = bands_asset + [mask_band] if (load_masks and mask_band) else bands_asset
    if sign_func is not None:
        plain_items = sign_func(ItemCollection(tile_dict))
    else:
        plain_items = [item.to_dict() for item in ItemCollection(tile_dict)]
    stacked_items = stackstac.stack(
        plain_items,
        assets=assets_to_load,
        chunksize=blocksize,
        properties=False,
        rescale=False,
        fill_value=fill_value,
        epsg=epsg,
        resolution=resolution,
    )

    bands = adjust_dims(stacked_items.sel(band=bands_asset))
    masks = adjust_dims(stacked_items.sel(band=[mask_band])) if (load_masks and mask_band) else None

    # Convert to float32 if we are dealing with Sentinel-1
    bands = bands.astype("float32") if bands_asset == S1_BANDS.ASSET else bands.astype("uint16")
    bands.attrs["scale_factor"] = 1

    return bands, masks, bands.crs
