"""
functions to expand dataclass to pandas dataframe (geospatial processing)
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# attach coordinate to the nearest taz centroid node
def attach_coord2taz_node(df_row, df_nodes, mode="orig"):
    filt = (df_nodes.nid == str(df_row[f"{mode}_taz"]))
    X = df_nodes.loc[filt, "x"].iloc[0]
    Y = df_nodes.loc[filt, "y"].iloc[0]
    node = df_nodes.loc[filt, "nid"].iloc[0]
    if mode == "orig":
        return pd.Series({"ox": X, "oy": Y, "o_node": str(node)})
    elif mode == "dest":
        return pd.Series({"dx": X, "dy": Y, "d_node": str(node)})
    else:
        raise IOError


def attach_pnr2taz_node(df_row, df_nodes):
    filt = (df_nodes.nid == str(df_row["taz"]))
    X = df_nodes.loc[filt, "x"].iloc[0]
    Y = df_nodes.loc[filt, "y"].iloc[0]
    node = df_nodes.loc[filt, "nid"].iloc[0]
    return pd.Series({"x": X, "y": Y, "node": str(node)})


# alias for attach_coord2taz_node
get_xy = attach_coord2taz_node
get_xy_pnr = attach_pnr2taz_node


# for each parking lot, search the corresponding TAZ located in
def get_contained_taz(x: Point, tazs: gpd.GeoDataFrame):
    filt = tazs.contains(x)
    if sum(filt) > 0:
        taz = tazs.loc[filt, 'taz_id'].iloc[0]
    else:
        taz = -1
    return taz

# alias for get_contained_taz
get_taz = get_contained_taz


def preprocess_trips(
    trips: pd.DataFrame,
    df_nodes: pd.DataFrame,
) -> pd.DataFrame:
    trips[["ox", "oy", "o_node"]] = trips.apply(
        get_xy,
        axis=1, df_nodes=df_nodes, mode="orig"
    )
    trips[["dx", "dy", "d_node"]] = trips.apply(
        get_xy,
        axis=1, df_nodes=df_nodes, mode="dest"
    )
    return trips


def preprocess_df_links(
    df_links: pd.DataFrame,
    grid_size: float = 25000.0,
) -> pd.DataFrame:
    # get the grid index of each link
    def get_bbox(df_row):
        xmin = min(df_row["ax"], df_row["bx"]) / grid_size
        xmax = max(df_row["ax"], df_row["bx"]) / grid_size
        ymin = min(df_row["ay"], df_row["by"]) / grid_size
        ymax = max(df_row["ay"], df_row["by"]) / grid_size
        xmin_sq = round(xmin, 0)
        xmax_sq = round(xmax, 0)
        ymin_sq = round(ymin, 0)
        ymax_sq = round(ymax, 0)
        return pd.Series(
            {"minx_sq": xmin_sq, "maxx_sq": xmax_sq, 
             "miny_sq": ymin_sq, "maxy_sq": ymax_sq}
        )
    
    df_links[["minx_sq", "maxx_sq", "miny_sq", "maxy_sq"]] = df_links.apply(get_bbox, axis=1)
    return df_links

