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



