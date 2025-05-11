import os

import geopandas as gpd
import pandas as pd

# load environment variables
import carpoolsim.basic_settings as bs
import carpoolsim.dataclass.utils as ut


# load dataframes for testing
test_input_dir = os.path.join(bs.os.environ['data_inputs'], "cleaned")
print(f"load example shapefiles/dataframes from {test_input_dir}")

print("load trips file")
trips = gpd.read_file(
    os.path.join(test_input_dir, "trips.shp")
)

print("load tazs file (polygon file)")
tazs = gpd.read_file(
    os.path.join(test_input_dir, "tazs.shp")
)

print("load nodes file")
df_nodes = gpd.read_file(
    os.path.join(test_input_dir, "nodes.shp")
)

print("load links file")
df_links = gpd.read_file(
    os.path.join(test_input_dir, "links.shp")
)

print("load pnr lots file")
pnr_lots = gpd.read_file(
    os.path.join(test_input_dir, "pnrs.shp")
)

print("Finished loading example shapefiles/dataframes")
print("preprocess dataframes")


# match origin/destination to nearest nodes represented in coordinates!
trips[["ox", "oy", "o_node"]] = trips.apply(
    ut.get_xy,
    axis=1, df_nodes=df_nodes, mode="orig"
)
trips[["dx", "dy", "d_node"]] = trips.apply(
    ut.get_xy,
    axis=1, df_nodes=df_nodes, mode="dest"
)

pnr_lots['taz'] = pnr_lots['geometry'].apply(
    ut.get_taz, tazs=tazs
)
pnr_lots[['x', 'y', 'node']] = pnr_lots.apply(
    ut.get_xy_pnr,
    axis=1, df_nodes=df_nodes
)

print("Finished preprocess dataframes")
