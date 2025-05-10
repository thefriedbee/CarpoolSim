import os

import geopandas as gpd
import pandas as pd

# load environment variables
import carpoolsim.basic_settings as bs


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

