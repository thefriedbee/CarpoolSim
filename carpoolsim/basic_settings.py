"""
It contains the most basic project settings at one place to change...
"""
import os
import sys
from pathlib import Path
import sqlalchemy
# basic path information
os.environ['project_root'] = Path(__file__).parent.parent.as_posix()
sys.path.append(os.environ['project_root'])


# data inputs and outputs folder
os.environ['data_inputs'] = os.path.join(
    os.environ['project_root'], "data_inputs"
)

os.environ['data_outputs'] = os.path.join(
    os.environ['project_root'], "data_outputs"
)


# more specific paths to required files
os.environ['network_nodes'] = os.path.join(
    os.environ['data_inputs'],
    'ABM2020 203K', '2020 nodes with latlon', '2020_nodes_latlon.shp'
)

os.environ['network_links'] = os.path.join(
    os.environ['data_inputs'],
    'ABM2020 203K', '2020 links', '2020_links.shp'
)

os.environ['taz'] = os.path.join(
    os.environ['data_inputs'],
    'ABM2020 203K', 'taz', 'taz.shp'
)

os.environ['parking_lots'] = os.path.join(
    os.environ['data_inputs'], 'ABM2020 203K', '2020 PNR nodes',
    '2020 pnr nodes.shp'
)

os.environ['trip_demands'] = os.path.join(
    os.environ['data_inputs'], 'gt_survey',
    '2022_campus_commute_survey_data_no_zip_na_05122023.csv'
)

# geographical information (the projection crs)
# CRS = "EPSG:3857"  # This is Mercator system that can work for most places globally
CRS = "EPSG:2240"  # this is for Georgia West
NUM_PROCESSES = 12  # number of multiprocess for parallel computing

DB_URL = f"sqlite:///{os.path.join(os.environ['data_inputs'], 'path_retention1.db')}"
engine = sqlalchemy.create_engine(DB_URL)
