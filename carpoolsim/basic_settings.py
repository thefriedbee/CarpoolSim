"""
It contains the most basic project settings at one place to change...
"""
import os
import sys
from pathlib import Path

# basic path information
os.environ['project_root'] = Path(__file__).parent.parent.as_posix()
sys.path.append(os.environ['project_root'])

os.environ['project_data'] = os.path.join(
    os.environ['project_root'], "data_inputs"
)

os.environ['network_links'] = os.path.join(
    os.environ['project_root'],
    'ABM2020 203K', '2020 nodes with latlon', '2020_nodes_latlon.shp'
)

os.environ['network_nodes'] = os.path.join(
    os.environ['project_root'],
    'ABM2020 203K', '2020 links', '2020_links.shp'
)

os.environ['taz'] = os.path.join(
    os.environ['project_root'],
    'ABM2020 203K', 'taz', 'TAZ_combine.shp'
)

os.environ['parking_lots'] = os.path.join(
    os.environ['project_data'], 'Park_and_Ride_locations',
    'Park_and_Ride_locations.shp'
)

# CRS = "EPSG:3857"  # This is Mercator system that can work for most places globally
CRS = "EPSG:2240"  # this is for Georgia West

