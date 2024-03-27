"""
Define the standard schema of network links file
"""
from dataclasses import dataclass

from shapely import LineString


@dataclass
class InputTrafficNetworkLink:
    node_a: int  # link's starting node id
    node_b: int  # link's ending node id
    a_b: str  # links id (defined by node_a and node_b)
    geometry: LineString  # geometry of the link (for visualization purpose)
    distance: float  # travel distance along the link


@dataclass
class TrafficNetworkNode:
    node_id: int
    node_lon: float
    node_lat: float
