"""
Define the standard schema of network links file
"""
from dataclasses import dataclass

from shapely import LineString

@dataclass
class TrafficNetworkLink:
    node_a: int  # link's starting node id
    node_b: int  # link's ending node id
    a_b: str  # links id (defined by node_a and node_b)
    geometry: LineString  # geometry of the link (for visualization purpose)
    distance: float  # length of the link
    link_length: float

    # detailed information
    pass


