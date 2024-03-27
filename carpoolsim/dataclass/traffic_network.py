"""
Define the standard schema of network links file
"""
from dataclasses import dataclass

from shapely import LineString, Polygon


@dataclass
class TrafficNetworkLink:
    a: int  # link's starting node id
    b: int  # link's ending node id
    a_b: str  # links id (defined by node_a and node_b)
    distance: float  # travel distance along the link
    geometry: LineString  # geometry of the link (for visualization purpose)

    def convert_to_dict(self):
        return {
            "a": self.a,
            "b": self.b,
            "a_b": self.a_b,
            "distance": self.distance,
            "geometry": self.geometry
        }


@dataclass
class TrafficNetworkNode:
    nid: int
    lon: float
    lat: float
    # projected coordinates (need to choose the correct projection)
    x: float
    y: float

    def convert_to_dict(self):
        return {
            "nid": self.nid,
            "lon": self.lon,
            "lat": self.lat,
            "x": self.x,
            "y": self.y,
        }


@dataclass
class TrafficAnalysisZone:
    taz_id: int
    geometry: Polygon

    def convert_to_dict(self):
        return {
            "taz_id": self.taz_id,
            "geometry": self.geometry
        }

