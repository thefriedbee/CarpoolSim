"""
Define the standard schema of network links file
"""
from dataclasses import dataclass

from shapely import Point, LineString, Polygon


@dataclass
class TrafficNetworkLink:
    a: int  # link's starting node id
    b: int  # link's ending node id
    a_b: str  # links id (defined by node_a and node_b)
    name: str  # name of the road link
    distance: float  # travel distance along the link
    factype: str  # type of the road (e.g., highway, etc.)
    speed_limit: float  # speed limit of the road (i.e., travel speed)
    geometry: LineString  # geometry of the link (for visualization purpose)

    def convert_to_dict(self):
        return {
            "a": self.a,
            "b": self.b,
            "a_b": self.a_b,
            "name": self.name,
            "distance": self.distance,
            "factype": self.factype,
            "speed_limit": self.speed_limit,
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
    geometry: Point

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
    group_id: str  # group TAZ into contingent groups
    geometry: Polygon

    def convert_to_dict(self):
        return {
            "taz_id": self.taz_id,
            "group_id": self.group_id,
            "geometry": self.geometry,
        }

