"""
Define the standard schema of network links file
"""
from pydantic.dataclasses import dataclass
from pydantic import Field, ConfigDict, field_validator
from shapely import Point, LineString, Polygon


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TrafficNetworkLink:
    a: int = Field(description="link's starting node id")
    b: int = Field(description="link's ending node id")
    a_b: str = Field(description="links id (defined by node_a and node_b)")
    name: str = Field(description="name of the road link")
    distance: float = Field(ge=0, description="The travel distance along the link")
    factype: int = Field(description="type of the road (e.g., highway, etc.)")
    speed_limit: float = Field(ge=0, le=300, description="The speed limit of the road (in mph)")
    geometry: LineString = Field(description="The geometry of the link")

    def convert_to_dict(self) -> dict:
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

    @field_validator("a_b")
    def convert_name_to_str(cls, v) -> str:
        return str(v)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TrafficNetworkNode:
    nid: int = Field(description="The traffic network node id")
    lon: float = Field(ge=-180, le=180, description="The longitude of the node")
    lat: float = Field(ge=-90, le=90, description="The latitude of the node")
    # projected coordinates (need to choose the correct projection)
    x: float = Field(description="The projected x-coordinate of the node")
    y: float = Field(description="The projected y-coordinate of the node")
    geometry: Point = Field(description="The geometry of the node")

    def convert_to_dict(self) -> dict:
        return {
            "nid": self.nid,
            "lon": self.lon,
            "lat": self.lat,
            "x": self.x,
            "y": self.y,
        }


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TrafficAnalysisZone:
    taz_id: int = Field(description="The traffic analysis zone id")
    group_id: str = Field(description="group TAZ into contingent groups")
    geometry: Polygon = Field(description="The geometry of the traffic analysis zone")

    def convert_to_dict(self) -> dict:
        return {
            "taz_id": self.taz_id,
            "group_id": self.group_id,
            "geometry": self.geometry,
        }
