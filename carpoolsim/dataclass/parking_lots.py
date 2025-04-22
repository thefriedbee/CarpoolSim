"""
Define the "schema" of parking lots
"""
from pydantic.dataclasses import dataclass
from pydantic import Field, ConfigDict
from shapely import Point


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ParkAndRideStation:
    """Standard class to represent a parking lot"""
    station_id: int = Field(description="The parking lot id")
    name: str | None = Field(description="The name of the parking lot")
    lon: float = Field(ge=-180, le=180, description="The longitude of the parking lot")
    lat: float = Field(ge=-90, le=90, description="The latitude of the parking lot")
    capacity: None | int = Field(description="The capacity of the parking lot (None if unknown)")
    geometry: Point = Field(description="The geometry of the parking lot")

    def convert_to_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "name": self.name,
            "lon": self.lon,
            "lat": self.lat,
            "size": self.capacity,
        }
