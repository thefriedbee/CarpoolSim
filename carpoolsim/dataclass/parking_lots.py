"""
Define the "schema" of parking lots
"""
from dataclasses import dataclass

from shapely import Geometry


@dataclass
class ParkAndRideStation:
    """Standard class to represent a parking lot"""
    station_id: int
    name: str
    lon: float
    lat: float
    capacity: None | int
    geometry: Geometry

    def convert_to_dict(self):
        return {
            "station_id": self.station_id,
            "name": self.name,
            "lon": self.lon,
            "lat": self.lat,
            "size": self.capacity,
        }
