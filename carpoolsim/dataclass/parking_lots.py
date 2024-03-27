"""
Define the "schema" of parking lots
"""
from dataclasses import dataclass

@dataclass
class ParkingLot:
    """Standard class to represent a parking lot"""
    object_id: int
    name: str
    lon: float
    lat: float
    capacity: None | int

    def convert_to_dict(self):
        return {
            "object_id": self.object_id,
            "name": self.name,
            "lon": self.lon,
            "lat": self.lat,
            "size": self.capacity,
        }
