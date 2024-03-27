"""
Define the standard schema of Traffic Analysis Zone file
"""
from dataclasses import dataclass
from shapely import Polygon


@dataclass
class InputTAZ:
    taz_id: int
    geometry: Polygon
    centroid_lon: float
    centroid_lat: float

    def convert_to_dict(self):
        return {
            "taz_id": self.taz_id,
            "geometry": self.geometry,
            "centroid_lon": self.centroid_lon,
            "centroid_lat": self.centroid_lat,
        }

