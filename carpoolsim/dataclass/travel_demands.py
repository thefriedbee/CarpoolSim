"""
Define the trip demands
"""
import datetime
from dataclasses import dataclass

import pandas as pd
from shapely import Point


@dataclass
class TripDemand:
    trip_id: int
    # spatial information
    orig_lon: float
    orig_lat: float

    dest_lon: float
    dest_lat: float
    # temporal information
    # I don't care the date, but time of the day is important
    new_min: float
    geometry: Point

    def convert_to_list(self):
        return [
            self.trip_id,
            self.orig_lon, self.orig_lat,
            self.dest_lon, self.dest_lat,
            self.new_min, self.geometry,
        ]

    def convert_to_pandas_series(self):
        lst = self.convert_to_list()
        idx = [
            "trip_id",
            "orig_lon", "orig_lat",
            "dest_lon", "dest_lat",
            "new_min", "geometry"
        ]
        ser = pd.Series(data=lst, index=idx)
        return ser

