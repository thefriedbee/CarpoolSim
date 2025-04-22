"""
Define the trip demands
"""
import datetime
from pydantic.dataclasses import dataclass
from pydantic import Field, ConfigDict

import pandas as pd
from shapely import Point


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TripDemand:
    trip_id: int = Field(description="The trip id")
    # spatial information
    orig_lon: float = Field(ge=-180, le=180, description="The longitude of the origin")
    orig_lat: float = Field(ge=-90, le=90, description="The latitude of the origin")

    dest_lon: float = Field(ge=-180, le=180, description="The longitude of the destination")
    dest_lat: float = Field(ge=-90, le=90, description="The latitude of the destination")
    # temporal information (the minute of the day)
    new_min: float = Field(ge=0, le=1440, description="The minute of the day")   
    geometry: Point = Field(description="The geometry of the trip demand")

    def convert_to_list(self) -> list:
        return [
            self.trip_id,
            self.orig_lon, self.orig_lat,
            self.dest_lon, self.dest_lat,
            self.new_min, self.geometry,
        ]

    def convert_to_pandas_series(self) -> pd.Series:
        lst = self.convert_to_list()
        idx = [
            "trip_id",
            "orig_lon", "orig_lat",
            "dest_lon", "dest_lat",
            "new_min", "geometry"
        ]
        ser = pd.Series(data=lst, index=idx)
        return ser

