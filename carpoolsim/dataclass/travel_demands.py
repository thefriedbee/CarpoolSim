"""
Define the trip demands
"""
import datetime
from dataclasses import dataclass

import pandas as pd


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

    def convert_to_list(self):
        return [
            self.trip_id,
            self.orig_lon, self.orig_lat,
            self.dest_lon, self.dest_lat,
            self.new_min
        ]

    def convert_to_pandas_series(self):
        lst = self.convert_to_list()
        idx = [
            "trip_id",
            "orig_lon", "orig_lat",
            "dest_lon", "dest_lat",
            "new_min"
        ]
        ser = pd.Series(data=lst, index=idx)
        return ser


class PrepareTripDemands:
    def __init__(
            self,
            df: pd.DataFrame,
            orig_lon_cn: str,
            orig_lat_cn: str,
            dest_lon_cn: str,
            dest_lat_cn: str,
    ):
        self.trip_demands: list[TripDemand] = []
        # iterate the data frame
        for index, row in df.iterrows():
            col_lst = [orig_lon_cn, orig_lat_cn, dest_lon_cn, dest_lat_cn]
            row_input = row[col_lst].iloc[0].tolist()
            self.trip_demands.append(TripDemand(*row_input))

    def convert_trip_demands_to_pd(self):
        df = []
        for trip_demand in self.trip_demands:
            df.append(trip_demand.convert_to_pandas_series())
        df = pd.DataFrame(df)
        return df

