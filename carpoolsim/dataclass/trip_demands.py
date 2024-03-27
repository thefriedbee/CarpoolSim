"""
Define the trip demands
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class TripDemand:
    # spatial information
    orig_lon: float
    orig_lat: float
    orig_taz: int  # TAZ id
    orig_taz_group: int   # TAZ group id (for multiprocessing)
    dest_lon: float
    dest_lat: float
    dest_taz: int  # TAZ id
    dest_taz_group: int  # TAZ group id (for multiprocessing)
    # temporal information
    new_min: float  # minute of the day for departure

    def convert_to_list(self):
        return [
            self.orig_lon, self.orig_lat, self.orig_taz, self.orig_taz_group,
            self.dest_lon, self.dest_lat, self.dest_taz, self.dest_taz_group,
            self.new_min
        ]

    def convert_to_pandas_series(self):
        lst = self.convert_to_list()
        idx = [
            "orig_lon", "orig_lat", "orig_taz", "orig_taz_group",
            "dest_lon", "dest_lat", "dest_taz", "dest_taz_group",
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

