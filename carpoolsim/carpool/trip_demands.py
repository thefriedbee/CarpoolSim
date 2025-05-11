"""
A basic class to store basic information of a set of SOV travelers.

This is a simple class just to store the basic information of travelers.
"""
import pandas as pd
import networkx as nx
import sqlalchemy

from carpoolsim.carpool.util.network_search import (
    naive_shortest_path_search
)


class TripDemands:
    def __init__(
        self,
        df: pd.DataFrame,
        network: nx.DiGraph,
        links: pd.DataFrame,
        engine: sqlalchemy.Engine,
        parking_lots: pd.DataFrame | None = None
    ) -> None:
        cols_to_drop = [
            'ox_sq', 'oy_sq', 'dx_sq', 'dy_sq',
            'depart_period', 'hh_id', 'newarr',
            'orig_inner_id', 'dest_inner_id'
        ]
        df = df.drop(cols_to_drop, axis=1, errors='ignore')
        df['pnr'] = None  # store accessible pnr station
        self.trips = df.copy()
        self.int2idx = {i: index for i, (index, row) in enumerate(df.iterrows())}
        self.idx2int = {index: i for i, (index, row) in enumerate(df.iterrows())}
        self.num_travelers = len(df)
        self.network = network
        self.links = links
        self.engine = engine
        self.parking_lots = parking_lots
        # record travel information
        self.soloPaths = []  # trip retention of SOV paths
        self.soloDists = []  # trip retention of SOV distance
        self.soloTimes = []  # trip retention of SOV time

    def compute_sov_info(self) -> None:
        trips = self.trips
        soloPaths, soloTimes, soloDists = [], [], []
        # search traveling paths for each trip
        for idx, trip in trips.iterrows():
            start_node, end_node = trip['o_node'], trip['d_node']
            # start_taz, end_taz = trip['orig_taz'], trip['dest_taz']
            pth_nodes, tt, dst = naive_shortest_path_search(
                self.network, start_node, end_node
            )
            soloPaths += [pth_nodes]
            soloTimes += [round(tt, 2)]
            soloDists += [round(dst, 2)]
        # save results to object
        self.soloPaths = soloPaths
        self.soloTimes = soloTimes
        self.soloDists = soloDists
