"""
Dataclasses to store all configurations/parameters of the experiment.
"""
from dataclasses import dataclass
from enum import Enum, auto

import pandas as pd
import geopandas as gpd
import networkx as nx

import carpoolsim.dataclass.utils as ut
from carpoolsim.network_prepare import (
    pnr_add_projection, 
    pnr_filter_within_TAZs,
    build_carpool_network
)


class CPMode(Enum):
    SOV = 1
    DC = 2
    PNR = 3

CPMode_MAP = {
    "SOV": CPMode.SOV,
    "sov": CPMode.SOV,
    "DC": CPMode.DC,
    "dc": CPMode.DC,
    "PNR": CPMode.PNR,
    "pnr": CPMode.PNR,
}


class SolveMethod(Enum):
    # available solvers
    bt = 1  # bipartite matching
    # lp = 2  # linear programming solver


# Configuration for EACH CARPOOL MODE
class TripClusterConfig:
    def __init__(
        self,
        mode: CPMode.DC,
        print_mat: bool = False,
        plot_all: bool = False,
        run_solver: bool = True,
        mu1: float = 1.5,
        mu2: float = 0.1,
        dist_max: float = 5*5280,
        Delta1: float = 15,
        Delta2: float = 10,
        Gamma: float = 0.2,
        delta: float = 10,
        gamma: float = 1.3,
        ita: float = 0.9,
    ):
        # basic settings
        self.solver = SolveMethod.bt
        self.mode = mode
        self.print_mat = print_mat
        self.plot_all = plot_all
        self.run_solver = run_solver
        # Euclidean distance filter
        self.mu1 = mu1  # carpool distance / total distance (driver)
        self.mu2 = mu2  # shared distance / total distance (driver)
        self.dist_max = dist_max  # pickup distance (driver)
        # time difference
        self.Delta1 = Delta1  # SOV departure time difference
        self.Delta2 = Delta2  # carpool waiting time
        self.Gamma = Gamma  # waiting time / passenger travel time
        # reroute time constraints
        self.delta = delta  # reroute time in minutes
        self.gamma = gamma  # carpool time / SOV travel time (for the driver)
        self.ita = ita  # shared travel time / passenger travel time
        # self.ita_pnr = 0.5  # shared travel time / passenger travel time (for PNR)

    def set_config(self, config: dict):
        for key, value in config.items():
            if key == "modes":
                self.modes = [CPMode_MAP[mode] for mode in value]
            else:
                setattr(self, key, value)


class NetworkConfig:
    def __init__(
        self, 
        links: gpd.GeoDataFrame, 
        nodes: gpd.GeoDataFrame,
        tazs: gpd.GeoDataFrame,
        walk_speed: float = 30,
        grid_size: int = 25000,
        ntp_dist_thresh: int = 5280,
    ):
        # basic networks
        self.links: gpd.GeoDataFrame = self.preprocess_network(links, grid_size)
        self.nodes: gpd.GeoDataFrame = nodes
        self.tazs: gpd.GeoDataFrame = tazs
        self.network: nx.DiGraph = self.build_graph()  # networkx directed graph
        # in this application, it is actually the driving speed
        # to the nearest node in the network
        self.walk_speed: float = walk_speed  # mph
        # for searching nearby links by grouping links to grids with width 25000 ft. for efficiency in searching
        self.grid_size: int = grid_size  # in feet
        # maximum distance to the nearest node in the network
        self.ntp_dist_thresh: int = ntp_dist_thresh  # in feet

    def preprocess_network(self, links: gpd.GeoDataFrame, grid_size: int) -> gpd.GeoDataFrame:
        # preprocess traffic links
        links = ut.preprocess_df_links(
            links,
            grid_size=grid_size
        )
        return links

    def build_graph(self) -> nx.DiGraph:
        return build_carpool_network(self.links)

    def preprocess_trips(self, trips: pd.DataFrame) -> pd.DataFrame:
        trips = ut.preprocess_trips(trips, self.nodes)
        return trips

    def preprocess_pnr_lots(self, pnr_lots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        pnr_lots = pnr_filter_within_TAZs(pnr_lots, self.tazs)
        pnr_lots = pnr_add_projection(pnr_lots, self)
        return pnr_lots
