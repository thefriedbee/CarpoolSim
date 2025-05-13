"""
Dataclasses to store all configurations/parameters of the experiment.
"""
from dataclasses import dataclass
from enum import Enum, auto


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
    def __init__(self):
        # basic settings
        self.solver = SolveMethod.bt
        self.mode = CPMode.DC
        self.print_mat = False
        self.plot_all = False
        self.run_solver = True
        # Euclidean distance filter
        self.mu1 = 1.5  # carpool distance / total distance (driver)
        self.mu2 = 0.1  # shared distance / total distance (driver)
        self.dist_max = 5*5280  # pickup distance (driver)
        # time difference
        self.Delta1 = 15  # SOV departure time difference
        self.Delta2 = 10  # carpool waiting time
        self.Gamma = 0.2  # waiting time / passenger travel time
        # reroute time constraints
        self.delta = 10  # reroute time in minutes
        self.gamma = 1.3  # carpool time / SOV travel time (for the driver)
        self.ita = 0.9  # shared travel time / passenger travel time
        # self.ita_pnr = 0.5  # shared travel time / passenger travel time (for PNR)

    def set_config(self, config: dict):
        for key, value in config.items():
            if key == "modes":
                self.modes = [CPMode_MAP[mode] for mode in value]
            else:
                setattr(self, key, value)


# class ExperimentConfig:
#     def __init__(self):
#         self.trip_cluster_config = TripClusterConfig()






