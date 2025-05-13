"""
Dataclasses to store all configurations/parameters of the experiment.
"""
from dataclasses import dataclass
from enum import Enum, auto


class CPMode(Enum):
    SOV = 1
    DC = 2
    PNR = 3


class SolveMethod(Enum):
    # available solvers
    bt = 1  # bipartite matching
    lp = 2  # linear programming solver


class Config:
    def __init__(self):
        self.solver = SolveMethod.bt
        self.mode = CPMode.PNR
        self.print_mat = False
        self.plot_all = False
        # Euclidean distance filter
        self.mu1 = 1.5
        self.mu2 = 0.1
        self.dist_max = 5*5280
        # time difference
        self.Delta1 = 15  # SOV departure time difference
        self.Delta2 = 10  # carpool waiting time
        self.Gamma = 0.2
        # reroute time constraints
        self.delta = 10
        self.gamma = 1.3
        self.ita = 0.9
        # self.ita_pnr = 0.5
        pass

    pass










