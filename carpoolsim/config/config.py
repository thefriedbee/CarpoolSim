"""
Dataclasses to store all configurations/parameters of the experiment.
"""
from dataclasses import dataclass
from enum import Enum, auto


class CPMode(Enum):
    SOV = "SOV"
    DC = "DC"
    PNR = "PNR"


class SolveMethod(Enum):
    # available solvers
    bipartite = auto()  # bipartite matching
    lp = auto()  # linear programming solver


class CoordinateFilter(Enum):
    # constraints the filter out unavailable carpool matches
    # available coordinate filters (in Manhattan distance)
    mu1 = auto()  # shared carpool distance / total distance
    mu2 = auto()  # travel distance of passenger after dropoff
    dist_max = auto()  # r: Euclidean distance between driver's origin and pickup location


class TimeFilter(Enum):
    # time constraint
    Delta1 = auto()  # control passenger/driver's departure time difference
    Delta2 = auto()  # control driver's maximum waiting time
    Gamma = auto()


class RerouteFilter(Enum):
    # reroute time constraints
    delta = auto()  # departure time difference
    gamma = auto()  # time window
    ita = auto()  # shared carpool time

