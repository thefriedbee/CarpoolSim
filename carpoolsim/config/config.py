"""
Dataclasses to store all configurations/parameters of the experiment.
"""
from dataclasses import dataclass, Enum, auto


# available match modes
class MatchMode(Enum):
    # for both modes, represent using a list of match modes
    dc = auto()  # direct carpool
    pnr = auto()  # Park-and-Ride carpool


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


# ====== Below are dataclasses ======
@dataclass
class Config:
    # basic settings
    match_mode: list[MatchMode]
    solve_method: SolveMethod
    coordinate_filter: list[CoordinateFilter]
    time_filter: list[TimeFilter]
    reroute_filter: list[RerouteFilter]



@dataclass
class RunSimConfigWithTime:
    delta_t: float  # max number of minutes in the near future for passenger to send travel request
    epsilon_t: float  # the max #minutes in the near future for drivers to depart
    w: float  # the #minutes to update simulation clock (granularity)





