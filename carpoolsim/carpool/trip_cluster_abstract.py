"""
Use a parent class to avoid cyclic import
"""
from abc import ABC, abstractmethod
import numpy as np

from carpoolsim.carpool.trip_demands import TripDemands
from carpoolsim.config import CPMode


class TripClusterAbstract(ABC):
    def __init__(
        self,
        trip_demands: TripDemands,
    ):
        self.mode = None
        # only store the trips information,
        # most other functions (e.g, query paths) are "outsourced" to free functions
        self.td = trip_demands
        N = len(self.trips)
        # matrices to store travel time and distance considering DC mode
        # all combinations are carpoolable in the beginning
        self.cp_matrix = np.full((N, N), 1, dtype="int8")
        self.tt_matrix = np.full((N, N), np.nan, dtype="float32")
        self.ml_matrix = np.full((N, N), np.nan, dtype="float32")
        # A CENTRIC VIEW OF DRIVERS (3 trip segments of a carpool driver)
        # p1: pickup travel time for driver
        # p2: shared travel time for driver and passenger
        # p3: drop-off travel time for driver
        self.tt_matrix_p1 = np.full((N, N), np.nan, dtype="float32")
        self.tt_matrix_p2 = np.full((N, N), np.nan, dtype="float32")
        self.tt_matrix_p3 = np.full((N, N), np.nan, dtype="float32")
        # briefly store the results of carpooling
        self.num_paired: int = 0
        self.paired_lst: list[tuple[int, int]] = []

    @property
    def shape(self):
        # (#rows, #columns)
        return self.cp_matrix.shape
    
    @property
    def nrow(self):
        return self.cp_matrix.shape[0]
    
    @property
    def ncol(self):
        return self.cp_matrix.shape[1]
    
    @property
    def trips(self):
        return self.td.trips
    
    @property
    def network(self):
        return self.td.network

    def fill_diagonal(self, tt_lst, dst_lst):
        np.fill_diagonal(self.cp_matrix, 1)
        np.fill_diagonal(self.tt_matrix, tt_lst)
        np.fill_diagonal(self.ml_matrix, dst_lst)

    @abstractmethod
    def compute_carpool(self, int_idx1, int_idx2, fixed_role: bool = False, **kwargs):
        pass
    
    @abstractmethod
    def compute_in_one_step(self, **kwargs):
        pass

    def _print_matrix(self, step: int = 0, print_mat: bool = False):
        if not print_mat:
            return
        print(f"after step {step}")
        print("cp matrix:", self.cp_matrix.astype(int).sum())
        print(self.cp_matrix[:8, :8].astype(int))

    def compute_optimal_bipartite(self) -> tuple[int, list[tuple[int, int]]]:
        # import the solver here
        import carpoolsim.carpool_solver.bipartite_solver as tg
        """
        Solve the pairing problem using traditional bipartite method.
        This is to compare results with that of linear programming one
        :return:
        """
        bipartite_obj = tg.CarpoolBipartite(self.cp_matrix, self.tt_matrix)
        num_pair, pairs = bipartite_obj.solve_bipartite_conflicts_naive()
        return num_pair, pairs
