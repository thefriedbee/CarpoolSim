"""
Use a parent class to avoid cyclic import
"""
from abc import ABC, abstractmethod
import numpy as np


class TripClusterAbstract(ABC):
    def __init__(self, trips):
        # only store the trips information,
        # most other functions (e.g, query paths) are "outsourced" to free functions
        self.trips = trips
        N = len(self.trips)
        # matrices to store travel time and distance considering DC mode
        self.cp_matrix = np.full((N, N), np.nan, dtype="int8")
        self.tt_matrix = np.full((N, N), np.nan, dtype="float32")
        self.ml_matrix = np.full((N, N), np.nan, dtype="float32")
        # A CENTRIC VIEW OF DRIVERS (3 trip segments of a carpool driver)
        # p1: pickup travel time for driver
        # p2: shared travel time for driver and passenger
        # p3: drop-off travel time for driver
        self.tt_matrix_p1 = np.full((N, N), np.nan, dtype="float32")
        self.tt_matrix_p2 = np.full((N, N), np.nan, dtype="float32")
        self.tt_matrix_p3 = np.full((N, N), np.nan, dtype="float32")

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
