"""
Classes for running the simulation.
"""
import pandas as pd

from carpoolsim.carpool.trip_demands import TripDemands, Infrastructure
from carpoolsim.carpool.trip_cluster_abstract import TripClusterAbstract


class Simulation:
    def __init__(
        self, 
        infrastructure: Infrastructure,
        trips: pd.DataFrame,
        # list of classes, not instances!
        carpool_types: list[TripClusterAbstract],
    ):
        self.infrastructure = infrastructure
        self.trips = trips
        self.carpool_types = carpool_types
        # unit: minute of the simulation day
        self.t0 = 0
        self.delta_t = 1

    def init_clusters(self, trip_demands: TripDemands):
        tcs = []
        for trip_cluster in self.trip_clusters:
            tcs.append(trip_cluster(trip_demands))
        return tcs

    def run(self):
        for tc in self.trip_clusters:
            tc.compute_carpool_in_one_step()

    def evaluate(self):
        pass


