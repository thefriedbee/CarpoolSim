"""
Classes for running the simulation.
"""
from carpoolsim.carpool.trip_demands import TripDemands
from carpoolsim.carpool.trip_cluster_abstract import TripClusterAbstract


class Simulation:
    def __init__(
            self, 
            trip_demands: TripDemands,
            # list of classes, not instances!
            trip_clusters: list[TripClusterAbstract],
    ):
        self.trip_demands = trip_demands
        self.trip_clusters = trip_clusters

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


