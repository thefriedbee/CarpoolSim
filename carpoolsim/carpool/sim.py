"""
Classes for running the simulation.
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np

from carpoolsim.carpool.trip_demands import TripDemands, Infrastructure
from carpoolsim.carpool.trip_cluster_abstract import TripClusterAbstract
from carpoolsim.carpool.util.evaluator import (
    evaluate_individual_trips_sim,
    summarize_results,
)


@dataclass
class Clock:
    # all units are in minutes
    t0: int = 0
    t1: int = 24 * 60  # 1440 minutes a day
    t: int = 0  # current time in the simulation
    w: int = 10  # the observation window
    delta_t: int = 1  # update step

    def update(self):
        self.t += self.delta_t
    
    def select_trips(self, trips: pd.DataFrame, time_col: str = "new_min"):
        # select trips to be assigned
        filt = (trips[time_col] >= self.t) & (trips[time_col] < self.t + self.w)
        return trips[filt]


class SimulationTask:
    def __init__(
        self, 
        infrastructure: Infrastructure,
        trips: pd.DataFrame,
        carpool_modes: list[TripClusterAbstract],  # list of classes, not instances!
    ):
        self.infrastructure = infrastructure
        self.trip_demands = TripDemands(trips, infrastructure)
        self.trip_demands.compute_sov_info()  # compute the SOV trips at once
        self.trip_clusters = [tc(self.trip_demands) for tc in carpool_modes]
        # combined cp results considering all modes
        self.mc_matrix = np.full((self.trip_demands.nrow, self.trip_demands.ncol), -1)
        self.cp_matrix = np.zeros((self.trip_demands.nrow, self.trip_demands.ncol))
        self.tt_matrix = np.zeros((self.trip_demands.nrow, self.trip_demands.ncol))
        self.ml_matrix = np.zeros((self.trip_demands.nrow, self.trip_demands.ncol))
        self.cp_pnr = np.full((self.trip_demands.nrow, self.trip_demands.ncol), -1)
        self.pnr_access_info = None
        self.fill_sov_info()
        # results
        self.num_pair: int = 0
        self.paired_lst: list[tuple[int, int]] = []
        # summary results
        self.df_trips: pd.DataFrame = pd.DataFrame()
        self.df_summ: pd.DataFrame = pd.DataFrame()
    
    def fill_sov_info(self):
        np.fill_diagonal(self.mc_matrix, 0)  # 0 for SOV mode
        np.fill_diagonal(self.cp_matrix, 1)
        np.fill_diagonal(self.tt_matrix, self.trip_demands.soloTimes)
        np.fill_diagonal(self.ml_matrix, self.trip_demands.soloDists)

    def run_in_one_step(self):
        for tc in self.trip_clusters:
            num_pair, _ = tc.compute_in_one_step(print_mat=False)
            print(f"Mode {tc.__class__.__name__}: {num_pair} pairs")
            print(tc.paired_lst)
            print()
            # store important information from cluster
            if tc.__class__.__name__ == "TripClusterPNR":
                self.cp_pnr = tc.cp_pnr
                self.pnr_access_info = tc.pnr_access_info

    def gather_results(self):
        mc_matrix = self.mc_matrix
        cp_matrix = self.cp_matrix
        tt_matrix = self.tt_matrix
        ml_matrix = self.ml_matrix
        # enumerate each solution and fill results in order
        for mode_idx, tc in enumerate(self.trip_clusters):
            mc_matrix = np.where(
                (mc_matrix == -1) & (tc.cp_matrix > 0), 
                mode_idx, mc_matrix
            )
            cp_matrix = np.where(
                (cp_matrix == 0) & (tc.cp_matrix > 0), 
                tc.cp_matrix, cp_matrix
            )
            tt_matrix = np.where(
                (tt_matrix == 0) & (tc.tt_matrix > 0), 
                tc.tt_matrix, tt_matrix
            )
            ml_matrix = np.where(
                (ml_matrix == 0) & (tc.ml_matrix > 0), 
                tc.ml_matrix, ml_matrix
            )
        self.mc_matrix = mc_matrix
        self.cp_matrix = cp_matrix
        self.tt_matrix = tt_matrix
        self.ml_matrix = ml_matrix

    def optimize_results(self):
        # import the solver here
        import carpoolsim.carpool_solver.bipartite_solver as tg
        bipartite_obj = tg.CarpoolBipartite(self.cp_matrix, self.tt_matrix)
        num_pair, paired_lst = bipartite_obj.solve_bipartite_conflicts_naive()
        self.num_pair = num_pair
        self.paired_lst = paired_lst

    def evaluate_assigned_results(self):
        df_trips = evaluate_individual_trips_sim(self)
        ret = summarize_results(df_trips, self.paired_lst)
        self.df_trips = df_trips
        self.df_summ = pd.DataFrame([ret], index=[0])

    def export_unassigned_trips(self):
        # leave the SOV trips in case needed for assignment in the future
        pairs = self.paired_lst
        assigned_idx = [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
        assigned_idx = list(set(assigned_idx))
        free_idx = [ix for ix in range(self.trip_demands.nrow) if ix not in assigned_idx]
        return self.trip_demands.trips.iloc[free_idx]


class SimulatorWithTime:
    def __init__(
        self, 
        infrastructure: Infrastructure,
        trips: pd.DataFrame,
        carpool_modes: list[TripClusterAbstract],
        clock: Clock
    ):
        self.infrastructure = infrastructure
        self.trips = trips
        self.carpool_modes = carpool_modes
        self.clock = clock
        # summary results
        self.df_trips: pd.DataFrame | list[pd.DataFrame] = []
        self.df_summ: pd.DataFrame | list[pd.DataFrame] = []

    def run_one_step(self):
        # select trips to be assigned
        trips1 = self.clock.select_trips(self.trips)
        # generate trip demands
        trip_demands = TripDemands(trips1, self.infrastructure)
        # create a simulation task
        sim_task = SimulationTask(
            self.infrastructure,
            trip_demands,
            self.carpool_modes
        )
        # run the simulation
        sim_task.run_in_one_step()
        sim_task.gather_results()
        sim_task.optimize_results()
        sim_task.evaluate_assigned_results()
        self.df_trips.append(sim_task.df_trips)
        self.df_summ.append(sim_task.df_summ)

    def run_all(self):
        while self.clock.t < self.clock.t1:
            self.run_one_step()
            self.clock.update()

