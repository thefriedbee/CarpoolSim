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
from carpoolsim.config import (
    CPMode,
    TripClusterConfig,
)


@dataclass
class Clock:
    # all units are in minutes
    t_start: int = 0
    t_end: int = 24 * 60  # 1440 minutes a day
    t: int = 0  # current time in the simulation
    w: int = 10  # the observation window
    delta_t: int = 5  # update step

    def update(self):
        self.t += self.delta_t
    
    def select_trips(self, trips: pd.DataFrame, time_col: str = "new_min"):
        # select trips to be assigned
        filt = (trips[time_col] >= self.t) & (trips[time_col] < self.t + self.w)
        return trips[filt].copy()


class SimulationTask:
    def __init__(
        self, 
        infrastructure: Infrastructure,
        trips: pd.DataFrame,
        carpool_modes: list[TripClusterAbstract],  # list of classes, not instances!
        mode_configs: list[TripClusterConfig],
    ):
        self.infrastructure = infrastructure
        self.trip_demands = TripDemands(trips, infrastructure)
        self.trip_demands.compute_sov_info()  # compute the SOV trips at once
        self.trip_clusters = [tc(self.trip_demands) for tc in carpool_modes]
        self.mode_configs = mode_configs
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
        for i, tc in enumerate(self.trip_clusters):
            num_pair, _ = tc.compute_in_one_step(
                config = self.mode_configs[i]
            )
            print(f"Mode {tc.mode}: {num_pair} pairs")
            # print(tc.paired_lst)
            # print()
            if tc.mode == CPMode.PNR:
                self.cp_pnr = tc.cp_pnr
                self.pnr_access_info = tc.pnr_access_info

    def gather_results(self):
        mc_matrix = self.mc_matrix
        cp_matrix = self.cp_matrix
        tt_matrix = self.tt_matrix
        ml_matrix = self.ml_matrix
        # enumerate each solution and fill results in order
        for tc in self.trip_clusters:
            tc_mode = tc.mode
            mc_matrix = np.where(
                (mc_matrix == -1) & (tc.cp_matrix > 0), 
                tc_mode.value, mc_matrix
            )
        
        for tc in self.trip_clusters:
            mode = tc.mode.value
            cp_matrix = np.where((mc_matrix == mode), tc.cp_matrix, cp_matrix)
            tt_matrix = np.where((mc_matrix == mode), tc.tt_matrix, tt_matrix)
            ml_matrix = np.where((mc_matrix == mode), tc.ml_matrix, ml_matrix)
        self.mc_matrix = mc_matrix
        self.cp_matrix = cp_matrix
        self.tt_matrix = tt_matrix
        self.ml_matrix = ml_matrix

    def optimize_results(self):
        import carpoolsim.carpool_solver.bipartite_solver as tg
        bipartite_obj = tg.CarpoolBipartite(self.cp_matrix, self.tt_matrix)
        num_pair, paired_lst = bipartite_obj.solve_bipartite_conflicts_naive(
            drop_sov_pairs=False
        )
        self.num_pair = num_pair
        self.paired_lst = paired_lst

    def evaluate_assigned_results(self):
        df_trips = evaluate_individual_trips_sim(self)
        ret = summarize_results(df_trips)
        self.df_trips = df_trips
        self.df_summ = pd.DataFrame([ret], index=[0])

    def get_cp_trip_index(self):
        df_trips = self.df_trips
        # leave the SOV trips in case needed for assignment in the future...
        cp_idx = df_trips[df_trips['MODE'] != CPMode.SOV.value].index.tolist()
        return cp_idx


class SimulationWithTime:
    def __init__(
        self, 
        infrastructure: Infrastructure,
        trips: pd.DataFrame,
        carpool_modes: list[TripClusterAbstract],
        mode_configs: list[TripClusterConfig],
        clock: Clock
    ):
        self.infrastructure = infrastructure
        self.trips = trips
        self.carpool_modes = carpool_modes
        self.mode_configs = mode_configs
        self.clock = clock
        # summary results
        self.df_trips: pd.DataFrame | list[pd.DataFrame] = []
        self.df_summ: pd.DataFrame | list[pd.DataFrame] = []
        # temporary results (records to drop)
        self.cp_idx: list[int] = []
    
    def run_one_step(self):
        # select trips to be assigned
        trips1 = self.clock.select_trips(self.trips)
        trips1 = self.deselect_trips(trips1)
        if len(trips1) == 0:
            return
        # create a simulation task
        sim_task = SimulationTask(
            infrastructure = self.infrastructure,
            trips = trips1,
            carpool_modes = self.carpool_modes,
            mode_configs = self.mode_configs
        )
        # run the simulation
        sim_task.run_in_one_step()
        sim_task.gather_results()
        sim_task.optimize_results()
        sim_task.evaluate_assigned_results()
        self.cp_idx = sim_task.get_cp_trip_index()
        self.df_trips.append(sim_task.df_trips)
        self.df_summ.append(sim_task.df_summ)

    def run_all(self):
        while self.clock.t < self.clock.t_end:
            self.run_one_step()
            print(f"Time range: {self.clock.t} - {self.clock.t + self.clock.w}: ", end="")
            self.clock.update()

    def deselect_trips(self, trips_subset: pd.DataFrame) -> pd.DataFrame:
        return trips_subset.drop(self.cp_idx, errors="ignore")

    def combine_results(self):
        df_trips = pd.concat(self.df_trips)
        print(f"Total number of trips: {len(df_trips)}")
        df_trips = df_trips.drop_duplicates()
        print(f"Total number of unique trips: {len(df_trips)}")
        df_summ = summarize_results(df_trips)
        return df_trips, df_summ
