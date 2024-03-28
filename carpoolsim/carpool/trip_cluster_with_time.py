import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
import sqlalchemy

from carpoolsim.carpool.trip_cluster import TripCluster

plt.rcParams.update({'font.size': 22})
np.set_printoptions(precision=3)


class TripClusterWithTime(TripCluster):
    def __init__(
            self,
            df_all: pd.DataFrame,
            network: nx.DiGraph,
            links: pd.DataFrame,
            engine: sqlalchemy.Engine,
            delta_t: float,
            epsilon_t: float,
            parking_lots: pd.DataFrame | None = None,
    ):
        """
        For PNR case and other cases, need to update all the matrices every iteration
        :param df_all: trips to init the computation/simulation
        :param delta_t: maximum time difference. It decides the maximal number of drivers.
        :param epsilon_t: the update size of time for every simulation frame
        :param parking_lots: a DataFrame contains the locations of all parking lots
        """
        self.mode = -1  # -1 denote computation mode not set
        self.priority_mode = 0
        # drop useless columns to save memory in mass computation
        df_all = df_all.drop([
            'ox_sq', 'oy_sq', 'dx_sq', 'dy_sq',
            'depart_period', 'hh_id', 'newarr',
            'orig_inner_id', 'dest_inner_id'],
            axis=1, errors='ignore'
        )
        self.df_all = df_all.sort_values(by=['new_min']).copy()
        self.df_all['pnr'] = None
        # time information
        self.t0 = 0  # now
        self.epsilon = epsilon_t
        self.delta = delta_t
        self.t1 = self.epsilon + self.delta  # latest future time to be considered carpool
        # select data to init CarpoolSim matching
        df = self.df_all.loc[self.df_all['new_min'] < self.t1, :]
        # notice, needs to overwrite many methods and matrices of Trip Cluster since we consider
        # a rectangular carpool-able matrix instead of square matrix,
        # (future trips queries can only be passengers)
        self.trips_front = df.loc[df['new_min'] < self.t0 + self.epsilon, :]
        # NOTE: here df becomes self.trips (passegners + drivers);
        #   self.trips_front is for all drivers
        super().__init__(
            df, network, links, engine,
            parking_lots, update_now_str=True
        )
        # print('Start with {} potential drivers; {} potential passengers!'.format(
        #     len(self.trips_front), len(self.trips)))
        self.nrow = len(self.trips_front)
        self.ncol = len(self.trips)
        # truncate square matrices to rectangular matrices
        self.truncate_matrices(update_diag_info=False)
        # two new fields to store stat values
        self.stats_df = []
        self.stats_ind_df = []
        if parking_lots is not None:
            # init the parking lots matrix (done)
            pass

    def truncate_matrices(self, update_diag_info: bool = True):
        """ Truncate matrix and update diagonal information (e.g., using self.soloTimes)
        :param update_diag_info:
        :return:
        """
        # number of driver; number of passenger
        nrow, ncol = self.nrow, self.ncol
        # print("truncation nrow: {}; ncol: {}".format(nrow, ncol))
        self.cp_matrix = np.ones((nrow, ncol)).astype(np.bool_)  # overall carpool-able matrix
        self.tt_matrix = self.tt_matrix[:nrow, :ncol]  # travel time matrix
        self.tt_matrix_p1 = self.tt_matrix_p1[:nrow, :ncol]
        self.tt_matrix_p3 = self.tt_matrix_p3[:nrow, :ncol]
        self.ml_matrix = self.ml_matrix[:nrow, :ncol]  # store travel distance (mileage in miles)
        if self.parking_lots is not None:  # PNR mode (only among available drivers)
            # TODO: for each iteration, only recompute the newly loaded entries
            pnr_col = self.pnr_ncol
            self.pnr_01_matrix = self.pnr_01_matrix[:ncol, :pnr_col]
            self.pnr_access_info = self.pnr_access_info[:ncol, :pnr_col]
            # available only among drivers
            self.cp_pnr_matrix = self.cp_pnr_matrix[:nrow, :ncol]
            self.tt_pnr_matrix = self.tt_pnr_matrix[:nrow, :ncol]
            self.ml_pnr_matrix = self.ml_pnr_matrix[:nrow, :ncol]
            # self.ml_pnr_matrix_p = self.ml_pnr_matrix_p[:nrow, :nrow]

    def run_a_step_epsilon(
            self, rt_bipartite: bool = True,
            verbose: bool = True, print_mat: bool = False,
            mu1: float = 15, mu2: float = 0.1, dst_max: float = 5*5280,
            Delta1: float = 15, Delta2: float = 10, Gamma: float = 0.2,
            delta: float = 15, gamma: float = 1.5, ita: float = 0.5, ita_pnr: float = 0.5,
            plot_all: bool = False, mode: int = 0
    ):
        """
        Run assignment for one epsilon time.
        one can either choose bipartite method or lp method, not both
        Step 1: compute optimal
        Step 2: drop paired trips and preserve computation results
        Step 3: load new trips, update fields, etc.
        :param mode: the traffic mode to choose from
            mode 0: simple carpooling + SOV
            mode 1: PNR carpooling + SOV
            mode 2: simple + PNR carpooling + SOV (priority to shortest time)
            mode 3: simple + PNR carpooling + SOV (priority to PNR)
            mode 4: simple + PNR carpooling + SOV (priority to SOV)
        :return:
        """
        self.mode = mode
        self.priority_mode = None
        if mode <= 2:
            self.priority_mode = 0
        elif mode == 3:
            self.priority_mode = 1
        elif mode == 4:
            self.priority_mode = 2
        # Step 1. compute optimal (by different mode) (filtering step)
        if self.mode == 0:  # simple carpool
            # 1. get cp matrix
            self.compute_in_one_step(
                mu1=mu1, mu2=mu2, dst_max=dst_max,
                Delta1=Delta1, Delta2=Delta2, Gamma=Gamma,
                delta=delta, gamma=gamma, ita=ita)
        if self.mode == 1:  # simple PNR
            # 1. get cp matrix
            self.compute_in_one_step_pnr(
                mu1=mu1, mu2=mu2, dst_max=dst_max,
                Delta1=Delta1, Delta2=Delta2, Gamma=Gamma,
                delta=delta, gamma=gamma, ita=ita, ita_pnr=ita_pnr,
                include_direct=False, print_mat=print_mat)
        if self.mode >= 2:  # simple + PNR carpool
            self.compute_in_one_step_pnr(
                mu1=mu1, mu2=mu2, dst_max=dst_max,
                Delta1=Delta1, Delta2=Delta2, Gamma=Gamma,
                delta=delta, gamma=gamma, ita=ita, ita_pnr=ita_pnr,
                include_direct=True, print_mat=print_mat)
        # 2. compute optimal values (optimization step)
        lp_summ, lp_summ_ind, bt_summ, bt_summ_ind = \
            self.optimize_in_one_step(
                rt_bipartite=rt_bipartite, rt_LP=(not rt_bipartite),
                verbose=verbose, plot_all=plot_all,
                mode=self.mode
            )
        # store optimized results
        if rt_bipartite:
            self.stats_df.append(bt_summ)
            self.stats_ind_df.append(bt_summ_ind)
        else:
            self.stats_df.append(lp_summ)
            self.stats_ind_df.append(lp_summ_ind)

        # Step 2. drop all matched trips
        # get the indexes to be kept (unpaired passengers)
        if rt_bipartite:
            ind_df = bt_summ_ind
        else:
            ind_df = lp_summ_ind
        keep_trips_filt = (ind_df.new_min >= self.t0 + self.epsilon) & ind_df.SOV
        dropped_indexes = ind_df.loc[~keep_trips_filt, :].index.tolist()
        if verbose:
            left_ind = ind_df.loc[keep_trips_filt, :].index.tolist()
            # print('current table indexes', ind_df.index.tolist())
            # print('indexes to be kept', left_ind)
            # print('dropped_indexes', dropped_indexes)
            pass
        # Step 3. Update trips (drop old or matched trips, add new trips)
        self.update_trips(dropped_indexes, verbose=verbose)
        # if no driver to be matched, update to the next simulation t0 until there is driver...
        while self.nrow == 0 or self.ncol == 0:
            self.update_trips(dropped_indexes, verbose=verbose)
            if self.t0 > 1440:  # stop if over a day...
                break

    def update_trips(self, dropped_indexes: list[int], verbose: bool = False):
        """
        Add new trips coming into the trip
        :param dropped_indexes: indexes to trips to be dropped from last simulation step/iteration of optimization
        :param verbose: print
        :return:
        """
        # step 1. update time
        self.t0 += self.epsilon
        self.t1 += self.epsilon
        if verbose:
            print()
            print('updated time: t1={}, t2={}'.format(self.t0, self.t1))
        # step 2. delete assigned trips from self.df_all
        dropped_indexes = self.df_all.index.isin(dropped_indexes)
        self.df_all = self.df_all[~dropped_indexes]  # no need to keep matched trips recorded in df
        # step 3. select new interested trips to re-init object use time filter
        time_filt = (self.df_all['new_min'] >= self.t0) & (self.df_all['new_min'] < self.t1)
        # # step 4. reset values for the next computation
        # self.trip is updated here
        df = self.df_all.loc[time_filt, :].copy()
        super().__init__(
            df, self.network, self.links, self.engine,
            parking_lots=self.parking_lots, update_now_str=False
        )
        # trips front are drivers
        self.trips_front = self.trips.loc[self.trips['new_min'] < self.t0 + self.epsilon, :].copy()
        self.nrow = len(self.trips_front)
        self.ncol = len(self.trips)
        # truncate square matrices to rectangular matrices
        self.truncate_matrices()  # also update truncation
        if verbose:
            print('{} potential drivers; {} potential passengers!'.format(len(self.trips_front), len(self.trips)))
            # print('trip depart time: ', self.trips.new_min.tolist())
            pass

    def run_all_sim(self, kwargs=None):
        """
        Run all simulation for the whole day (whole dataset)
        :param kwargs: (rt_bipartite=True, verbose=True, delta=15, gamma=1.5, plot_all=False)
        :return:
        """
        t0 = time.time()
        if kwargs is None:
            kwargs = {
                'rt_bipartite': True,
                'verbose': True, 'print_mat': False, 'plot_all': False,
                'mu1': 15, 'mu2': 0.1, 'dst_max': 5 * 5280,  # coordinate filters
                'delta': 15, 'gamma': 1.5, 'ita': 0.8, 'ita_pnr': 0.5,  # reroute filters
                'Delta1': 15, 'Delta2': 10, 'Gamma': 0.2,
                'mode': 0
            }

        # as long as time is done, continue to run sim and do optimization
        while self.t0 <= 1440:
            if kwargs['verbose']:
                print(f'Update a new time frame. t0 is {self.t0}, t1 is {self.t1}')
            self.run_a_step_epsilon(**kwargs)
        # gather stats
        summ_stats, summ_stats_ind = self.gather_stats()
        # return important stats
        delta_t = time.time() - t0
        print('Finished running all simulation in {:.2f} minutes for a trip cluster with {} trips!'.format(
            delta_t / 60, summ_stats.tot_num.sum()))
        return summ_stats, summ_stats_ind

    def gather_stats(self):
        summ_stats = pd.concat(self.stats_df)
        summ_stats = pd.DataFrame(summ_stats.sum(axis=0)).transpose()
        summ_stats_ind = pd.concat(self.stats_ind_df)
        return summ_stats, summ_stats_ind

    def compute_depart_01_matrix_pre(self, Delta1: float = 15, default_rule: bool = False):
        """
        Overload method for this case
        Filter out carpool trips based on time constraint conditions.
        Two trips are carpool-able if:
            Default rule: Driver departs before rider but no earlier than the given threshold.
            Alternative rule: The time difference between two departures are within a fixed time threshold.
        :return:
        """
        nrow, ncol = self.nrow, self.ncol
        passenger_lst = np.array(self.trips['new_min'].tolist()).reshape((1, -1))  # depart minute
        driver_lst = np.array(self.trips_front['new_min'].tolist()).reshape((1, -1))  # depart minute
        # compare departure time difference
        dri_dep = np.tile(driver_lst.transpose(), (1, ncol))
        pax_dep = np.tile(passenger_lst, (nrow, 1))
        mat = pax_dep - dri_dep  # passenger departs "mat" minutes after driver departs
        if default_rule:
            # driver should leave earlier than the passenger
            self.cp_matrix = (self.cp_matrix &
                              (mat >= 0) &
                              (np.absolute(mat) <= Delta1)).astype(np.bool_)
        else:
            # alternative rule: no option
            self.cp_matrix = (self.cp_matrix &
                              (np.absolute(mat) <= Delta1)).astype(np.bool_)

    def compute_depart_01_matrix_pre_pnr(self, Delta1: float = 15, default_rule: bool = False):
        nrow, ncol = self.nrow, self.ncol
        # simple method
        driver_lst = np.array(self.trips_front['new_min'].tolist()).reshape((1, -1))
        passenger_lst = np.array(self.trips['new_min'].tolist()).reshape((1, -1))  # depart minute
        mat = np.tile(driver_lst.transpose(), (1, ncol))
        mat = np.tile(passenger_lst, (nrow, 1)) - mat  # depart time difference (driver's depart - pax depart)
        if default_rule:
            # criterion 1. driver should leave earlier than the passenger, but not earlier than 15 minutes
            # criterion 2. driver should wait at most 5 minutes
            self.cp_pnr_matrix = (self.cp_pnr_matrix &
                                  (mat >= 0) &
                                  (np.absolute(mat) <= Delta1))[:nrow, :ncol].astype(np.bool_)
        else:
            # criterion 1. passenger/driver depart time within +/- 15 minutes
            # criterion 2. passenger/driver wait time +/- 5 minutes
            self.cp_pnr_matrix = (self.cp_pnr_matrix &
                                  (np.absolute(mat) <= Delta1))[:nrow, :ncol].astype(np.bool_)

    def compute_depart_01_matrix_post(self, Delta2: float = 10, Gamma: float = 0.2, default_rule: bool = True):
        """
        After tt_matrix_p1 is computed, filter by maximum waiting time for the driver at pickup location
        :param Delta2: driver's maximum waiting time
        :param default_rule: if True, strict time different; if False, absolute time difference
        :return:
        """
        # step 2. Maximum waiting time for driver is Delta2 (default is 5 minutes)
        nrow, ncol = self.nrow, self.ncol
        # print("nrow: {}; ncol: {}".format(nrow, ncol))
        # print(self.tt_matrix_p1.shape)
        driver_lst = np.array(self.trips_front['new_min'].tolist()).reshape((-1, 1))  # depart minute
        passenger_lst = np.array(self.trips['new_min'].tolist()).reshape((1, -1))  # depart minute
        # print("driver lst:", driver_lst.tolist())
        # print("passenger lst:", passenger_lst.tolist())
        # compare departure time difference
        dri_arr = np.tile(driver_lst, (1, ncol)) + self.tt_matrix_p1
        pax_dep = np.tile(passenger_lst, (nrow, 1))  # depart time difference
        # step 2. Maximum waiting time for driver is Delta2 (default is 5 minutes)
        pax_wait_time_mat = dri_arr - pax_dep  # wait time matrix
        # print(pax_wait_time_mat/self.tt_matrix)
        # print(np.argwhere(pax_wait_time_mat/self.tt_matrix <= Gamma))
        # for post analysis, directly update final cp_matrix
        passenger_time = np.array([self.soloTimes[i] for i in range(self.ncol)]).reshape(1, -1)
        passenger_time = np.tile(passenger_time, (nrow, 1))
        # print("self.tt_matrix_p1 not nan:", self.tt_matrix_p1[~np.isnan(self.tt_matrix_p1)])
        # print("passenger waiting time:", pax_wait_time_mat[~np.isnan(pax_wait_time_mat)])
        # print("num cp_matrix (before):", np.sum(self.cp_matrix))
        if default_rule:
            # passenger only waits the driver should wait at most Delta2 minutes
            self.cp_matrix = (self.cp_matrix &
                              (pax_wait_time_mat >= 0) & (np.absolute(pax_wait_time_mat) <= Delta2) &
                              (np.absolute(pax_wait_time_mat / passenger_time) <= Gamma)
                              ).astype(np.bool_)
        else:
            # passenger/driver waits the other party for at most Delta2 minutes
            self.cp_matrix = (self.cp_matrix &
                              (np.absolute(pax_wait_time_mat) <= Delta2) &
                              (np.absolute(pax_wait_time_mat / passenger_time) <= Gamma)
                              ).astype(np.bool_)

    def compute_depart_01_matrix_post_pnr(
            self, Delta2: float = 10, Gamma: float = 0.2, default_rule: bool = True
    ):
        """
        Nothing changed for post analysis except fix candidates to drivers
        :param Delta2:
        :param default_rule: True: driver should wait at most Delta2 minutes for the passenger
            False: driver/passenger should wait each other at most Delta2 minutes
        :return:
        """
        # complex and slow method for post analysis
        nrow, ncol = self.nrow, self.ncol
        # step 1. get the travel time to each feasible station
        # ind = list(zip(*np.where(self.cp_pnr_matrix == 1)))
        ind = np.argwhere(self.cp_pnr_matrix == 1)
        # print("inds:", ind)
        # pnr depart time matrix (diff between drivers)
        mat = np.full((nrow, ncol), np.nan)
        for ind_one in ind:  # only consider among drivers
            trip_id1, trip_id2 = ind_one
            # print("trips ids:", trip_id1, trip_id2, "; trips front len:", len(self.trips_front))
            if trip_id1 >= nrow or trip_id2 >= nrow:
                continue
            # get station id they share
            trip_row1 = self.trips.iloc[trip_id1]
            trip_row2 = self.trips.iloc[trip_id2]
            sid = self._check_trips_best_pnr(trip_row1, trip_row2, trip_id1, trip_id2)
            if sid is None or sid == -1:
                continue
            # access information (path, time, distance)
            info1 = self.pnr_access_info[trip_id1, sid]
            info2 = self.pnr_access_info[trip_id2, sid]
            t1 = self.trips['new_min'].iloc[trip_id1] + info1[1]  # arrival time at pnr for person 1
            t2 = self.trips['new_min'].iloc[trip_id2] + info2[1]  # arrival time at pnr for person 2
            # the # of minutes it takes for passenger to arrive after driver arrived
            mat[trip_id1][trip_id2] = t1 - t2
            mat[trip_id2][trip_id1] = t2 - t1

        passenger_time = np.array([self.soloTimes[i] for i in range(self.ncol)]).reshape(1, -1)
        passenger_time = np.tile(passenger_time, (nrow, 1))
        if default_rule:
            # passenger should arrive earlier than the driver, plus
            # only the driver should wait at most Delta2 minutes
            self.cp_pnr_matrix = (self.cp_pnr_matrix &
                                  (mat >= 0) &
                                  (mat <= Delta2) &
                                  (np.absolute(mat/passenger_time) <= Gamma)).astype(np.bool_)
        else:
            # passenger/driver depart time within +/- 10 minutes
            self.cp_pnr_matrix = (self.cp_pnr_matrix &
                                  (np.absolute(mat) <= Delta2) &
                                  (np.absolute(mat/passenger_time) <= Gamma)).astype(np.bool_)

    def compute_pickup_01_matrix(
            self, threshold_dist: float = 5280 * 5, mu1: float = 1.5, mu2: float = 0.1, use_mu2: bool = True
    ):
        """
        Compute feasibility matrix based on whether one can pick up/ drop off passengers.
        If A can pick up B in threshold (e.g., 5 miles), then it is feasible. Otherwise, it is not a feasible carpool.
        Use Euclidean Distance.

        This function run all the filtering tasks using coordinates.
        :param threshold_dist: the distance between pickups are within the distance in miles (default is 5 mile)
        :param mu1: the maximum ratio between carpool vector distance and SOV vector distance. Vector distance is
        the length connecting origin/destination coordinates.

        :param mu2: the maximum ratio of backward traveling distance after drop off passengers.
        :param use_mu2: If True, measure backward traveling distance.
        V_O1D1 defined as SOV trip vector
        V_D2D1 defined the last portion of carpool trip vector (vector connecting passenger's dest. to driver's origin)
        the backward ratio is defined as:  - (V_O1D1 * V_D2D1) / (V_D2D1 * V_D2D1).
        The filter holds for all (i,j) pairs with: - (V_O1D1 * V_D2D1) / (V_D2D1 * V_D2D1) < mu2
        """
        nrow, ncol = self.nrow, self.ncol

        oxs_driver = np.array(self.trips_front.ox.tolist()).reshape((1, -1))
        oys_driver = np.array(self.trips_front.oy.tolist()).reshape((1, -1))
        oxs_passenger = np.array(self.trips.ox.tolist()).reshape((1, -1))
        oys_passenger = np.array(self.trips.oy.tolist()).reshape((1, -1))

        dxs_driver = np.array(self.trips_front.dx.tolist()).reshape((1, -1))
        dys_driver = np.array(self.trips_front.dy.tolist()).reshape((1, -1))
        dxs_passenger = np.array(self.trips.dx.tolist()).reshape((1, -1))
        dys_passenger = np.array(self.trips.dy.tolist()).reshape((1, -1))

        # compute coordinate difference
        mat_ox = np.tile(oxs_driver.transpose(), (1, ncol))
        mat_ox = np.abs(mat_ox - np.tile(oxs_passenger, (nrow, 1)))
        mat_oy = np.tile(oys_driver.transpose(), (1, ncol))
        mat_oy = np.abs(mat_oy - np.tile(oys_passenger, (nrow, 1)))

        mat_dx = np.tile(dxs_driver.transpose(), (1, ncol))
        mat_dx = np.abs(mat_dx - np.tile(dxs_passenger, (nrow, 1)))
        mat_dy = np.tile(dys_driver.transpose(), (1, ncol))
        mat_dy = np.abs(mat_dy - np.tile(dys_passenger, (nrow, 1)))

        dst_o = np.sqrt(mat_ox ** 2 + mat_oy ** 2)
        dst_d = np.sqrt(mat_dx ** 2 + mat_dy ** 2)

        # compute euclidean travel distance
        mat_diag = np.sqrt((oxs_passenger - dxs_passenger) ** 2 + (oxs_passenger - dys_passenger) ** 2)
        # print(man_o.shape, man_d.shape)
        # compute manhattan travel distance
        # mat_diag = np.abs(oxs_passenger - dxs_passenger) + np.abs(oxs_passenger - dys_passenger)  # l1 distance
        rr = (dst_o + dst_d + np.tile(mat_diag, (nrow, 1)))
        ori = np.tile(mat_diag, (nrow, 1))
        mat_ratio = rr / ori

        # filtering condition only for community pickup scenarios
        self.cp_matrix = (self.cp_matrix &
                          (dst_o <= threshold_dist) &
                          (mat_ratio < mu1)).astype(np.bool_)

        if use_mu2:
            # now it is time for implementing backward constraint
            # compute the vector for all drivers V_{O1D1}
            mat_x_o1d1 = (dxs_driver - oxs_driver).reshape((1, -1))
            mat_y_o1d1 = (dys_driver - oys_driver).reshape((1, -1))
            # compute vector V_{D2D1} from passenger's destination to driver's destination
            mat_x_d2d1 = np.tile(dxs_driver.transpose(), (1, ncol))
            mat_x_d2d1 = np.abs(mat_x_d2d1 - np.tile(dxs_passenger, (nrow, 1)))
            mat_y_d2d1 = np.tile(dys_driver.transpose(), (1, ncol))
            mat_y_d2d1 = np.abs(mat_y_d2d1 - np.tile(dys_passenger, (nrow, 1)))
            # compute vector angle for each composition position
            # print(np.tile(mat_x_o1d1.transpose(), (1, ncol)).shape)
            part1 = -(np.tile(mat_x_o1d1.transpose(), (1, ncol)) * mat_x_d2d1 +
                      np.tile(mat_y_o1d1.transpose(), (1, ncol)) * mat_y_d2d1)
            part2 = (np.tile(mat_x_o1d1.transpose() ** 2, (1, ncol)) +
                     np.tile(mat_y_o1d1.transpose() ** 2, (1, ncol)))
            backward_index = part1 / part2
            # if backward_index.shape[0] > 0:
            #     print(backward_index.min(), backward_index.max())
            np.fill_diagonal(backward_index, -1)
            # self.cp_dropoff_matrix = (backward_index <= mu2).astype(int)
            self.cp_matrix = (self.cp_matrix &
                              (backward_index <= mu2)).astype(np.bool_)

    def compute_reroute_01_matrix(
            self, delta: float = 15, gamma: float = 1.5, ita: float = 0.9
    ):
        """
        Compute carpool-able matrix considering total reroute time (non-shared trip segments)
        This cannot be estimated before travel time matrix (self.tt_matrix) is fully computed
        :param delta: maximum reroute time (in minutes) acceptable for the driver
        :param gamma: the maximum ratio of extra travel time over driver's original travel time
        :param ita: passenger's travel time should be at greater than ita times travel time of that drivers
        :return:
        """
        nrow, ncol = self.nrow, self.ncol
        # print(nrow, ncol)
        # propagate drive alone matrix
        # drive_alone_tt = self.tt_matrix.diagonal().reshape(-1, 1)
        drive_alone_tt = np.array([self.soloTimes[i] for i in range(self.nrow)]).reshape(-1, 1)
        # print(f"nrow:{self.nrow}; ncol:{self.ncol};")
        # print(drive_alone_tt.shape)
        # condition 1: reroute time is smaller than threshold minutes
        cp_reroute_matrix = ((self.tt_matrix - np.tile(drive_alone_tt, (1, ncol))) <= delta).astype(int)
        # condition 2: ratio is smaller than a threshold
        cp_rerouteratio_matrix = ((self.tt_matrix / np.tile(drive_alone_tt, (1, ncol))) <= gamma).astype(int)
        # condition 3: rider should at least share ita of the trip
        shared_tt = self.tt_matrix - self.tt_matrix_p1 - self.tt_matrix_p3
        # notice that cp_time_similarity is not recorded by the obj
        cp_time_similarity = ((shared_tt / self.tt_matrix) >= ita).astype(int)
        # print('reroute time matrix')
        # print(cp_reroute_matrix[:8, :8])
        # print('reroute ratio matrix')
        # print(cp_rerouteratio_matrix[:8, :8])
        self.cp_matrix = (self.cp_matrix & cp_reroute_matrix * cp_rerouteratio_matrix)
        # update the overall carpool-able 0-1 matrix
        self.cp_matrix = (self.cp_matrix & cp_time_similarity).astype(np.bool_)
        # need to mask tt and ml matrix
        # self.tt_matrix = self.tt_matrix * self.cp_matrix
        # self.ml_matrix = self.ml_matrix * self.cp_matrix
        self.tt_matrix[self.cp_matrix == 0] = np.nan
        self.ml_matrix[self.cp_matrix == 0] = np.nan

    def compute_carpoolable_trips_pnr(self, reset_off_diag=False):
        nrow, ncol = self.nrow, self.ncol
        if reset_off_diag:  # wipe and reset out all off-diagonal values
            temp_diag_tt = self.tt_pnr_matrix.diagonal()
            temp_diag_ml = self.ml_pnr_matrix.diagonal()
            # reset travel time information
            self.tt_pnr_matrix = np.full((nrow, ncol), np.nan)
            self.ml_pnr_matrix = np.full((nrow, ncol), np.nan)
            np.fill_diagonal(self.tt_pnr_matrix, temp_diag_tt)  # many empty objects
            np.fill_diagonal(self.ml_pnr_matrix, temp_diag_ml)
        indexes_pairs = np.argwhere(self.cp_pnr_matrix == 1)
        indexes_pairs = [index for index in indexes_pairs if index[0] != index[1]]
        # indexes_pairs = [index for index in indexes_pairs if max(index[0], index[1]) < ncol]
        # print('PNR Indices matching 1: \n', indexes_pairs)
        for index in indexes_pairs:
            self.compute_carpool_pnr(index[0], index[1], fixed_role=True, trips=self.trips)

    def compute_reroute_01_matrix_pnr(
            self, delta: float = 15, gamma: float = 1.5, ita_pnr: float = 0.5,
            print_mat: bool = True
    ):
        """ Calculate specific traveling plan for each party of the PNR trip.
        :param delta: reroute minutes at most for each PNR traveler
        :param gamma: reroute ratio at most for each PNR traveler
        :param ita_pnr: ratio of shared trip for each PNR traveler
        :param print_mat:
        :return:
        """
        nrow, ncol = self.nrow, self.ncol
        # propagate drive alone matrix
        drive_alone_tt = np.array([self.soloTimes[i] for i in range(self.nrow)]).reshape(-1, 1)
        passenger_alone_tt = np.array([self.soloTimes[i] for i in range(self.ncol)]).reshape(1, -1)
        # print(f"nrow:{self.nrow}; ncol:{self.ncol}; npnr:{self.pnr_ncol}")
        # print("passenger alone tt")
        # print(pax_alone_tt)
        # condition 1: total reroute time is smaller than threshold minutes
        # print(self.tt_pnr_matrix.shape, nrow)
        # print(np.tile(drive_alone_tt, (1, ncol)).shape)
        cp_reroute_matrix1 = ((self.tt_pnr_matrix - np.tile(drive_alone_tt, (1, ncol)))
                              <= delta).astype(np.bool_)
        cp_reroute_matrix2 = ((self.tt_pnr_matrix - np.tile(passenger_alone_tt, (nrow, 1)))
                              <= delta).astype(np.bool_)
        cp_reroute_matrix = cp_reroute_matrix1 & cp_reroute_matrix2
        # condition 2: ratio is smaller than a threshold
        cp_reroute_ratio_matrix1 = ((self.tt_pnr_matrix / np.tile(drive_alone_tt, (1, ncol)))
                                    <= gamma).astype(np.bool_)
        cp_reroute_ratio_matrix2 = ((self.tt_pnr_matrix / np.tile(passenger_alone_tt, (nrow, 1)))
                                    <= gamma).astype(np.bool_)
        cp_reroute_ratio_matrix = cp_reroute_ratio_matrix1 & cp_reroute_ratio_matrix2
        # condition 3: rider should at least share ita_pnr of the total travel time
        cp_time_similarity = ((self.tt_pnr_matrix_shared[:self.nrow, :self.ncol] / self.tt_pnr_matrix)
                              >= ita_pnr).astype(np.bool_)
        # print(passenger_time / self.tt_pnr_matrix)
        # condition 4: after drop-off time / whole travel time?
        # print("drive alone...")
        # print(drive_alone_tt[:8, :8])
        if print_mat:
            print("PNR path-based filters results:")
            print("cp_reroute_matrix passed count:", cp_reroute_matrix.sum())
            # print(cp_reroute_matrix[:8, :8])
            print("cp_reroute_ratio_matrix passed count:", cp_reroute_ratio_matrix.sum())
            # print(cp_reroute_ratio_matrix[:8, :8])
            print("cp_time_similarity passed count:", cp_time_similarity.sum())
            # print(cp_time_similarity[:8, :8])
        # print("PNR path-based filters results:")
        # print("all passed count (before):", self.cp_pnr_matrix.sum())
        self.cp_pnr_matrix = (self.cp_pnr_matrix &
                              cp_reroute_matrix &
                              cp_reroute_ratio_matrix &
                              cp_time_similarity).astype(np.bool_)
        # print("cp_reroute_matrix passed count:", cp_reroute_matrix.sum())
        # # print(cp_reroute_matrix[:8, :8])
        # print("cp_reroute_ratio_matrix passed count:", cp_reroute_ratio_matrix.sum())
        # # print(cp_reroute_ratio_matrix[:8, :8])
        # print("cp_time_similarity passed count:", cp_time_similarity.sum())
        # print("all passed count (after):", self.cp_pnr_matrix.sum())
        # print()
        # need to mask tt and ml matrix
        self.tt_pnr_matrix[self.cp_pnr_matrix == 0] = np.nan
        self.ml_pnr_matrix[self.cp_pnr_matrix == 0] = np.nan
        # TODO: same filters applied to the passenger's side
        # passenger's travel time = "driver's travel time" - "driver's access "

    def combine_pnr_carpool(
            self, include_direct: bool = True, print_mat: bool = True
    ) -> None:
        nrow, ncol = self.nrow, self.ncol
        if include_direct:
            # 0 for simple travel, 1 for PNR travel
            self.choice_matrix = self._generate_choice_matrix(self.cp_matrix, self.cp_pnr_matrix)
            self.cp_matrix_all = np.fmax(self.cp_matrix, self.cp_pnr_matrix)
            # self.cp_matrix_all = np.hstack((self.cp_matrix_all, self.cp_matrix))
            # for PNR candidates
            inds = np.where(self.choice_matrix == 1)
            self.ml_matrix_all = self.ml_matrix.copy()
            self.tt_matrix_all = self.tt_matrix.copy()
            self.ml_matrix_all[inds] = self.ml_pnr_matrix[inds]
            self.tt_matrix_all[inds] = self.tt_pnr_matrix[inds]
        else:  # consider only PNR carpooling
            self.cp_matrix_all = self.cp_pnr_matrix.copy()
            self.tt_matrix_all = self.tt_pnr_matrix.copy()
            self.ml_matrix_all = self.ml_pnr_matrix.copy()
            # 0 for simple carpool, 1 for PNR travel
            self.choice_matrix = np.full(self.cp_pnr_matrix.shape, 1, dtype="int8")
        # drive alone as default along diagonal
        # fill diagonals
        np.fill_diagonal(self.cp_matrix_all, self.cp_matrix.diagonal())
        np.fill_diagonal(self.tt_matrix_all, self.tt_matrix.diagonal())
        np.fill_diagonal(self.ml_matrix_all, self.ml_matrix.diagonal())
        np.fill_diagonal(self.choice_matrix, 0)
        if print_mat:
            print("choice matrix head (one mode): {} vs. {}".format(
                (self.choice_matrix == 0).sum(), (self.choice_matrix == 1).sum()
            ))
            print(self.choice_matrix[:8, :8])
