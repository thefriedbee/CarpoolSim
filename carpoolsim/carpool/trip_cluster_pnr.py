"""
Apply PNR carpooling mode to a set of trips.
"""
import numpy as np
import pandas as pd
from carpoolsim.carpool.trip_demands import TripDemands
from carpoolsim.carpool.trip_cluster_abstract import TripClusterAbstract
from carpoolsim.carpool.util.network_search import (
    get_path_distance_and_tt, 
    dynamic_shortest_path_search
)
from carpoolsim.carpool.util.mat_operations import (
    get_trip_projected_ods,
    get_pnr_xys,
    get_distances_between_ods_sov,
    get_distances_between_ods_matrix,
)
from carpoolsim.carpool.util.filters_pnr import (
    generate_pnr_trip_map_filt,
    compute_depart_01_matrix_pre_pnr,
    compute_reroute_01_matrix_pnr,
    compute_depart_01_matrix_post_pnr
)


class TripClusterPNR(TripClusterAbstract):
    def __init__(
        self, 
        trip_demands: TripDemands
    ):
        super().__init__(trip_demands)
        pnr_ncol = len(self.parking_lots)
        N = len(self.td.trips)
        self.pnr_matrix = np.full((N, pnr_ncol), 1).astype(np.bool_)
        # store all info in one matrix of objects
        self.pnr_access_info = np.empty((N, pnr_ncol), dtype=object)
        # 0-1 matrix for PNR travelers, start with all available and shrink by filters
        self.cp_matrix = np.full((N, N), 1, dtype=np.bool_)
        self.tt_matrix = np.full((N, N), np.nan, dtype="float32")
        self.ml_matrix = np.full((N, N), np.nan, dtype="float32")
        # A CENTRIC VIEW OF DRIVERS (3 trip segments of a carpool driver)
        # p1: pickup travel time for driver
        # p2: shared travel time for driver and passenger
        # p3: drop-off travel time for driver
        self.tt_matrix_p2 = np.full((N, N), np.nan, dtype="float32")
    
    @property
    def pnr_ncol(self):
        return len(self.parking_lots)
    
    @property
    def trips(self):
        return self.td.trips
    
    @property
    def parking_lots(self):
        return self.td.parking_lots
    
    @property
    def network(self):
        return self.td.network
    
    def compute_pnr_access(self, trip_id: int, station_id: int) -> None:
        """
        Compute access time to PNR station
        :return:
        """
        # load dataframe series
        trip_row = self.trips.iloc[trip_id, :]
        pnr_row = self.parking_lots.iloc[station_id, :]
        network = self.network

        def calculateAccess(trip, pnr):
            O1, O1_taz = trip['o_node'], trip['orig_taz']
            M1, M1_taz = pnr['node'], pnr['taz']
            D1, D1_taz = trip['d_node'], trip['dest_taz']
            p1, t1, d1 = dynamic_shortest_path_search(network, O1, M1, O1_taz, M1_taz)
            p2, t2, d2 = dynamic_shortest_path_search(network, M1, D1, M1_taz, D1_taz)
            # return access info (nodes, time, and distance), and travel time goes through PNR station
            return p1, t1, d1, t1+t2
        p1, t1, d1, t_all = calculateAccess(trip_row, pnr_row)
        self.pnr_access_info[trip_id, station_id] = [p1, t1, d1, t_all]

    def _check_trips_best_pnr(
        self, 
        trip_row1: pd.Series, 
        trip_row2: pd.Series, 
        int_idx1: int, 
        int_idx2: int
    ):
        """
        Just check if two trips can share one PNR station.
        If so, among the stations, choose the best station for joint trip
        :param trip_row1: Pandas Series for trip 1
        :param trip_row2: Pandas Series for trip 2
        :return: the PNR station reachable by both trips (shortest total travel time)
        """
        if trip_row1['pnr'] is None or trip_row2['pnr'] is None:
            return None
        pnr1, pnr2 = trip_row1['pnr'], trip_row2['pnr']
        lst = list(set(pnr1) & set(pnr2))
        if len(lst) == 0:
            return None
        if len(lst) == 1:
            return lst[0]
        # if can share multiple stations
        time_lst = []
        for i, sid in enumerate(lst):
            # compute on demand
            self.compute_pnr_access(int_idx1, sid)
            self.compute_pnr_access(int_idx2, sid)
            info1 = self.pnr_access_info[int_idx1, sid]
            info2 = self.pnr_access_info[int_idx2, sid]
            tot_time = info1[1] + info2[1]  # two traveler's total access time to station
            time_lst.append(tot_time)
        time_lst = np.array(time_lst)
        i = np.argmin(time_lst)
        return lst[i]

    def compute_01_matrix_to_station_p1(
        self,
        threshold_dist: float = 5280 * 5,
        mu1: float = 1.3,
        trips: pd.DataFrame | None =None,
        print_mat: bool = True
    ) -> None:
        """
        For each trip, compute the parking lots that can be used as the PNR "meetings point" for carpooled trip.
        step 1 (this function). Use Euclidean distance between coordinates
        step 2. check actual travel time & distance (compute_01_matrix_to_station_p2)

        Assume trip's ODs are O1 and D1, consider stop at M1 for a middle PNR station.
        The following conditions should be satisfied (in Euclidean distance):
        1. O1-->M1 is within 5 miles
        2. (O1 M1) + (M1 D1) < p1 * (O1 D1)
        Notice: all above measures in Euclidean distance between coordinates
        :param threshold_dist: PNR should be relatively close at home
        :param mu1: maximum reroute ratio
        :param mu2: portion of the trip segment arriving at destination over all trip
        :param use_mu2: If True, measure backward traveling distance.
        :return:
        """
        if trips is None:
            trips = self.trips
        # nrow: num of drivers; ncol: num of passengers
        num_driver, num_all, lots_ncol = self.nrow, self.ncol, self.pnr_ncol
        oxs, oys, dxs, dys = get_trip_projected_ods(trips)
        # positions of the parking lots 
        # (mxs: middle x coordinates, mys: middle y coordinates)
        mxs, mys = get_pnr_xys(self.parking_lots)

        # SOV case
        #   distance of O1 --> D1
        mat_od_dx, mat_od_dy, man_od = get_distances_between_ods_sov(oxs, oys, dxs, dys)
        # PNR case (stop by M1)
        #   distance of O1 --> M1
        mat_om_dx, mat_om_dy, man_om = get_distances_between_ods_matrix(oxs, oys, mxs, mys)
        #   distance of M1 --> D1 (not the order of inputs are reversed to match the shapes above)
        mat_md_dx, mat_md_dy, man_md = get_distances_between_ods_matrix(dxs, dys, mxs, mys)   

        # compute reroute distance using pnr facility
        pre_dist = man_od.reshape(-1, 1)
        post_dist = (man_om + man_md)
        mat_ratio = (post_dist / pre_dist).astype("float32")

        self.pnr_matrix = (self.pnr_matrix &
                           (mat_ratio < mu1) &
                           (man_om <= threshold_dist))
        if print_mat:
            print("mat_ratio total pass is (<= reroute ratio):", (mat_ratio < mu1).sum())
            print("man_om total pass is (<= threshold miles):", (man_om <= threshold_dist).sum())
            print("pnr 0-1 matrix total pass is (after basic euclidean filters):", self.pnr_matrix.sum())

    def compute_01_matrix_to_station_p2(
        self,
        delta: float = 15,
        gamma: float = 1.5,
    ) -> None:
        """
        Make sure each SOV trip can go through PNR station without too much extra costs.
        step 2. check actual travel time & distance
        :param delta: maximum reroute time (in minutes) acceptable for the driver
        :param gamma: the maximum ratio of extra travel time over driver's original travel time
        :return:
        """
        # just compute travel information through PNR for all trips
        # meanwhile, for each trip, filter by delta and gamma condition
        # update matrix self.pnr_access_info
        generate_pnr_trip_map_filt(trip_cluster=self, delta=delta, gamma=gamma)

    def compute_carpool(
        self, 
        int_idx1: int, int_idx2: int,
        print_dist: bool = False,
        fill_mat: bool = True,
        fixed_role: bool = False,
        trips: pd.DataFrame | None = None
    ):
        """
        Given the integer index of two trips (of the trip DataFrame self.df),
        compute distances and paths of park and ride scenarios, that is:
        1. B meet A at a midpoint 2. A meet B at a midpoint
        Currently, implement first 2 scenarios. 3 and 4 will be done in the future
        :param int_idx1: integer index for the first traveler A
        :param int_idx2: integer index for the second traveler B
        :param fill_mat: store computation results to matrices
        :param print_dist: print trip plans (for debugging)

        :param fixed_role: if True, int_idx1 is for the driver, int_idx2 is for the passenger
            Otherwise, we are trying to call all PERMUTATIONS of carpool with roles,
            (call 1 picks up 2 then call 2 picks up 1, for future conclude pickup location) then set this to True.
            If call on COMBINATIONS, set this to False.
        :return: paths and links for the all scenarios (two for now)
        """
        if trips is None:
            trips = self.trips
        trip1, trip2 = trips.iloc[int_idx1, :], trips.iloc[int_idx2, :]

        def calculatePNRCarpool(trip1, trip2, station_id, reversed=False):
            """
            Calculate specific PNR plans for two trips
            :param trip1: the driver's trip info
            :param trip2: the passenger's trip info
            :param station_id: midpoint station to use
            :param reversed:
            :return: if False, trip1 is the driver. Otherwise, trip2 is the driver.
            """
            network = self.network
            O1, D1, O2, D2 = trip1['o_node'], trip1['d_node'], trip2['o_node'], trip2['d_node']
            O1_taz, D1_taz, O2_taz, D2_taz = trip1['orig_taz'], trip1['dest_taz'], trip2['orig_taz'], trip2['dest_taz']
            pnr_row = self.parking_lots.iloc[station_id, :]
            M1, M_taz = pnr_row['node'], pnr_row['taz']
            # print("trip_id: {}, station_id: {}".format(int_idx1, station_id))
            if not reversed:  # O1 ==> M1 ==> D2 ==> D1
                p0, t0, d0, t_all0 = self.pnr_access_info[int_idx2, station_id]  # O2 ==> M1
                p1, t1, d1, t_all1 = self.pnr_access_info[int_idx1, station_id]  # O1 ==> M1
                p2, t2, d2 = dynamic_shortest_path_search(network, M1, D2, M_taz, D2_taz)  # M1 ==> D2
                p3, t3, d3 = dynamic_shortest_path_search(network, D2, D1, D2_taz, D1_taz)  # D2 ==> D1
            else:  # O2 ==> M1 ==> D1 == D2
                p0, t0, d0, t_all0 = self.pnr_access_info[int_idx1, station_id]  # O1 ==> M1
                p1, t1, d1, t_all1 = self.pnr_access_info[int_idx2, station_id]  # O2 ==> M1
                p2, t2, d2 = dynamic_shortest_path_search(network, M1, D1, M_taz, D1_taz)  # M1 ==> D1
                p3, t3, d3 = dynamic_shortest_path_search(network, D1, D2, D2_taz, D2_taz)  # D1 ==> D2
            if print_dist:
                print(' d1: {}; d2: {}; d3: {}'.format(d1, d2, d3))
            return t0, t1, t2, t3, d0, d1, d2, d3, p0, p1, p2, p3
        # A pickup B at a midpoint
        # check if two trips can share one pnr station
        sid = self._check_trips_best_pnr(trip1, trip2, int_idx1, int_idx2)  # get the feasible midpoints
        if sid is None or sid == -1:
            # print("No feasible join carpool plan using PNR found", end=";")
            return None
        # else:  # contain feasible midpoints
        p2_tt_p0_pnr, d1_tt_p1_pnr, d1_tt_p2_pnr, d1_tt_p3_pnr, \
        p2_ml_p0_pnr, d1_ml_p1_pnr, d1_ml_p2_pnr, d1_ml_p3_pnr, \
        p2_p_p0_pnr, d1_p_p1_pnr, d1_p_p2_pnr, d1_p_p3_pnr = \
            calculatePNRCarpool(trip1, trip2, sid, reversed=False)
        if fixed_role is False:
            p1_tt_p0_pnr, d2_tt_p1_pnr, d2_tt_p2_pnr, d2_tt_p3_pnr, \
            p1_ml_p0_pnr, d2_ml_p1_pnr, d2_ml_p2_pnr, d2_ml_p3_pnr, \
            p1_p_p0_pnr, d2_p_p1_pnr, d2_p_p2_pnr, d2_p_p3_pnr = \
                calculatePNRCarpool(trip1, trip2, sid, reversed=True)
        # fill the matrix for pnr mode
        if fill_mat:  # and sid is not None
            # travel time for the driver during shared time
            self.tt_matrix[int_idx1][int_idx2] = d1_tt_p1_pnr + d1_tt_p2_pnr + d1_tt_p3_pnr
            self.tt_matrix_p2[int_idx1][int_idx2] = d1_tt_p2_pnr
            # total vehicular driving time
            self.ml_matrix[int_idx1][int_idx2] = d1_ml_p1_pnr + d1_ml_p2_pnr + d1_ml_p3_pnr
            # self.ml_pnr_matrix_p[int_idx1][int_idx2] = p2_ml_p0_pnr
            # print(f"ml_matrix[{int_idx1}][{int_idx2}]:{self.ml_matrix[int_idx1][int_idx2]}")
            if fixed_role is False:
                # (0.5 * (p1_tt_p0_pnr + d2_tt_p1_pnr))
                self.tt_matrix[int_idx2][int_idx1] = d2_tt_p1_pnr + d2_tt_p2_pnr + d2_tt_p3_pnr
                self.tt_matrix_p2[int_idx2][int_idx1] = d2_tt_p2_pnr
                self.ml_matrix[int_idx2][int_idx1] = d2_ml_p1_pnr + d2_ml_p2_pnr + d2_ml_p3_pnr
                # self.ml_pnr_matrix_p[int_idx2][int_idx1] = p1_ml_p0_pnr
        # return distances and links of two trips (only for debug/visualization)
        # dists_1, links_1, dists_2, links_2
        if fixed_role is False:
            return ((p2_ml_p0_pnr + d1_ml_p1_pnr + d1_ml_p2_pnr + d1_ml_p3_pnr),
                    (p2_p_p0_pnr, d1_p_p1_pnr, d1_p_p2_pnr, d1_p_p3_pnr),
                    (p1_ml_p0_pnr + d2_ml_p1_pnr + d2_ml_p2_pnr + d2_ml_p3_pnr),
                    (p1_p_p0_pnr, d2_p_p1_pnr, d2_p_p2_pnr, d2_p_p3_pnr), sid)
        return ((p2_ml_p0_pnr + d1_ml_p1_pnr + d1_ml_p2_pnr + d1_ml_p3_pnr),
                (p2_p_p0_pnr, d1_p_p1_pnr, d1_p_p2_pnr, d1_p_p3_pnr), sid)

    def compute_carpoolable_trips_pnr(self, reset_off_diag: bool = False) -> None:
        """
        Instead of computing all combinations, only compute all carpool-able trips.
        :param reset_off_diag: if True, reset all carpool trips information EXCEPT drive alone trips
            if False, only update based on carpool-able matrix information
        :return: None
        """
        nrow, ncol = self.nrow, self.ncol
        if reset_off_diag:  # wipe and reset out all off-diagonal values
            temp_diag_tt = self.tt_matrix.diagonal()
            temp_diag_ml = self.ml_matrix.diagonal()
            # reset travel time information
            self.tt_matrix = np.full((nrow, ncol), np.nan)
            self.ml_matrix = np.full((nrow, ncol), np.nan)
            np.fill_diagonal(self.tt_matrix, temp_diag_tt)  # many empty objects
            np.fill_diagonal(self.ml_matrix, temp_diag_ml)
        indexes_pairs = np.argwhere(self.cp_matrix == 1)
        indexes_pairs = [index for index in indexes_pairs if index[0] != index[1]]
        # print('PNR Indices matching0: \n', indexes_pairs)
        for index in indexes_pairs:
            self.compute_carpool(index[0], index[1], fixed_role=True)

    def compute_in_one_step(
        self, mu1: float = 1.3, mu2: float = 0.1, dst_max: float = 5 * 5280,
        Delta1: float = 15, Delta2: float = 10, Gamma: float = 0.2,  # for depart diff and wait time
        delta: float = 15, gamma: float =1.5, ita: float = 0.5,
        print_mat: bool = True,
    ):
        """
        For the park and ride case, use the same set of filtering parameters for normal case (6 steps).
        :param include_direct:
            if True, consider both PNR and simple carpool at the same time
            if False, consider only PNR trips
        :return:
        """
        # step 1. compute drive alone information
        self.td.compute_sov_info()
        tt_lst, dst_lst = self.td.soloTimes, self.td.soloDists
        self.fill_diagonal(tt_lst, dst_lst)
        self._print_matrix(step=1, print_mat=print_mat)

        # step 2. a set of filter based on euclidean distance between coordinates
        # (driver pass through a PNR station)
        self.compute_01_matrix_to_station_p1(
            threshold_dist=dst_max, mu1=mu1, 
            trips=self.trips, print_mat=print_mat
        )
        self._print_matrix(step=2, print_mat=print_mat)

        # step 3. make sure each SOV trip can travel through PNR
        # Note: filter by the passenger's before after traveling TIME
        self.compute_01_matrix_to_station_p2(delta=15, gamma=1.5)
        self._print_matrix(step=3, print_mat=print_mat)

        # step 4. check departure time difference to filter (for all reasonable pnr stations)
        # this may filter out useful matches for PNR mode
        compute_depart_01_matrix_pre_pnr(self, Delta1=Delta1)
        self._print_matrix(step=4, print_mat=print_mat)

        # step 5. combine all aforementioned filters to generate one big filter
        self.compute_carpoolable_trips_pnr(reset_off_diag=False)
        self._print_matrix(step=5, print_mat=print_mat)

        # step 6. filter by the maximum waiting time for the driver at pickup location
        self.cp_matrix = compute_depart_01_matrix_post_pnr(
            trip_cluster=self, Delta2=Delta2, Gamma=Gamma
        )
        self._print_matrix(step=6, print_mat=print_mat)

        # step 7. filter by real computed waiting time (instead of coordinates before)
        compute_reroute_01_matrix_pnr(
            trip_cluster=self,
            delta=delta, gamma=gamma, ita=ita,
            print_mat=print_mat
        )
        self._print_matrix(step=7, print_mat=print_mat)

        # step 8. solve the carpool conflicts
        num_pair, pairs = self.compute_optimal_bipartite()
        self._print_matrix(step=8, print_mat=print_mat)
        return num_pair, pairs
