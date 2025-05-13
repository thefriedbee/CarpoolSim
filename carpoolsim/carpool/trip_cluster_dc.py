"""
Apply DC carpooling mode to a set of trips.
"""
import numpy as np

from carpoolsim.carpool.trip_demands import TripDemands
from carpoolsim.carpool.trip_cluster_abstract import TripClusterAbstract
from carpoolsim.carpool.util.network_search import (
    get_path_distance_and_tt, 
    dynamic_shortest_path_search
)
from carpoolsim.carpool.util.mat_operations import (
    get_trip_projected_ods,
    get_distances_among_coordinates,
)
from carpoolsim.carpool.util.filters_dc import (
    compute_depart_01_matrix_pre,
    compute_reroute_01_matrix
)
from carpoolsim.config import CPMode, TripClusterConfig


# Direct Carpool Mode
class TripClusterDC(TripClusterAbstract):
    def __init__(self, trip_demands: TripDemands):
        super().__init__(trip_demands)
        self.mode = CPMode.DC

    def compute_carpool(
        self,
        int_idx1: int,
        int_idx2: int,
        fixed_role: bool = False,
        print_dist: bool = False,
    ):
        """
        Given the integer index of two trips (of the trip DataFrame self.df),
        compute distances and paths of 4 different scenarios, that is:
        1. A pickup B; 2. B pickup A;
        Note that park and ride scenarios 3 and 4 are in the function "compute_carpool_pnr"
        :param int_idx1: integer index for the first traveler A
        :param int_idx2: integer index for the second traveler B
        :param print_dist: print trip plans (for debugging)

        :param fixed_role: if True, int_idx1 is for the driver, int_idx2 is for the passenger
            Otherwise, we are trying to call all PERMUTATIONS of carpool with roles,
            (call 1 picks up 2 then call 2 picks up 1, for future conclude pickup location) then set this to True.
            If call on COMBINATIONS, set this to False.
        :return: paths and links for the all scenarios (two for now)
        """
        trips = self.td.trips
        network = self.td.network
        soloDists, soloTimes, soloPaths = self.td.soloDists, self.td.soloTimes, self.td.soloPaths
        trip1, trip2 = trips.iloc[int_idx1, :], trips.iloc[int_idx2, :]
        
        # part 2 (p2) below is the shared trip between passenger & driver
        # d1_tt_p1, d1_tt_p2, d1_tt_p3 = 0, 0, 0  # pickup, duration, drop-off time for driver 1
        # d1_ml_p1, d1_ml_p2, d1_ml_p3 = 0, 0, 0  # pickup, duration, drop-off mileage for driver 1
        # d2_tt_p1, d2_tt_p2, d2_tt_p3 = 0, 0, 0  # pickup, duration, drop-off time for driver 2
        # d2_ml_p1, d2_ml_p2, d2_ml_p3 = 0, 0, 0  # pickup, duration, drop-off mileage for driver 2

        # a helper function to calculate carpool speed
        def calculateCarpool(trip1, trip2, t1_idx, t2_idx, reversed=False):
            """
            Calculate the shortest carpool travel path.
            Similar to the self.compute_diagonal function, use trick to fast compute
            by setting values dynamically.
            :param trip1: the driver's trip info
            :param trip2: the passenger's trip info
            :param t1_idx: the index of trip 1
            :param t2_idx: the index of trip 2
            :param reversed: if False, trip1 is the driver. Otherwise, trip2 is the driver.
            """
            O1, D1, O2, D2 = trip1['o_node'], trip1['d_node'], trip2['o_node'], trip2['d_node']
            O1_taz, D1_taz, O2_taz, D2_taz = trip1['orig_taz'], trip1['dest_taz'], trip2['orig_taz'], trip2['dest_taz']
            if not reversed:  # O1 ==> O2 ==> D2 ==> D1
                p1, t1, d1 = dynamic_shortest_path_search(network, O1, O2, O1_taz, O2_taz)  # O1->O2
                # O2->D2, which is already computed (self.compute_diagonal should be called before)
                d2, p2 = soloDists[t2_idx], soloPaths[t2_idx]
                t2, __ = get_path_distance_and_tt(network, p2)
                p3, t3, d3 = dynamic_shortest_path_search(network, D2, D1, D2_taz, D1_taz)  # D2->D1
            else:  # O2 ==> O1 ==> D1 ==> D2
                p1, t1, d1 = dynamic_shortest_path_search(network, O2, O1, O2_taz, O1_taz)  # O2->O1
                # O1->D1, which is already computed (self.compute_diagonal should be called before)
                d2, p2 = soloDists[t1_idx], soloPaths[t1_idx]
                t2, __ = get_path_distance_and_tt(network, p2)
                p3, t3, d3 = dynamic_shortest_path_search(network, D1, D2, D1_taz, D2_taz)  # D1->D2
            if print_dist:
                print('d1: {}; d2: {}; d3: {}'.format(d1, d2, d3))
            return t1, t2, t3, d1, d2, d3, p1, p2, p3

        # scheme 1. A pickup B. Trip paths is O1 ==> O2 ==> D2 ==> D1
        # "d1_tt_p1" means: driver 1, travel time, part 1
        d1_tt_p1, d1_tt_p2, d1_tt_p3, d1_ml_p1, d1_ml_p2, d1_ml_p3, d1_p_p1, d1_p_p2, d1_p_p3 = \
            calculateCarpool(trip1, trip2, int_idx1, int_idx2, reversed=False)
        # scheme 2. B pickup A. Trip paths is O2 ==> O1 ==> D1 ==> D2
        if fixed_role is False:
            d2_tt_p1, d2_tt_p2, d2_tt_p3, d2_ml_p1, d2_ml_p2, d2_ml_p3, d2_p_p1, d2_p_p2, d2_p_p3 = \
                calculateCarpool(trip1, trip2, int_idx1, int_idx2, reversed=True)
        # let's fill the matrix, store vehicular hours of a trip
        self.tt_matrix[int_idx1][int_idx2] = d1_tt_p1 + d1_tt_p2 + d1_tt_p3
        self.tt_matrix_p1[int_idx1][int_idx2] = d1_tt_p1
        self.tt_matrix_p3[int_idx1][int_idx2] = d1_tt_p3
        self.ml_matrix[int_idx1][int_idx2] = d1_ml_p1 + d1_ml_p2 + d1_ml_p3
        # print(f"ml_matrix[{int_idx1}][{int_idx2}]:{self.ml_matrix[int_idx1][int_idx2]}")
        if fixed_role is False:
            self.tt_matrix[int_idx2][int_idx1] = d2_tt_p1 + d2_tt_p2 + d2_tt_p3
            self.tt_matrix_p1[int_idx2][int_idx1] = d2_tt_p1
            self.tt_matrix_p3[int_idx2][int_idx1] = d2_tt_p3
            self.ml_matrix[int_idx2][int_idx1] = d2_ml_p1 + d2_ml_p2 + d2_ml_p3
            return (d1_ml_p1 + d1_ml_p2 + d1_ml_p3), (d1_p_p1[:-1] + d1_p_p2[:-1] + d1_p_p3), \
                   (d2_ml_p1 + d2_ml_p2 + d2_ml_p3), (d2_p_p1[:-1] + d2_p_p2[:-1] + d2_p_p3)
        return (d1_ml_p1 + d1_ml_p2 + d1_ml_p3), (d1_p_p1[:-1] + d1_p_p2[:-1] + d1_p_p3)

    def compute_depart_01_matrix_post(
        self,
        Delta2: float = 10,
        Gamma: float = 0.2,
        default_rule: bool = True,
    ) -> None:
        """
        After tt_matrix_p1 is computed, filter by maximum waiting time for the driver at pickup location
        :param Delta2: driver's maximum waiting time
        :param default_rule: if True, strict time different; if False, absolute time difference
        :return:
        """
        # step 2. Maximum waiting time for driver is Delta2 (default is 5 minutes)
        nrow, ncol = self.shape
        trips = self.td.trips
        soloTimes = self.td.soloTimes
        tt_matrix_p1 = self.tt_matrix_p1
        cp_matrix = self.cp_matrix
        # driver_lst = np.array(self.trips_front['new_min'].tolist()).reshape((1, -1))  # depart minute
        # for non-simulation with time case, passenger <==> driver have the same scope
        passenger_lst = np.array(trips['new_min'].tolist()).reshape((1, -1))
        # compare departure time difference
        dri_arr = np.tile(passenger_lst.reshape((-1, 1)), (1, ncol)) + tt_matrix_p1
        pax_dep = np.tile(passenger_lst.reshape((1, -1)), (nrow, 1))  # depart time difference
        # step 2. Maximum waiting time for driver is Delta2 (default is 10 minutes)
        wait_time_mat = dri_arr - pax_dep  # wait time matrix for driver
        # for post analysis, directly update final cp_matrix
        passenger_time = np.array([soloTimes[i] for i in range(ncol)]).reshape(1, -1)
        passenger_time = np.tile(passenger_time, (nrow, 1))

        cp_matrix = (cp_matrix &
                     (np.absolute(wait_time_mat) <= Delta2) &
                     (np.absolute(wait_time_mat/passenger_time) <= Gamma)).astype(np.bool_)
        if default_rule:
            # passenger only waits the driver should wait at most Delta2 minutes
            cp_matrix = (cp_matrix & (wait_time_mat >= 0)).astype(np.bool_)
        self.cp_matrix = cp_matrix

    def compute_pickup_01_matrix(
        self,
        threshold_dist: float = 5280 * 5,
        mu1: float = 1.5,
        mu2: float = 0.1,
        use_mu2: bool = True,
    ):
        """
        Compute feasibility matrix based on whether one can pick up/ drop off passengers.
        If A can pick up B in threshold, then it is feasible. Otherwise, it is not a feasible carpool.
        Use Euclidean Distance.

        :param threshold_dist: the distance between pickups are within the distance in miles (default is 5 mile)
        :param mu1: the maximum ratio between carpool vector distance and SOV vector distance. Vector distance is
        the length connecting origin/destination coordinates.

        :param mu2: the maximum ratio of backward traveling distance after drop off passengers.
        :param use_mu2: If True, measure backward traveling distance.
        V_O1D1 defined as SOV trip vector
        V_D2D1 defined the last portion of carpool trip vector (vector connecting passenger's dest. to driver's origin)
        the backward ratio is defined as:  - (V_O1D1 * V_D2D1) / (V_D2D1 * V_D2D1).
        The filter holds for all (i,j) pairs with: - (V_O1D1 * V_D2D1) / (V_D2D1 * V_D2D1) < mu2
        :return:
        """
        nrow, ncol = self.shape
        trips = self.td.trips
        oxs, oys, dxs, dys = get_trip_projected_ods(trips)
        mat_ox, mat_oy, man_o = get_distances_among_coordinates(oxs, oys)
        mat_dx, mat_dy, man_d = get_distances_among_coordinates(dxs, dys)

        # compute euclidean travel distance
        mat_diag = np.sqrt((oxs - dxs) ** 2 + (oys - dys) ** 2)
        # compute reroute straight distance, then origin SOV distance for the driver
        rr = (man_o + man_d + np.tile(mat_diag, (nrow, 1)))
        ori = np.tile(mat_diag, (nrow, 1))
        mat_ratio = rr / ori

        # 1. coordinate distance; 2. coordinate distance ratio
        self.cp_matrix = (self.cp_matrix &
                          (man_o <= threshold_dist) &
                          (mat_ratio < mu1)).astype(bool)

        if use_mu2:
            # now it is time for implementing backward constraint
            # compute the vector for all drivers V_{O1D1}
            mat_x_o1d1 = (dxs - oxs).reshape((1, -1))
            mat_y_o1d1 = (dys - oys).reshape((1, -1))
            # compute vector V_{D2D1} from passenger's destination to driver's destination
            mat_x_d2d1 = np.tile(dxs.transpose(), (1, ncol))
            mat_x_d2d1 = np.abs(mat_x_d2d1 - np.tile(dxs, (nrow, 1)))
            mat_y_d2d1 = np.tile(dys.transpose(), (1, ncol))
            mat_y_d2d1 = np.abs(mat_y_d2d1 - np.tile(dys, (nrow, 1)))
            # compute vector angle for each composition position
            part1 = -(np.tile(mat_x_o1d1.transpose(), (1, ncol)) * mat_x_d2d1 +
                      np.tile(mat_y_o1d1.transpose(), (1, ncol)) * mat_y_d2d1)
            part2 = (np.tile(mat_x_o1d1.transpose() ** 2, (1, ncol)) +
                     np.tile(mat_y_o1d1.transpose() ** 2, (1, ncol)))
            backward_index = part1 / part2

            np.fill_diagonal(backward_index, -1)
            self.cp_matrix = (self.cp_matrix &
                              (backward_index <= mu2)).astype(bool)   

    def evaluate_carpoolable_trips(self, reset_off_diag: bool = False) -> None:
        """
        Instead of computing all combinations, only compute all carpool-able trips.
        :param reset_off_diag: 
            if True, reset all carpool trips information EXCEPT drive alone trips
            if False, only update based on carpool-able matrix information
        :return: None
        """
        nrow, ncol = self.shape
        if reset_off_diag:  # wipe and reset out all off-diagonal values
            temp_diag_tt = self.tt_matrix.diagonal()
            temp_diag_ml = self.ml_matrix.diagonal()
            # reset travel time/mileage information
            self.tt_matrix = np.full((nrow, ncol), np.nan)
            self.ml_matrix = np.full((nrow, ncol), np.nan)
            np.fill_diagonal(self.tt_matrix, temp_diag_tt)
            np.fill_diagonal(self.ml_matrix, temp_diag_ml)
        # print(self.cp_matrix[:5, :5])
        # indexes = np.where(self.cp_matrix == 1)
        indexes_pairs = np.argwhere(self.cp_matrix == 1)
        # print('Indices matching: \n', [index for index in indexes_pairs if index[0]!=index[1]])
        for index in indexes_pairs:
            self.compute_carpool(index[0], index[1], fixed_role=True)

    def compute_in_one_step(
        self, config: TripClusterConfig,
        # mu1: float = 1.5, mu2: float = 0.1, dst_max: float = 5 * 5280,
        # Delta1: float = 15, Delta2: float = 10, Gamma: float = 0.2,  # for depart diff and wait time
        # delta: float = 15, gamma: float = 1.5, ita: float = 0.5,
        # print_mat: bool = False, run_solver: bool = True,
    ) -> tuple[int, list[tuple[int, int]]]:
        # step 1. compute drive alone info
        self.td.compute_sov_info()
        tt_lst, dst_lst = self.td.soloTimes, self.td.soloDists
        self.fill_diagonal(tt_lst, dst_lst)
        self._print_matrix(step=1, print_mat=config.print_mat)

        # step 2. check departure time difference to filter
        compute_depart_01_matrix_pre(self, Delta1=config.Delta1)
        self._print_matrix(step=2, print_mat=config.print_mat)

        # step 3. a set of filter based on Euclidean distance between coordinates
        self.compute_pickup_01_matrix(threshold_dist=config.dist_max, mu1=config.mu1, mu2=config.mu2)
        self._print_matrix(step=3, print_mat=config.print_mat)

        # step 4. evaluate information using the actual traveling paths
        self.evaluate_carpoolable_trips(reset_off_diag=False)
        self._print_matrix(step=4, print_mat=config.print_mat)

        # step 5. filter by the maximum waiting time for the driver at pickup location
        self.compute_depart_01_matrix_post(Delta2=config.Delta2, Gamma=config.Gamma)
        self._print_matrix(step=5, print_mat=config.print_mat)

        # step 6. filter by real computed waiting time (instead of coordinates before)
        self = compute_reroute_01_matrix(self, delta=config.delta, gamma=config.gamma, ita=config.ita)
        self._print_matrix(step=6, print_mat=config.print_mat)

        # step 7. compute optimal bipartite pairing
        if config.run_solver:
            num_pair, pairs = self.compute_optimal_bipartite()
            self._print_matrix(step=7, print_mat=config.print_mat)
            self.num_paired = num_pair
            self.paired_lst = pairs
            return num_pair, pairs
        return None, None
