import os
import time
from datetime import datetime
import itertools

import numpy as np
import pandas as pd
import networkx as nx
from gurobipy import Model, GRB, quicksum
import sqlalchemy

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as cx

import carpoolsim.carpool_solver.bipartite_solver as tg
from carpoolsim.visualization.carpool_viz_seq import plot_seq
from carpoolsim.database.query_database import query_od_info

plt.rcParams.update({'font.size': 22})
np.set_printoptions(precision=3)


# TripHolder is depreciated. Use this class for all kinds of carpool computation tasks
class TripCluster:
    def __init__(
            self,
            df: pd.DataFrame,
            network: nx.DiGraph,
            links: pd.DataFrame,
            engine: sqlalchemy.Engine,
            parking_lots: pd.DataFrame | None = None,
            update_now_str: bool = True
    ) -> None:
        """
        This TripCluster class tries to substitute the old TripHolder class.
        It is more efficient in shortest path computation by checking cached data.
        :param df: corresponds to all trips within a time space cluster,
        df is raw trip data and it needs to be processed later to fit CarpoolSim's format.
        Some important columns are:
        'hh_id', 'person_id', 'orig_taz', 'dest_taz',
        'orig_inner_id', 'dest_inner_id',
        'orig_lon', 'orig_lat', 'dest_lon', 'dest_lat'

        Note: expected columns for CarpoolSim are:
            ox, oy, o_t, o_node, ox_sq, oy_sq, o_d
            dx, dy, d_t, d_node, dx_sq, dy_sq, d_d
        :param network: a directed graph on package networkx, travel network to be computed on
        :param engine: store database connection (should provide multiple options)
        :param parking_lots: a DataFrame contains the locations of all parking lots
        """
        if update_now_str:
            self.now_str = datetime.now().strftime("%y%m%d_%H%M%S")
        # traffic mode
        self.mode = -1  # -1 denote computation mode not set
        self.priority_mode = 0
        # drop useless columns to save memory in mass computation
        df = df.drop(['ox_sq', 'oy_sq', 'dx_sq', 'dy_sq',
                      'depart_period', 'hh_id', 'newarr',
                      'orig_inner_id', 'dest_inner_id'], axis=1, errors='ignore')
        self.trips = df.copy()
        self.trips['pnr'] = None  # store accessible pnr station
        # for simulation with time, this will also update as needed
        self.int2idx = {i: index for i, (index, row) in enumerate(df.iterrows())}
        self.idx2int = {index: i for i, (index, row) in enumerate(df.iterrows())}
        count = len(df)
        # this should be different for inherited class
        self.nrow, self.ncol = count, count
        self.network = network  # directed network
        self.links = links.to_crs(epsg=3857)
        # convert links to Web M. format for better visualization
        self.crs = self.links.crs  # store crs information, this project uses 'epsg:2240'
        self.engine = engine
        self.parking_lots = parking_lots  # dataframe of parking lots

        self.soloPaths = {}  # trip retention of SOV paths
        self.soloDists = {}  # trip retention of SOV distance
        self.soloTimes = {}  # trip retention of SOV time
        # TODO: accessibility from start to parking lot to destination
        if parking_lots is not None:
            self.pnr_ncol = len(self.parking_lots)
            self.pnr_01_matrix = np.full((count, len(self.parking_lots)), 1).astype(np.bool_)
            # store all info in one matrix of objects
            self.pnr_access_info = np.empty((count, len(self.parking_lots)), dtype=object)
            # 0-1 matrix for PNR travelers, start with all available and shrink by filters
            self.cp_pnr_matrix = np.full((count, count), 1, dtype=np.bool_)
            # travel time: "averaged travel time" for all
            self.tt_pnr_matrix = np.full((count, count), np.nan, dtype="float32")
            self.tt_pnr_matrix_shared = np.full((count, count), np.nan, dtype="float32")
            # mileage: use total vehicular mileage for all
            self.ml_pnr_matrix = np.full((count, count), np.nan, dtype="float32")
            # another matrix to record the PNR passengers drive
            # self.ml_pnr_matrix_p = np.full((count, count), np.nan, dtype="float32")
        # add more matrices for specific pickup/drop-off information
        # average travel time matrix for driver and passenger
        self.tt_matrix = np.full((count, count), np.nan, dtype="float32")
        self.tt_matrix_p1 = np.full((count, count), np.nan, dtype="float32")  # pickup travel time for driver
        self.tt_matrix_p3 = np.full((count, count), np.nan, dtype="float32")  # drop-off travel time for driver
        self.ml_matrix = np.full((count, count), np.nan, dtype="float32")  # store travel distance (miles) for driver

        # choice matrix applied when considering multiple modes
        self.choice_matrix = np.full((count, count), np.nan, dtype="int8")
        self.cp_matrix_all = np.full((count, count), np.nan, dtype=np.bool_)
        self.tt_matrix_all = np.full((count, count), np.nan, dtype="float32")
        self.ml_matrix_all = np.full((count, count), np.nan, dtype="float32")
        # carpool matrix: the 0-1 accessibility matrix
        self.cp_matrix = np.ones((count, count), dtype=np.bool_)  # overall carpool-able matrix
        # store matching results
        self.result_lst = None  # store the integer pairs of the optimized results (LP method)
        self.result_lst_bipartite = None  # store optimized results of bipartite method
        self.trip_summary_df = None  # a DataFrame stores before after numbers for summarized statistics for the cluster

    # connect to database and quickly check OD shortest path
    def query_od_info(self, o_taz: str | int, d_taz: str | int):
        """
        # returned: origin ID, destination ID, distance, path used
        :param o_taz: origin TAZ
        :param d_taz: destination TAZ
        :return:
        """
        results = query_od_info(engine=self.engine, o_taz=o_taz, d_taz=d_taz)
        row_dist = results[2]
        row_path = results[3]
        return str(o_taz), str(d_taz), row_dist, row_path

    def get_path_distance_and_tt(self, nodes: list[str]):
        """
        A helper function to get driving path given a set of network nodes.
        :param nodes: a list of traveling network node id
        :return: travel time and mileage distance of the path
        """
        tt, dst = 0, 0
        for i in range(len(nodes) - 1):
            tt += self.network[nodes[i]][nodes[i + 1]]['forward']
            dst += self.network[nodes[i]][nodes[i + 1]]['dist']
        return tt, dst

    def compute_diagonal(self):
        """
        Compute drive alone (SOV) trips for all travel demands in the cluster,
        For each SOV trip, check their OD coordinates and TAZ ID.

        The fast compute strategy:
        1. query the travel path between OD TAZ centroids, store this path in this trip cluster class object
        2. for the directed network, set the cost of all links used by this path as zeroes.
        3. Run dijkstra's algorithm again using specific start-end point (queried path is prioritized with zero cost)
        4. The new cost we get is the shortest paths between od node,
         given that a path must use at least one path segment of the queried path!

        Since copy a graph is expensive, we just modify travel time values and
        restore default settings after we are done.
        :return: None
        """
        nrow, ncol = self.nrow, self.ncol
        item_size = max(nrow, ncol)
        trips = self.trips.iloc[:item_size, :]

        integer_idx = 0  # use integer index for SOV trips
        tt_lst, dst_lst = [], []
        for idx, trip in trips.iterrows():
            # step 0. prepare OD nodes and OD TAZs
            start_node, end_node = trip['o_node'], trip['d_node']
            start_taz, end_taz = trip['orig_taz'], trip['dest_taz']
            # step 2-4. for diagonal, use slower but more accurate distance searcher
            # pth_nodes, tt, dst = self._dynamic_shortest_path_search(start_node, end_node, start_taz, end_taz)
            pth_nodes, tt, dst = self._naive_shortest_path_search(start_node, end_node)
            # store solo paths information
            self.soloPaths[integer_idx] = pth_nodes
            # step 5. store solo travel results
            self.soloTimes[integer_idx] = tt
            self.soloDists[integer_idx] = dst
            tt_lst.append(tt)
            dst_lst.append(dst)
            integer_idx += 1
        # update diagonal cp matrix
        np.fill_diagonal(self.cp_matrix, 1)
        # update tt matrix and ml matrix
        np.fill_diagonal(self.tt_matrix, tt_lst)
        np.fill_diagonal(self.ml_matrix, dst_lst)
        if self.parking_lots is not None:
            np.fill_diagonal(self.cp_pnr_matrix, 1)
            # update tt matrix and ml matrix
            np.fill_diagonal(self.tt_pnr_matrix, tt_lst)
            np.fill_diagonal(self.ml_pnr_matrix, dst_lst)

    def _dynamic_shortest_path_search(self, start_node, end_node, start_taz, end_taz):
        # step 1. get OD and query the shortest path between OD TAZ centroids.
        __, __, row_dist, nodes = self.query_od_info(start_taz, end_taz)
        # step 2. store graph distances, reset them to zeros for fast compute
        orig_distances = []
        for i in range(len(nodes) - 1):
            orig_distances.append(self.network[nodes[i]][nodes[i + 1]]['forward'])
            self.network[nodes[i]][nodes[i + 1]]['forward'] = 0.0001
        # step 3.1 Run dijkstra's algorithm to recompute the shortest paths
        __, pth_nodes = nx.single_source_dijkstra(self.network, str(start_node), str(end_node), weight='forward')
        # step 3.2 Restore all graph weights
        for i, d in enumerate(orig_distances):
            self.network[nodes[i]][nodes[i + 1]]['forward'] = d
        # step 4. Compute travel distance of the new path
        tt, dst = self.get_path_distance_and_tt(pth_nodes)
        return pth_nodes, tt, dst

    def _naive_shortest_path_search(self, start_node, end_node):
        # step 1. get OD and query the shortest path between OD TAZ centroids.
        __, pth_nodes = nx.single_source_dijkstra(self.network, str(start_node), str(end_node), weight='forward')
        tt, dst = self.get_path_distance_and_tt(pth_nodes)
        return pth_nodes, tt, dst

    def generate_pnr_trip_map_filt(
            self,
            delta: float = 15,
            gamma: float = 1.5,
            trips: pd.DataFrame | None = None,
    ):
        """
        :param delta: maximum reroute time (in minutes) acceptable for the driver
        :param gamma: the maximum ratio of extra travel time over driver's original travel time

        1. For each accessible PNR for a driver, compute their (arrival time, travel time to PNR)
        2. Store the dictionary of <station id --> (arrival time, travel time to PNR)> to pnr trips matrix
        :return:
        """
        if trips is None:
            trips = self.trips
        # return all indicies with value 1
        ind = np.argwhere(self.pnr_01_matrix == 1)
        # print('pnr trip map filt:', ind)
        for ind_one in ind:
            trip_id, station_id = ind_one
            # print(trip_id, station_id)
            self.compute_pnr_access(trip_id, station_id)
            __, t1, __, t_all = self.pnr_access_info[trip_id, station_id]
            # get travel alone time
            t2 = self.soloTimes[trip_id]
            # if not possible to access PNR station in time,
            # change access pnr matrix information
            if (t_all - t2) >= delta or t_all / (t2+0.1) >= gamma:
                self.pnr_01_matrix[trip_id, station_id] = 0
                # self.pnr_access_info[trip_id, station_id] = None
                continue
            # store station info to the list (could be multiple accessible PNR stations)
            if trips['pnr'].iloc[trip_id] is None:
                trips['pnr'].iloc[trip_id] = [station_id]
            else:
                trips['pnr'].iloc[trip_id].append(station_id)

        # print new filtered results
        # finally, prepare the big 0-1 matrix between travelers
        temp_cp_pnr_matrix = ((self.pnr_01_matrix @ self.pnr_01_matrix.T) > 0).astype(np.bool_)
        n_rows = self.cp_pnr_matrix.shape[0]
        self.cp_pnr_matrix = (self.cp_pnr_matrix &
                              temp_cp_pnr_matrix[:n_rows, :]).astype(np.bool_)

    def compute_pnr_access(self, trip_id: int, station_id: int) -> None:
        """
        Compute access time to PNR station
        :return:
        """
        # load dataframe series
        trip_row = self.trips.iloc[trip_id, :]
        pnr_row = self.parking_lots.iloc[station_id, :]

        def calculateAccess(trip, pnr):
            O1, O1_taz = trip['o_node'], trip['orig_taz']
            O2, O2_taz = pnr['node'], pnr['taz']
            D1, D1_taz = trip['d_node'], trip['dest_taz']
            p1, t1, d1 = self._dynamic_shortest_path_search(O1, O2, O1_taz, O2_taz)
            p2, t2, d2 = self._dynamic_shortest_path_search(O2, D1, O2_taz, D1_taz)
            # return access info, and travel time goes through PNR station
            return p1, t1, d1, t1+t2
        # return travel time, distance, and network nodes
        p1, t1, d1, t_all = calculateAccess(trip_row, pnr_row)
        # print("strange...")
        # print(trip_id, station_id)
        # print(p1, t1, d1, t_all)
        self.pnr_access_info[trip_id, station_id] = [p1, t1, d1, t_all]

    def _check_trips_best_pnr(self, trip_row1, trip_row2, int_idx1, int_idx2):
        """
        Just check if two trips can share one PNR station.
        If so, among the stations, choose the best station for joint trip
        :param trip_row1: Pandas Series for trip 1
        :param trip_row2: Pandas Series for trip 2
        :return: a list of PNR stations feasible by both trips
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
            mu2: float = 0.3,
            use_mu2: bool =True,
            trips: pd.DataFrame | None =None,
            print_mat: bool = True,
    ):
        """
        For each trip, compute the parking lots that can be used as the "meetings point" for carpooled trip.
        part 1. Use Euclidean distance between coordinates
        part 2. check precise travel time & distance passing stations (implemented in another function)

        Assume trip starting from O1 and ending at D1, consider stop at M1 for a midterm stop
        1. O1-->M1 is within 5 miles
        2. (O1 M1) + (M1 D1) < p1 * (O1 D1)
        3. (O1 D1) / [(O1 M1) + (M1 D1)] < p2 * (O1 D1)
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
        # print(f"nrow: {nrow}; nrol: {ncol}; lots_ncol: {lots_ncol}")
        oxs = np.array(trips.ox.tolist()).reshape((1, -1))
        oys = np.array(trips.oy.tolist()).reshape((1, -1))
        dxs = np.array(trips.dx.tolist()).reshape((1, -1))
        dys = np.array(trips.dy.tolist()).reshape((1, -1))

        mxs = np.array(self.parking_lots.x.tolist()).reshape((1, -1))
        mys = np.array(self.parking_lots.y.tolist()).reshape((1, -1))

        # origin distance in (x, y) axis between trip origin and parking lots (vectorized)
        mat_om_x = np.tile(oxs.transpose(), (1, lots_ncol))
        mat_om_x = np.abs(mat_om_x - np.tile(mxs, (num_all, 1)))
        mat_om_y = np.tile(oys.transpose(), (1, lots_ncol))
        mat_om_y = np.abs(mat_om_y - np.tile(mys, (num_all, 1)))
        # destination distance in (x, y) axis between trip destination and parking lots (vectorized)
        mat_md_x = np.tile(dxs.transpose(), (1, lots_ncol))
        mat_md_x = np.abs(np.tile(mxs, (num_all, 1)) - mat_md_x)
        mat_md_y = np.tile(dys.transpose(), (1, lots_ncol))
        mat_md_y = np.abs(np.tile(mys, (num_all, 1)) - mat_md_y)
        # original distance from o to d for each trip
        mat_od_x = dxs - oxs
        mat_od_y = dys - oys

        man_od = np.sqrt(mat_od_x ** 2 + mat_od_y ** 2).astype("float32")
        man_om = np.sqrt(mat_om_x ** 2 + mat_om_y ** 2).astype("float32")
        man_md = np.sqrt(mat_md_x ** 2 + mat_md_y ** 2).astype("float32")
        # compute reroute distance using pnr facility
        post_dist = (man_om + man_md)
        pre_dist = man_od.reshape(-1, 1)
        mat_ratio = (post_dist / pre_dist).astype("float32")
        if print_mat:
            # print("shapes:", self.pnr_01_matrix.shape, mat_ratio.shape, man_om.shape)
            print("mat_ratio total pass is:", (mat_ratio < mu1).sum())
            # print(mat_ratio[:8, :8])
            print("man_om total pass is:", (man_om <= threshold_dist).sum())
            # print(man_om[:8, :8])
            # row_idx = np.argmax(mat_ratio, axis=1)
            self.pnr_01_matrix = (self.pnr_01_matrix &
                                  (mat_ratio < mu1) &
                                  (man_om <= threshold_dist))
            print("pnr 0-1 matrix total pass is: (after basic euclidean filters)", self.pnr_01_matrix.sum())
            # print(self.pnr_01_matrix[:8, :8])
        if use_mu2:
            # now it is time for implementing backward constraint
            # compute the vector for all drivers V_{O1D1}
            part1 = -(mat_om_x * mat_md_x) + (mat_om_y * mat_md_y)
            part2 = (mat_om_x * mat_md_x) + (mat_om_y * mat_om_y)
            backward_index = part1 / part2
            # print(self.pnr_01_matrix.shape)
            self.pnr_01_matrix = (self.pnr_01_matrix &
                                  (backward_index <= mu2)).astype(np.bool_)
            if print_mat:
                print("pnr 0-1 matrix total pass is (after mu2 filter):", self.pnr_01_matrix.sum())
                # print(self.pnr_01_matrix[:8, :8])

    def compute_01_matrix_to_station_p2(
            self,
            delta: float = 15,
            gamma: float = 1.5,
    ) -> None:
        """
        Make sure each SOV trip can go through PNR station without too much extra costs
        :param delta: maximum reroute time (in minutes) acceptable for the driver
        :param gamma: the maximum ratio of extra travel time over driver's original travel time
        :return:
        """
        # just compute travel information through PNR for all trips
        # meanwhile, for each trip, filter by delta and gamma condition
        # update matrix self.pnr_access_info
        self.generate_pnr_trip_map_filt(delta=delta, gamma=gamma)

    def compute_carpool(
            self,
            int_idx1: int,
            int_idx2: int,
            print_dist: bool = False,
            fill_mat: bool = True,
            fixed_role: bool = False,
    ):
        """
        Given the integer index of two trips (of the trip DataFrame self.df),
        compute distances and paths of 4 different scenarios, that is:
        1. A pickup B; 2. B pickup A;
        Note that park and ride scenarios 3 and 4 are in the function "compute_carpool_pnr"
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
        trip1, trip2 = self.trips.iloc[int_idx1, :], self.trips.iloc[int_idx2, :]
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
                p1, t1, d1 = self._dynamic_shortest_path_search(O1, O2, O1_taz, O2_taz)  # O1->O2
                # O2->D2, which is already computed (self.compute_diagonal should be called before)
                d2, p2 = self.soloDists[t2_idx], self.soloPaths[t2_idx]
                t2, __ = self.get_path_distance_and_tt(p2)
                p3, t3, d3 = self._dynamic_shortest_path_search(D2, D1, D2_taz, D1_taz)  # D2->D1
            else:  # O2 ==> O1 ==> D1 ==> D2
                p1, t1, d1 = self._dynamic_shortest_path_search(O2, O1, O2_taz, O1_taz)  # O2->O1
                # O1->D1, which is already computed (self.compute_diagonal should be called before)
                d2, p2 = self.soloDists[t1_idx], self.soloPaths[t1_idx]
                t2, __ = self.get_path_distance_and_tt(p2)
                p3, t3, d3 = self._dynamic_shortest_path_search(D1, D2, D1_taz, D2_taz)  # D1->D2
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
        if fill_mat:
            self.tt_matrix[int_idx1][int_idx2] = d1_tt_p1 + d1_tt_p2 + d1_tt_p3
            self.tt_matrix_p1[int_idx1][int_idx2] = d1_tt_p1
            self.tt_matrix_p3[int_idx1][int_idx2] = d1_tt_p3
            self.ml_matrix[int_idx1][int_idx2] = d1_ml_p1 + d1_ml_p2 + d1_ml_p3
            # print(f"ml_matrix[{int_idx1}][{int_idx2}]:{self.ml_matrix[int_idx1][int_idx2]}")
            if not fixed_role:
                self.tt_matrix[int_idx2][int_idx1] = d2_tt_p1 + d2_tt_p2 + d2_tt_p3
                self.tt_matrix_p1[int_idx2][int_idx1] = d2_tt_p1
                self.tt_matrix_p3[int_idx2][int_idx1] = d2_tt_p3
                self.ml_matrix[int_idx2][int_idx1] = d2_ml_p1 + d2_ml_p2 + d2_ml_p3
                # print(f"ml_matrix[idx2][idx1]:{self.ml_matrix[int_idx2][int_idx1]}")

        # dists_1, links_1, dists_2, links_2
        if not fixed_role:
            return (d1_ml_p1 + d1_ml_p2 + d1_ml_p3), (d1_p_p1[:-1] + d1_p_p2[:-1] + d1_p_p3), \
                   (d2_ml_p1 + d2_ml_p2 + d2_ml_p3), (d2_p_p1[:-1] + d2_p_p2[:-1] + d2_p_p3)
        return (d1_ml_p1 + d1_ml_p2 + d1_ml_p3), (d1_p_p1[:-1] + d1_p_p2[:-1] + d1_p_p3)

    def compute_carpool_pnr(
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
            O1, D1, O2, D2 = trip1['o_node'], trip1['d_node'], trip2['o_node'], trip2['d_node']
            O1_taz, D1_taz, O2_taz, D2_taz = trip1['orig_taz'], trip1['dest_taz'], trip2['orig_taz'], trip2['dest_taz']
            pnr_row = self.parking_lots.iloc[station_id, :]
            M1, M_taz = pnr_row['node'], pnr_row['taz']
            # print("trip_id: {}, station_id: {}".format(int_idx1, station_id))
            if not reversed:  # O1 ==> M1 ==> D2 ==> D1
                p0, t0, d0, t_all0 = self.pnr_access_info[int_idx2, station_id]  # O2 ==> M1
                p1, t1, d1, t_all1 = self.pnr_access_info[int_idx1, station_id]  # O1 ==> M1
                p2, t2, d2 = self._dynamic_shortest_path_search(M1, D2, M_taz, D2_taz)  # M1 ==> D2
                p3, t3, d3 = self._dynamic_shortest_path_search(D2, D1, D2_taz, D1_taz)  # D2 ==> D1
            else:  # O2 ==> M1 ==> D1 == D2
                p0, t0, d0, t_all0 = self.pnr_access_info[int_idx1, station_id]  # O1 ==> M1
                p1, t1, d1, t_all1 = self.pnr_access_info[int_idx2, station_id]  # O2 ==> M1
                p2, t2, d2 = self._dynamic_shortest_path_search(M1, D1, M_taz, D1_taz)  # M1 ==> D1
                p3, t3, d3 = self._dynamic_shortest_path_search(D1, D2, D2_taz, D2_taz)  # D1 ==> D2
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
            self.tt_pnr_matrix[int_idx1][int_idx2] = d1_tt_p1_pnr + d1_tt_p2_pnr + d1_tt_p3_pnr
            self.tt_pnr_matrix_shared[int_idx1][int_idx2] = d1_tt_p2_pnr
            # total vehicular driving time
            self.ml_pnr_matrix[int_idx1][int_idx2] = d1_ml_p1_pnr + d1_ml_p2_pnr + d1_ml_p3_pnr
            # self.ml_pnr_matrix_p[int_idx1][int_idx2] = p2_ml_p0_pnr
            # print(f"ml_pnr_matrix[{int_idx1}][{int_idx2}]:{self.ml_pnr_matrix[int_idx1][int_idx2]}")
            if fixed_role is False:
                # (0.5 * (p1_tt_p0_pnr + d2_tt_p1_pnr))
                self.tt_pnr_matrix[int_idx2][int_idx1] = d2_tt_p1_pnr + d2_tt_p2_pnr + d2_tt_p3_pnr
                self.tt_pnr_matrix_shared[int_idx2][int_idx1] = d2_tt_p2_pnr
                self.ml_pnr_matrix[int_idx2][int_idx1] = d2_ml_p1_pnr + d2_ml_p2_pnr + d2_ml_p3_pnr
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

    def iterate_all_carpools(self):
        """
        Iterate all carpools to fill up the travel time matrix.
        This function is dumb and depreciated by more pin-point algorithm to compute
        only for the feasible pairs corresponding to the one value of the 0-1 carpool-able matrix
        :return: No return
        """
        t0 = time.time()
        nrow, ncol = self.nrow, self.ncol
        combs = itertools.product(list(range(nrow)), list(range(ncol)))
        # for each combs, call function computeCarpool
        for c in combs:
            __ = self.compute_carpool(c[0], c[1])
            __ = self.compute_carpool_pnr(c[0], c[1])  # compute carpool
        print_str = 'All carpool scenarios computed. Number of trips in this cluster is {}.' + \
                    ' Time spend is {} seconds'
        print(print_str.format(max(nrow, ncol), time.time() - t0))

    def compute_depart_01_matrix_pre_pnr(
            self,
            Delta1: float = 15,
            default_rule: bool = False,
    ):
        """
        For park and ride, the requirements are:
            1. both travelers can choose to use one parking facilities
            2. the arrival time at the station is within the time Delta1
        Use the most relaxed requirement for this, that is:
            (1) The two trip process don't overlap at all (for drive alone case)!!!

        Required: must run self.compute_pnr_access function to compute access time to station
        The code could be slow as many operations are not vectorized
        TODO: optimized by cutting into smaller matrices, one matrix for each station

        Two trips are carpool-able through PNR station if:
            (Absolute difference rule: The time difference between two departures are within a fixed time threshold)
            (Alternative rule: Driver departs before rider but no earlier than the given threshold.)
        :param Delta1: depart time different within Delta1 minutes
        :param default_rule: if True, strict time different; if False, absolute time difference
        :return:
        """
        trips = self.trips
        nrow, ncol = self.nrow, self.ncol
        # simple method
        depart_lst = np.array(trips['new_min'].tolist()).reshape((1, -1))  # depart minute
        mat = np.tile(depart_lst.transpose(), (1, ncol))
        mat = np.tile(depart_lst, (nrow, 1)) - mat  # depart time difference (driver's depart - pax depart)
        if default_rule:
            # criterion 1. driver should leave earlier than the passenger, but not earlier than 15 minutes
            # criterion 2. driver should wait at most 5 minutes
            self.cp_pnr_matrix = (self.cp_pnr_matrix &
                                  (mat >= 0) &
                                  (np.absolute(mat) <= Delta1)).astype(np.bool_)
        else:
            # criterion 1. passenger/driver depart time within +/- 15 minutes
            # criterion 2. passenger/driver wait time +/- 5 minutes
            self.cp_pnr_matrix = (self.cp_pnr_matrix &
                                  (np.absolute(mat) <= Delta1)).astype(np.bool_)

    def compute_depart_01_matrix_pre(
            self,
            Delta1: float = 15,
            default_rule: bool = True,
    ):
        """
        Required: must run self.compute_diagonal function to compute diagonal travel information?
        Filter out carpool trips based on time constraint conditions.
        Two trips are carpool-able if:
            (Absolute difference rule: The time difference between two departures are within a fixed time threshold)
            (Alternative rule: Driver departs before rider but no earlier than the given threshold.)
        :param Delta1: depart time different within Delta1 minutes
        :param default_rule: if True, strict time different; if False, absolute time difference
        :return:
        """
        # step 1. Measure depart time difference within threshold time (default is 15 minutes)
        nrow, ncol = self.nrow, self.ncol
        depart_lst = np.array(self.trips['new_min'].tolist()).reshape((1, -1))  # depart minute
        # print('depart list: ', depart_lst)
        # compare departure time difference
        mat = np.tile(depart_lst.transpose(), (1, ncol))
        mat = np.tile(depart_lst, (nrow, 1)) - mat  # depart time difference (driver's depart - pax depart)
        # print('tt matrix')
        # print(self.tt_matrix[0:20, 0:20])
        # print('tt matrix part 1')
        # print(self.tt_matrix_p1[0:20, 0:20])
        # print('depart time different matrix')
        # print(mat[0:20, 0:20])
        if default_rule:
            # condition 1. passenger should arrive earlier than the driver, plus
            # condition 2. only the driver start no later than Delta1 minutes after passenger leaves
            self.cp_matrix = (self.cp_matrix &
                              (mat >= 0) &
                              (np.absolute(mat) <= Delta1)).astype(np.bool_)
        else:
            # condition 1. passenger/driver depart time within +/- Delta1 minutes
            self.cp_matrix = (self.cp_matrix &
                              (np.absolute(mat) <= Delta1)).astype(np.bool_)

    def compute_depart_01_matrix_post_pnr(
            self,
            Delta2: float = 10,
            Gamma: float = 0.2,
            default_rule: bool = True,
    ):
        """
        Filter PNR trips considering driver's waiting time is limited
        :param Delta2: the driver's maximum waiting time
        :param default_rule:
            if True, strict time different (applies for driver)
            if False, absolute time difference (applies for both travelers)
        :return:
        """
        # complex and slow method for post analysis
        nrow, ncol = self.nrow, self.ncol
        # step 1. get the travel time to each feasible station
        ind = np.argwhere(self.cp_pnr_matrix == 1)
        # pnr depart time matrix (diff between drivers)
        mat = np.full((nrow, ncol), np.nan)
        for ind_one in ind:
            trip_id1, trip_id2 = ind_one
            # get station id they share
            trip_row1 = self.trips.iloc[trip_id1]
            trip_row2 = self.trips.iloc[trip_id2]
            sid = self._check_trips_best_pnr(trip_row1, trip_row2, trip_id1, trip_id2)
            # print(ind_one, ";", sid)
            if sid is None:
                continue
            # access information (path, time, distance)
            info1 = self.pnr_access_info[trip_id1, sid]
            info2 = self.pnr_access_info[trip_id2, sid]
            t1 = self.trips['new_min'].iloc[trip_id1] + info1[1]  # arrival time at pnr for person 1
            t2 = self.trips['new_min'].iloc[trip_id2] + info2[1]  # arrival time at pnr for person 2
            # the number of minutes it takes for passenger to wait for the driver
            mat[trip_id1][trip_id2] = t1 - t2
            mat[trip_id2][trip_id1] = t2 - t1

        passenger_time = np.array([self.soloTimes[i] for i in range(self.ncol)]).reshape(1, -1)
        passenger_time = np.tile(passenger_time, (nrow, 1))
        if default_rule:
            # passenger only waits the driver should wait at most Delta2 minutes
            self.cp_pnr_matrix = (self.cp_pnr_matrix &
                                  (mat >= 0) & (mat <= Delta2) &
                                  (np.absolute(mat/passenger_time) <= Gamma)).astype(np.bool_)
        else:
            # passenger/driver waits the other party for at most Delta2 minutes
            self.cp_pnr_matrix = (self.cp_pnr_matrix &
                                  (np.absolute(mat) <= Delta2) &
                                  (np.absolute(mat/passenger_time) <= Gamma).astype(np.bool_))

    def compute_depart_01_matrix_post(
            self,
            Delta2: float = 10,
            Gamma: float = 0.2,
            default_rule: bool = True,
    ):
        """
        After tt_matrix_p1 is computed, filter by maximum waiting time for the driver at pickup location
        :param Delta2: driver's maximum waiting time
        :param default_rule: if True, strict time different; if False, absolute time difference
        :return:
        """
        # step 2. Maximum waiting time for driver is Delta2 (default is 5 minutes)
        nrow, ncol = self.nrow, self.ncol
        # driver_lst = np.array(self.trips_front['new_min'].tolist()).reshape((1, -1))  # depart minute
        # for non-simulation with time case, passenger <==> driver have the same scope
        passenger_lst = np.array(self.trips['new_min'].tolist()).reshape((1, -1))
        # compare departure time difference
        dri_arr = np.tile(passenger_lst.reshape((-1, 1)), (1, ncol)) + self.tt_matrix_p1
        pax_dep = np.tile(passenger_lst.reshape((1, -1)), (nrow, 1))  # depart time difference
        # step 2. Maximum waiting time for driver is Delta2 (default is 10 minutes)
        wait_time_mat = dri_arr - pax_dep  # wait time matrix for driver
        # for post analysis, directly update final cp_matrix
        passenger_time = np.array([self.soloTimes[i] for i in range(self.ncol)]).reshape(1, -1)
        passenger_time = np.tile(passenger_time, (nrow, 1))
        if default_rule:
            # passenger only waits the driver should wait at most Delta2 minutes
            self.cp_matrix = (self.cp_matrix &
                              (wait_time_mat >= 0) & (np.absolute(wait_time_mat) <= Delta2) &
                              (np.absolute(wait_time_mat/passenger_time) <= Gamma)).astype(np.bool_)
        else:
            # passenger/driver waits the other party for at most Delta2 minutes
            self.cp_matrix = (self.cp_matrix &
                              (np.absolute(wait_time_mat) <= Delta2) &
                              (np.absolute(wait_time_mat/passenger_time) <= Gamma)).astype(np.bool_)

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
        nrow, ncol = self.nrow, self.ncol
        oxs = np.array(self.trips.ox.tolist()).reshape((1, -1))
        oys = np.array(self.trips.oy.tolist()).reshape((1, -1))
        dxs = np.array(self.trips.dx.tolist()).reshape((1, -1))
        dys = np.array(self.trips.dy.tolist()).reshape((1, -1))

        # origin distance in (x, y) axis between two trips (vectorized)
        mat_ox = np.tile(oxs.transpose(), (1, ncol))
        mat_ox = np.abs(mat_ox - np.tile(oxs, (nrow, 1)))
        mat_oy = np.tile(oys.transpose(), (1, ncol))
        mat_oy = np.abs(mat_oy - np.tile(oys, (nrow, 1)))

        # destination distance in (x, y) axis between two trips (vectorized)
        mat_dx = np.tile(dxs.transpose(), (1, ncol))
        mat_dx = np.abs(mat_dx - np.tile(dxs, (nrow, 1)))
        mat_dy = np.tile(dys.transpose(), (1, ncol))
        mat_dy = np.abs(mat_dy - np.tile(dys, (nrow, 1)))

        man_o = np.sqrt(mat_ox ** 2 + mat_oy ** 2)
        man_d = np.sqrt(mat_dx ** 2 + mat_dy ** 2)

        # compute euclidean travel distance
        mat_diag = np.sqrt((oxs - dxs) ** 2 + (oys - dys) ** 2)
        # compute reroute straight distance, then origin SOV distance for the driver
        rr = (man_o + man_d + np.tile(mat_diag, (nrow, 1)))
        ori = np.tile(mat_diag, (nrow, 1))
        mat_ratio = rr / ori

        # print('mat_orig', mat_ratio.shape)
        # print(mat_ratio[:5, :5])
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

    def compute_reroute_01_matrix_pnr(
            self,
            delta: float = 15,
            gamma: float = 1.5,
            ita_pnr: float = 0.5,
            print_mat: bool = True
    ):
        """
        Compute carpool-able matrix considering total reroute time (non-shared trip segments)
        This cannot be estimated before travel time matrix (self.tt_matrix) is fully computed
        :param delta: maximum reroute time (in minutes) acceptable for the driver
        :param gamma: the maximum ratio of extra travel time over driver's original travel time
        :param ita_pnr: the ratio between passenger's travel time and driver's travel time should be greater than ita
        :return:
        """
        nrow, ncol = self.nrow, self.ncol
        # propagate drive alone matrix
        drive_alone_tt = self.tt_matrix.diagonal().reshape(-1, 1)
        passenger_alone_tt = np.array([self.soloTimes[i] for i in range(self.ncol)]).reshape(1, -1)
        # condition 1: total reroute time is smaller than threshold minutes
        cp_reroute_matrix1 = ((self.tt_pnr_matrix - np.tile(drive_alone_tt, (1, ncol)))
                              <= delta).astype(bool)
        cp_reroute_matrix2 = ((self.tt_pnr_matrix - np.tile(passenger_alone_tt, (nrow, 1)))
                              <= delta).astype(bool)
        cp_reroute_matrix = cp_reroute_matrix1 & cp_reroute_matrix2
        # condition 2: ratio is smaller than a threshold
        cp_reroute_ratio_matrix1 = ((self.tt_pnr_matrix / np.tile(drive_alone_tt, (1, ncol)))
                                    <= gamma).astype(bool)
        cp_reroute_ratio_matrix2 = ((self.tt_pnr_matrix / np.tile(passenger_alone_tt, (nrow, 1)))
                                    <= gamma).astype(bool)
        cp_reroute_ratio_matrix = cp_reroute_ratio_matrix1 & cp_reroute_ratio_matrix2
        # condition 3: rider should at least share ita_pnr of the total travel time
        cp_time_similarity = ((self.tt_pnr_matrix_shared[:self.nrow, :self.ncol] / self.tt_pnr_matrix)
                              >= ita_pnr).astype(bool)
        # condition 4: after drop-off time / whole travel time?
        if print_mat:
            # print("drive alone...")
            # print(drive_alone_tt[:8, :8])
            print("PNR path-based filters results:")
            print("cp_reroute_matrix passed count:", cp_reroute_matrix.sum())
            # print(cp_reroute_matrix[:8, :8])
            print("cp_reroute_ratio_matrix passed count:", cp_reroute_ratio_matrix.sum())
            # print(cp_reroute_ratio_matrix[:8, :8])
            print("cp_time_similarity passed count:", cp_time_similarity.sum())
            # print(cp_time_similarity[:8, :8])
        self.cp_pnr_matrix = (self.cp_pnr_matrix &
                              cp_reroute_matrix &
                              cp_reroute_ratio_matrix &
                              cp_time_similarity).astype(bool)
        # need to mask tt and ml matrix
        self.tt_pnr_matrix[self.cp_pnr_matrix == 0] = np.nan
        self.ml_pnr_matrix[self.cp_pnr_matrix == 0] = np.nan

    def compute_reroute_01_matrix(
            self,
            delta: float = 15,
            gamma: float = 1.5,
            ita: float = 0.5,
    ) -> None:
        """
        Compute carpool-able matrix considering total reroute time (non-shared trip segments)
        This cannot be estimated before travel time matrix (self.tt_matrix) is fully computed
        :param ita: passenger's travel time should be greater than ita (minutes) compared to that of drivers
        :param delta: maximum reroute time (in minutes) acceptable for the driver
        :param gamma: the maximum ratio of extra travel time over driver's original travel time
        :return:
        """
        nrow, ncol = self.nrow, self.ncol
        # propagate drive alone matrix
        # drive_alone_tt = self.tt_matrix.diagonal().reshape(-1, 1)
        drive_alone_tt = np.array([self.soloTimes[i] for i in range(self.nrow)]).reshape(-1, 1)
        # condition 1: total reroute time is smaller than threshold minutes
        cp_reroute_matrix = ((self.tt_matrix - np.tile(drive_alone_tt, (1, ncol))) <= delta).astype(int)
        # condition 2: ratio is smaller than a threshold
        cp_reroute_ratio_matrix = ((self.tt_matrix / np.tile(drive_alone_tt, (1, ncol))) <= gamma).astype(int)
        # condition 3: rider should at least share 50% of the total travel time
        passenger_time = np.tile(drive_alone_tt.reshape(1, -1), (nrow, 1))
        cp_time_similarity = ((passenger_time / self.tt_matrix) >= ita).astype(np.bool_)
        # print('reroute time matrix...')
        # # print((self.tt_matrix - np.tile(drive_alone_tt, (1, ncol)))[:20, :20])
        # print(cp_reroute_matrix[:20, :20])
        # print('reroute ratio matrix...')
        # print(cp_reroute_ratio_matrix[:20, :20])
        # condition 4: after drop-off time / whole travel time?
        # self.cp_reroute_matrix = cp_reroute_matrix # TODO: delete this row when not needed
        self.cp_matrix = (self.cp_matrix &
                          cp_reroute_matrix &
                          cp_reroute_ratio_matrix &
                          cp_time_similarity).astype(np.bool_)
        # need to mask tt and ml matrix
        self.tt_matrix[self.cp_matrix == 0] = np.nan
        self.ml_matrix[self.cp_matrix == 0] = np.nan

    def compute_carpoolable_trips_pnr(self, reset_off_diag: bool = False) -> None:
        """
        Instead of computing all combinations, only compute all carpool-able trips.
        :param reset_off_diag: if True, reset all carpool trips information EXCEPT drive alone trips
            if False, only update based on carpool-able matrix information
        :return: None
        """
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
        # print('PNR Indices matching0: \n', indexes_pairs)
        for index in indexes_pairs:
            self.compute_carpool_pnr(index[0], index[1], fixed_role=True)

    def compute_carpoolable_trips(
            self,
            reset_off_diag: bool = False
    ) -> None:
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

    def compute_optimal_lp(
            self,
            mute: bool = True,
            use_both: bool = False,
    ) -> None:
        """
        Use Gurobipy to solve for optimal solutions
        :param mute: if True, don't print output results.
        :param use_both: if True, consider both modes
        :return: No return
        """
        nrow, ncol = self.nrow, self.ncol
        # below functions are already called before in
        # compute_in_one_step, no need to call again
        # self.compute_depart_01_matrix_pre(Delta1=15)
        # self.compute_pickup_01_matrix(threshold_dist=5280 * 5)
        # self.compute_diagonal()  # compute drive alone scenarios
        # self.compute_carpoolable_trips(reset_off_diag=False)
        # self.compute_depart_01_matrix_post(Delta2=10)
        # self.compute_reroute_01_matrix()

        # use Gurobi linear programming solver to get optimal solution
        md1 = Model("CP");
        if mute:
            md1.setParam('OutputFlag', 0)
        # travel time matrix
        if use_both:
            tt_matrix = self.tt_matrix_all
        else:
            tt_matrix = self.tt_matrix
        # a list of trips in the cluster
        travel_pairs = np.argwhere(~np.isnan(tt_matrix))
        tps = [tuple(tp.tolist()) for tp in travel_pairs]

        # get cost for each travel pairs
        c = {(i, j): tt_matrix[i, j] for i, j in tps}
        x = md1.addVars(tps, vtype=GRB.BINARY)
        # set objective function
        md1.modelSense = GRB.MINIMIZE
        md1.setObjective(quicksum(x[i, j] * c[i, j] for i, j in tps))

        # a help method to get carpoolable indexes
        def find_pairs(pairs, fixed_idx, fix_row=True):
            returned_lst = []
            for i, j in pairs:
                if fix_row:
                    if fixed_idx == i:
                        returned_lst.append((i, j))
                else:
                    if fixed_idx == j:
                        returned_lst.append((i, j))
            return returned_lst

        # add LP constraints
        diag_size = min(nrow, ncol)
        for i in range(nrow):
            tp_row_lst = find_pairs(tps, i)
            md1.addConstr(quicksum(x[p] for p in tp_row_lst) <= 1);  # row sum no greater than 1
        for j in range(ncol):
            tp_col_lst = find_pairs(tps, j, fix_row=False)
            md1.addConstr(quicksum(x[p] for p in tp_col_lst) <= 1);  # col sum no greater than 1
        # only assign roles for current queries, future ones can be left for assignment in the future
        for i in range(diag_size):
            tp_row_lst = find_pairs(tps, i)
            tp_col_lst = find_pairs(tps, i, fix_row=False)
            tp_all_lst = list(set(tp_row_lst + tp_col_lst))
            # one cannot be both driver and passenger at the same time
            md1.addConstr(quicksum(x[p] for p in tp_all_lst) == 1);  # row col sum equals to 1 exactly
        # solve the problem.
        md1.optimize()
        # get results
        self.result_lst = [p for p in tps if x[p].x > 0.5]

    def evaluate_individual_trips_both(
            self,
            verbose: bool = False,
            use_bipartite: bool = False,
            trips: pd.DataFrame | None = None,
    ):
        """
        After getting optimized results, expand the trip column with before after information for each person.
        This code works for general cases.

        If row not equals to column (in inherited classes), then only report the results of the lower left of the
        square matrix!!!
        :param verbose:
        :param use_bipartite:
        :return:
        """
        if trips is None:
            trips = self.trips
        self.trip_summary_df = trips[['new_min']].copy()
        # init new columns (before/after travel time and distances)
        # values are placeholder
        self.trip_summary_df = self.trip_summary_df.assign(
            **{'before_time': 0.0, 'before_dist': 0.0,
               'after_time': 0.0, 'after_dist': 0.0,
               'SOV': True, 'as_passenger': False,
               'partner_idx': 0, 'station': -1})
        # for each traveler, find its SOV trip time/distances,
        # then find the optimized trip information
        if use_bipartite is False:
            temp_records = self.result_lst
        else:
            temp_records = self.result_lst_bipartite

        index_paired = []
        for d_idx, p_idx in temp_records:
            d, p = self.int2idx[d_idx], self.int2idx[p_idx]
            choice = self.choice_matrix[d_idx][p_idx]
            sid = -1
            if choice == 1:
                trip1, trip2 = trips.loc[d], trips.loc[p]
                sid = self._check_trips_best_pnr(trip1, trip2, d_idx, p_idx)
                # note: need to select mode based on the choice matrix

            # this is strictly the driver's time
            row_d = [
                self.soloTimes[d_idx], self.soloDists[d_idx],
                self.tt_matrix_all[d_idx, p_idx], self.ml_matrix_all[d_idx, p_idx],
                sid
            ]

            # passenger costs assumed a fixed number
            # (need to recalculate passenger's new travel time in post analysis)
            row_p = [self.soloTimes[p_idx], self.soloDists[p_idx],
                     self.soloTimes[p_idx], self.soloDists[p_idx], sid]

            row_d = [round(r, 3) for r in row_d]
            row_p = [round(r, 3) for r in row_p]
            if verbose:
                print(d_idx, p_idx)
                print(row_d)

            self.trip_summary_df.loc[
                d,
                ['before_time', 'before_dist',
                 'after_time', 'after_dist',
                 'station']
            ] = row_d

            if p_idx != d_idx:  # carpool, not SOV in after case
                # for passenger, travel is same as before
                if verbose:
                    print(row_p)
                self.trip_summary_df.loc[
                    p,
                    ['before_time', 'before_dist',
                     'after_time', 'after_dist',
                     'station']
                ] = row_p

                # finally, update role info
                self.trip_summary_df.loc[d, ['SOV', 'as_passenger', 'partner_idx']] = [False, False, p]
                self.trip_summary_df.loc[p, ['SOV', 'as_passenger', 'partner_idx']] = [False, True, d]
                index_paired.append(p)
                index_paired.append(d)
            else:  # drive alone (SOV)
                # partner is the driver herself
                self.trip_summary_df.loc[d, ['SOV', 'as_passenger', 'partner_idx']] = [True, False, d]
                index_paired.append(p)
        if verbose:
            print('LP output results are updated.')
        self.trip_summary_df = self.trip_summary_df.loc[index_paired, :]
        return self.trip_summary_df

    def evaluate_individual_trips(
            self,
            verbose: bool = False,
            use_bipartite: bool = False
    ) -> pd.DataFrame:
        """
        After getting optimized results, expand the trip column with before after information for each person.
        This code works for general cases.

        If row not equals to column (in inherited classes), then only report the results of the lower left of the
        square matrix!!!
        :param verbose:
        :param use_bipartite:
        :return:
        """
        nrow, ncol = self.nrow, self.ncol
        # left_trips_filt = self.trips.SOV & (self.trips.new_min > self.t0 + self.epsilon)
        # dropped_indexes = self.trips.loc[~left_trips_filt, :].index.tolist()
        self.trip_summary_df = self.trips[['new_min']].copy()

        # init new columns (before/after travel time and distances)
        # values are placeholder
        self.trip_summary_df = self.trip_summary_df.assign(
            **{'before_time': 0.0, 'before_dist': 0.0,
               'after_time': 0.0, 'after_dist': 0.0,
               'SOV': True, 'as_passenger': False, 'partner_idx': 0})
        # downcast to save memory
        self.trip_summary_df = self.trip_summary_df.astype(
            {"before_time": "float32", "before_dist": "float32",
             'after_time': "float32", 'after_dist': "float32",
             'partner_idx': "int32"})
        # for each traveler, find its SOV trip time/distances,
        # then find the optimized trip information
        if use_bipartite is False:
            temp_records = self.result_lst
        else:
            temp_records = self.result_lst_bipartite
        # print('assignment list is:', temp_records)

        index_paired = []
        # index_int_paired = []
        for d_idx, p_idx in temp_records:
            d, p = self.int2idx[d_idx], self.int2idx[p_idx]
            row_d = [self.soloTimes[d_idx], self.soloDists[d_idx],
                     self.tt_matrix[d_idx, p_idx], self.ml_matrix[d_idx, p_idx]]
            row_d = [round(r, 3) for r in row_d]

            row_p = [self.soloTimes[p_idx], self.soloDists[p_idx],
                     self.soloTimes[p_idx], self.soloDists[p_idx]]
            row_p = [round(r, 3) for r in row_p]

            self.trip_summary_df.loc[
                d,
                ['before_time', 'before_dist',
                 'after_time', 'after_dist']
            ] = row_d

            if p_idx != d_idx:  # carpool, not SOV in after case
                # for passenger, travel is same as before
                if verbose:
                    # print(row_p)
                    pass
                self.trip_summary_df.loc[
                    p,
                    ['before_time', 'before_dist',
                     'after_time', 'after_dist']
                ] = row_p

                # finally, update role info
                self.trip_summary_df.loc[d, ['SOV', 'as_passenger', 'partner_idx']] = [False, False, p]
                self.trip_summary_df.loc[p, ['SOV', 'as_passenger', 'partner_idx']] = [False, True, d]
                index_paired.append(p)
                index_paired.append(d)
                # index_int_paired.append(p_idx)
                # index_int_paired.append(d_idx)
            else:  # drive alone (SOV)
                # partner is the driver herself
                self.trip_summary_df.loc[d, ['SOV', 'as_passenger', 'partner_idx']] = [True, False, d]
                index_paired.append(p)
        self.trip_summary_df = self.trip_summary_df.loc[index_paired, :]
        return self.trip_summary_df

    def evaluate_trips(
            self,
            verbose: bool = False,
            use_bipartite: bool = False
    ):
        """
        Evaluate the assignment's performances. Interested in:
        (1) The changes in total vehicular travel time
        (2) The changes in total travel time
        (3) Number of carpool pairs matched

        The extra travel time solely contributed by driver reroutes to pick up passengers.
        The first metric measures systemic benefits; The second measures driver's sacrifices.
        For future case, more statistics will be developed to measure extreme cases.

        If row not equals to column (in inherited classes), then only report the results of the left square matrix!!!
        :return:
        """
        nrow, ncol = self.nrow, self.ncol

        def count_parties(paired_lst):
            party_lst = []
            tot_count, num_paired = 0, 0
            for pr in paired_lst:
                if pr[0] == pr[1]:
                    tot_count += 1
                    party_lst.append(pr[0])
                else:
                    tot_count += 2
                    num_paired += 2
                    party_lst.append(pr[0])
                    party_lst.append(pr[1])
            return tot_count, num_paired, party_lst

        if use_bipartite:
            tot_count, num_paired, party_lst = count_parties(self.result_lst_bipartite)
            rl = self.result_lst_bipartite
        else:
            tot_count, num_paired, party_lst = count_parties(self.result_lst)
            rl = self.result_lst
        # special case: If num of columns is greater than rows, then self pairs in the right part should
        # not be concluded in the evaluation
        ori_tt = sum(self.soloTimes[p] for p in party_lst)
        ori_ml = sum(self.soloDists[p] for p in party_lst)
        new_tt, new_ml = 0, 0
        sid = None  # PNR station id
        for p in rl:
            if self.choice_matrix[p] == 0:
                # print('simple:', p, self.tt_matrix[p], self.ml_matrix[p])
                new_tt += self.tt_matrix[p]
                new_ml += self.ml_matrix[p]
            elif self.choice_matrix[p] == 1:
                # print('shared:', p, self.tt_pnr_matrix[p], self.ml_pnr_matrix[p])
                new_tt += self.tt_pnr_matrix[p]
                new_ml += self.ml_pnr_matrix[p]  # + self.ml_pnr_matrix_p[p]
                trip1, trip2 = self.trips.iloc[p[0], :], self.trips.iloc[p[1], :]
                sid = self._check_trips_best_pnr(trip1, trip2, p[0], p[1])
            else:  # no assign, drive alone
                # print('drive alone')
                new_tt += self.tt_pnr_matrix[p]
                new_ml += self.ml_pnr_matrix[p]  # + self.ml_pnr_matrix_p[p]
        if verbose:
            # if use_bipartite:
            #     print("*****Use bipartite method*****")
            # else:
            #     print("*****Use LP method*****")
            print("{} persons found carpooling in a cluster with {} persons".format(num_paired, tot_count))
            print_str = "Original total vehicular travel time is {} veh-min;\n"
            print_str += "New total vehicular travel time is {} veh-min "
            print(print_str.format(round(ori_tt, 2), round(new_tt, 2)))

            print_str = "Original total vehicular travel mileage is {} miles;\n"
            print_str += "New total vehicular travel mileage is {} miles."
            print(print_str.format(round(ori_ml, 2), round(new_ml, 2)))

        return tot_count, num_paired, ori_tt, new_tt, ori_ml, new_ml, sid

    def plot_single_trip(
            self,
            intind1: str, intind2: str,
            network_df: pd.DataFrame | None = None,
            fixed_role: bool = False,
    ):
        """
        Plot carpool trip for a single trip corresponding to the matrix position (ind 1, ind 2)
        Plot three carpool plans at once. A picks up B, B picks up A, one plot showing both schemes.
        :param intind1: integer index 1 for the driver
        :param intind2: integer index 2 for the passenger
        :param network_df: network file containing geometry information for plotting
        :return: lastly used plotting axes
        """
        # print depart time diff
        tod_1, tod_2 = self.trips.iloc[intind1, :]['new_min'], self.trips.iloc[intind2, :]['new_min']
        tt1 = self.tt_matrix_p1[intind1, intind2]
        sb = [ScaleBar(1, "m", dimension="si-length", fixed_value=5*1000/0.621371,
                       box_alpha=0.5, location="upper left",
                       scale_formatter=lambda value, unit: f'{round(0.621371 * value/1000)} mile')
              for i in range(2)]
        if network_df is None:
            network_df = self.links
        network_df = network_df.to_crs("3857")

        dists_1, links_1, dists_2, links_2 = self.compute_carpool(
            intind1, intind2,
            print_dist=False, fill_mat=False
        )
        # check shared route
        d2, p2 = self.soloDists[intind2], self.soloPaths[intind2]
        d1, p1 = self.soloDists[intind1], self.soloPaths[intind1]
        # intind1 is driver, intind2 is passenger
        # driver's trip or the whole trip
        fig1, ax1, h1 = plot_seq(
            trip_seq=links_1, gpd_df=network_df,
            linewidth=2, color='red', arrow_color='red',
            trip_df=self.trips.iloc[[intind1, intind2], :],
            skeleton="I-285"
        )

        # passenger's trip or the shared trip
        fig1, ax1, h2 = plot_seq(
            trip_seq=p2, gpd_df=network_df,
            ctd=True, color='green', linewidth=20,
            ax=ax1, plt_arrow=False
        )
        plt.title(
            f"Driver: {intind1}({tod_1}+{tt1:.2f}min); Passenger: {intind2}({tod_2}min)"
        )

        idx1 = self.int2idx[intind1]
        idx2 = self.int2idx[intind2]
        rec_driver = [idx1, "dc_driver", links_1]
        rec_passenger = [idx2, "dc_passenger", p2]
        rec_combined = [rec_driver, rec_passenger]

        # helper function
        def make_legend_arrow(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
            p = mpatches.FancyArrow(0, 0.5 * height, width, 0,
                                    length_includes_head=True, head_width=0.75 * height,
                                    shape='left')
            return p
        plt.legend(handles=h1+h2+[mpatches.Patch(color='green', label='shared trip', alpha=0.3)],
                   handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow)},
                   loc='lower right', fontsize=20)

        # add base map for fig1
        cx.add_basemap(ax1, source=cx.providers.CartoDB.Voyager)
        ax1.add_artist(sb[0])
        ax1.axis("off")

        # If fixed rule, stop plotting!!!
        if fixed_role:
            return fig1, ax1, rec_combined

        # intind2 is driver, intind1 is passenger
        fig2, ax2, h3 = plot_seq(
            trip_seq=links_2, gpd_df=network_df,
            color='red', arrow_color='red', linewidth=2,
            ax=None, trip_df=self.trips.iloc[[intind2, intind1], :]
        )
        fig2, ax2, h4 = plot_seq(
            trip_seq=p1, gpd_df=network_df,
            ctd=True, color='green', linewidth=20,
            ax=ax2, plt_arrow=False
        )
        plt.legend(handles=h3+h4+[mpatches.Patch(color='green', label='shared trip', alpha=0.3)],
                   handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow)},
                   loc='lower right', fontsize=20)
        # add base map for fig2
        plt.title(
            f"Driver: {intind2}({tod_2}min); Passenger: {intind1}({tod_1}min)"
        )
        cx.add_basemap(ax2, source=cx.providers.CartoDB.Voyager)
        ax2.add_artist(sb[1])
        ax2.axis("off")
        return fig1, fig2

    def plot_single_trip_pnr(
            self, intind1: int, intind2: int,
            trips: pd.DataFrame | None = None,
            network_df: pd.DataFrame | None = None,
            fixed_role: bool = False,
    ):
        idx1, idx2 = self.int2idx[intind1], self.int2idx[intind2]
        if trips is None:
            trips = self.trips
            sid = trips.loc[idx1, 'pnr'][0]
        else:
            sid = trips.loc[idx1, 'station']
        # print depart time diff
        tod_1, tod_2 = trips.loc[idx1, :]['new_min'], trips.loc[idx2, :]['new_min']
        tt1, tt2 = None, None

        if self.pnr_access_info[intind1, sid] is not None:
            tt1 = self.pnr_access_info[intind1, sid][1]
        if self.pnr_access_info[intind2, sid] is not None:
            tt2 = self.pnr_access_info[intind2, sid][1]
        # scale bar
        sb = [ScaleBar(1, "m", dimension="si-length", fixed_value=5*1000/0.621371,
                       box_alpha=0.5, location="upper left",
                       scale_formatter=lambda value, unit: f'{round(0.621371 * value/1000)} mile')
              for i in range(2)]
        if network_df is None:
            network_df = self.links
        # dists_1, links_1, dists_2, links_2
        dists_1, (links_1p, links_1d0, links_1d1, links_1d2), \
            dists_2, (links_2p, links_2d0, links_2d1, links_2d2), station_id = \
            self.compute_carpool_pnr(intind1, intind2, print_dist=False, fill_mat=False)

        # intind1 is driver, intind2 is passenger
        # (red arrow): vehicle trajectory
        fig1, ax1, h1 = plot_seq(
            trip_seq=links_1d0+links_1d1+links_1d2, gpd_df=network_df,
            linewidth=2, color='red', arrow_color='red',
            trip_df=self.trips.iloc[[intind1, intind2], :],
            station_df=self.parking_lots.iloc[[station_id]]
        )

        # (trip segment 1): from driver start --> meet point
        fig1, ax1, h2 = plot_seq(
            trip_seq=links_1p, gpd_df=network_df, ctd=True,
            linewidth=10, color='blue',
            ax=ax1, plt_arrow=False
        )

        # (trip segment 2): from meet point --> end
        fig1, ax1, h2 = plot_seq(
            trip_seq=links_1d1, gpd_df=network_df, ctd=True,
            linewidth=20, color='green',
            ax=ax1, plt_arrow=False
        )

        rec_driver = [idx1, "pnr_driver", links_1d0+links_1d1+links_1d2]
        rec_passenger = [idx2, "pnr_passenger", links_1d1]
        rec = [rec_driver, rec_passenger]

        # plot coordinate of station id
        plt.title(
            f"Driver: {intind1}({tod_1}+{tt1:.2f}min); Passenger: {intind2}({tod_2}min) (Park and Ride)"
        )
        cx.add_basemap(ax1, source=cx.providers.CartoDB.Voyager)
        ax1.add_artist(sb[0])
        ax1.axis("off")
        # helper function
        def make_legend_arrow(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
            p = mpatches.FancyArrow(0, 0.5 * height, width, 0,
                                    length_includes_head=True, head_width=0.75 * height,
                                    shape='left')
            return p
        plt.legend(handles=(h1 + h2 + [mpatches.Patch(color='green', label='shared trip', alpha=0.3)] +
                            [mpatches.Patch(color='blue', label="passenger's drive to PNR", alpha=0.3)]),
                   handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow)},
                   loc='lower right', fontsize=20)

        # If fixed rule, stop plotting!!!
        if fixed_role:
            return fig1, ax1, rec

        # intind2 is driver, intind1 is passenger
        fig2, ax2, h3 = plot_seq(
            trip_seq=links_2d0+links_2d1+links_2d2, gpd_df=network_df,
            color='red', arrow_color='red', linewidth=2,
            ax=None, trip_df=self.trips.iloc[[intind2, intind1], :],
            station_df=self.parking_lots.iloc[[station_id]]
        )
        fig2, ax2, h4 = plot_seq(
            trip_seq=links_2p, gpd_df=network_df, ctd=True,
            linewidth=10, color='blue', arrow_color='blue',
            ax=ax2, plt_arrow=False
        )
        fig2, ax2, h5 = plot_seq(
            trip_seq=links_2d1, gpd_df=network_df, ctd=True,
            linewidth=20, color='green',
            ax=ax2, plt_arrow=False
        )

        # plt.title("Driver: {}; Passenger: {} (Park and Ride)".format(intind2, intind1))
        plt.title(
            f"Driver: {intind2}({tod_2}+{tt2:.2f}min); Passenger: {intind1}({tod_1}min) (Park and Ride)"
        )
        plt.legend(handles=(h3+h4+[mpatches.Patch(color='green', label='shared trip', alpha=0.3)]
                            + [mpatches.Patch(color='blue', label="passenger's drive to PNR", alpha=0.3)]),
                   handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow)},
                   loc='lower right', fontsize=20)
        cx.add_basemap(ax2, source=cx.providers.CartoDB.Voyager)
        ax2.add_artist(sb[1])
        ax2.axis("off")
        return (fig1, ax1), (fig2, ax2)

    def compute_optimal_bipartite(self) -> None:
        """
        Solve the pairing problem using traditional bipartite method.
        This is to compare results with that of linear programming one
        :return:
        """
        bipartite_obj = tg.CarpoolBipartite(self.cp_matrix_all, self.tt_matrix_all)
        num_pair, pairs = bipartite_obj.solve_bipartite_conflicts_naive()
        self.result_lst_bipartite = pairs

    def _generate_choice_matrix(self, cp_mat1: np.ndarray, cp_mat2: np.ndarray):
        """
        TODO: add other two modes (prioritize to direct carpool, prioritize to pnr carpool)
        Generate choice matrix by comparing two matrices
        :param cp_mat1: carpool matrix mode one (cp_matrix)
        :param cp_mat2: carpool matrix mode two (cp_pnr_matrix)
        :return:
        """
        # priority: 0: based on driver's traveling time
        # 1: carpool mode > pnr mode when both are available
        # 2: pnr mode > carpool mode when both are available
        priority = self.priority_mode
        cp_mat1 = cp_mat1.astype("int8")
        cp_mat2 = cp_mat2.astype("int8")
        ind_mat1_one = np.where(cp_mat1 == 1)
        ind_mat2_one = np.where(cp_mat2 == 1)
        # compete_pos = np.where(cp_mat1 == 1 & cp_mat2 == 1)
        choice_mat = np.full(cp_mat1.shape, np.nan, dtype="int8")
        if priority == 0:
            # TODO: sanity check on this priority mode!
            mat1_better_inds = ((self.tt_matrix <= self.tt_pnr_matrix) &
                                (cp_mat1 == 1) & (cp_mat2 == 1))
            # print("mat1_better_inds:", mat1_better_inds)
            # cp_matrix may have more columns than cp_pnr_matrix in inherit??
            choice_mat[ind_mat1_one] = 0
            choice_mat[ind_mat2_one] = 1
            # for position both choices are good,
            # need to decide which is better by traveling time
            choice_mat[mat1_better_inds] = 0
            # np.put(choice_mat, mat1_better_inds, 0)
        elif priority == 1:  # prioritize for direct carpool mode
            choice_mat[ind_mat2_one] = 1
            choice_mat[ind_mat1_one] = 0
        elif priority == 2:  # prioritize for pnr carpool mode
            choice_mat[ind_mat1_one] = 0
            choice_mat[ind_mat2_one] = 1
        return choice_mat

    def combine_pnr_carpool(self, include_direct: bool = True, print_mat: bool = True):
        """
        Generate a final matrix combining two modes: direct pickup and pickup at PNR station
        If (i, j) form a carpool either by simple carpool or PNR,
            then choose the best mode in terms of driver's traveling time
        :param: include_direct:
            if True, consider both PNR and simple carpool at the same time
            if False, consider only PNR trips
        :return:
        """
        if include_direct:
            # 0 for simple travel, 1 for PNR travel
            self.choice_matrix = self._generate_choice_matrix(self.cp_matrix, self.cp_pnr_matrix)
            # get shortest traveling time for both trips
            self.cp_matrix_all = np.fmax(self.cp_matrix, self.cp_pnr_matrix)
            self.tt_matrix_all = np.fmin(self.tt_matrix, self.tt_pnr_matrix)
            self.ml_matrix_all = self.ml_matrix.copy()
            self.ml_matrix_all[self.choice_matrix > 0] = self.ml_pnr_matrix[self.choice_matrix > 0]
        else:
            self.cp_matrix_all = self.cp_pnr_matrix.copy()
            self.tt_matrix_all = self.tt_pnr_matrix.copy()
            self.ml_matrix_all = self.ml_pnr_matrix.copy()
            # 0 for simple carpool, 1 for PNR travel
            self.choice_matrix = np.full(self.cp_pnr_matrix.shape, 1, dtype="int8")

        # drive alone as default along diagonal
        np.fill_diagonal(self.cp_matrix_all, self.cp_matrix.diagonal())
        np.fill_diagonal(self.tt_matrix_all, self.tt_matrix.diagonal())
        np.fill_diagonal(self.ml_matrix_all, self.ml_matrix.diagonal())
        np.fill_diagonal(self.choice_matrix, 0)
        if print_mat:
            print("choice matrix head (one mode): {} vs. {}".format(
                (self.choice_matrix == 0).sum(), (self.choice_matrix == 1).sum()
            ))
            print(self.choice_matrix[:8, :8])

    def compute_in_one_step_pnr(
            self, mu1: float = 1.3, mu2: float = 0.1, dst_max: float = 5 * 5280,
            Delta1: float = 15, Delta2: float = 10, Gamma: float = 0.2,  # for depart diff and wait time
            delta: float = 15, gamma: float =1.5, ita: float = 0.8, ita_pnr: float = 0.5,
            include_direct: bool = True, print_mat: bool = True,
    ):
        """
        For the park and ride case, use the same set of filtering parameters for normal case (6 steps).
        :param include_direct:
            if True, consider both PNR and simple carpool at the same time
            if False, consider only PNR trips
        :return:
        """
        # step 0. compute diagonal
        self.compute_diagonal()
        if include_direct:
            self.compute_in_one_step(print_mat=False, skip_combine=True,  # combine modes later
                                     mu1=mu1, mu2=mu2, dst_max=dst_max,
                                     Delta1=Delta1, Delta2=Delta2, Gamma=Gamma,  # for depart diff and wait time
                                     delta=delta, gamma=gamma, ita=ita)
        # step 1. a set of filter based on euclidean distance between coordinates
        # (driver to station)
        self.compute_01_matrix_to_station_p1(threshold_dist=dst_max, mu1=mu1, mu2=mu2,
                                             trips=self.trips, use_mu2=True, print_mat=print_mat)
        if print_mat:
            # print("ml matrix (after PNR pre scan)")
            # print(self.ml_matrix[:10, :10])
            pass
        # step 2 make sure each SOV trip can travel through PNR
        # Note: filter by the passenger's before after traveling time (just as those for passengers)
        #  the above task should be cleverly done in step 2.
        # filter by precise path traveling distance
        self.compute_01_matrix_to_station_p2(delta=15, gamma=1.5)
        if print_mat:
            # print("ml matrix (after PNR pre scan)")
            # print(self.ml_matrix[:10, :10])
            pass
        # step 3. check departure time difference to filter (for all reasonable pnr stations)
        # this may filter out useful matches for PNR mode
        self.compute_depart_01_matrix_pre_pnr(Delta1=Delta1)
        if print_mat:
            # print("ml matrix (after PNR depart check)")
            # print(self.ml_matrix[:10, :10])
            pass
        # step 4. combine all aforementioned filters to generate one big filter
        self.compute_carpoolable_trips_pnr(reset_off_diag=False)
        if print_mat:
            # print("ml matrix (after step 4)")
            # print(self.ml_matrix[:10, :10])
            print("cp_pnr_matrix (after step 4):", self.cp_pnr_matrix.sum())
            print(self.cp_pnr_matrix[:8, :8])
            print("tt_pnr_matrix (after step 4):", self.tt_pnr_matrix.sum())
            print(self.tt_pnr_matrix[:8, :8])
            pass
        # step 5. filter by the maximum waiting time for the driver at pickup location
        self.compute_depart_01_matrix_post_pnr(Delta2=Delta2, Gamma=Gamma)
        if print_mat:
            # print("ml matrix (after step 5)")
            # print(self.ml_matrix[:10, :10])
            print("cp_pnr_matrix (after step 5):", self.cp_pnr_matrix.sum())
            print(self.cp_pnr_matrix[:8, :8])
            print("tt_pnr_matrix (after step 5):", self.tt_pnr_matrix.sum())
            print(self.tt_pnr_matrix[:8, :8])
            pass
        # step 6. filter by real computed waiting time (instead of coordinates before)
        self.compute_reroute_01_matrix_pnr(delta=delta, gamma=gamma, ita_pnr=ita_pnr,
                                           print_mat=print_mat)
        if print_mat:
            # print("ml matrix (after step 6)")
            # print(self.ml_matrix[:10, :10])
            print("cp_pnr_matrix (after step 6):", self.cp_pnr_matrix.sum())
            print(self.cp_pnr_matrix[:8, :8])
        # step 7. combine direct carpool and park and ride carpool
        self.combine_pnr_carpool(include_direct, print_mat=print_mat)
        if print_mat:
            # print("ml matrix (after step 7)")
            # print(self.ml_matrix[:10, :10])
            # print("combined carpool matrix:")
            # print(self.cp_matrix_all)
            # print("cppnr matrix (after step 7)")
            # print(self.cp_pnr_matrix[:10, :10])
            print("cp matrix (after step 7):", self.cp_matrix.sum())
            print(self.cp_matrix[:8, :8])
            print("combined matrix (after step 7):", self.cp_matrix_all.sum())
            print(self.cp_matrix_all[:8, :8])

    def combine_simple_carpool(self, print_mat: bool = True):
        # 0 denotes simple carpool; 1 denotes PNR
        # for DC mode only, all chose to use DC mode
        self.choice_matrix = np.full(self.cp_matrix.shape, 0, dtype="int8")
        self.cp_matrix_all = self.cp_matrix.copy()
        self.tt_matrix_all = self.tt_matrix.copy()
        self.ml_matrix_all = self.ml_matrix.copy()
        if print_mat:
            print("choice matrix head (only simple carpool): {} vs. {}".format(
                (self.choice_matrix == 0).sum(), (self.choice_matrix == 1).sum()
            ))
            print(self.choice_matrix[:8, :8])

    def optimize_in_one_step(
            self,
            rt_bipartite: bool,
            rt_LP: bool,
            verbose: bool,
            plot_all: bool,
            mode: int,
    ):
        lp_summ, lp_summ_ind, bt_summ, bt_summ_ind = None, None, None, None
        if rt_LP:
            self.compute_optimal_lp()
            # tot_count, num_paired, ori_tt, new_tt, ori_ml, new_ml
            g_total_num, g_paired_num, g_ori_tt, new_tt, ori_ml, new_ml, sid = self.evaluate_trips(verbose=verbose)
            lp_summ = pd.DataFrame(
                columns=['tot_num', 'paired_num',
                         'orig_tt', 'opti_tt', 'orig_ml', 'opti_ml'])
            lp_summ.loc[0] = [g_total_num, g_paired_num, g_ori_tt, new_tt, ori_ml, new_ml]
            if mode == 0:
                lp_summ_ind = self.evaluate_individual_trips(verbose=False, use_bipartite=False)
            else:
                lp_summ_ind = self.evaluate_individual_trips_both(verbose=False, use_bipartite=False)
        if rt_bipartite:
            self.compute_optimal_bipartite()
            g_total_num2, g_paired_num2, g_ori_tt2, new_tt2, ori_ml2, new_ml2, sid = self.evaluate_trips(
                verbose=verbose, use_bipartite=True)
            bt_summ = pd.DataFrame(
                columns=['tot_num', 'paired_num',
                         'orig_tt', 'opti_tt', 'orig_ml', 'opti_ml'])
            bt_summ.loc[0] = [g_total_num2, g_paired_num2, g_ori_tt2, new_tt2, ori_ml2, new_ml2]
            if mode == 0:
                bt_summ_ind = self.evaluate_individual_trips(verbose=False, use_bipartite=True)
            else:
                bt_summ_ind = self.evaluate_individual_trips_both(verbose=False, use_bipartite=True)

        if plot_all:
            travel_paths = None
            now_str = self.now_str
            folder_name = os.path.join("data_outputs", now_str)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            if rt_LP:
                for pair in self.result_lst:
                    if pair[0] == pair[1]:
                        continue
                    if bt_summ_ind.iloc[pair[0]]["station"] != -1:
                        fig, ax, travel_paths = self.plot_single_trip_pnr(pair[0], pair[1], trips=bt_summ_ind, fixed_role=True)
                    else:
                        fig, ax, travel_paths = self.plot_single_trip(pair[0], pair[1], fixed_role=True)
                    file_name = "{}_{}.PNG".format(*pair)
                    pn = os.path.join(folder_name, file_name)
                    fig.savefig(pn)
                    plt.clf()
            if rt_bipartite:
                for pair in self.result_lst_bipartite:
                    if pair[0] == pair[1]:
                        # get the travel path for the sov trip
                        paths1 = self.soloPaths[pair[0]]
                        idx1 = self.int2idx[pair[0]]
                        travel_paths = [[idx1, "sov", paths1]]
                        self._save_travel_path(travel_paths, folder_name)
                        continue
                    # print("bt_summ_ind")
                    # print(bt_summ_ind)
                    idx1 = self.int2idx[pair[0]]
                    if "station" in bt_summ_ind.columns and bt_summ_ind.loc[idx1, "station"] != -1:
                        fig, ax, travel_paths = self.plot_single_trip_pnr(
                            pair[0], pair[1], trips=bt_summ_ind, fixed_role=True
                        )
                    else:
                        fig, ax, travel_paths = self.plot_single_trip(
                            pair[0], pair[1], fixed_role=True
                        )
                    ax.get_legend().remove()
                    file_name = "{}_{}.PNG".format(*pair)
                    pn = os.path.join(folder_name, file_name)
                    fig.savefig(pn)
                    plt.clf()

                    if travel_paths is not None:
                        self._save_travel_path(travel_paths, folder_name)
        return lp_summ, lp_summ_ind, bt_summ, bt_summ_ind

    def _save_travel_path(
            self,
            travel_paths: str,
            folder_name: str,
    ) -> None:
        travel_paths = pd.DataFrame(
            travel_paths,
            columns=["person_idx", "role", "travel_path"]
        )
        fn = os.path.join(folder_name, "trip_paths.csv")
        with open(fn, 'a') as f:
            travel_paths.to_csv(
                f,
                index=False, mode='a', header=f.tell() == 0
            )

    def compute_in_one_step(
            self,  print_mat: bool = False,
            mu1: float = 1.5, mu2: float = 0.1, dst_max: float = 5 * 5280,
            Delta1: float = 15, Delta2: float = 10, Gamma: float = 0.2,  # for depart diff and wait time
            delta: float = 15, gamma: float = 1.5, ita: float = 0.5,
            skip_combine: bool = False
    ):
        # step 1. check departure time difference to filter
        self.compute_depart_01_matrix_pre(Delta1=Delta1)
        # step 2. a set of filter based on Euclidean distance between coordinates
        self.compute_pickup_01_matrix(threshold_dist=dst_max, mu1=mu1, mu2=mu2)
        # step 3. compute drive alone cases
        self.compute_diagonal()
        # step 4. combine all aforementioned filters to generate one big filter
        self.compute_carpoolable_trips(reset_off_diag=False)
        if print_mat:
            print("after step 4")
            print("cp matrix:", self.cp_matrix.sum())
        # step 5. filter by the maximum waiting time for the driver at pickup location
        self.compute_depart_01_matrix_post(Delta2=Delta2, Gamma=Gamma)
        # step 6. filter by real computed waiting time (instead of coordinates before)
        self.compute_reroute_01_matrix(delta=delta, gamma=gamma, ita=ita)
        if print_mat:
            print("after step 6")
            print("cp matrix:", self.cp_matrix.sum())
            # print(self.cp_matrix[:8, :8])
            print("tt matrix:", (self.tt_matrix > 0).sum())
            # print(self.tt_matrix[:8, :8])
            print("ml matrix:", (self.ml_matrix > 0).sum())
            # print(self.ml_matrix[:8, :8])
        if skip_combine is False:
            # step 7. just copy matrix value to "combined modes" matrices (for a uniformed computational framework)
            self.combine_simple_carpool(print_mat=print_mat)
        if print_mat:
            # print("ml matrix (after step 7)")
            # print(self.ml_matrix[:10, :10])
            # print("combined carpool matrix:")
            # print(self.cp_matrix_all)
            # print("cppnr matrix (after step 7)")
            # print(self.cp_pnr_matrix[:10, :10])
            print("cp matrix (after step 7):", self.cp_matrix.sum())
            print(self.cp_matrix[:8, :8])
            print("combined matrix (after step 7):", self.cp_matrix_all.sum())
            print(self.cp_matrix_all[:8, :8])
