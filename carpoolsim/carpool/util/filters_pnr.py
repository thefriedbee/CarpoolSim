"""
Filter out the carpool assignments for PNR mode.
"""
import numpy as np

from carpoolsim.carpool.trip_cluster_abstract import TripClusterAbstract


def generate_pnr_trip_map_filt(
    trip_cluster: TripClusterAbstract,
    delta: float = 15,
    gamma: float = 1.5
) -> TripClusterAbstract:
    """
    :param delta: maximum reroute time (in minutes) acceptable for the driver
    :param gamma: the maximum ratio of extra travel time over driver's original travel time

    1. For each accessible PNR for a driver, compute their (arrival time, travel time to PNR)
    2. Store the dictionary of <station id --> (arrival time, travel time to PNR)> to pnr trips matrix
    :return:
    """
    tc = trip_cluster
    trips = tc.trips
    soloTimes = tc.td.soloTimes
    pnr_matrix = tc.pnr_matrix
    pnr_access_info = tc.pnr_access_info
    cp_matrix = tc.cp_matrix
    # return all indicies with value 1
    ind = np.argwhere(pnr_matrix == 1)
    for ind_one in ind:
        trip_id, station_id = ind_one
        tc.compute_pnr_access(trip_id, station_id)
        __, t1, __, t_all, __ = pnr_access_info[trip_id, station_id]
        # get travel alone time
        t2 = soloTimes[trip_id]
        # 2 cases not available for PNR trip
        if (t_all - t2) >= delta or t_all / (t2+0.1) >= gamma:
            pnr_matrix[trip_id, station_id] = 0
            continue
        # store accessible PNR stations information into the list
        # avoid chained assignment warning
        trips_pnr_col = trips['pnr'].copy()
        if trips_pnr_col.iloc[trip_id] is None:
            trips_pnr_col.iloc[trip_id] = [station_id]
        else:
            trips_pnr_col.iloc[trip_id].append(station_id)
        trips['pnr'] = trips_pnr_col

    # finally, prepare the big 0-1 matrix between travelers
    # given that they can share at least one PNR station
    temp_cp_matrix = ((pnr_matrix @ pnr_matrix.T) > 0).astype(np.bool_)
    cp_matrix = (cp_matrix & temp_cp_matrix).astype(np.bool_)
    # np.fill_diagonal(cp_matrix, 1)
    tc.cp_matrix = cp_matrix
    return tc


def compute_depart_01_matrix_pre_pnr(
    trip_cluster: TripClusterAbstract,
    Delta1: float = 15,
    default_rule: bool = False,
) -> TripClusterAbstract:
    """
    For park and ride, the requirements are:
        1. both travelers can choose to use one parking facilities
        2. the arrival time at the station is within the time Delta1
    Use the most relaxed requirement for this, that is:
        (1) The two trip process don't overlap at all (for drive alone case)!!!

    Required: must run self.compute_pnr_access function to compute access time to station
    The code could be slow as many operations are not vectorized

    Two trips are carpool-able through PNR station if:
        (Absolute difference rule: The time difference between two departures are within a fixed time threshold)
        (Alternative rule: Driver departs before rider but no earlier than the given threshold.)
    :param Delta1: depart time different within Delta1 minutes
    :param default_rule: if True, strict time different; if False, absolute time difference
    :return:
    """
    tc = trip_cluster
    trips = tc.trips
    nrow, ncol = tc.nrow, tc.ncol
    cp_matrix = tc.cp_matrix
    # simple method
    depart_lst = np.array(trips['new_min'].tolist()).reshape((1, -1))  # depart minute
    mat = np.tile(depart_lst.transpose(), (1, ncol))
    mat = np.tile(depart_lst, (nrow, 1)) - mat  # depart time difference (driver's depart - pax depart)
    
    # departure time difference should be within Delta1 minutes for both parties
    cp_matrix = (cp_matrix & (np.absolute(mat) <= Delta1)).astype(np.bool_)
    if default_rule:
        # criterion 1. driver should leave earlier than the passenger
        cp_matrix = (cp_matrix & (mat >= 0)).astype(np.bool_)
    tc.cp_matrix = cp_matrix
    return tc


def compute_reroute_01_matrix_pnr(
    trip_cluster: TripClusterAbstract,
    delta: float = 15,
    gamma: float = 1.5,
    ita: float = 0.5,
    print_mat: bool = True
) -> TripClusterAbstract:
    """
    Compute carpool-able matrix considering total reroute time (non-shared trip segments)
    This cannot be estimated before travel time matrix (self.tt_matrix) is fully computed
    :param delta: maximum reroute time (in minutes) acceptable for the driver
    :param gamma: the maximum ratio of extra travel time over driver's original travel time
    :param ita: the ratio between passenger's travel time and driver's travel time should be greater than ita
    :return:
    """
    # unload parameters...
    tc = trip_cluster
    nrow, ncol = tc.nrow, tc.ncol
    tt_matrix = tc.tt_matrix
    tt_matrix_p2 = tc.tt_matrix_p2
    cp_matrix = tc.cp_matrix
    solo_times = tc.td.soloTimes
    # propagate drive alone matrix
    drive_alone_tt = tt_matrix.diagonal().reshape(-1, 1)
    passenger_alone_tt = np.array([solo_times[i] for i in range(ncol)]).reshape(1, -1)
    # condition 1: total reroute time is smaller than threshold minutes
    cp_reroute_matrix1 = ((tt_matrix - np.tile(drive_alone_tt, (1, ncol)))
                            <= delta).astype(bool)
    cp_reroute_matrix2 = ((tt_matrix - np.tile(passenger_alone_tt, (nrow, 1)))
                            <= delta).astype(bool)
    cp_reroute_matrix = cp_reroute_matrix1 & cp_reroute_matrix2
    # condition 2: ratio is smaller than a threshold
    cp_reroute_ratio_matrix1 = ((tt_matrix / np.tile(drive_alone_tt, (1, ncol)))
                                <= gamma).astype(bool)
    cp_reroute_ratio_matrix2 = ((tt_matrix / np.tile(passenger_alone_tt, (nrow, 1)))
                                <= gamma).astype(bool)
    cp_reroute_ratio_matrix = cp_reroute_ratio_matrix1 & cp_reroute_ratio_matrix2
    # condition 3: rider should at least share ita of the total travel time
    cp_time_similarity = ((tt_matrix_p2[:nrow, :ncol] / tt_matrix)
                            >= ita).astype(bool)
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
    cp_matrix = (cp_matrix &
                 cp_reroute_matrix &
                 cp_reroute_ratio_matrix &
                 cp_time_similarity).astype(bool)
    # need to mask tt and ml matrix at un-carpoolable positions also
    tc.tt_matrix[cp_matrix == 0] = np.nan
    tc.ml_matrix[cp_matrix == 0] = np.nan
    tc.cp_matrix = cp_matrix
    return tc


def compute_depart_01_matrix_post_pnr(
    trip_cluster: TripClusterAbstract,
    Delta2: float = 10,
    Gamma: float = 0.2,
    default_rule: bool = True,
)-> TripClusterAbstract:
    """
    Filter PNR trips considering driver's waiting time is limited
    :param Delta2: the driver's maximum waiting time
    :param Gamma: the maximum ratio of waiting time over passenger's travel time
    :param default_rule:
        if True, strict time different (applies for driver)
        if False, absolute time difference (applies for both travelers)
    :return:
    """
    # unload parameters...
    tc = trip_cluster
    nrow, ncol = tc.nrow, tc.ncol
    trips = tc.trips
    cp_matrix = tc.cp_matrix
    solo_times = tc.td.soloTimes
    pnr_access_info = tc.pnr_access_info
    # step 1. get the travel time to each feasible station
    ind = np.argwhere(cp_matrix == 1)
    # pnr depart time matrix (diff between drivers)
    mat = np.full((nrow, ncol), np.nan)
    for ind_one in ind:
        trip_id1, trip_id2 = ind_one
        # get station id they share
        trip_row1 = trips.iloc[trip_id1]
        trip_row2 = trips.iloc[trip_id2]
        # check the "best" PNR station for two travlers to meet
        sid = tc._check_trips_best_pnr(trip_row1, trip_row2, trip_id1, trip_id2)
        if sid is None:
            continue
        # access information (path, time, distance)
        info1 = pnr_access_info[trip_id1, sid]
        info2 = pnr_access_info[trip_id2, sid]
        # info1[1] is the access time to PNR station
        t1 = trips['new_min'].iloc[trip_id1] + info1[1]  # arrival time at pnr for person 1
        t2 = trips['new_min'].iloc[trip_id2] + info2[1]  # arrival time at pnr for person 2
        # the number of minutes it takes for passenger to wait for the driver
        mat[trip_id1][trip_id2] = t1 - t2
        mat[trip_id2][trip_id1] = t2 - t1

    passenger_time = np.array([solo_times[i] for i in range(ncol)]).reshape(1, -1)
    passenger_time = np.tile(passenger_time, (nrow, 1))
    # waiting time of the driver should be less than Gamma times of the passenger's travel time
    cp_matrix = (cp_matrix & 
                 (np.absolute(mat/passenger_time) <= Gamma) &
                 (np.absolute(mat) <= Delta2)).astype(np.bool_)
    
    if default_rule:
        # passenger only waits the driver, not the other way around
        cp_matrix = (cp_matrix & (mat >= 0)).astype(np.bool_)
    tc.cp_matrix = cp_matrix
    return tc
