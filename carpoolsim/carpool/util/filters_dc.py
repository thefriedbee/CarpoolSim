"""
Filter out the carpool assignments for DC mode.
"""
import numpy as np
from carpoolsim.carpool.trip_cluster_abstract import TripClusterAbstract


def compute_depart_01_matrix_pre(
    trip_cluster: TripClusterAbstract,
    Delta1: float = 15,
    default_rule: bool = True,
) -> TripClusterAbstract:
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
    tc = trip_cluster
    trips = tc.trips
    # step 1. Measure depart time difference within threshold time (default is 15 minutes)
    nrow, ncol = tc.nrow, tc.ncol
    depart_lst = np.array(trips['new_min'].tolist()).reshape((1, -1))  # depart minute
    cp_matrix = tc.cp_matrix
    # compare departure time difference
    mat = np.tile(depart_lst.transpose(), (1, ncol))
    mat = np.tile(depart_lst, (nrow, 1)) - mat  # depart time difference (driver's depart - pax depart)
    cp_matrix = (cp_matrix &
                 (np.absolute(mat) <= Delta1)).astype(np.bool_)
    # default rule: driver should depart earlier (no later) than passenger
    if default_rule:
        cp_matrix = cp_matrix & (mat >= 0)
    tc.cp_matrix = cp_matrix
    return tc


def compute_reroute_01_matrix(
    trip_cluster: TripClusterAbstract,
    delta: float = 15,
    gamma: float = 1.5,
    ita: float = 0.5,
) -> TripClusterAbstract:
    """
    Compute carpool-able matrix considering total reroute time (non-shared trip segments)
    This cannot be estimated before travel time matrix (self.tt_matrix) is fully computed
    :param ita: passenger's travel time should be greater than ita (minutes) compared to that of drivers
    :param delta: maximum reroute time (in minutes) acceptable for the driver
    :param gamma: the maximum ratio of extra travel time over driver's original travel time
    :return:
    """
    # unload parameters...
    tc = trip_cluster
    nrow, ncol = tc.nrow, tc.ncol
    soloTimes = tc.td.soloTimes
    tt_matrix = tc.tt_matrix
    cp_matrix = tc.cp_matrix
    ml_matrix = tc.ml_matrix
    # propagate drive alone matrix
    # drive_alone_tt = self.tt_matrix.diagonal().reshape(-1, 1)
    drive_alone_tt = np.array([soloTimes[i] for i in range(nrow)]).reshape(-1, 1)
    # condition 1: total reroute time is smaller than threshold minutes
    cp_reroute_matrix = ((tt_matrix - np.tile(drive_alone_tt, (1, ncol))) <= delta).astype(int)
    # condition 2: ratio is smaller than a threshold
    cp_reroute_ratio_matrix = ((tt_matrix / np.tile(drive_alone_tt, (1, ncol))) <= gamma).astype(int)
    # condition 3: rider should at least share 50% of the total travel time
    passenger_time = np.tile(drive_alone_tt.reshape(1, -1), (nrow, 1))
    cp_time_similarity = ((passenger_time / tt_matrix) >= ita).astype(np.bool_)
    # condition 4: after drop-off time / whole travel time?
    # self.cp_reroute_matrix = cp_reroute_matrix # TODO: delete this row when not needed
    cp_matrix = (cp_matrix &
                 cp_reroute_matrix &
                 cp_reroute_ratio_matrix &
                 cp_time_similarity).astype(np.bool_)
    # need to mask tt and ml matrix
    tt_matrix[cp_matrix == 0] = np.nan
    ml_matrix[cp_matrix == 0] = np.nan

    # update information
    tc.cp_matrix = cp_matrix
    tc.tt_matrix = tt_matrix
    tc.ml_matrix = ml_matrix
    return tc


def compute_depart_01_matrix_post(
    trip_cluster: TripClusterAbstract,
    Delta2: float = 10,
    Gamma: float = 0.2,
    default_rule: bool = True,
) -> TripClusterAbstract:
    """
    After tt_matrix_p1 is computed, filter by maximum waiting time for the driver at pickup location
    :param Delta2: driver's maximum waiting time
    :param default_rule: if True, strict time different; if False, absolute time difference
    :return:
    """
    # step 2. Maximum waiting time for driver is Delta2 (default is 5 minutes)
    # unload parameters...
    tc = trip_cluster
    nrow, ncol = tc.nrow, tc.ncol
    trips = tc.trips
    soloTimes = tc.soloTimes
    tt_matrix_p1 = tc.tt_matrix_p1
    cp_matrix = tc.cp_matrix
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
    # update information
    tc.cp_matrix = cp_matrix
    return tc
