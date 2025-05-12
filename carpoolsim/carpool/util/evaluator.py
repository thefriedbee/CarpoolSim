"""
Move all functions evaluating the carpool assignments results.
"""
import pandas as pd
from carpoolsim.carpool.trip_cluster_abstract import TripClusterAbstract


def evaluate_trips(
    trip_cluster: TripClusterAbstract,
    verbose: bool = False,
) -> tuple[int, int, float, float, float, float, int]:
    """
    Evaluate the assignment's performances at the aggregated level. Interested in:
    (1) The changes in total vehicular travel time
    (2) The changes in total travel time
    (3) Number of carpool pairs matched

    The extra travel time solely contributed by driver reroutes to pick up passengers.
    The first metric measures systemic benefits; The second measures driver's sacrifices.
    For future case, more statistics will be developed to measure extreme cases.

    If row not equals to column (in inherited classes), then only report the results of the left square matrix!!!
    :return:
    """
    tc = trip_cluster
    num_paired, paired_lst = tc.num_paired, tc.paired_lst
    num_travelers = tc.nrow
    # SOV results (before considering carpooling)
    ori_tt = sum(tc.soloTimes[p] for p in range(num_travelers))
    ori_ml = sum(tc.soloDists[p] for p in range(num_travelers))

    # Carpool results (after considering carpooling)
    def get_cp_members(paired_lst):
        cp_members = []
        for pr in paired_lst:
            if pr[0] != pr[1]:
                cp_members += [pr[0], pr[1]]
        return cp_members
    cp_members = get_cp_members(paired_lst)

    def get_after_tt_ml(num_travelers, cp_members):
        new_tt, new_ml = 0, 0
        for p in range(num_travelers):
            if p in cp_members:
                new_tt += tc.tt_matrix[p]
                new_ml += tc.ml_matrix[p]
            else:
                new_tt += tc.soloTimes[p]
                new_ml += tc.soloDists[p]
        return new_tt, new_ml
    new_tt, new_ml = get_after_tt_ml(num_travelers, cp_members)

    if verbose:
        print(f"{num_paired} persons found carpooling in a cluster with {num_travelers} persons")
        print_str = f"Before: Total VTT is {ori_tt:.2f} veh-min;\n"
        print_str += f"After: Total VTT is {new_tt:.2f} veh-min;\n"
        print_str += f"Before: Total VTM is {ori_ml:.2f} miles;\n"
        print_str += f"After: Total VTM is {new_ml:.2f} miles."
        print(print_str)
    return num_travelers, num_paired, ori_tt, new_tt, ori_ml, new_ml


def evaluate_individual_trips_pnr(
    trip_cluster: TripClusterAbstract,
) -> pd.DataFrame:
    """
    After getting optimized results, record before-after traveling plan of each traveler.
    This code works for general cases.
    :return:
    """
    tc = trip_cluster
    trips = tc.trips

    # trick: individual trip evaluation results are stored in a dataframe
    #   with the same number of rows as the original trips
    trip_summary_df = trips[['new_min']].copy()
    trip_summary_df = trip_summary_df.assign(
        **{'before_time': 0.0, 'before_dist': 0.0,
           'after_time': 0.0, 'after_dist': 0.0,
           'SOV': True, 'as_passenger': False,
           'partner_idx': 0, 'station': -1})
    
    # for each traveler, find its SOV trip time/distances,
    # then find the optimized trip information
    temp_records = tc.paired_lst
    index_paired = []
    # evaluate for each driver-passenger pair (assigned carpool trip)
    for d, p in temp_records:
        d_idx, p_idx = tc.int2idx[d], tc.int2idx[p]
        role_cols = ['SOV', 'as_passenger', 'partner_idx']
        info_cols = ['before_time', 'before_dist', 'after_time', 'after_dist', 'station']
        if d == p:  # SOV trips
            row_d = [tc.soloTimes[p], tc.soloDists[p],
                     tc.soloTimes[p], tc.soloDists[p], -1]
            trip_summary_df.loc[
                d_idx,
                ['before_time', 'before_dist', 'after_time', 'after_dist', 'station']
            ] = row_d
            trip_summary_df.loc[d_idx, role_cols] = [True, False, d]
            index_paired.append(d)
            continue
        
        # evaluate carpool trips
        trip1, trip2 = trips.iloc[d], trips.iloc[p]
        sid = tc.cp_pnr[d, p]

        # this is strictly the driver's time
        row_d = [tc.soloTimes[d], tc.soloDists[d],
                 tc.tt_matrix[d, p], tc.ml_matrix[d, p], sid]
        row_p = [tc.soloTimes[p], tc.soloDists[p],
                 tc.soloTimes[p], tc.soloDists[p], sid]
        row_d = [round(r, 3) for r in row_d]
        row_p = [round(r, 3) for r in row_p]

        trip_summary_df.loc[d_idx, info_cols] = row_d
        trip_summary_df.loc[p_idx, info_cols] = row_p
        # finally, update role info
        trip_summary_df.loc[d_idx, role_cols] = [False, False, p]
        trip_summary_df.loc[p_idx, role_cols] = [False, True, d]
        index_paired.append(p)
        index_paired.append(d)
    trip_summary_df = trip_summary_df.loc[index_paired, :]
    return trip_summary_df


def evaluate_individual_trips_sov(
    trip_cluster: TripClusterAbstract,
) -> pd.DataFrame:
    """
    After getting optimized results, expand the trip column with before after information for each person.
    This code works for general cases.
    :return:
    """
    tc = trip_cluster
    trips = tc.trips

    # trick: individual trip evaluation results are stored in a dataframe
    #   with the same number of rows as the original trips
    trip_summary_df = trips[['new_min']].copy()
    trip_summary_df = trip_summary_df.assign(
        **{'before_time': 0.0, 'before_dist': 0.0,
           'after_time': 0.0, 'after_dist': 0.0,
           'SOV': True, 'as_passenger': False, 'partner_idx': 0})
    
    # for each traveler, find its SOV trip time/distances,
    # then find the optimized trip information
    temp_records = tc.paired_lst

    index_paired = []
    for d, p in temp_records:
        d_idx, p_idx = tc.int2idx[d], tc.int2idx[p]
        role_cols = ['SOV', 'as_passenger', 'partner_idx']
        info_cols = ['before_time', 'before_dist', 'after_time', 'after_dist', 'station']
        if d == p:  # drive alone (SOV)
            row_d = [tc.soloTimes[d], tc.soloDists[d], tc.soloTimes[d], tc.soloDists[d], -1]
            trip_summary_df.loc[d_idx, info_cols] = row_d
            trip_summary_df.loc[d_idx, role_cols] = [True, False, d]
            index_paired.append(d)
            continue
        
        # carpool case
        row_d = [tc.soloTimes[d], tc.soloDists[d],
                 tc.tt_matrix[d, p], tc.ml_matrix[d, p], -1]
        row_d = [round(r, 3) for r in row_d]
        row_p = [tc.soloTimes[p], tc.soloDists[p],
                 tc.soloTimes[p], tc.soloDists[p], -1]
        row_p = [round(r, 3) for r in row_p]        

        # for passenger, travel is same as before
        trip_summary_df.loc[p_idx, info_cols] = row_p
        # finally, update role info
        trip_summary_df.loc[d_idx, role_cols] = [False, False, p]
        trip_summary_df.loc[p_idx, role_cols] = [False, True, d]
        index_paired.append(p)
        index_paired.append(d)
    
    trip_summary_df = trip_summary_df.loc[index_paired, :]
    return trip_summary_df


