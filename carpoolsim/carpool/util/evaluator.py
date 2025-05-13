"""
Move all functions evaluating the carpool assignments results.
"""
import pandas as pd

from carpoolsim.carpool.trip_cluster_abstract import TripClusterAbstract
from carpoolsim.config import CPMode


def evaluate_individual_trips_sim(
    sim_task,
) -> pd.DataFrame:
    """
    After getting optimized results, expand the trip column with before after information for each person.
    This code works for general cases.
    :return:
    """
    # unload data
    st = sim_task
    td = st.trip_demands
    trips = td.trips
    soloTimes = td.soloTimes
    soloDists = td.soloDists
    mc_matrix = st.mc_matrix  # mode choice matrix
    tt_matrix = st.tt_matrix
    ml_matrix = st.ml_matrix
    cp_pnr = st.cp_pnr
    paired_lst = st.paired_lst
    int2idx = td.int2idx
    pnr_access_info = st.pnr_access_info

    # trick: individual trip evaluation results are stored in a dataframe
    #   with the same number of rows as the original trips
    trip_summary_df = trips[['new_min']].copy()
    trip_summary_df = trip_summary_df.assign(
        **{'before_time': 0.0, 'before_dist': 0.0,
           'after_time': 0.0, 'after_dist': 0.0,
           'MODE': CPMode.SOV.value, 'as_passenger': False, 
           'partner_idx': 0, 'station': -1})
    
    # for each traveler, find its SOV trip time/distances,
    # then find the optimized trip information
    for d, p in paired_lst:
        d_idx, p_idx = int2idx[d], int2idx[p]
        role_cols = ['MODE', 'as_passenger', 'partner_idx']
        info_cols = ['before_time', 'before_dist', 'after_time', 'after_dist', 'station']
        if d == p:  # drive alone (SOV)
            row_d = [soloTimes[d], soloDists[d], soloTimes[d], soloDists[d], -1]
            trip_summary_df.loc[d_idx, info_cols] = row_d
            trip_summary_df.loc[d_idx, role_cols] = [CPMode.SOV.value, False, d_idx]
            continue
        
        # carpool case
        if mc_matrix[d, p] == CPMode.DC.value:  # DC mode
            mode = CPMode.DC
            row_d = [soloTimes[d], soloDists[d], tt_matrix[d, p], ml_matrix[d, p], -1]
            row_p = [soloTimes[p], soloDists[p], soloTimes[p], soloDists[p], -1]
        elif mc_matrix[d, p] == CPMode.PNR.value:  # PNR mode
            sid = cp_pnr[d, p]
            mode = CPMode.PNR
            pnr_pass_time, pnr_pass_dist = pnr_access_info[d, sid][3:5]
            # print(f"d:{d}, p:{p}, sid:{sid}") 
            # print(f"driver: cp: {cp_matrix[d, p]}, tt:{tt_matrix[d, p]:.2f}, ml:{ml_matrix[d, p]:.2f}, time:{pnr_pass_time:.2f}, dist:{pnr_pass_dist:.2f}")
            row_d = [soloTimes[d], soloDists[d], pnr_pass_time, pnr_pass_dist, sid]
            pnr_pass_time, pnr_pass_dist = pnr_access_info[p, sid][3:5]
            # print(f"passenger: cp: {cp_matrix[p, d]}, tt:{tt_matrix[p, d]:.2f}, ml:{ml_matrix[p, d]:.2f}, time:{pnr_pass_time:.2f}, dist:{pnr_pass_dist:.2f}")
            row_p = [soloTimes[p], soloDists[p], pnr_pass_time, pnr_pass_dist, sid]
        else:
            raise ValueError(f"Invalid mode choice matrix: {mc_matrix[d, p]}")
        row_d = [round(r, 2) for r in row_d]
        row_p = [round(r, 2) for r in row_p]        

        # for passenger, travel is same as before
        trip_summary_df.loc[d_idx, info_cols] = row_d
        trip_summary_df.loc[p_idx, info_cols] = row_p
        trip_summary_df.loc[d_idx, role_cols] = [mode.value, False, p_idx]
        trip_summary_df.loc[p_idx, role_cols] = [mode.value, True, d_idx]
    
    return trip_summary_df


def summarize_results(
    trip_summary_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.Series:
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
    # unload data
    num_travelers = trip_summary_df.shape[0]
    ori_tt = sum(trip_summary_df['before_time'])
    ori_ml = sum(trip_summary_df['before_dist'])
    new_tt = sum(trip_summary_df['after_time'])
    new_ml = sum(trip_summary_df['after_dist'])
    filt = trip_summary_df['MODE'] != CPMode.SOV.value
    num_paired = len(trip_summary_df[filt])

    if verbose:
        print(f"{num_paired} persons found carpooling in a cluster with {num_travelers} persons")
        print_str = f"Before: Total VTT is {ori_tt:.2f} veh-min;\n"
        print_str += f"After: Total VTT is {new_tt:.2f} veh-min;\n"
        print_str += f"Before: Total VTM is {ori_ml:.2f} miles;\n"
        print_str += f"After: Total VTM is {new_ml:.2f} miles."
        print(print_str)
    col_names = ["num_travelers", "num_paired", "ori_tt", "new_tt", "ori_ml", "new_ml"]
    return pd.Series([num_travelers, num_paired, ori_tt, new_tt, ori_ml, new_ml], index=col_names)
