"""
Move all functions evaluating the carpool assignments results.
"""
import pandas as pd
from carpoolsim.carpool.trip_cluster_abstract import TripClusterAbstract


def evaluate_trips(
    trip_cluster: TripClusterAbstract,
    verbose: bool = False,
    use_bipartite: bool = False
) -> tuple[int, int, float, float, float, float, int]:
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
    tc = trip_cluster
    nrow, ncol = tc.nrow, tc.ncol

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
        tot_count, num_paired, party_lst = count_parties(tc.result_lst_bipartite)
        rl = tc.result_lst_bipartite
    else:
        tot_count, num_paired, party_lst = count_parties(tc.result_lst)
        rl = tc.result_lst
    # special case: If num of columns is greater than rows, then self pairs in the right part should
    # not be concluded in the evaluation
    ori_tt = sum(tc.soloTimes[p] for p in party_lst)
    ori_ml = sum(tc.soloDists[p] for p in party_lst)
    new_tt, new_ml = 0, 0
    sid = None  # PNR station id
    for p in rl:
        if tc.choice_matrix[p] == 0:
            # print('simple:', p, self.tt_matrix[p], self.ml_matrix[p])
            new_tt += tc.tt_matrix[p]
            new_ml += tc.ml_matrix[p]
        elif tc.choice_matrix[p] == 1:
            # print('shared:', p, self.tt_pnr_matrix[p], self.ml_pnr_matrix[p])
            new_tt += tc.tt_pnr_matrix[p]
            new_ml += tc.ml_pnr_matrix[p]  # + self.ml_pnr_matrix_p[p]
            trip1, trip2 = tc.trips.iloc[p[0], :], tc.trips.iloc[p[1], :]
            sid = tc._check_trips_best_pnr(trip1, trip2, p[0], p[1])
        else:  # no assign, drive alone
            # print('drive alone')
            new_tt += tc.tt_pnr_matrix[p]
            new_ml += tc.ml_pnr_matrix[p]  # + self.ml_pnr_matrix_p[p]
    if verbose:
        print("{} persons found carpooling in a cluster with {} persons".format(num_paired, tot_count))
        print_str = "Original total vehicular travel time is {} veh-min;\n"
        print_str += "New total vehicular travel time is {} veh-min "
        print(print_str.format(round(ori_tt, 2), round(new_tt, 2)))

        print_str = "Original total vehicular travel mileage is {} miles;\n"
        print_str += "New total vehicular travel mileage is {} miles."
        print(print_str.format(round(ori_ml, 2), round(new_ml, 2)))

    return tot_count, num_paired, ori_tt, new_tt, ori_ml, new_ml, sid


def evaluate_individual_trips_both(
    trip_cluster: TripClusterAbstract,
    verbose: bool = False,
    use_bipartite: bool = False,
    trips: pd.DataFrame | None = None,
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
    tc = trip_cluster
    if trips is None:
        trips = tc.trips
    trip_summary_df = trips[['new_min']].copy()
    # init new columns (before/after travel time and distances)
    # values are placeholder
    trip_summary_df = trip_summary_df.assign(
        **{'before_time': 0.0, 'before_dist': 0.0,
            'after_time': 0.0, 'after_dist': 0.0,
            'SOV': True, 'as_passenger': False,
            'partner_idx': 0, 'station': -1})
    # for each traveler, find its SOV trip time/distances,
    # then find the optimized trip information
    if use_bipartite is False:
        temp_records = tc.result_lst
    else:
        temp_records = tc.result_lst_bipartite

    index_paired = []
    for d_idx, p_idx in temp_records:
        d, p = tc.int2idx[d_idx], tc.int2idx[p_idx]
        choice = tc.choice_matrix[d_idx][p_idx]
        sid = -1
        if choice == 1:
            trip1, trip2 = trips.loc[d], trips.loc[p]
            sid = tc._check_trips_best_pnr(trip1, trip2, d_idx, p_idx)
            # note: need to select mode based on the choice matrix

        # this is strictly the driver's time
        row_d = [
            tc.soloTimes[d_idx], tc.soloDists[d_idx],
            tc.tt_matrix_all[d_idx, p_idx], tc.ml_matrix_all[d_idx, p_idx],
            sid
        ]

        # passenger costs assumed a fixed number
        # (need to recalculate passenger's new travel time in post analysis)
        row_p = [tc.soloTimes[p_idx], tc.soloDists[p_idx],
                 tc.soloTimes[p_idx], tc.soloDists[p_idx], sid]

        row_d = [round(r, 3) for r in row_d]
        row_p = [round(r, 3) for r in row_p]
        if verbose:
            print(d_idx, p_idx)
            print(row_d)

        trip_summary_df.loc[
            d,
            ['before_time', 'before_dist', 'after_time', 'after_dist', 'station']
        ] = row_d

        if p_idx != d_idx:  # carpool, not SOV in after case
            # for passenger, travel is same as before
            if verbose:
                print(row_p)
            trip_summary_df.loc[
                p,
                ['before_time', 'before_dist', 'after_time', 'after_dist', 'station']
            ] = row_p

            # finally, update role info
            trip_summary_df.loc[d, ['SOV', 'as_passenger', 'partner_idx']] = [False, False, p]
            trip_summary_df.loc[p, ['SOV', 'as_passenger', 'partner_idx']] = [False, True, d]
            index_paired.append(p)
            index_paired.append(d)
        else:  # drive alone (SOV)
            # partner is the driver herself
            trip_summary_df.loc[d, ['SOV', 'as_passenger', 'partner_idx']] = [True, False, d]
            index_paired.append(p)
    if verbose:
        print('LP output results are updated.')
    trip_summary_df = trip_summary_df.loc[index_paired, :]
    return trip_summary_df


def evaluate_individual_trips(
    trip_cluster: TripClusterAbstract,
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
    tc = trip_cluster
    nrow, ncol = tc.nrow, tc.ncol
    # left_trips_filt = self.trips.SOV & (self.trips.new_min > self.t0 + self.epsilon)
    # dropped_indexes = self.trips.loc[~left_trips_filt, :].index.tolist()
    trip_summary_df = tc.trips[['new_min']].copy()

    # init new columns (before/after travel time and distances)
    # values are placeholder
    trip_summary_df = trip_summary_df.assign(
        **{'before_time': 0.0, 'before_dist': 0.0,
            'after_time': 0.0, 'after_dist': 0.0,
            'SOV': True, 'as_passenger': False, 'partner_idx': 0})
    # downcast to save memory
    trip_summary_df = trip_summary_df.astype(
        {"before_time": "float32", "before_dist": "float32",
            'after_time': "float32", 'after_dist': "float32",
            'partner_idx': "int32"})
    # for each traveler, find its SOV trip time/distances,
    # then find the optimized trip information
    if use_bipartite is False:
        temp_records = tc.result_lst
    else:
        temp_records = tc.result_lst_bipartite
    # print('assignment list is:', temp_records)

    index_paired = []
    # index_int_paired = []
    for d_idx, p_idx in temp_records:
        d, p = tc.int2idx[d_idx], tc.int2idx[p_idx]
        row_d = [tc.soloTimes[d_idx], tc.soloDists[d_idx],
                 tc.tt_matrix[d_idx, p_idx], tc.ml_matrix[d_idx, p_idx]]
        row_d = [round(r, 3) for r in row_d]

        row_p = [tc.soloTimes[p_idx], tc.soloDists[p_idx],
                 tc.soloTimes[p_idx], tc.soloDists[p_idx]]
        row_p = [round(r, 3) for r in row_p]

        trip_summary_df.loc[
            d,
            ['before_time', 'before_dist', 'after_time', 'after_dist']
        ] = row_d

        if p_idx != d_idx:  # carpool, not SOV in after case
            # for passenger, travel is same as before
            if verbose:
                # print(row_p)
                pass
            trip_summary_df.loc[
                p,
                ['before_time', 'before_dist', 'after_time', 'after_dist']
            ] = row_p

            # finally, update role info
            trip_summary_df.loc[d, ['SOV', 'as_passenger', 'partner_idx']] = [False, False, p]
            trip_summary_df.loc[p, ['SOV', 'as_passenger', 'partner_idx']] = [False, True, d]
            index_paired.append(p)
            index_paired.append(d)
            # index_int_paired.append(p_idx)
            # index_int_paired.append(d_idx)
        else:  # drive alone (SOV)
            # partner is the driver herself
            trip_summary_df.loc[d, ['SOV', 'as_passenger', 'partner_idx']] = [True, False, d]
            index_paired.append(p)
    trip_summary_df = trip_summary_df.loc[index_paired, :]
    return trip_summary_df


