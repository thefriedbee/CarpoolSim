import numpy as np
import pandas as pd
import os
import graph_tool.flow as gt
from graph_tool.all import *


# find maximal Bipartite matching for general graph (given feasibility matrix of any shape)
class CarpoolBipartite:
    def __init__(self, cp_matrix, tt_matrix):
        """
        Init graph & shape. Notice that driver and passenger with same integer index are the same person.
        :param cp_matrix: carpool-able matrix (0-1)
        :param tt_matrix: travel time matrix. Used to integrate greedy algorithm
        """
        # need to mask the diagonal for running bipartite
        self.cp_matrix = cp_matrix.copy().astype("int8")
        np.fill_diagonal(self.cp_matrix, 0)
        self.tt_matrix = tt_matrix
        self.driver, self.passenger = self.cp_matrix.shape

    def dfs_same_source(self, u, matchR, seen):
        """
        A dfs based recursive function that returns true if a matching from vertex u is possible.
        (u ==> v)
        :param u: integer id of row index for drivers
        :param matchR: a reversal dictionary records which driver u points to passenger v.
        If there is no assigned driver for the passenger, then denote it as '-1'
        :param seen: a dictionary records whether a passenger v
        :return:
        """
        # try every job one by one
        for v in range(self.passenger):
            # If driver u can pick up passenger v and v is not seen
            if self.cp_matrix[u][v] and seen[v] is False:
                seen[v] = True
                # a recursive structure
                """
                If passenger 'v' is not assigned to any driver before OR 
                previously the passenger is previously assigned to driver u1 (== matchR[v]).
                The driver u1 has an alternate passenger available. 
                In recursion, since v is marked as visited in the above line, matchR[v] in the following 
                recursive call will not get job 'v' again. In this way, it is DFS search.
                """
                if matchR[v] == -1 or self.dfs_same_source(matchR[v], matchR, seen):
                    matchR[v] = u
                    return True
        return False

    def solve_bipartite(self):
        # init trace back item
        match_reverse = [-1] * self.passenger
        # Count of passengers assigned to drivers
        count_pairs = 0
        for i in range(self.driver):
            # all passenger not seen for the driver
            seen = [False] * self.passenger
            # Find if the driver 'u' can get a passenger
            if self.dfs_same_source(i, match_reverse, seen):
                count_pairs += 1
        # return counts of matched pairs, matrix showing matched pairs, traceback
        return count_pairs, match_reverse

    def get_chain_loop(self, u, match_reverse, match_forward, detected_lst):
        """
        Given the person u, get the chain it contained in using two pointers to opposite directions.
        :param u: the integer index of person u
        :param match_reverse: map relation from passenger v to driver u.
        If value is -1, then u cannot be passenger to others.
        :param match_forward: map relation from
        :param detected_lst: a list record if a person is grouped into a set. If so, which set
        (set name as a person's index in the set)
        :return:
        """
        # print(u, match_forward[u], detected_lst[u], detected_lst[-1])
        p_forward, p_reverse = u, u
        # corner case: self loop
        if match_forward[u] == u:
            # print('the first driver (self loop)', u)
            detected_lst[u] = u
            return u
        # corner case: passenger without driving option
        # if match_reverse[u] == -1:
        #     return None
        # check if person u is ALREADY in a named set
        if detected_lst[u] != -1:
            return None

        is_turn1 = True
        last_p_reverse = u
        last_p_forward = u
        # print(p_reverse, p_forward, is_turn1)
        while True:
            # lr, lf = p_reverse, p_forward
            # if no driver found or out of scope, no need to continue
            # otherwise, try find the next driver in chain.
            if is_turn1:
                if p_reverse != -1:
                    last_p_reverse = p_reverse  # will record the last real driver in chain instead of -1
                    if p_reverse < len(match_reverse):
                        p_reverse = match_reverse[p_reverse]
                    else:
                        p_reverse = -1
            else:
                if p_forward != -1:
                    last_p_forward = p_forward  # will record the last real passenger in chain instead of -1
                    if p_forward < len(match_forward):
                        p_forward = match_forward[p_forward]
                    else:
                        p_forward = -1
            # if two pointers all points to -1 or meet at a midpoint
            # or (p_reverse == last_p_reverse and p_forward == last_p_forward)
            # print(p_reverse, p_forward, is_turn1, last_p_reverse, last_p_forward)
            # assign persons found in the chain a set name as u
            detected_lst[last_p_reverse] = u
            detected_lst[last_p_forward] = u
            is_turn1 = not is_turn1
            # print('reverse = {}, last_reverse = {}, forward = {}, last_forward = {}'.format(
            #     p_reverse, last_p_reverse, p_forward, last_p_forward))
            if p_reverse == p_forward == -1 or p_reverse == p_forward:
                # reach -1 on both ends of the chain or two pointers meet at one point of a loop
                break

        # reset the set names to the "first" driver's name
        # print('the first driver', last_p_reverse)
        set_name = last_p_reverse
        p = last_p_reverse
        while p != -1 and p < len(match_forward):
            # print(p)
            last_p = p
            p = match_forward[p]
            detected_lst[p] = set_name
            if p == set_name or last_p == p:  # back to loop or reach at end
                break
        return set_name

    def drop_even_matches(self, set_name, match_forward):
        """
        Special case: If num of columns is greater than num of rows,
        then self pairs in the right part should not be concluded in the evaluation!
        :param set_name: set name (as the first driver in chain)
        :param match_forward:
        :return:
        """
        # corner case: if itself forms a loop, return itself
        if match_forward[set_name] == set_name:
            # if it is an isolated passenger, return empty list
            if int(set_name) > self.driver:
                return []
            return [(set_name, set_name)]
        # other cases are long chain or loop
        # print('set name', set_name)
        return_matches = []
        p = set_name
        is_odd = True
        while p != -1 and p < len(match_forward):
            p_next = match_forward[p]
            if is_odd:
                if p_next != -1:
                    return_matches.append((p, p_next))
                if p_next == -1:
                    return_matches.append((p, p))
                    break
            is_odd = not is_odd
            p = p_next
            # break long loop (back to the last -1)
            if p == set_name:
                last_p, final_p = return_matches[-1]
                if final_p == p:  # need to solve conflict
                    return_matches = return_matches[:-1]
                    if p <= self.driver:
                        return_matches.append((last_p, last_p))
                break
        # print('bipartite matches', return_matches)
        return return_matches

    def solve_bipartite_conflicts_naive(self, use_graph_tool=False):
        """
        Solve role conflicts (one cannot be passenger and driver at the same time)
        1. find all matching chains/loops
        2. For each chain, delete about half of the matching until there are no conflicts
        :return: num of matches and a list of matches
        """
        if not use_graph_tool:
            count_pairs, match_reverse = self.solve_bipartite()
        else:
            count_pairs, match_reverse = self.solve_bipartite_gt()
            # match_reverse = [u if u != -1 else v for v, u in enumerate(match_reverse)]
        match_forward = [-1] * self.driver  # default is drive alone
        for v, u in enumerate(match_reverse):
            if u != -1:
                match_forward[u] = v
        # find isolated trips and set to self loops. In other words,
        # if one has neither role as passenger and as driver, set to drive alone.
        for i in range(max(self.driver, self.passenger)):
            if i < self.driver and i < self.passenger:
                # if person i is an isolated trip, set to self loop
                if match_reverse[i] == -1 and match_forward[i] == -1:
                    match_reverse[i], match_forward[i] = i, i
            # otherwise, only the drivers can drive, passenger without role of driver
            # cannot drive alone
            elif i >= self.driver:  # i can only be passenger,
                if match_reverse[i] == -1:  # no suitable driver
                    # match_reverse[i] = i
                    pass
            elif i >= self.passenger:
                if match_forward[i] == -1:
                    match_forward[i] = i
        # print('reverse match', match_reverse)
        # print('forward match', match_forward)
        # notice the length equals to all persons
        detected_lst = [-1] * max(self.driver, self.passenger)
        # efficient code to explore graph into several sets
        set_names = []
        for u in range(self.driver):
            i = self.get_chain_loop(u, match_reverse, match_forward, detected_lst)
            if i is not None:
                set_names.append(i)
        # solve conflicts in order. The set name must be either the first item of chain or it is a loop
        # always keep odd items and delete even items.
        all_matches = []
        for sn in set_names:
            lst = self.drop_even_matches(sn, match_forward)
            # print('set name is', sn,  'set matching is:', sorted(lst))
            all_matches.extend(lst)
        # print('Matching pairs are: ', sorted(all_matches))
        return len(all_matches), all_matches

    def solve_bipartite_gt(self):
        """
        Solve role conflicts (one cannot be passenger and driver at the same time)
        1. find all matching chains
        2. For each chain, select feasible matching greedily until no more matching
        :return:
        """
        count_pairs, match_reverse = solve_bipartite_maxflow(self.cp_matrix)
        return count_pairs, match_reverse


def solve_bipartite_maxflow(mat, connect_weight=1):
    """
    :param mat: the carpoolable 0-1 matrix
    :param connect_weight: weight of each link connecting source/sink to nodes
    :return:
    """
    nrow, ncol = mat.shape
    g = Graph()
    # construct network (add node and vertices)
    for i in range(nrow):  # nodes for "driver"
        g.add_vertex(i)
    for j in range(ncol):  # nodes for "passenger"
        g.add_vertex(j+nrow)
    # construct edge weight
    cap = g.new_edge_property("int")
    indicies = list(zip(*np.where(mat > 0)))
    for ind in indicies:
        i, j = ind
        I, J = i, j + nrow
        g.add_edge(g.vertex(I), g.vertex(J))
        e = g.edge(g.vertex(I), g.vertex(J))
        cap[e] = mat[i, j]
    # add source and sink node
    source_id = nrow + ncol
    target_id = source_id + 1
    g.add_vertex(source_id)
    g.add_vertex(target_id)
    src = g.vertex(source_id)
    tgt = g.vertex(target_id)
    # add direct links from source and to targets
    for i in range(nrow):
        g.add_edge(src, g.vertex(i))
        e = g.edge(src, g.vertex(i))
        cap[e] = 1
    for j in range(nrow, nrow+ncol):
        g.add_edge(g.vertex(j), tgt)
        e = g.edge(g.vertex(j), tgt)
        cap[e] = 1
    g.edge_properties["cap"] = cap
    # solve problem use max-flow algorithm
    res = gt.edmonds_karp_max_flow(g, g.vertex(source_id), g.vertex(target_id), cap)
    res.a = cap.a - res.a  # the actual flow
    # get maxflow (that is, the number of pairs)
    max_flow = sum(res[e] for e in g.vertex(target_id).in_edges())
    # print("max flow is:", max_flow)
    # print edges (carpool pairs) found by algorithm

    # init trace back item for each passenger
    match_reverse = [-1] * ncol
    for e in g.edges():
        edge_in, edge_out = e
        # exclude edges with source and sinks
        if edge_in == source_id or edge_out == target_id:
            continue
        if res[e] > 0:
            # print(edge_in, edge_out)
            in_id, out_id = int(edge_in), int(edge_out)
            out_id = out_id - nrow
            match_reverse[out_id] = in_id
    # should return count_pairs, match_reverse
    return max_flow, match_reverse


# Python program to find maximal Bipartite matching.
class GFG_Square:
    def __init__(self, graph):
        # residual graph, assume it is a square graph
        self.graph = graph
        self.ppl = len(graph)
        # self.jobs = len(graph[0])
        self.jobs = self.ppl  # assume it is a square graph

    # A DFS based recursive function that returns true
    # if a matching for vertex u is possible
    def bpm(self, u, matchR, seen):
        # Try every job one by one
        for v in range(self.jobs):
            # If applicant u is interested in job v and v is not seen
            if self.graph[u][v] and seen[v] is False:
                # Mark v as visited
                seen[v] = True
                """
                If job 'v' is not assigned to an applicant OR 
                previously assigned applicant for job v1 (== matchR[v]).
                has an alternate job available.
                In recursion, since v is marked as visited in the above line, matchR[v] in the following 
                recursive call will not get job 'v' again. In this way, it is DFS"""
                # a recursive structure
                if matchR[v] == -1 or self.bpm(matchR[v], matchR, seen):
                    matchR[v] = u
                    return True
        return False

    # Returns maximum number of matching
    def maxBPM(self):
        """An array to keep track of the applicants assigned to jobs.
           The value of matchR[i] is the applicant number assigned to job i,
           the value -1 indicates nobody is assigned."""
        matchR = [-1] * self.jobs
        # Count of jobs assigned to applicants
        result = 0
        for i in range(self.ppl):
            # Mark all jobs as not seen for next applicant.
            seen = [False] * self.jobs
            # Find if the applicant 'u' can get a job
            if self.bpm(i, matchR, seen):
                result += 1
        CM = self.getCM(matchR)
        return result, CM, matchR

    # Returns maximum number of carpools
    # first by running maxBPM, then decide conflicts
    # finally returns carpool pairing
    def maxCarpoolBPM(self, timeorder=None):
        nrow = self.ppl
        if timeorder is None:
            timeorder = list(range(nrow))
        # build a dict index==>order
        index2order = {}
        for k, v in enumerate(timeorder):
            index2order[v] = k

        __, CM, __ = self.maxBPM()
        CM = np.transpose(CM)  # make CM upper triangular matrix
        
        # heuristics algorithm to solve conflicts
        for i in timeorder:
            # check the sum of row and column
            rcsum = np.sum(CM[i, :]) + np.sum(CM[:, i])
            if rcsum > 1:
                a = CM[:, i]  # the column represents all arcs point to i
                b = CM[i, :]  # the row represents all arcs outgoing from i
                idxa, idxb = np.where(a == 1)[0][0].tolist(), np.where(b == 1)[0][0].tolist()
                # print('idxa', idxa, 'idx order a', index2order[idxa])
                # print('idxb', idxb, 'idx order b', index2order[idxb])
                # check which of the idxa, idxb node happens last in time order, delete that
                # print('timeorder:', timeorder)
                if index2order[idxa] > index2order[idxb]:
                    CM[:, i] = 0
                else:
                    CM[i, :] = 0
        for i in timeorder:
            rcsum = np.sum(CM[i, :]) + np.sum(CM[:, i])
            if rcsum == 0:
                CM[i, i] = 1
        # import the updated matrix to maxBPM
        self.graph = CM

        maxpairs2, CM2, matchR2 = self.maxBPM()
        return maxpairs2, np.transpose(CM), matchR2  # make CM upper triangular matrix

    # special version for carpool cluster objects
    def maxCarpoolBPM_v2(self, timeorder=None):
        nrow = self.ppl
        if timeorder is None:
            timeorder = list(range(nrow))
        # build a dict index==>order
        index2order = {}
        for k, v in enumerate(timeorder):
            index2order[v] = k

        __, CM, __ = self.maxBPM()
        CM = np.transpose(CM)  # make CM upper triangular matrix
        # heuristics algorithm to solve conflicts
        for i in timeorder:
            # check the sum of row and column
            rcsum = np.sum(CM[i, :]) + np.sum(CM[:, i])
            if rcsum > 1:
                a = CM[:, i]  # the column represents all arcs point to i
                b = CM[i, :]  # the row represents all arcs outgoing from i
                idxa, idxb = np.where(a == 1)[0][0].tolist(), np.where(b == 1)[0][0].tolist()
                # print('idxa', idxa, 'idx order a', index2order[idxa])
                # print('idxb', idxb, 'idx order b', index2order[idxb])
                # check which of the idxa, idxb node happens last in time order, delete that
                # print('timeorder:', timeorder)
                if index2order[idxa] > index2order[idxb]:
                    CM[:, i] = 0
                else:
                    CM[i, :] = 0
        for i in timeorder:
            rcsum = np.sum(CM[i, :]) + np.sum(CM[:, i])
            if rcsum == 0:
                CM[i, i] = 1
        # import the updated matrix to maxBPM
        # self.graph = CM
        CM = CM.astype(int)
        pairs = np.where(CM == 1)
        return CM, list(zip(*pairs))

    # Returns maximum number of matching
    # timeorder: a list showing the temporal sequence for the data in the matrix
    # given the traceback relations matchR, return connection (binary) matrix
    def getCM(self, matchR):
        nr = len(matchR)
        CM = np.zeros((nr, nr))
        for k, v in enumerate(matchR):
            if v != -1:
                CM[k, v] = 1
        return CM


# There is a linear process of get travel time
# between O-Ds (within the scope of Employment Centers TAZs)
def getECsTime():
    # os.chdir('F:\Data\ABM\ABM new output')
    os.environ['abm_new_output'] = '/Users/diyi93/Desktop/Data/ABM/ABM new output'
    ECs_taz = pd.read_csv(os.path.join(os.environ['abm_new_output'], 'activitycenter_TAZ.csv'))
    taz_time_am = pd.read_csv(os.path.join(os.environ['abm_new_output'], 'trip_sov_AM.csv'))
    # get ECs' names
    centers_name = ECs_taz.ACT_CNTR.unique().tolist()
    ECs_taz_dict = {}
    for c in centers_name:
        lst = ECs_taz['MTAZ10'].loc[ECs_taz['ACT_CNTR'] == c].tolist()
        ECs_taz_dict[c] = lst

    # get ECs TAZ list
    ECs_taz_lst = ECs_taz.MTAZ10.unique().tolist()
    ECs_taz_lst = sorted(ECs_taz_lst)
    # select sov trip data only by those
    # involves TAZ centers
    q1 = taz_time_am['orig_taz'].isin(ECs_taz_lst)
    q2 = taz_time_am['dest_taz'].isin(ECs_taz_lst)
    ECs_time = taz_time_am.loc[q1 & q2]
    ECs_time = ECs_time.sort_values(by=['orig_taz', 'dest_taz', 'depart_period'])
    ECs_time = ECs_time.drop_duplicates()
    # simplify dataframe by ignoring depart time
    # for O-D with multiple time, keep the worst one
    ECs_time_short = ECs_time.loc[:, ['orig_taz', 'dest_taz', 'travel_time']].copy()
    ECs_time_short = ECs_time_short.drop_duplicates()

    # a helper function to get mini travel time
    def get_mini(g):
        return g.loc[g['travel_time'].idxmin(), :]

    ECs_time_short = ECs_time_short.groupby(['orig_taz', 'dest_taz']).apply(get_mini)
    return ECs_time_short, ECs_taz_lst


# New function above getECsTime using CarpoolSim
def query_od_info(o_taz, d_taz, engine=None):
    """
    Connect to the Postgresql database table storing travel time and retention paths between any
    pair of taz centroids.

    :return:
    """
    if engine is None:
        from sqlalchemy import create_engine
        engine = create_engine('postgresql://diyi93:muwenyang0001@localhost/diyi93', use_batch_mode=True)

    results = engine.execute(
        "SELECT * FROM dists WHERE origin='{}' AND destination='{}'".format(o_taz, d_taz)).fetchall()
    for row in results:
        row_dist = row[2]
        row_path = row[3].strip("[]").replace("'", "").replace(" ", "").split(',')
    return str(o_taz), str(d_taz), row_dist, row_path


# a function feed ECs travel time between O-D into a matrix
def generateBigDistMat(ECs_time_short, ECs_taz_lst, withinODdist=5):
    """
    Given taz travel information, fill a lookup matrix for OD pairs between employment centers
    to lookup travel times. This is done for old method using ABM outputs, not for new method using RoadwaySim

    :param ECs_time_short: all trips between EC ODs to check trips between certain EC's OD pairs.
    :param ECs_taz_lst: a list containing all TAZs of ECs (employment centres)
    :param withinODdist: there might be no trips within the same OD for ABM, so fill the diagonal of travel time
        matrix with travel time of 5 minutes as default
    Returns.
        dist_mat: the O-D matrix records travel time
        check_tazind: a dictionary maps from taz number to taz index
    """
    # now start building up the big matrix
    nrow = len(ECs_taz_lst)
    dist_mat = np.full((nrow, nrow), np.inf)
    # build up a check dict for index given taz code
    check_tazind = {}
    for k, v in enumerate(ECs_taz_lst):
        check_tazind[v] = k
    # for each row in ECs_time_short
    for row in ECs_time_short.iterrows():
        orig, dest, tt = row[1]['orig_taz'], row[1]['dest_taz'], row[1]['travel_time']
        ind1, ind2 = check_tazind[orig], check_tazind[dest]
        dist_mat[ind1, ind2] = np.around(tt, 1)
    # assume O and D are the same taz, the cost is a fixed number, say 5 minutes
    for i in range(nrow):
        dist_mat[i, i] = withinODdist
    return dist_mat, check_tazind


def getTimeZeroOneMat(f_lst, time_thr=15):
    """
    This given a list of depart/arrive time, generate a depart/arrive time difference matrix by propagating list
    to two dimensions. The (absolute) difference is then converted to 0-1 carpool-able matrix with carpool-able
    trips denoted as 1. Trips are considered carpool-able if the difference between their depart time is greater
    than a given time threshold.

    :param f_lst: a list of depart/arrive time, depend on the input data
    :param time_thr: the given time threshold
    :return: zo_mat: a zero-one matrix
    """
    # f_time = np.array(f['newarr']).reshape((-1,1))
    f_time = np.array(f_lst).reshape((-1, 1))
    # time difference matrix
    mat1 = np.tile(f_time, (1, f_time.shape[0]))
    mat2 = np.transpose(mat1)
    mat_td = np.absolute(np.subtract(mat1, mat2))  # time difference matrix

    # set diagonal to inf
    # np.fill_diagonal(mat_td, np.inf)
    # apply func to each element in mat_td
    def myfunc(ele, time_thr):  # can only connects distance smaller than time_thr
        if ele <= time_thr + 0.01:
            return 1
        return 0

    myfunc = np.vectorize(myfunc)
    zo_mat = myfunc(mat_td, time_thr)  # get zero-one matrix
    zo_mat = np.triu(zo_mat, 1)
    return zo_mat


def getRerouteZeroOneMat(f, bigTAZMatrix, check_tazind, time_reroute):
    """
    Similar to getTimeZeroOneMat function, this function generates another zero-one carpool-able matrix measuring
    reroute traveling time after driver drops off the passenger.

    Notice that the reroute time is coarse since it has not considered role playing in a carpool trip. In other words,
    the reroute time of A drives B is different from that of B drives A to the destination.

    :param f: trip DataFrame (the cluster of trips to be processed together)
    bigTAZMatrix: the big matrix to lookup travel time between TAZs (in our project within ECs destinations)
    time_reroute: a threshold to decide which binary matrix.
            set binary matrix entry to one if there is a connection
    check_tazind: a helper dictionary to map every taz number to its positional number
    Output.
    """
    # get all dest taz values
    tazs = np.array(f['dest_taz']).tolist()
    # get indes from tazs code
    tazs_idx = [check_tazind[t] for t in tazs]
    # get small taz matrix by indices
    small_mat = bigTAZMatrix[tazs_idx, :]
    small_mat = small_mat[:, tazs_idx]
    # mirrow values since too many infinity
    # (as well as considering current carpooling model settings)
    small_mat_trans = np.transpose(small_mat)
    small_mat = np.minimum(small_mat_trans, small_mat)

    # np.fill_diagonal(small_mat, np.inf)
    # apply func to each element in mat_td
    def myfunc(ele, time_reroute):  # can only connects distance smaller than time_reroute (e.g.,5 min)
        if ele <= time_reroute + 0.01:
            return 1
        return 0

    myfunc = np.vectorize(myfunc)
    zo_mat = myfunc(small_mat, time_reroute)  # get zero-one matrix
    zo_mat = np.triu(zo_mat, 1)  # make it an upper triangular matrix
    return zo_mat


# sort trip cluster in the order of arrive/depart time
def get_time_order(g):
    g_time = g['newarr']
    timeorder_index = g_time.sort_values(ascending=True).index.tolist()
    timeorder = []
    for k in timeorder_index:
        timeorder.append(g_time.index.get_loc(k))
    return timeorder


# based on the output of GB_orig_temp, generate adjacency matrices for each trip cluster
def genFinalMats(trips, bigTAZMatrix, check_tazind, time_thr=10):
    trips1 = trips.loc[trips['temp_label'] != -1].copy()  # delete trips with DBSCAN label -1
    # trip_gb = trips1.groupby(['orig_taz','dest_taz']) # trip_gb is a groupby data
    trip_gb = trips1.groupby(['orig_taz'])

    # get spatial matrix
    def get_matrices(g):
        g_lst = g['newarr'].values.tolist()
        mat1 = getTimeZeroOneMat(g_lst, time_thr=time_thr)
        mat2 = getRerouteZeroOneMat(g, bigTAZMatrix, check_tazind, time_reroute=15)
        return np.minimum(mat1, mat2)  # return one 0-1 matrix by applying 'and' operator

    mat_final = trip_gb.apply(get_matrices)  # final carpool-able matrix
    time_order = trip_gb.apply(get_time_order)  # time_order for making heuristic decisions
    return mat_final, time_order


# After getting all the trips, a function running all trips
def Carpool_Bipartite(trips, mat_final):
    matindlst = mat_final.index.tolist()
    # we can directly solve on every matrix for bipartite
    # and summarize the result, could consier traceback trips later
    total_pairs = 0
    total_drivers = trips.shape[0]
    count100taz, count_rows = 0, 0

    # store matrices and number of pairs in each group
    CMs, maxPairs, rec_pairs = {}, {}, {}
    total_groupnum = len(matindlst)
    for n, i in enumerate(matindlst):
        if n % int(0.1 * total_groupnum) == 0 and n != 0:
            print('Finished {} %'.format(10 * n / int(0.1 * total_groupnum)))
            print('# carpools pairs found:', count100taz)
            print('# drivers scanned:', count_rows)
            print('Paired drivers Percentile:', round(count100taz * 2 / count_rows, 2))
            count100taz = 0
            count_rows = 0
        j = mat_final[i]
        g = GFG_Square(j)
        # maxpairs, CM, rec= g.maxCarpoolBPM(timeorder_j)
        maxpairs, CM, rec = g.maxCarpoolBPM()

        CMs[i], maxPairs[i], rec_pairs[i] = CM, maxpairs, rec
        total_pairs += maxpairs
        count100taz += maxpairs
        count_rows += j.shape[0]

    print('total drivers in dataset:', total_drivers)
    print('total drivers get paired:', total_pairs * 2)
    return CMs, maxPairs, rec_pairs


# below are two functions for calculating saved bipartite
def est_VMT_saved_bipartite(trips, rec_pairs):
    trip_gb = trips.groupby(['orig_taz'])
    key_lst = list(trip_gb.groups.keys())
    lst = []
    for i in key_lst:
        trip_cluster = trip_gb.get_group(i)
        record_clster = rec_pairs[i]
        for ind, j in enumerate(record_clster):
            if j != -1:
                lst.append(min(trip_cluster.loc[:, "distance"].iloc[list([j, ind])].tolist()))

    totalSavedVMT = sum(lst)
    print("Average VMT saved by each match:", totalSavedVMT / len(lst))
    print("Total mileage saved by all:", totalSavedVMT)
    return None


def est_VMT_saved_bipartite_return(trip_gb, rec_pairs):
    """ return an Dataframe for all paired trips with just O, D information"""
    key_lst = list(trip_gb.groups.keys())
    AllpairedODlst = []
    for i in key_lst:
        trip_cluster = trip_gb.get_group(i)
        record_clster = rec_pairs[i]
        for ind, j in enumerate(record_clster):
            if j != -1:
                # print('inner', trip_cluster.loc[:,['orig_taz', 'dest_taz']].iloc[list([j, ind]),:].shape)
                AllpairedODlst.append(trip_cluster.loc[:, ['orig_taz', 'dest_taz']].iloc[list([j, ind]), :])
    return AllpairedODlst
