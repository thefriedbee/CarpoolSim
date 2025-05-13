import numpy as np
# import graph_tool.flow as gt
# from graph_tool.all import *


# find maximal Bipartite matching for general graph (given feasibility matrix of any shape)
class CarpoolBipartite:
    def __init__(
            self,
            cp_matrix: np.ndarray,
            tt_matrix: np.ndarray,
    ):
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

    def dfs_same_source(self, u: int, match_reversed: list, seen: list[bool]):
        """
        A dfs based recursive function that returns true if a matching from vertex u is possible.
        (u ==> v)
        :param u: integer id of row index for drivers
        :param match_reversed: a reversal dictionary records which driver u points to passenger v.
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
                previously the passenger is previously assigned to driver u1 (== match_reversed[v]).
                The driver u1 has an alternate passenger available. 
                In recursion, since v is marked as visited in the above line, match_reversed[v] in the following 
                recursive call will not get job 'v' again. In this way, it is DFS search.
                """
                if match_reversed[v] == -1 or self.dfs_same_source(match_reversed[v], match_reversed, seen):
                    match_reversed[v] = u
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

    def get_chain_loop(
            self,
            u: int,
            match_reverse: list[int],
            match_forward: list[int],
            detected_lst: list
    ) -> int:
        """
        Given the person u, get the chain it contained in using two pointers to opposite directions.
        :param u: the integer index of person u
        :param match_reverse: map relation from passenger v to driver u.
        If value is -1, then u cannot be passenger to others.
        :param match_forward: map relation from
        :param detected_lst: a list record if a person is grouped into a set. If so, which set
        (set name as a person's index in the set)
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
            # if no driver found or out of scope, no need to continue
            # otherwise, try to find the next driver in chain.
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
        return return_matches

    def solve_bipartite_conflicts_naive(self, drop_sov_pairs: bool = True):
        """
        Solve role conflicts (one cannot be passenger and driver at the same time)
        1. find all matching chains/loops
        2. For each chain, delete about half of the matching until there are no conflicts
        :return: num of matches and a list of matches
        """
        count_pairs, match_reverse = self.solve_bipartite()
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
        if drop_sov_pairs:
            all_matches = self.drop_sov_pairs(all_matches)
        return len(all_matches), all_matches

    def drop_sov_pairs(self, all_matches: list[tuple[int, int]]):
        """
        Keep only carpool pairs in the match_reverse list.
        """
        return [pair for pair in all_matches if pair[0] != pair[1]]

    # def solve_bipartite_gt(self):
    #     """
    #     Solve role conflicts (one cannot be passenger and driver at the same time)
    #     1. find all matching chains
    #     2. For each chain, select feasible matching greedily until no more matching
    #     :return:
    #     """
    #     count_pairs, match_reverse = solve_bipartite_maxflow(self.cp_matrix)
    #     return count_pairs, match_reverse


# def solve_bipartite_maxflow(mat: np.ndarray):
#     """
#     :param mat: the carpool-able 0-1 matrix
#     """
#     nrow, ncol = mat.shape
#     g = Graph()
#     # construct network (add node and vertices)
#     for i in range(nrow):  # nodes for "driver"
#         g.add_vertex(i)
#     for j in range(ncol):  # nodes for "passenger"
#         g.add_vertex(j+nrow)
#     # construct edge weight
#     cap = g.new_edge_property("int")
#     indicies = list(zip(*np.where(mat > 0)))
#     for ind in indicies:
#         i, j = ind
#         I, J = i, j + nrow
#         g.add_edge(g.vertex(I), g.vertex(J))
#         e = g.edge(g.vertex(I), g.vertex(J))
#         cap[e] = mat[i, j]

#     # add source and sink node
#     source_id = nrow + ncol
#     target_id = source_id + 1
#     g.add_vertex(source_id)
#     g.add_vertex(target_id)
#     src = g.vertex(source_id)
#     tgt = g.vertex(target_id)

#     # add direct links from source and to targets
#     for i in range(nrow):
#         g.add_edge(src, g.vertex(i))
#         e = g.edge(src, g.vertex(i))
#         cap[e] = 1
#     for j in range(nrow, nrow+ncol):
#         g.add_edge(g.vertex(j), tgt)
#         e = g.edge(g.vertex(j), tgt)
#         cap[e] = 1
#     g.edge_properties["cap"] = cap
#     # solve problem use max-flow algorithm
#     res = gt.edmonds_karp_max_flow(g, g.vertex(source_id), g.vertex(target_id), cap)
#     res.a = cap.a - res.a  # the actual flow
#     # get maxflow (that is, the number of pairs)
#     max_flow = sum(res[e] for e in g.vertex(target_id).in_edges())
#     # print edges (carpool pairs) found by algorithm

#     # init trace back item for each passenger
#     match_reverse = [-1] * ncol
#     for edge in g.edges():
#         edge_in, edge_out = edge
#         # exclude edges with source and sinks
#         if edge_in == source_id or edge_out == target_id:
#             continue
#         if res[edge] > 0:
#             # print(edge_in, edge_out)
#             in_id, out_id = int(edge_in), int(edge_out)
#             out_id = out_id - nrow
#             match_reverse[out_id] = in_id

#     # should return count_pairs, match_reverse
#     return max_flow, match_reverse
