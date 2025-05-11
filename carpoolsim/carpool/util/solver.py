"""
The code used for solving the carpooling matrix.
"""
import numpy as np
from gurobipy import Model, GRB, quicksum

from carpoolsim.carpool.trip_cluster_abstract import TripClusterAbstract


def compute_optimal_lp(
    tc: TripClusterAbstract,
    mute: bool = True,
    use_both: bool = False,
) -> list[tuple[int, int]]:
    """
    Use Gurobipy to solve for optimal solutions
    :param mute: if True, don't print output results.
    :param use_both: if True, consider both modes
    :return: No return
    """
    nrow, ncol = tc.nrow, tc.ncol
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
        tt_matrix = tc.tt_matrix_all
    else:
        tt_matrix = tc.tt_matrix
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
    result_lst = [p for p in tps if x[p].x > 0.5]
    return result_lst

