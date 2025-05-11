import networkx as nx

from carpoolsim.database.query_database import query_od_info
import carpoolsim.basic_settings as bs


def get_path_distance_and_tt(network: nx.DiGraph, nodes: list[str]):
    """
    A helper function to get driving path given a set of network nodes.
    :param nodes: a list of traveling network node id
    :return: travel time and mileage distance of the path
    """
    tt, dst = 0, 0
    for i in range(len(nodes) - 1):
        tt += network[nodes[i]][nodes[i + 1]]['forward']
        dst += network[nodes[i]][nodes[i + 1]]['dist']
    return tt, dst


def dynamic_shortest_path_search(network: nx.DiGraph, start_node, end_node, start_taz, end_taz):
    engine = bs.engine
    # step 1. get OD and query the shortest path between OD TAZ centroids.
    __, __, row_dist, nodes = query_od_info(engine, start_taz, end_taz)
    # step 2. store graph distances, reset them to zeros for fast compute
    orig_distances = []
    for i in range(len(nodes) - 1):
        orig_distances.append(network[nodes[i]][nodes[i + 1]]['forward'])
        network[nodes[i]][nodes[i + 1]]['forward'] = 0.0001
    # step 3.1 Run dijkstra's algorithm to recompute the shortest paths
    __, pth_nodes = nx.single_source_dijkstra(network, str(start_node), str(end_node), weight='forward')
    # step 3.2 Restore all graph weights
    for i, d in enumerate(orig_distances):
        network[nodes[i]][nodes[i + 1]]['forward'] = d
    # step 4. Compute travel distance of the new path
    tt, dst = get_path_distance_and_tt(network, pth_nodes)
    return pth_nodes, tt, dst


def naive_shortest_path_search(network: nx.DiGraph, start_node, end_node):
    # step 1. get OD and query the shortest path between OD TAZ centroids.
    __, pth_nodes = nx.single_source_dijkstra(network, str(start_node), str(end_node), weight='forward')
    tt, dst = get_path_distance_and_tt(network, pth_nodes)
    return pth_nodes, tt, dst







