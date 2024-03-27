"""
Prepare inputs for the analysis...

A traffic network class is defined consists of three parts:
- network links: directed links of the network
- network nodes: nodes (end points) of the network
- network tazs: traffic analysis zones
"""
import time

import geopandas as gpd
import networkx as nx

import carpoolsim.network_prepare as net_prep


class TrafficNetwork:
    def __init__(
            self,
            network_links: gpd.GeoDataFrame,
            network_nodes: gpd.GeoDataFrame,
            tazs: gpd.GeoDataFrame,
    ) -> None:
        self.gdf_nodes = network_nodes
        self.gdf_links = network_links
        self.tazs = tazs
        self.network_dict = {}

    def convert_abm_links(self):
        gdf_links = net_prep.initialize_abm15_links(
            self.gdf_nodes,
            self.gdf_links,
            drop_connector=False
        )
        self.gdf_links = gdf_links

    def build_network(self):
        df_links = self.gdf_links

        network_dict = {
            "links": df_links,
            "DG": net_prep.build_carpool_network(df_links)
        }
        # example to check a links time
        # print(network_dict['DG']['1']['65666'])
        # print(network_dict['DG']['65666']['1'])
        self.network_dict = network_dict

    def run_batch(self):
        network_dict = self.network_dict

        # only record results to centroids
        destination_lst = [str(i) for i in range(1, 5874)]
        t1 = time.perf_counter()
        dists_dict, paths_dict = run_batch_ods(
            network_dict['DG'],
            source='1',
            destination_lst=destination_lst,
            weight='backward'
        )
        delta_t = time.perf_counter() - t1
        print('It takes {} seconds to run'.format(delta_t))


# Define a new function to simply run shortest path given origin and destination node id.
# No dict_settings or option is required, just simply compute od travel time
# given od nodes and a network with static speed.
# make it as simple as possible for efficiency
def run_batch_ods(nx_network, source, destination_lst=None, weight='forward'):
    """
    Use single source Dijkstra to compute the travel time from one origin to all listed destinations.
    This function also aims at testing compute efficiency.
    For example, if there are 1000 taz centroids, we need to run Dijkstra's algorithm 1000 times.
    NOTICE: if weight is chosen as backwards, the function reversed all node list to forward order
    before returned, the interpretation of list is also changed to the keyed value as origin while the
    parameter/argument 'source' as interested destination
    :param nx_network: a static Networkx graph for computing shortest path between two ods.
    :param source: a source taz centroid node
    :param destination_lst: a list of centroid nodes interested
    :param weight: the weight of the graph, either forward graph or backward graph
    :return: return a dataframe with all the information
    """
    len_dict, paths_dict = nx.single_source_dijkstra(nx_network, source, weight=weight)
    if destination_lst is not None:
        len_dict_output, paths_dict_output = {}, {}
        for k in destination_lst:
            len_dict_output[k] = len_dict[k]
            paths_dict_output[k] = paths_dict[k]
        len_dict = len_dict_output
        paths_dict = paths_dict_output
    return len_dict, paths_dict
