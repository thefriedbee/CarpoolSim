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
import multiprocess

import carpoolsim.network_prepare as net_prep
from carpoolsim.database.prepare_database import (
    sql_engine_factory,
    batch_store_from_lst
)


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
        self.tazs_ids = []
        self.network_dict = {}

    def get_taz_id_list(self):
        self.tazs.taz_id = self.tazs.taz_id.astype(int)
        self.tazs_ids = sorted(self.tazs.taz_id.tolist())

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

    def prepare_taz_lists(self, chunk_size=100):
        # break taz list to chunks for multiprocessing
        taz_lst = []
        num_tazs = len(self.tazs_ids)
        num_chucks = int(len(self.tazs_ids) / chunk_size) + 1
        for idx in range(num_chucks):
            start_index = idx * chunk_size
            end_index = min((idx + 1) * chunk_size, num_tazs)
            _lst = self.tazs_ids[start_index: end_index]
            taz_lst.append(_lst)
        return taz_lst

    def run_batch(self, source_id=1):
        network_dict = self.network_dict

        # only record results to centroids
        dists_dict, paths_dict = run_batch_ods(
            network_dict['DG'],
            source=f'{source_id}',
            destination_lst=self.tazs_ids,
            weight='forward'
        )
        return dists_dict, paths_dict


# Define a new function to simply run shortest path given origin and destination node id.
# No dict_settings or option is required, just simply compute od travel time
# given od nodes and a network with static speed.
# make it as simple as possible for efficiency
def run_batch_ods(
        nx_network: nx.DiGraph,
        source: int | str,
        destination_lst: list[int | str] | None = None,
        weight='forward'
):
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
    # convert formats
    source = str(source)
    destination_lst = [str(item) for item in destination_lst]

    len_dict, paths_dict = nx.single_source_dijkstra(
        nx_network, source, weight=weight)
    if destination_lst is not None:
        len_dict_output, paths_dict_output = {}, {}
        for k in destination_lst:
            len_dict_output[k] = len_dict[k]
            paths_dict_output[k] = paths_dict[k]
        len_dict = len_dict_output
        paths_dict = paths_dict_output
    return len_dict, paths_dict


def get_shortest_paths(
        network_dict,
        destination_lst,
        source_taz_lst
):
    path_retention_dists, path_retention_paths = {}, {}
    taz_lst = []
    for taz in source_taz_lst:
        dists_dict, paths_dict = run_batch_ods(
            network_dict['DG'],
            str(taz),
            destination_lst,
            weight='forward'
        )
        path_retention_dists[str(taz)] = dists_dict
        path_retention_paths[str(taz)] = paths_dict
        taz_lst.append(str(taz))

    print(f'Finished searching {len(source_taz_lst)} tazs')
    # store values in disk instead of holding all the memories
    db = {'dist': path_retention_dists, 'path': path_retention_paths}
    return db


if __name__ == "__main__":
    # import basic settings
    from carpoolsim.basic_settings import *

    # load traffic network data
    taz_centriod_nodes = gpd.read_file(os.environ["taz"])
    df_nodes_raw = gpd.read_file(os.environ['network_nodes'])
    df_links_raw = gpd.read_file(os.environ['network_links'])

    # init object
    traffic_network = TrafficNetwork(
        network_links=df_links_raw,
        network_nodes=df_nodes_raw,
        tazs=taz_centriod_nodes
    )

    traffic_network.get_taz_id_list()
    traffic_network.convert_abm_links()
    traffic_network.build_network()

    network_dict = traffic_network.network_dict
    destination_lst = traffic_network.tazs_ids
    taz_lst = traffic_network.prepare_taz_lists(chunk_size=100)
    task_inputs = [
        (network_dict, destination_lst, taz)
        for taz in taz_lst
    ]

    t0 = time.perf_counter()
    with multiprocess.Pool(NUM_PROCESSES) as pool:
        results = pool.starmap(get_shortest_paths, task_inputs)

    d1 = time.perf_counter() - t0
    print(f'It takes {d1 / 60:.1f} minutes to prepare objects')

    # store results in database
    t0 = time.perf_counter()
    engine = sql_engine_factory(DB_URL)
    batch_store_from_lst(results, DB_URL)
    d1 = time.perf_counter() - t0
    print(f'It takes {d1 / 60:.1f} minutes to prepare dataset')
    print("all data has loaded into the database!")

