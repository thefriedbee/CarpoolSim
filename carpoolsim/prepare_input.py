"""
Prepare inputs for the analysis...

A traffic network class is defined consists of three parts:
- network links: directed links of the network
- network nodes: nodes (end points) of the network
- network tazs: traffic analysis zones
"""

import geopandas as gpd

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

        self.network_dict["links"] = df_links
        self.network_dict["DG"] = net_prep.build_carpool_network(df_links)

    def run_batch(self):
        pass




