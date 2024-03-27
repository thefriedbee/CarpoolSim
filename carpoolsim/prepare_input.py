"""
Prepare inputs for the analysis...

A traffic network class is defined consists of three parts:
- network links: directed links of the network
- network nodes: nodes (end points) of the network
- network tazs: traffic analysis zones
"""

import geopandas as gpd

from carpoolsim.network_prepare import (
    initialize_abm15_links,
    build_carpool_network,
)


class TrafficNetwork:
    def __init__(
            self,
            network_links: gpd.GeoDataFrame,
            network_nodes: gpd.GeoDataFrame,
            tazs: gpd.GeoDataFrame,
    ) -> None:
        self.network_links = network_links
        self.network_nodes = network_nodes
        self.tazs = tazs

    pass




