import os

import pytest
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine


from carpoolsim.carpool.trip_cluster_basic import TripDemands
from carpoolsim.network_prepare import build_carpool_network

# load environment variables
import carpoolsim.basic_settings as bs
from tests.test_load_example import trips, tazs, df_nodes, df_links, pnr_lots

engine = create_engine(bs.DB_URL)


def prepare_network():
    dict_network = {
        'DG': build_carpool_network(df_links),
        'links': df_links,
        'nodes': df_nodes,
    }
    return dict_network


def test_trip_demands():
    dict_network = prepare_network()
    sample_trips = trips.sample(3)  # sample 3 trips
    td = TripDemands(
        sample_trips, 
        dict_network['DG'], 
        dict_network['links'], 
        engine
    )
    solo_paths, solo_dists, solo_times, tt_lst, dst_lst = td.compute_sov_info()
    print(solo_paths)
    print(solo_dists)
    print(solo_times)
    print(tt_lst)
    print(dst_lst)
    assert True
    assert len(solo_paths) == len(sample_trips)
    assert len(solo_dists) == len(sample_trips)
    assert len(solo_times) == len(sample_trips)
    assert len(tt_lst) == len(sample_trips)
    assert len(dst_lst) == len(sample_trips)













