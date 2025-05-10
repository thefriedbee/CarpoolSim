import os

import pytest
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine


from carpoolsim.carpool.trip_cluster_basic import TripDemands
from carpoolsim.carpool.trip_cluster_dc import TripClusterDC
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
dict_network = prepare_network()
sample_trips = trips.iloc[:3].copy()  # use 3 trips to test


def test_td_fill_diagonal():
    trip_demands = TripDemands(
        sample_trips, 
        dict_network['DG'], 
        dict_network['links'], 
        engine
    )
    trip_demands.compute_sov_info()
    tdc = TripClusterDC(trip_demands)
    tt_lst = trip_demands.soloTimes
    dst_lst = trip_demands.soloDists
    tdc.fill_diagonal(tt_lst, dst_lst)
    print("travel time matrix")
    print(tdc.tt_matrix)
    print("travel distance matrix")
    print(tdc.ml_matrix)
    assert True


def test_td_compute_carpool():
    trip_demands = TripDemands(
        sample_trips, 
        dict_network['DG'], 
        dict_network['links'], 
        engine
    )
    trip_demands.compute_sov_info()
    tdc = TripClusterDC(trip_demands)
    tt_lst = trip_demands.soloTimes
    dst_lst = trip_demands.soloDists
    tdc.fill_diagonal(tt_lst, dst_lst)
    ret1, ret2 = tdc.compute_carpool(0, 1)
    print(tdc.cp_matrix)
    print(ret1)
    print(ret2)
    assert False
