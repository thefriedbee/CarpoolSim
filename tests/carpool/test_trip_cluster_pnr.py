import os

import pytest
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine

from carpoolsim.carpool.trip_cluster_basic import TripDemands
from carpoolsim.carpool.trip_cluster_pnr import TripClusterPNR
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


def prepare_trip_demands():
    trip_demands = TripDemands(
        sample_trips, 
        dict_network['DG'], 
        dict_network['links'], 
        engine,
        pnr_lots
    )
    trip_demands.compute_sov_info()
    return trip_demands
trip_demands = prepare_trip_demands()
print(trip_demands.trips.columns)


def test_tc_pnr_access():
    tc = TripClusterPNR(trip_demands)
    tc.compute_pnr_access(
        trip_id=0,
        station_id=0
    )
    print(tc.pnr_access_info.shape)
    print("travel path:")
    print(tc.pnr_access_info[0, 0][0])
    print("travel time (access):")
    print(tc.pnr_access_info[0, 0][1])
    print("travel distance:")
    print(tc.pnr_access_info[0, 0][2])
    print("travel time (all):")
    print(tc.pnr_access_info[0, 0][3])
    assert True


def test_tc_check_trips_best_pnr():
    tc = TripClusterPNR(trip_demands)
    pnr_idx = tc._check_trips_best_pnr(
        trip_row1=tc.trips.iloc[0],
        trip_row2=tc.trips.iloc[1],
        int_idx1=0,
        int_idx2=1
    )
    print("pnr_idx:", pnr_idx)
    assert True


def test_compute_01_matrix_to_station_p1():
    tc = TripClusterPNR(trip_demands)
    tc.compute_01_matrix_to_station_p1()
    print(tc.pnr_matrix.shape)
    print(tc.pnr_matrix)
    assert True


def test_compute_01_matrix_to_station_p2():
    tc = TripClusterPNR(trip_demands)
    tc.compute_01_matrix_to_station_p2()
    print(tc.pnr_matrix.shape)
    print(tc.cp_matrix.shape)
    print(tc.pnr_matrix.sum(axis=1))
    print(tc.cp_matrix.sum(axis=1))
    assert True


def test_compute_in_one_step():
    tc = TripClusterPNR(trip_demands)
    tc.compute_in_one_step()
    print(tc.cp_matrix.shape)
    print(tc.cp_matrix.sum(axis=1))
    assert False



