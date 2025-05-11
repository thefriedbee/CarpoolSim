import pytest
import numpy as np

from carpoolsim.carpool.util.mat_operations import *


def test_get_projected_ods():
    oxs = np.array([1, 2, 3])
    oys = np.array([1, 2, 3])
    dxs = np.array([4, 5, 6])
    dys = np.array([4, 5, 6])
    print(oxs.shape)

    trips = pd.DataFrame({
        'ox': oxs,
        'oy': oys,
        'dx': dxs,
        'dy': dys,
    })

    oxs1, oys1, dxs1, dys1 = get_trip_projected_ods(trips)
    assert oxs1.shape == (1, 3)
    assert oys1.shape == (1, 3)
    assert dxs1.shape == (1, 3)
    assert dys1.shape == (1, 3)
    assert np.all(oxs1 == oxs)
    assert np.all(oys1 == oys)
    assert np.all(dxs1 == dxs)
    assert np.all(dys1 == dys)


def test_get_distances_among_coordinates():
    xs = np.array([1, 2, 3])
    ys = np.array([1, 2, 3])
    mat_dx, mat_dy, mat_dist = get_distances_among_coordinates(xs, ys)

    mat_dy_result = mat_dx_result = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0],
    ])
    print(mat_dx)
    assert mat_dx.shape == (3, 3)
    assert mat_dy.shape == (3, 3)
    assert mat_dist.shape == (3, 3)
    assert np.all(mat_dx == mat_dx_result)
    assert np.all(mat_dy == mat_dy_result)



