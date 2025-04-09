import pytest
import numpy as np

from carpoolsim.carpool.trip_filters import *


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









