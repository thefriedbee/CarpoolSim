"""
Reuse matrix operations in carpoolsim 
- Although inputs are sometimes Pandas DataFrame, operations are all numpy array operations.
"""
import numpy as np
import pandas as pd


def get_trip_projected_ods(
    trips: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the projected origin and destination of the trip.
    """
    oxs = np.array(trips.ox.tolist()).reshape((1, -1))
    oys = np.array(trips.oy.tolist()).reshape((1, -1))
    dxs = np.array(trips.dx.tolist()).reshape((1, -1))
    dys = np.array(trips.dy.tolist()).reshape((1, -1))
    return oxs, oys, dxs, dys


def get_pnr_xys(
    pnr: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the x and y coordinates of the parking lots.
    """
    xs = np.array(pnr.x.tolist()).reshape((1, -1))
    ys = np.array(pnr.y.tolist()).reshape((1, -1))
    return xs, ys


def get_distances_among_coordinates(
    xs: np.ndarray,  # 1 dim array of x coordinates among N travelers.
    ys: np.ndarray,  # 1 dim array of y coordinates among N travelers.
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the distance matrix among N travelers.
    Return arrays are [N, N] to describe the distance between each pair of travelers.
    """
    if xs.ndim == 1:
        xs = xs.reshape((1, -1))
    if ys.ndim == 1:
        ys = ys.reshape((1, -1))
    nrow = ncol = xs.shape[1]
    mat_dx = np.tile(xs.transpose(), (1, ncol))
    mat_dx = np.abs(mat_dx - np.tile(xs, (nrow, 1)))
    mat_dy = np.tile(ys.transpose(), (1, ncol))
    mat_dy = np.abs(mat_dy - np.tile(ys, (nrow, 1)))
    mat_dist = np.sqrt(mat_dx ** 2 + mat_dy ** 2)
    return mat_dx, mat_dy, mat_dist
