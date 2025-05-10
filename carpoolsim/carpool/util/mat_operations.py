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
    Xs = np.array(pnr.x.tolist()).reshape((1, -1))
    Ys = np.array(pnr.y.tolist()).reshape((1, -1))
    return Xs, Ys

def get_distances_among_coordinates(
    Xs: np.ndarray,  # 1 dim array of x coordinates among N travelers.
    Ys: np.ndarray,  # 1 dim array of y coordinates among N travelers.
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the distance matrix among N travelers.
    Return arrays are [N, N] to describe the distance between each pair of travelers.
    """
    if Xs.ndim == 1:
        Xs = Xs.reshape((1, -1))
    if Ys.ndim == 1:
        Ys = Ys.reshape((1, -1))
    nrow = ncol = Xs.shape[1]
    mat_x = np.tile(Xs.transpose(), (1, ncol))
    mat_x = np.abs(mat_x - np.tile(Xs, (nrow, 1)))
    mat_y = np.tile(Ys.transpose(), (1, ncol))
    mat_y = np.abs(mat_y - np.tile(Ys, (nrow, 1)))
    mat_dist = np.sqrt(mat_x ** 2 + mat_y ** 2)
    return mat_x, mat_y, mat_dist