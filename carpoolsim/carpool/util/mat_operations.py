"""
Reuse matrix operations in carpoolsim 
- Although inputs are sometimes Pandas DataFrame, operations are all numpy array operations.
"""
import numpy as np
import pandas as pd


def arrs2vec(arrs: list[np.ndarray]) -> np.ndarray:
    """
    Reshape a list of numpy arrays to one row, multiple columns.
    """
    arrs = [arr.reshape((1, -1)) for arr in arrs]
    return arrs


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
    xs, ys = arrs2vec([xs, ys])
    nrow = ncol = xs.shape[1]
    mat_dx = np.tile(xs.transpose(), (1, ncol))
    mat_dx = np.abs(mat_dx - np.tile(xs, (nrow, 1)))
    mat_dy = np.tile(ys.transpose(), (1, ncol))
    mat_dy = np.abs(mat_dy - np.tile(ys, (nrow, 1)))
    mat_dist = np.sqrt(mat_dx ** 2 + mat_dy ** 2)
    return mat_dx, mat_dy, mat_dist


def get_distances_between_ods_sov(
    oxs: np.ndarray,
    oys: np.ndarray,
    dxs: np.ndarray,
    dys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the distance matrix among N travelers.
    """
    oxs, oys, dxs, dys = arrs2vec([oxs, oys, dxs, dys])
    N = oxs.shape[1]
    # assure lengths are all the same
    assert oxs.shape == oys.shape == dxs.shape == dys.shape
    # not matrix operation, just vector operation for sov case
    mat_dx = np.abs(oxs - dxs)
    mat_dy = np.abs(oys - dys)
    mat_dist = np.sqrt(mat_dx ** 2 + mat_dy ** 2)
    return mat_dx, mat_dy, mat_dist


def get_distances_between_ods_matrix(
    oxs: np.ndarray,
    oys: np.ndarray,
    dxs: np.ndarray,
    dys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the distance matrix among N travelers.
    """
    oxs, oys, dxs, dys = arrs2vec([oxs, oys, dxs, dys])
    num_orig = oxs.shape[1]
    num_dest = dxs.shape[1]
    mat_dx = np.tile(oxs.transpose(), (1, num_dest))
    mat_dx = np.abs(mat_dx - np.tile(dxs, (num_orig, 1)))
    mat_dy = np.tile(oys.transpose(), (1, num_dest))
    mat_dy = np.abs(mat_dy - np.tile(dys, (num_orig, 1)))
    mat_dist = np.sqrt(mat_dx ** 2 + mat_dy ** 2)
    return mat_dx, mat_dy, mat_dist

