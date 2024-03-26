import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# a function creates the accurate graph representations!
def create_nx_graph(
        mat: np.ndarray,
        trips: pd.DataFrame | pd.Series,  # TODO: check if type annotation is accurate
        self_loop: bool = False,
        undirected: bool = True,
):
    mat = mat.copy()
    if self_loop is False:
        np.fill_diagonal(mat, 0)
    if undirected is True:
        # only keeps the upper diagonal matrix
        mat = np.triu(mat)
    G = nx.from_numpy_matrix(mat)
    # load depart time
    depart_lst = trips['newmin'].tolist()
    # iterate nodes (store depart time as datetime object)
    dct = {i: t + 180 for i, t in enumerate(depart_lst)}
    nx.set_node_attributes(G, dct, "depart_min")
    return G


# Post analysis after assigning all trips
# write a function to regenerate summary histograms for each trip
def plot_hist(
        trip_sum: pd.DataFrame,
        on: str = "time",
        title_info: str | None = None
) -> plt.Axes:
    """
    Given trip summary info, plot histogram for interested variable, which could be time, speed and distance.
    :param trip_sum: trip summary DataFrame table
    :param on: the variable to operate on, could be "time", "speed", and "dist"
    :param title_info: if not None, if not None, print the title based on given info
    :return: plt object
    """
    plt.figure(figsize=(18, 6))
    label_str = ''
    if on == "time":
        plt.hist(trip_sum['before_time'], bins=list(range(0, 30, 1)), alpha=0.5,
                 label='Before Case')
        plt.hist(trip_sum['after_time'], bins=list(range(0, 30, 1)), alpha=0.5,
                 label='After Case')
        plt.xlim(0, 30)
        plt.xlabel('Minutes')
    elif on == "dist":
        plt.hist(trip_sum['before_dist'], bins=list(range(0, 30, 1)), alpha=0.5,
                 label='Before Case ({}) '.format(label_str))
        plt.hist(trip_sum['after_dist'], bins=list(range(0, 30, 1)), alpha=0.5,
                 label='After Case ({}) '.format(label_str))
        plt.xlim(0, 30)
        plt.xlabel('Miles')
    plt.legend()
    if title_info is not None:
        title_str = '{on} of Trips Departing From {range} {lane}'.format(**title_info)
        plt.title(title_str)
    plt.ylabel('Counts')
    return plt.gca()


# plot saved time
def plot_diff_hist(
        trip_sum: pd.DataFrame,
        on: str = "time",
        title_info: str | None = None
) -> plt.Axes:
    """
    Plot the changed amount for between before case and after case.
    :param trip_sum: trip summary DataFrame table
    :param on: the variable to operate on, could be "time", "speed", and "dist"
    :param title_info: if not None, print the title based on given info
    :return: plt object
    """
    plt.figure(figsize=(12, 3))
    if on == "time":
        plt.hist(trip_sum['after_time'] - trip_sum['before_time'],
                 bins=list(range(-10, 20, 1)), alpha=0.5, label='Reroute Time (min)')
        plt.xlim(-10, 30)
        plt.xlabel('Minutes')
    elif on == "dist":
        plt.hist(trip_sum['after_dist'] - trip_sum['before_dist'],
                 bins=list(range(-10, 20, 1)), alpha=0.5, label='Reroute Mileage (mile)')
        plt.xlim(-20, 20)
        plt.xlabel('Miles')
    plt.legend()
    if title_info is not None:
        title_str = 'Saved {on} of Trips Departing From {range} {lane}'.format(**title_info)
        if on == 'dist':
            title_str = 'Changed {on} of Trips Departing From {range} {lane}'.format(**title_info)
        plt.title(title_str)
    plt.ylabel('Counts')
    return plt.gca()
