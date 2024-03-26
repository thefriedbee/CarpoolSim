import networkx as nx
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import (
    MultipleLocator,
    FormatStrFormatter
)
import numpy as np


# a function creates the accurate graph representations!
def create_nx_graph(mat, trips, self_loop=False, undirected=True):
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


# a function for plotting graph (in the correct formula)!
def plot_graph_simple(G: nx.DiGraph):
    plt.figure(figsize=(20, 5))
    # default positions
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color='black', node_shape="x")  # nodes
    nx.draw_networkx_edges(G, pos, width=1, edge_color='grey')  # edges
    plt.title('Shareability Network Graph of a Trip Clusters (default layout)')
    return plt.gca()


# a function for plotting graph (in the correct formula)!
def plot_graph_along_time(G, springY=True, edge_color='grey', edge_alpha=0.7, continued=False, fig=None, ax=None):
    if not continued:
        fig, ax = plt.subplots(figsize=(20, 5))
    # default positions
    pos = nx.spring_layout(G)
    pos_y = [pos[i][1] for i in range(len(pos))]
    pos_x = [n[1]['depart_min'] for i, n in enumerate(G.nodes(data=True))]
    # node pos random.uniform(0, 1)
    if springY:
        dct = {i: (pos_x[i], pos_y[i]) for i in range(len(pos_x))}
        plt.ylabel('Spring layout on Y axis')
    else:
        a = math.pi / 4
        dct = {i: (pos_x[i], math.sin(a * i)) for i in range(len(pos_x))}
        plt.ylabel('Use sin value on y axis in turn for visualization')
    nx.draw_networkx_nodes(G, dct, node_size=20,
                           node_color='black', node_shape="x", alpha=0.7)  # nodes
    nx.draw_networkx_edges(G, dct, width=1,
                           edge_color=edge_color, alpha=edge_alpha)  # edges
    # add lines for viz
    for pX in pos_x:
        plt.axvline(x=pX, ymin=0.0, ymax=1.0, color='grey', lw=0.5, alpha=0.5)
    plt.title('Shareability Network Graph')
    plt.xlabel('depart time (hour) (5 min minor ticks)')
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.tick_params(axis="x", direction="out", which="both", right=False, top=False)
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.xaxis.set_major_locator(MultipleLocator(60))

    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    x_labels = ax.get_xticks()
    # formatter = FuncFormatter(lambda x_val, tick_pos: "$2^{{+{:.0f}}}$".format(np.log2(x_val)))
    ax.set_xticklabels([str(int(x / 60)) for x in x_labels])
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    return fig, ax


# Post analysis after assigning all trips
# write a function to regenerate summary histograms for each trip
def plot_hist(trip_sum, on="time", title_info=None):
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
def plot_diff_hist(trip_sum, on="time", title_info=None):
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
