"""
Visualization code
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as cx

from carpoolsim.carpool.trip_cluster_abstract import TripClusterAbstract
from carpoolsim.visualization.carpool_viz_seq import plot_seq
plt.rcParams.update({'font.size': 22})


def plot_single_trip(
    trip_cluster: TripClusterAbstract,
    intind1: str, intind2: str,
    network_df: pd.DataFrame,
    fixed_role: bool = False,
):
    """
    Plot carpool trip for a single trip corresponding to the matrix position (ind 1, ind 2)
    Plot three carpool plans at once. A picks up B, B picks up A, one plot showing both schemes.
    :param intind1: integer index 1 for the driver
    :param intind2: integer index 2 for the passenger
    :param network_df: network file containing geometry information for plotting
    :return: lastly used plotting axes
    """
    # unload parameters
    trips = trip_cluster.trips
    solo_dists = trip_cluster.solo_dists
    solo_paths = trip_cluster.solo_paths
    int2idx = trip_cluster.int2idx
    tt_matrix_p1 = trip_cluster.tt_matrix_p1

    # print depart time diff
    tod_1, tod_2 = trips.iloc[intind1, :]['new_min'], trips.iloc[intind2, :]['new_min']
    tt1 = tt_matrix_p1[intind1, intind2]
    sb = [ScaleBar(1, "m", dimension="si-length", fixed_value=5*1000/0.621371,
                    box_alpha=0.5, location="upper left",
                    scale_formatter=lambda value, unit: f'{round(0.621371 * value/1000)} mile')
            for i in range(2)]
    
    network_df = network_df.to_crs("3857")
    dists_1, links_1, dists_2, links_2 = trip_cluster.compute_carpool(
        intind1, intind2,
        print_dist=False, fill_mat=False
    )

    # check shared route
    d2, p2 = solo_dists[intind2], solo_paths[intind2]
    d1, p1 = solo_dists[intind1], solo_paths[intind1]
    # intind1 is driver, intind2 is passenger
    # driver's trip or the whole trip
    fig1, ax1, h1 = plot_seq(
        trip_seq=links_1, gpd_df=network_df,
        linewidth=2, color='red', arrow_color='red',
        trip_df=trips.iloc[[intind1, intind2], :],
        skeleton="I-285"
    )

    # passenger's trip or the shared trip
    fig1, ax1, h2 = plot_seq(
        trip_seq=p2, gpd_df=network_df,
        ctd=True, color='green', linewidth=20,
        ax=ax1, plt_arrow=False
    )
    plt.title(
        f"Driver: {intind1}({tod_1}+{tt1:.2f}min); Passenger: {intind2}({tod_2}min)"
    )

    idx1, idx2 = int2idx[intind1], int2idx[intind2]
    rec_driver = [idx1, "dc_driver", links_1]
    rec_passenger = [idx2, "dc_passenger", p2]
    rec_combined = [rec_driver, rec_passenger]

    # helper function
    def make_legend_arrow(
        legend, orig_handle, xdescent, ydescent,
        width, height, fontsize
    ):
        p = mpatches.FancyArrow(0, 0.5 * height, width, 0,
                                length_includes_head=True, head_width=0.75 * height,
                                shape='left')
        return p
    plt.legend(handles=h1+h2+[mpatches.Patch(color='green', label='shared trip', alpha=0.3)],
                handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow)},
                loc='lower right', fontsize=20)

    # add base map for fig1
    cx.add_basemap(ax1, source=cx.providers.CartoDB.Voyager)
    ax1.add_artist(sb[0])
    ax1.axis("off")

    # If fixed rule, stop plotting!!!
    if fixed_role:
        return fig1, ax1, rec_combined

    # intind2 is driver, intind1 is passenger
    fig2, ax2, h3 = plot_seq(
        trip_seq=links_2, gpd_df=network_df,
        color='red', arrow_color='red', linewidth=2,
        ax=None, trip_df=trips.iloc[[intind2, intind1], :]
    )
    fig2, ax2, h4 = plot_seq(
        trip_seq=p1, gpd_df=network_df,
        ctd=True, color='green', linewidth=20,
        ax=ax2, plt_arrow=False
    )
    plt.legend(handles=h3+h4+[mpatches.Patch(color='green', label='shared trip', alpha=0.3)],
                handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow)},
                loc='lower right', fontsize=20)
    # add base map for fig2
    plt.title(
        f"Driver: {intind2}({tod_2}min); Passenger: {intind1}({tod_1}min)"
    )
    cx.add_basemap(ax2, source=cx.providers.CartoDB.Voyager)
    ax2.add_artist(sb[1])
    ax2.axis("off")
    return fig1, fig2


def plot_single_trip_pnr(
    trip_cluster,
    intind1: int, intind2: int,
    network_df: pd.DataFrame,
    fixed_role: bool = False,
):
    trips = trip_cluster.trips
    links = trip_cluster.links
    int2idx = trip_cluster.int2idx
    pnr_access_info = trip_cluster.pnr_access_info
    parking_lots = trip_cluster.parking_lots

    idx1, idx2 = int2idx[intind1], int2idx[intind2]
    sid = trips.loc[idx1, 'pnr'][0]
    
    # print depart time diff
    tod_1, tod_2 = trips.loc[idx1, :]['new_min'], trips.loc[idx2, :]['new_min']
    tt1, tt2 = None, None

    if pnr_access_info[intind1, sid] is not None:
        tt1 = pnr_access_info[intind1, sid][1]
    if pnr_access_info[intind2, sid] is not None:
        tt2 = pnr_access_info[intind2, sid][1]
    # scale bar
    sb = [ScaleBar(1, "m", dimension="si-length", fixed_value=5*1000/0.621371,
                    box_alpha=0.5, location="upper left",
                    scale_formatter=lambda value, unit: f'{round(0.621371 * value/1000)} mile')
            for i in range(2)]

    # dists_1, links_1, dists_2, links_2
    dists_1, (links_1p, links_1d0, links_1d1, links_1d2), \
        dists_2, (links_2p, links_2d0, links_2d1, links_2d2), station_id = \
        trip_cluster.compute_carpool_pnr(intind1, intind2, print_dist=False, fill_mat=False)

    # intind1 is driver, intind2 is passenger
    # (red arrow): vehicle trajectory
    fig1, ax1, h1 = plot_seq(
        trip_seq=links_1d0+links_1d1+links_1d2, gpd_df=network_df,
        linewidth=2, color='red', arrow_color='red',
        trip_df=trips.iloc[[intind1, intind2], :],
        station_df=parking_lots.iloc[[station_id]]
    )

    # (trip segment 1): from driver start --> meet point
    fig1, ax1, h2 = plot_seq(
        trip_seq=links_1p, gpd_df=network_df, ctd=True,
        linewidth=10, color='blue',
        ax=ax1, plt_arrow=False
    )

    # (trip segment 2): from meet point --> end
    fig1, ax1, h2 = plot_seq(
        trip_seq=links_1d1, gpd_df=network_df, ctd=True,
        linewidth=20, color='green',
        ax=ax1, plt_arrow=False
    )

    rec_driver = [idx1, "pnr_driver", links_1d0+links_1d1+links_1d2]
    rec_passenger = [idx2, "pnr_passenger", links_1d1]
    rec = [rec_driver, rec_passenger]

    # plot coordinate of station id
    plt.title(
        f"Driver: {intind1}({tod_1}+{tt1:.2f}min); Passenger: {intind2}({tod_2}min) (Park and Ride)"
    )
    cx.add_basemap(ax1, source=cx.providers.CartoDB.Voyager)
    ax1.add_artist(sb[0])
    ax1.axis("off")
    # helper function
    def make_legend_arrow(legend, orig_handle, xdescent, ydescent,
                            width, height, fontsize):
        p = mpatches.FancyArrow(0, 0.5 * height, width, 0,
                                length_includes_head=True, head_width=0.75 * height,
                                shape='left')
        return p
    
    plt.legend(handles=(h1 + h2 + [mpatches.Patch(color='green', label='shared trip', alpha=0.3)] +
                        [mpatches.Patch(color='blue', label="passenger's drive to PNR", alpha=0.3)]),
                handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow)},
                loc='lower right', fontsize=20)

    # If fixed rule, stop plotting!!!
    if fixed_role:
        return fig1, ax1, rec

    # intind2 is driver, intind1 is passenger
    fig2, ax2, h3 = plot_seq(
        trip_seq=links_2d0+links_2d1+links_2d2, gpd_df=network_df,
        color='red', arrow_color='red', linewidth=2,
        ax=None, trip_df=trips.iloc[[intind2, intind1], :],
        station_df=parking_lots.iloc[[station_id]]
    )
    fig2, ax2, h4 = plot_seq(
        trip_seq=links_2p, gpd_df=network_df, ctd=True,
        linewidth=10, color='blue', arrow_color='blue',
        ax=ax2, plt_arrow=False
    )
    fig2, ax2, h5 = plot_seq(
        trip_seq=links_2d1, gpd_df=network_df, ctd=True,
        linewidth=20, color='green',
        ax=ax2, plt_arrow=False
    )

    # plt.title("Driver: {}; Passenger: {} (Park and Ride)".format(intind2, intind1))
    plt.title(
        f"Driver: {intind2}({tod_2}+{tt2:.2f}min); Passenger: {intind1}({tod_1}min) (Park and Ride)"
    )
    plt.legend(handles=(h3+h4+[mpatches.Patch(color='green', label='shared trip', alpha=0.3)]
                        + [mpatches.Patch(color='blue', label="passenger's drive to PNR", alpha=0.3)]),
                handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow)},
                loc='lower right', fontsize=20)
    cx.add_basemap(ax2, source=cx.providers.CartoDB.Voyager)
    ax2.add_artist(sb[1])
    ax2.axis("off")
    return (fig1, ax1), (fig2, ax2)

