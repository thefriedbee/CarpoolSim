"""
Contains function to visually check the behavior of individual trips (SOV trips, carpool trips, etc.)
"""

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

import carpoolsim.basic_settings as bs

plt.rcParams.update({'font.size': 22})


# a helper method to plot carpool routes for visualization
def plot_seq(
        trip_seq: list[str],
        gpd_df: pd.DataFrame,
        skeleton: str | None = None,
        ctd: bool = False,
        color: str = 'red',
        linewidth: int = 3,
        ax: plt.Axes = None,
        plt_arrow: bool = True,
        arrow_color: str = 'black',
        trip_df: pd.DataFrame = None,
        station_df: pd.DataFrame = None,
) -> tuple[plt.Figure, plt.Axes, list]:
    """
    trip_seq: a list of link id (includes 'origin' and 'destination')
    gdf_df: Geopandas dataframe (shapefile) of abm15
    skeleton: a list of link ID.
        If it is None, DO NOT plot skeleton;
        If set as 'highway', those will be plotted as skeleton traffic network (e.g., I-85).
        Otherwise, input is a list of link segments.
    trip_df: If not None, corresponding input used for running RoadwaySim.
        This is just for plotting origin destination coordinates.
    station_df: the station used by two trips
    """
    # convert trip_df to geopataframe, reproject the trip dataframe
    from pyproj import Proj, transform
    in_crs = Proj(init=bs.CRS)
    out_crs = Proj(init='EPSG:3857')
    out_crs_code = 3857
    # store all legend handles
    handles = []
    if ctd is False:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        ax.set_aspect('equal')
    if trip_df is not None:
        # t1 is for driver, t2 is for passenger
        t1, t2 = trip_df.iloc[0], trip_df.iloc[1]
        point_o1 = transform(in_crs, out_crs, t1['ox'], t1['oy'])
        point_o2 = transform(in_crs, out_crs, t1['dx'], t1['dy'])
        point_d1 = transform(in_crs, out_crs, t2['ox'], t2['oy'])
        point_d2 = transform(in_crs, out_crs, t2['dx'], t2['dy'])
        h1, = plt.plot(*point_o1, 'r^', label='driver origin')
        h2, = plt.plot(*point_o2, 'ro', label='driver destination')
        h3, = plt.plot(*point_d1, 'b^', label='passenger origin')
        h4, = plt.plot(*point_d2, 'bo', label='passenger destination')
        handles += [h1, h2, h3, h4]
    if station_df is not None:
        pnr1 = station_df.iloc[0]
        station_m = transform(in_crs, out_crs, pnr1['x'], pnr1['y'])
        h5, = plt.plot(*station_m, 'ks', label='PNR (parking lots)')
        handles.append(h5)

    # load skeleton network for visualization (if needed)
    if skeleton == 'highway':
        trip_df_skeleton = gpd_df.loc[gpd_df['name'].str.contains('I-', na=False), :]
    elif skeleton is not None:  # input should be a list of names
        trip_df_skeleton = gpd_df.loc[gpd_df['name'].str.contains(skeleton, na=False), :]

    if skeleton == 'highway' or skeleton is not None:
        # project skeleton network
        gpd_df.to_crs(epsg=out_crs_code, inplace=True)
        ax = trip_df_skeleton.plot(ax=ax, color='gray', alpha=0.2, label="skeleton highways")

    # load trip path to plot path arrows or polygons
    # convert sequential node ID to link ID
    As, Bs = trip_seq[0:-1:1], trip_seq[1:len(trip_seq):1]
    trip_seq = [a + '_' + b for a, b in zip(As, Bs)]  # link IDs
    trip_df = gpd_df.loc[gpd_df['a_b'].isin(trip_seq), :]
    trip_df.to_crs(epsg=out_crs_code, inplace=True)  # re-project before plot
    # use merge method to keep the sequence of trajectory correct
    trip_seq = pd.DataFrame(trip_seq, columns=['seq'])
    trip_df = trip_df.merge(trip_seq, left_on='a_b', right_on='seq', how='left')
    trip_df.plot(ax=ax, color=color, alpha=0.3, linewidth=linewidth)

    if plt_arrow:
        for index, row in trip_df.iterrows():
            # original projection in feet
            dx = abs(row.bx - row.ax)
            dy = abs(row.by - row.ay)
            # only plot arrow if link length is greater than 2000 ft.
            head_width = 0
            if dx + dy > 2000:
                head_width = 300
            # don't plot arrow is distance if link smaller than a mile
            a_x, a_y = transform(in_crs, out_crs, row.ax, row.ay)
            b_x, b_y = transform(in_crs, out_crs, row.bx, row.by)
            ax.arrow(
                a_x, a_y, b_x - a_x, b_y - a_y,
                shape='left', width=150, color=arrow_color, head_width=head_width,
                length_includes_head=True
            )

    # create legend manually
    if skeleton is not None:
        skeleton_gray = mpatches.Patch(color='gray', label='skeleton network')
        handles.append(skeleton_gray)
    return plt.gcf(), ax, handles
