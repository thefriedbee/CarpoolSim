import os
import math
import warnings

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx

warnings.filterwarnings('ignore')


def initialize_abm15_links(
        drop_connector: bool = True,
        spd_column: str | None = None,
        input_folder: str | None = None,
        output_folder: str | None = None,
):
    """
    Import shape-file to GeoDataFrame, convert the format based on our needs.
    (Combining all useful information from 2 .shp file to one file)
    Store the output file back to .shp file and return the DataFrame.
    After prepared, output .shp file could be used the next time and
    there is no need to call this function again.
    :return: GeoDataFrame of network link information, along with its end node information.
    """
    # prepare input/output paths
    if input_folder is None:
        input_folder = os.path.join(os.environ['PROJ_LIB'], 'data_inputs', 'ABM2020 203K')
    if output_folder is None:
        output_folder = os.path.join(os.environ['PROJ_LIB'], 'data_outputs', 'ABM2020 203K')

    file_name_nodes = os.path.join('2020 nodes with latlon', '2020_nodes_latlon.shp')
    file_name_links = os.path.join('2020 links', '2020_links.shp')
    df_nodes_raw = gpd.read_file(os.path.join(input_folder, file_name_nodes))
    df_links_raw = gpd.read_file(os.path.join(input_folder, file_name_links))

    df_nodes = df_nodes_raw[['N', 'X', 'Y', 'lat', 'lon']]
    # notice FACTTYPE zeros stands for all connectors; FACTTYPE over 50 stands for transit links or its access!
    if drop_connector:
        df_links_raw = df_links_raw[df_links_raw['FACTYPE'] > 0][df_links_raw['FACTYPE'] < 50]
    else:
        df_links_raw = df_links_raw[df_links_raw['FACTYPE'] < 50]
    # if there is speed column
    if spd_column is not None:
        df_links_raw['SPEEDLIMIT'] = df_links_raw[spd_column]
    if 'SPEEDLIMIT' in df_links_raw.columns.tolist():
        spd = 'SPEEDLIMIT'
    else:
        spd = 'SPEED_LIMI'

    if 'A_B' not in df_links_raw.columns.tolist():
        df_links_raw['A_B'] = df_links_raw.apply(lambda x: str(int(x['A'])) + '_' + str(int(x['B'])), axis=1)

    # IMPORTANT: this step set the default traveling speed given road type!!!
    mapper = {0: 35, 1: 70, 2: 70, 3: 65, 4: 65, 7: 35, 10: 35, 11: 70, 14: 35}  # edited dl 09072020
    # mapper = {0: 35, 3: 65, 4: 65, 7: 35, 10: 35, 11: 55, 14: 35} # edited hl 04092019
    df_links_raw['tmp'] = df_links_raw['FACTYPE'].map(mapper)
    df_links_raw['tmp'] = df_links_raw['tmp'].fillna(35)
    df_links_raw.loc[df_links_raw[spd] == 0, spd] = df_links_raw.loc[df_links_raw[spd] == 0, 'tmp']
    df_links = df_links_raw[['A', 'B', 'A_B', 'geometry', spd, 'DISTANCE', 'NAME', 'FACTYPE']]
    df_links = df_links.merge(df_nodes.rename(columns={'N': 'A', 'X': 'Ax', 'Y': 'Ay', 'lat': 'A_lat', 'lon': 'A_lon'}),
                              how='left', on='A')
    df_links = df_links.merge(df_nodes.rename(columns={'N': 'B', 'X': 'Bx', 'Y': 'By', 'lat': 'B_lat', 'lon': 'B_lon'}),
                              how='left', on='B')

    def abm15_assignGrid(df_links, grid_size=25000.0):
        for col in ['minx', 'miny', 'maxx', 'maxy']:
            df_links[col + '_sq'] = round(df_links['geometry'].bounds[col] / grid_size, 0)
        return df_links

    df_links = abm15_assignGrid(df_links)
    df_links = gpd.GeoDataFrame(df_links, geometry=df_links['geometry'], crs=df_links.crs)
    if output_folder is None:
        output_folder = os.path.join(
            os.environ['PROJ_LIB'],
            'data_outputs',
            'step2',
            'abm_links_processed.shp'
        )
    df_links.to_file(
        output_folder
    )
    return df_links


def build_carpool_network(
        df_links: pd.DataFrame,
) -> nx.DiGraph:
    """
    Given original network like the DataFrame of abm15, create directed graph
     for shortest paths searching using the package networkx.
    Notice that in this is the vehicular speed, and its free flow speed is already estimated through
    the function initialize_abm15_links, so we don't need to set default speed anymore.
    :param df_links: The whole network, like the whole abm15 geo-dataframe
    :return: DGo: directed link graph.
    """
    # TODO: maybe prepare bike speed in previous step instead of this simplification
    # The measure to use as cost of traverse a link
    col = 'time'  # can expand to other measures, like considering grades, etc.

    def compute_link_cost(x, method):
        # in case there are still links with 0 speed value
        x[method] = x['DISTANCE'] / x['SPEED_LIMI'] if x['SPEED_LIMI'] > 0 else 30  # mile / mph = hour
        # could implement other methods based on needs (e.g., consider grades, etc.)
        return x[method]

    df_links[col] = df_links.apply(compute_link_cost, axis=1, method=col)

    DGo = nx.DiGraph()  # directed graph
    for ind, row2 in df_links.iterrows():
        # forward graph, time stored as minutes
        # dist stored link length in miles, forward/backward stores the key value of the travel time.
        DGo.add_weighted_edges_from([(str(row2['A']), str(row2['B']), float(row2[col]) * 60.0)],
                                    weight='forward', dist=row2['DISTANCE'], name=row2['NAME'])
        # add its backward links
        DGo.add_weighted_edges_from([(str(row2['B']), str(row2['A']), float(row2[col]) * 60.0)],
                                    weight='backward', dist=row2['DISTANCE'], name=row2['NAME'])

    for ind, row2 in df_links.iterrows():
        # iterate all edges, if a link has no 'forward' weights, set it to large number like 10000 to block it
        # Do the same thing for 'backward' weights
        if 'forward' not in DGo[str(row2['B'])][str(row2['A'])].keys():
            DGo[str(row2['B'])][str(row2['A'])]['forward'] = 1e6
        if 'backward' not in DGo[str(row2['A'])][str(row2['B'])].keys():
            DGo[str(row2['A'])][str(row2['B'])]['backward'] = 1e6
    # construct backward network
    return DGo


# add (x,y) given (lon, lat)
def add_xy(
        df: pd.DataFrame,
        lat: str, lon: str,
        x: str, y: str,
        x_sq: str, y_sq: str,
        grid_size: float = 25000.0
) -> pd.DataFrame:
    """
    Given (lat, lon) information, generate coordinates in local projection system
        Also, classify location into different categories using a grid and store the
        row and column it falls into.
    """
    crs = {'init': 'epsg:4326', 'no_defs': True}  # NAD83: EPSG 4326
    geometry = [Point(xy) for xy in zip(df[lon], df[lat])]
    df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    df = df.to_crs(epsg=2240)  # Georgia West (ftUS):  EPSG:2240
    df[x] = df['geometry'].apply(lambda x: x.coords[0][0])
    df[y] = df['geometry'].apply(lambda x: x.coords[0][1])
    df[x_sq] = round(df[x] / grid_size, 0)
    df[y_sq] = round(df[y] / grid_size, 0)
    return df


def point_to_node(
        df_points: pd.DataFrame,
        df_links: pd.DataFrame,
        use_grid: bool = False,
        walk_speed: float = 2.0,
        grid_size: float = 25000.0,
        dist_thresh: float = 5280.0,
        is_origin: bool = True,
        freeway_links: bool = False,
) -> pd.DataFrame:
    """
    Given a column of location projected to local coordinates (x, y), find nearest node in the network,
     record the node ID and the distance to walk to the node.
    Arguments:
        df_points: a DataFrame containing projected coordinates.
                   Each row corresponds to one point.
        df_links: GeoDataFrame network files like abm15.shp,
                  each row denotes a directed link with two end nodes A and B.
        use_grid: If False, compute the grid it false into. If True, grid info is stored in de_points.
        walk_speed: walking speed default is 2.0 mph.
        grid_size: (I guess it should be) the width of the grid. Default is 25000 ft or 4.7 mile.
        dist_thresh: the maximum distance a normal person willing walk. Default is 1 mile.

    Returns:
        df_points: expand same input DataFrame with information about the nearest node and
                   walking time from point to the node.
    """

    def find_grid(pt_x):
        return round(pt_x / grid_size), 0

    def define_gridid(df_pts):
        df_pts['x_sq'] = df_pts['geometry'].apply(lambda x: find_grid(x.coords[0][0]))
        df_pts['y_sq'] = df_pts['geometry'].apply(lambda x: find_grid(x.coords[0][1]))
        return df_pts

    def find_closestLink(point, lines):
        dists = lines.distance(point)
        # print(dists)
        # print('dists shapes', dists.shape)
        return [dists.idxmin(), dists.min()]

    def calculate_dist(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    # INITIALIZATION
    if use_grid:
        df_points = define_gridid(df_points)
    df_points['NodeID'] = 0
    df_points['Node_t'] = 0
    # CALCULATION
    print('{} points to prepare'.format(len(df_points)))
    i = 0
    for ind, row in df_points.iterrows():
        i += 1
        try:
            # find links in the grid
            df_links_i = df_links[df_links['minx_sq'] <= row['x_sq']][df_links['maxx_sq'] >= row['x_sq']][
                df_links['maxy_sq'] >= row['y_sq']][df_links['miny_sq'] <= row['y_sq']]
            # print(df_links_i.shape)
            # if cannot find df links, it is an external point, try to pick a highway access link
            if len(df_links_i) == 0:
                df_links_i = freeway_links
            # print('# of links in the grid', len(df_links_i))
            # print(df_links_i.index)
            # find the closest link and the distance
            LinkID_Dist = find_closestLink(row.geometry, gpd.GeoSeries(df_links_i.geometry))
            linki = df_links_i.loc[LinkID_Dist[0], :]
            # find the closest node on the link
            df_coords = df_points.loc[ind, 'geometry'].coords[0]
            # print('coords', df_coords)
            dist1 = calculate_dist(df_coords[0], df_coords[1], linki['Ax'], linki['Ay'])
            dist2 = calculate_dist(df_coords[0], df_coords[1], linki['Bx'], linki['By'])
            if (dist1 > dist_thresh) and (dist2 > dist_thresh):
                df_points.loc[ind, 'NodeID'] = -1
                df_points.loc[ind, 'Node_t'] = -1
            else:
                df_points.loc[ind, 'NodeID'] = linki['A'] if dist1 < dist2 else linki['B']
                df_points.loc[ind, 'Node_t'] = dist1 / walk_speed / 5280.0 if \
                    dist1 < dist2 else dist2 / walk_speed / 5280.0
            # add distance o_d, d_d to dataframe
            df_points.loc[ind, 'dist'] = min(dist1, dist2) / 5280.0
        except Exception as e:
            print('Error happens!', e)
            print('Trip number is', ind)
            if is_origin:
                print(row['x_sq'], row['y_sq'], row['ori_lat'], row['ori_lon'])
            else:
                print(row['x_sq'], row['y_sq'], row['dest_lat'], row['dest_lon'])
            df_points.loc[ind, 'NodeID'] = -1
            df_points.loc[ind, 'Node_t'] = 0
    if i % 100 == 0 and i > 90:
        print("Finished prepare {} points to nearest network nodes!".format(i))
    return df_points


def samp_pre_process(filename, dict_settings, input_as_df=False):
    df_links = dict_settings['network']['bike']['links']
    freeway_links = dict_settings['network']['bike']['links']
    walk_speed, grid_size, ntp_dist_thresh = dict_settings['walk_speed'], dict_settings['grid_size'], dict_settings[
        'ntp_dist_thresh']

    if input_as_df:
        df_points = filename
    else:
        df_points = pd.read_csv(filename)
    df_points = add_xy(df_points, 'ori_lat', 'ori_lon', 'x', 'y', 'x_sq', 'y_sq', grid_size=grid_size)

    df_points = point_to_node(df_points, df_links, False, walk_speed, grid_size, ntp_dist_thresh,
                              freeway_links=freeway_links) \
        .rename(columns={'NodeID': 'o_node', 'Node_t': 'o_t', 'x': 'ox', 'y': 'oy',
                         'x_sq': 'ox_sq', 'y_sq': 'oy_sq', 'dist': 'o_d'})

    df_points = add_xy(df_points, 'dest_lat', 'dest_lon', 'x', 'y', 'x_sq', 'y_sq', grid_size=grid_size)
    df_points = point_to_node(df_points, df_links, False, walk_speed, grid_size, ntp_dist_thresh,
                              freeway_links=freeway_links) \
        .rename(columns={'NodeID': 'd_node', 'Node_t': 'd_t', 'x': 'dx', 'y': 'dy',
                         'x_sq': 'dx_sq', 'y_sq': 'dy_sq', 'dist': 'd_d'})

    # origin, destination should be of different OD nodes and must found ODs to continue
    df_points = df_points[
        df_points['o_node'] != -1][
        df_points['d_node'] != -1][
        df_points['o_node'] != df_points['d_node']
    ]
    if input_as_df is False:  # record information
        df_points.to_csv(filename.replace('.csv', '_node.csv').replace('samples_in', 'samples_out'), index=False)
    return df_points


def pnr_pre_process(
        filename: str,
        dict_settings: dict,
        input_as_df: bool = False,
) -> pd.DataFrame:
    df_links = dict_settings['network']['bike']['links']
    freeway_links = dict_settings['network']['bike']['links']
    walk_speed = dict_settings['walk_speed']
    grid_size = dict_settings['grid_size']
    ntp_dist_thresh = dict_settings['ntp_dist_thresh']

    if input_as_df:
        df_points = filename
    else:
        df_points = pd.read_csv(filename)

    df_points = add_xy(df_points, 'lat', 'lon', 'x', 'y', 'x_sq', 'y_sq', grid_size=grid_size)
    # still use prefix o_ for convenience
    df_points = point_to_node(
        df_points, df_links, False, walk_speed,
        grid_size, ntp_dist_thresh,
        freeway_links=freeway_links) \
        .rename(
        columns={'NodeID': 'node', 'Node_t': 't', 'x': 'x', 'y': 'y',
                 'x_sq': 'x_sq', 'y_sq': 'y_sq', 'dist': 'd'}
    )
    return df_points

