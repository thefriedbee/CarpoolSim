import math
import warnings

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx

import carpoolsim.basic_settings as bs
import carpoolsim.dataclass.utils as ut
warnings.filterwarnings('ignore')


def project_gpd_to_local(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """
    Project a GeoDataFrame to a local coordinate system.
    """
    gdf = gdf.to_crs(crs)
    return gdf


def convert_all_df_column_names_to_lower(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [cn.lower() for cn in df.columns]
    return df


def initialize_abm15_links(
    df_nodes_raw: gpd.GeoDataFrame,
    df_links_raw: gpd.GeoDataFrame,
    drop_connector: bool = True,
    speed_column: str | None = None,
) -> pd.DataFrame:
    """
    Import shape-file to GeoDataFrame, convert the format based on our needs.
    (Combining all useful information from 2 .shp file to one file)
    Store the output file back to .shp file and return the DataFrame.
    After prepared, output .shp file could be used the next time and
    there is no need to call this function again.
    :return: GeoDataFrame of network link information, along with its end node information.
    """
    # keep all column names to lower cases
    df_nodes_raw = convert_all_df_column_names_to_lower(df_nodes_raw)
    df_links_raw = convert_all_df_column_names_to_lower(df_links_raw)
    df_nodes_raw = project_gpd_to_local(df_nodes_raw, bs.CRS)
    df_links_raw = project_gpd_to_local(df_links_raw, bs.CRS)

    df_nodes = df_nodes_raw[['nid', 'x', 'y', 'lat', 'lon']]
    # notice FACTTYPE zeros stands for all connectors; FACTTYPE over 50 stands for transit links or its access!
    if drop_connector:
        df_links_raw = df_links_raw[df_links_raw['factype'] > 0][df_links_raw['factype'] < 50]
    else:
        df_links_raw = df_links_raw[df_links_raw['factype'] < 50]
    # if there is speed column
    if speed_column is not None:
        df_links_raw['speed_limit'] = df_links_raw[speed_column]

    if 'speed_limit' in df_links_raw.columns.tolist():
        spd = 'speed_limit'
    else:
        spd = 'speed_limi'

    if 'a_b' not in df_links_raw.columns.tolist():
        df_links_raw['a_b'] = df_links_raw.apply(lambda x: str(int(x['a'])) + '_' + str(int(x['b'])), axis=1)

    # IMPORTANT: this step set the default traveling speed given road type!!!
    mapper = {0: 35, 1: 70, 2: 70, 3: 65, 4: 65, 7: 35, 10: 35, 11: 70, 14: 35}  # edited dl 09072020
    df_links_raw['tmp'] = df_links_raw['factype'].map(mapper)
    df_links_raw['tmp'] = df_links_raw['tmp'].fillna(35)
    df_links_raw.loc[df_links_raw[spd] == 0, spd] = df_links_raw.loc[df_links_raw[spd] == 0, 'tmp']
    df_links = df_links_raw[['a', 'b', 'a_b', 'name', 'geometry', spd, 'distance', 'factype']]

    # add node information to links
    df_links = df_links.merge(
        df_nodes.rename(columns={
            'nid': 'a',
            'x': 'ax', 'y': 'ay',
            'lat': 'a_lat', 'lon': 'a_lon'}),
        how='left', on='a'
    )
    df_links = df_links.merge(
        df_nodes.rename(columns={
            'nid': 'b',
            'x': 'bx', 'y': 'by',
            'lat': 'b_lat', 'lon': 'b_lon'}),
        how='left', on='b'
    )

    def abm15_assign_grid(df_links, grid_size=25000.0):
        for col in ['minx', 'miny', 'maxx', 'maxy']:
            df_links[col + '_sq'] = round(df_links['geometry'].bounds[col] / grid_size, 0)
        return df_links

    df_links = abm15_assign_grid(df_links)
    df_links = gpd.GeoDataFrame(df_links, geometry=df_links['geometry'], crs=df_links.crs)
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
    # convert all column names to lower cases from this function...
    df_links.columns = [cn.lower() for cn in df_links.columns]

    # The measure to use as cost of traverse a link
    col = 'time'  # can expand to other measures, like considering grades, etc.

    def compute_link_cost(x, method):
        # in case there are still links with 0 speed value
        x[method] = x['distance'] / x['speed_limi'] if x['speed_limi'] > 0 else 30  # mile / mph = hour
        # could implement other methods based on needs (e.g., consider grades, etc.)
        return x[method]

    df_links[col] = df_links.apply(compute_link_cost, axis=1, method=col)

    DGo = nx.DiGraph()  # directed graph
    for ind, row2 in df_links.iterrows():
        # forward graph, time stored as minutes
        # dist stored link length in miles, forward/backward stores the key value of the travel time.
        DGo.add_weighted_edges_from(
            [(str(row2['a']), str(row2['b']), float(row2[col]) * 60.0)],
            weight='forward', dist=row2['distance'], name=row2['a_b']
        )
        # add its backward links
        DGo.add_weighted_edges_from(
            [(str(row2['b']), str(row2['a']), float(row2[col]) * 60.0)],
            weight='backward', dist=row2['distance'], name=row2['a_b']
        )

    for ind, row2 in df_links.iterrows():
        # iterate all edges, if a link has no 'forward' weights, set it to large number like 10000 to block it
        # Do the same thing for 'backward' weights
        if 'forward' not in DGo[str(row2['b'])][str(row2['a'])].keys():
            DGo[str(row2['b'])][str(row2['a'])]['forward'] = 1e6
        if 'backward' not in DGo[str(row2['a'])][str(row2['b'])].keys():
            DGo[str(row2['a'])][str(row2['b'])]['backward'] = 1e6
    # construct backward network
    return DGo


# add projected coordinate (x,y) given (lon, lat)
def add_xy(
    df: pd.DataFrame | gpd.GeoDataFrame,
    lat: str, lon: str,
    x: str, y: str,
    x_sq: str, y_sq: str,
    grid_size: float = 25000.0,
    reset_crs: bool = True,
) -> gpd.GeoDataFrame:
    """
    Given (lat, lon) information, generate coordinates in local projection system
        Also, classify location into different categories using grids and 
        store the row and column the point falls into.
    """
    if isinstance(df, pd.DataFrame) or reset_crs:
        crs = {'init': 'epsg:4326', 'no_defs': True}  # NAD83: EPSG 4326
        geometry = [Point(xy) for xy in zip(df[lon], df[lat])]
        df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    df = project_gpd_to_local(df, bs.CRS)
    df[x] = df['geometry'].apply(lambda x: x.coords[0][0])
    df[y] = df['geometry'].apply(lambda x: x.coords[0][1])
    df[x_sq] = round(df[x] / grid_size, 0)
    df[y_sq] = round(df[y] / grid_size, 0)
    return df


def point_to_node(
    df_points: gpd.GeoDataFrame,
    df_links: gpd.GeoDataFrame,
    use_grid: bool = False,
    walk_speed: float = 2.0,
    grid_size: float = 25000.0,
    dist_thresh: float = 5280.0,
    is_origin: bool = True,
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
    df_points = project_gpd_to_local(df_points, bs.CRS)
    df_links = project_gpd_to_local(df_links, bs.CRS)

    def find_grid(pt_x):
        return round(pt_x / grid_size), 0

    def define_grid_id(df_pts):
        df_pts['x_sq'] = df_pts['geometry'].apply(lambda x: find_grid(x.coords[0][0]))
        df_pts['y_sq'] = df_pts['geometry'].apply(lambda x: find_grid(x.coords[0][1]))
        return df_pts

    def find_closest_link(point, lines):
        dists = lines.distance(point)
        return [dists.idxmin(), dists.min()]

    def calculate_dist(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    # INITIALIZATION
    if use_grid:
        df_points = define_grid_id(df_points)
    df_points['node_id'] = 0
    df_points['node_t'] = 0
    # CALCULATION
    print('{} points to prepare'.format(len(df_points)))
    i = 0
    for ind, row in df_points.iterrows():
        i += 1
        try:
            # find links in the grid
            filt1 = (df_links['minx_sq'] <= row['x_sq']) & (row['x_sq'] <= df_links['maxx_sq'])
            filt2 = (df_links['miny_sq'] <= row['y_sq']) & (row['y_sq'] <= df_links['maxy_sq'])
            df_links_i = df_links[filt1 & filt2]
            # if one cannot find df links, it is an external point, try to pick a highway access link
            if len(df_links_i) == 0:
                df_points.loc[ind, 'node_id'] = -1
                df_points.loc[ind, 'node_t'] = -1
                continue
            # find the closest link and the distance
            link_id_dist = find_closest_link(row.geometry, gpd.GeoSeries(df_links_i.geometry))
            linki = df_links_i.loc[link_id_dist[0], :]
            # find the closest node on the link
            df_coords = df_points.loc[ind, 'geometry'].coords[0]
            dist1 = calculate_dist(df_coords[0], df_coords[1], linki['ax'], linki['ay'])
            dist2 = calculate_dist(df_coords[0], df_coords[1], linki['bx'], linki['by'])
            if (dist1 > dist_thresh) and (dist2 > dist_thresh):
                df_points.loc[ind, 'node_id'] = -1
                df_points.loc[ind, 'node_t'] = -1
            else:
                df_points.loc[ind, 'node_id'] = linki['a'] if dist1 < dist2 else linki['b']
                df_points.loc[ind, 'node_t'] = dist1 / walk_speed / 5280.0 if \
                    dist1 < dist2 else dist2 / walk_speed / 5280.0
            # add distance o_d, d_d to dataframe
            df_points.loc[ind, 'dist'] = min(dist1, dist2) / 5280.0
        except Exception as e:
            print('Error happens: ', e)
            print('Trip number is', ind)
            if is_origin:
                print(row['x_sq'], row['y_sq'], row['orig_lat'], row['orig_lon'])
            else:
                print(row['x_sq'], row['y_sq'], row['dest_lat'], row['dest_lon'])
            df_points.loc[ind, 'node_id'] = -1
            df_points.loc[ind, 'node_t'] = 0
    if i % 100 == 0 and i > 90:
        print("Finished prepare {} points to nearest network nodes!".format(i))
    return df_points


def pnr_filter_within_TAZs(
    pnr_lots: gpd.GeoDataFrame,
    tazs: gpd.GeoDataFrame,
) -> pd.DataFrame:
    pnr_lots = pnr_lots.to_crs('epsg:4326')
    # for each point, search if it is contained in polygon
    pnr_lots['taz'] = pnr_lots['geometry'].apply(ut.get_taz, tazs=tazs)

    filt = (pnr_lots.taz != -1)
    num_lots = pnr_lots.shape[0]
    print(f"Filtered {num_lots - filt.sum()} points out of {num_lots} (because they are not in any TAZ polygon)")
    pnr_lots = pnr_lots.loc[filt, :].copy()
    return pnr_lots


def pnr_add_projection(
    pnr_lots: gpd.GeoDataFrame,
    network_config: 'NetworkConfig',
) -> pd.DataFrame:
    df_links = network_config.links
    # network parameters
    walk_speed = network_config.walk_speed
    grid_size = network_config.grid_size
    ntp_dist_thresh = network_config.ntp_dist_thresh

    pnr_lots = add_xy(
        pnr_lots,
        'lat', 'lon',
        'x', 'y',
        'x_sq', 'y_sq',
        grid_size=grid_size
    )

    pnr_lots = point_to_node(
        pnr_lots, df_links, False, walk_speed,
        grid_size, ntp_dist_thresh
    ).rename(columns={
        'node_id': 'node',
        'node_t': 't',
        'dist': 'd'
    })

    # make sure same results if this function is called multiple times
    duplicate_columns = pnr_lots.columns.duplicated()
    pnr_lots = pnr_lots.loc[:, ~duplicate_columns]
    return pnr_lots
