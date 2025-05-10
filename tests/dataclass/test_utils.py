import pytest

from carpoolsim.dataclass.utils import *
from tests.test_load_example import trips, tazs, df_nodes, df_links, pnr_lots


def test_attach_coord2taz_node():
    trips[["ox", "oy", "o_node"]] = trips.apply(
        attach_coord2taz_node,
        axis=1, df_nodes=df_nodes, mode="orig"
    )
    assert trips["ox"].notna().all()
    assert trips["oy"].notna().all()
    assert trips["o_node"].notna().all()
    trips[["dx", "dy", "d_node"]] = trips.apply(
        attach_coord2taz_node,
        axis=1, df_nodes=df_nodes, mode="dest"
    )
    assert trips["dx"].notna().all()
    assert trips["dy"].notna().all()
    assert trips["d_node"].notna().all()


def test_get_contained_taz():
    pnr_lots['taz'] = pnr_lots['geometry'].apply(
        get_contained_taz,
        tazs=tazs
    )
    assert pnr_lots['taz'].notna().all()
    assert pnr_lots['taz'].isin(tazs['taz_id']).all()


