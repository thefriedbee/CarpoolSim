from sqlalchemy import create_engine

from carpoolsim.basic_settings import DB_URL
from carpoolsim.database.query_database import (
    query_od_info, 
    execute_sql_command
)


def test_query_od_info():
    engine = create_engine(DB_URL)
    od_info = query_od_info(engine, '1', '2')
    assert od_info is not None
    # example format
    # ('1', '2', 1.6642285714285716, ['1', '80483', '2'])
    assert od_info[0] == '1'
    assert od_info[1] == '2'
    assert isinstance(od_info[2], float)
    assert isinstance(od_info[3], list)
    for node_id in od_info[3]:
        assert isinstance(node_id, str)









