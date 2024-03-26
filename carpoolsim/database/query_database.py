from sqlalchemy import text
from sqlalchemy import create_engine


def query_od_info(engine, o_taz, d_taz):
    sql_cmd = f"SELECT * FROM dists WHERE origin='{o_taz}' AND destination='{d_taz}'"
    with engine.connect() as connection:
        results = connection.execute(text(sql_cmd)).fetchall()
        res = results[0]
        row_dist = res[2]
        # convert printed path in string to list
        row_path = res[3].strip("[]").replace("'", "").replace(" ", "").split(',')

    return str(o_taz), str(d_taz), row_dist, row_path


def execute_sql_command(engine, sql_command, mode):
    with engine.connect() as connection:
        if mode == "fetchall":
            results = connection.execute(text(sql_command)).fetchall()
        elif mode == "scalar":
            results = connection.execute(text(sql_command)).scalar()
        elif mode == "one":
            results = connection.execute(text(sql_command)).scalar()
        else:
            raise IOError(f"sql execute mode {mode} not recognized")
    return results

