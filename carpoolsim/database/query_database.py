from sqlalchemy import text
import sqlalchemy


def query_od_info(
        engine: sqlalchemy.Engine,
        o_taz: str | int,
        d_taz: str | int,
):
    sql_cmd = f"SELECT * FROM dists WHERE origin='{o_taz}' AND destination='{d_taz}'"
    with engine.connect() as connection:
        results = connection.execute(text(sql_cmd)).fetchall()
        res = results[0]
        row_dist = res[2]
        # convert printed path in string to list
        row_path = res[3].strip("[]").replace("'", "").replace(" ", "").split(',')

    return str(o_taz), str(d_taz), row_dist, row_path


def execute_sql_command(
        engine: sqlalchemy.Engine,
        sql_command: str,
        mode: str,
):
    with engine.connect() as connection:
        if mode == "fetchall":
            results = connection.execute(text(sql_command)).fetchall()
        elif mode == "scalar":
            results = connection.execute(text(sql_command)).scalar()
        elif mode == "one":
            results = connection.execute(text(sql_command)).scalar()
        else:
            raise IOError(f"sql execute mode {mode} not recognized")
    # TODO: confirm the return type of results
    return results

