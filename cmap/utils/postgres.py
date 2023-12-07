import pandas as pd
import os
from glob import glob
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from tqdm import tqdm

from cmap.utils.constants import (
    ALL_BANDS,
    DATE_COL,
    LABEL_COL,
    POINT_ID_COL,
    SEASON_COL,
    TEMP_COL,
)


def write_chunks(
    connection,
    root,
    type,
    cols,
    table_name,
    check: bool = False,
    stages=["train", "eval"],
):
    insert_query = sql.SQL(
        f"INSERT INTO {table_name}\n" + f"({','.join(cols)})\n" + f"VALUES %s"
    )

    cursor = connection.cursor()
    already_written = set()
    if check:
        print(f"\tCheck already writtent samples")
        server_cursor = connection.cursor("server_side")
        query = f"SELECT DISTINCT poi_id from {table_name};"
        server_cursor.execute(query)
        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break
            already_written.update(row[0] for row in rows)

    for stage in stages:
        print(f"\t{stage}")
        files = glob(os.path.join(root, stage, type, "*.csv"))
        for file in tqdm(files):
            poi_id = os.path.basename(file)[:-4]
            if poi_id not in already_written:
                df = pd.read_csv(
                    file, index_col=0, parse_dates=["date"] if "date" in cols else None
                )
                df = df[cols]
                data_to_insert = [tuple(row) for row in df.values]
                execute_values(
                    cursor, insert_query, data_to_insert, page_size=len(data_to_insert)
                )

                connection.commit()


if __name__ == "__main__":
    ROOT = "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_23_tiles_sorted/"
    db_name = "crops"
    user = "michel"
    password = "cmap2023"
    host = "localhost"
    port = 6666
    conn = psycopg2.connect(
        dbname=db_name, user=user, password=password, host=host, port=port
    )

    features_col = [
        POINT_ID_COL,
        "tile_id",
        DATE_COL,
        "aws_index",
        "R20m_SCL",
    ] + ALL_BANDS
    labels_col = [POINT_ID_COL, SEASON_COL, LABEL_COL]
    temp_cols = [POINT_ID_COL, DATE_COL, "tile_id", TEMP_COL]

    # print("Start features")
    # write_chunks(conn, ROOT, "features", features_col, "points")
    # print("Start labels")
    # write_chunks(conn, ROOT, "labels", labels_col, "labels")
    print("Start temperatures")
    write_chunks(
        conn,
        ROOT,
        "features/temperatures",
        temp_cols,
        "temperatures",
        check=True,
        stages=["trai"],
    )
