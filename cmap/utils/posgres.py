import pandas as pd
import os
from glob import glob
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

from cmap.utils.constants import (
    ALL_BANDS,
    DATE_COL,
    LABEL_COL,
    POINT_ID_COL,
    SEASON_COL,
    TEMP_COL,
)


def write_chunks(connection, root, type, cols, table_name):
    insert_query = sql.SQL(
        f"INSERT INTO {table_name}\n" + f"({','.join(cols)})\n" + f"VALUES %s"
    )
    cursor = connection.cursor()
    for stage in ["train", "eval"]:
        print(f"\t{stage}")
        files = glob(os.path.join(root, stage, type, "*.csv"))
        for file in files:
            print(f"\t\t{file}")
            df = pd.read_csv(file, index_col=0, parse_dates=["date"] if 'date' in cols else None)
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

    #print("Start features")
    #write_chunks(conn, ROOT, "features", features_col, "points")
    print("Start labels")
    write_chunks(conn, ROOT, "labels", labels_col, "labels")
    print("Start temperatures")
    write_chunks(conn, ROOT, "features/temperatures", temp_cols, "temperatures")
