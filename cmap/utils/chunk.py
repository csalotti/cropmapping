from os.path import basename

import pandas as pd

from cmap.utils.constants import (CHUNK_ID_COL, DATE_COL, POINT_ID_COL,
                                  SIZE_COL, START_COL)


def get_indexes(chunk_df: pd.DataFrame):
    chunk_id = chunk_df[CHUNK_ID_COL].unique()[0]
    chunk_df = (
        chunk_df.sort_values([POINT_ID_COL, DATE_COL])
        .reset_index(drop=True)
        .drop(columns=[cn for cn in chunk_df.columns if "Unnamed" in cn])
    )
    indexes_df = chunk_df[[POINT_ID_COL]].assign(id=chunk_df.index)
    indexes_df = indexes_df.groupby(POINT_ID_COL).agg(["first", "last"])
    indexes_df.columns = indexes_df.columns.map("_".join)
    indexes_df = (
        # fmt: off
            indexes_df
            .assign(chunk_id=chunk_id)
            .assign(size=lambda x: x["id_last"] - x["id_first"])
            .rename(columns={"id_first" : START_COL})
        # fmt: on
    )

    return indexes_df, chunk_df


def get_temperatures(temp_files):
    chunk_data = []
    # Iterate through each CSV file in the chunk folder
    for csv_file in temp_files:
        df = pd.read_csv(csv_file, index_col=0).rename(columns={"id_point": "poi_id"})

        # Extract the date from the file name
        date = basename(csv_file)[:-4]  # Remove the '.csv' extension
        df["date"] = pd.to_datetime(date)

        # Append the DataFrame to the combined_data DataFrame
        chunk_data.append(df)

    chunk_temp_df = pd.concat(chunk_data).reset_index()

    return chunk_temp_df


def preprocessing(features_df, temp_files=[]):
    indexes_df, features_df = get_indexes(features_df)

    indexes_json = indexes_df.reset_index()[
        [POINT_ID_COL, START_COL, SIZE_COL, CHUNK_ID_COL]
    ].to_dict(orient="records")

    return indexes_json, features_df
