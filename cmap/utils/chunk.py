import re
from glob import glob
from os.path import join

import pandas as pd
from pandas.io.parquet import json

from cmap.utils.constants import (CHUNK_ID_COL, DATE_COL, POINT_ID_COL,
                                  SIZE_COL, START_COL)


def chunks_indexing(chunks_root: str, write_csv: bool = False):
    files = glob(join(chunks_root, "*.csv"))
    indexes = []
    for f in files:
        chunk_df = pd.read_csv(f)
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
        indexes.extend(
            indexes_df.reset_index()[
                [POINT_ID_COL, START_COL, SIZE_COL, CHUNK_ID_COL]
            ].to_dict(orient="records")
        )
        chunk_df.to_csv(f, index=False)

    with open(join(chunks_root, "indexes.json"), "w") as fw:
        json.dump(indexes, fw)
