import os
from functools import partial
from glob import glob
from typing import Callable, List

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map


from .constants import DATE_COL, POINT_ID_COL, ALL_BANDS, SEASON_COL, LABEL_COL

COLS = {
    "features": [POINT_ID_COL, DATE_COL],
    "labels": [POINT_ID_COL, SEASON_COL],
    "temperatures": [POINT_ID_COL, DATE_COL],
}


def date_season_filter(stage):
    """Season filterinig per date

    Args:
        stage : Train or val

    """
    start = 2018 if stage == "train" else 2020
    end = 2020 if stage == "train" else 2021
    return f"({DATE_COL} >= '{start}-11-01') & ({DATE_COL} < '{end}-12-01')"


def year_season_filter(stage):
    """Season filterinig per year

    Args:
        stage : Train or val

    """
    seasons = [2021] if stage == "val" else [2020, 2019]
    return f"{SEASON_COL} in {seasons}"


def get_temp_folder(f):
    """Retrieve temperatures data from foldeer

    Args:
        f : File path
    """
    df = pd.read_csv(f, engine="pyarrow", index_col=0).rename(
        columns={"id_point": POINT_ID_COL}
    )
    if DATE_COL not in df.columns:
        df[DATE_COL] = pd.to_datetime(os.path.basename(f)[:-4])
    if "temperature" in df.columns:
        df["temperature"] -= 273.15
        df["temperature"] = df["temperature"].astype("uint8")

    return df.query(s_filter)


def get_features(s_filter, f):
    """Retrieve features

    Args:
        s_filter : Season filter
        f: File path

    Returns:
        features dataframe with convereted fileds
    """
    df = pd.read_csv(f, engine="pyarrow", index_col=0)
    df[["chunk_id", "aws_index"]] = df[["chunk_id", "aws_index"]].astype("category")
    df[ALL_BANDS] = df[ALL_BANDS].astype("uint16")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.query(s_filter)


def get_labels(s_filter, f):
    """Retrieve labels

    Args:
        s_filter : Season filter
        f: File path

    Returns:
        features dataframe with convereted fileds
    """
    df = pd.read_csv(f, engine="pyarrow", index_col=0)
    df[[SEASON_COL, LABEL_COL]] = df[[SEASON_COL, LABEL_COL]].astype("category")
    return df.query(s_filter)


def merge_chunks(
    src: str,
    dst: str,
    dname: str,
    dfilter: Callable,
    dget: Callable,
    dcols: List[str],
    max_workers: int = 2,
    chunksize: int = 10,
):
    """Retrieve files chunks and merge them into single parquet files

    Args:
        src (str) : Files source folders
        dst (str) : Files destination folders
        dname (str) : Data name
        dfilter (Callable) : season fitler function
        dget (Callable) : data get function
        dcols (List[str]) : columns names
        max_workers (int , default 2): number of parrallel workers
        chunksize (int , defaut 10): Chunk for multiprocessing
    """
    for stage in ["train", "val"]:
        s_filter = dfilter(stage)
        in_folder = "eval" if stage == "val" else stage
        files = glob(os.path.join(src, in_folder, dname, "*.csv"))
        fn = partial(dget, s_filter)
        data = process_map(fn, files, max_workers=max_workers, chunksize=chunksize)
        df = pd.concat(data, ignore_index=True).sort_values(dcols)
        df.to_parquet(os.path.join(dst, stage, f"{dname}.pq"), index=False)


def convert_temps(src, dst, max_workers=2):
    for stage in ["train", "val"]:
        # Convert
        s_filter = date_season_filter(stage)
        ids = range(80) if stage == "train" else range(80, 100)
        for i in tqdm(ids):
            files = glob(os.path.join(src, "temperatures", str(i), "*.csv"))
            temp_data = process_map(
                get_temp_folder, files, max_workers=max_workers, chunksize=10
            )
            temps_df = pd.concat(temp_data, ignore_index=True).sort_values(
                [POINT_ID_COL, DATE_COL]
            )
            temps_df = temps_df.query(s_filter)
            temps_df.to_parquet(
                os.path.join(dst, stage, "temperatures", f"{i}.pq"), index=False
            )


def merge_temps(root: str):
    """Merge temperatures parquet files parts into a single one

    Args:
        root (str) : Files root
    """
    for stage in ["val", "train"]:
        # Merge
        files = glob(os.path.join(root, stage, "temperatures", "*.pq"))
        ids = pd.read_parquet(
            os.path.join(root, stage, "labels.pq"), columns=[POINT_ID_COL]
        )[POINT_ID_COL].tolist()
        schema = pq.ParquetFile(files[0]).schema_arrow
        with pq.ParquetWriter(
            os.path.join(root, stage, "temperatures.pq"), schema=schema
        ) as writer:
            for f in tqdm(files):
                writer.write_table(
                    pq.read_table(f, schema=schema, filters=[(POINT_ID_COL, "in", ids)])
                )


def interesect_ids(src: str):
    """Filter labels, features and temperatures s.t. all
    tables shares the same ids. Write them as parquet afterwards

    Args:
        src: Files root

    """
    print("Intersection")
    for stage in ["val", "train"]:
        print(f"\t{stage}")
        inter_ids = set()
        for dname in ["temperatures", "features", "labels"]:
            df = pd.read_parquet(
                os.path.join(src, stage, f"{dname}.pq"), columns=[POINT_ID_COL]
            )
            df_ids = set(df[POINT_ID_COL].unique().tolist())
            if len(inter_ids) == 0:
                inter_ids = df_ids
            else:
                inter_ids = inter_ids & df_ids

        for dname in ["temperatures", "features", "labels"]:
            df = pd.read_parquet(
                os.path.join(src, stage, f"{dname}.pq"),
                filters=[(POINT_ID_COL, "in", inter_ids)],
            )
            assert len(df[POINT_ID_COL].unique()) == len(inter_ids)
            df.sort_values(COLS[dname]).to_parquet(
                os.path.join(src, stage, f"{dname}.pq"), index=False
            )


if __name__ == "__main__":
    src = "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_23_tiles_01234"
    dst = "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_19_21"

    # merge_chunks(
    #        src,
    #        dst,
    #        "features",
    #        feat_season_filter,
    #        get_features,
    #        [POINT_ID_COL, DATE_COL],
    #        max_workers=16,
    #        chunksize=40
    #        )
    # merge_chunks(
    #         src,
    #         dst,
    #         "labels",
    #         labels_season_filter,
    #         get_labels,
    #         [POINT_ID_COL, SEASON_COL],
    #         max_workers=16,
    #         chunksize=40
    #         )
    # interesect_ids(dst)
    merge_temps(dst)
