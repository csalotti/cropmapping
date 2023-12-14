import os
from functools import partial
from glob import glob

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map


from .constants import DATE_COL, POINT_ID_COL, ALL_BANDS, SEASON_COL, LABEL_COL

COLS = {
    "features" : [POINT_ID_COL, DATE_COL],
    "labels": [POINT_ID_COL, SEASON_COL],
    "temperatures" : [POINT_ID_COL, DATE_COL],
}


def feat_season_filter(stage):
    start = 2018 if stage == "train" else 2020
    end = 2020 if stage == "train" else 2021
    return f"({DATE_COL} >= '{start}-11-01') & ({DATE_COL} < '{end}-12-01')"

def labels_season_filter(stage):
    seasons = [2021] if stage == 'val' else [2020, 2019] 
    return f"{SEASON_COL} in {seasons}"

def get_temp_folder(s_filter, f):
    df = pd.read_csv(f, engine="pyarrow", index_col=0).rename(
        columns={"id_point": POINT_ID_COL}
    )
    if DATE_COL not in df.columns:
        df[DATE_COL] = os.path.basename(f)[:-4]
    if "temperature" in df.columns:
        df["temperature"] -= 273.15
    
    return df.query(s_filter)

def get_features(s_filter, f):
    df = pd.read_csv(f, engine="pyarrow", index_col=0)
    df[[POINT_ID_COL, 'chunk_id', 'aws_index']] = df[[POINT_ID_COL, 'chunk_id', 'aws_index']].astype('category')
    df[ALL_BANDS] = df[ALL_BANDS].astype('uint16')
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.query(s_filter)

def get_labels(s_filter, f):
    df = pd.read_csv(f, engine="pyarrow", index_col=0)
    df[[POINT_ID_COL, SEASON_COL, LABEL_COL]] = df[[POINT_ID_COL, SEASON_COL, LABEL_COL]].astype('category')
    return df.query(s_filter)

def merge_chunks(src, dst, dname, dfilter, dget, dcols, max_workers=2, chunksize=10):
    for stage in ["train", "val"]:
        s_filter = dfilter(stage) 
        in_folder = "eval" if stage == 'val' else stage
        files = glob(os.path.join(src, in_folder, dname, "*.csv"))
        fn = partial(dget, s_filter)
        data = process_map(fn, files, max_workers=max_workers, chunksize=10)
        df = pd.concat(data, ignore_index=True).sort_values(dcols)
        df.to_parquet(os.path.join(dst, stage, f"{dname}.pq"), index=False)

def convert_temps(src, dst, max_workers=2):
    for stage in ["train", "val"]:
        # Convert
        s_filter = season_filter(stage) 
        ids = range(80) if stage == "train" else range(80, 100)
        for i in tqdm(ids):
            files = glob(os.path.join(src , "temperatures", str(i), "*.csv"))
            fn = partial(get_temp_folder, s_filter)
            temp_data = process_map(fn, files,max_workers=max_workers, chunksize=10)
            temps_df = pd.concat(temp_data, ignore_index=True).sort_values([POINT_ID_COL, DATE_COL])
            temps_df.to_parquet(os.path.join(dst, stage, "temperatures", f"{i}.pq"), index=False)

def merge_temps(root):
    for stage in ["val", "train"]:
        # Merge
        files = glob(os.path.join(root, stage, "temperatures", "*.pq"))
        ids = pd.read_parquet(os.path.join(root, stage, "labels.pq"), columns=[POINT_ID_COL])[POINT_ID_COL].tolist()
        schema = pq.ParquetFile(files[0]).schema_arrow
        with pq.ParquetWriter(os.path.join(root, stage, "temperatures.pq"), schema=schema) as writer:
            for f in tqdm(files):
                writer.write_table(pq.read_table(f, schema=schema, filters=[(POINT_ID_COL, "in", ids)]))

def interesect_ids(src):
    print("Intersection")
    for stage in ["val", "train"]:
        print(f"\t{stage}")
        inter_ids = set()
        for dname in ["temperatures", "features", "labels"]:
            df =  pd.read_parquet(os.path.join(src, stage, f"{dname}.pq"), columns=[POINT_ID_COL])
            df_ids = set(df[POINT_ID_COL].unique().tolist())
            if len(inter_ids) == 0:
                inter_ids = df_ids
            else:
                inter_ids = inter_ids & df_ids


        for dname in ["temperatures", "features", "labels"]:
            df =  pd.read_parquet(os.path.join(src, stage, f"{dname}.pq"), filters=[(POINT_ID_COL, "in", inter_ids)])
            assert len(df[POINT_ID_COL].unique()) == len(inter_ids)
            df.sort_values(COLS[dname]).to_parquet(os.path.join(src, stage, f"{dname}.pq"), index=False)

if __name__ == "__main__":
    src = "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_23_tiles_01234"
    dst = "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_19_21"

    #merge_chunks(
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
