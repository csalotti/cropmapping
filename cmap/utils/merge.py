import os
from functools import partial
from glob import glob

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map


from .constants import DATE_COL, POINT_ID_COL


def season_filter(stage):
    start = 2018 if stage == "train" else 2020
    end = 2020 if stage == "train" else 2021
    return f"({DATE_COL} >= '{start}-11-01') & ({DATE_COL} < '{end}-12-01')"

def parquet_filter(stage):
    start = 2018 if stage == "train" else 2020
    end = 2020 if stage == "train" else 2021
    return [(DATE_COL, ">=", f"{start}-11-01") , (DATE_COL, "<", f"{end}-12-01")]

def get_temp_folder(s_filter, f):
    df = pd.read_csv(f, engine="pyarrow", index_col=0).rename(
        columns={"id_point": POINT_ID_COL}
    )
    if DATE_COL not in df.columns:
        df[DATE_COL] = os.path.basename(f)[:-4]
    if "temperature" in df.columns:
        df["temperature"] -= 273.15
    
    return df.query(s_filter)

def convert_to_parquet(src, dst, max_workers=2):
    for stage in ["train", "val"]:
        s_filter = season_filter(stage) 
        ids = range(80) if stage == "train" else range(80, 100)
        for i in tqdm(ids):
            files = glob(os.path.join(src , "temperatures", str(i), "*.csv"))
            fn = partial(get_temp_folder, s_filter)
            temp_data = process_map(fn, files,max_workers=max_workers, chunksize=10)
            temps_df = pd.concat(temp_data, ignore_index=True).sort_values([POINT_ID_COL, DATE_COL])
            temps_df.to_parquet(os.path.join(dst, stage, "temperatures", f"{i}.pq"), index=False)



if __name__ == "__main__":
    src = "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_23_tiles_01234"
    dst = "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_19_21"

   convert_to_parquet(src, dst, max_workers=16)
