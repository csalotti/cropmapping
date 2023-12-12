import os
from glob import glob
from tqdm import tqdm
import pandas as pd
from .constants import DATE_COL, POINT_ID_COL

def season_filter(stage):
    start = 2018 if stage == "train" else 2020
    end = 2020 if stage == "train" else 2021
    return f"({DATE_COL} >= '{start}-11-01') & ({DATE_COL} < '{end}-12-01')"

def merge(src, dst, folder, data_name):
    print(data_name)
    for stage in [ "train", "val"]:
        ids = range(80) if stage == "train" else range(80,100)
        data = []
        for i in tqdm(ids):
            for f in tqdm(glob(os.path.join(src,"temperatures", str(i), "*.csv"))):
                df = pd.read_csv(f,engine='pyarrow', index_col=0)
                if DATE_COL not in df.columns:
                    df[DATE_COL] = os.path.basename(f)[:4]
                if 'temperature' in df.columns:
                    df['temperature'] -= 273.15
                data.append(df)
        pd.concat(data, ignore_index=True).sort_values([POINT_ID_COL, DATE_COL]).to_parquet(os.path.join(dst,stage,f"{data_name}.pq"))

    
if __name__ == "__main__":

    src = '/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_23_tiles_01234'
    dst = '/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_19_21'

    merge(src, dst, "temperatures", "temperatures")

