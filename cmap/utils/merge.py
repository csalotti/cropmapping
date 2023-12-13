import os
from glob import glob

import pandas as pd
from tqdm import tqdm

from .constants import DATE_COL, POINT_ID_COL


def season_filter(stage):
    start = 2018 if stage == "train" else 2020
    end = 2020 if stage == "train" else 2021
    return f"({DATE_COL} >= '{start}-11-01') & ({DATE_COL} < '{end}-12-01')"


def merge_temps(src, dst, data_name):
    print(data_name)
    for stage in ["train", "val"]:
        ids = range(12, 80) if stage == "train" else range(80, 100)
        data = []
        for i in tqdm(ids):
            for f in tqdm(glob(os.path.join(src, "temperatures", str(i), "*.csv"))):
                df = pd.read_csv(f, engine="pyarrow", index_col=0).rename(
                    columns={"id_point": POINT_ID_COL}
                )
                if DATE_COL not in df.columns:
                    df[DATE_COL] = os.path.basename(f)[:4]
                if "temperature" in df.columns:
                    df["temperature"] -= 273.15
                df = df.query(season_filter(stage))
                data.append(df)
            pd.concat(data, ignore_index=True).sort_values(
                [POINT_ID_COL, DATE_COL]
            ).to_parquet(os.path.join(dst, stage, data_name, f"{i}.pq"))


if __name__ == "__main__":
    src = "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_23_tiles_01234"
    dst = "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_19_21"

    merge_temps(src, dst, "temperatures")
