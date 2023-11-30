from glob import glob
from os.path import join, basename
import pandas as pd
import re

KELVIN_CST = 273.15


def get_chunk_temps(in_root, chunk_id):
    temp_files = glob(join(in_root, "temperatures", str(chunk_id), "*.csv"))
    temps_list = []
    for f in temp_files:
        df = pd.read_csv(f, index_col=0).rename(columns={"id_point": "poi_id"})
        date = basename(f)[:-4]  # Remove the '.csv' extension
        df["date"] = pd.to_datetime(date)
        df["temperature"] = df["temperature"] - KELVIN_CST

        temps_list.append(df)
    temp_df = pd.concat(temps_list).reset_index(drop=True)
    return temp_df


def get_temps(in_path, out_path):
    for stage in ["train", "eval"]:
        chunks = [
            re.findall(r"\d+", basename(f))[0]
            for f in glob(join(in_path, stage, "features", "*.csv"))
        ]
        print(f"{stage} chunks {chunks}")
        for chunk_id in chunks:
            print(f"Get chunk {chunk_id}")
            chunk_temp_df = get_chunk_temps(in_path, chunk_id)
            print(f"Writing chunk {chunk_id}")
            for poi, df in chunk_temp_df.groupby("poi_id"):
                df.to_csv(
                    join(
                        out_path,
                        stage,
                        "features",
                        "temperatures",
                        f"{poi}.csv",
                    )
                )


if __name__ == "__main__":
    input_root = "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_23_tiles_01234"
    output_root = (
        "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_23_tiles_sorted"
    )

    get_temps(input_root, output_root)
