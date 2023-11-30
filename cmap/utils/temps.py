import dask.dataframe as dd
from glob import glob
from os.path import join, basename
import pandas as pd
import re

KELVIN_CST = 273.15


def read_and_process_csv(file):
    df = pd.read_csv(file, index_col=0).rename(columns={"id_point": "poi_id"})
    date = basename(file)[:-4]  # Remove the '.csv' extension
    df["date"] = pd.to_datetime(date)
    df["temperature"] = df["temperature"] - KELVIN_CST
    return df


def get_chunk_temps(in_root, chunk_id):
    temp_files = glob(join(in_root, "temperatures", str(chunk_id), "*.csv"))

    dfs = [
        dd.from_pandas(read_and_process_csv(file), npartitions=1) for file in temp_files
    ]
    temp_df = dd.concat(dfs)
    temp_df["chunk_id"] = chunk_id

    return temp_df


def write_poi_csv(poi_df, out_path, poi):
    sorted_poi_df = poi_df.sort_values(["date"])
    poi_filename = join(out_path, f"{poi}.csv")
    dd.from_pandas(sorted_poi_df, npartitions=1).to_csv(
        poi_filename, index=False, single_file=True
    )


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

            print(f"Writing {stage} data for chunk {chunk_id}")
            output_folder = join(out_path, stage, "features", "temperatures")

            # Group by 'poi_id' and write each group to a separate file
            for poi, poi_df in chunk_temp_df.groupby("poi_id"):
                write_poi_csv(poi_df, output_folder, poi)


if __name__ == "__main__":
    input_root = "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_23_tiles_01234"
    output_root = (
        "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_23_tiles_sorted"
    )

    get_temps(input_root, output_root)
