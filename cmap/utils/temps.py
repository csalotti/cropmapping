import dask
from dask import delayed
import dask.dataframe as dd
import os
from os.path import join, basename
import re

KELVIN_CST = 273.15


def read_and_process_csv(file):
    df = dd.read_csv(file, assume_missing=True)
    date = basename(file)[:-4]  # Remove the '.csv' extension
    df["date"] = dd.to_datetime(date)
    df["temperature"] = df["temperature"] - KELVIN_CST
    return df


def get_chunk_temps(in_root, chunk_id):
    temp_files = [
        join(in_root, "temperatures", str(chunk_id), f)
        for f in os.listdir(join(in_root, "temperatures", str(chunk_id)))
        if f.endswith(".csv")
    ]

    dfs = [read_and_process_csv(file) for file in temp_files]
    temp_df = dd.concat(dfs)
    temp_df["chunk_id"] = chunk_id
    return temp_df


@delayed
def write_poi_csv(chunk_temp_df, out_path):
    output_folder = join(out_path, "features", "temperatures")
    poi_id = chunk_temp_df["poi_id"].iloc[0]
    output_filename = join(output_folder, f"{poi_id}.csv")

    dd.from_dask_array(chunk_temp_df.to_dask_array(lengths=True)).to_csv(
        output_filename, index=False, single_file=True
    )


def get_temps(in_path, out_path):
    for stage in ["train", "eval"]:
        chunks = [
            re.findall(r"\d+", f)[0]
            for f in os.listdir(join(in_path, stage, "features"))
            if f.endswith(".csv")
        ]
        print(f"{stage} chunks {chunks}")

        tasks = []
        for chunk_id in chunks:
            print(f"Get chunk {chunk_id}")
            chunk_temp_df = get_chunk_temps(in_path, chunk_id)

            print(f"Writing {stage} data for chunk {chunk_id}")

            # Use dask.delayed for parallelizing writing for each poi_id
            tasks.extend(
                [
                    write_poi_csv(poi_df, out_path)
                    for _, poi_df in chunk_temp_df.groupby("poi_id")
                ]
            )

        dask.compute(*tasks, scheduler="processes")


if __name__ == "__main__":
    input_root = "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_23_tiles_01234"
    output_root = (
        "/mnt/sda/geowatch/datasets/hackathon/crop_mapping/fra_23_tiles_sorted"
    )

    get_temps(input_root, output_root)
