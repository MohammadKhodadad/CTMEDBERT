import os
import shutil

import pandas as pd


def data_read_drg_codes(args):
    file_path = f'{args.data_dir}/physionet.org/files/mimiciv/3.0/hosp/drgcodes.csv.gz'

    print("\n=============================================")
    print(f"Reading DRG code file from: \n{file_path}")
    return pd.read_csv(file_path)


def data_write_drg_code_files(args, all_data):
    file_path = f"{args.data_dir}/MIMIC-Processed-Data/discharge"

    print("\n=============================================")
    print(f"Writing DRG code CSV files to: \n{file_path}")

    all_data.to_csv(f"{file_path}/drgcodes.csv", index=False)
    all_data.iloc[:100].to_csv(f"{file_path}/100_drgcodes.csv", index=False)


def data_read_discharge_file(args):
    file_path = f'{args.data_dir}/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz'

    print("\n=============================================")
    print(f"Reading discharge file from: \n{file_path}")
    return pd.read_csv(file_path)


def data_write_discharge_files(args, all_data):
    file_path = f"{args.data_dir}/MIMIC-Processed-Data/discharge"

    print("\n=============================================")
    print(f"Writing discharge CSV files to: \n{file_path}")

    all_data.to_csv(f"{file_path}/discharges.csv", index=False)
    all_data.iloc[:100].to_csv(f"{file_path}/100_discharge.csv", index=False)


def data_write_discharge_file_processed(args, all_data):
    file_path = f"{args.data_dir}/MIMIC-Processed-Data/discharge"

    print("\n=============================================")
    print(f"Writing PROCESSED discharge CSV files to: \n{file_path}")

    all_data.to_csv(f"{file_path}/discharge_processed.csv", index=False)
    all_data.iloc[:100].to_csv(f"{file_path}/100_discharge_processed.csv", index=False)


def data_read_discharge_file_chunks_csv(args, chunk_size=1000, ):
    file_path = f"{args.data_dir}/MIMIC-Processed-Data/discharge/discharges.csv"

    print("\n=============================================")
    print(f"Reading discharge file CHUNKS from: \n{file_path}")

    return pd.read_csv(file_path, chunksize=chunk_size)


def create_output_directory(args, delete_old: bool = False) -> bool:
    """
    Creates the output directory.
    """

    output_directory_path = os.path.abspath(f"{args.data_dir}/MIMIC-Processed-Data/discharge")

    print("\n=============================================")
    print("Checking directory: ")
    print(output_directory_path)

    if delete_old and os.path.exists(output_directory_path):
        shutil.rmtree(output_directory_path)
        print("\nOld output directory removed.")

    if not os.path.exists(output_directory_path):
        os.mkdir(output_directory_path)
        print("New output directory created.")
        print("Directory structure done.")
        return True
    else:
        print("Output directory already exist.")
        print("Directory structure done.\n")
        return True
