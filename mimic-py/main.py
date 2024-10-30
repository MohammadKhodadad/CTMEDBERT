import pandas as pd
from lib.discharge_clean import clean_discharge_in_chunks
from util.io import (data_read_discharge_file,
                     data_write_discharge_files,
                     data_write_discharge_file_processed,
                     data_read_discharge_file_chunks_csv,
                     create_output_directory,
                     data_read_drg_codes, data_write_drg_code_files)
from util.arg_parser import read_arguments


def main():
    """
    Provide absolute path to data directory containing physionet.org directory
    python3 main.py -d /Users/../SO_Data

    Set --full_run to run full data
    python3 main.py -f -d /Users/../SO_Data
    :return:
    """
    args = read_arguments()

    # Create output directories.
    create_output_directory(args)

    # Read zip file and write csv files.
    discharge_dataframe = data_read_discharge_file(args)
    data_write_discharge_files(args, discharge_dataframe)
    dgr_code_dataframe = data_read_drg_codes(args)
    data_write_drg_code_files(args, dgr_code_dataframe)

    # Processes discharge data from csv files.
    discharge_dataframe = data_read_discharge_file_chunks_csv(args, chunk_size=1000)
    cleaned_discharge_dataframe = clean_discharge_in_chunks(args, discharge_dataframe)
    data_write_discharge_file_processed(args, cleaned_discharge_dataframe)


if __name__ == "__main__":
    main()
