import argparse
from pathlib import Path


def read_arguments():
    _parser = argparse.ArgumentParser(
        prog="Discharge data clean util",
        description="MIMIC discharge notes clean util",
    )
    _parser.add_argument('-dd', '--data_dir',
                         type=str,
                         required=False,
                         default=f"{Path.home()}/SO_Data",
                         help="Path to data directory (default: ~/SO_Data)")

    _parser.add_argument('-f', '--full_run',
                         help="Test run with 1000 patient",
                         action="store_true",
                         default=False)

    return _parser.parse_args()
