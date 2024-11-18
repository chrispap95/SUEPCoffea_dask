"""
This script finds all datasets in the top directory of the supplied EOS path
and recursively searches for files within each dataset.
"""

import argparse
import json
import subprocess
import os

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", help="EOS directory path", required=True)
parser.add_argument(
    "-o",
    "--output",
    help="Output JSON file name",
    default="dataset_files.json",
    required=False,
)

xrootd_redirector = "root://cmsxrootd.fnal.gov/"


def get_files_recursive(path):
    files = []
    result = subprocess.run(
        f"eos {xrootd_redirector} ls {args.dir}", stdout=subprocess.PIPE, shell=True
    )
    items = result.stdout.decode("utf-8").splitlines()

    for item in items:
        if not item:  # Skip empty lines
            continue
        full_path = os.path.join(path, item)
        # Check if item is a directory by trying to list its contents
        check_dir = subprocess.run(
            ["eosls", args.dir],
            stdout=subprocess.PIPE,
            shell=False,
            env=os.environ.copy(),
        )
        if check_dir.stdout:  # If we can list contents, it's a directory
            files.extend(get_files_recursive(full_path))
        else:
            files.append(os.path.join(xrootd_redirector + full_path))

    return files


if __name__ == "__main__":
    args = parser.parse_args()

    # Get all datasets in the top directory
    print(f"Listing datasets in {args.dir}")
    result = subprocess.run(
        f"eos {xrootd_redirector} ls {args.dir}",
        stdout=subprocess.PIPE,
        shell=True,
    )
    datasets = result.stdout.decode("utf-8").splitlines()

    print(f"Found {len(datasets)} datasets")

    file_dict = {}

    for dataset in tqdm(datasets):
        if dataset:  # ignore empty lines
            dataset_path = os.path.join(args.dir, dataset)
            file_dict[dataset] = get_files_recursive(dataset_path)

    # Write the dictionary to a JSON file
    with open(args.output, "w") as f:
        json.dump(file_dict, f, indent=4)
