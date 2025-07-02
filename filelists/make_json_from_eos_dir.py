import argparse
import json
import logging
import os
import re
import subprocess

from tqdm import tqdm  # type: ignore[import]

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", help="EOS directory path", required=True)
parser.add_argument("--name", help="some egrep pattern for the dataset names")
parser.add_argument(
    "--depth",
    type=int,
    default=0,
    help="Depth of the directory tree. Default is 0 (list given directory).",
)
parser.add_argument(
    "-o", "--output", help="Output JSON file name", default="dataset_files.json"
)

xrootd_redirector = "root://cmseos.fnal.gov/"


def get_all_files_in_dir(eos_path):
    """Use eos find to get all files under the directory."""
    result = subprocess.run(
        f"eos {xrootd_redirector} find -f {xrootd_redirector}{eos_path}",
        capture_output=True,
        shell=True,
    )
    lines = result.stdout.decode("utf-8").splitlines()
    # Only return files (remove empty and weird entries)
    lines = [line for line in lines if line.strip() and line.endswith(".root")]
    return lines


if __name__ == "__main__":
    args = parser.parse_args()

    # Get top-level directories (datasets)
    print(f"Listing datasets in {args.dir}")
    result = subprocess.run(
        f"eos {xrootd_redirector} ls {args.dir}",
        capture_output=True,
        shell=True,
    )
    datasets = [
        line.strip()
        for line in result.stdout.decode("utf-8").splitlines()
        if line.strip()
    ]
    # Filter datasets if a name pattern is provided
    if args.name is not None:
        print(f"Filtering datasets with pattern: {args.name}")
        datasets = [dataset for dataset in datasets if re.search(args.name, dataset)]

    print(f"Found {len(datasets)} primary datasets.")

    file_dict = {}
    if args.depth == 0:
        for dataset in tqdm(datasets):
            dataset_path = os.path.join(args.dir, dataset)
            # Let's keep only the primary dataset name
            file_dict[dataset.split("+")[0]] = get_all_files_in_dir(dataset_path)

    # Implement only up to depth 1 - Need recursion for deeper levels but not needed for now
    if args.depth > 1:
        logging.warning(
            "Depth greater than 1 is not implemented. Going up to depth 1..."
        )
    if args.depth > 0:
        for path in tqdm(datasets):
            result = subprocess.run(
                f"eos {xrootd_redirector} ls {os.path.join(args.dir, path)}",
                capture_output=True,
                shell=True,
            )
            datasets_inside = [
                line.strip()
                for line in result.stdout.decode("utf-8").splitlines()
                if line.strip()
            ]
            for dataset in datasets_inside:
                dataset_path = os.path.join(args.dir, path, dataset)
                # Let's keep only the primary dataset name
                file_dict[dataset.split("/")[-1]] = get_all_files_in_dir(dataset_path)

    with open(args.output, "w") as f:
        json.dump(file_dict, f, indent=4)
