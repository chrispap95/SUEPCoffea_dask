import argparse
import json
import os
import subprocess

from tqdm import tqdm  # type: ignore[import]

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", help="EOS directory path", required=True)
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

    print(f"Found {len(datasets)} datasets")

    file_dict = {}
    for dataset in tqdm(datasets):
        dataset_path = os.path.join(args.dir, dataset)
        # Let's keep only the primary dataset name
        file_dict[dataset.split("+")[0]] = get_all_files_in_dir(dataset_path)

    with open(args.output, "w") as f:
        json.dump(file_dict, f, indent=4)
