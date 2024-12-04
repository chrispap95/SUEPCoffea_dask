"""
This script reads a list of datasets from a text file and uses dasgoclient to get the locations
of the files for each dataset. It will write the locations of the files to a JSON file.
"""

import argparse
import json
import subprocess

from rich.progress import track  # type: ignore[import]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file with list of datasets")
    args = parser.parse_args()

    xrootd_redirector = "root://cmsxrootd.fnal.gov/"

    with open(f"{args.input}") as f:
        datasets = f.read().splitlines()

    file_dict = {}

    for dataset in track(datasets, description="Processing datasets..."):
        key = dataset.replace("/", "+")[1:]
        file_dict[key] = []
        command = f'dasgoclient -query="file dataset={dataset}"'
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        command_str = " ".join(command)
        files = result.stdout.splitlines()
        for _file in files:
            if _file:  # ignore empty lines
                file_dict[key].append(xrootd_redirector + _file)

    # Write the dictionary to a JSON file
    with open(args.input.replace(".txt", "") + ".json", "w") as f:
        json.dump(file_dict, f, indent=4)
