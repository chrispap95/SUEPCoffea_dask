"""
This script reads a list of datasets from a text file and then looks for the files in the specified EOS directory.
It then creates a dictionary with pairs of datasets and lists of files and writes it to a JSON file.
"""

import argparse
import json
import os
import subprocess

from rich.progress import track  # type: ignore[import]

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", help="Input file with dataset names", required=True
)
parser.add_argument("-d", "--dir", help="Directory path", required=True)


if __name__ == "__main__":
    args = parser.parse_args()

    xrootd_redirector = "root://cmsxrootd.fnal.gov/"

    with open(f"{args.input}") as f:
        datasets = f.read().splitlines()

    file_dict = {}

    for dataset in track(datasets, description="Processing datasets..."):
        file_dict[dataset] = []
        result = subprocess.run(
            ["eos", xrootd_redirector, "ls", args.dir + dataset], stdout=subprocess.PIPE
        )
        files = result.stdout.decode("utf-8").split("\n")
        for _file in files:
            if _file:  # ignore empty lines
                file_dict[dataset].append(
                    os.path.join(xrootd_redirector + args.dir + dataset, _file)
                )

    # Write the dictionary to a JSON file
    with open(args.input.replace(".txt", "") + ".json", "w") as f:
        json.dump(file_dict, f, indent=4)
