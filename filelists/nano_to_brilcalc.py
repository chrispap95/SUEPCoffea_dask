import json
import sys
from collections import defaultdict

import ROOT  # type: ignore[import]


def create_lumi_json(filename, output_json="lumi_ranges.json"):
    """
    Create a JSON file for brilcalc from a NanoAOD file
    The format should be:
    {
        "run_number": [[lumi_start, lumi_end], [lumi_start, lumi_end], ...],
        ...
    }
    """

    # Open the file and get the LuminosityBlocks tree
    f = ROOT.TFile.Open(filename)
    tree = f.Get("LuminosityBlocks")

    # Dictionary to store run:lumis mapping
    run_lumi_dict = defaultdict(list)

    # Loop over entries
    for entry in tree:
        run = str(entry.run)  # brilcalc expects string keys
        luminosityBlock = entry.luminosityBlock
        run_lumi_dict[run].append(luminosityBlock)

    # Process the lumis to create ranges
    output_dict = {}
    for run, lumis in run_lumi_dict.items():
        lumis.sort()  # Sort luminosity sections
        ranges = []
        range_start = lumis[0]
        prev_lumi = lumis[0]

        for lumi in lumis[1:]:
            if lumi > prev_lumi + 1:
                # Gap found, close the current range
                ranges.append([range_start, prev_lumi])
                range_start = lumi
            prev_lumi = lumi

        # Add the last range
        ranges.append([range_start, prev_lumi])

        # Add to output dictionary
        output_dict[run] = ranges

    # Write to JSON file
    with open(output_json, "w") as f:
        # Add minimal formatting for readability (one run per line)
        formatted_str = (
            "{\n"
            + ",\n".join(
                f'"{run}":{json.dumps(ranges, separators=(",", ":"))}'
                for run, ranges in output_dict.items()
            )
            + "\n}"
        )
        f.write(formatted_str)

    print(f"Created JSON file: {output_json}")

    # Print some statistics
    total_runs = len(output_dict)
    total_ls_ranges = sum(len(ranges) for ranges in output_dict.values())
    print(f"\nStatistics:")
    print(f"Total number of runs: {total_runs}")
    print(f"Total number of LS ranges: {total_ls_ranges}")
    print("\nExample brilcalc command:")
    print(
        f"brilcalc lumi --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -i {output_json}"
    )

    f.close()
    return output_dict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_lumi_json.py <nanoaod_file> [output_json]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else "lumi_ranges.json"

    run_lumi_dict = create_lumi_json(input_file, output_json)
