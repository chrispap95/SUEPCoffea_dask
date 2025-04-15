import argparse
import sys
import pathlib

import uproot
from rich.progress import track  # type: ignore[import]

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / "plotting"))
import plot_utils  # type: ignore[import]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="PU_dist_Apr2025",
        help="Tag to identify the analysis",
    )
    return parser.parse_args()


eras = ["UL16", "UL16APV", "UL17", "UL18", "2022", "2022EE", "2023", "2023BPix"]

if "__main__" == __name__:
    args = parse_args()

    # for era in track(eras):
    era = "UL18"
    plots = plot_utils.loader(tag=f"{args.tag}_{era}", era=era.replace("UL", "20"))
    sample = f"QCD_Pt_MuEnrichedPt5_{era.replace('UL', '20')}"
    mc_pileup_hist = plots[sample]["nTrueInt"]

    # Create output file
    with uproot.recreate(f"mc_pileup_{era}.root") as f:
        f["mc_pileup"] = mc_pileup_hist
