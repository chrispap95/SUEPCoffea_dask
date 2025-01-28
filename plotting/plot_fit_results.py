import argparse
import os
import warnings

import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import mplhep as hep
import plot_utils
from cycler import cycler  # type: ignore[import]
from rich.progress import track  # type: ignore[import]

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

warnings.filterwarnings("ignore")

# Color palette for 10 colors
cmap_petroff_6 = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
cmap_petroff_10 = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]
CMS = {"axes.prop_cycle": cycler("color", cmap_petroff_10)}
plt.style.use(CMS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Dec2024",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        help="Custom integrated luminosity to be used (in pb^-1). For example, use 559.322 for "
        "the single data file in filelists/data/data_Run2018A_0p6fb_1file_unskimmed.json."
        "If not provided, the luminosity will be determined automatically for the year.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/muon_branches/SUEPCoffea_dask/plotting/fit_results_plots",
        help="Destination directory to save the plots",
    )
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots_CR = plot_utils.loader(
        tag=f"{args.tag}_CR", custom_lumi=args.lumi, load_data=False
    )
    plots_VR = plot_utils.loader(
        tag=f"{args.tag}_VR", custom_lumi=args.lumi, load_data=False
    )
    plots_SR = plot_utils.loader(
        tag=f"{args.tag}_SRs", custom_lumi=args.lumi, load_data=False
    )
    plots = {}
    for dataset in plots_CR:
        # Note: need to fix this to be mergeable even when data for SR is missing! (blinded...)
        # This merges two dicts!
        plots[dataset] = plots_CR[dataset] | plots_VR[dataset] | plots_SR[dataset]
    print("Done!", flush=True)

    print("Fit and extrapolation...", end=" ", flush=True)
    # QCD extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "VR_loose": slice(4j, None),
        "VR_tight": slice(3j, None),
        "SR_low_temp_loose": slice(4j, None),
        "SR_low_temp_tight": slice(3j, None),
        "SR_high_temp_loose": slice(4j, None),
        "SR_high_temp_tight": slice(3j, None),
    }

    qcd_extrapolation = plot_utils.Extrapolation(plots["QCD_Pt_MuEnrichedPt5_2018"])
    qcd_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

    print("Done!", flush=True)

    # Plot regions
    regions = [
        "SR_low_temp",
        "SR_high_temp",
        "VR",
    ]
    for region in track(regions):
        qcd_extrapolation.plot_fit(region)
        plt.tight_layout()
        plt.savefig(f"{args.dest}/plot_fit_{region}.pdf")
        plt.close()

    qcd_extrapolation.plot_overlay(regions=regions)
    plt.tight_layout()
    plt.savefig(f"{args.dest}/fit_overlay.pdf")
    plt.close()
