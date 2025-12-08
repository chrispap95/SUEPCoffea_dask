import argparse
import logging

import plot_utils
from colorama import Fore, Style  # type: ignore[import]
from rich.progress import track  # type: ignore[import]
from tabulate import tabulate  # type: ignore[import]

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)


def color_value(value):
    try:
        if float(value) > 0.15:
            return f"{Fore.RED}{value}{Style.RESET_ALL}"
        return value
    except (ValueError, TypeError):
        return value  # Return unchanged if not a number


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Apr2025",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="2018",
        help="Year of the data. Default is 2018. Pick only one year.",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        help="Custom integrated luminosity to be used (in pb^-1). For example, use 559.322 for "
        "the single data file in filelists/data/data_Run2018A_0p6fb_1file_unskimmed.json."
        "If not provided, the luminosity will be determined automatically for the year.",
    )
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots = plot_utils.loader(
        tag=f"{args.tag}_{args.year}_VR",
        era=args.year,
        custom_lumi=args.lumi,
        load_data=False,
    )
    print("Done!", flush=True)

    print("Fit and extrapolation...", end=" ", flush=True)
    # QCD extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "SR_low_temp_loose": slice(4j, 7j),
        "SR_low_temp_tight": slice(3j, 5j),
        "SR_high_temp_loose": slice(4j, 7j),
        "SR_high_temp_tight": slice(3j, 5j),
        "VR_loose": slice(3j, 6j),
        "VR_tight": slice(3j, 6j),
    }

    qcd_extrapolation = plot_utils.Extrapolation(
        plots[f"QCD_Pt_MuEnrichedPt5_{args.year}"], uncertainty_scheme="full"
    )
    qcd_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

    print("Done!", flush=True)

    QCD_VR_loose_vals = plots[f"QCD_Pt_MuEnrichedPt5_{args.year}"][
        "VR_loose_extrapolation"
    ].values()
    QCD_VR_tight_vals = plots[f"QCD_Pt_MuEnrichedPt5_{args.year}"][
        "VR_tight_extrapolation"
    ].values()

    # Add signal
    signal_models = [model for model in plots if "SUEP" in model]
    SoverB_loose = {}
    SoverB_tight = {}
    print(f"Processing {len(signal_models)} signal models.\n", flush=True)
    for model in sorted(signal_models):
        model_name = (
            model.replace("GluGluTo", "")
            .replace(".000", "")
            .replace(f"_13TeV_{args.year}", "")
            .replace("00_mode", "_mode")
            .replace("0_mode", "_mode")
            .replace("00_T", "_T")
        )
        SoverB_loose[model_name] = plots[model]["VR_loose"].values() / QCD_VR_loose_vals
        SoverB_tight[model_name] = plots[model]["VR_tight"].values() / QCD_VR_tight_vals

    header = (
        ["Model"]
        + [f"S/B loose bin {i}" for i in range(len(QCD_VR_loose_vals))]
        + [f"S/B tight bin {i}" for i in range(len(QCD_VR_tight_vals))]
    )
    SoverB_table = [
        [key]
        + [color_value(round(v, 2)) for v in value1]
        + [color_value(round(v, 2)) for v in value2]
        for (key, value1), value2 in zip(SoverB_loose.items(), SoverB_tight.values())
    ]

    print(tabulate(SoverB_table, headers=header))
