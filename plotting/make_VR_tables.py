import argparse
import warnings

import plot_utils
from colorama import Fore, Style  # type: ignore[import]
from tabulate import tabulate  # type: ignore[import]

warnings.filterwarnings("ignore")


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
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()

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

    QCD_VR_loose_vals = plots["QCD_Pt_MuEnrichedPt5_2018"]["VR_loose"].values()
    QCD_VR_tight_vals = plots["QCD_Pt_MuEnrichedPt5_2018"]["VR_tight"].values()

    # Add signal
    signal_models = [model for model in plots if "SUEP" in model]
    SoverB_loose = {}
    SoverB_tight = {}
    for model in sorted(signal_models):
        model_name = (
            model.replace("GluGluTo", "")
            .replace(".000", "")
            .replace("_13TeV_2018", "")
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
