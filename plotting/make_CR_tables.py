import argparse
import logging

import plot_utils
from colorama import Fore, Style  # type: ignore[import]
from tabulate import tabulate  # type: ignore[import]

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)


def color_value(value):
    try:
        if float(value) > 0.05:
            return f"{Fore.RED}{value}{Style.RESET_ALL}"
        return value
    except (ValueError, TypeError):
        return value  # Return unchanged if not a number


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Feb2026",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="2018",
        help="Year of the data to be used (e.g., 2016, 2017, 2018). Default is 2018.",
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
        tag=f"{args.tag}_{args.year}_CR",
        custom_lumi=args.lumi,
        load_data=False,
        era=args.year,
    )
    print("Done!", flush=True)

    mc_processes = [
        ("Higgs", "Higgs"),
        ("TTV", "TTV"),
        ("ST_NLO", "ST"),
        ("WJets", "WJets"),
        ("VV+VVV", "VV+VVV"),
        ("TT_powheg", "TT"),
        ("DY", f"DY+extr."),
        ("QCD_Pt_MuEnrichedPt5", f"QCD+extr."),
    ]
    regions = ["CR_cb", "CR_prompt"]

    # Calculate total bkg
    plots["total_bkg"] = {}
    for process, proc_label in mc_processes:
        for region in regions:
            if region not in plots["total_bkg"]:
                plots["total_bkg"][region] = plots[f"{process}_{args.year}"][
                    region
                ].copy()
            else:
                plots["total_bkg"][region] += plots[f"{process}_{args.year}"][region]

    bkg_CR_cb_vals = plots["total_bkg"]["CR_cb"].values()
    bkg_CR_prompt_vals = plots["total_bkg"]["CR_prompt"].values()

    # Add signal
    signal_models = [model for model in plots if "SUEP" in model]
    SoverB_CR_light = {}
    SoverB_CR_cb = {}
    SoverB_CR_prompt = {}
    for model in sorted(signal_models):
        model_name = (
            model.replace("GluGluTo", "")
            .replace(".000", "")
            .replace(f"_13TeV_{args.year}", "")
            .replace("00_mode", "_mode")
            .replace("0_mode", "_mode")
            .replace("00_T", "_T")
        )
        SoverB_CR_cb[model_name] = plots[model]["CR_cb"].values() / bkg_CR_cb_vals
        SoverB_CR_prompt[model_name] = (
            plots[model]["CR_prompt"].values() / bkg_CR_prompt_vals
        )

    header = (
        ["Model"]
        + [f"S/B CR_cb bin {i}" for i in range(len(bkg_CR_cb_vals))]
        + [f"S/B CR_prompt bin {i}" for i in range(len(bkg_CR_prompt_vals))]
    )
    SoverB_table = [
        [key]
        + [color_value(round(v, 2)) for v in value1]
        + [color_value(round(v, 2)) for v in value2]
        for (key, value1), value2 in zip(
            SoverB_CR_cb.items(), SoverB_CR_prompt.values()
        )
    ]

    print(tabulate(SoverB_table, headers=header))
