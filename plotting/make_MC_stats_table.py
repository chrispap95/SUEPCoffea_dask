import argparse
import logging
import math

import numpy as np
import plot_utils
import scipy.stats as stats  # type: ignore[import]
from tabulate import tabulate  # type: ignore[import]

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Apr2025",
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
        "--latex",
        action="store_true",
        help="Print the table in LaTeX format",
    )
    return parser.parse_args()


def get_poisson_errors(N, alpha=0.6827, scale=1):
    # Return the Garwood confidence interval for a Poisson distribution
    upper = stats.gamma.ppf((1 + alpha) / 2, N + 1) - N
    lower = N - stats.gamma.ppf((1 - alpha) / 2, N)
    return np.nan_to_num(lower) * scale, np.nan_to_num(upper) * scale


def smart_round(x):
    if x == 0:
        return x
    if x > 10:
        return round(x)
    return round(x, -math.floor(np.log10(x)) + 1)


if "__main__" == __name__:
    args = parse_args()

    tablefmt = "simple"
    if args.latex:
        tablefmt = "latex_raw"

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots_CR = plot_utils.loader(
        tag=f"{args.tag}_CR", custom_lumi=args.lumi, load_data=False
    )
    plots_SR = plot_utils.loader(
        tag=f"{args.tag}_SRs", custom_lumi=args.lumi, load_data=False
    )
    plots = {}
    for dataset in plots_CR:
        # Note: need to fix this to be mergeable even when data for SR is missing! (blinded...)
        # This merges two dicts!
        plots[dataset] = plots_CR[dataset] | plots_SR[dataset]
    print("Done!", flush=True)

    mc_processes = [
        ("Higgs_2018", "Higgs"),
        ("TTV_2018", "TTV"),
        ("ST_NLO_2018", "ST"),
        ("WJets_2018", "WJets"),
        ("VV+VVV_2018", "VV+VVV"),
        ("TT_powheg_2018", "TT"),
        ("DY_2018", f"DY"),
        ("QCD_Pt_MuEnrichedPt5_2018", f"QCD"),
    ]
    regions = ["SR_high_temp_tight", "SR_low_temp_tight"]
    regions_latex = [r"\SRhigh discovery bin", r"\SRlow discovery bin"]

    for region in regions:
        header = [
            "process",
            "highest populated nMuon bin",
            "event yield",
            "eff. weight (sumw2/sumw)",
            "eff. MC events ((sumw)^2/sumw2)",
            "Poisson unc. band for y=0",
        ]
        table = []
        for proc, label in mc_processes:
            h = plots[proc][region]
            # Get scale of rightmost non-zero bin
            values = h.values()[h.values() > 0]
            variances = h.variances()[h.values() > 0]
            scale = variances[-1] / values[-1]
            mc_events = values[-1] ** 2 / variances[-1]
            unc_band = get_poisson_errors(0, scale=scale)
            row = [
                label,
                np.where(plots[proc][region].values() > 0)[0][-1] + 3,
                f"{smart_round(values[-1])} +/- {smart_round(np.sqrt(variances[-1]))}",
                smart_round(scale),
                round(mc_events, 1),
                (smart_round(unc_band[0]), smart_round(unc_band[1])),
            ]
            table.append(row)

        print(region)
        print()
        print(tabulate(table, headers=header, tablefmt=tablefmt))
        print()
