import argparse
import logging

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
        "--poisson",
        action="store_true",
        help="Use Poisson errors for the table",
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


if "__main__" == __name__:
    args = parse_args()

    tablefmt = "simple"
    prefix = suffix = ""
    sep = "±"
    if args.latex:
        tablefmt = "latex_raw"
        prefix = "$"
        suffix = "$"
        sep = r"\pm"

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

    print("Fit and extrapolation...", end=" ", flush=True)
    # QCD extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "SR_low_temp_loose": slice(4j, None),
        "SR_low_temp_tight": slice(3j, None),
        "SR_high_temp_loose": slice(4j, None),
        "SR_high_temp_tight": slice(3j, None),
    }

    qcd_extrapolation = plot_utils.Extrapolation(
        plots["QCD_Pt_MuEnrichedPt5_2018"], uncertainty_scheme="full"
    )
    qcd_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

    # DY extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "SR_low_temp_loose": slice(4j, None),
        "SR_low_temp_tight": slice(3j, None),
        "SR_high_temp_loose": slice(4j, None),
        "SR_high_temp_tight": slice(4j, None),
    }
    dy_extrapolation = plot_utils.Extrapolation(
        plots["DY_2018"], uncertainty_scheme="full"
    )
    dy_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)
    print("Done!", flush=True)

    mc_processes = [
        # ("Higgs_2018", "Higgs"),
        # ("TTV_2018", "TTV"),
        # ("ST_NLO_2018", "ST"),
        # ("WJets_2018", "WJets"),
        # ("VV+VVV_2018", "VV+VVV"),
        # ("TT_powheg_2018", "TT"),
        # ("DY_2018", f"DY (no extr.)"),
        ("DY_2018", f"DY + extr."),
        # ("QCD_Pt_MuEnrichedPt5_2018", f"QCD (no extr.)"),
        ("QCD_Pt_MuEnrichedPt5_2018", f"QCD + extr."),
    ]
    regions = ["SR_high_temp_tight", "SR_low_temp_tight"]
    regions_latex = [r"\SRhigh discovery bin", r"\SRlow discovery bin"]

    # Calculate total bkg
    plots["total_bkg_2018"] = {}
    for process, proc_label in mc_processes:
        for region in regions:
            proc_suffix = ""
            if "+ extr." in proc_label and "SR" in region:
                proc_suffix = "_extrapolation"
            if region not in plots["total_bkg_2018"]:
                plots["total_bkg_2018"][region] = plots[process][
                    region + proc_suffix
                ].copy()
            else:
                plots["total_bkg_2018"][region] += plots[process][region + proc_suffix]

    header = ["process"] + regions_latex if args.latex else ["process"] + regions
    table = []
    for process, proc_label in mc_processes:
        proc_suffix = ""
        if "+ extr." in proc_label:
            proc_suffix = "_extrapolation"
        row = [proc_label]
        SR_high_temp = plots[process][f"SR_high_temp_tight{proc_suffix}"]
        SR_low_temp = plots[process][f"SR_low_temp_tight{proc_suffix}"]
        if args.poisson and args.latex:
            values = SR_high_temp.values()[SR_high_temp.values() > 0]
            variances = SR_high_temp.variances()[SR_high_temp.values() > 0]
            scale = variances[-1] / values[-1]
            poisson_unc = get_poisson_errors(SR_high_temp[7j].value, scale=scale)
            row.append(
                f"${SR_high_temp[7j].value:g}"
                + r"^{+"
                + f"{poisson_unc[1]:g}"
                + "}_{-"
                + f"{poisson_unc[0]:g}"
                + r"}$"
            )
            values = SR_low_temp.values()[SR_low_temp.values() > 0]
            variances = SR_low_temp.variances()[SR_low_temp.values() > 0]
            scale = variances[-1] / values[-1]
            poisson_unc = get_poisson_errors(SR_low_temp[7j].value, scale=scale)
            row.append(
                f"${SR_low_temp[7j].value:g}"
                + r"^{+"
                + f"{poisson_unc[1]:g}"
                + "}_{-"
                + f"{poisson_unc[0]:g}"
                + r"}$"
            )
        elif args.poisson and not args.latex:
            values = SR_high_temp.values()[SR_high_temp.values() > 0]
            variances = SR_high_temp.variances()[SR_high_temp.values() > 0]
            scale = variances[-1] / values[-1]
            poisson_unc = get_poisson_errors(SR_high_temp[7j].value, scale=scale)
            row.append(
                f"{SR_high_temp[7j].value:g} +{poisson_unc[1]:g} -{poisson_unc[0]:g}"
            )
            values = SR_low_temp.values()[SR_low_temp.values() > 0]
            variances = SR_low_temp.variances()[SR_low_temp.values() > 0]
            scale = variances[-1] / values[-1]
            poisson_unc = get_poisson_errors(SR_low_temp[7j].value, scale=scale)
            row.append(
                f"{SR_low_temp[7j].value:g} +{poisson_unc[1]:g} -{poisson_unc[0]:g}"
            )
        else:
            row.append(
                f"{prefix}{SR_high_temp[7j].value:g} {sep} {np.sqrt(SR_high_temp[7j].variance):g}{suffix}"
            )
            row.append(
                f"{prefix}{SR_low_temp[7j].value:g} {sep} {np.sqrt(SR_low_temp[7j].variance):g}{suffix}"
            )
        table.append(row)

    row = ["Total bkg"]
    SR_high_temp = plots["total_bkg_2018"][f"SR_high_temp_tight"][7j]
    row.append(
        f"{prefix}{SR_high_temp.value:g} {sep} {np.sqrt(SR_high_temp.variance):g}{suffix}"
    )
    SR_low_temp = plots["total_bkg_2018"][f"SR_low_temp_tight"][7j]
    row.append(
        f"{prefix}{SR_low_temp.value:g} {sep} {np.sqrt(SR_low_temp.variance):g}{suffix}"
    )
    table.append(row)

    print()
    print(tabulate(table, headers=header, tablefmt=tablefmt))
    print()
