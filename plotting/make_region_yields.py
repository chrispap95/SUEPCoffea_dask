import argparse
import logging

import numpy as np
import plot_utils
import scipy.stats as stats  # type: ignore[import]
from rich.progress import track  # type: ignore[import]
from tabulate import tabulate  # type: ignore[import]

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Jun2025",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--year",
        type=str,
        nargs="*",
        default=["2018"],
        help="Year of the data. Default is 2018. Can be a single year or multiple years.",
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
        "--data",
        action="store_true",
        help="Load data.",
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


def merge_runs(plots, run, args):
    names = [
        "Higgs",
        "TTV",
        "ST_NLO",
        "WJets",
        "VV+VVV",
        "TT_powheg",
        "DY",
        "QCD_Pt_MuEnrichedPt5",
    ]
    # Remove 2016APV for now
    # years = ["2016APV", "2016", "2017", "2018"]
    years = ["2016", "2017", "2018"]
    if run == "Run3":
        names.remove("TTV")
        names.remove("ST_NLO")
        years = ["2022", "2022EE", "2023", "2023BPix"]
    if args.data:
        names.append("Data")
    run_plots = {}
    for name in names:
        run_plots[f"{name}_{run}"] = {}
        for year in years:
            if f"{name}_{year}" not in plots.keys():
                continue
            for plot in plots[f"{name}_{year}"]:
                if plot not in run_plots[f"{name}_{run}"].keys():
                    run_plots[f"{name}_{run}"][plot] = plots[f"{name}_{year}"][
                        plot
                    ].copy()
                else:
                    run_plots[f"{name}_{run}"][plot] += plots[f"{name}_{year}"][plot]
    return run_plots


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
    years_to_load = args.year
    if "Run2" in args.year:
        # Commenting out 2016APV for now
        # years_to_load = ["2016APV", "2016", "2017", "2018"]
        years_to_load = ["2016", "2017", "2018"]
    if "Run3" in args.year:
        years_to_load = ["2022", "2022EE", "2023", "2023BPix"]
    if "Run2" in args.year and "Run3" in args.year:
        years_to_load = [
            # Commenting out 2016APV for now
            # "2016APV",
            "2016",
            "2017",
            "2018",
            "2022",
            "2022EE",
            "2023",
            "2023BPix",
        ]
    plots = {}
    for year in track(years_to_load, description="Loading plots"):
        plots_CR = plot_utils.loader(
            tag=f"{args.tag}_{year}_CR",
            era=year,
            custom_lumi=args.lumi,
            load_data=args.data,
        )
        plots_SR = plot_utils.loader(
            tag=f"{args.tag}_{year}_SRs",
            era=year,
            custom_lumi=args.lumi,
            load_data=args.data,
        )
        all_datasets = set(plots_CR.keys()) | set(plots_SR.keys())
        for dataset in list(all_datasets):
            if dataset not in plots_CR:
                plots_CR[dataset] = {}
            if dataset not in plots_SR:
                plots_SR[dataset] = {}
            plots[dataset] = plots_CR[dataset] | plots_SR[dataset]

    for year in track(years_to_load, description="Fitting and extrapolating"):
        # QCD extrapolation
        # Slice the first bin out where needed for fit stability
        slice_hists = {
            "SR_low_temp_loose": slice(4j, None),
            "SR_low_temp_tight": slice(3j, None),
            "SR_high_temp_loose": slice(4j, None),
            "SR_high_temp_tight": slice(3j, None),
        }
        qcd_extrapolation = plot_utils.Extrapolation(
            plots["QCD_Pt_MuEnrichedPt5_" + year], uncertainty_scheme="full"
        )
        qcd_extrapolation.fit_syst_variations(slice_hists=slice_hists, verbose=False)
        qcd_extrapolation.create_syst_variation(sample="QCD")
        # DY extrapolation
        # Slice the first bin out where needed for fit stability
        slice_hists = {
            "SR_low_temp_loose": slice(4j, None),
            "SR_low_temp_tight": slice(3j, None),
            "SR_high_temp_loose": slice(4j, None),
            "SR_high_temp_tight": slice(4j, None),
        }
        dy_extrapolation = plot_utils.Extrapolation(
            plots["DY_" + year], uncertainty_scheme="full"
        )
        dy_extrapolation.fit_syst_variations(slice_hists=slice_hists, verbose=False)
        dy_extrapolation.create_syst_variation(sample="DY")

    if "Run2" in args.year:
        run2_plots = merge_runs(plots, "Run2", args)
        plots = plots | run2_plots
    if "Run3" in args.year:
        run3_plots = merge_runs(plots, "Run3", args)
        plots = plots | run3_plots

    mc_processes = [
        # ("Higgs", "Higgs"),
        # ("TTV", "TTV"),
        # ("ST_NLO", "ST"),
        # ("WJets", "WJets"),
        # ("VV+VVV", "VV+VVV"),
        # ("TT_powheg", "TT"),
        # ("DY", f"DY (no extr.)"),
        ("DY", f"DY + extr."),
        # ("QCD_Pt_MuEnrichedPt5", f"QCD (no extr.)"),
        ("QCD_Pt_MuEnrichedPt5", f"QCD + extr."),
    ]
    regions = ["SR_high_temp_tight", "SR_low_temp_tight"]
    regions_latex = [r"\SRhigh discovery bin", r"\SRlow discovery bin"]

    # Calculate total bkg and fill the table
    header = ["process"] + regions_latex if args.latex else ["process"] + regions
    table = []
    for year in args.year:
        plots[f"total_bkg_{year}"] = {}
        for process, proc_label in mc_processes:
            for region in regions:
                proc_suffix = ""
                if "+ extr." in proc_label and "SR" in region:
                    proc_suffix = "_extrapolation"
                if region not in plots[f"total_bkg_{year}"]:
                    plots[f"total_bkg_{year}"][region] = plots[f"{process}_{year}"][
                        region + proc_suffix
                    ].copy()
                else:
                    plots[f"total_bkg_{year}"][region] += plots[f"{process}_{year}"][
                        region + proc_suffix
                    ]

        for process, proc_label in mc_processes:
            proc_suffix = ""
            if "+ extr." in proc_label:
                proc_suffix = "_extrapolation"
            row = [f"{proc_label} {year}"]
            SR_high_temp = plots[f"{process}_{year}"][
                f"SR_high_temp_tight{proc_suffix}"
            ]
            SR_low_temp = plots[f"{process}_{year}"][f"SR_low_temp_tight{proc_suffix}"]
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

        row = [f"Total bkg {year}"]
        SR_high_temp = plots[f"total_bkg_{year}"][f"SR_high_temp_tight"][7j]
        row.append(
            f"{prefix}{SR_high_temp.value:g} {sep} {np.sqrt(SR_high_temp.variance):g}{suffix}"
        )
        SR_low_temp = plots[f"total_bkg_{year}"][f"SR_low_temp_tight"][7j]
        row.append(
            f"{prefix}{SR_low_temp.value:g} {sep} {np.sqrt(SR_low_temp.variance):g}{suffix}"
        )
        table.append(row)

    print()
    print(tabulate(table, headers=header, tablefmt=tablefmt))
    print()
