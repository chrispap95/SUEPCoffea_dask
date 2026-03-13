"""
Script to create a table of bkg (QCD and DY) yields in the SR discovery bins for all years after extrapolation.
Optionally, produces the latex table for the AN.
"""

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
        default="full_analysis_Feb2026",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--year",
        type=str,
        nargs="*",
        default=["2016", "2017", "2018", "2022", "2022EE", "2023", "2023BPix"],
        help="Year of the data. Default is all years. Can be a single year or multiple years.",
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


def smart_round(value, unc):
    """
    Round the value and uncertainty to a reasonable number of significant digits based on the size of the uncertainty.
    If the uncertainty is very small, use scientific notation.
    Return the rounded value and uncertainty as formatted strings.
    """
    if unc == 0 and value == 0:
        return "0", "0"
    significant_digits = 1 - int(np.floor(np.log10(unc)))
    if significant_digits < 0:
        significant_digits = 0
    rounded_value = f"{round(value, significant_digits):.{significant_digits}f}"
    rounded_unc = f"{round(unc, significant_digits):.{significant_digits}f}"
    if unc < 1e-4:
        rounded_value = f"{value:.1e}"
        rounded_unc = f"{unc:.1e}"
    return rounded_value, rounded_unc


def format_entry(value, unc, latex_mode=False):
    prefix = suffix = ""
    sep = "±"
    if latex_mode:
        prefix = "$"
        suffix = "$"
        sep = r"\pm"
    rounded_value, rounded_unc = smart_round(value, unc)
    if unc < 1e-4:
        if latex_mode:
            return (
                prefix
                + f"{rounded_value}".replace("e-0", "e-").replace("e-", r"\times 10^{-")
                + "}"
                + sep
                + f"{rounded_unc}".replace("e-0", "e-").replace("e-", r"\times 10^{-")
                + "}"
                + suffix
            )
        return f"{prefix}{rounded_value} {sep} {rounded_unc}{suffix}"
    return f"{prefix}{rounded_value} {sep} {rounded_unc}{suffix}"


def render_latex_table(table, header, args):
    col_widths = [25, 18, 22, 22]
    print(r"\begin{tabular}{clcc}")
    print(r"\hline")
    header.insert(0, "year")
    header = [f"\\textbf{{{col}}}" for col in header]
    header = [f"{col:<{col_widths[i]}}" for i, col in enumerate(header)]
    print(" & ".join(header) + r" \\")
    print(r"\hline")
    i = 0
    for year in args.year:
        for j, row in enumerate(table[i : i + 3]):
            if "Total" in row[0]:
                print(r"\cline{2-4}")
                row = [r"\textbf{" + rr + "}" for rr in row]
            row.insert(0, "")  # Add an empty cell for the multirow
            row[1] = row[1].replace(f" {year}", "")
            if j == 0:
                row[0] = r"\multirow{3}{*}{" + year + "}"
            row = [f"{col:<{col_widths[k]}}" for k, col in enumerate(row)]
            print(" & ".join(row) + r" \\")
        i += 3
        print(r"\hline")
    print(" & ".join(table[-3]) + r" \\")
    print(r"\hline")
    print(" & ".join(table[-2]) + r" \\")
    print(r"\hline")
    print(" & ".join(table[-1]) + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")


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
        plots = plots | plot_utils.loader(
            tag=f"{args.tag}_{year}_SRs",
            era=year,
            custom_lumi=args.lumi,
            load_data=False,
        )

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

    mc_processes = [
        ("DY", f"DY + extr."),
        ("QCD_Pt_MuEnrichedPt5", f"QCD + extr."),
    ]
    regions = ["SR_high_temp_tight", "SR_low_temp_tight"]
    regions_latex = [r"\SRhigh discovery bin", r"\SRlow discovery bin"]

    SR_low_temp_tot_run2 = (
        plots[f"{mc_processes[0][0]}_{args.year[0]}"][regions[0]].copy().reset()
    )
    SR_low_temp_tot_run3 = (
        plots[f"{mc_processes[0][0]}_{args.year[0]}"][regions[0]].copy().reset()
    )
    SR_low_temp_tot_all = (
        plots[f"{mc_processes[0][0]}_{args.year[0]}"][regions[0]].copy().reset()
    )
    SR_high_temp_tot_run2 = (
        plots[f"{mc_processes[0][0]}_{args.year[0]}"][regions[0]].copy().reset()
    )
    SR_high_temp_tot_run3 = (
        plots[f"{mc_processes[0][0]}_{args.year[0]}"][regions[0]].copy().reset()
    )
    SR_high_temp_tot_all = (
        plots[f"{mc_processes[0][0]}_{args.year[0]}"][regions[0]].copy().reset()
    )

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
            # if args.poisson and args.latex:
            #     values = SR_high_temp.values()[SR_high_temp.values() > 0]
            #     variances = SR_high_temp.variances()[SR_high_temp.values() > 0]
            #     scale = variances[-1] / values[-1]
            #     poisson_unc = get_poisson_errors(SR_high_temp[7j].value, scale=scale)
            #     row.append(
            #         f"${SR_high_temp[7j].value:g}"
            #         + r"^{+"
            #         + f"{poisson_unc[1]:g}"
            #         + "}_{-"
            #         + f"{poisson_unc[0]:g}"
            #         + r"}$"
            #     )
            #     values = SR_low_temp.values()[SR_low_temp.values() > 0]
            #     variances = SR_low_temp.variances()[SR_low_temp.values() > 0]
            #     scale = variances[-1] / values[-1]
            #     poisson_unc = get_poisson_errors(SR_low_temp[7j].value, scale=scale)
            #     row.append(
            #         f"${SR_low_temp[7j].value:g}"
            #         + r"^{+"
            #         + f"{poisson_unc[1]:g}"
            #         + "}_{-"
            #         + f"{poisson_unc[0]:g}"
            #         + r"}$"
            #     )
            # elif args.poisson and not args.latex:
            #     values = SR_high_temp.values()[SR_high_temp.values() > 0]
            #     variances = SR_high_temp.variances()[SR_high_temp.values() > 0]
            #     scale = variances[-1] / values[-1]
            #     poisson_unc = get_poisson_errors(SR_high_temp[7j].value, scale=scale)
            #     row.append(
            #         f"{SR_high_temp[7j].value:g} +{poisson_unc[1]:g} -{poisson_unc[0]:g}"
            #     )
            #     values = SR_low_temp.values()[SR_low_temp.values() > 0]
            #     variances = SR_low_temp.variances()[SR_low_temp.values() > 0]
            #     scale = variances[-1] / values[-1]
            #     poisson_unc = get_poisson_errors(SR_low_temp[7j].value, scale=scale)
            #     row.append(
            #         f"{SR_low_temp[7j].value:g} +{poisson_unc[1]:g} -{poisson_unc[0]:g}"
            #     )
            # else:
            row.append(
                format_entry(
                    SR_high_temp[7j].value,
                    np.sqrt(SR_high_temp[7j].variance),
                    latex_mode=args.latex,
                )
            )
            row.append(
                format_entry(
                    SR_low_temp[7j].value,
                    np.sqrt(SR_low_temp[7j].variance),
                    latex_mode=args.latex,
                )
            )
            table.append(row)

        row = [f"Total bkg {year}"]
        SR_high_temp = plots[f"total_bkg_{year}"][f"SR_high_temp_tight"][7j]
        row.append(
            format_entry(
                SR_high_temp.value,
                np.sqrt(SR_high_temp.variance),
                latex_mode=args.latex,
            )
        )
        SR_low_temp = plots[f"total_bkg_{year}"][f"SR_low_temp_tight"][7j]
        row.append(
            format_entry(
                SR_low_temp.value,
                np.sqrt(SR_low_temp.variance),
                latex_mode=args.latex,
            )
        )
        table.append(row)

        # append an empty row between years
        if not args.latex:
            table.append(["-----------------"] * len(header))

        if year.startswith("201"):
            if SR_low_temp_tot_run2 is None:
                SR_low_temp_tot_run2 = SR_low_temp.copy()
                SR_high_temp_tot_run2 = SR_high_temp.copy()
            else:
                SR_low_temp_tot_run2 += SR_low_temp
                SR_high_temp_tot_run2 += SR_high_temp
        if year.startswith("202"):
            if SR_low_temp_tot_run3 is None:
                SR_low_temp_tot_run3 = SR_low_temp.copy()
                SR_high_temp_tot_run3 = SR_high_temp.copy()
            else:
                SR_low_temp_tot_run3 += SR_low_temp
                SR_high_temp_tot_run3 += SR_high_temp
        if SR_low_temp_tot_all is None:
            SR_low_temp_tot_all = SR_low_temp.copy()
            SR_high_temp_tot_all = SR_high_temp.copy()
        else:
            SR_low_temp_tot_all += SR_low_temp
            SR_high_temp_tot_all += SR_high_temp

    row = [r"\multicolumn{2}{c}{\textbf{Total bkg Run2}}"]
    row.append(
        format_entry(
            SR_high_temp_tot_run2[7j].value,
            np.sqrt(SR_high_temp_tot_run2[7j].variance),
            latex_mode=args.latex,
        )
    )
    row.append(
        format_entry(
            SR_low_temp_tot_run2[7j].value,
            np.sqrt(SR_low_temp_tot_run2[7j].variance),
            latex_mode=args.latex,
        )
    )
    table.append(row)

    row = [r"\multicolumn{2}{c}{\textbf{Total bkg Run3}}"]
    row.append(
        format_entry(
            SR_high_temp_tot_run3[7j].value,
            np.sqrt(SR_high_temp_tot_run3[7j].variance),
            latex_mode=args.latex,
        )
    )
    row.append(
        format_entry(
            SR_low_temp_tot_run3[7j].value,
            np.sqrt(SR_low_temp_tot_run3[7j].variance),
            latex_mode=args.latex,
        )
    )
    table.append(row)

    row = [r"\multicolumn{2}{c}{\textbf{Total bkg Run2 + Run3}}"]
    row.append(
        format_entry(
            SR_high_temp_tot_all[7j].value,
            np.sqrt(SR_high_temp_tot_all[7j].variance),
            latex_mode=args.latex,
        )
    )
    row.append(
        format_entry(
            SR_low_temp_tot_all[7j].value,
            np.sqrt(SR_low_temp_tot_all[7j].variance),
            latex_mode=args.latex,
        )
    )
    table.append(row)

    if not args.latex:
        print()
        print(tabulate(table, headers=header, tablefmt=tablefmt))
        print()
    else:
        # Custom rendering for LaTeX
        render_latex_table(table, header, args)
