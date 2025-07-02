import argparse
import logging
import os
import pathlib

import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.ticker as ticker  # type: ignore[import]
import mplhep as hep
import numpy as np
import plot_utils
from rich.progress import track  # type: ignore[import]

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)

pub_style = {
    "font.size": 26,
    "axes.labelsize": "large",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "legend.fontsize": "small",
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.5,
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "xtick.major.pad": 6,
    "ytick.major.size": 12,
    "ytick.minor.size": 6.0,
    "axes.linewidth": 2,
}
plt.style.use(pub_style)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_May2025",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "year_comparison_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'year_comparison_plots'}.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["data", "mc"],
        help="Mode of operation: 'data' for data plots, 'mc' for Monte Carlo plots.",
    )
    return parser.parse_args()


region_labels = {
    "CR_cb": r"$CR_{QCD}$",
    "CR_prompt": r"$CR_{DY}$",
}

y_ranges = {
    "CR_cb": (1e-3, 1),
    "CR_prompt": (1e-6, 2),
}


def plot_ratio(results, year, ax2):
    this_year = results[year]
    year_2018 = results["2018"]
    ratio = np.divide(
        this_year.values(),
        year_2018.values(),
        out=np.ones_like(this_year.values()),
        where=year_2018.values() != 0,
    )
    ratio_err = np.where(
        year_2018.values() > 0,
        np.sqrt(
            (year_2018.values() ** -2) * (this_year.variances())
            + (this_year.values() ** 2 * year_2018.values() ** -4)
            * (year_2018.variances())
        ),
        0,
    )
    hep.histplot(
        ratio,
        bins=this_year.axes.edges[0],
        yerr=ratio_err,
        lw=2,
        ax=ax2,
    )

    ax2.axhline(1, ls="--", color="gray")


def plot_region_data(args, region, year, ax1):
    plots = plot_utils.loader(tag=f"{args.tag}_{year}_CR", era=year, load_data=True)

    hist_data = plots[f"Data_{year}"][region].copy()

    values = hist_data.values()
    variances = hist_data.variances()
    values_norm = values / sum(values)
    variances_norm = (
        values_norm**2
        / sum(values) ** 2
        * (
            (sum(values) - values) ** 2 / values**2 * variances
            + (sum(variances) - variances)
        )
    )

    for i in range(len(values)):
        hist_data[i] = (values_norm[i], variances_norm[i])

    hep.histplot(
        hist_data,
        yerr=np.sqrt(hist_data.variances()),
        label=year,
        lw=3,
        ax=ax1,
    )

    return hist_data


def plot_region_mc(args, region, year, ax1):
    plots = plot_utils.loader(tag=f"{args.tag}_{year}_CR", era=year, load_data=False)

    mc_processes = [
        "Higgs",
        "TTV",
        "ST_NLO",
        "WJets",
        "VV+VVV",
        "TT_powheg",
        "DY",
        "QCD_Pt_MuEnrichedPt5",
    ]

    hists_mc = []
    hist_bkg_total = plots[f"QCD_Pt_MuEnrichedPt5_{year}"][region].copy().reset()

    for process in mc_processes:
        h_mc = plots[f"{process}_{year}"][region]
        hists_mc.append(h_mc)
        hist_bkg_total += h_mc.copy()

    values = hist_bkg_total.values()
    variances = hist_bkg_total.variances()
    values_norm = values / sum(values)
    variances_norm = (
        values_norm**2
        / sum(values) ** 2
        * (
            (sum(values) - values) ** 2 / values**2 * variances
            + (sum(variances) - variances)
        )
    )

    for i in range(len(values)):
        hist_bkg_total[i] = (values_norm[i], variances_norm[i])

    hep.histplot(
        hist_bkg_total,
        yerr=np.sqrt(hist_bkg_total.variances()),
        label=year,
        lw=3,
        ax=ax1,
    )

    return hist_bkg_total


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)

    # Plot regions
    regions = [
        "CR_prompt",
        "CR_cb",
    ]
    for region in track(regions):
        fig = plt.figure(figsize=(12, 13))
        plt.subplots_adjust(bottom=0.08, top=0.92, left=0.1, right=0.95)
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

        results = {}
        for year in ["2016APV", "2016", "2017", "2018"]:
            if args.mode == "mc":
                results[year] = plot_region_mc(args, region, year, ax1)
            elif args.mode == "data":
                results[year] = plot_region_data(args, region, year, ax1)

        for year in ["2016APV", "2016", "2017"]:
            plot_ratio(results, year, ax2)

        hep.cms.label(llabel="Preliminary", data=False, ax=ax1)

        region_label_coords = (0.4, 0.2)
        plt.text(
            *region_label_coords,
            f"{region_labels[region]} - {args.mode}",
            ha="center",
            weight="bold",
            transform=ax1.transAxes,
        )

        # modify last x tick label
        if "CR_cb" in region:
            ax1.set_xticks([0.75, 1, 2, 3, 4, 5.25])
            ax1.set_xticklabels(["", "1", "2", "3", "4+", ""])
        elif "CR_prompt" in region:
            ax1.set_xticks([1.75, 2, 3, 4, 5, 6.25])
            ax1.set_xticklabels(["", "2", "3", "4", "5+", ""])

        plt.sca(ax2)
        plt.ylim(0.0, 2.0)
        plt.ylabel("Year/2018")
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_xlabel("", visible=False)
        plt.xlabel(r"$n_{muon}$")
        plt.sca(ax1)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
        plt.ylim(y_ranges[region])
        plt.yscale("log")
        plt.legend()
        plt.ylabel("density")
        plt.savefig(
            f"{args.dest}/{region}_{args.mode}_{args.tag}.pdf", bbox_inches="tight"
        )
        plt.close()
