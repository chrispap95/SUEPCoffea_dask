import argparse
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

cmap = plt.get_cmap("viridis", 10)  # type: ignore[attr-defined]


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
        default=[
            "2016",
            "2017",
            "2018",
            "Run2",
            "2022",
            "2022EE",
            "2023",
            "2023BPix",
            "Run3",
        ],
        help="Year of the data. Default is all years. Can be a single year or multiple years.",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Plot data points in the regions. Default is False.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "tight_vs_loose_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'tight_vs_loose_plots'}",
    )
    return parser.parse_args()


# Latex labels for the regions
region_labels = {
    "SR_high_temp_loose": r"$SR^{loose}_{high~T}$",
    "SR_high_temp_tight": r"$SR^{tight}_{high~T}$",
    "SR_low_temp_loose": r"$SR^{loose}_{low~T}$",
    "SR_low_temp_tight": r"$SR^{tight}_{low~T}$",
}

# Abbreviations for the sample names
sample_names = {
    "QCD_Pt_MuEnrichedPt5": "QCD",
    "DY": "DY",
}


def make_plot(plots, sample, region):
    h_loose = (
        plots[sample][region + "_loose"] / plots[sample][region + "_loose"].sum().value
    )
    h_tight = (
        plots[sample][region + "_tight"] / plots[sample][region + "_tight"].sum().value
    )

    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.08, top=0.92, left=0.1, right=0.95, hspace=0.1)
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

    hep.histplot(
        h_loose,
        yerr=np.sqrt(h_loose.variances()),
        lw=2,
        label=region_labels[region + "_loose"],
        ax=ax1,
    )
    hep.histplot(
        h_tight,
        yerr=np.sqrt(h_tight.variances()),
        label=region_labels[region + "_tight"],
        lw=2,
        ax=ax1,
    )

    ratio = np.divide(
        h_tight.values(),
        h_loose.values(),
        out=np.zeros_like(h_tight.values()),
        where=h_loose.values() != 0,
    )
    ratio_err = np.sqrt(
        np.divide(
            h_tight.variances(),
            h_loose.values() ** 2,
            out=np.zeros_like(h_loose.values()),
            where=h_loose.values() != 0,
        )
        + np.divide(
            h_tight.values() ** 2 * h_loose.variances(),
            h_loose.values() ** 4,
            out=np.zeros_like(h_loose.values()),
            where=h_loose.values() != 0,
        )
    )
    hep.histplot(
        ratio,
        bins=h_loose.axes[0].edges,
        yerr=ratio_err,
        label="tight/loose",
        lw=2,
        ax=ax2,
        color="black",
        histtype="step",
    )
    ax2.axhline(1, ls="--", color="gray")

    ax1.text(
        0.5,
        0.95,
        sample_names["_".join(sample.split("_")[:-1])],
        ha="center",
        va="top",
        transform=ax1.transAxes,
    )
    hep.cms.label(llabel="Simulation", data=False, ax=ax1)

    plt.sca(ax2)
    plt.ylim(0, 2)
    plt.ylabel("tight/loose")
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlabel("", visible=False)
    plt.xlabel(r"$n_{muon}$")
    plt.sca(ax1)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
    plt.yscale("log")
    plt.legend()
    plt.ylabel("density")
    plt.savefig(
        os.path.join(
            args.dest,
            args.tag,
            f"tight_vs_loose_{region}_{year}_{sample_names['_'.join(sample.split('_')[:-1])]}.pdf",
        ),
        bbox_inches="tight",
    )
    plt.close()


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(os.path.join(args.dest, args.tag), exist_ok=True)

    # Load plots
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
            load_data=False,
        )

    for year in track(years_to_load, description="Fitting and extrapolations"):
        # QCD extrapolation
        # Slice the first bin out where needed for fit stability
        slice_hists = {
            "SR_low_temp_loose": slice(4j, None),
            "SR_low_temp_tight": slice(3j, None),
            "SR_high_temp_loose": slice(4j, None),
            "SR_high_temp_tight": slice(3j, None),
        }
        qcd_extrapolation = plot_utils.Extrapolation(
            plots[f"QCD_Pt_MuEnrichedPt5_{year}"], uncertainty_scheme="full"
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
            plots[f"DY_{year}"], uncertainty_scheme="full"
        )
        dy_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

    if "Run2" in args.year:
        run2_plots = plot_utils.merge_runs(plots, "Run2", data=args.data)
        plots = plots | run2_plots
    if "Run3" in args.year:
        run3_plots = plot_utils.merge_runs(plots, "Run3", data=args.data)
        plots = plots | run3_plots

    for year in track(args.year, description="Plotting regions"):
        for sample in [f"QCD_Pt_MuEnrichedPt5_{year}", f"DY_{year}"]:
            for region in ["SR_low_temp", "SR_high_temp"]:
                make_plot(plots, sample, region)
