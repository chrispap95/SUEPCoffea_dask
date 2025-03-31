import argparse
import logging
import os
import pathlib

import matplotlib as mpl  # type: ignore[import]
import matplotlib.gridspec as gridspec  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.ticker as ticker  # type: ignore[import]
import mplhep as hep
import numpy as np
import plot_utils

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag1",
        type=str,
        default="QCD_HT_Feb2025",
        help="Tag to identify QCD_HT",
    )
    parser.add_argument(
        "--tag2",
        type=str,
        default="full_analysis_Feb2025",
        help="Tag to identify QCD_Pt_MuEnrichedPt5",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        help="Custom integrated luminosity to be used (in pb^-1). For example, use 559.322 for "
        "the single data file in filelists/data/data_Run2018A_0p6fb_1file_unskimmed.json."
        "If not provided, the luminosity will be determined automatically for the year.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize QCD_Pt_MuEnrichedPt5_2018 to QCD_HT_2018",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "qcd_comparison"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'qcd_comparison'}",
    )
    return parser.parse_args()


def plot_QCD(args, plots, sample, region):
    fig = plt.figure(figsize=(13, 12))
    gs = gridspec.GridSpec(5, 1, left=0.1, right=0.95, bottom=0.1, top=0.95)

    ax1 = plt.subplot(gs[0:3, 0])  # Top 3 rows
    ax2 = plt.subplot(gs[3, 0], sharex=ax1)  # 4th row
    ax3 = plt.subplot(gs[4, 0], sharex=ax1)  # 4th row

    hep.histplot(
        plots["QCD_Pt_MuEnrichedPt5_2018"][region],
        yerr=np.sqrt(plots[sample][region].variances()),
        label="QCD_Pt_MuEnrichedPt5",
        lw=3,
        color="C0",
        ax=ax1,
    )
    hep.histplot(
        plots["QCD_HT_2018"][region],
        yerr=np.sqrt(plots[sample][region].variances()),
        label="QCD_HT",
        lw=3,
        color="C1",
        ax=ax1,
    )

    ax2.axhline(1, ls="--", color="gray")
    ratio = np.divide(
        plots["QCD_HT_2018"][region].values(),
        plots["QCD_Pt_MuEnrichedPt5_2018"][region].values(),
        out=np.ones_like(plots["QCD_Pt_MuEnrichedPt5_2018"][region].values()),
        where=(plots["QCD_Pt_MuEnrichedPt5_2018"][region].values() > 0),
    )
    ratio_unc = ratio * np.sqrt(
        np.divide(
            plots["QCD_HT_2018"][region].variances(),
            plots["QCD_HT_2018"][region].values() ** 2,
            out=np.ones_like(plots["QCD_HT_2018"][region].values()),
            where=(plots["QCD_HT_2018"][region].values() > 0),
        )
        + np.divide(
            plots["QCD_Pt_MuEnrichedPt5_2018"][region].variances(),
            plots["QCD_Pt_MuEnrichedPt5_2018"][region].values() ** 2,
            out=np.ones_like(plots["QCD_Pt_MuEnrichedPt5_2018"][region].values()),
            where=(plots["QCD_Pt_MuEnrichedPt5_2018"][region].values() > 0),
        )
    )
    hep.histplot(
        ratio,
        yerr=ratio_unc,
        bins=plots[sample][region].axes[0].edges,
        label="QCD_HT/QCD_Pt_MuEnrichedPt5",
        lw=3,
        color="C2",
        ax=ax2,
    )

    # mc_rel_unc = np.divide(
    #     np.sqrt(plots["QCD_Pt_MuEnrichedPt5_2018"][region].variances()),
    #     plots["QCD_Pt_MuEnrichedPt5_2018"][region].values(),
    #     out=np.zeros_like(plots["QCD_Pt_MuEnrichedPt5_2018"][region].values()),
    #     where=plots["QCD_Pt_MuEnrichedPt5_2018"][region].values() != 0,
    # )
    # x_hatch = np.vstack(
    #     (
    #         plots["QCD_Pt_MuEnrichedPt5_2018"][region].axes[0].edges[:-1],
    #         plots["QCD_Pt_MuEnrichedPt5_2018"][region].axes[0].edges[1:],
    #     )
    # ).reshape((-1,), order="F")
    # y_hatch2 = np.vstack(
    #     (
    #         np.ones_like(plots["QCD_Pt_MuEnrichedPt5_2018"][region].values()),
    #         np.ones_like(plots["QCD_Pt_MuEnrichedPt5_2018"][region].values()),
    #     )
    # ).reshape((-1,), order="F")
    # y_hatch2_unc = np.vstack((mc_rel_unc, mc_rel_unc)).reshape((-1,), order="F")
    # ax2.fill_between(
    #     x=x_hatch,
    #     y1=y_hatch2 - y_hatch2_unc,  # type: ignore[assign]
    #     y2=y_hatch2 + y_hatch2_unc,  # type: ignore[assign]
    #     step="pre",
    #     facecolor="none",
    #     edgecolor=(0, 0, 0, 0.5),
    #     linewidth=0,
    #     hatch="///",
    #     label="stat. unc.",
    # )

    pulls = (
        plots["QCD_HT_2018"][region].values()
        - plots["QCD_Pt_MuEnrichedPt5_2018"][region].values()
    ) / np.sqrt(plots["QCD_HT_2018"][region].variances())
    pulls_up = np.where(pulls >= 0, pulls, 0)
    pulls_down = np.where(pulls < 0, pulls, 0)

    x_hatch = np.vstack(
        (
            plots["QCD_Pt_MuEnrichedPt5_2018"][region].axes[0].edges[:-1],
            plots["QCD_Pt_MuEnrichedPt5_2018"][region].axes[0].edges[1:],
        )
    ).reshape((-1,), order="F")
    y_hatch_up = np.vstack((pulls_up, pulls_up)).reshape((-1,), order="F")
    y_hatch_down = np.vstack((pulls_down, pulls_down)).reshape((-1,), order="F")

    ax3.fill_between(
        x=x_hatch,
        y1=0,
        y2=y_hatch_up,  # type: ignore[arg-type]
        step="pre",
        facecolor="None",
        edgecolor="red",
        alpha=1,
        linewidth=0,
        hatch="///",
    )
    ax3.fill_between(
        x=x_hatch,
        y1=0,
        y2=y_hatch_down,  # type: ignore[arg-type]
        step="pre",
        facecolor="None",
        edgecolor="blue",
        alpha=1,
        linewidth=0,
        hatch="///",
    )
    ax3.axhline(0, ls="--", color="gray")
    ax3.set_xlabel("nMuon")
    ax3.set_ylabel("pull")
    ax3.set_ylim(-2.5, 2.5)

    ax1.set_yscale("log")
    ax1.set_ylabel("Events")
    ax1.legend()
    ax1.xaxis.set_minor_locator(ticker.NullLocator())
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_ylim(0.5, 1.5)
    ax2.legend()
    ax2.set_ylabel("ratio")
    ax2.set_xlabel("")
    ax1.set_xlabel("")
    for label in ax1.xaxis.get_ticklabels():
        label.set_visible(False)
    for label in ax2.xaxis.get_ticklabels():
        label.set_visible(False)
    plt.savefig(os.path.join(args.dest, f"QCD_comparison_{region}.pdf"))
    plt.close()
    return


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots_CR_1 = plot_utils.loader(
        tag=f"{args.tag1}_CR", custom_lumi=args.lumi, load_data=False
    )
    plots_CR_2 = plot_utils.loader(
        tag=f"{args.tag2}_CR", custom_lumi=args.lumi, load_data=False
    )
    plots = {}
    plots["QCD_HT_2018"] = plots_CR_1["QCD_HT_2018"]
    plots["QCD_Pt_MuEnrichedPt5_2018"] = plots_CR_2["QCD_Pt_MuEnrichedPt5_2018"]
    print("Done!", flush=True)

    # Normalize QCD_Pt_MuEnrichedPt5_2018 to QCD_HT_2018
    if args.normalize:
        print("Calculate and apply normalization...", end=" ", flush=True)
        norm_factor = (plots["QCD_HT_2018"]["CR_cb"].sum().value) / plots[
            "QCD_Pt_MuEnrichedPt5_2018"
        ]["CR_cb"].sum().value
        for plot in plots["QCD_Pt_MuEnrichedPt5_2018"]:
            plots["QCD_Pt_MuEnrichedPt5_2018"][plot] = (
                norm_factor * plots["QCD_Pt_MuEnrichedPt5_2018"][plot]
            )
        print(f" norm_factor = {norm_factor:.2f}  Done!", flush=True)

    # Plot systematics
    regions = [
        "CR_light",
        "CR_prompt",
        "CR_cb",
    ]
    for region in regions:
        plot_QCD(args, plots, "QCD_HT_2018", region)
