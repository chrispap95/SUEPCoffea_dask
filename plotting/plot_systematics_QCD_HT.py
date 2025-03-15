import argparse
import os
import warnings

import matplotlib as mpl  # type: ignore[import]
import matplotlib.gridspec as gridspec  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.ticker as ticker  # type: ignore[import]
import mplhep as hep
import numpy as np
import plot_utils

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="QCD_HT_Feb2025",
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
        "--dest",
        type=str,
        default="/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/muon_branches/SUEPCoffea_dask/plotting/systematics_plots",
        help="Destination directory to save the plots",
    )
    return parser.parse_args()


def plot_systematics(args, plots, sample, region):
    if region not in plots[sample]:
        return

    sample_tag = (
        sample.replace("_2018", "")
        .replace("_13TeV", "")
        .replace("mode", "")
        .replace(".000", "")
        .replace("0_", "_")
        + " - "
    )
    if len(sample_tag) > 10:
        sample_tag += "\n"

    systematics = set()
    for key in plots[sample]:
        if region in key:
            systematics.add(
                key.replace(region, "")
                .replace("extrapolation", "")
                .replace("_", "")
                .replace("Up", "")
                .replace("Down", "")
            )

    for syst in sorted(systematics):
        if not syst:
            continue

        fig = plt.figure(figsize=(13, 12))
        gs = gridspec.GridSpec(4, 1, left=0.08, right=0.92, bottom=0.15)

        ax1 = plt.subplot(gs[0:3, 0])  # Top 3 rows
        ax2 = plt.subplot(gs[3, 0], sharex=ax1)  # 4th row

        hep.histplot(
            plots[sample][region],
            yerr=np.sqrt(plots[sample][region].variances()),
            label="nominal",
            color="C0",
            ax=ax1,
        )
        hep.histplot(
            plots[sample][f"{region}_{syst}Up"],
            yerr=np.sqrt(plots[sample][f"{region}_{syst}Up"].variances()),
            label="up",
            color="C1",
            ax=ax1,
        )
        hep.histplot(
            plots[sample][f"{region}_{syst}Down"],
            yerr=np.sqrt(plots[sample][f"{region}_{syst}Down"].variances()),
            label="down",
            color="C2",
            ax=ax1,
        )

        ax2.axhline(1, ls="--", color="gray")
        hep.histplot(
            np.divide(
                plots[sample][f"{region}_{syst}Up"].values(),
                plots[sample][region].values(),
                out=np.ones_like(plots[sample][region].values()),
                where=(plots[sample][region].values() > 0),
            ),
            bins=plots[sample][region].axes[0].edges,
            label="up",
            color="C1",
            ax=ax2,
        )
        hep.histplot(
            np.divide(
                plots[sample][f"{region}_{syst}Down"].values(),
                plots[sample][region].values(),
                out=np.ones_like(plots[sample][region].values()),
                where=(plots[sample][region].values() > 0),
            ),
            bins=plots[sample][region].axes[0].edges,
            label="down",
            color="C2",
            ax=ax2,
        )

        ax1.set_yscale("log")
        ax1.set_ylabel("Events")
        ax1.set_title(f"{sample_tag}{region} - {syst}")
        ax1.legend()
        ax1.xaxis.set_minor_locator(ticker.NullLocator())
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.set_ylim(0.5, 1.5)
        ax2.set_ylabel("Ratio")
        ax2.set_xlabel("nMuon")
        ax1.set_xlabel("")
        for label in ax1.xaxis.get_ticklabels():
            label.set_visible(False)
        plt.savefig(f"{args.dest}/{region}_{sample}_{syst}.pdf")
        plt.close()
    return


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots_CR = plot_utils.loader(
        tag=f"{args.tag}_CR", custom_lumi=args.lumi, load_data=False
    )
    plots = {}
    for dataset in plots_CR:
        plots[dataset] = plots_CR[dataset]
    print("Done!", flush=True)

    # Plot systematics
    regions = [
        "CR_light",
        "CR_prompt",
        "CR_cb",
    ]
    for region in regions:
        plot_systematics(args, plots, "QCD_HT_2018", region)
