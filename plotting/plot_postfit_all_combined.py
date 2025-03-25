import argparse
import logging
import os
import pathlib

import cms_styles
import hist
import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.ticker as ticker  # type: ignore[import]
import matplotlib.transforms as transforms  # type: ignore[import]
import mplhep as hep
import numpy as np
import uproot
from matplotlib.lines import Line2D  # type: ignore[import]

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)

# Set to 10 color cycler
plt.style.use(cms_styles.CMS_petroff_10)

pub_style = {
    "font.size": 26,
    "axes.labelsize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
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
        "--input",
        type=str,
        default="/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/combine_stuff/Jan2025/CMSSW_11_3_4/src/postfit_plots.root",
        help="Path to the postfit plots root file. Default is "
        "/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/combine_stuff/Jan2025/CMSSW_11_3_4/src/postfit_plots.root",
    )
    parser.add_argument(
        "--signal-region",
        type=str,
        default="SR_high_temp",
        help="Signal region to plot. Default is SR_high_temp.",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Plot data points in the regions. Default is False.",
    )
    parser.add_argument(
        "--ratio",
        action="store_true",
        help="Plot the ratio of the data to the total background. "
        "Has an effect only when --data is passed as well. Default is False.",
    )
    parser.add_argument(
        "--unblind",
        action="store_true",
        help="Unblind the SRs. Default is False.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "postfit_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'postfit_plots'}",
    )
    return parser.parse_args()


regions = ["CR_QCD", "CR_DY", "SR_low_temp", "SR_high_temp"]
processes = [
    "DY",
    "GluGluToSUEP_mS125.000_mPhi8.000_T32.000_modeleptonic",
    "VV+VVV",
    "QCD",
    "ST",
    "TT",
    "TTV",
    "WJets",
    "Higgs",
    "TotalBkg",
    "data_obs",
]


def load_postfits(input_file, regions, processes):
    plots = {}
    with uproot.open(input_file) as f:
        for process in processes:
            plots[process] = {}
            for region in regions:
                if f"{region}_postfit/{process}" not in f:
                    continue
                plots[process][region] = f[f"{region}_postfit/{process}"].to_hist()  # type: ignore[attr-defined]
    return plots


def plot_ratio(hist_data, hist_bkg_total, ax, x_hatch, args):
    slc = slice(None)
    if not args.unblind:
        slc = slice(0, -1)
    ratio = np.divide(
        hist_data.values(),
        hist_bkg_total.values()[slc],
        out=np.ones_like(hist_data.values()),
        where=hist_bkg_total.values()[slc] != 0,
    )
    ratio_err = np.where(
        hist_bkg_total.values()[slc] > 0,
        np.sqrt(
            (hist_bkg_total.values()[slc] ** -2) * (hist_data.variances())
            + (hist_data.values() ** 2 * hist_bkg_total.values()[slc] ** -4)
            * (hist_bkg_total.variances()[slc])
        ),
        0,
    )
    ax.errorbar(
        hist_data.axes.centers[0],
        ratio,
        yerr=ratio_err,
        color="black",
        fmt="o",
        linestyle="none",
        markersize=8,
        lw=2,
    )

    # Draw a filled hatch area with the relative uncertainty of the MC in the ratio plot.
    mc_rel_unc = np.divide(
        np.sqrt(hist_bkg_total.variances()),
        hist_bkg_total.values(),
        out=np.zeros_like(hist_bkg_total.values()),
        where=hist_bkg_total.values() != 0,
    )
    y_hatch2 = np.vstack(
        (np.ones_like(hist_bkg_total.values()), np.ones_like(hist_bkg_total.values()))
    ).reshape((-1,), order="F")
    y_hatch2_unc = np.vstack((mc_rel_unc, mc_rel_unc)).reshape((-1,), order="F")
    ax.fill_between(
        x=x_hatch,
        y1=y_hatch2 - y_hatch2_unc,
        y2=y_hatch2 + y_hatch2_unc,
        step="pre",
        facecolor="none",
        edgecolor=(0, 0, 0, 0.5),
        linewidth=0,
        hatch="///",
    )


def plot_SUEP_combined(args, plots):
    mc_processes = [
        "Higgs",
        "TTV",
        "ST",
        "WJets",
        "VV+VVV",
        "TT",
        "DY",
        "QCD",
    ]
    data_name = ("data_obs", "Data")

    signal_processes = ["GluGluToSUEP_mS125.000_mPhi8.000_T32.000_modeleptonic"]
    signal_labels = [r"$m_\phi=8\,$GeV, $T=32\,$GeV"]

    hists_mc = []
    hist_bkg_total = plots["TotalBkg"]["SUEP"]

    for process in mc_processes:
        h_mc = plots[process]["SUEP"]
        hists_mc.append(h_mc)

    hists_signal = []
    for process in signal_processes:
        h_signal = plots[process]["SUEP"]
        hists_signal.append(h_signal)

    fig, ax1 = plt.subplots(figsize=(13.5, 11))

    if args.ratio:
        fig = plt.figure(figsize=(12, 11))
        plt.subplots_adjust(bottom=0.14, top=0.95, left=0.12, right=0.97)
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

    hep.histplot(
        hists_mc,
        yerr=[np.sqrt(h.variances()) for h in hists_mc],
        stack=True,
        label=mc_processes,
        histtype="fill",
        ec="black",
        lw=2,
        ax=ax1,
    )

    x_hatch = np.vstack(
        (hist_bkg_total.axes[0].edges[:-1], hist_bkg_total.axes[0].edges[1:])
    ).reshape((-1,), order="F")
    y_hatch1 = np.vstack((hist_bkg_total.values(), hist_bkg_total.values())).reshape(
        (-1,), order="F"
    )
    y_hatch1_unc = np.vstack(
        (np.sqrt(hist_bkg_total.variances()), np.sqrt(hist_bkg_total.variances()))
    ).reshape((-1,), order="F")
    ax1.fill_between(
        x=x_hatch,
        y1=y_hatch1 - y_hatch1_unc,  # type: ignore[assign]
        y2=y_hatch1 + y_hatch1_unc,  # type: ignore[assign]
        label="Stat. Unc.",
        step="pre",
        facecolor="none",
        edgecolor=(0, 0, 0, 0.5),
        linewidth=0,
        hatch="///",
        zorder=2,
    )

    if args.data and "SUEP" in plots[data_name[0]]:
        hep.histplot(
            plots[data_name[0]]["SUEP"],
            label=[data_name[1]],
            histtype="errorbar",
            mec="black",
            mfc="black",
            ecolor="black",
            markersize=15,
            lw=3,
            ax=ax1,
        )

    hep.histplot(
        hists_signal,
        yerr=[np.sqrt(h.variances()) for h in hists_signal],
        label=[s for s in signal_labels],
        color=["C8"],  # "C9"] + ["C" + str(i) for i in range(len(signal_labels) - 2)],
        lw=3,
        ls="--",
        ax=ax1,
    )

    if args.ratio and args.data:
        plot_ratio(plots[data_name[0]]["SUEP"], hist_bkg_total, ax2, x_hatch, args)

    hep.cms.label(llabel="Preliminary", data=True, lumi=59.8, ax=ax1)

    ln_x_positions = [0, 4, 8, 10]
    ln_y_upper = [2.2, 2.2, 2.1, 2.2]
    lines = []
    for ln_x_pos, ln_y_pos in zip(ln_x_positions, ln_y_upper):
        lines.append(
            Line2D(
                [ln_x_pos, ln_x_pos],
                [0.6, ln_y_pos],
                figure=fig,
                transform=ax2.transData,
                color="black",
                linestyle="-",
                linewidth=4,
            )
        )
        fig.add_artist(lines[-1])

    region_y = 0.63
    plt.text(
        2,
        region_y,
        r"$CR_{QCD}$",
        ha="center",
        weight="bold",
        fontsize=30,
        transform=ax2.transData,
    )
    plt.text(
        6,
        region_y,
        r"$CR_{DY}$",
        ha="center",
        weight="bold",
        fontsize=30,
        transform=ax2.transData,
    )
    plt.text(
        9,
        region_y,
        r"$SR_\text{high T}$",
        ha="center",
        weight="bold",
        fontsize=30,
        transform=ax2.transData,
    )

    plt.text(
        0.67,
        0.63,
        r"$m_S=125\,$GeV, $m_{A'}=0.5\,$GeV",
        ha="center",
        fontsize=24,
        transform=ax1.transAxes,
    )

    labels = ["1", "2", "3", "4+", "2", "3", "4", "5+", "7+", ""]
    major_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    ax1.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    ax1.set_xticklabels(labels)
    ax1.xaxis.set_minor_locator(ticker.NullLocator())
    # Create offset transform by 5 points in x direction
    dxs = np.array([25, 25, 25, 20, 25, 25, 25, 20, 50, 0]) / 72.0
    dy = 0 / 72.0
    for label, dx in zip(ax2.get_xticklabels(), dxs):
        label.set_horizontalalignment("left")
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)  # type: ignore[attr-defined]
        label.set_transform(label.get_transform() + offset)

    if args.ratio and args.data:
        plt.sca(ax2)
        plt.ylim(0.8, 1.2)
        plt.ylabel("Data/MC")
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_xlabel("", visible=False)
    plt.xlabel(r"$n_{muon}$", fontsize=36, labelpad=35)
    plt.sca(ax1)
    plt.ylim(1e-2, 1e12)
    plt.yscale("log")
    plt.legend(ncol=3, loc="upper center")
    plt.ylabel("events")
    plt.tight_layout()
    plt.savefig(f"{args.dest}/postfit_all_regions_combined.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    args = parse_args()

    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)

    print("Loading postfit plots...", end=" ", flush=True)
    plots = load_postfits(args.input, regions, processes)
    print("done!", flush=True)

    for sample in plots:
        h_comb = hist.Hist.new.Variable(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
            name="SUEP",
        ).Weight()
        if "data_obs" in sample and not args.unblind:
            h_comb = hist.Hist.new.Variable(
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                name="SUEP",
            ).Weight()

        h_comb[0] = plots[sample]["CR_QCD"][1j]
        h_comb[1] = plots[sample]["CR_QCD"][2j]
        h_comb[2] = plots[sample]["CR_QCD"][3j]
        h_comb[3] = plots[sample]["CR_QCD"][4j]
        h_comb[4] = plots[sample]["CR_DY"][2j]
        h_comb[5] = plots[sample]["CR_DY"][3j]
        h_comb[6] = plots[sample]["CR_DY"][4j]
        h_comb[7] = plots[sample]["CR_DY"][5j]
        if args.unblind or "data_obs" not in sample:
            if args.signal_region in plots[sample]:
                h_comb[8] = plots[sample][args.signal_region][7j]
        plots[sample]["SUEP"] = h_comb.copy()

    # Plot regions
    plot_SUEP_combined(args, plots)
