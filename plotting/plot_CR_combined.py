import argparse
import os
import pathlib
import warnings

import cms_styles
import hist
import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.ticker as ticker  # type: ignore[import]
import mplhep as hep
import numpy as np
import plot_utils

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

warnings.filterwarnings("ignore")

# Set to 10 color cycler
plt.style.use(cms_styles.CMS_petroff_10)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Dec2024",
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
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "regions_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'regions_plots'}",
    )
    return parser.parse_args()


def plot_ratio(hist_data, hist_bkg_total, ax, x_hatch):
    ratio = np.divide(
        hist_data.values(),
        hist_bkg_total.values(),
        out=np.ones_like(hist_data.values()),
        where=hist_bkg_total.values() != 0,
    )
    ratio_err = np.where(
        hist_bkg_total.values() > 0,
        np.sqrt(
            (hist_bkg_total.values() ** -2) * (hist_data.variances())
            + (hist_data.values() ** 2 * hist_bkg_total.values() ** -4)
            * (hist_bkg_total.variances())
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


def plot_CR_combined(args, plots):
    mc_processes = [
        ("Higgs_2018", "Higgs"),
        ("TTV_2018", "TTV"),
        ("ST_NLO_2018", "ST"),
        ("WJets_2018", "WJets"),
        ("VV+VVV_2018", "VV+VVV"),
        ("TT_powheg_2018", "TT"),
        ("DY_2018", "DY"),
        ("QCD_Pt_MuEnrichedPt5_2018", "QCD"),
    ]
    data_name = ("DoubleMuon_2018", "Data")

    signal_processes = [
        "GluGluToSUEP_mS125.000_mPhi8.000_T8.000_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS125.000_mPhi4.000_T16.000_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS125.000_mPhi8.000_T16.000_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS125.000_mPhi8.000_T32.000_modeleptonic_13TeV_2018",
    ]
    # signal_processes = [
    #     "GluGluToSUEP_mS125.000_mPhi4.000_T1.000_modeleptonic_13TeV_2018",
    #     "GluGluToSUEP_mS125.000_mPhi2.000_T2.000_modeleptonic_13TeV_2018",
    #     "GluGluToSUEP_mS125.000_mPhi2.000_T2.000_modeleptonic_13TeV_2018",
    #     "GluGluToSUEP_mS125.000_mPhi4.000_T4.000_modeleptonic_13TeV_2018",
    # ]

    hists_mc = []
    hist_bkg_total = plots["QCD_Pt_MuEnrichedPt5_2018"]["CR_combined"].copy().reset()

    for process, label in mc_processes:
        h_mc = plots[process]["CR_combined"]
        hists_mc.append(h_mc)
        hist_bkg_total += h_mc.copy()

    hists_signal = []
    for process in signal_processes:
        h_signal = plots[process]["CR_combined"]
        hists_signal.append(h_signal)

    fig, ax1 = plt.subplots(figsize=(13.5, 11))

    if args.ratio:
        fig = plt.figure(figsize=(14, 13))
        plt.subplots_adjust(bottom=0.15, top=0.92, left=0.1, right=0.95)
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

    hep.histplot(
        hists_mc,
        yerr=[np.sqrt(h.variances()) for h in hists_mc],
        stack=True,
        label=[p[1] for p in mc_processes],
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

    if args.data and "CR_combined" in plots[data_name[0]]:
        hep.histplot(
            plots[data_name[0]]["CR_combined"],
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
        label=[
            s.replace("GluGluTo", "")
            .replace(".000", "")
            .replace("mode", "")
            .replace("_13TeV_2018", "")
            for s in signal_processes
        ],
        color=["C8", "C9", "C0", "C1"],
        lw=3,
        ls="--",
        ax=ax1,
    )

    if args.ratio and args.data:
        plot_ratio(plots[data_name[0]]["CR_combined"], hist_bkg_total, ax2, x_hatch)

    hep.cms.label(llabel="Preliminary", data=True, lumi=59.8, ax=ax1)

    plt.text(
        0.5,
        0.62,
        "CR",
        ha="center",
        weight="bold",
        fontsize=32,
        transform=ax1.transAxes,
    )

    if args.ratio and args.data:
        plt.sca(ax2)
        plt.ylim(0.5, 1.5)
        plt.ylabel("Data/MC")
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_xlabel("", visible=False)
    plt.xticks(rotation=330)
    plt.xlabel("CR bin")
    plt.sca(ax1)
    plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
    # plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.ylim(10, 1e11)
    plt.yscale("log")
    plt.legend(ncol=3)
    plt.ylabel("events")
    plt.tight_layout()
    plt.savefig(f"{args.dest}/CR_combined.pdf")
    plt.close()


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots = plot_utils.loader(
        tag=f"{args.tag}_CR", custom_lumi=args.lumi, load_data=args.data
    )
    print("Done!", flush=True)

    # Make combined CR plots
    for sample in plots:
        h_comb = hist.Hist.new.StrCat(
            [
                "CR_QCD bin 1",
                "CR_QCD bin 2",
                "CR_QCD bin 3",
                "CR_QCD bin 4",
                "CR_DY bin 1",
            ],
            name="CR",
        ).Weight()
        h_comb["CR_QCD bin 1"] = plots[sample]["CR_cb"][1j]
        h_comb["CR_QCD bin 2"] = plots[sample]["CR_cb"][2j]
        h_comb["CR_QCD bin 3"] = plots[sample]["CR_cb"][3j]
        h_comb["CR_QCD bin 4"] = plots[sample]["CR_cb"][4j]
        h_comb["CR_DY bin 1"] = plots[sample]["CR_prompt"][2j]
        plots[sample]["CR_combined"] = h_comb.copy()

    # Plot regions
    plot_CR_combined(args, plots)
