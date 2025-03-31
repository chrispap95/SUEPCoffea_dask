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
import plot_utils
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
        "--normalize",
        action="store_true",
        help="Normalize the QCD to the data. Default is False.",
    )
    parser.add_argument(
        "--unblind",
        action="store_true",
        help="Unblind the SRs. Default is False.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "regions_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'regions_plots'}",
    )
    return parser.parse_args()


def calculate_k_factor(plots, region="CR_cb", process="QCD_Pt_MuEnrichedPt5_2018"):
    mc_processes = [
        "Higgs_2018",
        "ST_NLO_2018",
        "WJets_2018",
        "VV+VVV_2018",
        "TT_powheg_2018",
        "DY_2018",
        "QCD_Pt_MuEnrichedPt5_2018",
    ]
    tot_bkg = plots["DY_2018"][region].copy().reset()
    for mc_proc in mc_processes:
        if process == mc_proc:
            continue
        tot_bkg += plots[mc_proc][region]
    k_factor = (
        plots["DoubleMuon_2018"][region].sum().value - tot_bkg.sum().value
    ) / plots[process][region].sum().value
    return k_factor


def plot_ratio(hist_data, hist_bkg_total, ax, x_hatch, args):
    slc = slice(None)
    if not args.unblind:
        slc = slice(0, -2)
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
        "GluGluToSUEP_mS125.000_mPhi1.000_T0.250_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS125.000_mPhi2.000_T2.000_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS125.000_mPhi4.000_T4.000_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS125.000_mPhi8.000_T8.000_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS125.000_mPhi8.000_T16.000_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS125.000_mPhi8.000_T32.000_modeleptonic_13TeV_2018",
    ]
    signal_labels = [
        r"$m_\phi=1\,$GeV, $T=0.25\,$GeV",
        r"$m_\phi=2\,$GeV, $T=2\,$GeV",
        r"$m_\phi=4\,$GeV, $T=4\,$GeV",
        r"$m_\phi=8\,$GeV, $T=8\,$GeV",
        r"$m_\phi=8\,$GeV, $T=16\,$GeV",
        r"$m_\phi=8\,$GeV, $T=32\,$GeV",
    ]

    hists_mc = []
    hist_bkg_total = plots["QCD_Pt_MuEnrichedPt5_2018"]["SUEP"].copy().reset()

    for process, label in mc_processes:
        h_mc = plots[process]["SUEP"]
        hists_mc.append(h_mc)
        hist_bkg_total += h_mc.copy()

    hists_signal = []
    for process in signal_processes:
        h_signal = plots[process]["SUEP"]
        hists_signal.append(h_signal)

    fig, ax1 = plt.subplots(figsize=(13.5, 11))

    if args.ratio:
        fig = plt.figure(figsize=(14, 12.5))
        plt.subplots_adjust(bottom=0.12, top=0.95, left=0.1, right=0.97)
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
        label="MC Stat. Unc.",
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
        color=["C8", "C9"] + ["C" + str(i) for i in range(len(signal_labels) - 2)],
        lw=3,
        ls="--",
        ax=ax1,
    )

    if args.ratio and args.data:
        plot_ratio(plots[data_name[0]]["SUEP"], hist_bkg_total, ax2, x_hatch, args)

    hep.cms.label(llabel="Preliminary", data=True, lumi=59.8, ax=ax1)

    ln_x_positions = [0, 4, 8, 10, 12]
    ln_y_upper = [3.83, 3.83, 3.63, 3.63, 3.83]
    lines = []
    for ln_x_pos, ln_y_pos in zip(ln_x_positions, ln_y_upper):
        lines.append(
            Line2D(
                [ln_x_pos, ln_x_pos],
                [0.1, ln_y_pos],
                figure=fig,
                transform=ax2.transData,
                color="black",
                linestyle="-",
                linewidth=4,
            )
        )
        fig.add_artist(lines[-1])

    region_y = 0.13
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
        r"$SR_\text{low T}$",
        ha="center",
        weight="bold",
        fontsize=30,
        transform=ax2.transData,
    )
    plt.text(
        11,
        region_y,
        r"$SR_\text{high T}$",
        ha="center",
        weight="bold",
        fontsize=30,
        transform=ax2.transData,
    )

    plt.text(
        0.65,
        0.6,
        r"$m_S=125\,$GeV, $m_{A'}=0.5\,$GeV",
        ha="center",
        fontsize=24,
        transform=ax1.transAxes,
    )

    labels = ["1", "2", "3", "4+", "2", "3", "4", "5+", "7+", "7+", ""]
    major_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    ax1.xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    ax1.set_xticklabels(labels)
    ax1.xaxis.set_minor_locator(ticker.NullLocator())
    # Create offset transform by 5 points in x direction
    dxs = np.array([25, 25, 25, 20, 25, 25, 25, 20, 45, 45, 0]) / 72.0
    dy = 0 / 72.0
    for label, dx in zip(ax2.get_xticklabels(), dxs):
        label.set_horizontalalignment("left")
        offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)  # type: ignore[attr-defined]
        label.set_transform(label.get_transform() + offset)

    if args.ratio and args.data:
        plt.sca(ax2)
        plt.ylim(0.5, 1.5)
        plt.ylabel("Data/MC")
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_xlabel("", visible=False)
    plt.xlabel(r"$n_{muon}$", fontsize=36, labelpad=35)
    plt.sca(ax1)
    plt.ylim(1e-2, 1e13)
    plt.yscale("log")
    plt.legend(ncol=3, loc="upper center")
    plt.ylabel("events")
    plt.tight_layout()
    plt.savefig(f"{args.dest}/prefit_all_regions_combined.pdf", bbox_inches="tight")
    plt.close()


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots_CR = plot_utils.loader(
        tag=f"{args.tag}_CR", custom_lumi=args.lumi, load_data=args.data
    )
    plots_SR = plot_utils.loader(
        tag=f"{args.tag}_SRs", custom_lumi=args.lumi, load_data=args.data
    )
    plots = {}
    all_datasets = set(plots_CR.keys()) | set(plots_SR.keys())
    for dataset in list(all_datasets):
        if dataset not in plots_CR:
            plots_CR[dataset] = {}
        if dataset not in plots_SR:
            plots_SR[dataset] = {}
        plots[dataset] = plots_CR[dataset] | plots_SR[dataset]
    print("Done!", flush=True)

    # Apply k-factor to QCD
    if args.data and args.normalize:
        print("Calculate and apply k-factors...", end=" ", flush=True)
        k_factor_QCD = calculate_k_factor(
            plots, region="CR_cb", process="QCD_Pt_MuEnrichedPt5_2018"
        )
        for plot in plots["QCD_Pt_MuEnrichedPt5_2018"]:
            plots["QCD_Pt_MuEnrichedPt5_2018"][plot] = (
                k_factor_QCD * plots["QCD_Pt_MuEnrichedPt5_2018"][plot]
            )
        print(f"k_QCD = {k_factor_QCD:.2f}  Done!", flush=True)
        k_factor_DY = calculate_k_factor(plots, region="CR_prompt", process="DY_2018")
        for plot in plots["DY_2018"]:
            plots["DY_2018"][plot] = k_factor_DY * plots["DY_2018"][plot]
        print(f"k_DY = {k_factor_DY:.2f}  Done!", flush=True)

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
        "SR_high_temp_tight": slice(3j, None),
    }
    dy_extrapolation = plot_utils.Extrapolation(
        plots["DY_2018"], uncertainty_scheme="full"
    )
    dy_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)
    print("Done!", flush=True)

    # Make combined plots
    for sample in plots:
        h_comb = hist.Hist.new.Variable(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
            name="SUEP",
        ).Weight()
        if "DoubleMuon" in sample and not args.unblind:
            h_comb = hist.Hist.new.Variable(
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                name="SUEP",
            ).Weight()

        h_comb[0] = plots[sample]["CR_cb"][1j]
        h_comb[1] = plots[sample]["CR_cb"][2j]
        h_comb[2] = plots[sample]["CR_cb"][3j]
        h_comb[3] = plots[sample]["CR_cb"][4j]
        h_comb[4] = plots[sample]["CR_prompt"][2j]
        h_comb[5] = plots[sample]["CR_prompt"][3j]
        h_comb[6] = plots[sample]["CR_prompt"][4j]
        h_comb[7] = plots[sample]["CR_prompt"][5j]
        if args.unblind or "DoubleMuon" not in sample:
            if "SR_low_temp_tight" in plots[sample]:
                h_comb[8] = plots[sample]["SR_low_temp_tight"][7j]
            if "SR_low_temp_tight_extrapolation" in plots[sample]:
                h_comb[8] = plots[sample]["SR_low_temp_tight_extrapolation"][7j]
            if "SR_high_temp_tight" in plots[sample]:
                h_comb[9] = plots[sample]["SR_high_temp_tight"][7j]
            if "SR_high_temp_tight_extrapolation" in plots[sample]:
                h_comb[9] = plots[sample]["SR_high_temp_tight_extrapolation"][7j]
        plots[sample]["SUEP"] = h_comb.copy()

    # Plot regions
    plot_SUEP_combined(args, plots)
