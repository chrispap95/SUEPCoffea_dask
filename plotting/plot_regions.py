import argparse
import logging
import os
import pathlib

import cms_styles
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

# Set to 10 color cycler
plt.style.use(cms_styles.CMS_petroff_10)

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
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "regions_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'regions_plots'}.",
    )
    parser.add_argument(
        "--CRs",
        action="store_true",
        help="Process only CRs. Default is False.",
    )
    parser.add_argument(
        "--SRs",
        action="store_true",
        help="Process only SRs. Default is False.",
    )
    return parser.parse_args()


region_labels = {
    "CR_cb": r"$CR_{QCD}$",
    "CR_prompt": r"$CR_{DY}$",
    "SR_high_temp_loose": r"$SR^{loose}_{high~T}$",
    "SR_high_temp_loose_extrapolation": r"$SR^{loose}_{high~T}$ + extrapolation",
    "SR_high_temp_tight": r"$SR^{tight}_{high~T}$",
    "SR_high_temp_tight_extrapolation": r"$SR^{tight}_{high~T}$ + extrapolation",
    "SR_low_temp_loose": r"$SR^{loose}_{low~T}$",
    "SR_low_temp_loose_extrapolation": r"$SR^{loose}_{low~T}$ + extrapolation",
    "SR_low_temp_tight": r"$SR^{tight}_{low~T}$",
    "SR_low_temp_tight_extrapolation": r"$SR^{tight}_{low~T}$ + extrapolation",
}

y_ranges = {
    "CR_cb": (10, 1e11),
    "CR_light": (10, 1e11),
    "CR_prompt": (1, 1e8),
    "SR_high_temp_loose_extrapolation": (1e-2, 1e12),
    "SR_high_temp_loose": (1e-2, 1e12),
    "SR_high_temp_tight_extrapolation": (1e-2, 1e12),
    "SR_high_temp_tight": (1e-2, 1e12),
    "SR_low_temp_loose_extrapolation": (1e-2, 1e12),
    "SR_low_temp_loose": (1e-2, 1e12),
    "SR_low_temp_tight_extrapolation": (1e-2, 1e12),
    "SR_low_temp_tight": (1e-2, 1e12),
}


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
        markersize=7,
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
    ax.axhline(1, ls="--", color="gray")


def plot_region(args, plots, region):
    extrapolation = "extrapolation" in region
    region = region.replace("_extrapolation", "")
    extrapolation_tag = ""
    if extrapolation:
        extrapolation_tag = "+extr."
    mc_processes = [
        ("Higgs_2018", "Higgs"),
        ("TTV_2018", "TTV"),
        ("ST_NLO_2018", "ST"),
        ("WJets_2018", "WJets"),
        ("VV+VVV_2018", "VV+VVV"),
        ("TT_powheg_2018", "TT"),
        ("DY_2018", f"DY{extrapolation_tag}"),
        ("QCD_Pt_MuEnrichedPt5_2018", f"QCD{extrapolation_tag}"),
    ]
    data_name = ("DoubleMuon_2018", "Data")

    signal_processes = [
        "GluGluToSUEP_mS125.000_mPhi8.000_T8.000_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS125.000_mPhi4.000_T16.000_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS125.000_mPhi8.000_T16.000_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS125.000_mPhi8.000_T32.000_modeleptonic_13TeV_2018",
    ]
    signal_labels = [
        r"$m_S=125\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=8\,$GeV, lep. decays",
        r"$m_S=125\,$GeV,$m_\phi=4\,$GeV," + "\n" + r"$T=16\,$GeV, lep. decays",
        r"$m_S=125\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=16\,$GeV, lep. decays",
        r"$m_S=125\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=32\,$GeV, lep. decays",
    ]
    if "low_temp" in region:
        signal_processes = [
            "GluGluToSUEP_mS125.000_mPhi4.000_T1.000_modeleptonic_13TeV_2018",
            "GluGluToSUEP_mS125.000_mPhi2.000_T2.000_modeleptonic_13TeV_2018",
            "GluGluToSUEP_mS125.000_mPhi2.000_T2.000_modeleptonic_13TeV_2018",
            "GluGluToSUEP_mS125.000_mPhi4.000_T4.000_modeleptonic_13TeV_2018",
        ]
        signal_labels = [
            r"$m_S=125\,$GeV,$m_\phi=4\,$GeV," + "\n" + r"$T=1\,$GeV, lep. decays",
            r"$m_S=125\,$GeV,$m_\phi=2\,$GeV," + "\n" + r"$T=2\,$GeV, lep. decays",
            r"$m_S=125\,$GeV,$m_\phi=2\,$GeV," + "\n" + r"$T=2\,$GeV, lep. decays",
            r"$m_S=125\,$GeV,$m_\phi=4\,$GeV," + "\n" + r"$T=4\,$GeV, lep. decays",
        ]

    hists_mc = []
    hist_bkg_total = plots["QCD_Pt_MuEnrichedPt5_2018"][region].copy().reset()

    for process, label in mc_processes:
        h_mc = plots[process][
            (f"{region}_extrapolation" if "extr" in label else region)
        ]
        hists_mc.append(h_mc)
        hist_bkg_total += h_mc.copy()

    hists_signal = []
    for process in signal_processes:
        h_signal = plots[process][region]
        hists_signal.append(h_signal)

    fig, ax1 = plt.subplots(figsize=(12, 12))

    if args.ratio:
        fig = plt.figure(figsize=(12, 13))
        plt.subplots_adjust(bottom=0.08, top=0.92, left=0.1, right=0.95)
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

    if args.data and region in plots[data_name[0]]:
        hep.histplot(
            plots[data_name[0]][region],
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
        lw=3,
        ls="--",
        ax=ax1,
    )

    if args.ratio and args.data:
        plot_ratio(plots[data_name[0]][region], hist_bkg_total, ax2, x_hatch)

    if "SR" in region:
        plt.vlines(x=7, color="red", ymin=1e-3, ymax=1e6, lw=7)
        plt.annotate(
            "",
            xy=(7.5, 1e5),
            xytext=(7, 1e5),
            arrowprops=dict(
                arrowstyle="simple",  # Gives a filled arrow with an outline
                facecolor="red",  # Fills arrow with red
                edgecolor="black",  # Black outline
                linewidth=2,  # Outline thickness
                mutation_scale=40,  # Increases overall arrow size
            ),
        )

    hep.cms.label(llabel="Preliminary", data=True, lumi=59.8, ax=ax1)

    if extrapolation:
        region = f"{region}_extrapolation"
    region_label_coords = (0.51, 0.61)
    if "CR" in region:
        region_label_coords = (0.4, 0.62)
    plt.text(
        *region_label_coords,
        region_labels[region],
        ha="center",
        weight="bold",
        transform=ax1.transAxes,
    )

    # modify last x tick label
    if "CR_cb" in region:
        labels = ["", "1", "2", "3", "4+", ""]
        ax1.set_xticklabels(labels)
    elif "CR_prompt" in region:
        labels = ["", "2", "3", "4", "5+", ""]
        ax1.set_xticklabels(labels)
    elif "SR" in region:
        ax1.set_xticks([2.75, 3, 4, 5, 6, 7, 8.25])
        ax1.set_xticklabels(["", "3", "4", "5", "6", "7+", ""])

    if args.ratio and args.data:
        plt.sca(ax2)
        plt.ylim(0.5, 1.5)
        plt.ylabel("Data/MC")
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_xlabel("", visible=False)
    plt.xlabel(r"$n_{muon}$")
    plt.sca(ax1)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
    plt.ylim(y_ranges[region])
    plt.yscale("log")
    plt.legend(ncol=3, columnspacing=1, loc="upper center")
    plt.ylabel("events")
    plt.tight_layout()
    plt.savefig(f"{args.dest}/{region}.pdf", bbox_inches="tight")
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
        "SR_high_temp_tight": slice(4j, None),
    }
    dy_extrapolation = plot_utils.Extrapolation(
        plots["DY_2018"], uncertainty_scheme="full"
    )
    dy_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)
    print("Done!", flush=True)

    # Plot regions
    regions = [
        "CR_prompt",
        "CR_cb",
        "SR_low_temp_loose",
        "SR_low_temp_loose_extrapolation",
        "SR_low_temp_tight",
        "SR_low_temp_tight_extrapolation",
        "SR_high_temp_loose",
        "SR_high_temp_loose_extrapolation",
        "SR_high_temp_tight",
        "SR_high_temp_tight_extrapolation",
    ]
    for region in track(regions):
        if args.CRs and "CR" not in region:
            continue
        if args.SRs and "SR" not in region:
            continue
        plot_region(args, plots, region)
