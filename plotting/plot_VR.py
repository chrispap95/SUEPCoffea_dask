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
        "--year",
        type=str,
        default="2018",
        help="Year of the data. Default is 2018.",
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
        help="Plot data as well. Default is False.",
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
        default=str(pathlib.Path(__file__).parent / "vr_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'vr_plots'}.",
    )
    return parser.parse_args()


region_labels = {
    "VR_loose": r"$VR_{loose}$",
    "VR_loose_extrapolation": r"$VR_{loose}$ + extrapolation",
    "VR_tight": r"$VR_{tight}$",
    "VR_tight_extrapolation": r"$VR_{tight}$ + extrapolation",
}


def calculate_QCD_k_factor(args, plots, region="VR_loose"):
    mc_processes = [
        "Higgs",
        "ST_NLO",
        "WJets",
        "VV+VVV",
        "TT_powheg",
        "DY",
    ]
    non_QCD_bkg = plots["DY_" + args.year][region].copy().reset()
    for process in mc_processes:
        non_QCD_bkg += plots[f"{process}_{args.year}"][region]
    k_factor = (
        plots["DoubleMuon_" + args.year][region].sum().value - non_QCD_bkg.sum().value
    ) / plots["QCD_Pt_MuEnrichedPt5_" + args.year][region].sum().value
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


def plot_VR(args, plots, region):
    extrapolation = "extrapolation" in region
    region = region.replace("_extrapolation", "")
    extrapolation_tag = ""
    if extrapolation:
        extrapolation_tag = "+extr."
    mc_processes = [
        ("Higgs", "Higgs"),
        ("TTV", r"$t\bar{t}+V$"),
        ("ST_NLO", r"single $t$"),
        ("WJets", r"$W+jets$"),
        ("VV+VVV", r"$VV+VVV$"),
        ("TT_powheg", r"$t\bar{t}$"),
        ("DY", "Drell-Yan"),
        ("QCD_Pt_MuEnrichedPt5", f"QCD{extrapolation_tag}"),
    ]
    data_name = ("DoubleMuon", "Data")

    # The following are signal samples with contamination in VR
    signal_processes = [
        "GluGluToSUEP_mS500.000_mPhi2.000_T8.000_modehadronic_13TeV",
        "GluGluToSUEP_mS1000.000_mPhi2.000_T4.000_modeleptonic_13TeV",
        "GluGluToSUEP_mS1000.000_mPhi4.000_T4.000_modeleptonic_13TeV",
        "GluGluToSUEP_mS125.000_mPhi1.000_T0.250_modeleptonic_13TeV",
        "GluGluToSUEP_mS500.000_mPhi8.000_T2.000_modeleptonic_13TeV",
    ]
    signal_labels = [
        r"$m_S=500\,$GeV,$m_\phi=2\,$GeV," + "\n" + r"$T=8\,$GeV, lep. decays",
        r"$m_S=1000\,$GeV,$m_\phi=2\,$GeV," + "\n" + r"$T=4\,$GeV, lep. decays",
        r"$m_S=1000\,$GeV,$m_\phi=4\,$GeV," + "\n" + r"$T=4\,$GeV, lep. decays",
        r"$m_S=125\,$GeV,$m_\phi=1\,$GeV," + "\n" + r"$T=0.25\,$GeV, lep. decays",
        r"$m_S=500\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=2\,$GeV, lep. decays",
    ]

    hists_mc = []
    hist_bkg_total = plots[f"QCD_Pt_MuEnrichedPt5_{args.year}"][region].copy().reset()

    for process, label in mc_processes:
        h_mc = plots[f"{process}_{args.year}"][
            (f"{region}_extrapolation" if "extr" in label else region)
        ]
        hists_mc.append(h_mc)
        hist_bkg_total += h_mc.copy()

    hists_signal = []
    for process in signal_processes:
        h_signal = plots[f"{process}_{args.year}"][region]
        hists_signal.append(h_signal)

    fig, ax1 = plt.subplots(figsize=(12, 12))

    if args.ratio:
        fig = plt.figure(figsize=(12, 13))
        plt.subplots_adjust(bottom=0.08, top=0.94, left=0.11, right=0.96)
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

    if args.data and region in plots[f"{data_name[0]}_{args.year}"]:
        hep.histplot(
            plots[f"{data_name[0]}_{args.year}"][region],
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
        plot_ratio(
            plots[f"{data_name[0]}_{args.year}"][region], hist_bkg_total, ax2, x_hatch
        )

    hep.cms.label(
        llabel="Preliminary",
        data=True,
        year=args.year,
        lumi=round(plot_utils.lumis[args.year] / 1000, 1),
        ax=ax1,
    )

    if extrapolation:
        region = f"{region}_extrapolation"
    region_label_coords = (0.7, 0.37)
    plt.text(
        *region_label_coords,
        region_labels[region],
        ha="center",
        weight="bold",
        transform=ax1.transAxes,
    )

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
    plt.ylim(1e-2, 1e10)
    plt.yscale("log")
    plt.legend(ncol=3)
    plt.ylabel("events")
    # plt.tight_layout()
    plt.savefig(f"{args.dest}/{region}_{args.year}_{args.tag}.pdf", bbox_inches="tight")
    plt.close()


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots = plot_utils.loader(
        tag=f"{args.tag}_{args.year}_VR",
        era=args.year,
        custom_lumi=args.lumi,
        load_data=args.data,
    )
    print("Done!", flush=True)

    # Apply k-factor to QCD
    if args.data and args.normalize:
        print("Calculate and apply k-factor to QCD...", end=" ", flush=True)
        k_factor = calculate_QCD_k_factor(args, plots)
        for plot in plots["QCD_Pt_MuEnrichedPt5_" + args.year]:
            plots["QCD_Pt_MuEnrichedPt5_" + args.year][plot] = (
                k_factor * plots["QCD_Pt_MuEnrichedPt5_" + args.year][plot]
            )
        print(f"k_QCD = {k_factor:.2f}  Done!", flush=True)

    print("Fit and extrapolation...", end=" ", flush=True)
    # QCD extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "VR_loose": slice(3j, None),
        "VR_tight": slice(3j, None),
    }

    qcd_extrapolation = plot_utils.Extrapolation(
        plots["QCD_Pt_MuEnrichedPt5_" + args.year], uncertainty_scheme="full"
    )
    qcd_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

    print("Done!", flush=True)

    # Plot regions
    regions = [
        "VR_loose",
        "VR_loose_extrapolation",
        "VR_tight",
        "VR_tight_extrapolation",
    ]
    for region in track(regions):
        plot_VR(args, plots, region)
