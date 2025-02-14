import argparse
import os
import warnings

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
        "--dest",
        type=str,
        default="/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/muon_branches/SUEPCoffea_dask/plotting/regions_plots",
        help="Destination directory to save the plots",
    )
    return parser.parse_args()


y_ranges = {
    "CR_cb": (10, 1e11),
    "CR_light": (10, 1e11),
    "CR_prompt": (10, 1e11),
    "SR_high_temp_loose_extrapolation": (1e-2, 1e12),
    "SR_high_temp_loose": (1e-2, 1e12),
    "SR_high_temp_tight_extrapolation": (1e-2, 1e12),
    "SR_high_temp_tight": (1e-2, 1e12),
    "SR_low_temp_loose_extrapolation": (1e-2, 1e14),
    "SR_low_temp_loose": (1e-2, 1e14),
    "SR_low_temp_tight_extrapolation": (1e-2, 1e14),
    "SR_low_temp_tight": (1e-2, 1e14),
}


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
    if "low_temp" in region:
        signal_processes = [
            "GluGluToSUEP_mS125.000_mPhi4.000_T1.000_modeleptonic_13TeV_2018",
            "GluGluToSUEP_mS125.000_mPhi2.000_T2.000_modeleptonic_13TeV_2018",
            "GluGluToSUEP_mS125.000_mPhi2.000_T2.000_modeleptonic_13TeV_2018",
            "GluGluToSUEP_mS125.000_mPhi4.000_T4.000_modeleptonic_13TeV_2018",
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

    fig, ax = plt.subplots(figsize=(13.5, 11))

    hep.histplot(
        hists_mc,
        yerr=[np.sqrt(h.variances()) for h in hists_mc],
        stack=True,
        label=[p[1] for p in mc_processes],
        histtype="fill",
        ec="black",
        lw=2,
        ax=ax,
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
    ax.fill_between(
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
            ax=ax,
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
        lw=3,
        ls="--",
        ax=ax,
    )

    if "SR" in region:
        plt.vlines(x=7, color="red", ymin=1e-3, ymax=1e6, lw=4)
        plt.annotate(
            "",
            xy=(7.5, 1e5),
            xytext=(7, 1e5),
            arrowprops=dict(facecolor="red", shrink=0),
        )

    hep.cms.label(llabel="Preliminary", data=True, lumi=59.8, ax=ax)

    if extrapolation:
        region = f"{region}_extrapolation"
    plt.text(
        0.5,
        0.67,
        region.replace("temp", "T").replace("prompt", "DY").replace("cb", "QCD"),
        ha="center",
        weight="bold",
        transform=ax.transAxes,
    )

    plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.ylim(y_ranges[region])
    plt.yscale("log")
    plt.legend(ncol=3)
    plt.xlabel(r"$n_{muon}$")
    plt.ylabel("events")
    plt.tight_layout()
    plt.savefig(f"{args.dest}/{region}.pdf")
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

    print("Fit and extrapolation...", end=" ", flush=True)
    # QCD extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "SR_low_temp_loose": slice(4j, None),
        "SR_low_temp_tight": slice(3j, None),
        "SR_high_temp_loose": slice(4j, None),
        "SR_high_temp_tight": slice(3j, None),
    }

    qcd_extrapolation = plot_utils.Extrapolation(plots["QCD_Pt_MuEnrichedPt5_2018"])
    qcd_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

    # DY extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "SR_low_temp_loose": slice(4j, None),
        "SR_low_temp_tight": slice(3j, None),
        "SR_high_temp_loose": slice(4j, None),
        "SR_high_temp_tight": slice(3j, None),
    }
    dy_extrapolation = plot_utils.Extrapolation(plots["DY_2018"])
    dy_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)
    print("Done!", flush=True)

    # Plot regions
    regions = [
        "CR_light",
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
        plot_region(args, plots, region)
