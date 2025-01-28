import argparse
import os
import warnings

import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.ticker as ticker  # type: ignore[import]
import mplhep as hep
import numpy as np
import plot_utils
from cycler import cycler  # type: ignore[import]
from rich.progress import track  # type: ignore[import]

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

warnings.filterwarnings("ignore")

# Color palette for 10 colors
cmap_petroff_6 = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
cmap_petroff_10 = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]
CMS = {"axes.prop_cycle": cycler("color", cmap_petroff_10)}
plt.style.use(CMS)


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
        "--dest",
        type=str,
        default="/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/muon_branches/SUEPCoffea_dask/plotting/regions_plots",
        help="Destination directory to save the plots",
    )
    return parser.parse_args()


def plot_region(args, plots, region):
    extrapolation = "extrapolation" in region
    region = region.replace("_extrapolation", "")
    extrapolation_tag = ""
    if extrapolation:
        extrapolation_tag = "+extr."
    mc_processes = [
        ("Higgs_2018", "Higgs"),
        # ('TTV_2018', 'TTV'),
        ("ST_NLO_2018", "ST"),
        ("WJets_2018", "WJets"),
        ("VV+VVV_2018", "VV+VVV"),
        ("TT_powheg_2018", "TT"),
        ("DY_2018", f"DY{extrapolation_tag}"),
        ("QCD_Pt_MuEnrichedPt5_2018", f"QCD{extrapolation_tag}"),
    ]

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

    fig, ax = plt.subplots(figsize=(12.5, 11))

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

    hep.cms.label(llabel="Preliminary", data=True, lumi=55, ax=ax)

    region_loc = (ax.get_xlim()[1] + ax.get_xlim()[0]) * 0.5
    plt.text(
        region_loc,
        2e5,
        region.replace("temp", "T").replace("prompt", "DY").replace("cb", "QCD"),
        ha="center",
        weight="bold",
    )

    plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.ylim(1e-3, 1e11)
    plt.yscale("log")
    plt.legend(ncol=2)
    plt.xlabel(r"$n_{muon}$")
    plt.ylabel("events")
    plt.tight_layout()
    if extrapolation:
        region = f"{region}_extrapolation"
    plt.savefig(f"{args.dest}/{region}.pdf")
    plt.close()


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots_CR = plot_utils.loader(
        tag=f"{args.tag}_CR", custom_lumi=args.lumi, load_data=False
    )
    plots_VR = plot_utils.loader(
        tag=f"{args.tag}_VR", custom_lumi=args.lumi, load_data=False
    )
    plots_SR = plot_utils.loader(
        tag=f"{args.tag}_SRs", custom_lumi=args.lumi, load_data=False
    )
    plots = {}
    for dataset in plots_CR:
        # Note: need to fix this to be mergeable even when data for SR is missing! (blinded...)
        # This merges two dicts!
        plots[dataset] = plots_CR[dataset] | plots_VR[dataset] | plots_SR[dataset]
    print("Done!", flush=True)

    print("Fit and extrapolation...", end=" ", flush=True)
    # QCD extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "VR_loose": slice(4j, None),
        "VR_tight": slice(3j, None),
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
        "VR_loose": slice(4j, None),
        "VR_tight": slice(3j, None),
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
        "SR_low_temp_tight",
        "SR_low_temp_tight_extrapolation",
        "SR_high_temp_loose",
        "SR_high_temp_tight",
        "SR_high_temp_tight_extrapolation",
        "VR_loose",
        "VR_tight",
        "VR_tight_extrapolation",
    ]
    for region in track(regions):
        plot_region(args, plots, region)
