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
        default="/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/muon_branches/SUEPCoffea_dask/plotting/vr_plots",
        help="Destination directory to save the plots",
    )
    return parser.parse_args()


def calculate_k_factor(plots):
    mc_processes = [
        "Higgs_2018",
        "ST_NLO_2018",
        "WJets_2018",
        "VV+VVV_2018",
        "TT_powheg_2018",
        "DY_2018",
    ]
    non_QCD_bkg = plots["DY_2018"]["VR_loose"].copy().reset()
    for process in mc_processes:
        non_QCD_bkg += plots[process]["VR_loose"]
    k_factor = (
        plots["DoubleMuon_2018"]["VR_loose"][0].value - non_QCD_bkg[0].value
    ) / plots["QCD_Pt_MuEnrichedPt5_2018"]["VR_loose"][0].value
    return k_factor


def plot_VR(args, plots, region):
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
        ("DY_2018", "DY"),
        ("QCD_Pt_MuEnrichedPt5_2018", f"QCD{extrapolation_tag}"),
    ]
    data_name = ("DoubleMuon_2018", "Data")

    # The following are signal samples with contamination in VR
    signal_processes = [
        "GluGluToSUEP_mS500.000_mPhi2.000_T8.000_modehadronic_13TeV_2018",
        "GluGluToSUEP_mS1000.000_mPhi2.000_T4.000_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS1000.000_mPhi4.000_T4.000_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS125.000_mPhi1.000_T0.250_modeleptonic_13TeV_2018",
        "GluGluToSUEP_mS500.000_mPhi8.000_T2.000_modeleptonic_13TeV_2018",
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
            .replace(".250", ".25")
            .replace("mode", "")
            .replace("_13TeV_2018", "")
            for s in signal_processes
        ],
        lw=3,
        ls="--",
        ax=ax,
    )

    hep.cms.label(llabel="Preliminary", data=True, lumi=59.8, ax=ax)

    if extrapolation:
        region = f"{region}_extrapolation"
    plt.text(
        0.5,
        0.67,
        region.replace("temp", "T"),
        ha="center",
        weight="bold",
        transform=ax.transAxes,
    )

    plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.ylim(1e-2, 1e10)
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
    plots = plot_utils.loader(
        tag=f"{args.tag}_VR", custom_lumi=args.lumi, load_data=args.data
    )
    print("Done!", flush=True)

    # Apply k-factor to QCD
    if args.data and args.normalize:
        print("Calculate and apply k-factor to QCD...", end=" ", flush=True)
        k_factor = calculate_k_factor(plots)
        for plot in plots["QCD_Pt_MuEnrichedPt5_2018"]:
            plots["QCD_Pt_MuEnrichedPt5_2018"][plot] = (
                k_factor * plots["QCD_Pt_MuEnrichedPt5_2018"][plot]
            )
        print(f"k_QCD = {k_factor:.2f}  Done!", flush=True)

    print("Fit and extrapolation...", end=" ", flush=True)
    # QCD extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "VR_loose": slice(3j, None),
        "VR_tight": slice(3j, None),
    }

    qcd_extrapolation = plot_utils.Extrapolation(plots["QCD_Pt_MuEnrichedPt5_2018"])
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
