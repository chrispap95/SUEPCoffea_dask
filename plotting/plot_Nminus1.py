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
        default="Nminus1_Mar2025",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/muon_branches/SUEPCoffea_dask/plotting/Nminus1_plots",
        help="Destination directory to save the plots",
    )
    return parser.parse_args()


def make_plot(plots, region, plot):
    mc_processes = [
        ("Higgs_2018", "Higgs"),
        ("TTV_2018", "TTV"),
        ("ST_NLO_2018", "ST"),
        ("WJets_2018", "WJets"),
        ("VV+VVV_2018", "VV+VVV"),
        ("TT_powheg_2018", "TT"),
        ("DY_2018", f"DY"),
        ("QCD_Pt_MuEnrichedPt5_2018", f"QCD"),
    ]

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
        # signal_processes = [
        #     "GluGluToSUEP_mS200.000_mPhi1.000_T0.250_modeleptonic_13TeV_2018",
        #     "GluGluToSUEP_mS1000.000_mPhi1.000_T0.250_modeleptonic_13TeV_2018",
        #     "GluGluToSUEP_mS300.000_mPhi1.400_T0.350_modehadronic_13TeV_2018",
        #     "GluGluToSUEP_mS400.000_mPhi1.400_T0.350_modehadronic_13TeV_2018",
        #     "GluGluToSUEP_mS500.000_mPhi1.400_T0.350_modehadronic_13TeV_2018",
        #     "GluGluToSUEP_mS1000.000_mPhi1.400_T0.350_modehadronic_13TeV_2018",
        # ]
        # signal_labels = [
        #     r"$m_S=200\,$GeV,$m_\phi=1\,$GeV," + "\n" + r"$T=0.25\,$GeV, lep. decays",
        #     r"$m_S=1000\,$GeV,$m_\phi=1\,$GeV," + "\n" + r"$T=0.25\,$GeV, lep. decays",
        #     r"$m_S=300\,$GeV,$m_\phi=1.4\,$GeV," + "\n" + r"$T=0.35\,$GeV, had. decays",
        #     r"$m_S=400\,$GeV,$m_\phi=1.4\,$GeV," + "\n" + r"$T=0.35\,$GeV, had. decays",
        #     r"$m_S=500\,$GeV,$m_\phi=1.4\,$GeV," + "\n" + r"$T=0.35\,$GeV, had. decays",
        #     r"$m_S=1000\,$GeV,$m_\phi=1.4\,$GeV,"
        #     + "\n"
        #     + r"$T=0.35\,$GeV, had. decays",
        # ]

        signal_processes = [
            "GluGluToSUEP_mS125.000_mPhi1.400_T1.400_modehadronic_13TeV_2018",
            "GluGluToSUEP_mS125.000_mPhi4.000_T1.000_modehadronic_13TeV_2018",
            "GluGluToSUEP_mS125.000_mPhi2.000_T2.000_modeleptonic_13TeV_2018",
            "GluGluToSUEP_mS125.000_mPhi8.000_T4.000_modehadronic_13TeV_2018",
        ]
        signal_labels = [
            r"$m_S=125\,$GeV,$m_\phi=1.4\,$GeV," + "\n" + r"$T=1.4\,$GeV, had. decays",
            r"$m_S=125\,$GeV,$m_\phi=4\,$GeV," + "\n" + r"$T=1\,$GeV, had. decays",
            r"$m_S=125\,$GeV,$m_\phi=2\,$GeV," + "\n" + r"$T=2\,$GeV, lep. decays",
            r"$m_S=125\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=4\,$GeV, had. decays",
        ]

    xlabels = {
        "Nminus1_muon_pt_tight": r"muon $p_{T}$ (GeV)",
        "Nminus1_muon_pt_loose": r"muon $p_{T}$ (GeV)",
        "Nminus1_muon_ip3d_tight": r"muon $IP_{3D}$ (cm)",
        "Nminus1_muon_ip3d_loose": r"muon $IP_{3D}$ (cm)",
        "Nminus1_muon_iso_tight": "muon isolation",
        "Nminus1_muon_iso_loose": "muon isolation",
        "Nminus1_muon_neutral_iso_tight": "muon neutral isolation",
        "Nminus1_muon_neutral_iso_loose": "muon neutral isolation",
        "Nminus1_sph1_tight": r"$S_{1}$",
        "Nminus1_sph1_loose": r"$S_{1}$",
        "Nminus1_Z_mass_diff_tight": r"$m_\text{Z cand} - m_{Z}$ (GeV)",
        "Nminus1_Z_mass_diff_loose": r"$m_\text{Z cand} - m_{Z}$ (GeV)",
    }

    hists_mc = []
    hist_bkg_total = plots["QCD_Pt_MuEnrichedPt5_2018"][plot].copy().reset()

    for process, label in mc_processes:
        h_mc = plots[process][plot]
        hists_mc.append(h_mc)
        hist_bkg_total += h_mc.copy()

    hists_signal = []
    for process in signal_processes:
        h_signal = plots[process][plot]
        hists_signal.append(h_signal)

    fig, ax1 = plt.subplots(figsize=(18, 14))

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

    hep.histplot(
        hists_signal,
        yerr=[np.sqrt(h.variances()) for h in hists_signal],
        label=[s for s in signal_labels],
        lw=3,
        ls="--",
        ax=ax1,
    )

    hep.cms.label(llabel="Preliminary", data=True, lumi=59.8, ax=ax1)

    logx_plots = [
        "Nminus1_muon_ip3d_tight",
        "Nminus1_muon_ip3d_loose",
        "Nminus1_muon_iso_tight",
        "Nminus1_muon_iso_loose",
        "Nminus1_muon_neutral_iso_tight",
        "Nminus1_muon_neutral_iso_loose",
    ]
    plt.xlabel(xlabels[plot])
    if plot in logx_plots:
        plt.xscale("log")
    plt.yscale("log")
    if "loose" in plot:
        plt.ylim(1e2, 1e10)
    else:
        plt.ylim(1, 1e8)
    plt.legend(ncol=3)
    # plt.ylabel("events")
    plt.tight_layout()
    plt.savefig(f"{args.dest}/{region}_{plot}.pdf")
    plt.close()


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)

    # Load plots and merge them
    for region in ["SR_high_temp", "SR_low_temp"]:
        plots = plot_utils.loader(tag=f"{args.tag}_{region}")
        available_plots = list(plots["QCD_Pt_MuEnrichedPt5_2018"].keys())

        for plot in available_plots:
            make_plot(plots, region, plot)
