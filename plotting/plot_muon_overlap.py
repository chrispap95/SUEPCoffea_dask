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

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)

# Set to 10 color cycler
# plt.style.use(cms_styles.CMS_petroff_10)

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
        default="signal_dR_Oct2025",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--year",
        type=str,
        default="2018",
        help="Year of the data. Default is 2018.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "muon_overlap_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'muon_overlap_plots'}.",
    )
    return parser.parse_args()


def make_plot(args, plots, year, plot, decay_mode="leptonic"):
    cm_energy = "13TeV"
    if year.startswith("202") or year == "Run3":
        cm_energy = "13p6TeV"
    signal_processes = [
        f"GluGluToSUEP_mS125.000_mPhi1.000_T0.250_modeleptonic_{cm_energy}",
        f"GluGluToSUEP_mS125.000_mPhi4.000_T1.000_modeleptonic_{cm_energy}",
        f"GluGluToSUEP_mS125.000_mPhi2.000_T2.000_modeleptonic_{cm_energy}",
        f"GluGluToSUEP_mS125.000_mPhi8.000_T8.000_modeleptonic_{cm_energy}",
        f"GluGluToSUEP_mS125.000_mPhi8.000_T16.000_modeleptonic_{cm_energy}",
        f"GluGluToSUEP_mS125.000_mPhi8.000_T32.000_modeleptonic_{cm_energy}",
    ]
    signal_labels = [
        r"$m_S=125\,$GeV,$m_\phi=1\,$GeV," + "\n" + r"$T=0.25\,$GeV, lep. decays",
        r"$m_S=125\,$GeV,$m_\phi=4\,$GeV," + "\n" + r"$T=1\,$GeV, lep. decays",
        r"$m_S=125\,$GeV,$m_\phi=2\,$GeV," + "\n" + r"$T=2\,$GeV, lep. decays",
        r"$m_S=125\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=8\,$GeV, lep. decays",
        r"$m_S=125\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=16\,$GeV, lep. decays",
        r"$m_S=125\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=32\,$GeV, lep. decays",
    ]
    if decay_mode == "hadronic":
        signal_processes = [
            f"GluGluToSUEP_mS125.000_mPhi1.400_T0.350_modehadronic_{cm_energy}",
            f"GluGluToSUEP_mS125.000_mPhi4.000_T1.000_modehadronic_{cm_energy}",
            f"GluGluToSUEP_mS125.000_mPhi2.000_T2.000_modehadronic_{cm_energy}",
            f"GluGluToSUEP_mS125.000_mPhi8.000_T8.000_modehadronic_{cm_energy}",
            f"GluGluToSUEP_mS125.000_mPhi8.000_T16.000_modehadronic_{cm_energy}",
            f"GluGluToSUEP_mS125.000_mPhi8.000_T32.000_modehadronic_{cm_energy}",
        ]
        signal_labels = [
            r"$m_S=125\,$GeV,$m_\phi=1.4\,$GeV," + "\n" + r"$T=0.35\,$GeV, had. decays",
            r"$m_S=125\,$GeV,$m_\phi=4\,$GeV," + "\n" + r"$T=1\,$GeV, had. decays",
            r"$m_S=125\,$GeV,$m_\phi=2\,$GeV," + "\n" + r"$T=2\,$GeV, had. decays",
            r"$m_S=125\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=8\,$GeV, had. decays",
            r"$m_S=125\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=16\,$GeV, had. decays",
            r"$m_S=125\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=32\,$GeV, had. decays",
        ]

    hists_signal = []
    for process in signal_processes:
        h_signal = plots[f"{process}_{year}"][plot]
        hists_signal.append(h_signal)

    fig, ax1 = plt.subplots(figsize=(12.5, 12))

    hep.histplot(
        hists_signal,
        yerr=[np.sqrt(h.variances()) for h in hists_signal],
        label=[s for s in signal_labels],
        lw=3,
        ls="--",
        ax=ax1,
    )

    lumi_label = plot_utils.lumis[year]
    lumi_label = lumi_label / 1000  # Convert pb^-1 to fb^-1
    lumi_label = round(lumi_label, 2) if lumi_label < 1 else round(lumi_label, 1)
    hep.cms.label(
        llabel="Preliminary",
        data=True,
        year=year,
        lumi=lumi_label,
        com=13.6 if year.startswith("202") or year == "Run3" else 13,
        ax=ax1,
    )

    plt.sca(ax1)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
    plt.xscale("log")
    plt.ylim(0, 2e5)
    # plt.yscale("log")
    # plt.ylim(30, 5e6)

    ax1.legend(
        loc="upper left",
        ncol=2,
        columnspacing=1.1,
        frameon=False,
    )

    # plt.xlabel("OS dimuon pair dR")
    plt.ylabel("dimuon pairs")
    plt.savefig(
        os.path.join(args.dest, args.tag, f"{plot}_{year}_{decay_mode}.pdf"),
        bbox_inches="tight",
    )
    plt.close()


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(os.path.join(args.dest, args.tag), exist_ok=True)

    plots = plot_utils.loader(
        tag=f"{args.tag}_{args.year}",
        era=args.year,
        custom_lumi=None,
        load_data=False,
    )

    for plot in plots[list(plots.keys())[0]]:
        make_plot(args, plots, args.year, plot, decay_mode="leptonic")
        make_plot(args, plots, args.year, plot, decay_mode="hadronic")
