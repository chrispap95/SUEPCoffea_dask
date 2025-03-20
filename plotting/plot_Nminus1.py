import argparse
import os

import cms_styles
import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import mplhep as hep
import numpy as np
import plot_utils
from rich.progress import track  # type: ignore[import]

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

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


cuts = {
    # The tuples contains: (cut value, arrow location, arrow height)
    "CR_prompt_Nminus1_dimuon_mass": [(80, ">"), (100, "<")],
    "CR_prompt_Nminus1_cand_muon_pt": [(25, ">")],
    "CR_prompt_Nminus1_cand_muon_iso": [(0.1, "<")],
    "CR_prompt_Nminus1_cand_muon_ip3d": [(0.01, "<")],
    "CR_prompt_Nminus1_cand_muon_dxy": [(0.008, "<")],
    "CR_prompt_Nminus1_cand_muon_dz": [(0.01, "<")],
    "CR_prompt_Nminus1_muon_iso": [(0.1, ">")],
    "CR_prompt_Nminus1_muon_ip3d": [(0.015, ">")],
    "CR_prompt_Nminus1_muon_dxy": [(0.01, ">")],
    "CR_prompt_Nminus1_muon_dz": [(0.01, ">")],
    "CR_cb_Nminus1_muon_dxy": [(0.01, ">"), (0.2, "<")],
    "SR_low_temp_loose_Nminus1_muon_pt": [(45, "<")],
    "SR_low_temp_tight_Nminus1_muon_pt": [(35, "<")],
    "SR_low_temp_loose_Nminus1_muon_ip3d": [(0.1, "<")],
    "SR_low_temp_tight_Nminus1_muon_ip3d": [(0.007, "<")],
    "SR_low_temp_loose_Nminus1_sph1": [(0.2, ">")],
    "SR_low_temp_tight_Nminus1_sph1": [(0.7, ">")],
    "SR_low_temp_loose_Nminus1_dimuon_mass": [(45, "<")],
    "SR_low_temp_tight_Nminus1_dimuon_mass": [(35, "<")],
    "SR_high_temp_loose_Nminus1_muon_ip3d": [(0.1, "<")],
    "SR_high_temp_tight_Nminus1_muon_ip3d": [(0.007, "<")],
    "SR_high_temp_loose_Nminus1_muon_iso": [(5, "<")],
    "SR_high_temp_tight_Nminus1_muon_iso": [(0.65, "<")],
    "SR_high_temp_loose_Nminus1_muon_neutral_iso": [(3, "<")],
    "SR_high_temp_tight_Nminus1_muon_neutral_iso": [(0.5, "<")],
    "SR_high_temp_loose_Nminus1_dimuon_mass": [(70, "<")],
    "SR_high_temp_tight_Nminus1_dimuon_mass": [(70, "<")],
}

ylims = {
    "CR_prompt_Nminus1_dimuon_mass": (1e0, 1e8),
    "CR_prompt_Nminus1_cand_muon_pt": (1e0, 1e8),
    "CR_prompt_Nminus1_cand_muon_iso": (1e0, 1e8),
    "CR_prompt_Nminus1_cand_muon_ip3d": (1e0, 1e8),
    "CR_prompt_Nminus1_cand_muon_dxy": (1e0, 1e8),
    "CR_prompt_Nminus1_cand_muon_dz": (1e0, 1e8),
    "CR_prompt_Nminus1_muon_iso": (1e0, 1e8),
    "CR_prompt_Nminus1_muon_ip3d": (1e0, 1e8),
    "CR_prompt_Nminus1_muon_dxy": (1e0, 1e8),
    "CR_prompt_Nminus1_muon_dz": (1e0, 1e8),
    "CR_cb_Nminus1_muon_dxy": (1e2, 1e10),
    "SR_low_temp_tight_Nminus1_muon_pt": (1, 1e8),
    "SR_low_temp_loose_Nminus1_muon_pt": (1e2, 1e10),
    "SR_low_temp_tight_Nminus1_muon_ip3d": (1e2, 1e10),
    "SR_low_temp_loose_Nminus1_muon_ip3d": (1e2, 1e10),
    "SR_low_temp_tight_Nminus1_sph1": (10, 1e9),
    "SR_low_temp_loose_Nminus1_sph1": (1e2, 1e10),
    "SR_low_temp_tight_Nminus1_dimuon_mass": (1, 1e8),
    "SR_low_temp_loose_Nminus1_dimuon_mass": (1e2, 1e10),
    "SR_high_temp_tight_Nminus1_muon_ip3d": (1e2, 1e10),
    "SR_high_temp_loose_Nminus1_muon_ip3d": (1e2, 1e10),
    "SR_high_temp_tight_Nminus1_muon_iso": (10, 1e9),
    "SR_high_temp_loose_Nminus1_muon_iso": (1e2, 1e10),
    "SR_high_temp_tight_Nminus1_muon_neutral_iso": (10, 1e9),
    "SR_high_temp_loose_Nminus1_muon_neutral_iso": (1e2, 1e10),
    "SR_high_temp_tight_Nminus1_dimuon_mass": (1, 1e8),
    "SR_high_temp_loose_Nminus1_dimuon_mass": (1e2, 1e10),
}

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


def get_xlabel(plot):
    xlabels = {
        "Nminus1_muon_pt": r"muon $p_{T}$ (GeV)",
        "Nminus1_muon_ip3d": r"muon $IP_{3D}$ (cm)",
        "Nminus1_muon_iso": "muon isolation",
        "Nminus1_muon_neutral_iso": "muon neutral isolation",
        "Nminus1_sph1": r"$S_{1}$",
        "Nminus1_dimuon_mass": r"$m_{\mu\mu}$ (GeV)",
        "Nminus1_cand_muon_pt": r"prompt muon $p_{T}$ (GeV)",
        "Nminus1_cand_muon_iso": "prompt muon isolation",
        "Nminus1_cand_muon_ip3d": r"prompt muon $IP_{3D}$ (cm)",
        "Nminus1_cand_muon_dxy": r"prompt muon $|d_{xy}|$ (cm)",
        "Nminus1_cand_muon_dz": r"prompt muon $|d_{z}|$ (cm)",
        "Nminus1_muon_ip3d": r"muon $IP_{3D}$ (cm)",
        "Nminus1_muon_dxy": r"muon $|d_{xy}|$ (cm)",
        "Nminus1_muon_dz": r"muon $|d_{z}|$ (cm)",
    }
    for key in xlabels:
        if key in plot:
            return xlabels[key]
    return plot


logx_plots = [
    "muon_ip3d",
    "muon_dxy",
    "muon_dz",
    "muon_iso",
    "muon_neutral_iso",
]


def make_plot(plots, plot):
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
    if "low_temp" in plot:
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

    hists_mc = []
    slc = (
        slice(None, None, 2j)
        if "sph1" in plot or "dimuon" in plot
        else (slice(None, None, 2j), slice(None, None, sum))
    )
    hist_bkg_total = plots["QCD_Pt_MuEnrichedPt5_2018"][plot][slc].copy().reset()

    for process, label in mc_processes:
        h_mc = plots[process][plot][slc]
        hists_mc.append(h_mc)
        hist_bkg_total += h_mc.copy()

    hists_signal = []
    for process in signal_processes:
        h_signal = plots[process][plot][slc]
        hists_signal.append(h_signal)

    fig, ax1 = plt.subplots(figsize=(12, 12))

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

    # Check if logx
    islogx = False
    for key in logx_plots:
        if key in plot:
            islogx = True
            break

    # Draw cuts
    # Calculate relative positions and lengths
    cut_list = cuts[plot]
    xrange_min, xrange_max = (
        hist_bkg_total.axes[0].edges[0],
        hist_bkg_total.axes[0].edges[-1],
    )
    arrow_length_rel = 0.07
    arrow_length_abs = arrow_length_rel * (xrange_max - xrange_min)
    if islogx:
        arrow_length_abs = 10 ** (arrow_length_rel * np.log10(xrange_max / xrange_min))
    yrange_min, yrange_max = ylims[plot]
    y_position_rel = 0.6
    y_position_abs = yrange_min * 10 ** (
        y_position_rel * np.log10(yrange_max / yrange_min)
    )
    for cut in cut_list:
        if cut[1] == "<":
            arrow_position = (
                cut[0] / arrow_length_abs if islogx else cut[0] - arrow_length_abs
            )
        else:
            arrow_position = (
                cut[0] * arrow_length_abs if islogx else cut[0] + arrow_length_abs
            )
        plt.vlines(x=cut[0], color="black", ymin=1e-3, ymax=2 * y_position_abs, lw=7)
        plt.annotate(
            "",
            xy=(arrow_position, y_position_abs),
            xytext=(cut[0], y_position_abs),
            arrowprops=dict(
                arrowstyle="simple",  # Gives a filled arrow with an outline
                facecolor="red",  # Fills arrow with red
                edgecolor="black",  # Black outline
                linewidth=2,  # Outline thickness
                mutation_scale=40,  # Increases overall arrow size
            ),
        )

    plt.text(
        0.3,
        0.68,
        region_labels[plot.split("_Nminus1")[0]],
        ha="center",
        weight="bold",
        fontsize=30,
        transform=ax1.transAxes,
    )

    hep.cms.label(llabel="Preliminary", data=True, lumi=59.8, ax=ax1)

    plt.xlabel(get_xlabel(plot))
    if islogx:
        plt.xscale("log")
    plt.yscale("log")
    plt.ylim(ylims[plot])
    plt.legend(ncol=3, loc="upper center")
    plt.ylabel("muons")
    if "sph1" in plot or "dimuon" in plot:
        plt.ylabel("events")
    plt.tight_layout()
    plt.savefig(f"{args.dest}/{plot}.pdf", bbox_inches="tight")
    plt.close()


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)

    # Load plots
    print("Loading plots...", end=" ", flush=True)
    plots_CRs = plot_utils.loader(tag=f"{args.tag}_CRs")
    plots_SR_high_temp = plot_utils.loader(tag=f"{args.tag}_SR_high_temp")
    plots_SR_low_temp = plot_utils.loader(tag=f"{args.tag}_SR_low_temp")
    plots = {}
    all_datasets = (
        set(plots_CRs.keys())
        | set(plots_SR_high_temp.keys())
        | set(plots_SR_low_temp.keys())
    )
    for dataset in list(all_datasets):
        if dataset not in plots_CRs:
            plots_CRs[dataset] = {}
        if dataset not in plots_SR_high_temp:
            plots_SR_high_temp[dataset] = {}
        if dataset not in plots_SR_low_temp:
            plots_SR_low_temp[dataset] = {}
        plots[dataset] = (
            plots_CRs[dataset]
            | plots_SR_high_temp[dataset]
            | plots_SR_low_temp[dataset]
        )
    print("Done!", flush=True)

    # Load plots and merge them
    for plot in track(plots["QCD_Pt_MuEnrichedPt5_2018"].keys()):
        make_plot(plots, plot)
