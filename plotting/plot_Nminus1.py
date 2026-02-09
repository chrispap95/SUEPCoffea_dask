import argparse
import os
import pathlib

import cms_styles
import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import mplhep as hep
import numpy as np
import plot_utils
from rich.progress import Progress, track  # type: ignore[import]

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
        default="Nminus1_Feb2026",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--year",
        type=str,
        nargs="*",
        default=["2018"],
        help="Year of the data. Default is 2018. Can be a single year or multiple years.",
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
        default=str(pathlib.Path(__file__).parent / "Nminus1_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'Nminus1_plots'}",
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


cuts = {
    # The tuples contains: (cut value, arrow location)
    "CR_prompt_Nminus1_dimuon_mass": [(86.2, ">"), (96.2, "<")],
    "CR_prompt_Nminus1_prompt_muon_pt": [(25, ">")],
    "CR_prompt_Nminus1_prompt_muon_iso": [(0.1, "<")],
    "CR_prompt_Nminus1_prompt_muon_dxy": [(0.01, "<")],
    "CR_prompt_Nminus1_prompt_muon_dz": [(0.01, "<")],
    "CR_prompt_Nminus1_qcd_muon_iso": [(0.1, ">")],
    "CR_prompt_Nminus1_qcd_muon_dxy": [(0.01, ">")],
    "CR_prompt_Nminus1_qcd_muon_dz": [(0.01, ">")],
    "CR_cb_Nminus1_muon_dxy": [(0.01, ">"), (0.2, "<")],
    "SR_low_temp_loose_Nminus1_muon_dxy": [(0.1, "<")],
    "SR_low_temp_tight_Nminus1_muon_dxy": [(0.007, "<")],
    "SR_low_temp_loose_Nminus1_muon_dz": [(0.1, "<")],
    "SR_low_temp_tight_Nminus1_muon_dz": [(0.007, "<")],
    "SR_low_temp_loose_Nminus1_muon_pt": [(45, "<")],
    "SR_low_temp_tight_Nminus1_muon_pt": [(35, "<")],
    "SR_low_temp_loose_Nminus1_sph1": [(0.2, ">")],
    "SR_low_temp_tight_Nminus1_sph1": [(0.7, ">")],
    "SR_low_temp_loose_Nminus1_dimuon_mass": [(45, "<")],
    "SR_low_temp_tight_Nminus1_dimuon_mass": [(35, "<")],
    "SR_high_temp_loose_Nminus1_muon_dxy": [(0.1, "<")],
    "SR_high_temp_tight_Nminus1_muon_dxy": [(0.007, "<")],
    "SR_high_temp_loose_Nminus1_muon_dz": [(0.1, "<")],
    "SR_high_temp_tight_Nminus1_muon_dz": [(0.007, "<")],
    "SR_high_temp_loose_Nminus1_muon_iso": [(5, "<")],
    "SR_high_temp_tight_Nminus1_muon_iso": [(0.65, "<")],
    "SR_high_temp_loose_Nminus1_muon_neutral_iso": [(3, "<")],
    "SR_high_temp_tight_Nminus1_muon_neutral_iso": [(0.5, "<")],
    "SR_high_temp_loose_Nminus1_dimuon_mass": [(70, "<")],
    "SR_high_temp_tight_Nminus1_dimuon_mass": [(70, "<")],
}

blinding_cuts = {
    "CR_prompt_Nminus1_dimuon_mass": slice(80j, None),
    "CR_prompt_Nminus1_prompt_muon_pt": slice(None),
    "CR_prompt_Nminus1_prompt_muon_iso": slice(None),
    "CR_prompt_Nminus1_prompt_muon_dxy": slice(None),
    "CR_prompt_Nminus1_prompt_muon_dz": slice(None),
    "CR_prompt_Nminus1_qcd_muon_iso": slice(None),
    "CR_prompt_Nminus1_qcd_muon_dxy": slice(None),
    "CR_prompt_Nminus1_qcd_muon_dz": slice(None),
    "CR_cb_Nminus1_muon_dxy": slice(None),
}

ylims = {
    "CR_prompt_Nminus1_dimuon_mass": (1e0, 1e8),
    "CR_prompt_Nminus1_prompt_muon_pt": (1e0, 1e8),
    "CR_prompt_Nminus1_prompt_muon_iso": (1e0, 1e8),
    "CR_prompt_Nminus1_prompt_muon_dxy": (1e0, 1e8),
    "CR_prompt_Nminus1_prompt_muon_dz": (1e0, 1e8),
    "CR_prompt_Nminus1_qcd_muon_iso": (1e0, 1e8),
    "CR_prompt_Nminus1_qcd_muon_dxy": (1e0, 1e8),
    "CR_prompt_Nminus1_qcd_muon_dz": (1e0, 1e8),
    "CR_cb_Nminus1_muon_dxy": (1e2, 1e12),
    "SR_low_temp_tight_Nminus1_muon_pt": (1, 1e8),
    "SR_low_temp_loose_Nminus1_muon_pt": (1e2, 1e10),
    "SR_low_temp_tight_Nminus1_muon_dxy": (10, 1e9),
    "SR_low_temp_loose_Nminus1_muon_dxy": (1e2, 1e10),
    "SR_low_temp_tight_Nminus1_muon_dz": (10, 1e9),
    "SR_low_temp_loose_Nminus1_muon_dz": (1e2, 1e10),
    "SR_low_temp_tight_Nminus1_sph1": (10, 1e9),
    "SR_low_temp_loose_Nminus1_sph1": (1e2, 1e10),
    "SR_low_temp_tight_Nminus1_dimuon_mass": (1, 1e8),
    "SR_low_temp_loose_Nminus1_dimuon_mass": (1e2, 1e10),
    "SR_high_temp_tight_Nminus1_muon_dxy": (10, 1e9),
    "SR_high_temp_loose_Nminus1_muon_dxy": (1e2, 1e10),
    "SR_high_temp_tight_Nminus1_muon_dz": (10, 1e9),
    "SR_high_temp_loose_Nminus1_muon_dz": (1e2, 1e10),
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
        "Nminus1_muon_iso": "muon isolation",
        "Nminus1_muon_dxy": r"muon $|d_{xy}|$ (cm)",
        "Nminus1_muon_dz": r"muon $|d_{z}|$ (cm)",
        "Nminus1_muon_neutral_iso": "muon neutral isolation",
        "Nminus1_sph1": r"$S_{1}$",
        "Nminus1_dimuon_mass": r"$m_{\mu\mu}$ (GeV)",
        "Nminus1_prompt_muon_pt": r"prompt muon $p_{T}$ (GeV)",
        "Nminus1_prompt_muon_iso": "prompt muon isolation",
        "Nminus1_prompt_muon_dxy": r"prompt muon $|d_{xy}|$ (cm)",
        "Nminus1_prompt_muon_dz": r"prompt muon $|d_{z}|$ (cm)",
        "Nminus1_qcd_muon_iso": "qcd muon isolation",
        "Nminus1_qcd_muon_dxy": r"qcd muon $|d_{xy}|$ (cm)",
        "Nminus1_qcd_muon_dz": r"qcd muon $|d_{z}|$ (cm)",
    }
    for key in xlabels:
        if key in plot:
            return xlabels[key]
    return plot


logx_plots = [
    "muon_dxy",
    "muon_dz",
    "muon_iso",
    "muon_neutral_iso",
]


def plot_ratio(hist_data, hist_bkg_total, ax, x_hatch, blinding_cut):
    ratio = np.divide(
        hist_data[blinding_cut].values(),
        hist_bkg_total[blinding_cut].values(),
        out=np.ones_like(hist_data[blinding_cut].values()),
        where=hist_bkg_total[blinding_cut].values() != 0,
    )
    ratio_err = np.where(
        hist_bkg_total[blinding_cut].values() > 0,
        np.sqrt(
            (hist_bkg_total[blinding_cut].values() ** -2)
            * (hist_data[blinding_cut].variances())
            + (
                hist_data[blinding_cut].values() ** 2
                * hist_bkg_total[blinding_cut].values() ** -4
            )
            * (hist_bkg_total[blinding_cut].variances())
        ),
        0,
    )
    ax.errorbar(
        hist_data[blinding_cut].axes.centers[0],
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


def make_plot(plots, plot, year, args):
    mc_processes = [
        ("Higgs", "Higgs"),
        ("TTV", r"$t\bar{t}+V$"),
        ("ST_NLO", r"single $t$"),
        ("WJets", r"$W+jets$"),
        ("VV+VVV", r"$VV+VVV$"),
        ("TT_powheg", r"$t\bar{t}$"),
        ("DY", "Drell-Yan"),
        ("QCD_Pt_MuEnrichedPt5", "QCD"),
    ]

    cm_energy = "13TeV"
    if year.startswith("202") or year == "Run3":
        cm_energy = "13p6TeV"
    signal_processes = [
        f"GluGluToSUEP_mS125.000_mPhi8.000_T8.000_modeleptonic_{cm_energy}",
        f"GluGluToSUEP_mS125.000_mPhi4.000_T16.000_modeleptonic_{cm_energy}",
        f"GluGluToSUEP_mS125.000_mPhi8.000_T16.000_modeleptonic_{cm_energy}",
        f"GluGluToSUEP_mS125.000_mPhi8.000_T32.000_modeleptonic_{cm_energy}",
    ]
    signal_labels = [
        r"$m_S=125\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=8\,$GeV, lep. decays",
        r"$m_S=125\,$GeV,$m_\phi=4\,$GeV," + "\n" + r"$T=16\,$GeV, lep. decays",
        r"$m_S=125\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=16\,$GeV, lep. decays",
        r"$m_S=125\,$GeV,$m_\phi=8\,$GeV," + "\n" + r"$T=32\,$GeV, lep. decays",
    ]
    if "low_temp" in plot:
        # signal_processes = [
        #     f"GluGluToSUEP_mS200.000_mPhi1.000_T0.250_modeleptonic_{cm_energy}",
        #     f"GluGluToSUEP_mS1000.000_mPhi1.000_T0.250_modeleptonic_{cm_energy}",
        #     f"GluGluToSUEP_mS300.000_mPhi1.400_T0.350_modehadronic_{cm_energy}",
        #     f"GluGluToSUEP_mS400.000_mPhi1.400_T0.350_modehadronic_{cm_energy}",
        #     f"GluGluToSUEP_mS500.000_mPhi1.400_T0.350_modehadronic_{cm_energy}",
        #     f"GluGluToSUEP_mS1000.000_mPhi1.400_T0.350_modehadronic_{cm_energy}",
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
            f"GluGluToSUEP_mS125.000_mPhi1.400_T1.400_modehadronic_{cm_energy}",
            f"GluGluToSUEP_mS125.000_mPhi4.000_T1.000_modehadronic_{cm_energy}",
            f"GluGluToSUEP_mS125.000_mPhi2.000_T2.000_modeleptonic_{cm_energy}",
            f"GluGluToSUEP_mS125.000_mPhi8.000_T4.000_modehadronic_{cm_energy}",
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
    hist_bkg_total = plots[f"QCD_Pt_MuEnrichedPt5_{year}"][plot][slc].copy().reset()

    for process, label in mc_processes:
        h_mc = plots[f"{process}_{year}"][plot][slc]
        hists_mc.append(h_mc)
        hist_bkg_total += h_mc.copy()

    hists_signal = []
    for process in signal_processes:
        h_signal = plots[f"{process}_{year}"][plot][slc]
        hists_signal.append(h_signal)

    fig, ax1 = plt.subplots(figsize=(12.5, 12))

    if args.ratio:
        fig = plt.figure(figsize=(12, 12.5))
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

    if args.data and plot in plots[f"Data_{year}"]:
        blinding_cut = slice(None)
        if plot in blinding_cuts:
            blinding_cut = blinding_cuts[plot]
        hep.histplot(
            plots[f"Data_{year}"][plot][slc][blinding_cut],
            label=["Data"],
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
    y_position_rel = 0.5
    y_position_abs = yrange_min * 10 ** (
        y_position_rel * np.log10(yrange_max / yrange_min)
    )
    plt.sca(ax1)
    for cut in cut_list:
        if cut[1] == "<":
            arrow_position = (
                cut[0] / arrow_length_abs if islogx else cut[0] - arrow_length_abs
            )
        else:
            arrow_position = (
                cut[0] * arrow_length_abs if islogx else cut[0] + arrow_length_abs
            )
        plt.vlines(
            x=cut[0],
            color="black",
            ymin=1e-3,
            ymax=2 * y_position_abs,
            lw=7,
        )
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
        y_position_rel + 0.07,
        region_labels[plot.split("_Nminus1")[0]],
        ha="center",
        weight="bold",
        fontsize=30,
        transform=ax1.transAxes,
    )

    if args.ratio and args.data:
        plot_ratio(
            plots[f"Data_{year}"][plot][slc], hist_bkg_total, ax2, x_hatch, blinding_cut
        )

    lumi_label = plot_utils.lumis[year] if args.lumi is None else args.lumi
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

    if args.ratio and args.data:
        plt.sca(ax2)
        plt.ylim(0.7, 1.3)
        plt.ylabel("Data/MC")
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_xlabel("", visible=False)
    plt.xlabel(get_xlabel(plot))
    if islogx:
        ax1.set_xscale("log")
        if args.ratio:
            ax2.set_xscale("log")
    if args.ratio and args.data:
        plt.sca(ax1)
    plt.yscale("log")
    plt.ylim(ylims[plot])
    plt.legend(ncol=3, loc="upper center", columnspacing=1)
    plt.ylabel("muons")
    if "sph1" in plot or "dimuon" in plot:
        plt.ylabel("events")
    plt.savefig(
        os.path.join(args.dest, args.tag, f"{plot}_{year}.pdf"),
        bbox_inches="tight",
    )
    plt.close()


if "__main__" == __name__:
    args = parse_args()

    if args.CRs and args.SRs:
        raise ValueError("Please choose either CRs or SRs, not both.")

    # Create destination directory
    os.makedirs(os.path.join(args.dest, args.tag), exist_ok=True)

    # Load plots and merge them
    years_to_load = args.year
    if "Run2" in args.year:
        # Commenting out 2016APV for now
        # years_to_load = ["2016APV", "2016", "2017", "2018"]
        years_to_load = ["2016", "2017", "2018"]
    if "Run3" in args.year:
        years_to_load = ["2022", "2022EE", "2023", "2023BPix"]
    if "Run2" in args.year and "Run3" in args.year:
        years_to_load = [
            # Commenting out 2016APV for now
            # "2016APV",
            "2016",
            "2017",
            "2018",
            "2022",
            "2022EE",
            "2023",
            "2023BPix",
        ]
    plots = {}
    for year in track(years_to_load, description="Loading plots"):
        if args.CRs:
            plots = plots | plot_utils.loader(
                tag=f"{args.tag}_{year}_CRs",
                era=year,
                custom_lumi=args.lumi,
                load_data=args.data,
            )
        if args.SRs:
            plots = plots | plot_utils.loader(
                tag=f"{args.tag}_{year}_SR_low_temp",
                era=year,
                custom_lumi=args.lumi,
                load_data=args.data,
            )
            plots = plots | plot_utils.loader(
                tag=f"{args.tag}_{year}_SR_high_temp",
                era=year,
                custom_lumi=args.lumi,
                load_data=args.data,
            )

    # Apply k-factor to QCD and DY
    if args.data and args.normalize:
        k_factor_qcd = {}
        k_factor_dy = {}
        for year in track(years_to_load, description="Calculating k-factors"):
            k_factor_qcd[year] = plot_utils.calculate_k_factor(
                plots,
                year,
                region="CR_cb_Nminus1_muon_dxy",
                process="QCD_Pt_MuEnrichedPt5",
            )
            for plot in plots[f"QCD_Pt_MuEnrichedPt5_{year}"]:
                plots[f"QCD_Pt_MuEnrichedPt5_{year}"][plot] = (
                    k_factor_qcd[year] * plots[f"QCD_Pt_MuEnrichedPt5_{year}"][plot]
                )
            k_factor_dy[year] = plot_utils.calculate_k_factor(
                plots, year, region="CR_prompt_Nminus1_prompt_muon_dxy", process="DY"
            )
            for plot in plots[f"DY_{year}"]:
                plots[f"DY_{year}"][plot] = (
                    k_factor_dy[year] * plots[f"DY_{year}"][plot]
                )
        print("QCD k_factors =", k_factor_qcd, flush=True)
        print("DY k_factors =", k_factor_dy, flush=True)

    # Load plots and merge them
    with Progress() as progress:
        task = progress.add_task(
            "Plotting regions...",
            total=len(args.year) * len(plots[f"QCD_Pt_MuEnrichedPt5_{args.year[0]}"]),
        )
        for year in args.year:
            for plot in plots[f"QCD_Pt_MuEnrichedPt5_{year}"].keys():
                make_plot(plots, plot, year, args)
                progress.update(task, advance=1)
