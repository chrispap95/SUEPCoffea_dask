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
        default="muon_kinematics_Oct2025",
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
        default=str(pathlib.Path(__file__).parent / "muon_kinematics_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'muon_kinematics_plots'}",
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
    parser.add_argument(
        "--VRs",
        action="store_true",
        help="Process only VRs. Default is False.",
    )
    return parser.parse_args()


ylims = {
    "CR_cb": (1e2, 1e10),
    "CR_prompt": (1e1, 1e9),
    "VR_loose": (1e1, 1e9),
    "VR_tight": (1e1, 1e9),
    "SR_high_temp_loose": (1e2, 1e10),
    "SR_high_temp_tight": (1e1, 1e9),
    "SR_low_temp_loose": (1e2, 1e10),
    "SR_low_temp_tight": (1e0, 1e8),
}

region_labels = {
    "CR_cb": r"$CR_{QCD}$",
    "CR_prompt": r"$CR_{DY}$",
    "VR_loose": r"$VR^{loose}$",
    "VR_tight": r"$VR^{tight}$",
    "SR_high_temp_loose": r"$SR^{loose}_{high~T}$",
    "SR_high_temp_tight": r"$SR^{tight}_{high~T}$",
    "SR_low_temp_loose": r"$SR^{loose}_{low~T}$",
    "SR_low_temp_tight": r"$SR^{tight}_{low~T}$",
}


def get_xlabel(plot):
    xlabels = {
        "muon_pt": r"muon $p_{T}$ (GeV)",
        "muon_eta": r"muon $\eta$",
        "muon_phi": r"muon $\phi$",
        "muon_iso": "muon isolation",
        "muon_dxy": r"muon $|d_{xy}|$ (cm)",
        "muon_dz": r"muon $|d_{z}|$ (cm)",
        "dimuon_dr": r"$\Delta R_{\mu\mu}$ (GeV)",
        "dimuon_mass": r"$m_{\mu\mu}$ (GeV)",
    }
    for key in xlabels:
        if key in plot:
            return xlabels[key]
    return plot


logx_plots = [
    "muon_pt",
    "muon_dxy",
    "muon_dz",
    "muon_iso",
    "dimuon_dr",
    "dimuon_mass",
]


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
    hist_bkg_total = (
        plots[f"QCD_Pt_MuEnrichedPt5_{year}"][plot][:, ::sum].copy().reset()
    )

    for process, label in mc_processes:
        h_mc = plots[f"{process}_{year}"][plot][:, ::sum]
        hists_mc.append(h_mc)
        hist_bkg_total += h_mc.copy()

    hists_signal = []
    for process in signal_processes:
        h_signal = plots[f"{process}_{year}"][plot][:, ::sum]
        hists_signal.append(h_signal)

    fig, ax1 = plt.subplots(figsize=(13, 12))

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
        hep.histplot(
            plots[f"Data_{year}"][plot][:, ::sum],
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

    plt.sca(ax1)
    region_name = (
        plot.split("_muon")[0] if "_muon" in plot else plot.split("_dimuon")[0]
    )
    label_ypos = 0.65 if "SR" in region_name else 0.6
    plt.text(
        0.3,
        label_ypos,
        region_labels[region_name],
        ha="center",
        weight="bold",
        fontsize=30,
        transform=ax1.transAxes,
    )

    if args.ratio and args.data:
        plot_ratio(
            plots[f"Data_{year}"][plot][:, ::sum],
            hist_bkg_total,
            ax2,
            x_hatch,
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
        plt.xlabel(r"$n_{muon}$")
        plt.ylim(0.5, 1.5)
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
    plt.ylim(ylims[region_name])
    plt.legend(ncol=3, loc="upper center", columnspacing=1)
    plt.ylabel("muons")
    if "dimuon" in plot:
        plt.ylabel("dimuon pairs")
    plt.savefig(
        os.path.join(args.dest, args.tag, f"{plot}_{year}.pdf"),
        bbox_inches="tight",
    )
    plt.close()


if "__main__" == __name__:
    args = parse_args()

    if args.CRs and args.SRs:
        raise ValueError("Please choose either CRs or SRs, not both.")
    if args.CRs and args.VRs:
        raise ValueError("Please choose either CRs or VRs, not both.")
    if args.SRs and args.VRs:
        raise ValueError("Please choose either SRs or VRs, not both.")

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
        plots = plot_utils.loader(
            tag=f"{args.tag}_{year}_CR",
            era=year,
            custom_lumi=args.lumi,
            load_data=args.data,
        )
        if args.SRs:
            plots_SR_high_temp = plot_utils.loader(
                tag=f"{args.tag}_{year}_SR_low_temp",
                era=year,
                custom_lumi=args.lumi,
                load_data=args.data,
            )
            plots_SR_low_temp = plot_utils.loader(
                tag=f"{args.tag}_{year}_SR_high_temp",
                era=year,
                custom_lumi=args.lumi,
                load_data=args.data,
            )
            for process in plots:
                plots[process] = (
                    plots[process]
                    | plots_SR_low_temp[process]
                    | plots_SR_high_temp[process]
                )
        if args.VRs:
            plots_VR = plot_utils.loader(
                tag=f"{args.tag}_{year}_VR",
                era=year,
                custom_lumi=args.lumi,
                load_data=args.data,
            )
            for process in plots:
                plots[process] = plots[process] | plots_VR[process]

    # Apply k-factor to QCD and DY
    if args.data and args.normalize:
        k_factor_qcd = {}
        k_factor_dy = {}
        for year in track(years_to_load, description="Calculating k-factors"):
            k_factor_qcd[year] = plot_utils.calculate_k_factor(
                plots, year, region="CR_cb_muon_pt", process="QCD_Pt_MuEnrichedPt5"
            )
            for plot in plots[f"QCD_Pt_MuEnrichedPt5_{year}"]:
                plots[f"QCD_Pt_MuEnrichedPt5_{year}"][plot] = (
                    k_factor_qcd[year] * plots[f"QCD_Pt_MuEnrichedPt5_{year}"][plot]
                )
            k_factor_dy[year] = plot_utils.calculate_k_factor(
                plots, year, region="CR_prompt_muon_dxy", process="DY"
            )
            for plot in plots[f"DY_{year}"]:
                plots[f"DY_{year}"][plot] = (
                    k_factor_dy[year] * plots[f"DY_{year}"][plot]
                )
        print("QCD k_factors =", k_factor_qcd, flush=True)
        print("DY k_factors =", k_factor_dy, flush=True)

    # Load plots and merge them
    with Progress() as progress:
        n_subtract = (
            len([p for p in plots[f"QCD_Pt_MuEnrichedPt5_{args.year[0]}"] if "CR" in p])
            if not args.CRs
            else 0
        )
        task = progress.add_task(
            "Plotting regions",
            total=len(args.year)
            * (len(plots[f"QCD_Pt_MuEnrichedPt5_{args.year[0]}"]) - n_subtract),
        )
        for year in args.year:
            for plot in plots[f"QCD_Pt_MuEnrichedPt5_{year}"].keys():
                if not args.CRs and "CR" in plot:
                    continue
                if not args.SRs and "SR" in plot:
                    continue
                if not args.VRs and "VR" in plot:
                    continue
                make_plot(plots, plot, year, args)
                progress.update(task, advance=1)
