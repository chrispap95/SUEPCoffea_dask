import argparse
import os
import pathlib

import cms_styles
import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import mplhep as hep
import numpy as np
import plot_utils
from rich.progress import track

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

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
        default="muon_kinematics_Apr2026",
        help="Tag for the run with IP scale factors applied.",
    )
    parser.add_argument(
        "--tag-noIPcorr",
        type=str,
        default="muon_kinematics_noIPcorr_Apr2026",
        help="Tag for the run without IP scale factors applied.",
    )
    parser.add_argument(
        "--year",
        type=str,
        nargs="*",
        default=["2018"],
        help="Year(s) of the data. Default is 2018.",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        help="Custom integrated luminosity in pb^-1.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "IP_comparison_kinematics_plots"),
        help="Destination directory to save the plots.",
    )
    parser.add_argument(
        "--CRs",
        action="store_true",
        help="Process only CRs.",
    )
    parser.add_argument(
        "--SRs",
        action="store_true",
        help="Process only SRs.",
    )
    parser.add_argument(
        "--VRs",
        action="store_true",
        help="Process only VRs.",
    )
    return parser.parse_args()


mc_processes = [
    "Higgs",
    "TTV",
    "ST_NLO",
    "WJets",
    "VV+VVV",
    "TT_powheg",
    "DY",
    "QCD_Pt_MuEnrichedPt5",
]

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

logx_plots = [
    "muon_pt",
    "muon_dxy",
    "muon_dz",
    "muon_iso",
    "dimuon_dr",
    "dimuon_mass",
]


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


def sum_mc(plots, plot, year):
    """Sum all MC processes into a single total histogram."""
    hist_total = plots[f"QCD_Pt_MuEnrichedPt5_{year}"][plot][:, ::sum].copy().reset()
    for process in mc_processes:
        hist_total += plots[f"{process}_{year}"][plot][:, ::sum].copy()
    return hist_total


def make_comparison_plot(plots_ipcorr, plots_noipcorr, plot, year, args):
    region_name = (
        plot.split("_muon")[0] if "_muon" in plot else plot.split("_dimuon")[0]
    )

    h_with = sum_mc(plots_ipcorr, plot, year)
    h_without = sum_mc(plots_noipcorr, plot, year)

    fig = plt.figure(figsize=(12, 12.5))
    plt.subplots_adjust(bottom=0.08, top=0.92, left=0.1, right=0.95)
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

    # Uncertainty hatches helper
    def make_hatch_arrays(h):
        x_hatch = np.vstack((h.axes[0].edges[:-1], h.axes[0].edges[1:])).reshape(
            (-1,), order="F"
        )
        y_hatch = np.vstack((h.values(), h.values())).reshape((-1,), order="F")
        y_hatch_unc = np.vstack(
            (np.sqrt(h.variances()), np.sqrt(h.variances()))
        ).reshape((-1,), order="F")
        return x_hatch, y_hatch, y_hatch_unc

    # Main panel: both distributions
    hep.histplot(
        h_with,
        yerr=np.sqrt(h_with.variances()),
        label="with IP SF",
        color="C0",
        lw=2,
        ax=ax1,
    )
    x_hatch, y_hatch, y_hatch_unc = make_hatch_arrays(h_with)
    ax1.fill_between(
        x=x_hatch,
        y1=y_hatch - y_hatch_unc,
        y2=y_hatch + y_hatch_unc,
        step="pre",
        facecolor="none",
        edgecolor="C0",
        alpha=0.5,
        linewidth=0,
        hatch="///",
    )

    hep.histplot(
        h_without,
        yerr=np.sqrt(h_without.variances()),
        label="without IP SF",
        color="C1",
        lw=2,
        ls="--",
        ax=ax1,
    )
    x_hatch2, y_hatch2, y_hatch2_unc = make_hatch_arrays(h_without)
    ax1.fill_between(
        x=x_hatch2,
        y1=y_hatch2 - y_hatch2_unc,
        y2=y_hatch2 + y_hatch2_unc,
        step="pre",
        facecolor="none",
        edgecolor="C1",
        alpha=0.5,
        linewidth=0,
        hatch="\\\\\\",
    )

    # Region label
    label_ypos = 0.65 if "SR" in region_name else 0.6
    ax1.text(
        0.3,
        label_ypos,
        region_labels[region_name],
        ha="center",
        weight="bold",
        fontsize=30,
        transform=ax1.transAxes,
    )

    lumi_label = plot_utils.lumis[year] if args.lumi is None else args.lumi
    lumi_label = lumi_label / 1000
    lumi_label = round(lumi_label, 2) if lumi_label < 1 else round(lumi_label, 1)
    hep.cms.label(
        llabel="Preliminary",
        data=True,
        year=year,
        lumi=lumi_label,
        com=13.6 if year.startswith("202") or year == "Run3" else 13,
        ax=ax1,
    )

    ax1.legend(ncol=2, loc="upper center")
    ax1.set_yscale("log")
    ax1.set_ylim(ylims[region_name])
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlabel("", visible=False)
    plt.ylabel("muons" if "dimuon" not in plot else "dimuon pairs")

    # Ratio panel: (with IP SF) / (without IP SF)
    plt.sca(ax2)
    ratio = np.divide(
        h_with.values(),
        h_without.values(),
        out=np.ones_like(h_with.values()),
        where=h_without.values() != 0,
    )
    ratio_err = np.where(
        h_without.values() > 0,
        np.sqrt(
            (h_without.values() ** -2) * h_with.variances()
            + (h_with.values() ** 2 * h_without.values() ** -4) * h_without.variances()
        ),
        0,
    )
    ax2.errorbar(
        h_with.axes.centers[0],
        ratio,
        yerr=ratio_err,
        color="black",
        fmt="o",
        linestyle="none",
        markersize=7,
        lw=2,
    )
    ax2.axhline(1, ls="--", color="gray")
    ax2.set_ylim(0.8, 1.2)
    ax2.set_ylabel("with / without")

    islogx = any(key in plot for key in logx_plots)
    plt.xlabel(get_xlabel(plot))
    if islogx:
        ax1.set_xscale("log")
        ax2.set_xscale("log")

    plt.savefig(
        os.path.join(args.dest, args.tag, f"{plot}_{year}.pdf"),
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    args = parse_args()

    if args.CRs and args.SRs:
        raise ValueError("Please choose either CRs or SRs, not both.")
    if args.CRs and args.VRs:
        raise ValueError("Please choose either CRs or VRs, not both.")
    if args.SRs and args.VRs:
        raise ValueError("Please choose either SRs or VRs, not both.")

    out_dir = os.path.join(args.dest, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    years_to_load = args.year
    if "Run2" in args.year:
        years_to_load = ["2016", "2017", "2018"]
    if "Run3" in args.year:
        years_to_load = ["2022", "2022EE", "2023", "2023BPix"]
    if "Run2" in args.year and "Run3" in args.year:
        years_to_load = ["2016", "2017", "2018", "2022", "2022EE", "2023", "2023BPix"]

    tag_noipcorr = args.tag_noIPcorr
    load_all = not args.CRs and not args.SRs and not args.VRs

    for year in track(years_to_load, description="Processing years"):
        plots_ipcorr = plot_utils.loader(
            tag=f"{args.tag}_{year}_CR",
            era=year,
            custom_lumi=args.lumi,
        )
        plots_noipcorr = plot_utils.loader(
            tag=f"{tag_noipcorr}_{year}_CR",
            era=year,
            custom_lumi=args.lumi,
        )

        if args.SRs or load_all:
            for region_suffix in ["SR_low_temp", "SR_high_temp"]:
                plots_ipcorr_sr = plot_utils.loader(
                    tag=f"{args.tag}_{year}_{region_suffix}",
                    era=year,
                    custom_lumi=args.lumi,
                )
                plots_noipcorr_sr = plot_utils.loader(
                    tag=f"{tag_noipcorr}_{year}_{region_suffix}",
                    era=year,
                    custom_lumi=args.lumi,
                )
                for process in plots_ipcorr:
                    if process in plots_ipcorr_sr:
                        plots_ipcorr[process] = (
                            plots_ipcorr[process] | plots_ipcorr_sr[process]
                        )
                for process in plots_noipcorr:
                    if process in plots_noipcorr_sr:
                        plots_noipcorr[process] = (
                            plots_noipcorr[process] | plots_noipcorr_sr[process]
                        )

        if args.VRs or load_all:
            plots_ipcorr_vr = plot_utils.loader(
                tag=f"{args.tag}_{year}_VR",
                era=year,
                custom_lumi=args.lumi,
            )
            plots_noipcorr_vr = plot_utils.loader(
                tag=f"{tag_noipcorr}_{year}_VR",
                era=year,
                custom_lumi=args.lumi,
            )
            for process in plots_ipcorr:
                if process in plots_ipcorr_vr:
                    plots_ipcorr[process] = (
                        plots_ipcorr[process] | plots_ipcorr_vr[process]
                    )
            for process in plots_noipcorr:
                if process in plots_noipcorr_vr:
                    plots_noipcorr[process] = (
                        plots_noipcorr[process] | plots_noipcorr_vr[process]
                    )

        ref_process = f"QCD_Pt_MuEnrichedPt5_{year}"
        for plot in plots_ipcorr[ref_process]:
            if args.CRs and "CR" not in plot:
                continue
            if args.SRs and "SR" not in plot:
                continue
            if args.VRs and "VR" not in plot:
                continue
            if plot not in plots_noipcorr.get(ref_process, {}):
                continue
            make_comparison_plot(plots_ipcorr, plots_noipcorr, plot, year, args)
