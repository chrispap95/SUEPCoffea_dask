import argparse
import os

import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import mplhep as hep
import numpy as np
import plot_utils
from rich.progress import track  # type: ignore[import]

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"


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
        default="/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/muon_branches/SUEPCoffea_dask/plotting/Nminus1_plots_sources",
        help="Destination directory to save the plots",
    )
    return parser.parse_args()


cuts = {
    # The tuples contains: (cut value, arrow location)
    "CR_prompt_Nminus1_cand_muon_pt": [(25, ">")],
    "CR_prompt_Nminus1_cand_muon_iso": [(0.1, "<")],
    "CR_prompt_Nminus1_cand_muon_ip3d": [(0.01, "<")],
    "CR_prompt_Nminus1_cand_muon_dxy": [(0.008, "<")],
    "CR_prompt_Nminus1_cand_muon_dz": [(0.01, "<")],
    "CR_prompt_Nminus1_muon_iso": [(0.1, ">")],
    "CR_prompt_Nminus1_muon_ip3d": [(0.015, ">")],
    "CR_prompt_Nminus1_muon_dxy": [(0.01, ">")],
    "CR_prompt_Nminus1_muon_dz": [(0.01, ">")],
    "CR_cb_Nminus1_muon_dxy": [(0.01, 2), (0.2, "<")],
    "SR_low_temp_loose_Nminus1_muon_pt": [(45, "<")],
    "SR_low_temp_tight_Nminus1_muon_pt": [(35, "<")],
    "SR_low_temp_loose_Nminus1_muon_ip3d": [(0.1, "<")],
    "SR_low_temp_tight_Nminus1_muon_ip3d": [(0.007, "<")],
    "SR_high_temp_loose_Nminus1_muon_ip3d": [(0.1, "<")],
    "SR_high_temp_tight_Nminus1_muon_ip3d": [(0.007, "<")],
    "SR_high_temp_loose_Nminus1_muon_iso": [(5, "<")],
    "SR_high_temp_tight_Nminus1_muon_iso": [(0.65, "<")],
    "SR_high_temp_loose_Nminus1_muon_neutral_iso": [(3, "<")],
    "SR_high_temp_tight_Nminus1_muon_neutral_iso": [(0.5, "<")],
}

ylims = {
    "CR_prompt_Nminus1_cand_muon_pt": (1e1, 1e8),
    "CR_prompt_Nminus1_cand_muon_iso": (1e1, 1e7),
    "CR_prompt_Nminus1_cand_muon_ip3d": (1e1, 1e8),
    "CR_prompt_Nminus1_cand_muon_dxy": (1e1, 1e7),
    "CR_prompt_Nminus1_cand_muon_dz": (1e1, 1e8),
    "CR_prompt_Nminus1_muon_iso": (1e1, 1e7),
    "CR_prompt_Nminus1_muon_ip3d": (1e1, 1e7),
    "CR_prompt_Nminus1_muon_dxy": (1e1, 1e7),
    "CR_prompt_Nminus1_muon_dz": (1e1, 1e7),
    "CR_cb_Nminus1_muon_dxy": (1e3, 1e9),
    "SR_low_temp_tight_Nminus1_muon_pt": (1e2, 1e10),
    "SR_low_temp_loose_Nminus1_muon_pt": (1e2, 1e10),
    "SR_low_temp_tight_Nminus1_muon_ip3d": (1e2, 1e10),
    "SR_low_temp_loose_Nminus1_muon_ip3d": (1e2, 1e10),
    "SR_high_temp_tight_Nminus1_muon_ip3d": (1e2, 1e10),
    "SR_high_temp_loose_Nminus1_muon_ip3d": (1e2, 1e10),
    "SR_high_temp_tight_Nminus1_muon_iso": (1e2, 1e10),
    "SR_high_temp_loose_Nminus1_muon_iso": (1e2, 1e10),
    "SR_high_temp_tight_Nminus1_muon_neutral_iso": (1e2, 1e9),
    "SR_high_temp_loose_Nminus1_muon_neutral_iso": (1e2, 1e9),
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
        "Nminus1_muon_dxy": r"muon $|d_{xy}|$ (cm)",
        "Nminus1_muon_dz": r"muon $|d_{z}|$ (cm)",
        "Nminus1_muon_ip3d": r"muon $IP_{3D}$ (cm)",
        "Nminus1_muon_iso": "muon isolation",
        "Nminus1_muon_neutral_iso": "muon neutral isolation",
        "Nminus1_cand_muon_pt": r"prompt muon $p_{T}$ (GeV)",
        "Nminus1_cand_muon_iso": "prompt muon isolation",
        "Nminus1_cand_muon_ip3d": r"prompt muon $IP_{3D}$ (cm)",
        "Nminus1_cand_muon_dxy": r"prompt muon $|d_{xy}|$ (cm)",
        "Nminus1_cand_muon_dz": r"prompt muon $|d_{z}|$ (cm)",
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
        "tau",
        "unmatched",
        "light",
        "prompt",
        "c",
        "b",
    ]

    hists_mc = []
    hist_bkg_total = plots["unmatched"][plot][::2j].copy().reset()

    for process in mc_processes:
        h_mc = plots[process][plot][::2j]
        hists_mc.append(h_mc)
        hist_bkg_total += h_mc.copy()

    fig, ax1 = plt.subplots(figsize=(10, 10))

    hep.histplot(
        hists_mc,
        yerr=[np.sqrt(h.variances()) for h in hists_mc],
        stack=True,
        label=mc_processes,
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
    arrow_length_rel = 0.08
    arrow_length_abs = arrow_length_rel * (xrange_max - xrange_min)
    if islogx:
        arrow_length_abs = 10 ** (arrow_length_rel * np.log10(xrange_max / xrange_min))
    yrange_min, yrange_max = ylims[plot]
    y_position_rel = 0.67
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
        plt.vlines(x=cut[0], color="black", ymin=1e-3, ymax=2.5 * y_position_abs, lw=7)
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
        0.5,
        0.75,
        region_labels[plot.split("_Nminus1")[0]],
        ha="center",
        weight="bold",
        transform=ax1.transAxes,
    )

    hep.cms.label(llabel="Preliminary", data=True, lumi=59.8, ax=ax1)

    plt.xlabel(get_xlabel(plot))
    if islogx:
        plt.xscale("log")
    plt.yscale("log")
    plt.ylim(ylims[plot])
    plt.legend(ncol=2, loc="upper center")
    plt.ylabel("muons")
    if "sph1" in plot or "dimuon" in plot:
        plt.ylabel("events")
    plt.tight_layout()
    plt.savefig(f"{args.dest}/{plot}_sources.pdf")
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

    # Initialize empty plots for sources
    print("Creating plots for sources...", end=" ", flush=True)
    plots_sources = {
        "unmatched": {},
        "prompt": {},
        "light": {},
        "c": {},
        "b": {},
        "tau": {},
    }
    for source in plots_sources:
        for plot in plots["QCD_Pt_MuEnrichedPt5_2018"].keys():
            if "dimuon" in plot or "sph1" in plot:
                continue
            plots_sources[source][plot] = (
                plots["QCD_Pt_MuEnrichedPt5_2018"][plot][:, ::sum].copy().reset()
            )

    for dataset in plots:
        for plot in plots[dataset]:
            if plot not in plots_sources["unmatched"]:
                continue
            plots_sources["unmatched"][plot] += plots[dataset][plot][:, 0].copy()
            plots_sources["prompt"][plot] += plots[dataset][plot][:, 1j].copy()
            plots_sources["light"][plot] += plots[dataset][plot][:, 3j].copy()
            plots_sources["c"][plot] += plots[dataset][plot][:, 4j].copy()
            plots_sources["b"][plot] += plots[dataset][plot][:, 5j].copy()
            plots_sources["tau"][plot] += plots[dataset][plot][:, 15j].copy()
    print("Done!", flush=True)

    # Load plots and merge them
    for plot in track(plots_sources["unmatched"].keys()):
        make_plot(plots_sources, plot)
