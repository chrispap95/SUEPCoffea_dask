import argparse
import os
import pathlib

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
        default="full_analysis_Apr2026",
        help="Tag for the run with IP scale factors applied.",
    )
    parser.add_argument(
        "--tag-noIPcorr",
        type=str,
        default="full_analysis_noIPcorr_Apr2026",
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
        default=str(pathlib.Path(__file__).parent / "IP_comparison_plots"),
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
    ("Higgs", "Higgs"),
    ("TTV", r"$t\bar{t}+V$"),
    ("ST_NLO", r"single $t$"),
    ("WJets", r"$W+jets$"),
    ("VV+VVV", r"$VV+VVV$"),
    ("TT_powheg", r"$t\bar{t}$"),
    ("DY", "DY"),
    ("QCD_Pt_MuEnrichedPt5", "QCD"),
]

y_ranges = {
    "CR_cb": (100, 1e12),
    "CR_prompt": (1, 1e9),
    "CR_prompt_prompt": (1, 1e9),
    "CR_prompt_qcd": (1, 1e9),
    "VR_loose": (1, 1e9),
    "VR_tight": (1, 1e9),
    "SR_high_temp_loose": (1e-2, 1e13),
    "SR_high_temp_tight": (1e-2, 1e13),
    "SR_low_temp_loose": (1e-2, 1e13),
    "SR_low_temp_tight": (1e-2, 1e13),
}

region_labels = {
    "CR_cb": r"$CR_{QCD}$",
    "CR_prompt": r"$CR_{DY}$",
    "CR_prompt_prompt": r"$CR_{DY}$ prompt $\mu$",
    "CR_prompt_qcd": r"$CR_{DY}$ QCD $\mu$",
    "VR_loose": r"$VR^{loose}$",
    "VR_tight": r"$VR^{tight}$",
    "SR_high_temp_loose": r"$SR^{loose}_{high~T}$",
    "SR_high_temp_tight": r"$SR^{tight}_{high~T}$",
    "SR_low_temp_loose": r"$SR^{loose}_{low~T}$",
    "SR_low_temp_tight": r"$SR^{tight}_{low~T}$",
}

regions = [
    "CR_prompt",
    "CR_cb",
    "SR_low_temp_loose",
    "SR_low_temp_tight",
    "SR_high_temp_loose",
    "SR_high_temp_tight",
    "VR_loose",
    "VR_tight",
]


def get_mc_hists(plots, region, year):
    """Get individual MC histograms, their labels, and the summed total."""
    hists = []
    labels = []
    hist_total = None
    for process_name, label in mc_processes:
        process_key = f"{process_name}_{year}"
        if process_key not in plots or region not in plots[process_key]:
            continue
        h = plots[process_key][region].copy()
        hists.append(h)
        labels.append(label)
        if hist_total is None:
            hist_total = h.copy().reset()
        hist_total += h
    if hist_total is None:
        raise KeyError(f"No MC histogram found for {region}_{year}")
    return hists, labels, hist_total


def set_nmuon_ticks(ax, region):
    if region == "CR_cb":
        ax.set_xticks([0.75, 1, 2, 3, 4, 5.25])
        ax.set_xticklabels(["", "1", "2", "3", "4+", ""])
    elif region == "CR_prompt":
        ax.set_xticks([1.75, 2, 3, 4, 5, 6.25])
        ax.set_xticklabels(["", "2", "3", "4", "5+", ""])
    elif region in {"CR_prompt_prompt", "CR_prompt_qcd"}:
        ax.set_xticks([-0.25, 0, 1, 2, 3, 4, 5.25])
        ax.set_xticklabels(["", "0", "1", "2", "3", "4+", ""])
    elif "SR" in region or "VR" in region:
        ax.set_xticks([2.75, 3, 4, 5, 6, 7, 8.25])
        ax.set_xticklabels(["", "3", "4", "5", "6", "7+", ""])


def get_mc_total(plots, region, year):
    """Return only the summed total MC histogram for a region, or None if unavailable."""
    try:
        _, _, hist_total = get_mc_hists(plots, region, year)
    except KeyError:
        return None
    return hist_total


def make_comparison_plot(plots_ipcorr, plots_noipcorr, region, year, args):
    hists_with, labels_with, h_total_with = get_mc_hists(plots_ipcorr, region, year)
    _, _, h_total_without = get_mc_hists(plots_noipcorr, region, year)

    h_total_with_up = get_mc_total(plots_ipcorr, f"{region}_MuonSFUp", year)
    h_total_with_down = get_mc_total(plots_ipcorr, f"{region}_MuonSFDown", year)

    fig = plt.figure(figsize=(12, 12.5))
    plt.subplots_adjust(bottom=0.08, top=0.92, left=0.1, right=0.95)
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)

    # Stacked MC for "with IP SF"
    hep.histplot(
        hists_with,
        yerr=[np.sqrt(h.variances()) for h in hists_with],
        stack=True,
        label=labels_with,
        histtype="fill",
        ec="black",
        lw=2,
        ax=ax1,
    )

    # MC stat uncertainty hatch on the total "with IP SF"
    x_hatch = np.vstack(
        (h_total_with.axes[0].edges[:-1], h_total_with.axes[0].edges[1:])
    ).reshape((-1,), order="F")
    y_hatch = np.vstack((h_total_with.values(), h_total_with.values())).reshape(
        (-1,), order="F"
    )
    y_hatch_unc = np.vstack(
        (np.sqrt(h_total_with.variances()), np.sqrt(h_total_with.variances()))
    ).reshape((-1,), order="F")
    ax1.fill_between(
        x=x_hatch,
        y1=y_hatch - y_hatch_unc,
        y2=y_hatch + y_hatch_unc,
        label="MC Stat. Unc.",
        step="pre",
        facecolor="none",
        edgecolor=(0, 0, 0, 0.5),
        linewidth=0,
        hatch="///",
        zorder=2,
    )

    # Total MC without IP SF overlaid as a dashed line
    hep.histplot(
        h_total_without,
        yerr=np.sqrt(h_total_without.variances()),
        label="without IP SF",
        color="red",
        lw=2,
        ls="--",
        ax=ax1,
    )

    if "SR" in region:
        ax1.vlines(x=7, color="red", ymin=1e-3, ymax=1e6, lw=7)
        ax1.annotate(
            "",
            xy=(7.5, 1e5),
            xytext=(7, 1e5),
            arrowprops=dict(
                arrowstyle="simple",
                facecolor="red",
                edgecolor="black",
                linewidth=2,
                mutation_scale=40,
            ),
        )

    label_coords = (0.51, 0.58)
    if "CR" in region:
        label_coords = (0.4, 0.59)
    ax1.text(
        *label_coords,
        region_labels[region],
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

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(
        handles,
        labels,
        loc="upper left",
        ncol=2,
        columnspacing=1.1,
        frameon=False,
    )

    ax1.set_yscale("log")
    ax1.set_ylim(y_ranges[region])
    set_nmuon_ticks(ax1, region)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlabel("", visible=False)
    ax1.set_ylabel("events")
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.xaxis.set_minor_locator(ticker.NullLocator())

    # Ratio panel: (with IP SF) / (without IP SF)
    plt.sca(ax2)
    denom = h_total_without.values()
    ratio = np.divide(
        h_total_with.values(),
        denom,
        out=np.ones_like(h_total_with.values()),
        where=denom != 0,
    )
    if h_total_with_up is not None and h_total_with_down is not None:
        ratio_up = np.divide(
            h_total_with_up.values(),
            denom,
            out=np.ones_like(h_total_with_up.values()),
            where=denom != 0,
        )
        ratio_down = np.divide(
            h_total_with_down.values(),
            denom,
            out=np.ones_like(h_total_with_down.values()),
            where=denom != 0,
        )
        ratio_err = [
            np.clip(ratio - ratio_down, 0, None),
            np.clip(ratio_up - ratio, 0, None),
        ]
    else:
        ratio_err = np.zeros_like(ratio)
        nonzero = denom > 0
        ratio_err[nonzero] = np.sqrt(
            h_total_with.variances()[nonzero] / denom[nonzero] ** 2
            + h_total_with.values()[nonzero] ** 2
            * h_total_without.variances()[nonzero]
            / denom[nonzero] ** 4
        )
    ax2.errorbar(
        h_total_with.axes.centers[0],
        ratio,
        yerr=ratio_err,
        color="black",
        fmt="o",
        linestyle="none",
        markersize=7,
        lw=2,
    )
    ax2.axhline(1, ls="--", color="gray")
    ax2.set_ylim(0.5, 1.5)
    ax2.set_ylabel("with / without")
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.xaxis.set_minor_locator(ticker.NullLocator())
    set_nmuon_ticks(ax2, region)
    plt.xlabel(r"$n_{muon}$")

    plt.savefig(
        os.path.join(args.dest, args.tag, f"{region}_{year}.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def merge_plot_dicts(plots, new_plots):
    for process, process_plots in new_plots.items():
        if process not in plots:
            plots[process] = process_plots
            continue
        plots[process] = plots[process] | process_plots
    return plots


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
        years_to_load = [
            "2016",
            "2017",
            "2018",
            "2022",
            "2022EE",
            "2023",
            "2023BPix",
        ]

    tag_noipcorr = args.tag_noIPcorr
    load_all = not args.CRs and not args.SRs and not args.VRs

    plots_ipcorr = {}
    plots_noipcorr = {}
    for year in track(years_to_load, description="Loading plots"):
        if args.CRs or load_all:
            plots_ipcorr = merge_plot_dicts(
                plots_ipcorr,
                plot_utils.loader(
                    tag=f"{args.tag}_{year}_CR",
                    era=year,
                    custom_lumi=args.lumi,
                ),
            )
            plots_noipcorr = merge_plot_dicts(
                plots_noipcorr,
                plot_utils.loader(
                    tag=f"{tag_noipcorr}_{year}_CR",
                    era=year,
                    custom_lumi=args.lumi,
                ),
            )

        if args.SRs or load_all:
            plots_ipcorr = merge_plot_dicts(
                plots_ipcorr,
                plot_utils.loader(
                    tag=f"{args.tag}_{year}_SRs",
                    era=year,
                    custom_lumi=args.lumi,
                ),
            )
            plots_noipcorr = merge_plot_dicts(
                plots_noipcorr,
                plot_utils.loader(
                    tag=f"{tag_noipcorr}_{year}_SRs",
                    era=year,
                    custom_lumi=args.lumi,
                ),
            )

        if args.VRs or load_all:
            plots_ipcorr = merge_plot_dicts(
                plots_ipcorr,
                plot_utils.loader(
                    tag=f"{args.tag}_{year}_VR",
                    era=year,
                    custom_lumi=args.lumi,
                ),
            )
            plots_noipcorr = merge_plot_dicts(
                plots_noipcorr,
                plot_utils.loader(
                    tag=f"{tag_noipcorr}_{year}_VR",
                    era=year,
                    custom_lumi=args.lumi,
                ),
            )

    if "Run2" in args.year:
        plots_ipcorr = plots_ipcorr | plot_utils.merge_runs(
            plots_ipcorr, "Run2", data=False
        )
        plots_noipcorr = plots_noipcorr | plot_utils.merge_runs(
            plots_noipcorr, "Run2", data=False
        )
    if "Run3" in args.year:
        plots_ipcorr = plots_ipcorr | plot_utils.merge_runs(
            plots_ipcorr, "Run3", data=False
        )
        plots_noipcorr = plots_noipcorr | plot_utils.merge_runs(
            plots_noipcorr, "Run3", data=False
        )

    regions_to_plot = list(regions)
    if args.CRs:
        regions_to_plot += ["CR_prompt_prompt", "CR_prompt_qcd"]

    for year in track(args.year, description="Plotting regions"):
        ref_process = f"QCD_Pt_MuEnrichedPt5_{year}"
        if ref_process not in plots_ipcorr or ref_process not in plots_noipcorr:
            continue
        for region in regions_to_plot:
            if args.CRs and "CR" not in region:
                continue
            if args.SRs and "SR" not in region:
                continue
            if args.VRs and "VR" not in region:
                continue
            if region not in plots_ipcorr[ref_process]:
                continue
            if region not in plots_noipcorr[ref_process]:
                continue
            make_comparison_plot(plots_ipcorr, plots_noipcorr, region, year, args)
