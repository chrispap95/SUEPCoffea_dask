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
import scipy.stats as stats  # type: ignore[import]
from rich.pretty import pprint  # type: ignore[import]
from rich.progress import track  # type: ignore[import]

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

np.seterr(divide="ignore", invalid="ignore")

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)

# Set to 10 color cycler
plt.style.use(cms_styles.CMS_petroff_10)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="VR_JPsi_Jun2026",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--year",
        type=str,
        nargs="*",
        default=["2018"],
        help="Year of the data. Default is all years. Can be a single year or multiple years.",
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
        "--nMuon",
        type=int,
        default=None,
        help="Number of muons to select. Default is None, which means all events are selected.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "vr_plots_JPsi"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'vr_plots_JPsi'}.",
    )
    return parser.parse_args()


ylims = {
    "VR_loose": (1e0, 1e9),
    "VR_tight": (1e0, 1e9),
}

region_labels = {
    "VR_loose": r"$VR_{loose}$",
    "VR_tight": r"$VR_{tight}$",
}

# Plot regions
regions = [
    "VR_loose",
    "VR_tight",
]
variables = [
    "muon_pt",
    "dimuon_mass",
]


def get_xlabel(plot):
    xlabels = {
        "pt": r"muon $p_{T}$ (GeV)",
        "mass": r"$m_{\mu\mu}$ (GeV)",
    }
    for key in xlabels:
        if key in plot:
            return xlabels[key]
    return plot


logx_plots = [
    "pt",
    "mass",
]


def get_poisson_errors(N, alpha=0.6827):
    # Return the Garwood confidence interval for a Poisson distribution
    upper = stats.gamma.ppf((1 + alpha) / 2, N + 1) - N
    lower = N - stats.gamma.ppf((1 - alpha) / 2, N)
    return np.nan_to_num(lower), np.nan_to_num(upper)


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


def plot_pull(hist_data, hist_bkg_total, ax, x_hatch):
    # Get poisson errors for the data
    data_unc = get_poisson_errors(hist_data.values())
    # Select correct side for poisson errors
    data_unc = np.where(
        hist_data.values() > hist_bkg_total.values(), data_unc[0], data_unc[1]
    )

    pulls = (hist_data.values() - hist_bkg_total.values()) / np.sqrt(
        hist_bkg_total.variances() + data_unc**2
    )
    pulls_up = np.where(pulls >= 0, pulls, 0)
    pulls_down = np.where(pulls < 0, pulls, 0)

    y_hatch_up = np.vstack((pulls_up, pulls_up)).reshape((-1,), order="F")
    y_hatch_down = np.vstack((pulls_down, pulls_down)).reshape((-1,), order="F")

    ax.fill_between(
        x=x_hatch,
        y1=0,
        y2=y_hatch_up,  # type: ignore[arg-type]
        step="pre",
        facecolor="None",
        edgecolor="red",
        alpha=1,
        linewidth=0,
        hatch="///",
    )
    ax.fill_between(
        x=x_hatch,
        y1=0,
        y2=y_hatch_down,  # type: ignore[arg-type]
        step="pre",
        facecolor="None",
        edgecolor="blue",
        alpha=1,
        linewidth=0,
        hatch="///",
    )
    ax.axhline(0, ls="--", color="gray")
    ax.set_xlabel("nMuon")
    ax.set_ylabel("pull")
    ax.set_ylim(-2.5, 2.5)


def make_plot(args, plots, year, region, variable):
    sel_sum = slice(None, None, sum)
    if args.nMuon is not None:
        sel_sum = slice(args.nMuon * 1j, (args.nMuon + 1) * 1j, sum)

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

    hists_mc = []
    hist_bkg_total = (
        plots[f"QCD_Pt_MuEnrichedPt5_{year}"][f"{region}_{variable}"][:, sel_sum]
        .copy()
        .reset()
    )

    for process, label in mc_processes:
        h_mc = plots[f"{process}_{year}"][f"{region}_{variable}"][:, sel_sum]
        hists_mc.append(h_mc)
        hist_bkg_total += h_mc.copy()

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

    if args.data and f"{region}_{variable}" in plots[f"Data_{year}"]:
        hep.histplot(
            plots[f"Data_{year}"][f"{region}_{variable}"][:, sel_sum],
            label=["Data"],
            histtype="errorbar",
            mec="black",
            mfc="black",
            ecolor="black",
            markersize=15,
            lw=3,
            ax=ax1,
        )

    # Check if logx
    islogx = False
    for key in logx_plots:
        if key in f"{region}_{variable}":
            islogx = True
            break

    plt.sca(ax1)
    region_name = f"{region}_{variable}"
    label_ypos = 0.65
    plt.text(
        0.3,
        label_ypos,
        region_labels[region],
        ha="center",
        weight="bold",
        fontsize=30,
        transform=ax1.transAxes,
    )

    if args.ratio and args.data:
        plot_ratio(
            plots[f"Data_{year}"][f"{region}_{variable}"][:, sel_sum],
            hist_bkg_total,
            ax2,
            x_hatch,
        )
        # plot_pull(plots[f"Data_{year}"][region], hist_bkg_total, ax3, x_hatch)

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
    plt.xlabel(get_xlabel(region_name))
    if islogx:
        ax1.set_xscale("log")
        if args.ratio:
            ax2.set_xscale("log")
    if args.ratio and args.data:
        plt.sca(ax1)
    plt.yscale("log")
    plt.ylim(ylims[region])
    plt.legend(ncol=3, loc="upper center", columnspacing=1)
    plt.ylabel("muons")
    if "mass" in region_name:
        plt.ylabel("dimuon pairs")
    nMuon_label = "" if args.nMuon is None else f"_{args.nMuon}"
    plt.savefig(
        os.path.join(
            args.dest, args.tag, f"{region}_{variable}{nMuon_label}_{year}.pdf"
        ),
        bbox_inches="tight",
    )
    plt.close()


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(os.path.join(args.dest, args.tag), exist_ok=True)

    # Load plots and merge them
    print("Loading plots...", flush=True)
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
        plots = plots | plot_utils.loader(
            tag=f"{args.tag}_{year}_VR_JPsi",
            era=year,
            custom_lumi=args.lumi,
            load_data=args.data,
        )

    # Apply k-factor to QCD
    if args.data and args.normalize:
        print("Calculate and apply k-factor to QCD...", flush=True)
        k_factor = {}
        for year in track(years_to_load, description="Calculating k-factors"):
            k_factor[year] = plot_utils.calculate_k_factor(
                plots, year, region="VR_loose", process="QCD_Pt_MuEnrichedPt5"
            )
            for plot in plots["QCD_Pt_MuEnrichedPt5_" + year]:
                plots["QCD_Pt_MuEnrichedPt5_" + year][plot] = (
                    k_factor[year] * plots["QCD_Pt_MuEnrichedPt5_" + year][plot]
                )
        pprint("QCD k_factors:")
        pprint(k_factor)

    # print("Fit and extrapolation...", flush=True)
    # # QCD extrapolation
    # # Slice the first bin out where needed for fit stability
    # slice_hists = {
    #     "VR_loose": slice(3j, 8j),
    #     "VR_tight": slice(3j, 8j),
    # }
    # for year in track(years_to_load, description="Fitting and extrapolations"):
    #     qcd_extrapolation = plot_utils.Extrapolation(
    #         plots["QCD_Pt_MuEnrichedPt5_" + year], uncertainty_scheme="full"
    #     )
    #     qcd_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

    if "Run2" in args.year:
        run2_plots = plot_utils.merge_runs(plots, "Run2", data=args.data)
        plots = plots | run2_plots
    if "Run3" in args.year:
        run3_plots = plot_utils.merge_runs(plots, "Run3", data=args.data)
        plots = plots | run3_plots

    for year in track(args.year, description="Plotting regions"):
        for region in regions:
            for variable in variables:
                variable = variable.replace("dimuon_", "").replace("muon_", "")
                make_plot(args, plots, year, region, variable)
