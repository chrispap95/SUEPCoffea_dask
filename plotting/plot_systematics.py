import argparse
import logging
import os
import pathlib
import re

import matplotlib as mpl  # type: ignore[import]
import matplotlib.gridspec as gridspec  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.ticker as ticker  # type: ignore[import]
import mplhep as hep
import numpy as np
import plot_utils
from rich.progress import Progress  # type: ignore[import]
from rich.progress import track  # type: ignore[import]

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Dec2025",
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
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "systematics_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'systematics_plots'}.",
    )
    return parser.parse_args()


def merge_runs(plots, run):
    names = [
        "Higgs",
        "TTV",
        "ST_NLO",
        "WJets",
        "VV+VVV",
        "TT_powheg",
        "DY",
        "QCD_Pt_MuEnrichedPt5",
    ]
    # Add signal processes
    signal_processes = list(
        {p[:-5] for p in plots.keys() if re.search("GluGluToSUEP.*13TeV", p)}
    )
    # Remove 2016APV for now
    # years = ["2016APV", "2016", "2017", "2018"]
    years = ["2016", "2017", "2018"]
    if run == "Run3":
        years = ["2022", "2022EE", "2023", "2023BPix"]
        # Add signal processes
        signal_processes = list(
            {p[:-5] for p in plots.keys() if re.search("GluGluToSUEP.*13p6TeV", p)}
        )
    names.extend(signal_processes)
    run_plots = {}
    for name in names:
        run_plots[f"{name}_{run}"] = {}
        for year in years:
            if f"{name}_{year}" not in plots.keys():
                continue
            for plot in plots[f"{name}_{year}"]:
                if plot not in run_plots[f"{name}_{run}"].keys():
                    run_plots[f"{name}_{run}"][plot] = plots[f"{name}_{year}"][
                        plot
                    ].copy()
                else:
                    run_plots[f"{name}_{run}"][plot] += plots[f"{name}_{year}"][plot]
    return run_plots


def plot_systematics(args, plots, sample, region):
    if region not in plots[sample]:
        return

    year = sample.split("_")[-1]
    com_energy = "13TeV" if (year == "Run2") or year.startswith("201") else "13p6TeV"

    sample_tag = (
        sample.replace(f"_{year}", "")
        .replace(f"_{com_energy}", "")
        .replace("mode", "")
        .replace(".000", "")
        .replace("0_", "_")
        + " - "
    )
    if len(sample_tag) > 10:
        sample_tag += "\n"

    systematics = set()
    for key in plots[sample]:
        if "LHEPdf" in key:
            if "LHEPdfUp" not in key:
                continue
        if region in key:
            systematics.add(
                key.replace(region, "")
                .replace("extrapolation", "")
                .replace("_", "")
                .replace("Up", "")
                .replace("Down", "")
            )

    # print(region, sample, sorted(list(systematics)))

    for syst in sorted(systematics):
        if not syst:
            continue

        fig = plt.figure(figsize=(13, 12))
        gs = gridspec.GridSpec(4, 1, left=0.08, right=0.92, bottom=0.15)

        ax1 = plt.subplot(gs[0:3, 0])  # Top 3 rows
        ax2 = plt.subplot(gs[3, 0], sharex=ax1)  # 4th row

        hep.histplot(
            plots[sample][region],
            yerr=np.sqrt(plots[sample][region].variances()),
            label="nominal",
            color="C0",
            ax=ax1,
        )
        hep.histplot(
            plots[sample][f"{region}_{syst}Up"],
            yerr=np.sqrt(plots[sample][f"{region}_{syst}Up"].variances()),
            label="up",
            color="C1",
            ax=ax1,
        )
        hep.histplot(
            plots[sample][f"{region}_{syst}Down"],
            yerr=np.sqrt(plots[sample][f"{region}_{syst}Down"].variances()),
            label="down",
            color="C2",
            ax=ax1,
        )

        ax2.axhline(1, ls="--", color="gray")
        hep.histplot(
            np.divide(
                plots[sample][f"{region}_{syst}Up"].values(),
                plots[sample][region].values(),
                out=np.ones_like(plots[sample][region].values()),
                where=(plots[sample][region].values() > 0),
            ),
            bins=plots[sample][region].axes[0].edges,
            label="up",
            color="C1",
            ax=ax2,
        )
        hep.histplot(
            np.divide(
                plots[sample][f"{region}_{syst}Down"].values(),
                plots[sample][region].values(),
                out=np.ones_like(plots[sample][region].values()),
                where=(plots[sample][region].values() > 0),
            ),
            bins=plots[sample][region].axes[0].edges,
            label="down",
            color="C2",
            ax=ax2,
        )

        ax1.set_yscale("log")
        ax1.set_ylabel("Events")
        ax1.set_title(f"{sample_tag}{region} - {syst}")
        ax1.legend()
        ax1.xaxis.set_minor_locator(ticker.NullLocator())
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.set_ylim(0.5, 1.5)
        ax2.set_ylabel("Ratio")
        ax2.set_xlabel("nMuon")
        ax1.set_xlabel("")
        for label in ax1.xaxis.get_ticklabels():
            label.set_visible(False)
        # plt.tight_layout()
        plt.savefig(
            os.path.join(
                args.dest,
                args.tag,
                year,
                f"{region}_{sample.replace('.', 'p')}_{syst}_{year}.pdf",
            ),
            bbox_inches="tight",
        )
        plt.close()
    return


if "__main__" == __name__:
    args = parse_args()

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
        plots_CR = plot_utils.loader(
            tag=f"{args.tag}_{year}_CR",
            era=year,
            custom_lumi=args.lumi,
            load_data=False,
        )
        plots_SRs = plot_utils.loader(
            tag=f"{args.tag}_{year}_SRs",
            era=year,
            custom_lumi=args.lumi,
            load_data=False,
        )
        for sample in set(list(plots_CR.keys()) + list(plots_SRs.keys())):
            plots[sample] = plots_CR[sample] | plots_SRs[sample]

    plots = plot_utils.make_lhepdf_systematic(plots)

    for year in track(years_to_load, description="Fitting and extrapolations"):
        # QCD extrapolation
        # Slice the first bin out where needed for fit stability
        slice_hists = {
            "SR_low_temp_loose": slice(4j, None),
            "SR_low_temp_tight": slice(3j, None),
            "SR_high_temp_loose": slice(4j, None),
            "SR_high_temp_tight": slice(3j, None),
        }
        qcd_extrapolation = plot_utils.Extrapolation(
            plots[f"QCD_Pt_MuEnrichedPt5_{year}"], uncertainty_scheme="full"
        )
        qcd_extrapolation.fit_syst_variations(slice_hists=slice_hists, verbose=False)

        # DY extrapolation
        # Slice the first bin out where needed for fit stability
        slice_hists = {
            "SR_low_temp_loose": slice(4j, None),
            "SR_low_temp_tight": slice(3j, None),
            "SR_high_temp_loose": slice(4j, None),
            "SR_high_temp_tight": slice(4j, None),
        }
        dy_extrapolation = plot_utils.Extrapolation(
            plots[f"DY_{year}"], uncertainty_scheme="full"
        )
        dy_extrapolation.fit_syst_variations(slice_hists=slice_hists, verbose=False)

    if "Run2" in args.year:
        run2_plots = merge_runs(plots, "Run2")
        plots = plots | run2_plots
    if "Run3" in args.year:
        run3_plots = merge_runs(plots, "Run3")
        plots = plots | run3_plots

    # Plot systematics
    # To plot all signal samples, use the following line:
    # samples = [key for key in plots if "SUEP" in key]
    # For now, plot only a few samples
    samples = []
    for year in track(args.year, description="Plotting regions"):
        os.makedirs(os.path.join(args.dest, args.tag, year), exist_ok=True)
        com_energy = (
            "13TeV" if (year == "Run2") or year.startswith("201") else "13p6TeV"
        )
        samples.append(
            f"GluGluToSUEP_mS125.000_mPhi8.000_T32.000_modeleptonic_{com_energy}_{year}"
        )
        samples.append(
            f"GluGluToSUEP_mS125.000_mPhi1.400_T1.400_modehadronic_{com_energy}_{year}"
        )
        samples.append(f"QCD_Pt_MuEnrichedPt5_{year}")
        samples.append(f"DY_{year}")
        samples.append(f"VV+VVV_{year}")
    regions = [
        "CR_prompt",
        "CR_cb",
        "SR_low_temp_loose",
        "SR_low_temp_tight",
        "SR_low_temp_tight_extrapolation",
        "SR_high_temp_loose",
        "SR_high_temp_tight",
        "SR_high_temp_tight_extrapolation",
    ]
    with Progress() as progress:
        task = progress.add_task(
            "Plotting systematics...", total=len(samples) * len(regions)
        )
        for sample in samples:
            for region in regions:
                plot_systematics(args, plots, sample, region)
                progress.update(task, advance=1)
