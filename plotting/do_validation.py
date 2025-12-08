"""
Let's put the entire validation here.
I would like to have the following plots:
 - fit results plots both for QCD and data
 - overlay plots with data and QCD for both extrapolations as well and with a ratio plot for QCD/data

Total of 3 + 3 = 6 plots for the fit results and
"""

import argparse
import logging
import os
import pathlib
import re

import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import mplhep as hep
import plot_utils
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
        default=str(pathlib.Path(__file__).parent / "validation_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'validation_plots'}",
    )
    return parser.parse_args()


def merge_runs(plots, run, data=True):
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
    if data:
        names.append("Data")
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


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    output_dir = os.path.join(args.dest, args.tag)
    os.makedirs(output_dir, exist_ok=True)

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
        plots = plots | plot_utils.loader(
            tag=f"{args.tag}_{year}_VR",
            era=year,
            custom_lumi=args.lumi,
            load_data=True,
        )

    for year in track(years_to_load, description="Fitting and extrapolations"):
        # QCD extrapolation
        # Slice the first bin out where needed for fit stability
        slice_hists = {
            "VR_loose": slice(4j, None),
            "VR_tight": slice(3j, None),
        }

        qcd_extrapolation = plot_utils.Extrapolation(
            plots[f"QCD_Pt_MuEnrichedPt5_{year}"], uncertainty_scheme="full"
        )
        qcd_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

        data_extrapolation = plot_utils.Extrapolation(
            plots[f"Data_{year}"], is_data=True, uncertainty_scheme="full"
        )
        data_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

        qcd_extrapolation.plot_fit("VR", add_label=True, add_text="QCD")
        plt.savefig(
            os.path.join(output_dir, f"plot_fit_VR_qcd_{year}.pdf"), bbox_inches="tight"
        )
        plt.close()

        qcd_extrapolation.plot_overlay(regions=["VR"], add_label=True, add_text="QCD")
        plt.savefig(
            os.path.join(output_dir, f"fit_overlay_qcd_{year}.pdf"), bbox_inches="tight"
        )
        plt.close()

        data_extrapolation.plot_fit("VR", add_label=True, add_text="Data")
        plt.savefig(
            os.path.join(output_dir, f"plot_fit_VR_data_{year}.pdf"),
            bbox_inches="tight",
        )
        plt.close()

        data_extrapolation.plot_overlay(regions=["VR"], add_label=True, add_text="Data")
        plt.savefig(
            os.path.join(output_dir, f"fit_overlay_data_{year}.pdf"),
            bbox_inches="tight",
        )
        plt.close()

    if "Run2" in args.year:
        run2_plots = merge_runs(plots, "Run2", data=True)
        plots = plots | run2_plots

        # QCD extrapolation
        # Slice the first bin out where needed for fit stability
        slice_hists = {
            "VR_loose": slice(4j, None),
            "VR_tight": slice(3j, None),
        }

        qcd_extrapolation = plot_utils.Extrapolation(
            plots[f"QCD_Pt_MuEnrichedPt5_Run2"], uncertainty_scheme="full"
        )
        qcd_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

        data_extrapolation = plot_utils.Extrapolation(
            plots[f"Data_Run2"], is_data=True, uncertainty_scheme="full"
        )
        data_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

        qcd_extrapolation.plot_fit("VR", add_label=True, add_text="QCD")
        plt.savefig(
            os.path.join(output_dir, f"plot_fit_VR_qcd_Run2.pdf"), bbox_inches="tight"
        )
        plt.close()

        qcd_extrapolation.plot_overlay(regions=["VR"], add_label=True, add_text="QCD")
        plt.savefig(
            os.path.join(output_dir, f"fit_overlay_qcd_Run2.pdf"), bbox_inches="tight"
        )
        plt.close()

        data_extrapolation.plot_fit("VR", add_label=True, add_text="Data")
        plt.savefig(
            os.path.join(output_dir, f"plot_fit_VR_data_Run2.pdf"),
            bbox_inches="tight",
        )
        plt.close()

        data_extrapolation.plot_overlay(regions=["VR"], add_label=True, add_text="Data")
        plt.savefig(
            os.path.join(output_dir, f"fit_overlay_data_Run2.pdf"),
            bbox_inches="tight",
        )
        plt.close()

    if "Run3" in args.year:
        run3_plots = merge_runs(plots, "Run3", data=True)
        plots = plots | run3_plots

        # QCD extrapolation
        # Slice the first bin out where needed for fit stability
        slice_hists = {
            "VR_loose": slice(4j, None),
            "VR_tight": slice(3j, None),
        }

        qcd_extrapolation = plot_utils.Extrapolation(
            plots[f"QCD_Pt_MuEnrichedPt5_Run3"], uncertainty_scheme="full"
        )
        qcd_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

        data_extrapolation = plot_utils.Extrapolation(
            plots[f"Data_Run3"], is_data=True, uncertainty_scheme="full"
        )
        data_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

        qcd_extrapolation.plot_fit("VR", add_label=True, add_text="QCD")
        plt.savefig(
            os.path.join(output_dir, f"plot_fit_VR_qcd_Run3.pdf"), bbox_inches="tight"
        )
        plt.close()

        qcd_extrapolation.plot_overlay(regions=["VR"], add_label=True, add_text="QCD")
        plt.savefig(
            os.path.join(output_dir, f"fit_overlay_qcd_Run3.pdf"), bbox_inches="tight"
        )
        plt.close()

        data_extrapolation.plot_fit("VR", add_label=True, add_text="Data")
        plt.savefig(
            os.path.join(output_dir, f"plot_fit_VR_data_Run3.pdf"),
            bbox_inches="tight",
        )
        plt.close()

        data_extrapolation.plot_overlay(regions=["VR"], add_label=True, add_text="Data")
        plt.savefig(
            os.path.join(output_dir, f"fit_overlay_data_Run3.pdf"),
            bbox_inches="tight",
        )
        plt.close()
