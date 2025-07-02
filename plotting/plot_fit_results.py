import argparse
import logging
import os
import pathlib

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
        default="full_analysis_Jun2025",
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
        default=str(pathlib.Path(__file__).parent / "fit_results_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'fit_results_plots'}",
    )
    parser.add_argument(
        "--VR",
        action="store_true",
        help="If set, will load the validation region plots as well.",
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
    years = ["2016APV", "2016", "2017", "2018"]
    if run == "Run3":
        names.remove("TTV")
        names.remove("ST_NLO")
        years = ["2022", "2022EE", "2023", "2023BPix"]
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
    os.makedirs(args.dest, exist_ok=True)

    # Load plots and merge them
    years_to_load = args.year
    if "Run2" in args.year:
        years_to_load = ["2016APV", "2016", "2017", "2018"]
    if "Run3" in args.year:
        years_to_load = ["2022", "2022EE", "2023", "2023BPix"]
    if "Run2" in args.year and "Run3" in args.year:
        years_to_load = [
            "2016APV",
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
            tag=f"{args.tag}_{year}_CR",
            era=year,
            custom_lumi=args.lumi,
            load_data=False,
        )
        plots = plots | plot_utils.loader(
            tag=f"{args.tag}_{year}_SRs",
            era=year,
            custom_lumi=args.lumi,
            load_data=False,
        )
        if args.VR:
            plots = plots | plot_utils.loader(
                tag=f"{args.tag}_{year}_VR",
                era=year,
                custom_lumi=args.lumi,
                load_data=False,
            )

    qcd_extrapolations = {}
    dy_extrapolations = {}
    for year in track(years_to_load, description="Fitting and extrapolations"):
        # Slice the first bin out where needed for fit stability
        slice_hists = {
            "SR_low_temp_loose": slice(4j, None),
            "SR_low_temp_tight": slice(3j, None),
            "SR_high_temp_loose": slice(4j, None),
            "SR_high_temp_tight": slice(3j, None),
        }
        if year == "2016APV":
            slice_hists["SR_high_temp_loose"] = slice(3j, 6j)
            slice_hists["SR_high_temp_tight"] = slice(3j, 6j)
        qcd_extrapolations[year] = plot_utils.Extrapolation(
            plots[f"QCD_Pt_MuEnrichedPt5_{year}"],
            uncertainty_scheme="full",
        )
        qcd_extrapolations[year].fit_syst_variations(
            slice_hists=slice_hists, verbose=False
        )

        slice_hists = {
            "SR_low_temp_loose": slice(4j, None),
            "SR_low_temp_tight": slice(3j, None),
            "SR_high_temp_loose": slice(4j, None),
            "SR_high_temp_tight": slice(4j, None),
        }
        dy_extrapolations[year] = plot_utils.Extrapolation(
            plots[f"DY_{year}"], uncertainty_scheme="full"
        )
        dy_extrapolations[year].fit_syst_variations(
            slice_hists=slice_hists, verbose=False
        )

    if "Run2" in args.year:
        run2_plots = merge_runs(plots, "Run2")
        plots = plots | run2_plots
    if "Run3" in args.year:
        run3_plots = merge_runs(plots, "Run3")
        plots = plots | run3_plots

    # Plot regions
    regions = [
        "SR_low_temp",
        "SR_high_temp",
    ]
    if args.VR:
        regions += ["VR"]

    for year in track(args.year, description="Plotting regions"):
        for region in regions:
            qcd_extrapolations[year].plot_fit(region, add_text="QCD")
            plt.savefig(
                f"{args.dest}/plot_fit_QCD_{region}_{year}_{args.tag}.pdf",
                bbox_inches="tight",
            )
            plt.close()

            dy_extrapolations[year].plot_fit(region, add_text="DY")
            plt.savefig(
                f"{args.dest}/plot_fit_DY_{region}_{year}_{args.tag}.pdf",
                bbox_inches="tight",
            )
            plt.close()

        qcd_extrapolations[year].plot_overlay(regions=regions, add_text="QCD")
        plt.savefig(
            f"{args.dest}/fit_overlay_QCD_{year}_{args.tag}.pdf", bbox_inches="tight"
        )
        plt.close()

        dy_extrapolations[year].plot_overlay(regions=regions, add_text="DY")
        plt.savefig(
            f"{args.dest}/fit_overlay_DY_{year}_{args.tag}.pdf", bbox_inches="tight"
        )
        plt.close()
