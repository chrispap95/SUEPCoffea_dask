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
        default="full_analysis_Feb2026",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--year",
        type=str,
        nargs="*",
        default=[
            "2016",
            "2017",
            "2018",
            "Run2",
            "2022",
            "2022EE",
            "2023",
            "2023BPix",
            "Run3",
        ],
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
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "fit_results_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'fit_results_plots'}",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="If set, will load the data plots as well. Note that the fits are performed on MC only.",
    )
    parser.add_argument(
        "--VR",
        action="store_true",
        help="If set, will load the validation region plots as well.",
    )
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(os.path.join(args.dest, args.tag), exist_ok=True)

    # Load plots and merge them
    years_to_load = args.year
    if "Run2" in args.year:
        # years_to_load = ["2016APV", "2016", "2017", "2018"]
        years_to_load = ["2016", "2017", "2018"]
    if "Run3" in args.year:
        years_to_load = ["2022", "2022EE", "2023", "2023BPix"]
    if "Run2" in args.year and "Run3" in args.year:
        years_to_load = [
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
            tag=f"{args.tag}_{year}_SRs",
            era=year,
            custom_lumi=args.lumi,
            load_data=False,
        )
        if args.VR:
            plots_vr = plot_utils.loader(
                tag=f"{args.tag}_{year}_VR",
                era=year,
                custom_lumi=args.lumi,
                load_data=False,
            )
            for key in plots:
                if key not in plots_vr:
                    continue
                plots[key] = plots[key] | plots_vr[key]

    qcd_extrapolations = {}
    dy_extrapolations = {}
    for year in track(years_to_load, description="Fitting and extrapolations"):
        # Slice the first bin out where needed for fit stability
        slice_hists = {
            "SR_low_temp_loose": slice(4j, 7j),
            "SR_low_temp_tight": slice(3j, 5j),
            "SR_high_temp_loose": slice(4j, 7j),
            "SR_high_temp_tight": slice(3j, 5j),
            "VR_loose": slice(3j, 6j),
            "VR_tight": slice(3j, 6j),
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
        run2_plots = plot_utils.merge_runs(plots, "Run2", data=args.data)
        plots = plots | run2_plots
    if "Run3" in args.year:
        run3_plots = plot_utils.merge_runs(plots, "Run3", data=args.data)
        plots = plots | run3_plots

    # Plot regions
    regions = [
        "SR_low_temp",
        "SR_high_temp",
    ]
    if args.VR:
        regions += ["VR"]

    for year in track(years_to_load, description="Plotting regions"):
        for region in regions:
            qcd_extrapolations[year].plot_fit(region, add_text="QCD")
            plt.savefig(
                os.path.join(args.dest, args.tag, f"plot_fit_QCD_{region}_{year}.pdf"),
                bbox_inches="tight",
            )
            plt.close()

            dy_extrapolations[year].plot_fit(region, add_text="DY")
            plt.savefig(
                os.path.join(args.dest, args.tag, f"plot_fit_DY_{region}_{year}.pdf"),
                bbox_inches="tight",
            )
            plt.close()

        qcd_extrapolations[year].plot_overlay(regions=regions, add_text="QCD")
        plt.savefig(
            os.path.join(args.dest, args.tag, f"fit_overlay_QCD_{year}.pdf"),
            bbox_inches="tight",
        )
        plt.close()

        dy_extrapolations[year].plot_overlay(regions=regions, add_text="DY")
        plt.savefig(
            os.path.join(args.dest, args.tag, f"fit_overlay_DY_{year}.pdf"),
            bbox_inches="tight",
        )
        plt.close()
