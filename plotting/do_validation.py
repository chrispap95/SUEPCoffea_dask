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

import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.ticker as ticker  # type: ignore[import]
import mplhep as hep
import numpy as np
import plot_utils
from rich.progress import track  # type: ignore[import]

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)

SLICE_HISTS = {
    "VR_loose": slice(4j, None),
    "VR_tight": slice(3j, None),
}

EXTRAPOLATED_REGIONS = {
    "VR loose": "VR_loose_extrapolation",
    "VR tight": "VR_tight_extrapolation",
}


def snapshot_extrapolated_hists(plots):
    return {
        region: plots[region].copy()
        for region in EXTRAPOLATED_REGIONS.values()
        if region in plots
    }


def calculate_ratio(numerator, denominator):
    numerator_vals = numerator.values()
    denominator_vals = denominator.values()
    numerator_vars = numerator.variances()
    denominator_vars = denominator.variances()

    ratio = np.divide(
        numerator_vals,
        denominator_vals,
        out=np.zeros_like(numerator_vals, dtype=float),
        where=denominator_vals != 0,
    )
    ratio_unc = np.sqrt(
        np.divide(
            numerator_vars,
            denominator_vals**2,
            out=np.zeros_like(numerator_vars, dtype=float),
            where=denominator_vals != 0,
        )
        + np.divide(
            numerator_vals**2 * denominator_vars,
            denominator_vals**4,
            out=np.zeros_like(denominator_vars, dtype=float),
            where=denominator_vals != 0,
        )
    )

    return ratio, ratio_unc


def plot_run_stability(
    summed_yearly_extrapolation,
    total_run_extrapolation,
    run,
    sample_label,
    output_dir,
    is_data=False,
):
    max_y = 1
    min_y = np.inf
    for region in EXTRAPOLATED_REGIONS.values():
        for plots in (summed_yearly_extrapolation, total_run_extrapolation):
            values = plots[region].values()
            positive_values = values[values > 0]
            if len(positive_values) == 0:
                continue
            max_y = max(max_y, positive_values.max())
            min_y = min(min_y, positive_values.min())

    if not np.isfinite(min_y):
        min_y = 0.1

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 10),
        sharex="col",
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    for column, (title, region) in enumerate(EXTRAPOLATED_REGIONS.items()):
        ax_top = axes[0, column]
        ax_bottom = axes[1, column]

        h_sum = summed_yearly_extrapolation[region]
        h_total = total_run_extrapolation[region]

        h_sum.plot(
            yerr=np.sqrt(h_sum.variances()),
            label="sum of yearly fits",
            color="C0",
            ax=ax_top,
        )
        h_total.plot(
            yerr=np.sqrt(h_total.variances()),
            label=f"{run} fit",
            color="C1",
            ls="--",
            ax=ax_top,
        )

        ax_top.set_title(title)
        ax_top.set_yscale("log")
        ax_top.set_ylim(
            10 ** np.floor(np.log10(0.5 * min_y)),
            10 ** np.ceil(np.log10(2 * max_y)),
        )
        ax_top.set_ylabel("Events")
        ax_top.xaxis.set_minor_locator(ticker.NullLocator())
        ax_top.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_top.legend()

        if column == 0:
            hep.cms.label(
                llabel="Preliminary" if is_data else "Simulation",
                data=is_data,
                ax=ax_top,
            )
            ax_top.text(
                0.5,
                0.9,
                f"{sample_label}, {run}",
                transform=ax_top.transAxes,
                verticalalignment="top",
                horizontalalignment="center",
            )

        ratio, ratio_unc = calculate_ratio(h_sum, h_total)
        ax_bottom.errorbar(
            h_sum.axes[0].centers,
            ratio,
            yerr=ratio_unc,
            color="black",
            fmt="o",
            linestyle="none",
        )
        ax_bottom.axhline(1, ls="--", color="gray")
        ax_bottom.set_ylim(0, 2)
        ax_bottom.set_xlim(h_sum.axes[0].edges[0], h_sum.axes[0].edges[-1])
        ax_bottom.set_xlabel("nMuon")
        ax_bottom.set_ylabel("year sum / run")
        ax_bottom.xaxis.set_minor_locator(ticker.NullLocator())
        ax_bottom.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        for label in ax_top.xaxis.get_ticklabels():
            label.set_visible(False)

    plt.savefig(
        os.path.join(
            output_dir,
            f"fit_stability_{sample_label.lower()}_{run}.pdf",
        ),
        bbox_inches="tight",
    )
    plt.close(fig)


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
        qcd_extrapolation = plot_utils.Extrapolation(
            plots[f"QCD_Pt_MuEnrichedPt5_{year}"], uncertainty_scheme="full"
        )
        qcd_extrapolation.extrapolate(slice_hists=SLICE_HISTS, verbose=False)

        data_extrapolation = plot_utils.Extrapolation(
            plots[f"Data_{year}"], is_data=True, uncertainty_scheme="full"
        )
        data_extrapolation.extrapolate(slice_hists=SLICE_HISTS, verbose=False)

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
        run2_plots = plot_utils.merge_runs(plots, "Run2", data=True)
        plots = plots | run2_plots

        # The merged run plots already contain the sum of the per-year
        # extrapolated histograms. Snapshot them before the combined fit
        # overwrites the same keys with the run-wide extrapolation.
        qcd_summed_yearly_extrapolation = snapshot_extrapolated_hists(
            plots["QCD_Pt_MuEnrichedPt5_Run2"]
        )
        data_summed_yearly_extrapolation = snapshot_extrapolated_hists(
            plots["Data_Run2"]
        )

        qcd_extrapolation = plot_utils.Extrapolation(
            plots[f"QCD_Pt_MuEnrichedPt5_Run2"], uncertainty_scheme="full"
        )
        qcd_extrapolation.extrapolate(slice_hists=SLICE_HISTS, verbose=False)

        data_extrapolation = plot_utils.Extrapolation(
            plots[f"Data_Run2"], is_data=True, uncertainty_scheme="full"
        )
        data_extrapolation.extrapolate(slice_hists=SLICE_HISTS, verbose=False)

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

        plot_run_stability(
            qcd_summed_yearly_extrapolation,
            qcd_extrapolation.plots,
            "Run2",
            "QCD",
            output_dir,
        )

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

        plot_run_stability(
            data_summed_yearly_extrapolation,
            data_extrapolation.plots,
            "Run2",
            "Data",
            output_dir,
            is_data=True,
        )

    if "Run3" in args.year:
        run3_plots = plot_utils.merge_runs(plots, "Run3", data=True)
        plots = plots | run3_plots

        qcd_summed_yearly_extrapolation = snapshot_extrapolated_hists(
            plots["QCD_Pt_MuEnrichedPt5_Run3"]
        )
        data_summed_yearly_extrapolation = snapshot_extrapolated_hists(
            plots["Data_Run3"]
        )

        qcd_extrapolation = plot_utils.Extrapolation(
            plots[f"QCD_Pt_MuEnrichedPt5_Run3"], uncertainty_scheme="full"
        )
        qcd_extrapolation.extrapolate(slice_hists=SLICE_HISTS, verbose=False)

        data_extrapolation = plot_utils.Extrapolation(
            plots[f"Data_Run3"], is_data=True, uncertainty_scheme="full"
        )
        data_extrapolation.extrapolate(slice_hists=SLICE_HISTS, verbose=False)

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

        plot_run_stability(
            qcd_summed_yearly_extrapolation,
            qcd_extrapolation.plots,
            "Run3",
            "QCD",
            output_dir,
        )

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

        plot_run_stability(
            data_summed_yearly_extrapolation,
            data_extrapolation.plots,
            "Run3",
            "Data",
            output_dir,
            is_data=True,
        )
