"""
Compare the sum of yearly SR extrapolations with the extrapolation fit on the
full run for QCD and DY MC.
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

RUN_YEARS = {
    "Run2": ["2016", "2017", "2018"],
    "Run3": ["2022", "2022EE", "2023", "2023BPix"],
}

PROCESS_CONFIG = {
    "QCD": {
        "sample": "QCD_Pt_MuEnrichedPt5",
        "slice_hists": {
            "SR_low_temp_loose": slice(4j, None),
            "SR_low_temp_tight": slice(3j, None),
            "SR_high_temp_loose": slice(4j, None),
            "SR_high_temp_tight": slice(3j, None),
        },
    },
    "DY": {
        "sample": "DY",
        "slice_hists": {
            "SR_low_temp_loose": slice(4j, None),
            "SR_low_temp_tight": slice(3j, None),
            "SR_high_temp_loose": slice(4j, None),
            "SR_high_temp_tight": slice(4j, None),
        },
    },
}

EXTRAPOLATED_REGIONS = {
    r"$SR^{loose}_{low~T}$": "SR_low_temp_loose_extrapolation",
    r"$SR^{tight}_{low~T}$": "SR_low_temp_tight_extrapolation",
    r"$SR^{loose}_{high~T}$": "SR_high_temp_loose_extrapolation",
    r"$SR^{tight}_{high~T}$": "SR_high_temp_tight_extrapolation",
}


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
        default=["Run2", "Run3"],
        help="Runs to compare. Include Run2 and/or Run3. Component years are loaded automatically.",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        help="Custom integrated luminosity to be used (in pb^-1).",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "sr_fit_stability_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'sr_fit_stability_plots'}",
    )
    return parser.parse_args()


def resolve_years_to_load(requested_years):
    years_to_load = requested_years
    if "Run2" in requested_years:
        years_to_load = RUN_YEARS["Run2"]
    if "Run3" in requested_years:
        years_to_load = RUN_YEARS["Run3"]
    if "Run2" in requested_years and "Run3" in requested_years:
        years_to_load = RUN_YEARS["Run2"] + RUN_YEARS["Run3"]
    return years_to_load


def requested_runs(requested_years):
    return [run for run in RUN_YEARS if run in requested_years]


def snapshot_extrapolated_hists(plots):
    missing = [
        region for region in EXTRAPOLATED_REGIONS.values() if region not in plots
    ]
    if missing:
        raise KeyError(f"Missing extrapolated histograms: {missing}")
    return {region: plots[region].copy() for region in EXTRAPOLATED_REGIONS.values()}


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
        len(EXTRAPOLATED_REGIONS),
        figsize=(26, 8),
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
        ax_top.set_xlim(h_sum.axes[0].edges[0], h_sum.axes[0].edges[-1])
        ax_top.xaxis.set_minor_locator(ticker.NullLocator())
        ax_top.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax_top.legend()
        if column == 0:
            ax_top.set_ylabel("Events")
            # hep.cms.label(llabel="Simulation", data=False, ax=ax_top)
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
        ax_bottom.xaxis.set_minor_locator(ticker.NullLocator())
        ax_bottom.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        if column == 0:
            ax_bottom.set_ylabel("year sum / run")

        for label in ax_top.xaxis.get_ticklabels():
            label.set_visible(False)

    plt.savefig(
        os.path.join(output_dir, f"sr_fit_stability_{sample_label.lower()}_{run}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


if "__main__" == __name__:
    args = parse_args()

    runs_to_process = requested_runs(args.year)
    if not runs_to_process:
        raise ValueError(
            "Need Run2 and/or Run3 in --year to build SR fit stability plots."
        )

    output_dir = os.path.join(args.dest, args.tag)
    os.makedirs(output_dir, exist_ok=True)

    years_to_load = resolve_years_to_load(args.year)

    plots = {}
    for year in track(years_to_load, description="Loading SR plots"):
        plots = plots | plot_utils.loader(
            tag=f"{args.tag}_{year}_SRs",
            era=year,
            custom_lumi=args.lumi,
            load_data=False,
        )

    for year in track(years_to_load, description="Fitting yearly extrapolations"):
        for process in PROCESS_CONFIG.values():
            extrapolation = plot_utils.Extrapolation(
                plots[f"{process['sample']}_{year}"],
                uncertainty_scheme="full",
            )
            extrapolation.extrapolate(
                slice_hists=process["slice_hists"],
                verbose=False,
            )

    for run in track(runs_to_process, description="Building stability plots"):
        run_plots = plot_utils.merge_runs(plots, run, data=False)
        plots = plots | run_plots

        for sample_label, process in PROCESS_CONFIG.items():
            sample_key = f"{process['sample']}_{run}"

            # The merged run plot already contains the sum of the per-year
            # extrapolated histograms. Snapshot them before the combined fit
            # overwrites the same keys with the run-wide extrapolation.
            summed_yearly_extrapolation = snapshot_extrapolated_hists(plots[sample_key])

            run_extrapolation = plot_utils.Extrapolation(
                plots[sample_key],
                uncertainty_scheme="full",
            )
            run_extrapolation.extrapolate(
                slice_hists=process["slice_hists"],
                verbose=False,
            )

            plot_run_stability(
                summed_yearly_extrapolation,
                run_extrapolation.plots,
                run,
                sample_label,
                output_dir,
            )
