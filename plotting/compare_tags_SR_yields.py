"""
Compare the yields in the signal regions for two different analysis tags: tag1 and tag2.
Table will print the yields for tab1 minus yields for tab2.
"""

import argparse
import math

import hist
import plot_utils
from colorama import Fore, Style  # type: ignore[import]
from rich.progress import track  # type: ignore[import]
from tabulate import tabulate  # type: ignore[import]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--newtag",
        type=str,
        required=True,
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--oldtag",
        type=str,
        required=True,
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
        "--details",
        action="store_true",
        help="Print detailed information for each signal region. Otherwise, only the maximum region is printed.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print the table in LaTeX format",
    )
    return parser.parse_args()


def smart_rounding(value, uncertainty, mode="simple"):
    prefix = suffix = ""
    sep = "±"
    if mode == "latex_raw":
        prefix = "$"
        suffix = "$"
        sep = r"\pm"
    if uncertainty == 0 and value == 0:
        return f"{prefix}0{sep}0{suffix}"
    elif uncertainty < 0.1:
        return f"{prefix}{value:.3f} {sep} {uncertainty:.3f}{suffix}"
    elif uncertainty < 1:
        return f"{prefix}{value:.2f} {sep} {uncertainty:.2f}{suffix}"
    elif uncertainty < 10:
        return f"{prefix}{value:.1f} {sep} {uncertainty:.1f}{suffix}"
    else:
        return f"{prefix}{value:.0f} {sep} {uncertainty:.0f}{suffix}"


if "__main__" == __name__:
    args = parse_args()

    # Load plots and merge them
    plots_new = {}
    plots_old = {}
    for year in track(args.year, description="Loading plots"):
        plots_new = plots_new | plot_utils.loader(
            tag=f"{args.newtag}_{year}_SRs",
            era=year,
            load_data=False,
        )
        plots_old = plots_old | plot_utils.loader(
            tag=f"{args.oldtag}_{year}_SRs",
            era=year,
            load_data=False,
        )

    if len(plots_new) == 0 or len(plots_old) == 0:
        raise ValueError("No plots found for the given tags and years.")

    tablefmt = "simple"
    if args.latex:
        tablefmt = "latex_raw"

    header = ["SUEP model", "Maximum region"]
    if args.details:
        header = ["SUEP model", "SR high temp", "SR low temp", "Maximum region"]
    yield_table = []

    # Let's build the SUEP model names manually so that they are in the order we want
    masses_S = [125, 200, 300, 400, 500, 600, 800, 1000]
    masses_phi = [1, 1.4, 2, 3, 4, 6, 8]
    T_over_mPhi = [0.25, 0.5, 1, 2, 4]
    modes = ["leptonic", "hadronic"]

    com_energy = lambda year: "13TeV" if year.startswith("201") else "13p6TeV"

    sum_rel_diff_max = 0.0
    n_points = 0

    for mPhi in masses_phi:
        for r in T_over_mPhi:
            T = r * mPhi
            for mode in modes:
                for mS in masses_S:
                    scan_point = f"mS{mS:.3f}_mPhi{mPhi:.3f}_T{T:.3f}_mode{mode}"
                    sample = f"GluGluToSUEP_{scan_point}"
                    name = f"mS{mS}_mPhi{mPhi}_T{T}_{mode}"
                    if tablefmt == "latex_raw":
                        name = (
                            f"$m_S={mS}$, $m_" + r"\phi" + f"={mPhi}$, $T={T}$, {mode}"
                        )

                    sample_yields = []
                    y_high_new = hist.accumulators.WeightedSum()
                    y_low_new = hist.accumulators.WeightedSum()
                    y_high_old = hist.accumulators.WeightedSum()
                    y_low_old = hist.accumulators.WeightedSum()
                    exists = False
                    for year in args.year:
                        sample_year = f"{sample}_{com_energy(year)}_{year}"
                        if sample_year in plots_new:
                            exists = True
                            y_high_new += plots_new[sample_year]["SR_high_temp_tight"][
                                7j::sum
                            ]
                            y_low_new += plots_new[sample_year]["SR_low_temp_tight"][
                                7j::sum
                            ]
                        if sample_year in plots_old:
                            exists = True
                            y_high_old += plots_old[sample_year]["SR_high_temp_tight"][
                                7j::sum
                            ]
                            y_low_old += plots_old[sample_year]["SR_low_temp_tight"][
                                7j::sum
                            ]

                    if not exists:
                        continue

                    y_max_old = y_high_old
                    if y_low_old.value > y_high_old.value:
                        y_max_old = y_low_old
                    y_max_new = y_high_new
                    if y_low_new.value > y_high_new.value:
                        y_max_new = y_low_new

                    rel_diff_high = rel_diff_low = rel_diff_max = 0.0
                    if y_high_old.value > 0:
                        rel_diff_high = (
                            (y_high_new.value - y_high_old.value)
                            / y_high_old.value
                            * 100
                        )
                    elif y_high_new.value > 0:
                        rel_diff_high = 100.0
                    if y_low_old.value > 0:
                        rel_diff_low = (
                            (y_low_new.value - y_low_old.value) / y_low_old.value * 100
                        )
                    elif y_low_new.value > 0:
                        rel_diff_low = 100.0
                    if y_max_old.value > 0:
                        rel_diff_max = (
                            (y_max_new.value - y_max_old.value) / y_max_old.value * 100
                        )
                    elif y_max_new.value > 0:
                        rel_diff_max = 100.0

                    style_high = Fore.WHITE
                    if rel_diff_high > 0.5:
                        style_high = Fore.GREEN
                    if rel_diff_high < -0.5:
                        style_high = Fore.YELLOW
                    if rel_diff_high < -5:
                        style_high = Fore.RED

                    style_low = Fore.WHITE
                    if rel_diff_low > 0.5:
                        style_low = Fore.GREEN
                    if rel_diff_low < -0.5:
                        style_low = Fore.YELLOW
                    if rel_diff_low < -5:
                        style_low = Fore.RED

                    style_max = Fore.WHITE
                    if rel_diff_max > 0.5:
                        style_max = Fore.GREEN
                    if rel_diff_max < -0.5:
                        style_max = Fore.YELLOW
                    if rel_diff_max < -5:
                        style_max = Fore.RED

                    style_reset = Style.RESET_ALL

                    row = [name]
                    if args.details:
                        row += [
                            style_high
                            + smart_rounding(
                                y_high_new.value - y_high_old.value,
                                math.sqrt(y_high_new.variance + y_high_old.variance),
                                mode=tablefmt,
                            )
                            + f" ({rel_diff_high:.1f}%)"
                            + style_reset,
                            style_low
                            + smart_rounding(
                                y_low_new.value - y_low_old.value,
                                math.sqrt(y_low_new.variance + y_low_old.variance),
                                mode=tablefmt,
                            )
                            + f" ({rel_diff_low:.1f}%)"
                            + style_reset,
                        ]
                    row += [
                        style_max
                        + smart_rounding(
                            y_max_new.value - y_max_old.value,
                            math.sqrt(y_max_new.variance + y_max_old.variance),
                            mode=tablefmt,
                        )
                        + f" ({rel_diff_max:.1f}%)"
                        + style_reset,
                    ]
                    yield_table.append(row)

                    sum_rel_diff_max += rel_diff_max
                    n_points += 1

    print()
    print(tabulate(yield_table, headers=header, tablefmt=tablefmt))
    print()

    print(
        f"Average relative difference in maximum region: {sum_rel_diff_max / n_points:.2f}%"
    )
