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
    plots = {}
    for year in track(args.year, description="Loading plots"):
        plots = plots | plot_utils.loader(
            tag=f"{args.tag}_{year}_SRs",
            era=year,
            load_data=False,
        )

    tablefmt = "simple"
    if args.latex:
        tablefmt = "latex_raw"

    header = ["SUEP model", "SR high temp", "SR low temp"]
    yield_table = []

    # Let's build the SUEP model names manually so that they are in the order we want
    masses_S = [125, 200, 300, 400, 500, 600, 800, 1000]
    masses_phi = [1, 1.4, 2, 3, 4, 6, 8]
    T_over_mPhi = [0.25, 0.5, 1, 2, 4]
    modes = ["leptonic", "hadronic"]

    com_energy = lambda year: "13TeV" if year.startswith("201") else "13p6TeV"

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
                    y_high = hist.accumulators.WeightedSum()
                    y_low = hist.accumulators.WeightedSum()
                    exists = False
                    for year in args.year:
                        sample_year = f"{sample}_{com_energy(year)}_{year}"
                        if sample_year not in plots:
                            continue
                        exists = True
                        y_high += plots[sample_year]["SR_high_temp_tight"][7j::sum]
                        y_low += plots[sample_year]["SR_low_temp_tight"][7j::sum]

                    if not exists:
                        continue

                    if y_high.value > y_low.value:
                        style1 = Fore.GREEN
                        style2 = Fore.RED
                        reset1 = reset2 = Style.RESET_ALL
                        if tablefmt == "latex_raw":
                            style1 = r"\textbf{"
                            style2 = ""
                            reset1 = "}"
                            reset2 = ""
                    elif y_high.value < y_low.value:
                        style1 = Fore.RED
                        style2 = Fore.GREEN
                        reset1 = reset2 = Style.RESET_ALL
                        if tablefmt == "latex_raw":
                            style1 = ""
                            style2 = r"\textbf{"
                            reset1 = ""
                            reset2 = "}"
                    else:
                        style1 = Fore.WHITE
                        style2 = Fore.WHITE
                        reset1 = reset2 = Style.RESET_ALL
                        if tablefmt == "latex_raw":
                            style1 = ""
                            style2 = ""
                            reset1 = ""
                            reset2 = ""

                    yield_table.append(
                        [
                            name,
                            style1
                            + smart_rounding(
                                y_high.value,
                                math.sqrt(y_high.variance),
                                mode=tablefmt,
                            )
                            + reset1,
                            style2
                            + smart_rounding(
                                y_low.value,
                                math.sqrt(y_low.variance),
                                mode=tablefmt,
                            )
                            + reset2,
                        ]
                    )
    print()
    print(tabulate(yield_table, headers=header, tablefmt=tablefmt))
    print()
