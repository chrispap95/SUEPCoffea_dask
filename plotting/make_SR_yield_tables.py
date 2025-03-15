import argparse
import math

import plot_utils
from colorama import Fore, Style  # type: ignore[import]
from tabulate import tabulate  # type: ignore[import]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Feb2025",
        help="Tag to identify the analysis",
    )
    return parser.parse_args()


def smart_rounding(value, uncertainty):
    if uncertainty == 0 and value == 0:
        return "0 ± 0"
    elif uncertainty < 1:
        return f"{value:.2f} ± {uncertainty:.2f}"
    elif uncertainty < 10:
        return f"{value:.1f} ± {uncertainty:.1f}"
    else:
        return f"{value:.0f} ± {uncertainty:.0f}"


if "__main__" == __name__:
    args = parse_args()

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots = plot_utils.loader(tag=f"{args.tag}_SRs", load_data=False)
    print("Done!", flush=True)

    tablefmt = "simple"
    # tablefmt = "latex_raw"

    header = ["SUEP model", "SR high temp", "SR low temp"]
    yield_table = []

    # Let's build the SUEP model names manually so that they are in the order we want
    masses_S = [125, 200, 300, 400, 500, 600, 700, 800, 1000]
    masses_phi = [1, 1.4, 2, 3, 4, 6, 8]
    T_over_mPhi = [0.25, 0.5, 1, 2, 4]
    modes = ["leptonic", "hadronic"]
    for mPhi in masses_phi:
        for r in T_over_mPhi:
            T = r * mPhi
            for mode in modes:
                for mS in masses_S:
                    sample = f"GluGluToSUEP_mS{mS:.3f}_mPhi{mPhi:.3f}_T{T:.3f}_mode{mode}_13TeV_2018"
                    if sample not in plots:
                        continue
                    name = f"mS{mS}_mPhi{mPhi}_T{T}_{mode}"

                    if (
                        plots[sample]["SR_high_temp_tight"][7j::sum].value
                        > plots[sample]["SR_low_temp_tight"][7j::sum].value
                    ):
                        color1 = Fore.GREEN
                        color2 = Fore.RED
                    elif (
                        plots[sample]["SR_high_temp_tight"][7j::sum].value
                        < plots[sample]["SR_low_temp_tight"][7j::sum].value
                    ):
                        color1 = Fore.RED
                        color2 = Fore.GREEN
                    else:
                        color1 = Fore.WHITE
                        color2 = Fore.WHITE

                    yield_table.append(
                        [
                            name,
                            color1
                            + smart_rounding(
                                plots[sample]["SR_high_temp_tight"][7j::sum].value,
                                math.sqrt(
                                    plots[sample]["SR_high_temp_tight"][
                                        7j::sum
                                    ].variance
                                ),
                            )
                            + Style.RESET_ALL,
                            color2
                            + smart_rounding(
                                plots[sample]["SR_low_temp_tight"][7j::sum].value,
                                math.sqrt(
                                    plots[sample]["SR_low_temp_tight"][7j::sum].variance
                                ),
                            )
                            + Style.RESET_ALL,
                        ]
                    )
    print()
    print(tabulate(yield_table, headers=header, tablefmt=tablefmt))
    print()
