import argparse

import plot_utils
from rich.progress import track  # type: ignore[import]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Feb2026",
        help="Tag to identify the analysis.",
    )
    parser.add_argument(
        "--year",
        type=str,
        nargs="*",
        default=["2016", "2017", "2018", "2022", "2022EE", "2023", "2023BPix"],
        help="Year of the data. Default is run all years. Can be a single year or multiple years.",
    )
    def_out_path = "/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/combine_stuff"
    parser.add_argument(
        "--dest",
        type=str,
        default=f"{def_out_path}/Mar2026/CMSSW_14_1_0_pre4/src/auxiliaries/input/",
        help="Destination folder for the config files.",
    )
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()

    if len(args.year) == 0:
        raise ValueError("At least one year must be specified.")

    # Load plots and merge them
    plots = {}
    for year in track(args.year, description="Loading plots"):
        plots = plots | plot_utils.loader(
            tag=f"{args.tag}_{year}_SRs",
            era=year,
            load_data=False,
        )

    # Let's build the SUEP model names manually so that they are in the order we want
    masses_S = [125, 200, 300, 400, 500, 600, 700, 800, 1000]
    masses_phi = [1, 1.4, 2, 3, 4, 6, 8]
    T_over_mPhi = [0.25, 0.5, 1, 2, 4]
    modes = ["leptonic", "hadronic"]

    com_energy = lambda year: "13TeV" if year.startswith("201") else "13p6TeV"

    SR_high_temp_yields = {}
    SR_low_temp_yields = {}

    for mPhi in masses_phi:
        for r in T_over_mPhi:
            T = r * mPhi
            for mode in modes:
                for mS in masses_S:
                    scan_point = f"mS{mS:.3f}_mPhi{mPhi:.3f}_T{T:.3f}_mode{mode}"
                    sample = f"GluGluToSUEP_{scan_point}"
                    SR_high_temp_yields[sample] = 0
                    SR_low_temp_yields[sample] = 0
                    for year in args.year:
                        sample_year = f"{sample}_{com_energy(year)}_{year}"
                        if sample_year not in plots:
                            continue
                        SR_high_temp_yields[sample] += plots[sample_year][
                            "SR_high_temp_tight"
                        ][7j::sum].value
                        SR_low_temp_yields[sample] += plots[sample_year][
                            "SR_low_temp_tight"
                        ][7j::sum].value

    models_for_high_temp = []
    models_for_low_temp = []
    n_skipped_models = 0
    for sample in SR_high_temp_yields.keys():
        if (SR_high_temp_yields[sample] == 0) and (SR_low_temp_yields[sample] == 0):
            n_skipped_models += 1
            continue
        if SR_high_temp_yields[sample] >= SR_low_temp_yields[sample]:
            models_for_high_temp.append(f"{sample}\n")
        elif SR_high_temp_yields[sample] < SR_low_temp_yields[sample]:
            models_for_low_temp.append(f"{sample}\n")

    with open(f"{args.dest}/models_for_high_temp.txt", "w") as f:
        f.writelines(models_for_high_temp)
    with open(f"{args.dest}/models_for_low_temp.txt", "w") as f:
        f.writelines(models_for_low_temp)
    print(f"Skipped {n_skipped_models} models with no yield in any year.")
    print(f"Models for high temp: {len(models_for_high_temp)}")
    print(f"Models for low temp: {len(models_for_low_temp)}")
    print("Done!")
