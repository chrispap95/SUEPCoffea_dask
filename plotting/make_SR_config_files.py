import argparse

import plot_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Apr2025",
        help="Tag to identify the analysis.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/combine_stuff/Jan2025/CMSSW_11_3_4/src/auxiliaries/input/",
        help="Destination folder for the config files.",
    )
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots = plot_utils.loader(tag=f"{args.tag}_SRs", load_data=False)
    print("Done!", flush=True)

    # Let's build the SUEP model names manually so that they are in the order we want
    masses_S = [125, 200, 300, 400, 500, 600, 700, 800, 1000]
    masses_phi = [1, 1.4, 2, 3, 4, 6, 8]
    T_over_mPhi = [0.25, 0.5, 1, 2, 4]
    modes = ["leptonic", "hadronic"]

    models_for_high_temp = []
    models_for_low_temp = []
    for mPhi in masses_phi:
        for r in T_over_mPhi:
            T = r * mPhi
            for mode in modes:
                for mS in masses_S:
                    sample = f"GluGluToSUEP_mS{mS:.3f}_mPhi{mPhi:.3f}_T{T:.3f}_mode{mode}_13TeV_2018"
                    if sample not in plots:
                        continue

                    y_high_temp = plots[sample]["SR_high_temp_tight"][7j::sum].value
                    y_low_temp = plots[sample]["SR_low_temp_tight"][7j::sum].value
                    if (y_high_temp == 0) and (y_low_temp == 0):
                        continue
                    if y_high_temp >= y_low_temp:
                        models_for_high_temp.append(f"{sample}\n")
                    elif y_high_temp < y_low_temp:
                        models_for_low_temp.append(f"{sample}\n")

    with open(f"{args.dest}/models_for_high_temp.txt", "w") as f:
        f.writelines(models_for_high_temp)
    with open(f"{args.dest}/models_for_low_temp.txt", "w") as f:
        f.writelines(models_for_low_temp)
    print(f"Models for high temp: {len(models_for_high_temp)}")
    print(f"Models for low temp: {len(models_for_low_temp)}")
    print("Done!")
