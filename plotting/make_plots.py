import argparse
import shutil

import plot_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Nov2024",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        help="Custom integrated luminosity to be used (in pb^-1). For example, use 559.322 for "
        "the single data file in filelists/data/data_Run2018A_0p6fb_1file_unskimmed.json."
        "If not provided, the luminosity will be determined automatically for the year.",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Load data",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/combine_stuff/Nov2024/CMSSW_11_3_4/src/auxiliaries/input/",
        help="Destination directory for the ROOT files",
    )
    return parser.parse_args()


if "__main__" in __name__:
    args = parse_args()

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots_CR = plot_utils.loader(
        tag=f"{args.tag}_CR", custom_lumi=args.lumi, load_data=args.data
    )
    plots_VR = plot_utils.loader(
        tag=f"{args.tag}_VR", custom_lumi=args.lumi, load_data=args.data
    )
    plots_SR_low_temp = plot_utils.loader(
        tag=f"{args.tag}_SR_low_temp", custom_lumi=args.lumi, load_data=args.data
    )
    plots_SR_high_temp = plot_utils.loader(
        tag=f"{args.tag}_SR_high_temp", custom_lumi=args.lumi, load_data=args.data
    )
    plots = {}
    for dataset in plots_CR:
        # Note: need to fix this to be mergeable even when data for SR is missing! (blinded...)
        # This merges two dicts!
        plots[dataset] = (
            plots_CR[dataset]
            | plots_VR[dataset]
            | plots_SR_low_temp[dataset]
            | plots_SR_high_temp[dataset]
        )
    print("Done!", flush=True)

    print("Fit and extrapolation...", end=" ", flush=True)
    # QCD extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "SR_low_temp_loose": slice(4j, None),
        "SR_low_temp_tight": slice(3j, None),
        "SR_high_temp_loose": slice(4j, None),
        "SR_high_temp_tight": slice(3j, None),
    }

    qcd_extrapolation = plot_utils.Extrapolation(plots["QCD_Pt_MuEnrichedPt5_2018"])
    qcd_extrapolation.extrapolate(slice_hists=slice_hists)
    # qcd_extrapolation.plot_overlay()
    # qcd_extrapolation.plot_fit("SR_low_temp")
    # qcd_extrapolation.plot_fit("SR_high_temp")

    # DY extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "SR_low_temp_loose": slice(4j, None),
        "SR_low_temp_tight": slice(3j, None),
        "SR_high_temp_loose": slice(4j, None),
        "SR_high_temp_tight": slice(3j, None),
    }
    dy_extrapolation = plot_utils.Extrapolation(plots["DY_2018"])
    dy_extrapolation.extrapolate(slice_hists=slice_hists)
    # dy_extrapolation.plot_overlay()
    # dy_extrapolation.plot_fit("SR_low_temp")
    # dy_extrapolation.plot_fit("SR_high_temp")
    print("Done!", flush=True)

    print("Converting to ROOT and exporting...", end=" ", flush=True)
    # Prepare plots for export
    plots_for_export = {}

    # Add signal
    signal_models = [model for model in plots if "SUEP" in model]
    for model in signal_models:
        plots_for_export[model] = plot_utils.convert_to_root(model, plots[model])

    # QCD bkg
    plots_for_export["QCD_13TeV_2018"] = plot_utils.convert_to_root(
        "QCD_Pt_MuEnrichedPt5_2018",
        plots["QCD_Pt_MuEnrichedPt5_2018"],
        extrapolation=True,
    )

    # DY bkg
    plots_for_export["DY_13TeV_2018"] = plot_utils.convert_to_root(
        "DY_2018", plots["DY_2018"], extrapolation=True
    )

    # Export histograms to ROOT files
    plot_utils.export_histograms_to_root(plots_for_export, "exports")

    # Copy to destination
    shutil.copytree("exports", args.dest, dirs_exist_ok=True)
    print("Done!", flush=True)
