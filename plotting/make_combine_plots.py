import argparse
import logging
import math
import shutil

import plot_utils

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Apr2025",
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
        help="Load data.",
    )
    parser.add_argument(
        "--unblind",
        action="store_true",
        help="Unblind the SRs.",
    )
    parser.add_argument(
        "--signal_scale",
        type=float,
        default=0.0001,
        help="Scale signal by this factor. Can be used to scale r value in combine. This is the inverse of the scaling of the signal strength.",
    )
    parser.add_argument(
        "--inject_signal",
        type=int,
        default=0,
        help="Inject signal in the data_obs plot for the SR. This is the number (integer) of signal events to inject.",
    )
    parser.add_argument(
        "--signal_filter",
        type=str,
        default="",
        help="Export only signal containing this string. E.g., 'mPhi8.000_T32.000'.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/combine_stuff/Apr2025/CMSSW_14_2_2/src/auxiliaries/input/",
        help="Destination directory for the ROOT files.",
    )
    return parser.parse_args()


if "__main__" in __name__:
    args = parse_args()

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots_CR = plot_utils.loader(
        tag=f"{args.tag}_CR", custom_lumi=args.lumi, load_data=args.data
    )
    plots_SR = plot_utils.loader(
        tag=f"{args.tag}_SRs", custom_lumi=args.lumi, load_data=args.data
    )

    # Make sure the SR is blinded if needed
    if not args.unblind:
        for dataset in [d for d in plots_CR if "DoubleMuon" in d]:
            plots_SR[dataset] = {}
            for region in [
                "SR_high_temp_tight",
                "SR_high_temp_loose",
                "SR_low_temp_tight",
                "SR_low_temp_loose",
            ]:
                plots_SR[dataset][region] = (
                    plots_SR["QCD_Pt_MuEnrichedPt5_2018"][region].copy().reset()
                )

    if args.inject_signal:
        for dataset in [d for d in plots_CR if "DoubleMuon" in d]:
            for region in [
                "SR_high_temp_tight",
                "SR_high_temp_loose",
                "SR_low_temp_tight",
                "SR_low_temp_loose",
            ]:
                old_content = plots_SR[dataset][region][7j]
                plots_SR[dataset][region][7j] = (
                    old_content.value + float(args.inject_signal),
                    old_content.variance + float(args.inject_signal) ** 2,
                )

    # Merge CR and SR plots into one dictionary
    plots = {}
    for dataset in plots_CR:
        plots[dataset] = plots_CR[dataset] | plots_SR[dataset]

    # Scale signal
    if not math.isclose(args.signal_scale, 1.0):
        for model in [model for model in plots if "SUEP" in model]:
            for plot in plots[model]:
                plots[model][plot] *= args.signal_scale

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

    qcd_extrapolation = plot_utils.Extrapolation(
        plots["QCD_Pt_MuEnrichedPt5_2018"], uncertainty_scheme="full"
    )
    qcd_extrapolation.fit_syst_variations(slice_hists=slice_hists, verbose=False)
    qcd_extrapolation.create_syst_variation(sample="QCD")

    # DY extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "SR_low_temp_loose": slice(4j, None),
        "SR_low_temp_tight": slice(3j, None),
        "SR_high_temp_loose": slice(4j, None),
        "SR_high_temp_tight": slice(4j, None),
    }
    dy_extrapolation = plot_utils.Extrapolation(
        plots["DY_2018"], uncertainty_scheme="full"
    )
    dy_extrapolation.fit_syst_variations(slice_hists=slice_hists, verbose=False)
    dy_extrapolation.create_syst_variation(sample="DY")
    print("Done!", flush=True)

    print("Converting to ROOT...", end=" ", flush=True)
    # Prepare plots for export
    plots_for_export = {}

    # Add signal
    signal_models = [model for model in plots if "SUEP" in model]
    signal_models = [model for model in signal_models if args.signal_filter in model]
    for model in signal_models:
        plots_for_export[model] = plot_utils.convert_to_root(
            model, plots[model], do_syst=True
        )

    # MC bkg
    mc_processes = [
        ("Higgs_2018", "Higgs"),
        ("TTV_2018", "TTV"),
        ("ST_NLO_2018", "ST"),
        ("WJets_2018", "WJets"),
        ("VV+VVV_2018", "VV+VVV"),
        ("TT_powheg_2018", "TT"),
        ("DY_2018", "DY"),
        ("QCD_Pt_MuEnrichedPt5_2018", "QCD"),
    ]
    for process, process_name in mc_processes:
        do_extrapolation = process_name in ["QCD", "DY"]
        plots_for_export[process_name + "_13TeV_2018"] = plot_utils.convert_to_root(
            process, plots[process], extrapolation=do_extrapolation, do_syst=True
        )

    # Data
    if args.data:
        plots_for_export["data_obs_13TeV_2018"] = plot_utils.convert_to_root(
            "DoubleMuon_2018", plots["DoubleMuon_2018"]
        )
    print("Done!", flush=True)

    # Export histograms to ROOT files
    print(
        "Exporting histograms to ROOT and copying to final destination...",
        end=" ",
        flush=True,
    )
    output_name = "SUEP"
    if not math.isclose(args.signal_scale, 1.0):
        output_name += f"_signal_scale{args.signal_scale}"
    if args.inject_signal:
        output_name += f"_signal_injected{args.inject_signal}"

    plot_utils.export_histograms_to_root(
        plots_for_export,
        output_path="exports",
        output_name=f"{output_name}.root",
    )

    # Copy to destination
    shutil.copytree("exports", args.dest, dirs_exist_ok=True)
    print("Done!", flush=True)
