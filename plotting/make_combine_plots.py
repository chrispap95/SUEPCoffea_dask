import argparse
import logging
import math
import shutil

import hist
import plot_utils

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Dec2024",
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
        "--make_combined_regions",
        action="store_true",
        help="Make combined regions: CR and SUEP(CR+SR).",
    )
    parser.add_argument(
        "--inject_signal",
        type=int,
        default=0,
        help="Inject signal in the data_obs plot for the SR. This is the number (integer) of signal events to inject.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/combine_stuff/Jan2025/CMSSW_11_3_4/src/auxiliaries/input/",
        help="Destination directory for the ROOT files.",
    )
    return parser.parse_args()


def combine_CR_regions(plots):
    for sample in plots:
        systematic_vars = {""}
        for region in plots[sample]:
            if "SR_high_temp_tight_" in region:
                systematic_vars.add(region.replace(f"SR_high_temp_tight", ""))

        systematic_vars = list(systematic_vars)

        for syst in systematic_vars:
            CR_cb_plot = (
                plots[sample]["CR_cb"]
                if f"CR_cb{syst}" not in plots[sample]
                else plots[sample][f"CR_cb{syst}"]
            )
            CR_prompt_plot = (
                plots[sample]["CR_prompt"]
                if f"CR_prompt{syst}" not in plots[sample]
                else plots[sample][f"CR_prompt{syst}"]
            )
            h_comb = hist.Hist.new.StrCat(
                [
                    "CR_QCD bin 1",
                    "CR_QCD bin 2",
                    "CR_QCD bin 3",
                    "CR_QCD bin 4",
                    "CR_DY bin 1",
                ],
                name=f"CR{syst}",
            ).Weight()
            h_comb["CR_QCD bin 1"] = CR_cb_plot[1j]
            h_comb["CR_QCD bin 2"] = CR_cb_plot[2j]
            h_comb["CR_QCD bin 3"] = CR_cb_plot[3j]
            h_comb["CR_QCD bin 4"] = CR_cb_plot[4j]
            h_comb["CR_DY bin 1"] = CR_prompt_plot[2j]
            plots[sample][f"CR{syst}"] = h_comb.copy()


def combine_all_regions(plots, sr="high_temp"):
    for sample in plots:
        systematic_vars = {""}
        suffix = ""
        for region in plots[sample]:
            if "extrapolation" in region:
                suffix = "_extrapolation"
                break
        for region in plots[sample]:
            if f"SR_{sr}_tight{suffix}_" in region:
                systematic_vars.add(region.replace(f"SR_{sr}_tight{suffix}", ""))

        systematic_vars = list(systematic_vars)

        for syst in systematic_vars:
            CR_cb_plot = (
                plots[sample]["CR_cb"]
                if f"CR_cb{syst}" not in plots[sample]
                else plots[sample][f"CR_cb{syst}"]
            )
            CR_prompt_plot = (
                plots[sample]["CR_prompt"]
                if f"CR_prompt{syst}" not in plots[sample]
                else plots[sample][f"CR_prompt{syst}"]
            )
            SR_high_temp_plot = (
                plots[sample][f"SR_{sr}_tight{suffix}"]
                if f"SR_{sr}_tight{suffix}{syst}" not in plots[sample]
                else plots[sample][f"SR_{sr}_tight{suffix}{syst}"]
            )
            h_comb = hist.Hist.new.StrCat(
                [
                    "CR_QCD bin 1",
                    "CR_QCD bin 2",
                    "CR_QCD bin 3",
                    "CR_QCD bin 4",
                    "CR_DY bin 1",
                    f"SR_{sr} bin 1",
                ],
                name=f"SUEP_{sr}{suffix}{syst}",
            ).Weight()
            h_comb["CR_QCD bin 1"] = CR_cb_plot[1j]
            h_comb["CR_QCD bin 2"] = CR_cb_plot[2j]
            h_comb["CR_QCD bin 3"] = CR_cb_plot[3j]
            h_comb["CR_QCD bin 4"] = CR_cb_plot[4j]
            h_comb["CR_DY bin 1"] = CR_prompt_plot[2j]
            h_comb[f"SR_{sr} bin 1"] = SR_high_temp_plot[7j]
            plots[sample][f"SUEP_{sr}{suffix}{syst}"] = h_comb.copy()


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

    qcd_extrapolation = plot_utils.Extrapolation(plots["QCD_Pt_MuEnrichedPt5_2018"])
    qcd_extrapolation.fit_syst_variations(slice_hists=slice_hists, verbose=False)

    # DY extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "SR_low_temp_loose": slice(4j, None),
        "SR_low_temp_tight": slice(3j, None),
        "SR_high_temp_loose": slice(4j, None),
        "SR_high_temp_tight": slice(3j, None),
    }
    dy_extrapolation = plot_utils.Extrapolation(plots["DY_2018"])
    dy_extrapolation.fit_syst_variations(slice_hists=slice_hists, verbose=False)
    print("Done!", flush=True)

    if args.make_combined_regions:
        print("Creating combined regions...", end=" ", flush=True)
        combine_CR_regions(plots)
        combine_all_regions(plots)
        print("Done!", flush=True)

    print("Converting to ROOT...", end=" ", flush=True)

    # Prepare plots for export
    plots_for_export = {}

    # Add signal
    signal_models = [model for model in plots if "SUEP" in model]
    # For now just do the mPhi8.000_T32.000_modeleptonic
    signal_models = [
        model for model in signal_models if "mPhi8.000_T32.000_modeleptonic" in model
    ]
    for model in signal_models:
        plots_for_export[model] = plot_utils.convert_to_root(
            model, plots[model], do_syst=True
        )

    # QCD bkg
    plots_for_export["QCD_13TeV_2018"] = plot_utils.convert_to_root(
        "QCD_Pt_MuEnrichedPt5_2018",
        plots["QCD_Pt_MuEnrichedPt5_2018"],
        extrapolation=True,
        do_syst=True,
    )

    # DY bkg
    plots_for_export["DY_13TeV_2018"] = plot_utils.convert_to_root(
        "DY_2018", plots["DY_2018"], extrapolation=True, do_syst=True
    )

    # TT bkg
    plots_for_export["TT_13TeV_2018"] = plot_utils.convert_to_root(
        "TT_powheg_2018", plots["TT_powheg_2018"], do_syst=True
    )

    # ST bkg
    plots_for_export["ST_13TeV_2018"] = plot_utils.convert_to_root(
        "ST_NLO_2018", plots["ST_NLO_2018"], do_syst=True
    )

    # VV+VVV bkg
    plots_for_export["Multiboson_13TeV_2018"] = plot_utils.convert_to_root(
        "VV+VVV_2018", plots["VV+VVV_2018"], do_syst=True
    )

    # Data
    if args.data:
        plots_for_export["data_obs_13TeV_2018"] = plot_utils.convert_to_root(
            "DoubleMuon_2018", plots["DoubleMuon_2018"]
        )
    print("Done!", flush=True)

    # Export histograms to ROOT files
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
