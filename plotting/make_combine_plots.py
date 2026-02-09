import argparse
import logging
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import plot_utils
from rich.progress import track  # type: ignore[import]

# Suppress warnings from Extrapolation class
logging.getLogger().setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Dec2025",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--year",
        type=str,
        nargs="*",
        default=["2016", "2017", "2018", "2022", "2022EE", "2023", "2023BPix"],
        help="Year of the data. Default is 2018. Can be a single year or multiple years.",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        help="Custom integrated luminosity to be used (in pb^-1). For example, use 559.322 "
        "for the single data file in filelists/data/data_Run2018A_0p6fb_1file_unskimmed.json. "
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
        help="Scale signal by this factor. Can be used to scale r value in combine. "
        "This is the inverse of the scaling of the signal strength.",
    )
    parser.add_argument(
        "--inject_signal",
        type=int,
        default=0,
        help="Inject signal in the data_obs plot for the SR. "
        "This is the number (integer) of signal events to inject.",
    )
    parser.add_argument(
        "--signal_filter",
        type=str,
        default="",
        help="Export only signal containing this string. E.g., 'mPhi8.000_T32.000'.",
    )
    def_out_path = "/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/combine_stuff"
    parser.add_argument(
        "--dest",
        type=str,
        default=f"{def_out_path}/Dec2025/CMSSW_14_1_0_pre4/src/auxiliaries/input/",
        help="Destination directory for the ROOT files.",
    )
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Sanitize the plots before exporting them to ROOT. "
        "This will set all negative values to zero.",
    )
    parser.add_argument(
        "--multiproc",
        action="store_true",
        help="Use multiprocessing to convert signal plots to ROOT.",
    )
    return parser.parse_args()


if "__main__" in __name__:
    start_time = time.time()
    args = parse_args()

    if len(args.year) == 0:
        raise ValueError("At least one year must be specified.")

    # Load plots and merge them
    plots = {}
    for year in track(args.year, description="Loading plots"):
        plots_CR = plot_utils.loader(
            tag=f"{args.tag}_{year}_CR",
            era=year,
            custom_lumi=args.lumi,
            load_data=args.data,
        )
        plots_SR = plot_utils.loader(
            tag=f"{args.tag}_{year}_SRs",
            era=year,
            custom_lumi=args.lumi,
            load_data=args.data,
        )
        all_datasets = set(plots_CR.keys()) | set(plots_SR.keys())
        for dataset in list(all_datasets):
            if dataset not in plots_CR:
                plots_CR[dataset] = {}
            if dataset not in plots_SR:
                plots_SR[dataset] = {}
            plots[dataset] = plots_CR[dataset] | plots_SR[dataset]

    print("Calculating LHE PDF systematics...", end=" ", flush=True)
    plots = plot_utils.make_lhepdf_systematic(plots, cleanup=True)
    print("Done!", flush=True)

    # Sanitize plots if requested
    if args.sanitize:
        for dataset in track(plots, description="Sanitizing plots"):
            for region in plots[dataset]:
                if any(plots[dataset][region].values() < 0):
                    print(
                        f"Sanitizing negative values for dataset {dataset} in region {region}...",
                        flush=True,
                    )
                    h = plots[dataset][region]
                    for i in np.arange(len(h.values()))[h.values() < 0]:
                        h[i] = (1e-9, h[i].variance)

    # If the nominal histogram exists but not the systematics, create null systematics histograms
    print("Populating null systematics histograms...", end=" ", flush=True)
    systematics = [
        "MuonSF",
        "PUReweight",
        "ISR",
        "FSR",
        "LHEScaleMuF",
        "LHEScaleMuR",
        "LHEPdf",
        "TrigSF",
    ]
    regions = [
        "CR_cb",
        "CR_prompt",
        "SR_high_temp_loose",
        "SR_high_temp_tight",
        "SR_low_temp_loose",
        "SR_low_temp_tight",
    ]
    for dataset in plots:
        for region in regions:
            if (
                region in plots[dataset]
                and f"{region}_{systematics[0]}Down" not in plots[dataset]
            ):
                for syst in systematics:
                    plots[dataset][f"{region}_{syst}Down"] = plots[dataset][
                        region
                    ].copy()
                    plots[dataset][f"{region}_{syst}Up"] = plots[dataset][region].copy()
                if region == "SR_low_temp_loose":
                    plots[dataset]["SR_low_temp_loose_TrkEffDown"] = plots[dataset][
                        "SR_low_temp_loose"
                    ].copy()
                    plots[dataset]["SR_low_temp_loose_TrkEffUp"] = plots[dataset][
                        "SR_low_temp_loose"
                    ].copy()
    print("Done!", flush=True)

    # Make sure the SR is blinded if needed
    if not args.unblind:
        print("Blinding SRs...", end=" ", flush=True)
        for dataset in [d for d in plots if "Data" in d]:
            # plots[dataset] = {}
            for region in [
                "SR_high_temp_tight",
                "SR_high_temp_loose",
                "SR_low_temp_tight",
                "SR_low_temp_loose",
            ]:
                plots[dataset][region] = (
                    plots[f"QCD_Pt_MuEnrichedPt5_{args.year[0]}"][region].copy().reset()
                )
        print("Done!", flush=True)

    # Inject signal if requested
    if args.inject_signal:
        print("Injecting signal...", end=" ", flush=True)
        for dataset in [d for d in plots if "Data" in d]:
            for region in [
                "SR_high_temp_tight",
                "SR_high_temp_loose",
                "SR_low_temp_tight",
                "SR_low_temp_loose",
            ]:
                old_content = plots[dataset][region][7j]
                plots[dataset][region][7j] = (
                    old_content.value + float(args.inject_signal),
                    old_content.variance + float(args.inject_signal) ** 2,
                )
        print("Done!", flush=True)

    # Scale signal
    if not math.isclose(args.signal_scale, 1.0):
        for model in track(
            [model for model in plots if "SUEP" in model], description="Scaling signal"
        ):
            for plot in plots[model]:
                plots[model][plot] *= args.signal_scale

    for year in track(args.year, description="Fitting and extrapolating"):
        # QCD extrapolation
        # Slice the first bin out where needed for fit stability
        slice_hists = {
            "SR_low_temp_loose": slice(4j, None),
            "SR_low_temp_tight": slice(3j, None),
            "SR_high_temp_loose": slice(4j, None),
            "SR_high_temp_tight": slice(3j, None),
        }
        qcd_extrapolation = plot_utils.Extrapolation(
            plots["QCD_Pt_MuEnrichedPt5_" + year], uncertainty_scheme="full"
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
            plots["DY_" + year], uncertainty_scheme="full"
        )
        dy_extrapolation.fit_syst_variations(slice_hists=slice_hists, verbose=False)
        dy_extrapolation.create_syst_variation(sample="DY")

    # Prepare plots for export
    plots_for_export = {}

    # Add signal
    signal_models = [model for model in plots if "SUEP" in model]
    signal_models = [model for model in signal_models if args.signal_filter in model]

    if args.multiproc:

        def convert_model_pair(pair):
            model, histogram = pair
            import plot_utils

            return model, plot_utils.convert_to_root(model, histogram, do_syst=True)

        pairs = [(model, plots[model]) for model in signal_models]
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(convert_model_pair, pair): pair[0] for pair in pairs
            }
            for future in track(
                as_completed(futures),
                total=len(futures),
                description="Converting signal plots to ROOT",
            ):
                model, converted = future.result()
                plots_for_export[model] = converted
    else:
        for model in track(
            signal_models, description="Converting signal plots to ROOT"
        ):
            plots_for_export[model] = plot_utils.convert_to_root(
                model, plots[model], do_syst=True
            )

    # MC bkg
    mc_processes = [
        ("Higgs", "Higgs"),
        ("TTV", "TTV"),
        ("ST_NLO", "ST"),
        ("WJets", "WJets"),
        ("VV+VVV", "VV+VVV"),
        ("TT_powheg", "TT"),
        ("DY", "DY"),
        ("QCD_Pt_MuEnrichedPt5", "QCD"),
    ]
    for year in track(args.year, description="Converting bkg & data plots to ROOT"):
        com_energy = "13TeV" if year.startswith("201") else "13p6TeV"
        for process, process_name in mc_processes:
            do_extrapolation = process_name in ["QCD", "DY"]
            plots_for_export[f"{process_name}_{com_energy}_{year}"] = (
                plot_utils.convert_to_root(
                    f"{process}_{year}",
                    plots[f"{process}_{year}"],
                    extrapolation=do_extrapolation,
                    do_syst=True,
                )
            )

        # Data
        if args.data:
            plots_for_export[f"data_obs_{com_energy}_{year}"] = (
                plot_utils.convert_to_root(f"Data_{year}", plots[f"Data_{year}"])
            )

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

    # Create a fresh exports directory
    if os.path.exists("exports"):
        shutil.rmtree("exports")
    os.makedirs("exports")

    plot_utils.export_histograms_to_root(
        plots_for_export,
        output_path="exports",
        output_name=f"{output_name}.root",
        years=args.year,
    )

    # Copy to destination
    shutil.copytree("exports", args.dest, dirs_exist_ok=True)
    print("Done!", flush=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds", flush=True)
