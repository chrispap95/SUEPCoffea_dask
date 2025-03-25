import argparse
import logging

import plot_utils
from tabulate import tabulate  # type: ignore[import]

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
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots_CR = plot_utils.loader(
        tag=f"{args.tag}_CR", custom_lumi=args.lumi, load_data=False
    )
    plots_SR = plot_utils.loader(
        tag=f"{args.tag}_SRs", custom_lumi=args.lumi, load_data=False
    )
    plots = {}
    for dataset in plots_CR:
        # Note: need to fix this to be mergeable even when data for SR is missing! (blinded...)
        # This merges two dicts!
        plots[dataset] = plots_CR[dataset] | plots_SR[dataset]
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
    qcd_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

    # DY extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "SR_low_temp_loose": slice(4j, None),
        "SR_low_temp_tight": slice(3j, None),
        "SR_high_temp_loose": slice(4j, None),
        "SR_high_temp_tight": slice(3j, None),
    }
    dy_extrapolation = plot_utils.Extrapolation(
        plots["DY_2018"], uncertainty_scheme="full"
    )
    dy_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)
    print("Done!", flush=True)

    mc_processes = [
        ("Higgs_2018", "Higgs"),
        ("TTV_2018", "TTV"),
        ("ST_NLO_2018", "ST"),
        ("WJets_2018", "WJets"),
        ("VV+VVV_2018", "VV+VVV"),
        ("TT_powheg_2018", "TT"),
        ("DY_2018", f"DY+extr."),
        ("QCD_Pt_MuEnrichedPt5_2018", f"QCD+extr."),
    ]
    regions = ["CR_cb", "CR_prompt", "SR_high_temp_tight", "SR_low_temp_tight"]

    # Calculate total bkg
    plots["total_bkg_2018"] = {}
    for process, proc_label in mc_processes:
        for region in regions:
            suffix = ""
            if ("DY" in process or "QCD" in process) and "SR" in region:
                suffix = "_extrapolation"
            if region not in plots["total_bkg_2018"]:
                plots["total_bkg_2018"][region] = plots[process][region + suffix].copy()
            else:
                plots["total_bkg_2018"][region] += plots[process][region + suffix]

    print(plots["total_bkg_2018"]["SR_low_temp_tight"].values())

    header = [
        "process",
        "CR_QCD_bin1",
        "CR_QCD_bin2",
        "CR_QCD_bin3",
        "CR_QCD_bin4",
        "CR_DY_bin1",
        "SR_high_temp_bin1",
    ]
    table = []
    for process, proc_label in mc_processes:
        suffix = ""
        if "QCD" in process or "DY" in process:
            suffix = "_extrapolation"
        row = [proc_label]
        row.append(
            round(
                plots[process]["CR_cb"][1j].value
                / plots["total_bkg_2018"]["CR_cb"][1j].value,
                2,
            )
        )
        row.append(
            round(
                plots[process]["CR_cb"][2j].value
                / plots["total_bkg_2018"]["CR_cb"][2j].value,
                2,
            )
        )
        row.append(
            round(
                plots[process]["CR_cb"][3j].value
                / plots["total_bkg_2018"]["CR_cb"][3j].value,
                2,
            )
        )
        row.append(
            round(
                plots[process]["CR_cb"][4j].value
                / plots["total_bkg_2018"]["CR_cb"][4j].value,
                2,
            )
        )
        row.append(
            round(
                plots[process]["CR_prompt"][2j].value
                / plots["total_bkg_2018"]["CR_prompt"][2j].value,
                2,
            )
        )
        row.append(
            round(
                plots[process][f"SR_high_temp_tight{suffix}"][7j].value
                / plots["total_bkg_2018"]["SR_high_temp_tight"][7j].value,
                2,
            )
        )
        table.append(row)

    print(tabulate(table, headers=header))
