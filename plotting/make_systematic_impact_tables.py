import argparse
import warnings

import plot_utils
from tabulate import tabulate  # type: ignore[import]

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Feb2025",
        help="Tag to identify the analysis",
    )
    return parser.parse_args()


def print_table(plots, region, samples, tablefmt):
    systematics_to_processes_map = {
        "PUReweight": ["SUEP", "QCD", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "L1PreFire": ["SUEP", "QCD", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "MuonSF": ["SUEP", "QCD", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "LHEScaleMuR": ["DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "LHEScaleMuF": ["DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "LHEPdf": ["DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "ISR": ["SUEP", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "FSR": ["SUEP", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
    }

    systematics = {
        "PUReweight": [],
        "L1PreFire": [],
        "MuonSF": [],
        "LHEScaleMuR": [],
        "LHEScaleMuF": [],
        "LHEPdf": [],
        "ISR": [],
        "FSR": [],
    }

    header = ["Syst. Unc."]
    for sample, name in samples:
        suffix = ""
        if "DY" in sample or "QCD" in sample:
            suffix = "_extrapolation"
        header.append(name)
        for syst in systematics:
            if "SUEP" not in sample and name not in systematics_to_processes_map[syst]:
                systematics[syst].append("-")
                continue
            if "SUEP" in sample and "SUEP" not in systematics_to_processes_map[syst]:
                systematics[syst].append("-")
                continue
            if f"{region}{suffix}_{syst}Up" not in plots[sample]:
                systematics[syst].append("-")
                continue
            nominal = plots[sample][f"{region}{suffix}"][7j::sum].value
            syst_up = plots[sample][f"{region}{suffix}_{syst}Up"][7j::sum].value
            syst_down = plots[sample][f"{region}{suffix}_{syst}Down"][7j::sum].value

            # print(f"{region} {sample} {suffix} {syst}: {nominal} {syst_up} {syst_down}")

            if nominal == 0:
                systematics[syst].append("-")
                continue
            else:
                impact_up = (syst_up - nominal) / nominal
                impact_down = (syst_down - nominal) / nominal
            impact = max(abs(impact_up), abs(impact_down))
            systematics[syst].append(f"{100*impact:.1f}")

    syst_table = [[key] + value for key, value in systematics.items()]

    print("\nRegion:", region, "\n")
    print(tabulate(syst_table, headers=header, tablefmt=tablefmt))
    print()


if "__main__" == __name__:
    args = parse_args()

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots = plot_utils.loader(tag=f"{args.tag}_SRs", load_data=False)
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
    dy_extrapolation.fit_syst_variations(slice_hists=slice_hists, verbose=False)
    print("Done!", flush=True)

    sig_high_temp_samples = [
        (
            "GluGluToSUEP_mS125.000_mPhi8.000_T32.000_modeleptonic_13TeV_2018",
            "SUEP_mS125_mPhi8_T32_leptonic",
        ),
        (
            "GluGluToSUEP_mS400.000_mPhi8.000_T32.000_modeleptonic_13TeV_2018",
            "SUEP_mS400_mPhi8_T32_leptonic",
        ),
        (
            "GluGluToSUEP_mS1000.000_mPhi8.000_T32.000_modeleptonic_13TeV_2018",
            "SUEP_mS1000_mPhi8_T32_leptonic",
        ),
        (
            "GluGluToSUEP_mS125.000_mPhi8.000_T16.000_modeleptonic_13TeV_2018",
            "SUEP_mS125_mPhi8_T16_leptonic",
        ),
        (
            "GluGluToSUEP_mS125.000_mPhi4.000_T4.000_modeleptonic_13TeV_2018",
            "SUEP_mS125_mPhi4_T4_leptonic",
        ),
    ]
    sig_low_temp_samples = [
        (
            "GluGluToSUEP_mS400.000_mPhi4.000_T1.000_modeleptonic_13TeV_2018",
            "SUEP_mS400_mPhi4_T1_leptonic",
        ),
        (
            "GluGluToSUEP_mS400.000_mPhi4.000_T1.000_modehadronic_13TeV_2018",
            "SUEP_mS400_mPhi4_T1_leptonic",
        ),
        (
            "GluGluToSUEP_mS600.000_mPhi4.000_T1.000_modeleptonic_13TeV_2018",
            "SUEP_mS600_mPhi4_T1_leptonic",
        ),
        (
            "GluGluToSUEP_mS600.000_mPhi4.000_T1.000_modehadronic_13TeV_2018",
            "SUEP_mS600_mPhi4_T1_leptonic",
        ),
    ]
    bkg_samples = [
        ("QCD_Pt_MuEnrichedPt5_2018", "QCD"),
        ("DY_2018", "DY"),
    ]

    # tablefmt = "plain"
    tablefmt = "latex_raw"

    print_table(
        plots, "SR_high_temp_tight", sig_high_temp_samples + bkg_samples, tablefmt
    )
    print_table(
        plots, "SR_low_temp_tight", sig_low_temp_samples + bkg_samples, tablefmt
    )
