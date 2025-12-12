import argparse
import logging

import plot_utils
from rich.progress import track  # type: ignore[import]
from tabulate import tabulate  # type: ignore[import]

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
        default=["2018"],
        help="Year of the data. Default is 2018. Can be a single year or multiple years.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print the table in LaTeX format",
    )
    return parser.parse_args()


def print_table(plots, region, year, samples, tablefmt):
    systematics_to_processes_map = {
        "PUReweight": ["SUEP", "QCD", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "L1PreFire": ["SUEP", "QCD", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "MuonSF": ["SUEP", "QCD", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "TrkEff": ["SUEP", "QCD", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "TrigSF": ["SUEP", "QCD", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "LHEScaleMuR": ["SUEP", "QCD", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "LHEScaleMuF": ["SUEP", "QCD", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "LHEPdf": ["SUEP", "QCD", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "ISR": ["SUEP", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
        "FSR": ["SUEP", "DY", "TT", "VV+VVV", "WJets", "ST", "Higgs"],
    }

    systematics = {
        "PUReweight": [],
        "MuonSF": [],
        "TrkEff": [],
        "TrigSF": [],
        "LHEScaleMuR": [],
        "LHEScaleMuF": [],
        "LHEPdf": [],
        "ISR": [],
        "FSR": [],
    }
    if year.startswith("201") or year == "Run2":
        systematics["L1PreFire"] = []

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
            if f"{region}{suffix}_{syst}Up" not in plots[f"{sample}_{year}"]:
                systematics[syst].append("-")
                continue
            nominal = plots[f"{sample}_{year}"][f"{region}{suffix}"][7j::sum].value
            syst_up = plots[f"{sample}_{year}"][f"{region}{suffix}_{syst}Up"][
                7j::sum
            ].value
            syst_down = plots[f"{sample}_{year}"][f"{region}{suffix}_{syst}Down"][
                7j::sum
            ].value

            # print(f"{region} {sample} {suffix} {syst}: {nominal} {syst_up} {syst_down}")

            if nominal == 0:
                systematics[syst].append("-")
                continue
            else:
                impact_up = (syst_up - nominal) / nominal
                impact_down = (syst_down - nominal) / nominal
            impact = max(abs(impact_up), abs(impact_down))
            sign = "" if abs(impact_up) > abs(impact_down) else "-"
            systematics[syst].append(f"{sign}{100*impact:.1f}")

    syst_table = [[key] + value for key, value in systematics.items()]

    # There is no TrkEff for high temp region
    if "high_temp" in region:
        for i in range(len(syst_table)):
            if syst_table[i][0] == "TrkEff":
                syst_table.remove(syst_table[i])
                break

    print("\nRegion:", region, "\n")
    print(tabulate(syst_table, headers=header, tablefmt=tablefmt))
    print()


if "__main__" == __name__:
    args = parse_args()

    # Load plots and merge them
    years_to_load = args.year
    if "Run2" in args.year:
        # Commenting out 2016APV for now
        # years_to_load = ["2016APV", "2016", "2017", "2018"]
        years_to_load = ["2016", "2017", "2018"]
    if "Run3" in args.year:
        years_to_load = ["2022", "2022EE", "2023", "2023BPix"]
    if "Run2" in args.year and "Run3" in args.year:
        years_to_load = [
            # Commenting out 2016APV for now
            # "2016APV",
            "2016",
            "2017",
            "2018",
            "2022",
            "2022EE",
            "2023",
            "2023BPix",
        ]
    plots = {}
    for year in track(years_to_load, description="Loading plots"):
        plots = plots | plot_utils.loader(
            tag=f"{args.tag}_{year}_SRs",
            era=year,
            load_data=False,
        )

    plots = plot_utils.make_lhepdf_systematic(plots)

    for year in track(years_to_load, description="Fitting and extrapolations"):
        # QCD extrapolation
        # Slice the first bin out where needed for fit stability
        slice_hists = {
            "SR_low_temp_loose": slice(4j, None),
            "SR_low_temp_tight": slice(3j, None),
            "SR_high_temp_loose": slice(4j, None),
            "SR_high_temp_tight": slice(3j, None),
        }
        qcd_extrapolation = plot_utils.Extrapolation(
            plots[f"QCD_Pt_MuEnrichedPt5_{year}"], uncertainty_scheme="full"
        )
        qcd_extrapolation.fit_syst_variations(slice_hists=slice_hists, verbose=False)

        # DY extrapolation
        # Slice the first bin out where needed for fit stability
        slice_hists = {
            "SR_low_temp_loose": slice(4j, None),
            "SR_low_temp_tight": slice(3j, None),
            "SR_high_temp_loose": slice(4j, None),
            "SR_high_temp_tight": slice(4j, None),
        }
        dy_extrapolation = plot_utils.Extrapolation(
            plots[f"DY_{year}"], uncertainty_scheme="full"
        )
        dy_extrapolation.fit_syst_variations(slice_hists=slice_hists, verbose=False)

    if "Run2" in args.year:
        run2_plots = plot_utils.merge_runs(plots, "Run2", args)
        plots = plots | run2_plots
    if "Run3" in args.year:
        run3_plots = plot_utils.merge_runs(plots, "Run3", args)
        plots = plots | run3_plots

    for year in track(args.year, description="Making tables"):
        com_energy = (
            "13TeV" if year.startswith("201") or (year == "Run2") else "13p6TeV"
        )
        sig_high_temp_samples = [
            (
                f"GluGluToSUEP_mS125.000_mPhi8.000_T32.000_modeleptonic_{com_energy}",
                "SUEP_mS125_mPhi8_T32_leptonic",
            ),
            (
                f"GluGluToSUEP_mS400.000_mPhi8.000_T32.000_modeleptonic_{com_energy}",
                "SUEP_mS400_mPhi8_T32_leptonic",
            ),
            (
                f"GluGluToSUEP_mS1000.000_mPhi8.000_T32.000_modeleptonic_{com_energy}",
                "SUEP_mS1000_mPhi8_T32_leptonic",
            ),
            (
                f"GluGluToSUEP_mS125.000_mPhi8.000_T16.000_modeleptonic_{com_energy}",
                "SUEP_mS125_mPhi8_T16_leptonic",
            ),
            (
                f"GluGluToSUEP_mS125.000_mPhi4.000_T4.000_modeleptonic_{com_energy}",
                "SUEP_mS125_mPhi4_T4_leptonic",
            ),
        ]
        sig_low_temp_samples = [
            (
                f"GluGluToSUEP_mS400.000_mPhi4.000_T1.000_modeleptonic_{com_energy}",
                "SUEP_mS400_mPhi4_T1_leptonic",
            ),
            (
                f"GluGluToSUEP_mS400.000_mPhi4.000_T1.000_modehadronic_{com_energy}",
                "SUEP_mS400_mPhi4_T1_leptonic",
            ),
            (
                f"GluGluToSUEP_mS600.000_mPhi4.000_T1.000_modeleptonic_{com_energy}",
                "SUEP_mS600_mPhi4_T1_leptonic",
            ),
            (
                f"GluGluToSUEP_mS600.000_mPhi4.000_T1.000_modehadronic_{com_energy}",
                "SUEP_mS600_mPhi4_T1_leptonic",
            ),
        ]
        bkg_samples = [
            ("QCD_Pt_MuEnrichedPt5", "QCD"),
            ("DY", "DY"),
        ]

        tablefmt = "simple"
        if args.latex:
            tablefmt = "latex_raw"

        print("\nYear:", year)
        print_table(
            plots,
            "SR_high_temp_tight",
            year,
            sig_high_temp_samples + bkg_samples,
            tablefmt,
        )
        print_table(
            plots,
            "SR_low_temp_tight",
            year,
            sig_low_temp_samples + bkg_samples,
            tablefmt,
        )
        print("\n" + "=" * 80 + "\n")
