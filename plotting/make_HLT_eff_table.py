import argparse
import math

import numpy as np
import plot_utils
import ROOT  # type: ignore[import]
import uproot
from tabulate import tabulate  # type: ignore[import]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="HLT_eff_Feb2026",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--year",
        type=str,
        nargs="*",
        default=["2016", "2017", "2018", "2022", "2022EE", "2023", "2023BPix"],
        help="Year of the data. Default is all years. Can be a single year or multiple years.",
    )
    parser.add_argument(
        "--normalize533data",
        action="store_true",
        help="If set, normalize the data histograms for HLT_TripleMu_5_3_3.",
    )
    return parser.parse_args()


trigger_paths = {
    "2022": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        "paths": [
            "HLT_TripleMu_5_3_3_Mass3p8_DZ",
            "HLT_TripleMu_10_5_5_DZ",
            "HLT_TripleMu_12_10_5",
        ],
    },
    "2022EE": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        "paths": [
            "HLT_TripleMu_5_3_3_Mass3p8_DZ",
            "HLT_TripleMu_10_5_5_DZ",
            "HLT_TripleMu_12_10_5",
        ],
    },
    "2023": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        "paths": [
            "HLT_TripleMu_5_3_3_Mass3p8_DZ",
            "HLT_TripleMu_10_5_5_DZ",
            "HLT_TripleMu_12_10_5",
        ],
    },
    "2023BPix": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        "paths": [
            "HLT_TripleMu_5_3_3_Mass3p8_DZ",
            "HLT_TripleMu_10_5_5_DZ",
            "HLT_TripleMu_12_10_5",
        ],
    },
    "2018": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        "paths": [
            "HLT_TripleMu_5_3_3_Mass3p8_DZ",
            "HLT_TripleMu_10_5_5_DZ",
            "HLT_TripleMu_12_10_5",
        ],
    },
    "2017": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        "paths": [
            "HLT_TripleMu_5_3_3_Mass3p8to60_DZ",
            "HLT_TripleMu_10_5_5_DZ",
            "HLT_TripleMu_12_10_5",
        ],
    },
    "2016": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
        "paths": [
            "HLT_TripleMu_5_3_3",
            "HLT_TripleMu_12_10_5",
        ],
    },
    "2016APV": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
        "paths": [
            "HLT_TripleMu_5_3_3",
            "HLT_TripleMu_12_10_5",
        ],
    },
}


def calculate_efficiency(
    hlt_path: str,
    h_data_NUM: ROOT.TH1,
    h_data_DEN: ROOT.TH1,
    h_mc_NUM: ROOT.TH1,
    h_mc_DEN: ROOT.TH1,
) -> tuple[float, float]:
    h_eff_mc = ROOT.TEfficiency(h_mc_NUM, h_mc_DEN)
    h_eff_data = ROOT.TEfficiency(h_data_NUM, h_data_DEN)
    h_ratio = h_mc_NUM.Clone("h_ratio")
    h_ratio.Reset()
    for i in range(1, h_mc_NUM.GetNbinsX() + 1):
        eff_mc = h_eff_mc.GetEfficiency(i)
        eff_dt = h_eff_data.GetEfficiency(i)
        err_mc = 0.5 * (
            h_eff_mc.GetEfficiencyErrorLow(i) + h_eff_mc.GetEfficiencyErrorUp(i)
        )
        err_dt = 0.5 * (
            h_eff_data.GetEfficiencyErrorLow(i) + h_eff_data.GetEfficiencyErrorUp(i)
        )
        if eff_mc > 0:
            val = eff_dt / eff_mc
            err = math.sqrt((err_dt / eff_mc) ** 2 + (eff_dt * err_mc / eff_mc**2) ** 2)
            h_ratio.SetBinContent(i, val)
            h_ratio.SetBinError(i, err)
        else:
            h_ratio.SetBinContent(i, 0.0)
            h_ratio.SetBinError(i, 0.0)

    pt_min = 3 if "5_3_3" in hlt_path else 5
    pol0_ratio = ROOT.TF1("pol0_ratio", "pol0", pt_min, 20)
    h_ratio.Fit(pol0_ratio, "R", "", pt_min, 20)

    return pol0_ratio.GetParameter(0), pol0_ratio.GetParError(0)


def weighted_average(
    values: list[float], uncertainties: list[float]
) -> tuple[float, float, float]:
    weights = 1.0 / (np.array(uncertainties) ** 2)
    sum_wx = np.sum(weights * np.array(values))
    sum_w = np.sum(weights)
    global_sf = sum_wx / sum_w
    global_sf_unc = math.sqrt(1.0 / sum_w)
    sf_spread = np.array(values) - global_sf
    max_deviation = np.max(np.abs(sf_spread))
    return global_sf, global_sf_unc, max_deviation  # type: ignore[return-value]


if __name__ == "__main__":
    args = parse_args()

    rebin = 2j
    per_year_global_SFs = {}

    header = ["year", "SF_5_3_3", "SF_10_5_5", "SF_12_10_5", "global2_SF", "global3_SF"]
    table = []

    for year in args.year:
        plots = plot_utils.loader(
            tag=f"{args.tag}_{year}",
            era=year,
            custom_lumi=None,
            load_data=True,
        )

        qcd_sample = f"QCD_Pt_MuEnrichedPt5_{year}"

        SFs_2 = []
        unc_SFs_2 = []
        SFs_3 = []
        unc_SFs_3 = []

        row = [year]

        for hlt_path in trigger_paths[year]["paths"]:
            h_mc_NUM = uproot.to_writable(
                plots[qcd_sample][f"NUM_{hlt_path}"][::rebin]
            ).to_pyroot()  # type: ignore[attr-defined]
            h_mc_DEN = uproot.to_writable(
                plots[qcd_sample][f"DEN_{hlt_path}"][::rebin]
            ).to_pyroot()  # type: ignore[attr-defined]
            h_data_NUM = uproot.to_writable(
                plots[f"Data_{year}"][f"NUM_{hlt_path}"][::rebin]
            ).to_pyroot()  # type: ignore[attr-defined]
            h_data_DEN = uproot.to_writable(
                plots[f"Data_{year}"][f"DEN_{hlt_path}"][::rebin]
            ).to_pyroot()  # type: ignore[attr-defined]
            if (
                args.normalize533data
                and hlt_path == "HLT_TripleMu_5_3_3"
                and year == "2016"
            ):
                ratio = np.divide(
                    plots[f"Data_{year}"][f"NUM_{hlt_path}"][::rebin].values(),
                    plots[f"Data_{year}"][f"DEN_{hlt_path}"][::rebin].values(),
                    out=np.zeros_like(
                        plots[f"Data_{year}"][f"NUM_{hlt_path}"][::rebin].values()
                    ),
                    where=plots[f"Data_{year}"][f"DEN_{hlt_path}"][::rebin].values()
                    != 0,
                )
                h_data_DEN = uproot.to_writable(
                    plots[f"Data_{year}"][f"DEN_{hlt_path}"][::rebin]
                    * max(ratio)
                    * 1.000001  # needed to avoid rounding errors
                ).to_pyroot()  # type: ignore[attr-defined]
            h_mc_NUM.Sumw2()
            h_mc_NUM.Rebuild()
            h_mc_NUM.ResetStats()
            h_mc_NUM.ComputeIntegral()
            h_mc_DEN.Sumw2()
            h_mc_DEN.Rebuild()
            h_mc_DEN.ResetStats()
            h_mc_DEN.ComputeIntegral()

            sf, unc_sf = calculate_efficiency(
                hlt_path,
                h_data_NUM.Clone(),
                h_data_DEN.Clone(),
                h_mc_NUM.Clone(),
                h_mc_DEN.Clone(),
            )
            if not "5_3_3" in hlt_path:
                SFs_2.append(sf)
                unc_SFs_2.append(unc_sf)
            SFs_3.append(sf)
            unc_SFs_3.append(unc_sf)

            row.append(f"{sf:.3f} ± {unc_sf:.3f}")

        global_sf_2, global_sf_unc_2, max_deviation_2 = weighted_average(
            SFs_2, unc_SFs_2
        )
        row.append(f"{global_sf_2:.3f} ± {global_sf_unc_2:.3f} ± {max_deviation_2:.3f}")
        global_sf_3, global_sf_unc_3, max_deviation_3 = weighted_average(
            SFs_3, unc_SFs_3
        )
        row.append(f"{global_sf_3:.3f} ± {global_sf_unc_3:.3f} ± {max_deviation_3:.3f}")

        if year == "2016":
            row.insert(2, "-")
        table.append(row)

    print()
    print(tabulate(table, headers=header, tablefmt="simple"))
    print()
