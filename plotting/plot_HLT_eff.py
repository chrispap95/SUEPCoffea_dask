import argparse
import math
import os
import pathlib

import cmsstyle as CMS  # type: ignore[import]
import numpy as np
import plot_utils
import ROOT  # type: ignore[import]
import uproot
from tqdm import tqdm  # type: ignore[import]

CMS.setCMSStyle()
CMS.SetExtraText("Preliminary")
CMS.SetLumi("2022")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="HLT_eff_Dec2025",
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
        help="If set, normalize the data histograms for HLT_TripleMu_5_3_3 in 2016."
        "This is to account for the prescale factor that multiplies the denominator for data."
        "This gives some efficiency bins slightly above 1, thus breaking the calculation.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "HLT_effs"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'HLT_effs'}.",
    )
    return parser.parse_args()


variable = "subsubleading_muon_pt"
variable_label = "p^{3rd muon}_{T} [GeV]"
legend_position = (0.5, 0.1, 0.65, 0.2)

trigger_paths = {
    "2022": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        "paths": [
            "HLT_TripleMu_12_10_5",
            "HLT_TripleMu_10_5_5_DZ",
            "HLT_TripleMu_5_3_3_Mass3p8_DZ",
        ],
    },
    "2022EE": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        "paths": [
            "HLT_TripleMu_12_10_5",
            "HLT_TripleMu_10_5_5_DZ",
            "HLT_TripleMu_5_3_3_Mass3p8_DZ",
        ],
    },
    "2023": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        "paths": [
            "HLT_TripleMu_12_10_5",
            "HLT_TripleMu_10_5_5_DZ",
            "HLT_TripleMu_5_3_3_Mass3p8_DZ",
        ],
    },
    "2023BPix": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        "paths": [
            "HLT_TripleMu_12_10_5",
            "HLT_TripleMu_10_5_5_DZ",
            "HLT_TripleMu_5_3_3_Mass3p8_DZ",
        ],
    },
    "2018": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        "paths": [
            "HLT_TripleMu_12_10_5",
            "HLT_TripleMu_10_5_5_DZ",
            "HLT_TripleMu_5_3_3_Mass3p8_DZ",
        ],
    },
    "2017": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        "paths": [
            "HLT_TripleMu_12_10_5",
            "HLT_TripleMu_10_5_5_DZ",
            "HLT_TripleMu_5_3_3_Mass3p8to60_DZ",
        ],
    },
    "2016": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
        "paths": [
            "HLT_TripleMu_12_10_5",
            "HLT_TripleMu_5_3_3",
        ],
    },
    "2016APV": {
        "reference": "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
        "paths": [
            "HLT_TripleMu_12_10_5",
            "HLT_TripleMu_5_3_3",
        ],
    },
}


def plot_cms_header(year: str):
    y_position = 0.935
    cms_text_size = 0.06

    cms_text = ROOT.TLatex()
    cms_text.SetNDC()
    cms_text.SetTextFont(61)
    cms_text.SetTextSize(cms_text_size)
    cms_text.DrawLatex(0.16, y_position, "CMS")

    # ratio of 'CMS' and extra text size
    extraOverCmsTextSize = 0.76
    extra_text = ROOT.TLatex()
    extra_text.SetNDC()
    extra_text.SetTextFont(52)
    extra_text.SetTextSize(cms_text_size * extraOverCmsTextSize)
    extra_text.DrawLatex(0.25, y_position, f"Preliminary")

    extraOverCmsTextSize = 0.8
    lumi_text = ROOT.TLatex()
    lumi_text.SetNDC()
    lumi_text.SetTextFont(41)
    lumi_text.SetTextSize(cms_text_size * extraOverCmsTextSize)
    lumi_text.SetTextAlign(31)  # align right
    com_energy = "13 TeV" if year.startswith("201") else "13.6 TeV"
    lumi_text.DrawLatex(0.95, y_position, f"{year} ({com_energy})")


def plot_counts(
    args: argparse.Namespace,
    year: str,
    hlt_path: str,
    sample_type: str,
    h_NUM: ROOT.TH1,
    h_DEN: ROOT.TH1,
):
    # Let's plot the counts first
    canvas = CMS.cmsDiCanvas(
        canvName="counts",
        x_min=0,
        x_max=20,
        y_min=0,
        y_max=1e10,
        r_min=0,
        r_max=2,
        nameXaxis=variable_label,
        nameYaxis="probes",
        nameRatio="NUM/DEN",
        square=CMS.kSquare,
        extraSpace=0,
        iPos=11,
    )
    # --- 1) Make pad margins identical & sized for a readable ratio pad
    # (numbers are typical CMS-like; adjust once and forget)
    canvas.cd(1)
    pad1 = ROOT.gPad
    pad1.SetLeftMargin(0.16)
    pad1.SetRightMargin(0.04)
    pad1.SetTopMargin(0.08)
    pad1.SetBottomMargin(0.02)  # tiny, since we hide top x labels
    pad1.SetTicks(1, 1)

    canvas.cd(2)
    pad2 = ROOT.gPad
    pad2.SetLeftMargin(0.16)
    pad2.SetRightMargin(0.04)
    pad2.SetTopMargin(0.03)
    pad2.SetBottomMargin(0.40)  # big enough for large x labels
    pad2.SetTicks(1, 1)

    canvas.cd(1)
    h_NUM.SetLineColor(ROOT.kBlue)
    h_DEN.SetLineColor(ROOT.kBlack)
    h_NUM.Draw("E1")
    h_DEN.Draw("E1 SAME")
    legend_counts = ROOT.TLegend(0.65, 0.7, 0.85, 0.85)
    legend_counts.AddEntry(h_NUM, f"{sample_type} - numerator", "lp")
    legend_counts.AddEntry(h_DEN, f"{sample_type} - denominator", "lp")
    legend_counts.Draw()

    h_NUM.GetYaxis().SetTitle("# of probe muons")
    h_NUM.GetXaxis().SetLabelSize(0)  # hide labels on top pad
    h_NUM.GetXaxis().SetTitleSize(0)  # hide title on top pad
    h_NUM.GetXaxis().SetLabelOffset(999)  # belt-and-suspenders

    h_NUM.GetYaxis().SetRangeUser(0, h_DEN.GetMaximum() * 1.1)

    plot_cms_header(year)

    # --- Ratio plot
    canvas.cd(2)

    # Use a TH1 as frame to control axis styling precisely
    frame = h_mc_NUM.Clone("ratio_frame")
    frame.Reset()
    frame.GetXaxis().SetRangeUser(0, 20)
    frame.GetYaxis().SetRangeUser(0.0, 1.1)
    frame.SetTitle(f";{variable_label};NUM/DEN")

    # 3) Scale text sizes up on the small ratio pad (ROOT sizes are relative to pad)
    frame.GetXaxis().SetTitleSize(0.13)
    frame.GetXaxis().SetLabelSize(0.12)
    frame.GetXaxis().SetTitleOffset(1.0)
    frame.GetYaxis().SetTitleSize(0.12)
    frame.GetYaxis().SetLabelSize(0.11)
    frame.GetYaxis().SetTitleOffset(0.55)
    frame.GetYaxis().SetNdivisions(505)
    frame.Draw("AXIS")

    # Build the ratio values
    h_ratio = h_NUM.Clone("h_ratio")
    h_ratio.Divide(h_DEN)

    h_ratio.SetLineColor(ROOT.kBlack)
    h_ratio.SetMarkerColor(ROOT.kBlack)
    h_ratio.SetMarkerStyle(20)
    h_ratio.Draw("E1 SAME")

    # Horizontal line at 1
    one_bottom = ROOT.TLine(0, 1.0, 20, 1.0)
    one_bottom.SetLineStyle(2)
    one_bottom.Draw()

    # Final save
    canvas.SaveAs(f"{args.dest}/{args.tag}_{year}/counts_{sample_type}_{hlt_path}.pdf")


def plot_efficiency(
    args: argparse.Namespace,
    year: str,
    hlt_path: str,
    h_data_NUM: ROOT.TH1,
    h_data_DEN: ROOT.TH1,
    h_mc_NUM: ROOT.TH1,
    h_mc_DEN: ROOT.TH1,
) -> tuple[float, float]:
    canvas = CMS.cmsDiCanvas(
        canvName="HLT_efficiency",
        x_min=0,
        x_max=20,
        y_min=0,
        y_max=1,
        r_min=0,
        r_max=2,
        nameXaxis=variable_label,
        nameYaxis="Efficiency",
        nameRatio="Data/MC",
        square=CMS.kSquare,
        extraSpace=0,
        iPos=11,
    )

    # --- 1) Make pad margins identical & sized for a readable ratio pad
    # (numbers are typical CMS-like; adjust once and forget)
    canvas.cd(1)
    pad1 = ROOT.gPad
    pad1.SetLeftMargin(0.16)
    pad1.SetRightMargin(0.04)
    pad1.SetTopMargin(0.08)
    pad1.SetBottomMargin(0.02)  # tiny, since we hide top x labels
    pad1.SetTicks(1, 1)

    canvas.cd(2)
    pad2 = ROOT.gPad
    pad2.SetLeftMargin(0.16)
    pad2.SetRightMargin(0.04)
    pad2.SetTopMargin(0.03)
    pad2.SetBottomMargin(0.40)  # big enough for large x labels
    pad2.SetTicks(1, 1)

    canvas.cd(1)

    legend = ROOT.TLegend(*legend_position)

    h_eff_mc = ROOT.TEfficiency(h_mc_NUM, h_mc_DEN)
    h_eff_mc.SetTitle(f"QCD MC;{variable_label};Efficiency")
    h_eff_data = ROOT.TEfficiency(h_data_NUM, h_data_DEN)
    h_eff_data.SetTitle(f"Data;{variable_label};Efficiency")

    # Draw, then style the *painted graph*
    h_eff_mc.Draw()
    h_eff_mc.SetLineColor(ROOT.kBlue)
    h_eff_mc.SetMarkerColor(ROOT.kBlue)
    h_eff_data.SetLineColor(ROOT.kBlack)
    h_eff_data.SetMarkerColor(ROOT.kBlack)
    h_eff_data.Draw("SAME")

    ROOT.gPad.Update()  # needed before GetPaintedGraph()
    gr = h_eff_mc.GetPaintedGraph()

    # --- 2) Force same x range and hide top x labels so they don't peek through
    gr.GetXaxis().SetLimits(0, 20)  # equivalent to SetRangeUser for TGraph
    gr.GetXaxis().SetLabelSize(0)  # hide labels on top pad
    gr.GetXaxis().SetTitleSize(0)  # hide title on top pad
    gr.GetXaxis().SetLabelOffset(999)  # belt-and-suspenders
    gr.GetYaxis().SetRangeUser(0, 1.2)

    # Keep y-axis readable on the tall pad
    gr.GetYaxis().SetTitleOffset(1.2)
    gr.GetYaxis().SetLabelSize(0.050)
    gr.GetYaxis().SetTitleSize(0.055)

    legend.AddEntry(h_eff_mc, "QCD MC", "lp")
    legend.AddEntry(h_eff_data, "Data", "lp")
    legend.SetBorderSize(0)
    legend.Draw()

    # Horizontal line at 1
    one_top = ROOT.TLine(0, 1.0, 20, 1.0)
    one_top.SetLineStyle(2)
    one_top.Draw()

    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextFont(42)
    text.SetTextSize(0.035)
    text.DrawLatex(0.2, 0.86, f"selection = 3#mu + {trigger_paths[year]['reference']}")
    text.DrawLatex(
        0.2, 0.80, f"efficiency = #frac{{{hlt_path} + selection}}{{selection}}"
    )

    plot_cms_header(year)

    # --- Ratio plot
    canvas.cd(2)

    # Use a TH1 as frame to control axis styling precisely
    frame = h_mc_NUM.Clone("ratio_frame")
    frame.Reset()
    frame.GetXaxis().SetRangeUser(0, 20)
    frame.GetYaxis().SetRangeUser(0.7, 1.3)
    frame.SetTitle(f";{variable_label};Data/MC")

    # 3) Scale text sizes up on the small ratio pad (ROOT sizes are relative to pad)
    frame.GetXaxis().SetTitleSize(0.13)
    frame.GetXaxis().SetLabelSize(0.12)
    frame.GetXaxis().SetTitleOffset(1.0)
    frame.GetYaxis().SetTitleSize(0.12)
    frame.GetYaxis().SetLabelSize(0.11)
    frame.GetYaxis().SetTitleOffset(0.55)
    frame.GetYaxis().SetNdivisions(505)
    frame.Draw("AXIS")

    # Build the ratio values
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

    h_ratio.SetLineColor(ROOT.kBlack)
    h_ratio.SetMarkerColor(ROOT.kBlack)
    h_ratio.SetMarkerStyle(20)
    h_ratio.Draw("E1 SAME")

    # Horizontal line at 1
    one_bottom = ROOT.TLine(0, 1.0, 20, 1.0)
    one_bottom.SetLineStyle(2)
    one_bottom.Draw()

    pt_min = 3 if "5_3_3" in hlt_path else 5
    pol0_ratio = ROOT.TF1("pol0_ratio", "pol0", pt_min, 20)
    pol0_ratio.SetLineColor(ROOT.kRed)
    h_ratio.Fit(pol0_ratio, "R", "", pt_min, 20)
    pol0_ratio.Draw("SAME")

    pol1_ratio = ROOT.TF1("pol1_ratio", "pol1", pt_min, 20)
    pol1_ratio.SetLineColor(ROOT.kGreen)
    h_ratio.Fit(pol1_ratio, "R", "", pt_min, 20)
    pol1_ratio.Draw("SAME")

    legend_ratio = ROOT.TLegend(0.83, 0.77, 0.93, 0.93)
    legend_ratio.AddEntry(pol0_ratio, "Fit: pol0", "l")
    legend_ratio.AddEntry(pol1_ratio, "Fit: pol1", "l")
    legend_ratio.SetBorderSize(0)
    legend_ratio.SetTextSize(0.06)
    legend_ratio.Draw()

    sf_text = ROOT.TLatex()
    sf_text.SetNDC()
    sf_text.SetTextFont(42)
    sf_text.SetTextSize(0.07)
    sf_text.DrawLatex(
        0.5,
        0.45,
        f"#LTSF#GT = {pol0_ratio.GetParameter(0):.3f}#pm{pol0_ratio.GetParError(0):.3f}",
    )

    # ROOT.gStyle.SetOptFit(0000)

    # Final save
    canvas.SaveAs(f"{args.dest}/{args.tag}_{year}/{hlt_path}.pdf")

    return pol0_ratio.GetParameter(0), pol0_ratio.GetParError(0)


if __name__ == "__main__":
    args = parse_args()

    rebin = 2j
    per_year_global_SFs = {}

    for year in args.year:
        plots = plot_utils.loader(
            tag=f"{args.tag}_{year}",
            era=year,
            custom_lumi=None,
            load_data=True,
        )

        qcd_sample = f"QCD_Pt_MuEnrichedPt5_{year}"

        print("Making export directory for efficiency plots...", flush=True)
        os.makedirs(f"{args.dest}/{args.tag}_{year}", exist_ok=True)

        SFs = []
        unc_SFs = []

        tot_effs_mc = []
        tot_effs_data = []

        for hlt_path in tqdm(
            trigger_paths[year]["paths"], desc=f"Processing HLT paths for {year}"
        ):
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

            # Populate total eff arrays
            cut = 3j if "5_3_3" in hlt_path else 5j
            tot_effs_mc.append(
                plots[qcd_sample][f"NUM_{hlt_path}"][cut::sum].value
                / plots[qcd_sample][f"DEN_{hlt_path}"][cut::sum].value
            )
            tot_effs_data.append(
                plots[f"Data_{year}"][f"NUM_{hlt_path}"][cut::sum].value
                / plots[f"Data_{year}"][f"DEN_{hlt_path}"][cut::sum].value
            )

            plot_counts(
                args, year, hlt_path, "data", h_data_NUM.Clone(), h_data_DEN.Clone()
            )
            plot_counts(args, year, hlt_path, "mc", h_mc_NUM.Clone(), h_mc_DEN.Clone())
            sf, unc_sf = plot_efficiency(
                args,
                year,
                hlt_path,
                h_data_NUM.Clone(),
                h_data_DEN.Clone(),
                h_mc_NUM.Clone(),
                h_mc_DEN.Clone(),
            )

            # Calculate and print global scale factors
            print(f"Global SF for {hlt_path} in {year}: {sf} ± {unc_sf}", flush=True)
            SFs.append(sf)
            unc_SFs.append(unc_sf)

        weights = 1.0 / (np.array(unc_SFs) ** 2)
        sum_wx = np.sum(weights * np.array(SFs))
        sum_w = np.sum(weights)
        global_sf = sum_wx / sum_w
        global_sf_unc = math.sqrt(1.0 / sum_w)
        sf_spread = np.array(SFs) - global_sf
        max_deviation = np.max(np.abs(sf_spread))

        print(tot_effs_mc)
        tot_eff_OR_mc = 1.0 - np.prod(1.0 - np.array(tot_effs_mc))
        print(tot_effs_data)
        tot_eff_OR_data = 1.0 - np.prod(1.0 - np.array(tot_effs_data))
        overall_sf = tot_eff_OR_data / tot_eff_OR_mc

        per_year_global_SFs[year] = (
            overall_sf,
            global_sf,
            global_sf_unc,
            max_deviation,
        )

    print("\nSummary of global scale factors per year:", flush=True)
    for year, (overall_sf, global_sf, unc_sf, spread) in per_year_global_SFs.items():
        print(
            f"{year}: {overall_sf:.3f}, {global_sf:.3f} ± {unc_sf:.3f} (w. av. err.) ± {spread:.3f} (spread among HLT paths)",
            flush=True,
        )
