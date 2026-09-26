"""Plot conditional trigger efficiencies for the three offline probe pT ranks.

Use output from SUEP_coffea_HLT_eff_ranked. Each rank has its own counts,
efficiency and data/MC ratio plots. No averages across ranks or paths are made.
"""

import argparse
import math
import pathlib
import warnings

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
        default="HLT_eff_ranked",
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
        default=str(pathlib.Path(__file__).parent / "HLT_effs_ranked"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'HLT_effs_ranked'}.",
    )
    parser.add_argument(
        "--muon-rank",
        type=int,
        nargs="+",
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Probe pT ranks to plot separately (default: all three).",
    )
    parser.add_argument(
        "--rebin",
        type=int,
        default=2,
        help="Merge this many adjacent bins (default: 2).",
    )
    args = parser.parse_args()
    if args.rebin < 1:
        parser.error("--rebin must be positive")
    return args


muon_variables = {
    1: ("leading_muon_pt", "p^{1st muon probe}_{T} [GeV]"),
    2: ("subleading_muon_pt", "p^{2nd muon probe}_{T} [GeV]"),
    3: ("subsubleading_muon_pt", "p^{3rd muon probe}_{T} [GeV]"),
}


def plot_directory(args, year, rank):
    return pathlib.Path(args.dest) / f"{args.tag}_{year}" / muon_variables[rank][0]


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
    rank: int,
):
    variable_label = muon_variables[rank][1]
    x_max = h_DEN.GetXaxis().GetXmax()
    # Let's plot the counts first
    canvas = CMS.cmsDiCanvas(
        canvName="counts",
        x_min=0,
        x_max=x_max,
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

    h_NUM.GetYaxis().SetRangeUser(0, max(1.0, h_DEN.GetMaximum() * 1.1))

    plot_cms_header(year)

    # --- Ratio plot
    canvas.cd(2)

    # Use a TH1 as frame to control axis styling precisely
    frame = h_NUM.Clone("ratio_frame")
    frame.Reset()
    frame.GetXaxis().SetRangeUser(0, x_max)
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
    one_bottom = ROOT.TLine(0, 1.0, x_max, 1.0)
    one_bottom.SetLineStyle(2)
    one_bottom.Draw()

    # Final save
    canvas.SaveAs(
        f"{plot_directory(args, year, rank)}/counts_{sample_type}_{hlt_path}.pdf"
    )
    canvas.Close()


def plot_efficiency(
    args: argparse.Namespace,
    year: str,
    hlt_path: str,
    h_data_NUM: ROOT.TH1,
    h_data_DEN: ROOT.TH1,
    h_mc_NUM: ROOT.TH1,
    h_mc_DEN: ROOT.TH1,
    rank: int,
):
    variable_label = muon_variables[rank][1]
    x_max = h_mc_DEN.GetXaxis().GetXmax()
    canvas = CMS.cmsDiCanvas(
        canvName="HLT_efficiency",
        x_min=0,
        x_max=x_max,
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
    for efficiency in (h_eff_mc, h_eff_data):
        efficiency.SetUseWeightedEvents()
        efficiency.SetStatisticOption(ROOT.TEfficiency.kFNormal)

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
    gr.GetXaxis().SetLimits(0, x_max)  # equivalent to SetRangeUser for TGraph
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
    one_top = ROOT.TLine(0, 1.0, x_max, 1.0)
    one_top.SetLineStyle(2)
    one_top.Draw()

    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextFont(42)
    text.SetTextSize(0.035)
    text.DrawLatex(0.2, 0.86, f"Reference: {trigger_paths[year]['reference']}")
    text.DrawLatex(
        0.2, 0.80, f"efficiency = #frac{{{hlt_path} + selection}}{{selection}}"
    )

    text.DrawLatex(
        0.2, 0.74, f"Probe rank {rank}; #geq 2 other reference-matched muons"
    )

    plot_cms_header(year)

    # --- Ratio plot
    canvas.cd(2)

    # Use a TH1 as frame to control axis styling precisely
    frame = h_mc_NUM.Clone("ratio_frame")
    frame.Reset()
    frame.GetXaxis().SetRangeUser(0, x_max)
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
    one_bottom = ROOT.TLine(0, 1.0, x_max, 1.0)
    one_bottom.SetLineStyle(2)
    one_bottom.Draw()

    # Final save
    canvas.SaveAs(f"{plot_directory(args, year, rank)}/{hlt_path}.pdf")

    canvas.Close()


def main():
    args = parse_args()
    ROOT.gROOT.SetBatch(True)
    rebin = complex(0, args.rebin)
    for year in args.year:
        plots = plot_utils.loader(
            tag=f"{args.tag}_{year}",
            era=year,
            custom_lumi=None,
            load_data=True,
        )
        for rank in dict.fromkeys(args.muon_rank):
            variable = muon_variables[rank][0]
            plot_directory(args, year, rank).mkdir(parents=True, exist_ok=True)
            for path in tqdm(
                trigger_paths[year]["paths"], desc=f"{year}, probe rank {rank}"
            ):
                histograms = {}
                for sample_type, sample in (
                    ("mc", f"QCD_Pt_MuEnrichedPt5_{year}"),
                    ("data", f"Data_{year}"),
                ):
                    for prefix in ("NUM", "DEN"):
                        key = f"{prefix}_{path}_{variable}"
                        if key not in plots.get(sample, {}):
                            raise KeyError(
                                f"Missing {key} in {sample}. Run SUEP_coffea_HLT_eff_ranked "
                                "for both data and MC with this tag first."
                            )
                        h = plots[sample][key]
                        if len(h.axes[0]) % args.rebin:
                            raise ValueError(
                                f"--rebin {args.rebin} does not divide the bin count of {key}"
                            )
                        histograms[sample_type, prefix] = h[::rebin]

                if (
                    args.normalize533data
                    and path == "HLT_TripleMu_5_3_3"
                    and year == "2016"
                ):
                    numerator = histograms["data", "NUM"].values()
                    denominator = histograms["data", "DEN"].values()
                    ratios = np.divide(
                        numerator,
                        denominator,
                        out=np.zeros_like(numerator),
                        where=denominator != 0,
                    )
                    scale = max(1.0, float(np.max(ratios)) * 1.000001)
                    histograms["data", "DEN"] *= scale
                    print(
                        f"{year}, rank {rank}, {path}: data denominator normalization = {scale:g}"
                    )

                root_hists = {
                    key: uproot.to_writable(h).to_pyroot()  # type: ignore[no-any-return]
                    for key, h in histograms.items()
                }
                for sample_type in ("data", "mc"):
                    plot_counts(
                        args,
                        year,
                        path,
                        sample_type,
                        root_hists[sample_type, "NUM"].Clone(),
                        root_hists[sample_type, "DEN"].Clone(),
                        rank,
                    )
                if any(
                    root_hists[sample, "DEN"].Integral() <= 0
                    for sample in ("data", "mc")
                ):
                    warnings.warn(
                        f"Skipping efficiency for {year}, rank {rank}, {path}: empty/nonpositive denominator; counts were saved."
                    )
                    continue
                if not all(
                    ROOT.TEfficiency.CheckConsistency(
                        root_hists[sample, "NUM"], root_hists[sample, "DEN"], "w"
                    )
                    for sample in ("data", "mc")
                ):
                    warnings.warn(
                        f"Skipping efficiency for {year}, rank {rank}, {path}: inconsistent numerator/denominator; counts were saved."
                    )
                    continue
                plot_efficiency(
                    args,
                    year,
                    path,
                    root_hists["data", "NUM"].Clone(),
                    root_hists["data", "DEN"].Clone(),
                    root_hists["mc", "NUM"].Clone(),
                    root_hists["mc", "DEN"].Clone(),
                    rank,
                )


if __name__ == "__main__":
    main()
