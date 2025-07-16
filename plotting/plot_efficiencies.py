import os

import arrow
import cmsstyle as CMS  # type: ignore[import]
import plot_utils
import ROOT  # type: ignore[import]
import uproot
from tqdm import tqdm  # type: ignore[import]

CMS.SetExtraText("Simulation Preliminary")
CMS.SetLumi("")

plots = plot_utils.loader(
    tag="signal_effs_Jul2025_2018",
    era="2018",
    custom_lumi=None,
    load_data=False,
)

signal_points = [
    "GluGluToSUEP_mS125.000_mPhi1.000_T0.250_modeleptonic_13TeV_2018",
    "GluGluToSUEP_mS125.000_mPhi2.000_T2.000_modeleptonic_13TeV_2018",
    "GluGluToSUEP_mS125.000_mPhi4.000_T4.000_modeleptonic_13TeV_2018",
    "GluGluToSUEP_mS125.000_mPhi8.000_T8.000_modeleptonic_13TeV_2018",
    "GluGluToSUEP_mS125.000_mPhi8.000_T32.000_modeleptonic_13TeV_2018",
]
signal_labels = [
    r"m_{#phi} = 1 GeV, T = 0.25 GeV",
    r"m_{#phi} = 2 GeV, T = 2 GeV",
    r"m_{#phi} = 4 GeV, T = 4 GeV",
    r"m_{#phi} = 8 GeV, T = 8 GeV",
    r"m_{#phi} = 8 GeV, T = 32 GeV",
]
variable_labels = {
    "muon_pt": "cut (>) p^{#mu}_{T} [GeV]",
    "muon_abseta": "cut (<) |#eta^{#mu}|",
    "muon_id": "select muon ID",
    "muon_absdxy": "cut (<) |d^{#mu}_{xy}| [cm]",
    "muon_absdz": "cut (<) |d^{#mu}_{z}| [cm]",
}
legend_positions = {
    "muon_pt": (0.45, 0.7, 0.8, 0.9),
    "muon_abseta": (0.45, 0.2, 0.8, 0.4),
    "muon_id": (0.2, 0.25, 0.55, 0.45),
    "muon_absdxy": (0.45, 0.2, 0.8, 0.4),
    "muon_absdz": (0.45, 0.2, 0.8, 0.4),
}
id_labels = [
    "LooseId",
    "MediumId",
    "MediumPromptId",
    "TightId",
]
cut_positions = {
    # "muon_pt": 3,
    # "muon_abseta": 2.4,
    # "muon_id": "mediumId",
    "muon_absdxy": 0.2,
    "muon_absdz": 0.2,
}
text_x_positions = {
    "muon_pt": 0.7,
    "muon_abseta": 0.6,
    "muon_id": 0.3,
    "muon_absdxy": 0.5,
    "muon_absdz": 0.5,
}

petroff_6_colors = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]


def plot_efficiency(variable):
    canvas = CMS.cmsCanvas(
        "",
        0,
        1,
        0,
        1,
        "muon_pt cut",
        "Efficiency",
        square=CMS.kSquare,
        extraSpace=0.01,  # type: ignore[no-untyped-call]
        iPos=0,
    )

    legend = ROOT.TLegend(*legend_positions[variable])
    efficiencies = []
    for i, signal_point in enumerate(signal_points):
        h_NUM = uproot.to_writable(plots[signal_point][f"{variable}_NUM"]).to_pyroot()  # type: ignore[no-untyped-call]
        h_DEN = uproot.to_writable(plots[signal_point][f"{variable}_DEN"]).to_pyroot()  # type: ignore[no-untyped-call]

        if variable == "muon_id":
            h_NUM = plot_utils.convert_strcat_hist_to_root(
                plots[signal_point][f"{variable}_NUM"], "muon_id", "muon_id"
            )
            h_DEN = plot_utils.convert_strcat_hist_to_root(
                plots[signal_point][f"{variable}_DEN"], "muon_id", "muon_id"
            )
            for j in range(h_NUM.GetNbinsX()):
                h_NUM.GetXaxis().SetBinLabel(j + 1, id_labels[j])
                h_DEN.GetXaxis().SetBinLabel(j + 1, id_labels[j])

        if ROOT.TEfficiency.CheckConsistency(h_NUM, h_DEN):
            h_efficiency = ROOT.TEfficiency(h_NUM, h_DEN)
            h_efficiency.SetTitle(
                f"{signal_point};{variable_labels[variable]};Efficiency"
            )
            efficiencies.append(h_efficiency)
            legend.AddEntry(h_efficiency, signal_labels[i], "lp")
        else:
            raise RuntimeError(
                "Histograms h_NUM and h_DEN are not consistent for TEfficiency."
            )

    efficiencies[0].Draw("A")
    efficiencies[0].SetLineColor(ROOT.TColor.GetColor(petroff_6_colors[0]))
    efficiencies[0].SetMarkerColor(ROOT.TColor.GetColor(petroff_6_colors[0]))
    for i, eff in enumerate(efficiencies[1:]):
        eff.Draw("SAME")
        eff.SetLineColor(ROOT.TColor.GetColor(petroff_6_colors[i + 1]))
        eff.SetMarkerColor(ROOT.TColor.GetColor(petroff_6_colors[i + 1]))

    # Move y-axis to the right side
    ROOT.gPad.SetLeftMargin(0.16)
    ROOT.gPad.SetBottomMargin(0.16)
    ROOT.gPad.Update()
    efficiencies[0].GetPaintedGraph().GetXaxis().SetTitleOffset(1.2)
    efficiencies[0].GetPaintedGraph().GetYaxis().SetTitleOffset(1.3)
    if variable == "muon_id":
        efficiencies[0].GetPaintedGraph().GetHistogram().SetBins(4, 0, 4)
        efficiencies[0].GetPaintedGraph().GetXaxis().LabelsOption("h")
        efficiencies[0].GetPaintedGraph().GetXaxis().SetLabelSize(0.04)

    # if variable == "muon_pt":
    #     ROOT.gPad.Update()
    #     efficiencies[0].GetPaintedGraph().GetXaxis().SetLimits(1, 100)

    if variable in cut_positions:
        line = ROOT.TLine(
            cut_positions[variable],
            ROOT.gPad.GetUymin(),
            cut_positions[variable],
            ROOT.gPad.GetUymax(),
        )
        line.SetLineColor(ROOT.kRed)
        line.SetLineStyle(ROOT.kDashed)
        line.SetLineWidth(4)
        line.Draw("SAME")

        arrow = ROOT.TArrow(
            cut_positions[variable] * 0.2,
            ROOT.gPad.GetUymax() * 0.8,
            cut_positions[variable],
            ROOT.gPad.GetUymax() * 0.8,
            0.03,
            "<|",
        )
        arrow.SetLineColor(ROOT.kRed)
        arrow.SetFillColor(ROOT.kRed)
        arrow.SetLineWidth(2)
        arrow.Draw()

    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextSize(0.034)
    text.SetTextAlign(11)  # Right align
    text.DrawLatex(
        text_x_positions[variable],
        0.55,
        "m_{S} = 125 GeV",
    )
    text.DrawLatex(
        text_x_positions[variable],
        0.5,
        "m_{A'} = 0.5 GeV",
    )

    legend.Draw()

    if variable in ["muon_pt", "muon_absdxy", "muon_absdz"]:
        canvas.SetLogx()
    canvas.SaveAs(f"efficiencies/efficiency_plot_{variable}.pdf")


if __name__ == "__main__":
    print("Making export directory for efficiency plots...", flush=True)
    os.makedirs("efficiencies", exist_ok=True)

    for variable in tqdm(
        ["muon_pt", "muon_abseta", "muon_id", "muon_absdxy", "muon_absdz"],
        desc="Plotting efficiencies",
    ):
        plot_efficiency(variable)
