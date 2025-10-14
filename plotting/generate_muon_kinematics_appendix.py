import glob

mu_kin_plots_per_dimuon = ["dimuon_dr", "dimuon_mass"]

mu_kin_plots_per_muon = [
    "muon_pt",
    "muon_eta",
    "muon_phi",
    "muon_dxy",
    "muon_dz",
    "muon_iso",
]

regions = {
    "CR_cb": r"\CRQCD",
    "CR_prompt": r"\CRDY",
    "SR_low_temp_tight": r"\SRlowt",
    "SR_low_temp_loose": r"\SRlowl",
    "SR_high_temp_tight": r"\SRhight",
    "SR_high_temp_loose": r"\SRhighl",
    "VR_loose": r"\VRloose",
    "VR_tight": r"\VRtight",
}

tag = "muon_kinematics_Oct2025"


def create_figure(document, plot):
    fig_width = "0.65"

    document.write(r"\begin{figure}[htbp]" + "\n")
    document.write(4 * " " + r"\centering" + "\n")
    document.write(
        4 * " "
        + r"\includegraphics[width="
        + fig_width
        + r"\textwidth]{fig/"
        + plot
        + "}\n"
    )

    name = (
        plot.split("/")[-1]
        .replace(".pdf", "")
        .replace("CR_prompt", r"\CRDY")
        .replace("CR_cb", r"\CRQCD")
        .replace("SR_high_temp_loose", r"\SRhighl")
        .replace("SR_high_temp_tight", r"\SRhight")
        .replace("SR_low_temp_loose", r"\SRlowl")
        .replace("SR_low_temp_tight", r"\SRlowt")
        .replace("VR_loose", r"\VRloose")
        .replace("VR_tight", r"\VRtight")
        .replace("_", " ")
    )
    label = plot.split("/")[-1].replace(".pdf", "").replace("_", "-")
    document.write(
        4 * " " + r"\caption{" + f"Plot for {name}." + r"\label{fig:" + label + "}}\n"
    )
    document.write(r"\end{figure}" + "\n")


def create_subsection(document, name, label=None):
    document.write(r"\subsection{" + name + r"}" + "\n")
    if label is None:
        label = name.replace(" ", "-").replace("$", "").replace("\\", "")
    document.write(r"\label{subsec:" + label + r"}" + "\n")


if __name__ == "__main__":
    with open("Appendix_muon_kinematics.tex", "w") as document:
        for region in regions:
            create_subsection(
                document,
                r"Muon kinematics plots for "
                + r"\texorpdfstring{"
                + regions[region]
                + "}{"
                + region.replace("_", " ")
                + "}",
                "muon-kinematics-plots-for-" + region.replace("_", "-"),
            )
            for plot in glob.glob(f"muon_kinematics_plots/{tag}/{region}*.pdf"):
                create_figure(document, plot)
            document.write(r"\clearpage" + "\n")
            document.write("\n")
