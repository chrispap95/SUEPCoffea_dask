import glob
import os

Nminus1_plots_per_event = ["dimuon_mass", "sph1"]

Nminus1_plots_per_muon = [
    "muon_dxy",
    "muon_dz",
    "muon_ip3d",
    "muon_iso",
    "muon_pt",
    "muon_neutral_iso",
]


def has_source_plot(plot):
    basename = os.path.basename(plot).replace(".pdf", "")
    for source_plot in Nminus1_plots_per_muon:
        if source_plot in basename:
            return True
    return False


def create_figure(document, plot):
    is_source_plot = has_source_plot(plot)
    fig_width = "0.49"

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
    if is_source_plot:
        document.write(
            4 * " "
            + r"\includegraphics[width="
            + fig_width
            + r"\textwidth]{fig/"
            + plot.replace(".pdf", "_sources.pdf").replace(
                "Nminus1_plots/", "Nminus1_plots_sources/"
            )
            + "}\n"
        )

    name = (
        plot.split("/")[-1]
        .replace(".pdf", "")
        .replace("_Nminus1_", " ")
        .replace("CR_prompt", r"\CRDY")
        .replace("CR_cb", r"\CRQCD")
        .replace("SR_high_temp_loose", r"\SRhighl")
        .replace("SR_high_temp_tight", r"\SRhight")
        .replace("SR_low_temp_loose", r"\SRlowl")
        .replace("SR_low_temp_tight", r"\SRlowt")
        .replace("_", " ")
    )
    label = plot.split("/")[-1].replace(".pdf", "").replace("_", "-")
    document.write(
        4 * " "
        + r"\caption{"
        + f"$N-1$ plot for {name}."
        + r"\label{fig:"
        + label
        + "}}\n"
    )
    document.write(r"\end{figure}" + "\n")


def create_subsection(document, name, label=None):
    document.write(r"\subsection{" + name + r"}" + "\n")
    if label is None:
        label = name.replace(" ", "-").replace("$", "").replace("\\", "")
    document.write(r"\label{subsec:" + label + r"}" + "\n")


regions = {
    "CR_cb": r"\CRQCD",
    "CR_prompt": r"\CRDY",
    "SR_low_temp": r"\SRlow",
    "SR_high_temp": r"\SRhigh",
}

if __name__ == "__main__":
    with open("Appendix_Nminus1.tex", "w") as document:
        for region in regions:
            create_subsection(
                document,
                r"\texorpdfstring{$N-1$}{N-1} plots for "
                + r"\texorpdfstring{"
                + regions[region]
                + "}{"
                + region.replace("_", " ")
                + "}",
                "N-1-plots-for-" + region.replace("_", "-"),
            )
            for plot in glob.glob(f"Nminus1_plots/{region}*.pdf"):
                create_figure(document, plot)
            document.write(r"\clearpage" + "\n")
            document.write("\n")
