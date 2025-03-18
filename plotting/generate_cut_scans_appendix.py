import glob


def create_figure(document, plot):
    document.write(r"\begin{figure}[H]" + "\n")
    document.write(4 * " " + r"\centering" + "\n")
    document.write(
        4 * " " + r"\includegraphics[width=0.67\textwidth]{fig/" + plot + "}\n"
    )

    name = (
        plot.split("/")[-1]
        .replace(".pdf", "")
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
        + f"Cut scan plot for {name} ."
        + r"\label{fig:"
        + label
        + "}}\n"
    )
    document.write(r"\end{figure}" + "\n")


def create_subsection(document, name):
    document.write(r"\subsection{" + name + r"}" + "\n")
    document.write(
        r"\label{subsec:" + name.replace(" ", "-").replace("$", "") + r"}" + "\n"
    )


if __name__ == "__main__":
    with open("Appendix_cut_scans.tex", "w") as document:
        for plot in glob.glob("cut_scans_plots/*.pdf"):
            create_figure(document, plot)
        document.write("\n")
