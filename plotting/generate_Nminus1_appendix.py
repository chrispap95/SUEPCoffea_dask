import glob


def create_figure(document, plot):
    document.write(r"\begin{figure}[H]" + "\n")
    document.write(4 * " " + r"\centering" + "\n")
    document.write(
        4 * " " + r"\includegraphics[width=0.7\textwidth]{fig/" + plot + "}\n"
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
        + f"$N-1$ plot for {name} ."
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
    with open("Appendix_Nminus1.tex", "w") as document:
        # create_subsection(document, "$N-1$ plots")
        for plot in glob.glob("Nminus1_plots/*.pdf"):
            create_figure(document, plot)
        document.write("\n")
