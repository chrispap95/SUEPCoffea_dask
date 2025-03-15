import glob


def create_figure(document, plot):
    document.write(r"\begin{figure}[H]" + "\n")
    document.write(4 * " " + r"\centering" + "\n")
    document.write(
        4 * " " + r"\includegraphics[width=0.6\textwidth]{fig/" + plot + "}\n"
    )

    name = (
        plot.split("/")[-1].replace(".pdf", "").replace("limits_", "").replace("_", "-")
    )
    document.write(
        4 * " "
        + r"\caption{"
        + f"1D limit for {name} (both decay modes) as a function of "
        + r"$m_S$.\label{fig:"
        + name
        + "}}\n"
    )
    document.write(r"\end{figure}" + "\n")


def create_subsection(document, name):
    document.write(r"\subsection{" + name + r"}" + "\n")
    document.write(r"\label{subsec:" + name.replace(" ", "-") + r"}" + "\n")


if __name__ == "__main__":
    with open("Appendix_limits.tex", "w") as document:
        create_subsection(document, "1D limits")
        for plot in glob.glob("limit_plots/limits_mPhi*_T*.pdf"):
            create_figure(document, plot)
        document.write("\n")
