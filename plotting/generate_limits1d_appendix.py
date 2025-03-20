import glob


def create_figure(document, plot):
    document.write(r"\begin{figure}[htbp]" + "\n")
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


temperatures = [0.25, 0.35, 1, 1.4, 2, 2.8, 3, 4, 5.6, 6, 8, 12, 16, 24, 32]

if __name__ == "__main__":
    with open("Appendix_limits.tex", "w") as document:
        # create_subsection(document, "1D limits")
        for temp in temperatures:
            for plot in glob.glob(
                f"limit_plots/limits_mPhi*_T{temp:.3f}".replace(".", "p") + ".pdf"
            ):
                create_figure(document, plot)
        document.write("\n")
