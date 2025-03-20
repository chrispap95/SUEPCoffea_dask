import os

verbosity = 0


def create_figure(document, systematic, region, samples, year):
    document.write(r"\begin{figure}[htbp]" + "\n")
    document.write(4 * " " + r"\centering" + "\n")
    for sample in samples:
        if systematic not in samples[sample]:
            if verbosity > 0:
                print(f"Skipping {systematic} for {sample} in {region}")
            continue
        if not os.path.exists(
            f"systematics_plots/{region}_{sample.replace('.', 'p')}_{year}_{systematic}.pdf"
        ):
            if verbosity > 0:
                print(
                    f"File systematics_plots/{region}_{sample.replace('.', 'p')}_{year}_{systematic}.pdf does not exist."
                )
            continue
        document.write(
            4 * " "
            + r"\includegraphics[width=0.49\textwidth]{"
            + f"fig/systematics_plots/{region}_{sample.replace('.', 'p')}_{year}_{systematic}"
            + r".pdf}"
            + "\n"
        )

    document.write(
        4 * " "
        + r"\caption{"
        + f"Shape systematic {systematic.replace('_', '-')} in {region.replace('_', '-')}."
        + r"\label{fig:"
        + f"{systematic.replace('_', '-')}-{region.replace('_', '-')}"
        + r"}}"
        + "\n"
    )
    document.write(r"\end{figure}" + "\n")


def create_subsection(document, systematic):
    document.write(r"\subsection{" + systematic + r"}" + "\n")
    document.write(r"\label{subsec:" + systematic + r"}" + "\n")


systematics = [
    "PUReweight",
    "ISR",
    "FSR",
    "L1PreFire",
    "MuonSF",
    "LHEPdf",
    "LHEScaleMuR",
    "LHEScaleMuF",
]

regions = [
    "CR_cb",
    "CR_prompt",
    "SR_low_temp_loose",
    "SR_low_temp_tight",
    "SR_low_temp_tight_extrapolation",
    "SR_high_temp_loose",
    "SR_high_temp_tight",
    "SR_high_temp_tight_extrapolation",
]

samples = {
    "DY": [
        "PUReweight",
        "ISR",
        "FSR",
        "L1PreFire",
        "MuonSF",
        "LHEPdf",
        "LHEScaleMuR",
        "LHEScaleMuF",
    ],
    "GluGluToSUEP_mS125.000_mPhi1.000_T0.250_modeleptonic_13TeV": [
        "PUReweight",
        "ISR",
        "FSR",
        "L1PreFire",
        "MuonSF",
    ],
    "GluGluToSUEP_mS125.000_mPhi8.000_T32.000_modeleptonic_13TeV": [
        "PUReweight",
        "ISR",
        "FSR",
        "L1PreFire",
        "MuonSF",
    ],
    "QCD_Pt_MuEnrichedPt5": [
        "PUReweight",
        "L1PreFire",
        "MuonSF",
    ],
}

year = 2018

if __name__ == "__main__":
    with open("Appendix_systematic_shapes.tex", "w") as document:
        for systematic in systematics:
            create_subsection(document, systematic)
            for region in regions:
                create_figure(document, systematic, region, samples, year)
            document.write(r"\clearpage" + "\n")
            document.write("\n")
