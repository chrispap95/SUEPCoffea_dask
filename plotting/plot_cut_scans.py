import argparse
import os
import pathlib

import matplotlib as mpl  # type: ignore[import]
import matplotlib.cm as cmx  # type: ignore[import]
import matplotlib.colors as colors  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import mplhep as hep
import numpy as np
import plot_utils

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

cmap = plt.get_cmap("viridis", 10)  # type: ignore[attr-defined]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="scans_Mar2025",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(pathlib.Path(__file__).parent / "cut_scans_plots"),
        help="Destination directory to save the plots. Default is "
        f"{pathlib.Path(__file__).parent / 'cut_scans_plots'}",
    )
    return parser.parse_args()


# Latex labels for the regions
region_labels = {
    "SR_high_temp_loose": r"$SR^{loose}_{high~T}$",
    "SR_high_temp_tight": r"$SR^{tight}_{high~T}$",
    "SR_low_temp_loose": r"$SR^{loose}_{low~T}$",
    "SR_low_temp_tight": r"$SR^{tight}_{low~T}$",
}

# Abbreviations for the sample names
sample_names = {
    "QCD_Pt_MuEnrichedPt5_2018": "QCD",
    "DY_2018": "DY",
}


# Latex labels for the plot names
def get_axis_label(plot):
    axis_labels = {
        "ip3d": r"cut on muon $IP_{3D}$ (cm)",
        "iso": "cut on muon isolation",
        "sph1": r"cut on $S_{1}$",
    }
    for key in axis_labels:
        if key in plot:
            return axis_labels[key]
    return plot


def make_plot(plots, plot, sample):
    hist_bkg = plots[sample][plot]

    # Split the histograms by the cuts and normalize to unit area
    hists_split_by_cuts = [hist_bkg[i, :] / hist_bkg[i, ::sum].value for i in range(10)]
    hists_split_by_cuts_errors = [
        (
            np.where(hist_bkg[i, :].values() < 0, 0, hist_bkg[i, :].values())
            / hist_bkg[i, ::sum].value
        )
        * np.sqrt(
            hist_bkg[i, :].variances() / hist_bkg[i, :].values() ** 2
            + hist_bkg[i, ::sum].variance / hist_bkg[i, ::sum].value ** 2
        )
        for i in range(10)
    ]

    fig, ax1 = plt.subplots(figsize=(10, 9))
    hep.histplot(
        hists_split_by_cuts,
        yerr=hists_split_by_cuts_errors,
        color=[cmap(i) for i in range(10)],
        lw=2,
        ax=ax1,
    )

    # Extract region and cut name from plot name
    if "tight" in plot:
        region = plot.split("tight")[0] + "tight"
        cut_name = plot.split("tight")[1]
    elif "loose" in plot:
        region = plot.split("loose")[0] + "loose"
        cut_name = plot.split("loose")[1]

    # Axis range
    edges = hist_bkg[:, ::sum].axes[0].edges
    if "sph1" in plot:
        norm = colors.Normalize(vmin=edges[0], vmax=edges[-1])
    else:
        norm = colors.LogNorm(vmin=edges[0], vmax=edges[-1])
    sm = cmx.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label=get_axis_label(plot), orientation="vertical")

    plt.text(
        0.7,
        0.85,
        region_labels[region] + " $-$ MC " + sample_names[sample],
        ha="center",
        weight="bold",
        transform=ax1.transAxes,
    )

    hep.cms.label(llabel="Preliminary", data=False, ax=ax1)

    plt.xlabel(r"$n_{muon}$")
    plt.yscale("log")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(
        f"{args.dest}/{region}_{sample_names[sample]}_{plot}.pdf", bbox_inches="tight"
    )
    plt.close()


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)

    # Load plots
    print("Loading plots...", end=" ", flush=True)
    plots_SR_high_temp = plot_utils.loader(tag=f"{args.tag}_SR_high_temp")
    plots_SR_low_temp = plot_utils.loader(tag=f"{args.tag}_SR_low_temp")
    plots = {}
    all_datasets = set(plots_SR_high_temp.keys()) | set(plots_SR_low_temp.keys())
    for dataset in list(all_datasets):
        if dataset not in plots_SR_high_temp:
            plots_SR_high_temp[dataset] = {}
        if dataset not in plots_SR_low_temp:
            plots_SR_low_temp[dataset] = {}
        plots[dataset] = plots_SR_high_temp[dataset] | plots_SR_low_temp[dataset]
    print("Done!", flush=True)

    for sample in ["QCD_Pt_MuEnrichedPt5_2018", "DY_2018"]:
        for plot in plots[sample]:
            make_plot(plots, plot, sample)
