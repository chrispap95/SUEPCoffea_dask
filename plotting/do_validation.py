import argparse
import os
import warnings

import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import mplhep as hep
import plot_utils

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Dec2024",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        help="Custom integrated luminosity to be used (in pb^-1). For example, use 559.322 for "
        "the single data file in filelists/data/data_Run2018A_0p6fb_1file_unskimmed.json."
        "If not provided, the luminosity will be determined automatically for the year.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/muon_branches/SUEPCoffea_dask/plotting/vr_plots",
        help="Destination directory to save the plots",
    )
    return parser.parse_args()


if "__main__" == __name__:
    args = parse_args()

    # Create destination directory
    os.makedirs(args.dest, exist_ok=True)

    # Load plots and merge them
    print("Loading plots...", end=" ", flush=True)
    plots = plot_utils.loader(
        tag=f"{args.tag}_VR", custom_lumi=args.lumi, load_data=True
    )
    print("Done!", flush=True)

    print("Fit and extrapolation...", end=" ", flush=True)
    # QCD extrapolation
    # Slice the first bin out where needed for fit stability
    slice_hists = {
        "VR_loose": slice(4j, None),
        "VR_tight": slice(3j, None),
    }

    qcd_extrapolation = plot_utils.Extrapolation(plots["QCD_Pt_MuEnrichedPt5_2018"])
    qcd_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

    data_extrapolation = plot_utils.Extrapolation(plots["DoubleMuon_2018"])
    data_extrapolation.extrapolate(slice_hists=slice_hists, verbose=False)

    print("Done!", flush=True)

    qcd_extrapolation.plot_fit("VR")
    plt.text(4, 1e6, "QCD", fontsize=20, ha="center")
    hep.cms.label(llabel="Preliminary", data=False, lumi=55)
    plt.tight_layout()
    plt.savefig(f"{args.dest}/plot_fit_VR_qcd.pdf")
    plt.close()

    qcd_extrapolation.plot_overlay(regions=["VR"])
    plt.text(4, 1e6, "QCD", fontsize=20, ha="center")
    hep.cms.label(llabel="Preliminary", data=False, lumi=55)
    plt.tight_layout()
    plt.savefig(f"{args.dest}/fit_overlay_qcd.pdf")
    plt.close()

    data_extrapolation.plot_fit("VR")
    plt.text(4, 1e6, "Data", fontsize=20, ha="center")
    hep.cms.label(llabel="Preliminary", data=True, lumi=55)
    plt.tight_layout()
    plt.savefig(f"{args.dest}/plot_fit_VR_data.pdf")
    plt.close()

    data_extrapolation.plot_overlay(regions=["VR"])
    plt.text(4, 1e6, "Data", fontsize=20, ha="center")
    hep.cms.label(llabel="Preliminary", data=True, lumi=55)
    plt.tight_layout()
    plt.savefig(f"{args.dest}/fit_overlay_data.pdf")
    plt.close()
