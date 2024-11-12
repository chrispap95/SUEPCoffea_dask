import glob
import itertools

import plot_utils


def subtract_histograms(h1, h2):
    hist_out = h1.copy().reset()
    values = h1.values() - h2.values()
    variances = h1.variances() + h2.variances()
    iterables = [range(len(axis.edges) - 1) for axis in h1.axes]
    for indices in itertools.product(*iterables):
        hist_out[indices] = (values[indices], variances[indices])
    return hist_out


def loader(
    tag="test",
    custom_lumi=None,
    load_data=False,
    scale_qcd=0,
    verbosity=0,
):
    # input .pkl files
    plotDir = f"../../processor_output_files/{tag}_output_histograms/"
    filenames = glob.glob(plotDir + "*histograms.pkl")

    # separate the files into signal, background, and data
    files_SUEP = [f for f in filenames if ("SUEP" in f)]
    files_SUEP += [f for f in filenames if ("ggHBSMpythia" in f)]
    files_bkg = [f for f in filenames if ("pythia8" in f) and ("SUEP" not in f)]
    files_data = [f for f in filenames if ("DoubleMuon" in f)]
    if verbosity > 0:
        print(files_bkg)

    # merge the histograms, apply lumis, exclude low HT bins
    plots_SUEP_2018 = plot_utils.loader(files_SUEP, year=2018, custom_lumi=custom_lumi)
    plots_bkg_2018 = plot_utils.loader(files_bkg, year=2018, custom_lumi=custom_lumi)
    if load_data:
        plots_data_2018 = plot_utils.loader(files_data, year=2018, is_data=True)

    if verbosity > 1:
        print(plots_SUEP_2018)

    # put everything in one dictionary
    plots = {}
    for plot in plots_SUEP_2018:
        plots[plot + "_2018"] = plots_SUEP_2018[plot]
    for plot in plots_bkg_2018:
        plots[plot + "_2018"] = plots_bkg_2018[plot]
    if load_data:
        for plot in plots_data_2018:
            plots[plot + "_2018"] = plots_data_2018[plot]

    # Combine DYJetsToLL_NLO with DYLowMass_NLO
    if "DYLowMass_NLO_2018" in plots.keys():
        dy_nlo_all = {}
        for plt_i in plots["DYLowMass_NLO_2018"].keys():
            dy_nlo_all[plt_i] = (
                plots["DYLowMass_NLO_2018"][plt_i] + plots["DYJetsToLL_NLO_2018"][plt_i]
            )
        plots["DY_2018"] = dy_nlo_all

    # Combine TTbar powheg
    if "TTToHadronic_2018" in plots.keys():
        ttbar_powheg = {}
        for plt_i in plots["TTToHadronic_2018"].keys():
            ttbar_powheg[plt_i] = (
                plots["TTToHadronic_2018"][plt_i]
                + plots["TTToSemiLeptonic_2018"][plt_i]
                + plots["TTTo2L2Nu_2018"][plt_i]
            )
        plots["TT_powheg_2018"] = ttbar_powheg

    # Combine ZZZ with WWZ
    if "WWZ_4F_2018" in plots.keys():
        vvv_combined = {}
        for plt_i in plots["WWZ_4F_2018"].keys():
            vvv_combined[plt_i] = plots["WWZ_4F_2018"][plt_i] + plots["ZZZ_2018"][plt_i]
        plots["VVV_2018"] = vvv_combined

    # Combine ZZ, WZ, and WW
    if "WZ_all_2018" in plots.keys():
        vv_combined = {}
        for plt_i in plots["WZ_all_2018"].keys():
            vv_combined[plt_i] = (
                plots["WW_all_2018"][plt_i]
                + plots["WZ_all_2018"][plt_i]
                + plots["ZZTo4L_2018"][plt_i]
            )
        plots["VV_2018"] = vv_combined

    # Combine ST
    if "ST_tW_2018" in plots.keys():
        st_combined = {}
        for plt_i in plots["ST_t-channel_2018"].keys():
            st_combined[plt_i] = (
                plots["ST_t-channel_2018"][plt_i] + plots["ST_tW_2018"][plt_i]
            )
        plots["ST_2018"] = st_combined

    # Combine WJetsHT and WJets_inclusive
    if "WJetsToLNu_HT_2018" in plots.keys():
        wjets_combined = {}
        for plt_i in plots["WJetsToLNu_HT_2018"].keys():
            wjets_combined[plt_i] = (
                plots["WJetsToLNu_HT_2018"][plt_i]
                + plots["WJets_inclusive_2018"][plt_i]
            )
        plots["WJets_all_2018"] = wjets_combined

    others_combined = {}
    if "ST_t-channel_2018" in plots.keys():
        for plt_i in plots["ST_t-channel_2018"].keys():
            others_combined[plt_i] = (
                plots["VVV_2018"][plt_i]
                + plots["ST_2018"][plt_i]
                + plots["WJets_all_2018"][plt_i]
                + plots["ttZJets_2018"][plt_i]
            )
        plots["Other_2018"] = others_combined

    # Normalize QCD MuEnriched if it exists
    if "QCD_Pt_MuEnriched_2018" in plots.keys() and scale_qcd > 0:
        for plot in plots["QCD_Pt_MuEnriched_2018"].keys():
            plots["QCD_Pt_MuEnriched_2018"][plot] = (
                plots["QCD_Pt_MuEnriched_2018"][plot] * scale_qcd
            )
        plots["QCD_2018"] = plots["QCD_Pt_MuEnriched_2018"]

    return plots
