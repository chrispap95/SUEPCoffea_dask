import os
import pickle

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/RA2b13TeVProduction#Dataset_luminosities_2016_pb_1
lumis = {
    "2016_apv": 19497.914,
    "2016": 16810.813,
    "2017": 41471.589,
    # NOTE: 2018 lumi is only for the main trigger path
    "2018": 54540.000,
}

sample_names = {
    "QCD_Pt_": "QCD_Pt",
    "QCD_HT": "QCD_HT",
    "MuEnriched": "QCD_Pt_MuEnriched",
    "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX": "DYJetsToLL_NLO",
    "DYJetsToMuMu": "DYJetsToMuMu",
    "DY0JetsToLL": "DY0JetsToLL",
    "DY1JetsToLL": "DYNJetsToLL",
    "DY2JetsToLL": "DYNJetsToLL",
    "DY3JetsToLL": "DYNJetsToLL",
    "DY4JetsToLL": "DYNJetsToLL",
    "DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8": "DYJetsToLL_NJ",
    "DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8": "DYJetsToLL_NJ",
    "DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8": "DYJetsToLL_NJ",
    "DYJetsToLL_M-50_HT": "DYJetsToLL_HT",
    "DYJetsToLL_M-4to50_HT": "DYJetsToLL_HT",
    "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8": "DYLowMass_NLO",
    "DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8+": "DYLowMass_LO",
    "TTJets": "TTJets",
    "TTTo2L2Nu": "TTTo2L2Nu",
    "TTToSemiLeptonic": "TTToSemiLeptonic",
    "TTToHadronic": "TTToHadronic",
    "ttZJets": "ttZJets",
    "WWTo": "WW_all",
    "WZTo": "WZ_all",
    "ST_t-channel_": "ST_t-channel",
    "ST_tW_": "ST_tW",
    "WWZ_4F": "WWZ_4F",
    "WJetsToLNu_HT": "WJetsToLNu_HT",
    "WJetsToLNu_TuneCP5": "WJets_inclusive",
    "ZZTo4L": "ZZTo4L",
    "ZZZ": "ZZZ",
    "ZToMuMu": "ZToMuMu",
    "JetHT+Run": "data",
    "ScoutingPFHT": "data",
}


def lumiLabel(year):
    if year in ["2017", "2018"]:
        return round(lumis[year] / 1000, 1)
    elif year == "2016":
        return round((lumis[year] + lumis[year + "_apv"]) / 1000, 1)


def findLumi(year, auto_lumi, infile_name):
    if auto_lumi:
        if "20UL16MiniAODv2" in infile_name:
            lumi = lumis["2016"]
        if "20UL17MiniAODv2" in infile_name:
            lumi = lumis["2017"]
        if "20UL16MiniAODAPVv2" in infile_name:
            lumi = lumis["2016_apv"]
        if "20UL18" in infile_name:
            lumi = lumis["2018"]
        if "SUEP-m" in infile_name or "SUEP_m" in infile_name:
            lumi = lumis["2018"]
        if "JetHT+Run" in infile_name:
            lumi = 1
    if year and not auto_lumi:
        lumi = lumis[str(year)]
    if year and auto_lumi:
        raise Exception("Apply lumis automatically or based on year")
    return lumi


def fillSample(infile_name, plots, lumi):
    found_name = False
    sample = None
    for name in sample_names.keys():
        if name in infile_name:
            if found_name:
                raise Exception(f"Found multiple sample names in file name: {name}")
            sample = sample_names[name]
            found_name = True

    is_binned = False
    binned_samples = [
        "QCD_Pt_",
        "QCD_HT",
        "MuEnriched",
        "DY1JetsToLL",
        "DY2JetsToLL",
        "DY3JetsToLL",
        "DY4JetsToLL",
        "DYJetsToLL_0J",
        "DYJetsToLL_1J",
        "DYJetsToLL_2J",
        "DYJetsToLL_NJ",
        "DYJetsToLL_M-50_HT",
        "DYJetsToLL_M-4to50_HT",
        "ST_t-channel",
        "WJetsToLNu_HT",
        "WWTo",
        "WZTo",
        "ZToMuMu",
    ]
    for binned_sample in binned_samples:
        if binned_sample in infile_name:
            is_binned = True

    if is_binned:
        # include this block to import the bins individually
        temp_sample = infile_name.split("/")[-1].split(".pkl")[0]
        plots[temp_sample] = openpkl(infile_name)
        for plot in list(plots[temp_sample].keys()):
            plots[temp_sample][plot] = plots[temp_sample][plot] * lumi
    elif "SUEP" in infile_name or "ggHBSMpythia" in infile_name:
        if "+" in infile_name:
            sample = infile_name.split("/")[-1].split("+")[0]
        elif "new_generic" in infile_name:
            sample = infile_name.split("/")[-1].split("_")[
                1
            ]  # hack for Carlos naming convention
        else:
            sample = infile_name.split("/")[-1].replace("_histograms.pkl", "")
    elif "DoubleMuon" in infile_name:
        sample = infile_name.split("/")[-1].split(".pkl")[0]
    elif sample is None:
        sample = infile_name
    return sample, plots


# load file(s)
def loader(
    infile_names,
    year=None,
    auto_lumi=False,
    custom_lumi=None,
    is_data=False,
):
    plots = {}
    for infile_name in infile_names:
        if not os.path.isfile(infile_name):
            print("WARNING:", infile_name, "doesn't exist")
            continue
        elif ".pkl" not in infile_name:
            continue

        # set the lumi based on year or override using a custom value (in /pb). Data shouldn't be scaled.
        lumi = findLumi(year, auto_lumi, infile_name)
        if custom_lumi is not None:
            lumi = custom_lumi
        if is_data:
            lumi = 1

        # plots[sample] sample is filled here
        sample, plots = fillSample(infile_name, plots, lumi)

        if sample not in plots:
            plots[sample] = openpkl(infile_name)
            for plot in plots[sample]:
                plots[sample][plot] = plots[sample][plot] * lumi
        else:
            plotsToAdd = openpkl(infile_name)
            for plot in plotsToAdd:
                if plot not in plots[sample]:
                    plots[sample][plot] = plotsToAdd[plot] * lumi
                else:
                    plots[sample][plot] = plots[sample][plot] + plotsToAdd[plot] * lumi
    return plots


# function to load files from pickle
def openpkl(infile_name):
    plots = {}
    with open(infile_name, "rb") as openfile:
        while True:
            try:
                plots.update(pickle.load(openfile))
            except EOFError:
                break
    return plots
