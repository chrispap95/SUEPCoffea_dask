import os
import glob
import re
import pickle
from typing import Optional
from rich.pretty import pprint

import dataset_groups

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/RA2b13TeVProduction#Dataset_luminosities_2016_pb_1
lumis = {
    "2016_apv": 19497.914,
    "2016": 16810.813,
    "2017": 41471.589,
    # NOTE: 2018 lumi is only for the main trigger path
    "2018": 54540.000,
}


def lumi_Label(year: str) -> float:
    if year == "2016":
        return round((lumis[year] + lumis[year + "_apv"]) / 1000, 1)
    return round(lumis[year] / 1000, 1)


def find_lumi(
    infile_name: str,
    auto_lumi: Optional[bool] = True,
    year: Optional[str | None] = None,
) -> float:
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
        if "DoubleMuon" in infile_name:
            lumi = 1
    if year and not auto_lumi:
        lumi = lumis[year]
    if year and auto_lumi:
        raise Exception("Apply lumis automatically or based on year")
    return lumi


def load_samples(
    infile_names: list[str],
    year: Optional[int | str | None] = None,
    auto_lumi: Optional[bool | None] = None,
    custom_lumi: Optional[float | None] = None,
    is_data: Optional[bool | None] = False,
) -> dict:
    if isinstance(year, int):
        year = str(year)

    plots_ = {}
    # Load histograms and scale to lumi
    for infile_name in infile_names:
        if not os.path.isfile(infile_name):
            print("WARNING:", infile_name, "doesn't exist")
            continue
        elif ".pkl" not in infile_name:
            print("WARNING:", infile_name, "is not a .pkl file")
            continue

        # set the lumi based on year or override using a custom value (in /pb). Data shouldn't be scaled.
        lumi = find_lumi(
            infile_name,
            auto_lumi,
            year,
        )
        if custom_lumi is not None:
            lumi = custom_lumi
        if is_data:
            lumi = 1

        sample = infile_name.split("/")[-1].replace("_histograms.pkl", "")

        with open(infile_name, "rb") as f:
            plots_[sample] = pickle.load(f)
        for plot in plots_[sample]:
            plots_[sample][plot] = plots_[sample][plot] * lumi

    # Create combined histograms
    for combined_dataset in dataset_groups.dataset_groups_new:
        plots_[combined_dataset] = {}
        for pattern in dataset_groups.dataset_groups_new[combined_dataset]:
            for sample in plots_:
                if re.search(pattern, sample):
                    for plot in plots_[sample]:
                        if plot not in plots_[combined_dataset]:
                            plots_[combined_dataset][plot] = plots_[sample][plot].copy()
                        else:
                            plots_[combined_dataset][plot] += plots_[sample][plot]

    return plots_


def loader(
    tag="test",
    custom_lumi=None,
    load_data=False,
    verbosity=0,
):
    # input .pkl files
    plotDir = f"../../processor_output_files/{tag}_output_histograms/"
    filenames = glob.glob(plotDir + "*histograms.pkl")

    # separate the files into signal, background, and data
    files_SUEP = [f for f in filenames if ("SUEP" in f) or ("ggHBSMpythia" in f)]
    files_bkg = [
        f
        for f in filenames
        if ("pythia8" in f) and ("SUEP" not in f) and ("ggHBSMpythia" not in f)
    ]
    files_data = [f for f in filenames if ("DoubleMuon" in f)]
    if verbosity > 0:
        pprint(files_bkg)

    # load histograms and scale to lumi
    plots_SUEP_2018 = load_samples(files_SUEP, year=2018, custom_lumi=custom_lumi)
    plots_bkg_2018 = load_samples(files_bkg, year=2018, custom_lumi=custom_lumi)
    if load_data:
        plots_data_2018 = load_samples(files_data, year=2018, is_data=True)

    if verbosity > 1:
        pprint(plots_SUEP_2018)

    # put everything in one dictionary
    plots_ = {}
    for sample in plots_SUEP_2018:
        if sample + "_2018" not in plots_:
            plots_[sample + "_2018"] = plots_SUEP_2018[sample]
        else:
            plots_[sample + "_2018"].update(plots_SUEP_2018[sample])
    for sample in plots_bkg_2018:
        if sample + "_2018" not in plots_:
            plots_[sample + "_2018"] = plots_bkg_2018[sample]
        else:
            plots_[sample + "_2018"].update(plots_bkg_2018[sample])
    if load_data:
        for sample in plots_data_2018:
            if sample + "_2018" not in plots_:
                plots_[sample + "_2018"] = plots_data_2018[sample]
            else:
                plots_[sample + "_2018"].update(plots_data_2018[sample])

    return plots_
