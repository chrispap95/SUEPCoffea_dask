"""
SUEP_coffea.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer and Luca Lavezzo, 2021
"""

import itertools
from typing import Optional

import awkward as ak
import hist
import numba as nb
import numpy as np
import pandas as pd
import vector
from coffea import processor
from rich.pretty import pprint

import workflows.SUEP_utils as SUEP_utils

# Importing CMS corrections
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.CMS_corrections.pileup_utils import pileup_weight
from workflows.CMS_corrections.Prefire_utils import GetPrefireWeights
from workflows.pandas_accumulator import pandas_accumulator

# Set vector behavior
vector.register_awkward()


@nb.njit
def numba_n_unique(flat_array, starts, stops, placeholder):
    result = np.empty(len(starts), dtype=np.int64)

    # Loop over each sublist
    for i in range(len(starts)):
        seen = set()  # Set to track unique elements
        for j in range(starts[i], stops[i]):  # Loop over elements of the sublist
            elem = flat_array[j]
            if elem != placeholder:  # Skip placeholder values (e.g., -1)
                seen.add(elem)
        result[i] = len(seen)  # Store the count of unique elements

    return result


def n_unique(ak_array):
    # Flatten the awkward array
    # Use a placeholder (e.g., -1) for None values
    flat_array = ak.fill_none(ak.flatten(ak_array), -1)
    flat_array = np.array(flat_array)  # Convert to numpy array

    # Get the start and stop positions for each sublist and convert them to numpy arrays
    layout = ak_array.layout
    starts = np.array(layout.starts)
    stops = np.array(layout.stops)

    # Call numba function to count unique elements
    unique_counts = numba_n_unique(flat_array, starts, stops, -1)

    # Return result as an awkward array
    return ak.Array(unique_counts)


class SUEP_cluster(processor.ProcessorABC):
    def __init__(
        self,
        isMC: int,
        era: str,
        sample: str,
        do_syst: bool,
        syst_var: str,
        weight_syst: bool,
        flag: bool,
        output_location: Optional[str],
        accum: Optional[bool] = None,
        trigger: Optional[str] = None,
        blind: Optional[bool] = False,
        debug: Optional[bool] = None,
    ) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.era = era
        self.isMC = bool(isMC)
        self.sample = sample
        self.syst_var, self.syst_suffix = (
            (syst_var, f"_sys_{syst_var}") if do_syst and syst_var else ("", "")
        )
        self.weight_syst = weight_syst
        self.prefixes = {"SUEP": "SUEP"}
        self.accum = accum
        self.trigger = trigger
        self.blind = blind
        self.debug = debug

    def eventSelection(self, events):
        """
        Applies trigger, returns events.
        """
        trigger1 = np.ones(len(events), dtype=bool)
        trigger2 = np.ones(len(events), dtype=bool)
        trigger3 = np.ones(len(events), dtype=bool)
        if self.era in ["2016", "2016APV"]:
            if "TripleMu_5_3_3" in events.HLT.fields:
                trigger1 = events.HLT.TripleMu_5_3_3 == 1
            if "TripleMu_5_3_3_DZ_Mass3p8" in events.HLT.fields:
                trigger2 = events.HLT.TripleMu_5_3_3_DZ_Mass3p8 == 1
        elif self.era == "2017":
            if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
                trigger1 = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger2 = events.HLT.TripleMu_10_5_5_DZ == 1
        elif self.era in ["2018"]:
            if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
                trigger1 = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                trigger2 = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger3 = events.HLT.TripleMu_10_5_5_DZ == 1
        elif self.era in ["2022", "2023"]:
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                trigger1 = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger2 = events.HLT.TripleMu_10_5_5_DZ == 1
        else:
            raise ValueError("Invalid era")
        trigger = np.any(np.array([trigger1, trigger2, trigger3]).T, axis=-1)
        events = events[trigger]
        return events

    def get_weights(self, events):
        if not self.isMC:
            return np.ones(len(events))
        # Pileup weights (need to be fed with integers)
        pu_weights = pileup_weight(
            self.era, ak.values_astype(events.Pileup.nTrueInt, np.int32)
        )
        # L1 prefire weights
        prefire_weights = GetPrefireWeights(events)
        # Trigger scale factors
        # To be implemented
        return events.genWeight * pu_weights * prefire_weights

    def ht(self, events):
        jet_Cut = (events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4)
        jets = events.Jet[jet_Cut]
        return ak.sum(jets.pt, axis=-1)

    def muon_filter(self, events):
        """
        Filter events after the TripleMu trigger.
        Cleans muons and electrons.
        Requires at least nMuons with mediumId, pt, dxy, dz, and eta cuts.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dxy) <= 0.02)
            & (abs(events.Muon.dz) <= 0.1)
        )

        muons = muons[clean_muons]
        select_by_muons_high = ak.num(muons, axis=-1) >= 3
        events = events[select_by_muons_high]
        muons = muons[select_by_muons_high]
        return events, muons

    def fill_preclustering_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_, muons = self.muon_filter(events)
        if (len(events_) == 0) or (len(muons) == 0):
            return

        weights = self.get_weights(events_)

        for (
            cut_on_ip3d,
            cut_on_mini_iso,
            cut_on_neutral_iso,
            cut_on_pt,
        ) in itertools.product([True, False], repeat=4):
            cut = muons.pt > 0
            if cut_on_ip3d:
                cut = cut & (muons.ip3d < 0.01)
            if cut_on_mini_iso:
                cut = cut & (muons.miniPFRelIso_all < 1)
            if cut_on_neutral_iso:
                cut = cut & ((muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 0.1)
            if cut_on_pt:
                cut = cut & (muons.pt < 35)

            output[dataset]["histograms"][
                "ip3d_cut_vs_mini_iso_cut_vs_neutral_iso_cut_vs_pt_cut_vs_nMuon"
            ].fill(
                cut_on_ip3d,
                cut_on_mini_iso,
                cut_on_neutral_iso,
                cut_on_pt,
                ak.sum(cut, axis=-1),
                weight=weights,
            )
            muon_ip3d = ak.flatten(muons[cut].ip3d)
            muon_ip3d = ak.where(
                muon_ip3d < 1e-4,
                1.01 * 1.0e-4,
                ak.where(muon_ip3d >= 1, 1 * 0.99, muon_ip3d),
            )
            output[dataset]["histograms"][
                "ip3d_cut_vs_mini_iso_cut_vs_neutral_iso_cut_vs_pt_cut_vs_muon_ip3d"
            ].fill(
                cut_on_ip3d,
                cut_on_mini_iso,
                cut_on_neutral_iso,
                cut_on_pt,
                ak.flatten(muons[cut].ip3d),
                weight=ak.flatten(ak.broadcast_arrays(weights, muons[cut].pt)[0]),
            )
            muon_miniPFRelIso_all = ak.flatten(muons[cut].miniPFRelIso_all)
            muon_miniPFRelIso_all = ak.where(
                muon_miniPFRelIso_all < 1e-2,
                1.01 * 1e-2,
                ak.where(
                    muon_miniPFRelIso_all >= 100, 100 * 0.99, muon_miniPFRelIso_all
                ),
            )
            output[dataset]["histograms"][
                "ip3d_cut_vs_mini_iso_cut_vs_neutral_iso_cut_vs_pt_cut_vs_muon_miniPFRelIso_all"
            ].fill(
                cut_on_ip3d,
                cut_on_mini_iso,
                cut_on_neutral_iso,
                cut_on_pt,
                ak.flatten(muons[cut].miniPFRelIso_all),
                weight=ak.flatten(ak.broadcast_arrays(weights, muons[cut].pt)[0]),
            )
            muon_neutral_iso = ak.flatten(
                muons[cut].miniPFRelIso_all - muons[cut].miniPFRelIso_chg
            )
            muon_neutral_iso = ak.where(
                muon_neutral_iso < 1e-5,
                1.01 * 1e-5,
                ak.where(muon_neutral_iso >= 10, 10 * 0.99, muon_neutral_iso),
            )
            output[dataset]["histograms"][
                "ip3d_cut_vs_mini_iso_cut_vs_neutral_iso_cut_vs_pt_cut_vs_muon_neutral_iso"
            ].fill(
                cut_on_ip3d,
                cut_on_mini_iso,
                cut_on_neutral_iso,
                cut_on_pt,
                ak.flatten(muons[cut].miniPFRelIso_all - muons[cut].miniPFRelIso_chg),
                weight=ak.flatten(ak.broadcast_arrays(weights, muons[cut].pt)[0]),
            )
            muon_pt = ak.flatten(muons[cut].pt)
            muon_pt = ak.where(
                muon_pt < 3, 1.01 * 3, ak.where(muon_pt >= 300, 300 * 0.99, muon_pt)
            )
            output[dataset]["histograms"][
                "ip3d_cut_vs_mini_iso_cut_vs_neutral_iso_cut_vs_pt_cut_vs_muon_pt"
            ].fill(
                cut_on_ip3d,
                cut_on_mini_iso,
                cut_on_neutral_iso,
                cut_on_pt,
                ak.flatten(muons[cut].pt),
                weight=ak.flatten(ak.broadcast_arrays(weights, muons[cut].pt)[0]),
            )

        return

    def analysis(self, events, output):
        #####################################################################################
        # ---- Trigger event selection
        # Cut based on ak4 jets to replicate the trigger
        #####################################################################################

        # get dataset name
        dataset = events.metadata["dataset"]

        # take care of weights
        weights = self.get_weights(events)

        # Fill the cutflow columns for all
        output[dataset]["cutflow"].fill(len(events) * ["all"], weight=weights)

        # golden jsons for offline data
        if not self.isMC:
            events = applyGoldenJSON(self, events)

        events = self.eventSelection(events)

        # Apply HT selection for WJets stiching
        if "WJetsToLNu_HT" in dataset:
            events = events[self.ht(events) >= 70]
        elif "WJetsToLNu_TuneCP5" in dataset:
            events = events[self.ht(events) < 70]

        weights = self.get_weights(events)

        # Fill the cutflow columns for trigger
        output[dataset]["cutflow"].fill(
            len(events) * ["trigger"],
            weight=weights,
        )

        self.fill_preclustering_histograms(events, output)

        return

    def process(self, events):
        dataset = events.metadata["dataset"]
        cutflow = hist.Hist.new.StrCategory(
            [
                "all",
                "trigger",
            ],
            name="cutflow",
            label="cutflow",
        ).Weight()
        histograms = {
            "ip3d_cut_vs_mini_iso_cut_vs_neutral_iso_cut_vs_pt_cut_vs_nMuon": hist.Hist.new.Bool(
                name="ip3d_cut", label="ip3d_cut"
            )
            .Bool(name="mini_iso_cut", label="mini_iso_cut")
            .Bool(name="neutral_iso_cut", label="neutral_iso_cut")
            .Bool(name="pt_cut", label="pt_cut")
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "ip3d_cut_vs_mini_iso_cut_vs_neutral_iso_cut_vs_pt_cut_vs_muon_ip3d": hist.Hist.new.Bool(
                name="ip3d_cut", label="ip3d_cut"
            )
            .Bool(name="mini_iso_cut", label="mini_iso_cut")
            .Bool(name="neutral_iso_cut", label="neutral_iso_cut")
            .Bool(name="pt_cut", label="pt_cut")
            .Regular(
                100,
                1e-4,
                1,
                name="muon_ip3d",
                label="muon_ip3d",
                transform=hist.axis.transform.log,
            )
            .Weight(),
            "ip3d_cut_vs_mini_iso_cut_vs_neutral_iso_cut_vs_pt_cut_vs_muon_miniPFRelIso_all": hist.Hist.new.Bool(
                name="ip3d_cut", label="ip3d_cut"
            )
            .Bool(name="mini_iso_cut", label="mini_iso_cut")
            .Bool(name="neutral_iso_cut", label="neutral_iso_cut")
            .Bool(name="pt_cut", label="pt_cut")
            .Regular(
                100,
                1e-2,
                100,
                name="muon_miniPFRelIso_all",
                label="muon_miniPFRelIso_all",
                transform=hist.axis.transform.log,
            )
            .Weight(),
            "ip3d_cut_vs_mini_iso_cut_vs_neutral_iso_cut_vs_pt_cut_vs_muon_neutral_iso": hist.Hist.new.Bool(
                name="ip3d_cut", label="ip3d_cut"
            )
            .Bool(name="mini_iso_cut", label="mini_iso_cut")
            .Bool(name="neutral_iso_cut", label="neutral_iso_cut")
            .Bool(name="pt_cut", label="pt_cut")
            .Regular(
                100,
                1e-5,
                10,
                name="muon_neutral_iso",
                label="muon_neutral_iso",
                transform=hist.axis.transform.log,
            )
            .Weight(),
            "ip3d_cut_vs_mini_iso_cut_vs_neutral_iso_cut_vs_pt_cut_vs_muon_pt": hist.Hist.new.Bool(
                name="ip3d_cut", label="ip3d_cut"
            )
            .Bool(name="mini_iso_cut", label="mini_iso_cut")
            .Bool(name="neutral_iso_cut", label="neutral_iso_cut")
            .Bool(name="pt_cut", label="pt_cut")
            .Regular(
                100,
                3,
                300,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Weight(),
        }

        output = {
            dataset: {
                "cutflow": cutflow,
                "gensumweight": processor.value_accumulator(float, 0),
                "vars": pandas_accumulator(pd.DataFrame()),
                "histograms": histograms,
            },
        }

        # gen weights
        if self.isMC:
            self.gensumweight = ak.sum(events.genWeight)
            output[dataset]["gensumweight"].add(self.gensumweight)

        # run the analysis
        self.analysis(events, output)

        return output

    def postprocess(self, accumulator):
        pass
