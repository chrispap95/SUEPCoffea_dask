import itertools
from typing import Optional

import awkward as ak
import hist
import numpy as np
import vector  # type: ignore[import]
from coffea import processor

# Set vector behavior
vector.register_awkward()


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
        # # Pileup weights (need to be fed with integers)
        # pu_weights = pileup_weight(
        #     self.era, ak.values_astype(events.Pileup.nTrueInt, np.int32)
        # )
        # # L1 prefire weights
        # prefire_weights = GetPrefireWeights(events)
        # # Trigger scale factors
        # # To be implemented
        return events.genWeight  # * pu_weights * prefire_weights

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

        ip3d_cuts = 1.001 * np.logspace(-3, -1, 11)[:-1]
        mini_iso_cuts = 1.001 * np.logspace(-1, 1, 11)[:-1]
        pt_cuts = 1.001 * np.linspace(30, 60, 11)[:-1]

        for (
            ip3d_cut,
            mini_iso_cut,
            pt_cut,
        ) in itertools.product(ip3d_cuts, mini_iso_cuts, pt_cuts):
            cut = (
                (muons.pt < pt_cut)
                & (muons.ip3d < ip3d_cut)
                & (muons.miniPFRelIso_all < mini_iso_cut)
                & ((muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 0.5)
            )

            output[dataset]["histograms"][
                "ip3d_cut_vs_mini_iso_cut_vs_pt_cut_vs_nMuon"
            ].fill(
                ip3d_cut,
                mini_iso_cut,
                pt_cut,
                ak.sum(cut, axis=-1),
                weight=weights,
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

        # # golden jsons for offline data
        # if not self.isMC:
        #     events = golden_json_utils.apply_golden_JSON(events, self.era)

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
            "ip3d_cut_vs_mini_iso_cut_vs_pt_cut_vs_nMuon": hist.Hist.new.Reg(
                10,
                1e-3,
                0.1,
                name="ip3d_cut",
                label="ip3d_cut",
                transform=hist.axis.transform.log,
            )
            .Reg(
                10,
                0.1,
                10,
                name="mini_iso_cut",
                label="mini_iso_cut",
                transform=hist.axis.transform.log,
            )
            .Reg(10, 30, 60, name="pt_cut", label="pt_cut")
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
        }

        output = {
            dataset: {
                "cutflow": cutflow,
                "gensumweight": processor.value_accumulator(float, 0),
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
