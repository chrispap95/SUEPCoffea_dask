import awkward as ak
import hist
import numpy as np
import vector  # type: ignore[import]
from coffea import processor

# Importing CMS corrections
import workflows.CMS_corrections.golden_json_utils as golden_json_utils
import workflows.CMS_corrections.systematics_utils as systematics_utils

# Set vector behavior
vector.register_awkward()


class SUEP_cluster(processor.ProcessorABC):
    def __init__(
        self,
        isMC: bool,
        era: str | int,
        syst_var: str = "",
    ) -> None:
        self.isMC = isMC
        self.era = era if isinstance(era, str) else str(era)
        self.syst_var = syst_var
        self.syst_suffix = f"_sys_{syst_var}" if syst_var != "" else ""
        self.gensumweight = 1.0

    def trigger_selection(self, events):
        """
        Applies trigger, returns events.
        """
        trigger = np.zeros(len(events), dtype=bool)
        if self.era in ["2016", "2016APV"]:
            if "TripleMu_5_3_3" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_5_3_3 == 1)
            if "TripleMu_5_3_3_DZ_Mass3p8" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_5_3_3_DZ_Mass3p8 == 1)
            if "TripleMu_12_10_5" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_12_10_5 == 1)
        elif self.era == "2017":
            if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1)
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_10_5_5_DZ == 1)
            if "TripleMu_12_10_5" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_12_10_5 == 1)
        elif self.era == "2018":
            if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1)
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1)
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_10_5_5_DZ == 1)
            if "TripleMu_12_10_5" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_12_10_5 == 1)
        elif self.era in ["2022", "2023"]:
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1)
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_10_5_5_DZ == 1)
            if "TripleMu_12_10_5" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_12_10_5 == 1)
        else:
            raise ValueError(f"Invalid era: {self.era}")
        events = events[trigger]
        return events

    def get_weights(self, events):
        if not self.isMC:
            return np.ones(len(events))
        # Pileup weights (need to be fed with integers)
        pu_weights = systematics_utils.pileup_weight(
            self.era, ak.values_astype(events.Pileup.nTrueInt, np.int32)
        )
        # L1 prefire weights
        prefire_weights = systematics_utils.get_prefire_weights(events)
        # Trigger scale factors
        # To be implemented
        return events.genWeight * pu_weights * prefire_weights

    def muon_filter(self, events):
        """
        Filter events after the TripleMu trigger.
        Cleans muons and electrons.
        Requires at least nMuons with mediumId, pt, dz, and eta cuts.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dz) < 0.2)
        )
        muons = muons[clean_muons]
        select_by_muons_high = ak.num(muons, axis=-1) < 5
        select_by_muons_low = ak.num(muons, axis=-1) > 2
        events = events[select_by_muons_high & select_by_muons_low]

        return events

    def apply_CR_prompt(self, events):
        """
        Apply the CR_prompt selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        # Apply basic muon cuts
        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dz) < 0.2)
        )

        # Apply extra very tight cuts for CR_prompt
        prompt_muons = (
            (events.Muon.pt > 25)
            & (events.Muon.miniPFRelIso_all < 0.1)
            & (abs(events.Muon.dxy) < 0.005)
            & (abs(events.Muon.dz) < 0.01)
            & (abs(events.Muon.ip3d < 0.008))
        )
        muons = muons[clean_muons & prompt_muons]

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_high = ak.num(muons, axis=-1) < 5
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_high & select_by_muons_low]
        muons = muons[select_by_muons_high & select_by_muons_low]

        return events, muons

    def apply_CR_light(self, events):
        """
        Apply the CR_light selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        # Apply basic muon cuts
        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dz) < 0.2)
        )

        # Apply extra very tight cuts for CR_light
        prompt_muons = (
            (abs(events.Muon.dxy) <= 0.02)
            & (abs(events.Muon.dz) <= 0.1)
            & (abs(events.Muon.ip3d) <= 0.02)
        )
        non_isolated_muons = events.Muon.miniPFRelIso_all > 0.65
        light_muons = prompt_muons & non_isolated_muons
        muons = muons[clean_muons & light_muons]

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_high = ak.num(muons, axis=-1) < 5
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_high & select_by_muons_low]
        muons = muons[select_by_muons_high & select_by_muons_low]

        return events, muons

    def apply_CR_cb(self, events):
        """
        Apply the CR_cb selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        # Apply basic muon cuts
        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dz) < 0.2)
        )

        # Apply extra very tight cuts for CR_cb
        cb_muons = (abs(events.Muon.dxy) >= 0.01) & (abs(events.Muon.dxy) <= 0.2)
        muons = muons[clean_muons & cb_muons]

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_high = ak.num(muons, axis=-1) < 5
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_high & select_by_muons_low]
        muons = muons[select_by_muons_high & select_by_muons_low]

        return events, muons

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        events_CR_prompt, muons_CR_prompt = self.apply_CR_prompt(events_)
        weights_CR_prompt = self.get_weights(events_CR_prompt)
        output[dataset]["histograms"]["CR_prompt"].fill(
            ak.num(muons_CR_prompt, axis=-1),
            weight=weights_CR_prompt,
        )

        events_CR_light, muons_CR_light = self.apply_CR_light(events_)
        weights_CR_light = self.get_weights(events_CR_light)
        output[dataset]["histograms"]["CR_light"].fill(
            ak.num(muons_CR_light, axis=-1),
            weight=weights_CR_light,
        )

        events_CR_cb, muons_CR_cb = self.apply_CR_cb(events_)
        weights_CR_cb = self.get_weights(events_CR_cb)
        output[dataset]["histograms"]["CR_cb"].fill(
            ak.num(muons_CR_cb, axis=-1),
            weight=weights_CR_cb,
        )

        return

    def analysis(self, events, output):
        #######################################################################
        # ---- Trigger event selection
        # Cut based on ak4 jets to replicate the trigger
        #######################################################################

        # get dataset name
        dataset = events.metadata["dataset"]

        # take care of weights
        weights = self.get_weights(events)

        # Fill the cutflow columns for all
        output[dataset]["cutflow"].fill(len(events) * ["all"], weight=weights)

        # golden jsons for offline data
        if not self.isMC:
            events = golden_json_utils.apply_golden_JSON(events, self.era)

        events = self.trigger_selection(events)

        # Apply HT selection for WJets stiching
        if "WJetsToLNu_HT" in dataset:
            events = events[events.LHE.HT >= 70]
        elif "WJetsToLNu_TuneCP5" in dataset:
            events = events[events.LHE.HT < 70]

        # Keep only events with Zpt == 0 for the bug in LHEPt binned samples
        if "DYJetsToLL_LHEFilterPtZ-0_MatchEWPDG20" in dataset:
            events = events[events.LHE.Vpt == 0]

        weights = self.get_weights(events)

        # Fill the cutflow columns for trigger
        output[dataset]["cutflow"].fill(
            len(events) * ["trigger"],
            weight=weights,
        )

        self.fill_histograms(events, output)

        return

    def process(self, events):
        dataset = events.metadata["dataset"]
        cutflow = hist.Hist.new.StrCategory(
            ["all", "trigger"],
            name="cutflow",
            label="cutflow",
        ).Weight()
        histograms = {
            "CR_prompt": hist.Hist.new.Regular(
                4, 1, 5, name="nMuon", label="nMuon"
            ).Weight(),
            "CR_light": hist.Hist.new.Regular(
                4, 1, 5, name="nMuon", label="nMuon"
            ).Weight(),
            "CR_cb": hist.Hist.new.Regular(
                4, 1, 5, name="nMuon", label="nMuon"
            ).Weight(),
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
