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
        prefire_weights = events.L1PreFiringWeight.Nom
        # Trigger scale factors
        # To be implemented
        return events.genWeight * pu_weights * prefire_weights

    def muon_filter(self, events):
        """
        Filter events after the TripleMu trigger. Cleans muons.
        Requires at least nMuons with mediumId, pt, dxy, dz, and eta cuts.
        """
        muons = events.Muon
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
        muons = muons[select_by_muons_high & select_by_muons_low]

        # Cuts for this CR
        CR_requirement = (abs(muons.dxy) >= 0.01) & (abs(muons.dxy) <= 0.2)
        muons = muons[CR_requirement]

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_final = ak.num(CR_requirement, axis=-1) > 0
        events = events[select_by_muons_final]
        muons = muons[select_by_muons_final]

        return events, muons

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_, muons = self.muon_filter(events)
        if (len(events_) == 0) or (len(muons) == 0):
            return

        weights = self.get_weights(events_)

        nMuon = ak.num(muons, axis=-1)

        output[dataset]["histograms"]["CR_cb"].fill(
            ak.where(nMuon > 5, 5, nMuon),
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

        # golden jsons for offline data
        if not self.isMC:
            events = golden_json_utils.apply_golden_JSON(events, self.era)

        events = self.trigger_selection(events)

        # Apply HT selection for WJets stiching
        if "WJetsToLNu_HT" in dataset:
            events = events[events.LHE.HT >= 70]
        elif "WJetsToLNu_TuneCP5" in dataset:
            events = events[events.LHE.HT < 70]

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
            "CR_cb": hist.Hist.new.Regular(
                6, 0, 6, name="nMuon", label="nMuon"
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
