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
        isMC: bool,
        era: str | int,
        syst_var: str = "",
    ) -> None:
        self.isMC = isMC
        self.era = era if isinstance(era, str) else str(era)
        self.syst_var = syst_var
        self.syst_suffix = f"_sys_{syst_var}" if syst_var != "" else ""
        self.gensumweight = 1.0

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

        # Fill histograms
        scalar = events_.GenPart[
            (events_.GenPart.pdgId == 25) & (events_.GenPart.status == 62)
        ]
        output[dataset]["histograms"]["SUEP_pt"].fill(
            ak.firsts(scalar.pt, axis=-1),
            weight=weights,
        )
        output[dataset]["histograms"]["SUEP_eta"].fill(
            ak.firsts(scalar.eta, axis=-1),
            weight=weights,
        )
        output[dataset]["histograms"]["SUEP_phi"].fill(
            ak.firsts(scalar.phi, axis=-1),
            weight=weights,
        )

        output[dataset]["histograms"]["Muon_pt"].fill(
            ak.flatten(muons.pt),
            weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
        )
        output[dataset]["histograms"]["Muon_eta"].fill(
            ak.flatten(muons.eta),
            weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
        )
        output[dataset]["histograms"]["Muon_phi"].fill(
            ak.flatten(muons.phi),
            weight=ak.flatten(ak.broadcast_arrays(weights, muons.pt)[0]),
        )

        output[dataset]["histograms"]["nMuon"].fill(
            ak.num(muons),
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
            events = events[events.LHE.HT >= 70]
        elif "WJetsToLNu_TuneCP5" in dataset:
            events = events[events.LHE.HT < 70]

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
            "SUEP_pt": hist.Hist.new.Reg(
                50,
                0,
                300,
                name="SUEP_pt",
                label="SUEP_pt",
            ).Weight(),
            "SUEP_eta": hist.Hist.new.Reg(
                50,
                -5,
                5,
                name="SUEP_eta",
                label="SUEP_eta",
            ).Weight(),
            "SUEP_phi": hist.Hist.new.Reg(
                50,
                -np.pi,
                np.pi,
                name="SUEP_phi",
                label="SUEP_phi",
            ).Weight(),
            "Muon_pt": hist.Hist.new.Reg(
                50,
                3,
                300,
                name="Muon_pt",
                label="Muon_pt",
                transform=hist.axis.transform.log,
            ).Weight(),
            "Muon_eta": hist.Hist.new.Reg(
                50,
                -2.5,
                2.55,
                name="Muon_eta",
                label="Muon_eta",
            ).Weight(),
            "Muon_phi": hist.Hist.new.Reg(
                50,
                -np.pi,
                np.pi,
                name="Muon_phi",
                label="Muon_phi",
            ).Weight(),
            "nMuon": hist.Hist.new.Reg(
                20,
                0,
                20,
                name="nMuon",
                label="nMuon",
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
