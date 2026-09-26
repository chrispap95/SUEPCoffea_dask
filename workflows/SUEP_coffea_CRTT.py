import awkward as ak
import hist
import vector  # type: ignore[import]
from coffea import processor

import workflows.SUEP_common as SUEP_common

# Importing CMS corrections
from workflows.CMS_corrections import (
    golden_json_utils,
    muon_sf_utils,
    systematics_utils,
)

# Set vector behavior
vector.register_awkward()


class SUEP_processor(SUEP_common.SUEP_base):
    def __init__(
        self,
        isMC: bool,
        era: str | int,
        do_syst: bool = False,
        do_rochester: bool = False,
        do_lhepdfsyst: bool = False,
    ) -> None:
        self.isMC = isMC
        self.era = era if isinstance(era, str) else str(era)
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.do_rochester = do_rochester
        self.do_lhepdfsyst = do_lhepdfsyst

    def apply_CR_TT(self, events):
        """
        Apply the CR_TT selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        clean_muons = (
            (muons.tightId)
            & (muons.pfRelIso04_all < 0.15)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        pt_selection = (ak.sum(muons.pt > 25, axis=-1) >= 1) & (
            ak.sum(muons.pt > 20, axis=-1) >= 2
        )
        events = events[pt_selection]
        muons = muons[pt_selection]

        jets = events.Jet
        at_least_2_bjets = ak.sum(jets.btagDeepFlavB > 0.2783, axis=-1) >= 2
        events = events[at_least_2_bjets]
        muons = muons[at_least_2_bjets]

        return events, muons

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        events_CR_TT, muons_CR_TT = self.apply_CR_TT(events_)
        if len(events_CR_TT) > 0:
            weights_CR_TT = self.get_weights(
                events_CR_TT, do_vars=True, apply_lumi_factors=True
            )
            nMuon_CR_TT = ak.num(muons_CR_TT, axis=-1)
            output[dataset]["histograms"]["CR_TT"].fill(
                ak.where(nMuon_CR_TT > 4, 4, nMuon_CR_TT),
                weight=weights_CR_TT.weight(),
            )

        return

    def analysis(self, events, output):
        # get dataset name
        dataset = events.metadata["dataset"]

        # take care of weights
        weights = self.get_weights(events)

        # Fill the cutflow columns for all
        output[dataset]["cutflow"].fill(len(events) * ["all"], weight=weights.weight())

        # golden jsons for offline data
        if not self.isMC:
            events = golden_json_utils.apply_golden_JSON(events, self.era)

        events = self.trigger_selection(events)
        trigger_plateau = self.apply_trigger_plateau(events, pt3_threshold=4)
        events = events[trigger_plateau]

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
            weight=weights.weight(),
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
            "CR_TT": hist.Hist.new.Regular(
                4, 2, 6, name="nMuon", label="nMuon"
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
