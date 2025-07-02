import awkward as ak
import hist
import vector  # type: ignore[import]
from coffea import processor

# Importing CMS corrections
import workflows.CMS_corrections.golden_json_utils as golden_json_utils
import workflows.SUEP_common as SUEP_common

# Set vector behavior
vector.register_awkward()


class SUEP_processor(SUEP_common.SUEP_base):
    def __init__(
        self,
        isMC: bool,
        era: str | int,
        do_syst: bool = False,
        do_rochester: bool = False,
    ) -> None:
        self.isMC = isMC
        self.era = era if isinstance(era, str) else str(era)
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.do_rochester = do_rochester

    def analysis(self, events, output):
        dataset = events.metadata["dataset"]

        # apply golden JSON for data
        if not self.isMC:
            events = golden_json_utils.apply_golden_JSON(events, self.era)

        events_ = events[events.HLT.IsoMu27]
        muons = events_.Muon
        clean_muons = (
            (muons.pt > 27)
            & (abs(muons.eta) < 2.4)
            & (muons.mediumId)
            & (muons.pfIsoId > 1)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]
        events_ = events_[ak.num(muons) > 0]
        if len(events_) > 0:
            weights = self.get_weights(events_)
            output[dataset]["histograms"]["HLT_IsoMu27"].fill(
                len(events_) * ["HLT_IsoMu27"],
                weight=weights.weight(),
            )

        events_ = events[events.HLT.Mu50]
        muons = events_.Muon
        clean_muons = (
            (muons.pt > 50)
            & (abs(muons.eta) < 2.4)
            & (muons.mediumId)
            & (muons.pfIsoId > 1)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]
        events_ = events_[ak.num(muons) > 0]
        if len(events_) > 0:
            weights = self.get_weights(events_)
            output[dataset]["histograms"]["HLT_Mu50"].fill(
                len(events_) * ["HLT_Mu50"],
                weight=weights.weight(),
            )

        events_ = events[events.HLT.TripleMu_12_10_5]
        muons = events_.Muon
        clean_muons = (
            (muons.pt > 5)
            & (abs(muons.eta) < 2.4)
            & (muons.mediumId)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        trigger_plateau = (
            (ak.sum(muons.pt > 7, axis=1) >= 3)
            & (ak.sum(muons.pt > 12, axis=1) >= 2)
            & (ak.sum(muons.pt > 14, axis=1) >= 1)
        )

        events_ = events_[trigger_plateau]
        if len(events_) > 0:
            weights = self.get_weights(events_)
            output[dataset]["histograms"]["HLT_TripleMu_12_10_5"].fill(
                len(events_) * ["HLT_TripleMu_12_10_5"],
                weight=weights.weight(),
            )

        return

    def process(self, events):
        dataset = events.metadata["dataset"]
        histograms = {
            "HLT_IsoMu27": hist.Hist.new.StrCat(
                ["HLT_IsoMu27"],
                name="events",
                label="events",
            ).Weight(),
            "HLT_Mu50": hist.Hist.new.StrCat(
                ["HLT_Mu50"],
                name="events",
                label="events",
            ).Weight(),
            "HLT_TripleMu_12_10_5": hist.Hist.new.StrCat(
                ["HLT_TripleMu_12_10_5"],
                name="events",
                label="events",
            ).Weight(),
        }

        output = {
            dataset: {
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
