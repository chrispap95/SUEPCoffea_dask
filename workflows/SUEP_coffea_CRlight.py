import awkward as ak
import hist
import vector  # type: ignore[import]
from coffea import processor

# Importing CMS corrections
import workflows.CMS_corrections.golden_json_utils as golden_json_utils
import workflows.CMS_corrections.muon_sf_utils as muon_sf_utils
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

    def apply_CR_light(self, events):
        """
        Apply the CR_light selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dz) < 0.2)
        )

        # Apply extra very tight cuts for CR_light
        prompt_muons = (
            (abs(muons.dxy) <= 0.02)
            & (abs(muons.dz) <= 0.1)
            & (abs(muons.ip3d) <= 0.02)
        )
        non_isolated_muons = muons.miniPFRelIso_all > 0.65
        light_muons = prompt_muons & non_isolated_muons
        muons = muons[clean_muons & light_muons]

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

        events_CR_light, muons_CR_light = self.apply_CR_light(events_)
        if len(events_CR_light) > 0:
            weights_CR_light = self.get_weights(events_CR_light, do_vars=True)
            weights_CR_light.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(muons_CR_light, self.era, syst=""),
                    axis=-1,
                ),
                weightUp=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_CR_light, self.era, syst="up"
                    ),
                    axis=-1,
                ),
                weightDown=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_CR_light, self.era, syst="down"
                    ),
                    axis=-1,
                ),
            )
            output[dataset]["histograms"]["CR_light"].fill(
                ak.num(muons_CR_light, axis=-1),
                weight=weights_CR_light.weight(),
            )
            if self.do_syst:
                for syst in weights_CR_light.variations:
                    output[dataset]["histograms"][f"CR_light_{syst}"] = (
                        output[dataset]["histograms"]["CR_light"].copy().reset()
                    )
                    output[dataset]["histograms"][f"CR_light_{syst}"].fill(
                        ak.num(muons_CR_light, axis=-1),
                        weight=weights_CR_light.weight(syst),
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
        output[dataset]["cutflow"].fill(len(events) * ["all"], weight=weights.weight())

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
            "CR_light": hist.Hist.new.Regular(
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
