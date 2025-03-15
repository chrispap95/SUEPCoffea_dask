import awkward as ak
import hist
import vector  # type: ignore[import]
from coffea import processor
from prompt_toolkit import prompt

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

    def apply_CR_prompt(self, events):
        """
        Apply the CR_prompt selection to the events.
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

        # Apply extra very tight cuts for CR_prompt
        # prompt_muons = (
        #     (muons.pt > 25)
        #     & (muons.miniPFRelIso_all < 0.1)
        #     & (abs(muons.dxy) < 0.005)
        #     & (abs(muons.dz) < 0.01)
        #     & (abs(muons.ip3d < 0.008))
        # )

        # Get the Z candidates and make sure they are close to the peak
        muons = muons[clean_muons]
        events, muons, Z_cands, candidates_indices = self.find_Z_candidates(
            events, muons
        )
        inside_mass_window = (
            abs(Z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
        )
        muons = muons[inside_mass_window]
        events = events[inside_mass_window]
        candidates_indices = candidates_indices[inside_mass_window]

        # Make sure both muons from the Z candidates are prompt
        candidate_muons = muons[candidates_indices]
        prompt_muons = muons[
            (candidate_muons.pt > 25)
            & (candidate_muons.miniPFRelIso_all < 0.1)
            & (abs(candidate_muons.dxy) < 0.008)
            & (abs(candidate_muons.dz) < 0.01)
            & (abs(candidate_muons.ip3d) < 0.01)
        ]
        muons = muons[ak.num(prompt_muons, axis=-1) > 0]
        events = events[ak.num(prompt_muons, axis=-1) > 0]
        prompt_muons = prompt_muons[ak.num(prompt_muons, axis=-1) > 0]

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            (muons.miniPFRelIso_all > 0.1)
            & (abs(muons.dxy) > 0.01)
            & (abs(muons.dz) > 0.01)
            & (abs(muons.ip3d) > 0.015)
        ]
        # if len(qcd_muons) > 0:
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)
        # else:
        #     muons = prompt_muons

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_high = True  # ak.num(muons, axis=-1) < 5
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
        select_by_muons_high = True  # ak.num(muons, axis=-1) < 5
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

        # Apply extra very tight cuts for CR_cb
        cb_muons = (abs(muons.dxy) >= 0.01) & (abs(muons.dxy) <= 0.2)
        muons = muons[clean_muons & cb_muons]

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_high = True  # ak.num(muons, axis=-1) < 5
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
        if len(events_CR_prompt) > 0:
            weights_CR_prompt = self.get_weights(events_CR_prompt)
            weights_CR_prompt.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(muons_CR_prompt, syst=""),
                    axis=-1,
                ),
                weightUp=ak.prod(
                    muon_sf_utils.muon_efficiencies(muons_CR_prompt, syst="up"),
                    axis=-1,
                ),
                weightDown=ak.prod(
                    muon_sf_utils.muon_efficiencies(muons_CR_prompt, syst="down"),
                    axis=-1,
                ),
            )
            nMuon_CR_prompt = ak.num(muons_CR_prompt, axis=-1)
            output[dataset]["histograms"]["CR_prompt"].fill(
                ak.where(nMuon_CR_prompt > 5, 5, nMuon_CR_prompt),
                weight=weights_CR_prompt.weight(),
            )
            if self.do_syst:
                for syst in weights_CR_prompt.variations:
                    output[dataset]["histograms"][f"CR_prompt_{syst}"] = (
                        output[dataset]["histograms"]["CR_prompt"].copy().reset()
                    )
                    output[dataset]["histograms"][f"CR_prompt_{syst}"].fill(
                        ak.where(nMuon_CR_prompt > 5, 5, nMuon_CR_prompt),
                        weight=weights_CR_prompt.weight(syst),
                    )

        events_CR_light, muons_CR_light = self.apply_CR_light(events_)
        if len(events_CR_light) > 0:
            weights_CR_light = self.get_weights(events_CR_light)
            weights_CR_light.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(muons_CR_light, syst=""),
                    axis=-1,
                ),
                weightUp=ak.prod(
                    muon_sf_utils.muon_efficiencies(muons_CR_light, syst="up"),
                    axis=-1,
                ),
                weightDown=ak.prod(
                    muon_sf_utils.muon_efficiencies(muons_CR_light, syst="down"),
                    axis=-1,
                ),
            )
            nMuon_CR_light = ak.num(muons_CR_light, axis=-1)
            output[dataset]["histograms"]["CR_light"].fill(
                ak.where(nMuon_CR_light > 4, 4, nMuon_CR_light),
                weight=weights_CR_light.weight(),
            )
            if self.do_syst:
                for syst in weights_CR_light.variations:
                    output[dataset]["histograms"][f"CR_light_{syst}"] = (
                        output[dataset]["histograms"]["CR_light"].copy().reset()
                    )
                    output[dataset]["histograms"][f"CR_light_{syst}"].fill(
                        ak.where(nMuon_CR_light > 4, 4, nMuon_CR_light),
                        weight=weights_CR_light.weight(syst),
                    )

        events_CR_cb, muons_CR_cb = self.apply_CR_cb(events_)
        if len(events_CR_cb) > 0:
            weights_CR_cb = self.get_weights(events_CR_cb)
            weights_CR_cb.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(muons_CR_cb, syst=""),
                    axis=-1,
                ),
                weightUp=ak.prod(
                    muon_sf_utils.muon_efficiencies(muons_CR_cb, syst="up"),
                    axis=-1,
                ),
                weightDown=ak.prod(
                    muon_sf_utils.muon_efficiencies(muons_CR_cb, syst="down"),
                    axis=-1,
                ),
            )
            nMuon_CR_cb = ak.num(muons_CR_cb, axis=-1)
            output[dataset]["histograms"]["CR_cb"].fill(
                ak.where(nMuon_CR_cb > 4, 4, nMuon_CR_cb),
                weight=weights_CR_cb.weight(),
            )
            if self.do_syst:
                for syst in weights_CR_cb.variations:
                    output[dataset]["histograms"][f"CR_cb_{syst}"] = (
                        output[dataset]["histograms"]["CR_cb"].copy().reset()
                    )
                    output[dataset]["histograms"][f"CR_cb_{syst}"].fill(
                        ak.where(nMuon_CR_cb > 4, 4, nMuon_CR_cb),
                        weight=weights_CR_cb.weight(syst),
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
            "CR_prompt": hist.Hist.new.Regular(
                4, 2, 6, name="nMuon", label="nMuon"
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
