import awkward as ak
import hist
import numpy as np
import scipy.special  # type: ignore[import]
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

    def apply_CR_prompt(self, events, apply_dxy_cut=True, apply_dz_cut=True):
        """
        Apply the CR_prompt selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        # Apply basic muon cuts
        clean_muons = (muons.mediumId) & (muons.pt > 3) & (abs(muons.eta) < 2.4)
        if apply_dxy_cut:
            clean_muons = clean_muons & (abs(muons.dxy) < 0.2)
        if apply_dz_cut:
            clean_muons = clean_muons & (abs(muons.dz) < 0.2)
        muons = muons[clean_muons]

        # Apply tight miniIso id corresponding to miniIso < 0.1
        prompt_muon_mask = (muons.pt > 25) & (muons.miniIsoId >= 3)
        if apply_dxy_cut:
            prompt_muon_mask = prompt_muon_mask & (abs(muons.dxy) < 0.01)
        if apply_dz_cut:
            prompt_muon_mask = prompt_muon_mask & (abs(muons.dz) < 0.01)
        prompt_muons = muons[prompt_muon_mask]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        # events, prompt_muons, Z_cands, candidates_indices, muons = (
        #     self.find_Z_candidates(events, prompt_muons, muons, apply_dR_cut=False)
        # )
        # inside_mass_window = (
        #     abs(Z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
        # )
        # muons = muons[inside_mass_window]
        # events = events[inside_mass_window]
        # prompt_muons = prompt_muons[inside_mass_window]
        # candidates_indices = candidates_indices[inside_mass_window]

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons_mask = (
            muons.miniIsoId < 3  # miniIsoId < 3 corresponds to miniIso > 0.1
        )
        if apply_dxy_cut:
            qcd_muons_mask = qcd_muons_mask & (abs(muons.dxy) > 0.01)
        if apply_dz_cut:
            qcd_muons_mask = qcd_muons_mask & (abs(muons.dz) > 0.01)
        qcd_muons = muons[qcd_muons_mask]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons, prompt_muons, qcd_muons

    def apply_CR_cb(self, events, apply_dxy_cut=True, apply_dz_cut=True):
        """
        Apply the CR_cb selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        # Apply basic muon cuts
        clean_muons = (muons.mediumId) & (muons.pt > 3) & (abs(muons.eta) < 2.4)
        if apply_dz_cut:
            clean_muons = clean_muons & (abs(muons.dz) < 0.2)

        # Apply extra very tight cuts for CR_cb
        cb_muons = muons.pt > 0
        if apply_dxy_cut:
            cb_muons = cb_muons & (abs(muons.dxy) >= 0.01) & (abs(muons.dxy) <= 0.2)
        muons = muons[clean_muons & cb_muons]

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        (
            events_CR_prompt,
            muons_CR_prompt,
            prompt_muons_CR_prompt,
            qcd_muons_CR_prompt,
        ) = self.apply_CR_prompt(events_)
        if len(events_CR_prompt) > 0:
            weights = self.get_weights(
                events_CR_prompt, do_vars=False, apply_lumi_factors=True
            )
            output[dataset]["histograms"]["muon_dxy_CR_prompt_prompt"].fill(
                ak.flatten(prompt_muons_CR_prompt.dxy),
                weight=ak.flatten(
                    ak.broadcast_arrays(weights.weight(), prompt_muons_CR_prompt.dxy)[0]
                ),
            )
            output[dataset]["histograms"]["muon_dz_CR_prompt_prompt"].fill(
                ak.flatten(prompt_muons_CR_prompt.dz),
                weight=ak.flatten(
                    ak.broadcast_arrays(weights.weight(), prompt_muons_CR_prompt.dz)[0]
                ),
            )
            output[dataset]["histograms"]["muon_dxy_CR_prompt_qcd"].fill(
                ak.flatten(qcd_muons_CR_prompt.dxy),
                weight=ak.flatten(
                    ak.broadcast_arrays(weights.weight(), qcd_muons_CR_prompt.dxy)[0]
                ),
            )
            output[dataset]["histograms"]["muon_dz_CR_prompt_qcd"].fill(
                ak.flatten(qcd_muons_CR_prompt.dz),
                weight=ak.flatten(
                    ak.broadcast_arrays(weights.weight(), qcd_muons_CR_prompt.dz)[0]
                ),
            )

        events_CR_cb, muons_CR_cb = self.apply_CR_cb(events_)
        if len(events_CR_cb) > 0:
            weights = self.get_weights(
                events_CR_cb, do_vars=False, apply_lumi_factors=True
            )
            output[dataset]["histograms"]["muon_dxy_CR_cb"].fill(
                ak.flatten(muons_CR_cb.dxy),
                weight=ak.flatten(
                    ak.broadcast_arrays(weights.weight(), muons_CR_cb.dxy)[0]
                ),
            )
            output[dataset]["histograms"]["muon_dz_CR_cb"].fill(
                ak.flatten(muons_CR_cb.dz),
                weight=ak.flatten(
                    ak.broadcast_arrays(weights.weight(), muons_CR_cb.dz)[0]
                ),
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
        trigger_plateau = self.apply_trigger_plateau(events)
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
            "muon_dxy_CR_prompt_prompt": hist.Hist.new.Regular(
                40,
                -0.01,
                0.01,
                name="muon_dxy",
                label="muon_dxy",
            ).Weight(),
            "muon_dz_CR_prompt_prompt": hist.Hist.new.Regular(
                40,
                -0.01,
                0.01,
                name="muon_dz",
                label="muon_dz",
            ).Weight(),
            "muon_dxy_CR_prompt_qcd": hist.Hist.new.Regular(
                40,
                -0.2,
                0.2,
                name="muon_dxy",
                label="muon_dxy",
            ).Weight(),
            "muon_dz_CR_prompt_qcd": hist.Hist.new.Regular(
                40,
                -0.2,
                0.2,
                name="muon_dz",
                label="muon_dz",
            ).Weight(),
            "muon_dxy_CR_cb": hist.Hist.new.Regular(
                40,
                -0.2,
                0.2,
                name="muon_dxy",
                label="muon_dxy",
            ).Weight(),
            "muon_dz_CR_cb": hist.Hist.new.Regular(
                40,
                -0.2,
                0.2,
                name="muon_dz",
                label="muon_dz",
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
