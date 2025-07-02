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

    def apply_CR_prompt(
        self,
        events,
        prompt_dxyErr_cut=0,
        prompt_dzErr_cut=0,
        qcd_dxyErr_cut=0,
        qcd_dzErr_cut=0,
    ):
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
            & (abs(muons.dxy) < 0.2)
        )
        muons = muons[clean_muons]

        #### New code begin ####
        # Apply tight miniIso id corresponding to miniIso < 0.1
        prompt_muons = muons[
            (muons.pt > 25)
            & (muons.miniIsoId >= 3)
            & (abs(muons.dxy) < 0.01)
            & (abs(muons.dz) < 0.01)
            & (abs(muons.ip3d) < 0.01)
            & (muons.dxyErr > prompt_dxyErr_cut)
            & (muons.dzErr > prompt_dzErr_cut)
        ]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        events, muons, prompt_muons, Z_cands, candidates_indices = (
            self.find_Z_candidates(events, muons, prompt_muons)
        )
        inside_mass_window = (
            abs(Z_cands.mass - SUEP_common.Z_MASS) < 3 * SUEP_common.Z_WIDTH
        )
        muons = muons[inside_mass_window]
        events = events[inside_mass_window]
        prompt_muons = prompt_muons[inside_mass_window]
        candidates_indices = candidates_indices[inside_mass_window]
        #### New code end ####

        #### Old code begin ####
        # # Get the Z candidates and make sure they are close to the peak
        # events, muons, Z_cands, candidates_indices = self.find_Z_candidates(
        #     events, muons
        # )
        # inside_mass_window = (
        #     abs(Z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
        # )
        # muons = muons[inside_mass_window]
        # events = events[inside_mass_window]
        # candidates_indices = candidates_indices[inside_mass_window]

        # # Make sure both muons from the Z candidates are prompt
        # # Apply tight miniIso id corresponding to miniIso < 0.1
        # candidate_muons = muons[candidates_indices]
        # prompt_muons = muons[
        #     (candidate_muons.pt > 25)
        #     & (candidate_muons.miniIsoId >= 3)
        #     & (abs(candidate_muons.dxy) < 0.008)
        #     & (abs(candidate_muons.dz) < 0.01)
        #     & (abs(candidate_muons.ip3d) < 0.01)
        # ]
        # muons = muons[ak.num(prompt_muons, axis=-1) > 0]
        # events = events[ak.num(prompt_muons, axis=-1) > 0]
        # prompt_muons = prompt_muons[ak.num(prompt_muons, axis=-1) > 0]
        #### Old code end ####

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            (muons.miniIsoId < 3)
            & (abs(muons.dxy) > 0.01)
            & (abs(muons.dz) > 0.01)
            & (abs(muons.ip3d) > 0.015)
            & (muons.dxyErr > qcd_dxyErr_cut)
            & (muons.dzErr > qcd_dzErr_cut)
        ]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons

    def apply_CR_cb(self, events, dxyErr_cut=0, dzErr_cut=0):
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
            & (muons.dzErr > 0.003)
        )

        # Apply extra very tight cuts for CR_cb
        cb_muons = (abs(muons.dxy) >= 0.01) & (abs(muons.dxy) <= 0.2)
        err_cuts = (muons.dxyErr > dxyErr_cut) & (muons.dzErr > dzErr_cut)
        muons = muons[clean_muons & cb_muons & err_cuts]

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

        prompt_dxyErr_cuts, prompt_dzErr_cuts, qcd_dxyErr_cuts, qcd_dzErr_cuts, _ = (
            output[dataset]["histograms"]["CR_prompt"].axes
        )
        for prompt_dxyErr_cut in prompt_dxyErr_cuts.edges[:-1]:
            for prompt_dzErr_cut in prompt_dzErr_cuts.edges[:-1]:
                for qcd_dxyErr_cut in qcd_dxyErr_cuts.edges[:-1]:
                    for qcd_dzErr_cut in qcd_dzErr_cuts.edges[:-1]:
                        events_CR_prompt, muons_CR_prompt = self.apply_CR_prompt(
                            events_,
                            prompt_dxyErr_cut,
                            prompt_dzErr_cut,
                            qcd_dxyErr_cut,
                            qcd_dzErr_cut,
                        )
                        if len(events_CR_prompt) > 0:
                            weights_CR_prompt = self.get_weights(
                                events_CR_prompt, do_vars=True, apply_lumi_factors=True
                            )
                            if self.isMC:
                                weights_CR_prompt.add(
                                    "MuonSF",
                                    weight=ak.prod(
                                        muon_sf_utils.muon_efficiencies(
                                            muons_CR_prompt, self.era, peak="Z", syst=""
                                        ),
                                        axis=-1,
                                    ),
                                )
                            nMuon_CR_prompt = ak.num(muons_CR_prompt, axis=-1)
                            output[dataset]["histograms"]["CR_prompt"].fill(
                                prompt_dxyErr_cut
                                * 1.01
                                * ak.ones_like(nMuon_CR_prompt),
                                prompt_dzErr_cut * 1.01 * ak.ones_like(nMuon_CR_prompt),
                                qcd_dxyErr_cut * 1.01 * ak.ones_like(nMuon_CR_prompt),
                                qcd_dzErr_cut * 1.01 * ak.ones_like(nMuon_CR_prompt),
                                ak.where(nMuon_CR_prompt > 5, 5, nMuon_CR_prompt),
                                weight=weights_CR_prompt.weight(),
                            )

        dxyErr_cuts, dzErr_cuts, _ = output[dataset]["histograms"]["CR_cb"].axes
        for dxyErr_cut in dxyErr_cuts.edges[:-1]:
            for dzErr_cut in dzErr_cuts.edges[:-1]:
                events_CR_cb, muons_CR_cb = self.apply_CR_cb(events_)
                if len(events_CR_cb) > 0:
                    weights_CR_cb = self.get_weights(
                        events_CR_cb, do_vars=True, apply_lumi_factors=True
                    )
                    if self.isMC:
                        weights_CR_cb.add(
                            "MuonSF",
                            weight=ak.prod(
                                muon_sf_utils.muon_efficiencies(
                                    muons_CR_cb, self.era, peak="JPsi", syst=""
                                ),
                                axis=-1,
                            ),
                        )
                    nMuon_CR_cb = ak.num(muons_CR_cb, axis=-1)
                    output[dataset]["histograms"]["CR_cb"].fill(
                        dxyErr_cut * 1.01 * ak.ones_like(nMuon_CR_cb),
                        dzErr_cut * 1.01 * ak.ones_like(nMuon_CR_cb),
                        ak.where(nMuon_CR_cb > 4, 4, nMuon_CR_cb),
                        weight=weights_CR_cb.weight(),
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
        # events = self.trigger_plateau(events)

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
                8, 5e-4, 3e-3, name="prompt dxyErr cut", label="prompt dxyErr cut"
            )
            .Regular(7, 8e-4, 5e-3, name="prompt dzErr cut", label="prompt dzErr cut")
            .Regular(7, 7e-4, 4e-3, name="qcd dxyErr cut", label="qcd dxyErr cut")
            .Regular(4, 1e-3, 5e-3, name="qcd dzErr cut", label="qcd dzErr cut")
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_cb": hist.Hist.new.Regular(
                9, 6e-4, 5e-3, name="dxyErr cut", label="dxyErr cut"
            )
            .Regular(8, 7e-4, 5e-3, name="dzErr cut", label="dzErr cut")
            .Regular(4, 1, 5, name="nMuon", label="nMuon")
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
