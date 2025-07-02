import awkward as ak
import hist
import numpy as np
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

    def trigger_plateau(self, events):
        muons = events.Muon
        events, muons = events[ak.num(muons) > 2], muons[ak.num(muons) > 2]

        muons1 = muons[muons.charge == 1]
        muons2 = muons[muons.charge == -1]
        enough_muons = (ak.num(muons1) > 0) & (ak.num(muons2) > 0)
        muons1 = muons1[enough_muons]
        muons2 = muons2[enough_muons]
        muons = muons[enough_muons]
        events = events[enough_muons]

        muon_pairs = ak.unzip(ak.cartesian([muons1, muons2]))

        dz = muon_pairs[0].dz - muon_pairs[1].dz  # type: ignore[attr-defined]
        muons1 = muon_pairs[0][abs(dz) < 0.2]  # type: ignore[attr-defined]
        muons2 = muon_pairs[1][abs(dz) < 0.2]  # type: ignore[attr-defined]

        os_dimuons = muons1 + muons2
        events = events[ak.sum(os_dimuons.mass > 3.8, axis=-1) > 0]

        return events

    def apply_CR_prompt(self, events):
        """
        Apply the CR_prompt selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

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
        ]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        (
            events,
            muons,
            Z_cands,
            candidates_indices,
            prompt_muons,
        ) = self.find_Z_candidates(
            events,
            muons,
            prompt_muons,
        )
        inside_mass_window = (
            abs(Z_cands.mass - SUEP_common.Z_MASS) < 3 * SUEP_common.Z_WIDTH
        )
        muons = muons[inside_mass_window]
        events = events[inside_mass_window]
        prompt_muons = prompt_muons[inside_mass_window]
        candidates_indices = candidates_indices[inside_mass_window]

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            (muons.miniIsoId < 3)
            & (abs(muons.dxy) > 0.01)
            & (abs(muons.dz) > 0.01)
            & (abs(muons.ip3d) > 0.015)
        ]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]
        prompt_muons = prompt_muons[select_by_muons_low]
        qcd_muons = qcd_muons[select_by_muons_low]

        return events, muons, prompt_muons, qcd_muons

    def apply_CR_cb(self, events):
        """
        Apply the CR_cb selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

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

        # CR_prompt
        (
            events_CR_prompt,
            muons_CR_prompt,
            prompt_muons_CR_prompt,
            qcd_muons_CR_prompt,
        ) = self.apply_CR_prompt(events_)
        if len(events_CR_prompt) > 0:
            weights_CR_prompt = self.get_weights(
                events_CR_prompt, do_vars=False, apply_lumi_factors=True
            )
            if self.isMC:
                prompt_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        prompt_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_prompt",
                        syst="",
                    ),
                    axis=-1,
                )
                prompt_muon_SFs_up = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        prompt_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_prompt",
                        syst="up",
                    ),
                    axis=-1,
                )
                prompt_muon_SFs_down = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        prompt_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_prompt",
                        syst="down",
                    ),
                    axis=-1,
                )
                qcd_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        qcd_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_qcd",
                        syst="",
                    ),
                    axis=-1,
                )
                qcd_muon_SFs_up = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        qcd_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_qcd",
                        syst="up",
                    ),
                    axis=-1,
                )
                qcd_muon_SFs_down = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        qcd_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_qcd",
                        syst="down",
                    ),
                    axis=-1,
                )
                weights_CR_prompt.add(
                    "MuonSF",
                    weight=prompt_muon_SFs * qcd_muon_SFs,
                    weightUp=prompt_muon_SFs_up * qcd_muon_SFs_up,
                    weightDown=prompt_muon_SFs_down * qcd_muon_SFs_down,
                )
            nMuon_CR_prompt = ak.num(muons_CR_prompt, axis=-1)
            nMuon_CR_prompt = ak.where(nMuon_CR_prompt > 5, 5, nMuon_CR_prompt)
            nMuon_CR_prompt_prompt_bd = ak.broadcast_arrays(
                nMuon_CR_prompt, prompt_muons_CR_prompt.pt
            )[0]
            weights_CR_prompt_prompt_bd = ak.broadcast_arrays(
                weights_CR_prompt.weight(), prompt_muons_CR_prompt.pt
            )[0]
            nMuon_CR_prompt_qcd_bd = ak.broadcast_arrays(
                nMuon_CR_prompt, qcd_muons_CR_prompt.pt
            )[0]
            weights_CR_prompt_qcd_bd = ak.broadcast_arrays(
                weights_CR_prompt.weight(), qcd_muons_CR_prompt.pt
            )[0]
            output[dataset]["histograms"]["CR_prompt_prompt_pt"].fill(
                ak.flatten(prompt_muons_CR_prompt.pt),
                ak.flatten(nMuon_CR_prompt_prompt_bd),
                weight=ak.flatten(weights_CR_prompt_prompt_bd),
            )
            output[dataset]["histograms"]["CR_prompt_prompt_eta"].fill(
                ak.flatten(prompt_muons_CR_prompt.eta),
                ak.flatten(nMuon_CR_prompt_prompt_bd),
                weight=ak.flatten(weights_CR_prompt_prompt_bd),
            )
            output[dataset]["histograms"]["CR_prompt_prompt_dxy"].fill(
                ak.flatten(abs(prompt_muons_CR_prompt.dxy)),
                ak.flatten(nMuon_CR_prompt_prompt_bd),
                weight=ak.flatten(weights_CR_prompt_prompt_bd),
            )
            output[dataset]["histograms"]["CR_prompt_prompt_dxyErr"].fill(
                ak.flatten(abs(prompt_muons_CR_prompt.dxyErr)),
                ak.flatten(nMuon_CR_prompt_prompt_bd),
                weight=ak.flatten(weights_CR_prompt_prompt_bd),
            )
            output[dataset]["histograms"]["CR_prompt_prompt_dz"].fill(
                ak.flatten(abs(prompt_muons_CR_prompt.dz)),
                ak.flatten(nMuon_CR_prompt_prompt_bd),
                weight=ak.flatten(weights_CR_prompt_prompt_bd),
            )
            output[dataset]["histograms"]["CR_prompt_prompt_dzErr"].fill(
                ak.flatten(abs(prompt_muons_CR_prompt.dzErr)),
                ak.flatten(nMuon_CR_prompt_prompt_bd),
                weight=ak.flatten(weights_CR_prompt_prompt_bd),
            )
            output[dataset]["histograms"]["CR_prompt_prompt_ip3d"].fill(
                ak.flatten(abs(prompt_muons_CR_prompt.ip3d)),
                ak.flatten(nMuon_CR_prompt_prompt_bd),
                weight=ak.flatten(weights_CR_prompt_prompt_bd),
            )
            output[dataset]["histograms"]["CR_prompt_prompt_iso"].fill(
                ak.flatten(prompt_muons_CR_prompt.miniPFRelIso_all),
                ak.flatten(nMuon_CR_prompt_prompt_bd),
                weight=ak.flatten(weights_CR_prompt_prompt_bd),
            )
            output[dataset]["histograms"]["CR_prompt_qcd_pt"].fill(
                ak.flatten(qcd_muons_CR_prompt.pt),
                ak.flatten(nMuon_CR_prompt_qcd_bd),
                weight=ak.flatten(weights_CR_prompt_qcd_bd),
            )
            output[dataset]["histograms"]["CR_prompt_qcd_eta"].fill(
                ak.flatten(qcd_muons_CR_prompt.eta),
                ak.flatten(nMuon_CR_prompt_qcd_bd),
                weight=ak.flatten(weights_CR_prompt_qcd_bd),
            )
            output[dataset]["histograms"]["CR_prompt_qcd_dxy"].fill(
                ak.flatten(abs(qcd_muons_CR_prompt.dxy)),
                ak.flatten(nMuon_CR_prompt_qcd_bd),
                weight=ak.flatten(weights_CR_prompt_qcd_bd),
            )
            output[dataset]["histograms"]["CR_prompt_qcd_dxyErr"].fill(
                ak.flatten(abs(qcd_muons_CR_prompt.dxyErr)),
                ak.flatten(nMuon_CR_prompt_qcd_bd),
                weight=ak.flatten(weights_CR_prompt_qcd_bd),
            )
            output[dataset]["histograms"]["CR_prompt_qcd_dz"].fill(
                ak.flatten(abs(qcd_muons_CR_prompt.dz)),
                ak.flatten(nMuon_CR_prompt_qcd_bd),
                weight=ak.flatten(weights_CR_prompt_qcd_bd),
            )
            output[dataset]["histograms"]["CR_prompt_qcd_dzErr"].fill(
                ak.flatten(abs(qcd_muons_CR_prompt.dzErr)),
                ak.flatten(nMuon_CR_prompt_qcd_bd),
                weight=ak.flatten(weights_CR_prompt_qcd_bd),
            )
            output[dataset]["histograms"]["CR_prompt_qcd_ip3d"].fill(
                ak.flatten(abs(qcd_muons_CR_prompt.ip3d)),
                ak.flatten(nMuon_CR_prompt_qcd_bd),
                weight=ak.flatten(weights_CR_prompt_qcd_bd),
            )
            output[dataset]["histograms"]["CR_prompt_qcd_iso"].fill(
                ak.flatten(qcd_muons_CR_prompt.miniPFRelIso_all),
                ak.flatten(nMuon_CR_prompt_qcd_bd),
                weight=ak.flatten(weights_CR_prompt_qcd_bd),
            )

        events_CR_cb, muons_CR_cb = self.apply_CR_cb(events_)
        if len(events_CR_cb) > 0:
            weights_CR_cb = self.get_weights(
                events_CR_cb, do_vars=False, apply_lumi_factors=True
            )
            if self.isMC:
                weights_CR_cb.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_CR_cb, era=self.era, region="CR_cb", syst=""
                        ),
                        axis=-1,
                    ),
                )
            nMuon_CR_cb = ak.num(muons_CR_cb, axis=-1)
            nMuon_CR_cb = ak.where(nMuon_CR_cb > 4, 4, nMuon_CR_cb)
            nMuon_CR_cb_bd = ak.broadcast_arrays(nMuon_CR_cb, muons_CR_cb.pt)[0]
            weights_CR_cb_bd = ak.broadcast_arrays(
                weights_CR_cb.weight(), muons_CR_cb.pt
            )[0]
            output[dataset]["histograms"]["CR_cb_pt"].fill(
                ak.flatten(muons_CR_cb.pt),
                ak.flatten(nMuon_CR_cb_bd),
                weight=ak.flatten(weights_CR_cb_bd),
            )
            output[dataset]["histograms"]["CR_cb_eta"].fill(
                ak.flatten(muons_CR_cb.eta),
                ak.flatten(nMuon_CR_cb_bd),
                weight=ak.flatten(weights_CR_cb_bd),
            )
            output[dataset]["histograms"]["CR_cb_dxy"].fill(
                ak.flatten(abs(muons_CR_cb.dxy)),
                ak.flatten(nMuon_CR_cb_bd),
                weight=ak.flatten(weights_CR_cb_bd),
            )
            output[dataset]["histograms"]["CR_cb_dxyErr"].fill(
                ak.flatten(abs(muons_CR_cb.dxyErr)),
                ak.flatten(nMuon_CR_cb_bd),
                weight=ak.flatten(weights_CR_cb_bd),
            )
            output[dataset]["histograms"]["CR_cb_dz"].fill(
                ak.flatten(abs(muons_CR_cb.dz)),
                ak.flatten(nMuon_CR_cb_bd),
                weight=ak.flatten(weights_CR_cb_bd),
            )
            output[dataset]["histograms"]["CR_cb_dzErr"].fill(
                ak.flatten(abs(muons_CR_cb.dzErr)),
                ak.flatten(nMuon_CR_cb_bd),
                weight=ak.flatten(weights_CR_cb_bd),
            )
            output[dataset]["histograms"]["CR_cb_ip3d"].fill(
                ak.flatten(abs(muons_CR_cb.ip3d)),
                ak.flatten(nMuon_CR_cb_bd),
                weight=ak.flatten(weights_CR_cb_bd),
            )
            output[dataset]["histograms"]["CR_cb_iso"].fill(
                ak.flatten(muons_CR_cb.miniPFRelIso_all),
                ak.flatten(nMuon_CR_cb_bd),
                weight=ak.flatten(weights_CR_cb_bd),
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
            "CR_prompt_prompt_pt": hist.Hist.new.Regular(
                15,
                20,
                200,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_prompt_eta": hist.Hist.new.Regular(
                10,
                -2.5,
                2.5,
                name="muon_eta",
                label="muon_eta",
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_prompt_dxy": hist.Hist.new.Regular(
                30,
                5e-6,
                2e-2,
                name="muon_dxy",
                label="muon_dxy",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_prompt_dxyErr": hist.Hist.new.Regular(
                20,
                4e-4,
                8e-3,
                name="muon_dxyErr",
                label="muon_dxyErr",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_prompt_dz": hist.Hist.new.Regular(
                20,
                1e-5,
                2e-2,
                name="muon_dz",
                label="muon_dz",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_prompt_dzErr": hist.Hist.new.Regular(
                20,
                1e-3,
                4e-2,
                name="muon_dzErr",
                label="muon_dzErr",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_prompt_ip3d": hist.Hist.new.Regular(
                20,
                2e-5,
                2e-2,
                name="muon_ip3d",
                label="muon_ip3d",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_prompt_iso": hist.Hist.new.Regular(
                30,
                2e-4,
                0.2,
                name="muon_iso",
                label="muon_iso",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_qcd_pt": hist.Hist.new.Regular(
                15,
                3,
                150,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_qcd_eta": hist.Hist.new.Regular(
                10,
                -2.5,
                2.5,
                name="muon_eta",
                label="muon_eta",
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_qcd_dxy": hist.Hist.new.Regular(
                20,
                5e-3,
                0.5,
                name="muon_dxy",
                label="muon_dxy",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_qcd_dxyErr": hist.Hist.new.Regular(
                20,
                6e-4,
                3e-2,
                name="muon_dxyErr",
                label="muon_dxyErr",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_qcd_dz": hist.Hist.new.Regular(
                20,
                5e-3,
                0.5,
                name="muon_dz",
                label="muon_dz",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_qcd_dzErr": hist.Hist.new.Regular(
                20,
                1e-3,
                0.1,
                name="muon_dzErr",
                label="muon_dzErr",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_qcd_ip3d": hist.Hist.new.Regular(
                20,
                1e-2,
                1,
                name="muon_ip3d",
                label="muon_ip3d",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt_qcd_iso": hist.Hist.new.Regular(
                30,
                1e-3,
                100,
                name="muon_iso",
                label="muon_iso",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_cb_pt": hist.Hist.new.Regular(
                15,
                3,
                150,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 1, 5, name="nMuon", label="nMuon")
            .Weight(),
            "CR_cb_eta": hist.Hist.new.Regular(
                10,
                -2.5,
                2.5,
                name="muon_eta",
                label="muon_eta",
            )
            .Regular(4, 1, 5, name="nMuon", label="nMuon")
            .Weight(),
            "CR_cb_dxy": hist.Hist.new.Regular(
                20,
                5e-3,
                0.5,
                name="muon_dxy",
                label="muon_dxy",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_cb_dxyErr": hist.Hist.new.Regular(
                20,
                7e-4,
                2e-2,
                name="muon_dxyErr",
                label="muon_dxyErr",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_cb_dz": hist.Hist.new.Regular(
                20,
                5e-4,
                0.5,
                name="muon_dz",
                label="muon_dz",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_cb_dzErr": hist.Hist.new.Regular(
                20,
                1e-3,
                0.1,
                name="muon_dzErr",
                label="muon_dzErr",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "CR_cb_ip3d": hist.Hist.new.Regular(
                20,
                5e-3,
                0.5,
                name="muon_ip3d",
                label="muon_ip3d",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 1, 5, name="nMuon", label="nMuon")
            .Weight(),
            "CR_cb_iso": hist.Hist.new.Regular(
                25,
                0.01,
                100,
                name="muon_iso",
                label="muon_iso",
                transform=hist.axis.transform.log,
            )
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
