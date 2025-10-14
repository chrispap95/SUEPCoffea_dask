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
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Apply tight miniIso id corresponding to miniIso < 0.1
        prompt_muons = muons[
            (muons.pt > 25)
            & (muons.miniIsoId >= 3)
            & (abs(muons.dxy) < 0.01)
            & (abs(muons.dz) < 0.01)
            # & (abs(muons.ip3d) < 0.01)
        ]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        events, prompt_muons, Z_cands, candidates_indices, muons = (
            self.find_Z_candidates(events, prompt_muons, muons, apply_dR_cut=False)
        )
        inside_mass_window = (
            abs(Z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
        )
        muons = muons[inside_mass_window]
        events = events[inside_mass_window]
        prompt_muons = prompt_muons[inside_mass_window]
        candidates_indices = candidates_indices[inside_mass_window]

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            (muons.miniIsoId < 3)  # miniIsoId < 3 corresponds to miniIso > 0.1
            & (abs(muons.dxy) > 0.01)
            & (abs(muons.dz) > 0.01)
            # & (abs(muons.ip3d) > 0.015)
            # & (muons.dzErr > 0.003)
        ]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons, prompt_muons, qcd_muons

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
            # & (muons.dzErr > 0.003)
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

        (
            events_CR_prompt,
            muons_CR_prompt,
            prompt_muons_CR_prompt,
            qcd_muons_CR_prompt,
        ) = self.apply_CR_prompt(events_)

        if len(events_CR_prompt) > 0:
            muon_pairs_0, muon_pairs_1 = self.find_dimuon_pairs(muons_CR_prompt)
            dimuon_dr_mask = muon_pairs_0.delta_r(muon_pairs_1) < 0.3
            dimuon_mass = (muon_pairs_0 + muon_pairs_1).mass
            dimuon_mass_mask = ((dimuon_mass > 2.7) & (dimuon_mass < 3.5)) | (
                (dimuon_mass > 8.8) & (dimuon_mass < 11.2)
            )
            events_CR_prompt = events_CR_prompt[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muons_CR_prompt = muons_CR_prompt[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            prompt_muons_CR_prompt = prompt_muons_CR_prompt[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            qcd_muons_CR_prompt = qcd_muons_CR_prompt[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muon_pairs_0 = muon_pairs_0[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muon_pairs_1 = muon_pairs_1[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]

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
                    qcd_muon_SFs = ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            qcd_muons_CR_prompt,
                            era=self.era,
                            region="CR_prompt_qcd",
                            syst="",
                        ),
                        axis=-1,
                    )
                    weights_CR_prompt.add(
                        "MuonSF",
                        weight=prompt_muon_SFs * qcd_muon_SFs,
                    )
                nMuon_CR_prompt = ak.num(muons_CR_prompt, axis=-1)
                nMuon_CR_prompt = ak.where(nMuon_CR_prompt > 5, 5, nMuon_CR_prompt)
                output[dataset]["histograms"]["CR_prompt_muon_pt"].fill(
                    ak.flatten(muons_CR_prompt.pt),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_CR_prompt, muons_CR_prompt.pt)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_CR_prompt.weight(), muons_CR_prompt.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["CR_prompt_muon_eta"].fill(
                    ak.flatten(muons_CR_prompt.eta),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_CR_prompt, muons_CR_prompt.eta)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_CR_prompt.weight(), muons_CR_prompt.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["CR_prompt_muon_eta"].fill(
                    ak.flatten(muons_CR_prompt.eta),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_CR_prompt, muons_CR_prompt.eta)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_CR_prompt.weight(), muons_CR_prompt.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["CR_prompt_muon_phi"].fill(
                    ak.flatten(muons_CR_prompt.phi),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_CR_prompt, muons_CR_prompt.phi)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_CR_prompt.weight(), muons_CR_prompt.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["CR_prompt_muon_iso"].fill(
                    ak.flatten(muons_CR_prompt.miniPFRelIso_all),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_CR_prompt,
                            muons_CR_prompt.miniPFRelIso_all,
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_CR_prompt.weight(), muons_CR_prompt.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["CR_prompt_muon_dxy"].fill(
                    ak.flatten(muons_CR_prompt.dxy),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_CR_prompt, muons_CR_prompt.dxy)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_CR_prompt.weight(), muons_CR_prompt.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["CR_prompt_muon_dz"].fill(
                    ak.flatten(muons_CR_prompt.dz),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_CR_prompt, muons_CR_prompt.dz)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_CR_prompt.weight(), muons_CR_prompt.pt
                        )[0]
                    ),
                )
                muon_pairs_dr = muon_pairs_0.delta_r(muon_pairs_1)
                output[dataset]["histograms"]["CR_prompt_dimuon_dr"].fill(
                    ak.flatten(muon_pairs_dr),
                    ak.flatten(ak.broadcast_arrays(nMuon_CR_prompt, muon_pairs_dr)[0]),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_CR_prompt.weight(), muon_pairs_dr)[
                            0
                        ]
                    ),
                )
                muon_pairs_mass = (muon_pairs_0 + muon_pairs_1).mass
                output[dataset]["histograms"]["CR_prompt_dimuon_mass"].fill(
                    ak.flatten(muon_pairs_mass),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_CR_prompt, muon_pairs_mass)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_CR_prompt.weight(), muon_pairs_mass
                        )[0]
                    ),
                )

        events_CR_cb, muons_CR_cb = self.apply_CR_cb(events_)
        if len(events_CR_cb) > 0:
            muon_pairs_0, muon_pairs_1 = self.find_dimuon_pairs(muons_CR_cb)
            dimuon_dr_mask = muon_pairs_0.delta_r(muon_pairs_1) < 0.3
            dimuon_mass = (muon_pairs_0 + muon_pairs_1).mass
            dimuon_mass_mask = ((dimuon_mass > 2.7) & (dimuon_mass < 3.5)) | (
                (dimuon_mass > 8.8) & (dimuon_mass < 11.2)
            )
            events_CR_cb = events_CR_cb[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muons_CR_cb = muons_CR_cb[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muon_pairs_0 = muon_pairs_0[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muon_pairs_1 = muon_pairs_1[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]

            if len(events_CR_cb) > 0:
                weights_CR_cb = self.get_weights(
                    events_CR_cb, do_vars=False, apply_lumi_factors=True
                )
                if self.isMC:
                    weights_CR_cb.add(
                        "MuonSF",
                        weight=ak.prod(
                            muon_sf_utils.muon_efficiencies(
                                muons_CR_cb, self.era, region="CR_cb", syst=""
                            ),
                            axis=-1,
                        ),
                    )
                nMuon_CR_cb = ak.num(muons_CR_cb, axis=-1)
                nMuon_CR_cb = ak.where(nMuon_CR_cb > 4, 4, nMuon_CR_cb)
                output[dataset]["histograms"]["CR_cb_muon_pt"].fill(
                    ak.flatten(muons_CR_cb.pt),
                    ak.flatten(ak.broadcast_arrays(nMuon_CR_cb, muons_CR_cb.pt)[0]),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_CR_cb.weight(), muons_CR_cb.pt)[0]
                    ),
                )
                output[dataset]["histograms"]["CR_cb_muon_eta"].fill(
                    ak.flatten(muons_CR_cb.eta),
                    ak.flatten(ak.broadcast_arrays(nMuon_CR_cb, muons_CR_cb.eta)[0]),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_CR_cb.weight(), muons_CR_cb.pt)[0]
                    ),
                )
                output[dataset]["histograms"]["CR_cb_muon_eta"].fill(
                    ak.flatten(muons_CR_cb.eta),
                    ak.flatten(ak.broadcast_arrays(nMuon_CR_cb, muons_CR_cb.eta)[0]),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_CR_cb.weight(), muons_CR_cb.pt)[0]
                    ),
                )
                output[dataset]["histograms"]["CR_cb_muon_phi"].fill(
                    ak.flatten(muons_CR_cb.phi),
                    ak.flatten(ak.broadcast_arrays(nMuon_CR_cb, muons_CR_cb.phi)[0]),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_CR_cb.weight(), muons_CR_cb.pt)[0]
                    ),
                )
                output[dataset]["histograms"]["CR_cb_muon_iso"].fill(
                    ak.flatten(muons_CR_cb.miniPFRelIso_all),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_CR_cb,
                            muons_CR_cb.miniPFRelIso_all,
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_CR_cb.weight(), muons_CR_cb.pt)[0]
                    ),
                )
                output[dataset]["histograms"]["CR_cb_muon_dxy"].fill(
                    ak.flatten(muons_CR_cb.dxy),
                    ak.flatten(ak.broadcast_arrays(nMuon_CR_cb, muons_CR_cb.dxy)[0]),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_CR_cb.weight(), muons_CR_cb.pt)[0]
                    ),
                )
                output[dataset]["histograms"]["CR_cb_muon_dz"].fill(
                    ak.flatten(muons_CR_cb.dz),
                    ak.flatten(ak.broadcast_arrays(nMuon_CR_cb, muons_CR_cb.dz)[0]),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_CR_cb.weight(), muons_CR_cb.pt)[0]
                    ),
                )
                muon_pairs_dr = muon_pairs_0.delta_r(muon_pairs_1)
                output[dataset]["histograms"]["CR_cb_dimuon_dr"].fill(
                    ak.flatten(muon_pairs_dr),
                    ak.flatten(ak.broadcast_arrays(nMuon_CR_cb, muon_pairs_dr)[0]),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_CR_cb.weight(), muon_pairs_dr)[0]
                    ),
                )
                muon_pairs_mass = (muon_pairs_0 + muon_pairs_1).mass
                output[dataset]["histograms"]["CR_cb_dimuon_mass"].fill(
                    ak.flatten(muon_pairs_mass),
                    ak.flatten(ak.broadcast_arrays(nMuon_CR_cb, muon_pairs_mass)[0]),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_CR_cb.weight(), muon_pairs_mass)[0]
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
            f"CR_prompt_muon_pt": hist.Hist.new.Regular(
                50,
                3,
                300,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_prompt_muon_eta": hist.Hist.new.Regular(
                50,
                -3,
                3,
                name="muon_eta",
                label="muon_eta",
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_prompt_muon_phi": hist.Hist.new.Regular(
                50,
                -np.pi,
                np.pi,
                name="muon_phi",
                label="muon_phi",
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_prompt_muon_iso": hist.Hist.new.Regular(
                50,
                0.01,
                10,
                name="muon_iso",
                label="muon_iso",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_prompt_muon_dxy": hist.Hist.new.Regular(
                50,
                1e-4,
                1,
                name="muon_dxy",
                label="muon_dxy",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_prompt_muon_dz": hist.Hist.new.Regular(
                50,
                1e-4,
                1,
                name="muon_dz",
                label="muon_dz",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_prompt_dimuon_dr": hist.Hist.new.Regular(
                50,
                1e-2,
                10,
                name="dimuon_dr",
                label="dimuon_dr",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_prompt_dimuon_mass": hist.Hist.new.Regular(
                50,
                0.1,
                300,
                name="dimuon_mass",
                label="dimuon_mass",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_cb_muon_pt": hist.Hist.new.Regular(
                50,
                3,
                300,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 1, 5, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_cb_muon_eta": hist.Hist.new.Regular(
                50,
                -3,
                3,
                name="muon_eta",
                label="muon_eta",
            )
            .Regular(4, 1, 5, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_cb_muon_phi": hist.Hist.new.Regular(
                50,
                -np.pi,
                np.pi,
                name="muon_phi",
                label="muon_phi",
            )
            .Regular(4, 1, 5, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_cb_muon_iso": hist.Hist.new.Regular(
                50,
                0.01,
                10,
                name="muon_iso",
                label="muon_iso",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_cb_muon_dxy": hist.Hist.new.Regular(
                50,
                1e-4,
                1,
                name="muon_dxy",
                label="muon_dxy",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 1, 5, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_cb_muon_dz": hist.Hist.new.Regular(
                50,
                1e-4,
                1,
                name="muon_dz",
                label="muon_dz",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 1, 5, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_cb_dimuon_dr": hist.Hist.new.Regular(
                50,
                1e-2,
                10,
                name="dimuon_dr",
                label="dimuon_dr",
                transform=hist.axis.transform.log,
            )
            .Regular(4, 1, 5, name="nMuon", label="nMuon")
            .Weight(),
            f"CR_cb_dimuon_mass": hist.Hist.new.Regular(
                50,
                0.1,
                100,
                name="dimuon_mass",
                label="dimuon_mass",
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
