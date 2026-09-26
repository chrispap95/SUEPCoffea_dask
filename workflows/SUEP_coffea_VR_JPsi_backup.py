import awkward as ak
import hist
import vector  # type: ignore[import]
from coffea import processor

import workflows.SUEP_common as SUEP_common

# Importing CMS corrections
from workflows.CMS_corrections import golden_json_utils, muon_sf_utils

# Set vector behavior
vector.register_awkward()

Z_MASS = 91.1876
Z_WIDTH = 2.4952


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

    def apply_VR(self, events):
        """
        Apply the VR selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 1], muons[ak.num(muons) > 1]

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 5)  # I suppose this is to reduce signal contamination
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Make sure we are the trigger plateau
        events, muons = events[ak.num(muons) > 2], muons[ak.num(muons) > 2]

        # Find dimuon pairs and require a J/psi resonance in the event
        muon_pairs_0, muon_pairs_1, muon_pairs_idx_0, muon_pairs_idx_1 = (  # type: ignore[assignment]
            self.find_dimuon_pairs(muons, return_indices=True)
        )
        if len(muon_pairs_0):
            dimuon_dr = muon_pairs_0.delta_r(muon_pairs_1)
            dimuon_mass = (muon_pairs_0 + muon_pairs_1).mass
            dimuon_dr_mask = dimuon_dr < 0.3

            # J/psi resonance window
            jpsi_mask = (dimuon_mass > 2.7) & (dimuon_mass < 3.5)
            # Resonances whose muons we tag and remove (low-mass, J/psi, Upsilon)
            dimuon_mass_mask = (
                ((dimuon_mass > 0.4) & (dimuon_mass < 0.8))
                | jpsi_mask
                | ((dimuon_mass > 8.8) & (dimuon_mass < 11.2))
            )

            # Keep only events that contain a J/psi candidate
            # has_jpsi = ak.sum(dimuon_dr_mask & jpsi_mask, axis=-1) == 1
            has_jpsi = ak.sum(jpsi_mask, axis=-1) == 1
            events = events[has_jpsi]
            muons = muons[has_jpsi]
            muon_pairs_idx_0 = muon_pairs_idx_0[has_jpsi]
            muon_pairs_idx_1 = muon_pairs_idx_1[has_jpsi]
            dimuon_dr_mask = dimuon_dr_mask[has_jpsi]
            dimuon_mass_mask = dimuon_mass_mask[has_jpsi]
            dimuon_mass = dimuon_mass[has_jpsi]
            jpsi_mask = jpsi_mask[has_jpsi]

            # Remove the resonance dimuon
            dimuon_mass = dimuon_mass[~jpsi_mask]

            # Tag the resonance muons (incl. the J/psi pair) and remove them
            # res_mu_idx_0 = muon_pairs_idx_0[dimuon_dr_mask & jpsi_mask]
            # res_mu_idx_1 = muon_pairs_idx_1[dimuon_dr_mask & jpsi_mask]
            res_mu_idx_0 = muon_pairs_idx_0[jpsi_mask]
            res_mu_idx_1 = muon_pairs_idx_1[jpsi_mask]
            res_mu_idx = ak.concatenate([res_mu_idx_0, res_mu_idx_1], axis=-1)

            pairs = ak.cartesian(
                {"idx": ak.local_index(muons), "rm": res_mu_idx},
                axis=1,
                nested=True,  # make a sublist over "rm" for each muon
            )

            # True if this (muon, remove) pair matches
            same = pairs["idx"] == pairs["rm"]
            # per muon: True if its index is in res_mu_idx
            remove_mask = ak.any(same, axis=-1)
            keep_mask = ~remove_mask  # invert to keep the others

            muons = muons[keep_mask]
        else:
            # No OS dimuon pairs in the batch -> no J/psi -> drop all events
            events = events[:0]
            muons = muons[:0]
            dimuon_mass = ak.Array([])

        # Form loose VR & make sure there is at least one muon in the event after the cuts
        muons_VR_loose = muons[
            ((abs(muons.dxy) > 0.01) | (abs(muons.dz) > 0.01))
            & (muons.miniPFRelIso_all > 0.2)
        ]
        events_VR_loose = events[ak.num(muons_VR_loose, axis=-1) > 0]
        dimuon_mass_VR_loose = dimuon_mass[ak.num(muons_VR_loose, axis=-1) > 0]
        muons_VR_loose = muons_VR_loose[ak.num(muons_VR_loose, axis=-1) > 0]

        # # Cut on the max OS dimuon mass
        # muons1 = muons_VR_loose[muons_VR_loose.charge == 1]
        # muons2 = muons_VR_loose[muons_VR_loose.charge == -1]
        # enough_muons = (ak.num(muons1) > 0) & (ak.num(muons2) > 0)
        # muons1 = muons1[enough_muons]
        # muons2 = muons2[enough_muons]
        # muon_pairs = ak.unzip(ak.cartesian([muons1, muons2]))
        # os_dimuons = muon_pairs[0] + muon_pairs[1]  # type: ignore[index]
        # events_VR_loose = events_VR_loose[ak.max(os_dimuons.mass, axis=-1) > 20]  # type: ignore[op_type]
        # muons_VR_loose = muons_VR_loose[ak.max(os_dimuons.mass, axis=-1) > 20]  # type: ignore[op_type]

        # Form tight VR & make sure there is at least one muon in the event after the cuts
        muons_VR_tight = muons[
            ((abs(muons.dxy) > 0.02) | (abs(muons.dz) > 0.02))
            & (muons.miniPFRelIso_all > 0.4)
        ]
        events_VR_tight = events[ak.num(muons_VR_tight, axis=-1) > 0]
        dimuon_mass_VR_tight = dimuon_mass[ak.num(muons_VR_tight, axis=-1) > 0]
        muons_VR_tight = muons_VR_tight[ak.num(muons_VR_tight, axis=-1) > 0]

        # # Cut on the max OS dimuon mass
        # muons1 = muons_VR_tight[muons_VR_tight.charge == 1]
        # muons2 = muons_VR_tight[muons_VR_tight.charge == -1]
        # enough_muons = (ak.num(muons1) > 0) & (ak.num(muons2) > 0)
        # muons1 = muons1[enough_muons]
        # muons2 = muons2[enough_muons]
        # muon_pairs = ak.unzip(ak.cartesian([muons1, muons2]))
        # os_dimuons = muon_pairs[0] + muon_pairs[1]  # type: ignore[index]
        # events_VR_tight = events_VR_tight[ak.max(os_dimuons.mass, axis=-1) > 20]  # type: ignore[op_type]
        # muons_VR_tight = muons_VR_tight[ak.max(os_dimuons.mass, axis=-1) > 20]  # type: ignore[op_type]

        return (
            events_VR_tight,
            events_VR_loose,
            muons_VR_tight,
            muons_VR_loose,
            dimuon_mass_VR_tight,
            dimuon_mass_VR_loose,
        )

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        (
            events_VR_tight,
            events_VR_loose,
            muons_VR_tight,
            muons_VR_loose,
            dimuon_mass_VR_tight,
            dimuon_mass_VR_loose,
        ) = self.apply_VR(events_)

        if len(events_VR_tight) > 0:
            weights_VR_tight = self.get_weights(
                events_VR_tight, do_vars=True, apply_lumi_factors=True
            )
            if self.isMC:
                weights_VR_tight.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_VR_tight, era=self.era, region="VR_tight", syst=""
                        ),
                        axis=-1,
                    ),
                )
            nMuon_VR_tight = ak.num(muons_VR_tight, axis=-1)
            nMuon_VR_tight = ak.where(nMuon_VR_tight > 7, 7, nMuon_VR_tight)
            output[dataset]["histograms"]["VR_tight"].fill(
                nMuon_VR_tight,
                weight=weights_VR_tight.weight(),
            )
            output[dataset]["histograms"]["VR_tight_mass"].fill(
                ak.flatten(dimuon_mass_VR_tight),
                ak.flatten(
                    ak.broadcast_arrays(nMuon_VR_tight, dimuon_mass_VR_tight)[0]
                ),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        weights_VR_tight.weight(), dimuon_mass_VR_tight
                    )[0]
                ),
            )
            output[dataset]["histograms"]["VR_tight_pt"].fill(
                ak.flatten(muons_VR_tight.pt),
                ak.flatten(ak.broadcast_arrays(nMuon_VR_tight, muons_VR_tight.pt)[0]),
                weight=ak.flatten(
                    ak.broadcast_arrays(weights_VR_tight.weight(), muons_VR_tight.pt)[0]
                ),
            )

        if len(events_VR_loose) > 0:
            weights_VR_loose = self.get_weights(
                events_VR_loose, do_vars=True, apply_lumi_factors=True
            )
            if self.isMC:
                weights_VR_loose.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_VR_loose, era=self.era, region="VR_loose", syst=""
                        ),
                        axis=-1,
                    ),
                )
            nMuon_VR_loose = ak.num(muons_VR_loose, axis=-1)
            output[dataset]["histograms"]["VR_loose"].fill(
                ak.where(nMuon_VR_loose > 7, 7, nMuon_VR_loose),
                weight=weights_VR_loose.weight(),
            )
            output[dataset]["histograms"]["VR_loose_mass"].fill(
                ak.flatten(dimuon_mass_VR_loose),
                ak.flatten(
                    ak.broadcast_arrays(nMuon_VR_loose, dimuon_mass_VR_loose)[0]
                ),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        weights_VR_loose.weight(), dimuon_mass_VR_loose
                    )[0]
                ),
            )
            output[dataset]["histograms"]["VR_loose_pt"].fill(
                ak.flatten(muons_VR_loose.pt),
                ak.flatten(ak.broadcast_arrays(nMuon_VR_loose, muons_VR_loose.pt)[0]),
                weight=ak.flatten(
                    ak.broadcast_arrays(weights_VR_loose.weight(), muons_VR_loose.pt)[0]
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
            "VR_tight": hist.Hist.new.Regular(
                7, 1, 8, name="nMuon", label="nMuon"
            ).Weight(),
            "VR_loose": hist.Hist.new.Regular(
                7, 1, 8, name="nMuon", label="nMuon"
            ).Weight(),
            "VR_loose_mass": hist.Hist.new.Regular(
                50,
                0.1,
                100,
                name="dimuon_mass",
                label="dimuon_mass",
                transform=hist.axis.transform.log,
            )
            .Regular(7, 1, 8, name="nMuon", label="nMuon")
            .Weight(),
            "VR_tight_mass": hist.Hist.new.Regular(
                50,
                0.1,
                100,
                name="dimuon_mass",
                label="dimuon_mass",
                transform=hist.axis.transform.log,
            )
            .Regular(7, 1, 8, name="nMuon", label="nMuon")
            .Weight(),
            "VR_loose_pt": hist.Hist.new.Regular(
                50,
                1,
                100,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(7, 1, 8, name="nMuon", label="nMuon")
            .Weight(),
            "VR_tight_pt": hist.Hist.new.Regular(
                50,
                1,
                100,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(7, 1, 8, name="nMuon", label="nMuon")
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
