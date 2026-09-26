import awkward as ak
import hist
import numpy as np
import vector  # type: ignore[import]
from coffea import processor

import workflows.SUEP_common as SUEP_common

# Importing CMS corrections
from workflows.CMS_corrections import golden_json_utils, muon_sf_utils

# Set vector behavior
vector.register_awkward()

Z_MASS = 91.1876
Z_WIDTH = 2.4952

JET_PT_MIN = 15
JET_ETA_MAX = 2.4
JET_MUON_DR_MAX = 0.4


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

        # Form every possible opposite-sign dimuon pair. The shared helper uses
        # greedy one-to-one matching, which would omit pairs that share a muon.
        positive_muons = muons[muons.charge == 1]
        negative_muons = muons[muons.charge == -1]
        muon_indices = ak.local_index(muons)
        positive_indices = muon_indices[muons.charge == 1]
        negative_indices = muon_indices[muons.charge == -1]
        muon_pairs = ak.cartesian(
            {"positive": positive_muons, "negative": negative_muons},
            axis=1,
            nested=False,
        )
        muon_pair_indices = ak.cartesian(
            {"positive": positive_indices, "negative": negative_indices},
            axis=1,
            nested=False,
        )
        muon_pairs_0 = muon_pairs["positive"]
        muon_pairs_1 = muon_pairs["negative"]
        muon_pairs_idx_0 = muon_pair_indices["positive"]
        muon_pairs_idx_1 = muon_pair_indices["negative"]
        if len(muon_pairs_0):
            dimuon_mass = (muon_pairs_0 + muon_pairs_1).mass

            # J/psi resonance window
            jpsi_mask = (dimuon_mass > 2.7) & (dimuon_mass < 3.5)

            # Keep only events with exactly one J/psi among all OS pairs.
            n_jpsi = ak.sum(jpsi_mask, axis=-1)
            has_jpsi = n_jpsi == 1
            events = events[has_jpsi]
            muons = muons[has_jpsi]
            muon_pairs_idx_0 = muon_pairs_idx_0[has_jpsi]
            muon_pairs_idx_1 = muon_pairs_idx_1[has_jpsi]
            dimuon_mass = dimuon_mass[has_jpsi]
            jpsi_mask = jpsi_mask[has_jpsi]

            # Save the unique J/psi mass, then remove its two daughter muons.
            jpsi_mass = ak.firsts(dimuon_mass[jpsi_mask])
            jpsi_idx_0 = ak.firsts(muon_pairs_idx_0[jpsi_mask])
            jpsi_idx_1 = ak.firsts(muon_pairs_idx_1[jpsi_mask])
            muon_indices = ak.local_index(muons)
            muons = muons[(muon_indices != jpsi_idx_0) & (muon_indices != jpsi_idx_1)]
        else:
            # No OS dimuon pairs in the batch -> no J/psi -> drop all events
            events = events[:0]
            muons = muons[:0]
            jpsi_mass = ak.Array([])

        # Form loose VR & make sure there is at least one muon in the event after the cuts
        muons_VR_loose = muons[
            ((abs(muons.dxy) > 0.01) | (abs(muons.dz) > 0.01))
            & (muons.miniPFRelIso_all > 0.2)
        ]
        has_loose_muon = ak.num(muons_VR_loose, axis=-1) > 0
        events_VR_loose = events[has_loose_muon]
        jpsi_mass_VR_loose = jpsi_mass[has_loose_muon]
        muons_VR_loose = muons_VR_loose[has_loose_muon]

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
        has_tight_muon = ak.num(muons_VR_tight, axis=-1) > 0
        events_VR_tight = events[has_tight_muon]
        jpsi_mass_VR_tight = jpsi_mass[has_tight_muon]
        muons_VR_tight = muons_VR_tight[has_tight_muon]

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

        return {
            "VR_tight": (
                events_VR_tight,
                muons_VR_tight,
                jpsi_mass_VR_tight,
            ),
            "VR_loose": (
                events_VR_loose,
                muons_VR_loose,
                jpsi_mass_VR_loose,
            ),
        }

    def get_jet_matched_muons(self, events, muons):
        """Return muons within dR < 0.4 of a clean AK4 jet."""
        jets = events.Jet
        passes_tight_jet_id = (jets.jetId & 2) != 0
        passes_tight_pu_id = ak.ones_like(jets.pt, dtype=bool)
        if "puId" in jets.fields:
            passes_tight_pu_id = (jets.pt >= 50) | ((jets.puId & 4) != 0)
        jets = jets[
            (jets.pt > JET_PT_MIN)
            & (abs(jets.eta) < JET_ETA_MAX)
            & passes_tight_jet_id
            & passes_tight_pu_id
        ]

        muon_jet_pairs = ak.cartesian({"muon": muons, "jet": jets}, axis=1, nested=True)
        delta_eta = muon_jet_pairs["muon"].eta - muon_jet_pairs["jet"].eta
        delta_phi = abs(muon_jet_pairs["muon"].phi - muon_jet_pairs["jet"].phi)
        delta_phi = ak.where(delta_phi > np.pi, 2 * np.pi - delta_phi, delta_phi)
        delta_r = np.sqrt(delta_eta**2 + delta_phi**2)  # type: ignore[operator]
        matched_to_jet = ak.any(
            delta_r < JET_MUON_DR_MAX,
            axis=-1,
        )
        return muons[matched_to_jet]

    def fill_region_histograms(
        self, events, muons, jpsi_mass, output, region, sf_region
    ):
        """Fill one inclusive or jet-matched loose/tight histogram set."""
        if len(events) == 0:
            return

        dataset = events.metadata["dataset"]
        weights = self.get_weights(events, do_vars=True, apply_lumi_factors=True)
        if self.isMC:
            weights.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons, era=self.era, region=sf_region, syst=""
                    ),
                    axis=-1,
                ),
            )

        n_muon = ak.num(muons, axis=-1)
        n_muon_capped = ak.where(n_muon > 7, 7, n_muon)
        output[dataset]["histograms"][region].fill(
            n_muon_capped,
            weight=weights.weight(),
        )
        output[dataset]["histograms"][f"{region}_mass"].fill(
            jpsi_mass,
            n_muon_capped,
            weight=weights.weight(),
        )
        output[dataset]["histograms"][f"{region}_pt"].fill(
            ak.flatten(muons.pt),
            ak.flatten(ak.broadcast_arrays(n_muon_capped, muons.pt)[0]),
            weight=ak.flatten(ak.broadcast_arrays(weights.weight(), muons.pt)[0]),
        )

    def fill_histograms(self, events, output):
        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        regions = self.apply_VR(events_)
        for sf_region in ("VR_tight", "VR_loose"):
            region_events, region_muons, jpsi_mass = regions[sf_region]
            self.fill_region_histograms(
                region_events,
                region_muons,
                jpsi_mass,
                output,
                region=sf_region,
                sf_region=sf_region,
            )

            jet_matched_muons = self.get_jet_matched_muons(region_events, region_muons)
            has_jet_matched_muon = ak.num(jet_matched_muons, axis=-1) > 0
            self.fill_region_histograms(
                region_events[has_jet_matched_muon],
                jet_matched_muons[has_jet_matched_muon],
                jpsi_mass[has_jet_matched_muon],
                output,
                region=f"{sf_region}_jetMatched",
                sf_region=sf_region,
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

        def make_nmuon_histogram():
            return hist.Hist.new.Regular(7, 1, 8, name="nMuon", label="nMuon").Weight()

        def make_mass_histogram():
            return (
                hist.Hist.new.Regular(
                    50,
                    2.7,
                    3.5,
                    name="dimuon_mass",
                    label=r"$m_{\mu\mu}$ (GeV)",
                )
                .Regular(7, 1, 8, name="nMuon", label="nMuon")
                .Weight()
            )

        def make_pt_histogram():
            return (
                hist.Hist.new.Regular(
                    50,
                    1,
                    100,
                    name="muon_pt",
                    label="muon_pt",
                    transform=hist.axis.transform.log,
                )
                .Regular(7, 1, 8, name="nMuon", label="nMuon")
                .Weight()
            )

        histograms = {}
        for region in (
            "VR_tight",
            "VR_loose",
            "VR_tight_jetMatched",
            "VR_loose_jetMatched",
        ):
            histograms[region] = make_nmuon_histogram()
            histograms[f"{region}_mass"] = make_mass_histogram()
            histograms[f"{region}_pt"] = make_pt_histogram()

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
