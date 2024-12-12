import awkward as ak
import fastjet
import hist
import numpy as np
import vector  # type: ignore[import]
from coffea import processor

# Importing CMS corrections
import workflows.CMS_corrections.golden_json_utils as golden_json_utils
import workflows.CMS_corrections.systematics_utils as systematics_utils

# Set vector behavior
vector.register_awkward()

Z_MASS = 91.1876
Z_WIDTH = 2.4952


class SUEP_cluster(processor.ProcessorABC):
    def __init__(
        self,
        isMC: bool,
        era: str | int,
        syst_var: str = "",
    ) -> None:
        self.isMC = isMC
        self.era = era if isinstance(era, str) else str(era)
        self.syst_var = syst_var
        self.syst_suffix = f"_sys_{syst_var}" if syst_var != "" else ""
        self.gensumweight = 1.0

    def eventSelection(self, events):
        """
        Applies trigger, returns events.
        """
        trigger1 = np.ones(len(events), dtype=bool)
        trigger2 = np.ones(len(events), dtype=bool)
        trigger3 = np.ones(len(events), dtype=bool)
        trigger4 = np.ones(len(events), dtype=bool)
        if self.era in ["2016", "2016APV"]:
            if "TripleMu_5_3_3" in events.HLT.fields:
                trigger1 = events.HLT.TripleMu_5_3_3 == 1
            if "TripleMu_5_3_3_DZ_Mass3p8" in events.HLT.fields:
                trigger2 = events.HLT.TripleMu_5_3_3_DZ_Mass3p8 == 1
            if "TripleMu_12_10_5" in events.HLT.fields:
                trigger3 = events.HLT.TripleMu_12_10_5 == 1
        elif self.era == "2017":
            if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
                trigger1 = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger2 = events.HLT.TripleMu_10_5_5_DZ == 1
            if "TripleMu_12_10_5" in events.HLT.fields:
                trigger3 = events.HLT.TripleMu_12_10_5 == 1
        elif self.era == "2018":
            if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
                trigger1 = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                trigger2 = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger3 = events.HLT.TripleMu_10_5_5_DZ == 1
            if "TripleMu_12_10_5" in events.HLT.fields:
                trigger4 = events.HLT.TripleMu_12_10_5 == 1
        elif self.era in ["2022", "2023"]:
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                trigger1 = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger2 = events.HLT.TripleMu_10_5_5_DZ == 1
            if "TripleMu_12_10_5" in events.HLT.fields:
                trigger3 = events.HLT.TripleMu_12_10_5 == 1
        else:
            raise ValueError(f"Invalid era: {self.era}")
        trigger = np.any(np.array([trigger1, trigger2, trigger3, trigger4]).T, axis=-1)
        events = events[trigger]
        return events

    def get_weights(self, events):
        if not self.isMC:
            return np.ones(len(events))
        # Pileup weights (need to be fed with integers)
        pu_weights = systematics_utils.pileup_weight(
            self.era, ak.values_astype(events.Pileup.nTrueInt, np.int32)
        )
        # L1 prefire weights
        prefire_weights = systematics_utils.get_prefire_weights(events)
        # Trigger scale factors
        # To be implemented
        return events.genWeight * pu_weights * prefire_weights

    def sphericity_eigenvalues(self, particles, r):
        """
        Calculate the sphericity tensor for a set of particles and return the eigenvalues.
        """
        norm = ak.sum(particles.p**r, axis=1, keepdims=True)
        # Remove particles with 0 momentum
        particles = particles[particles.p > 0]
        s = np.array(
            [
                [
                    ak.sum(
                        particles.px * particles.px * particles.p ** (r - 2.0),
                        axis=1,
                        keepdims=True,
                    )
                    / norm,
                    ak.sum(
                        particles.px * particles.py * particles.p ** (r - 2.0),
                        axis=1,
                        keepdims=True,
                    )
                    / norm,
                    ak.sum(
                        particles.px * particles.pz * particles.p ** (r - 2.0),
                        axis=1,
                        keepdims=True,
                    )
                    / norm,
                ],
                [
                    ak.sum(
                        particles.py * particles.px * particles.p ** (r - 2.0),
                        axis=1,
                        keepdims=True,
                    )
                    / norm,
                    ak.sum(
                        particles.py * particles.py * particles.p ** (r - 2.0),
                        axis=1,
                        keepdims=True,
                    )
                    / norm,
                    ak.sum(
                        particles.py * particles.pz * particles.p ** (r - 2.0),
                        axis=1,
                        keepdims=True,
                    )
                    / norm,
                ],
                [
                    ak.sum(
                        particles.pz * particles.px * particles.p ** (r - 2.0),
                        axis=1,
                        keepdims=True,
                    )
                    / norm,
                    ak.sum(
                        particles.pz * particles.py * particles.p ** (r - 2.0),
                        axis=1,
                        keepdims=True,
                    )
                    / norm,
                    ak.sum(
                        particles.pz * particles.pz * particles.p ** (r - 2.0),
                        axis=1,
                        keepdims=True,
                    )
                    / norm,
                ],
            ]
        )
        s = np.squeeze(np.moveaxis(s, 2, 0), axis=3)
        evals = np.sort(np.linalg.eigvalsh(s))
        return evals

    def S1(self, SUEP_candidate, SUEP_cluster_tracks):
        """
        Calculate the S1 variable for the SUEP cluster.
        """
        if len(SUEP_candidate) == 0:
            return ak.Array([])
        boost_vector = ak.zip(
            {
                "px": SUEP_candidate.px * -1,
                "py": SUEP_candidate.py * -1,
                "pz": SUEP_candidate.pz * -1,
                "mass": SUEP_candidate.mass,
            },
            with_name="Momentum4D",
        )
        boosted_tracks = SUEP_cluster_tracks.boost_p4(boost_vector)
        # NOTE: is this the correct way to handle NaNs?
        boosted_tracks = ak.nan_to_num(boosted_tracks)
        eigs_tracks = self.sphericity_eigenvalues(boosted_tracks, 1.0)
        return 1.5 * (eigs_tracks[:, 1] + eigs_tracks[:, 0])

    def fastjet_reclustering(self, tracks, r, min_pt):
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, r)
        cluster = fastjet.ClusterSequence(tracks, jetdef)
        ak_inc_jets = cluster.inclusive_jets(min_pt=min_pt)
        ak_inc_cluster = cluster.constituents(min_pt)
        return ak_inc_jets, ak_inc_cluster

    def find_SUEP_candidate(self, tracks):
        """
        Find the SUEP candidate by reclustering the tracks with AK15 and selecting the
        jet with the most tracks. Return the SUEP candidate and the tracks in the cluster.
        Will return None if no SUEP candidate is found in the event.
        """
        # Recluster tracks with AK15 algorithm
        ak15_inc_jets, ak15_inc_cluster = self.fastjet_reclustering(
            tracks, r=1.5, min_pt=0
        )

        # Discard single track clusters
        at_least_two_tracks_per_cluster = ak.num(ak15_inc_cluster, axis=-1) > 1
        ak15_inc_jets = ak15_inc_jets[at_least_two_tracks_per_cluster]
        ak15_inc_cluster = ak15_inc_cluster[at_least_two_tracks_per_cluster]

        # Order the reclustered jets by pT and keep only up to the top 2
        jets_pt_order = ak.argsort(ak15_inc_jets.pt, axis=1, ascending=False)  # type: ignore[attr-defined]
        jets_pt_sorted = ak15_inc_jets[jets_pt_order]
        clusters_pt_sorted = ak15_inc_cluster[jets_pt_order]
        jets_pt_sorted = jets_pt_sorted[:, :2]  # type: ignore[attr-defined]
        clusters_pt_sorted = clusters_pt_sorted[:, :2]  # type: ignore[attr-defined]

        # Find
        nconst_pt_sorted = ak.num(clusters_pt_sorted, axis=-1)
        SUEP_cand_index = ak.argmax(nconst_pt_sorted, axis=1, keepdims=True)
        SUEP_cand = ak.firsts(jets_pt_sorted[SUEP_cand_index])  # type: ignore[attr-defined]
        SUEP_cluster = ak.firsts(clusters_pt_sorted[SUEP_cand_index])  # type: ignore[attr-defined]

        return SUEP_cand, SUEP_cluster

    def get_clean_tracks(self, events):
        pfcands = ak.zip(
            {
                "pt": events.PFCands.trkPt,
                "eta": events.PFCands.trkEta,
                "phi": events.PFCands.trkPhi,
                "mass": events.PFCands.mass,
            },
            with_name="Momentum4D",
        )
        cut = (
            (events.PFCands.fromPV > 1)
            & (events.PFCands.trkPt >= 0.75)
            & (abs(events.PFCands.trkEta) <= 2.5)
            & (abs(events.PFCands.dz) < 10)
            & (events.PFCands.dzErr < 0.05)
        )
        cleaned_pfcands = pfcands[cut]
        cleaned_pfcands = ak.packed(cleaned_pfcands)

        lost_tracks = ak.zip(
            {
                "pt": events.lostTracks.pt,
                "eta": events.lostTracks.eta,
                "phi": events.lostTracks.phi,
                "mass": ak.zeros_like(events.lostTracks.pt),
            },
            with_name="Momentum4D",
        )
        cut = (
            (events.lostTracks.fromPV > 1)
            & (events.lostTracks.pt >= 0.75)
            & (abs(events.lostTracks.eta) <= 1.0)
            & (abs(events.lostTracks.dz) < 10)
            & (events.lostTracks.dzErr < 0.05)
        )
        cleaned_lost_tracks = lost_tracks[cut]
        cleaned_lost_tracks = ak.packed(cleaned_lost_tracks)

        return ak.concatenate([cleaned_pfcands, cleaned_lost_tracks], axis=1)

    def find_Z_candidates(self, events, muons):
        """
        Find the Z candidates by forming all possible pairs of OS muons
        and selecting the one closest to the Z mass.
        """
        # Make sure there are at least two muons with opposite charge
        muons1 = muons[muons.charge == 1]
        muons2 = muons[muons.charge == -1]
        enough_muons = (ak.num(muons1) > 0) & (ak.num(muons2) > 0)
        muons1 = muons1[enough_muons]
        muons2 = muons2[enough_muons]
        muons = muons[enough_muons]
        events = events[enough_muons]

        # Create all possible pairs of OS muons
        muon_pairs = ak.unzip(ak.cartesian([muons1, muons2]))

        # Find the pair closest to the Z mass
        Z_cands = muon_pairs[0] + muon_pairs[1]  # type: ignore[attr-defined]
        closest_to_peak = ak.argmin(abs(Z_cands.mass - Z_MASS), axis=1)
        Z_cands = ak.firsts(Z_cands[ak.singletons(closest_to_peak)])

        return events, muons, Z_cands

    def muon_filter(self, events):
        """
        Filter events after the TripleMu trigger.
        Cleans muons and electrons.
        Requires at least nMuons with mediumId, pt, dz, and eta cuts.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dz) < 0.2)
        )
        muons = muons[clean_muons]
        select_by_muons_low = ak.num(muons, axis=-1) > 2
        events = events[select_by_muons_low]

        return events

    def apply_SR_low_temp(self, events):
        """
        Apply the SR_low_temp selection to the events.
        """
        muons = events.Muon

        # Filter events with at least one muon and at least one pfcand
        filter_empty_events = (ak.num(events.Muon) > 0) & (ak.num(events.PFCands) > 0)
        events = events[filter_empty_events]
        muons = muons[filter_empty_events]

        # Apply basic muon cuts
        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dz) < 0.2)
        )

        # Tight SR: selection for muons
        tight_cut = (events.Muon.pt < 35) & (events.Muon.ip3d < 0.008)
        muons_tight_cut = muons[clean_muons & tight_cut]

        # Tight SR: Z mass window cut
        events_tight_cut, muons_tight_cut, Z_cands_tight_cut = self.find_Z_candidates(
            events, muons_tight_cut
        )
        outside_mass_window_tight_cut = (
            abs(Z_cands_tight_cut.mass - Z_MASS) > 2 * Z_WIDTH
        )
        events_tight_cut = events_tight_cut[outside_mass_window_tight_cut]
        muons_tight_cut = muons_tight_cut[outside_mass_window_tight_cut]

        # Tight SR: at least 3 muons
        select_by_muons_tight = ak.num(muons_tight_cut, axis=-1) > 2
        events_tight_cut = events_tight_cut[select_by_muons_tight]
        muons_tight_cut = muons_tight_cut[select_by_muons_tight]

        # Tight SR: SUEP candidate
        tracks_tight_cut = self.get_clean_tracks(events_tight_cut)
        SUEP_cand_tight_cut, SUEP_cluster_tight_cut = self.find_SUEP_candidate(
            tracks_tight_cut
        )
        found_SUEP_cand_tight_cut = ~ak.is_none(SUEP_cand_tight_cut)
        events_tight_cut = events_tight_cut[found_SUEP_cand_tight_cut]
        muons_tight_cut = muons_tight_cut[found_SUEP_cand_tight_cut]
        tracks_tight_cut = tracks_tight_cut[found_SUEP_cand_tight_cut]
        SUEP_cand_tight_cut = SUEP_cand_tight_cut[found_SUEP_cand_tight_cut]
        SUEP_cluster_tight_cut = SUEP_cluster_tight_cut[found_SUEP_cand_tight_cut]

        # Tight SR: sphericity cut
        sph1_tight_cut = self.S1(SUEP_cluster_tight_cut, SUEP_cand_tight_cut)
        events_tight_cut = events[sph1_tight_cut > 0.75]
        muons_tight_cut = muons_tight_cut[sph1_tight_cut > 0.75]

        # Loose SR: selection for muons
        loose_cut = events.Muon.ip3d < 0.1
        muons_loose_cut = muons[clean_muons & loose_cut]

        # Loose SR: Z mass window cut
        events_loose_cut, muons_loose_cut, Z_cands_loose_cut = self.find_Z_candidates(
            events, muons_loose_cut
        )
        outside_mass_window_loose_cut = (
            abs(Z_cands_loose_cut.mass - Z_MASS) > 2 * Z_WIDTH
        )
        events_loose_cut = events_loose_cut[outside_mass_window_loose_cut]
        muons_loose_cut = muons_loose_cut[outside_mass_window_loose_cut]

        # Loose SR: at least 3 muons
        select_by_muons_loose_cut = ak.num(muons_loose_cut, axis=-1) > 2
        events_loose_cut = events_loose_cut[select_by_muons_loose_cut]
        muons_loose_cut = muons_loose_cut[select_by_muons_loose_cut]

        return events_tight_cut, events_loose_cut, muons_tight_cut, muons_loose_cut

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        (
            events_SR_low_temp_tight,
            events_SR_low_temp_loose,
            muons_SR_low_temp_tight,
            muons_SR_low_temp_loose,
        ) = self.apply_SR_low_temp(events_)

        weights_SR_low_temp_tight = self.get_weights(events_SR_low_temp_tight)
        nMuon_SR_low_temp_tight = ak.num(muons_SR_low_temp_tight, axis=-1)
        output[dataset]["histograms"]["SR_low_temp_tight"].fill(
            ak.where(nMuon_SR_low_temp_tight > 7, 7, nMuon_SR_low_temp_tight),
            weight=weights_SR_low_temp_tight,
        )

        weights_SR_low_temp_loose = self.get_weights(events_SR_low_temp_loose)
        nMuon_SR_low_temp_loose = ak.num(muons_SR_low_temp_loose, axis=-1)
        output[dataset]["histograms"]["SR_low_temp_loose"].fill(
            ak.where(nMuon_SR_low_temp_loose > 7, 7, nMuon_SR_low_temp_loose),
            weight=weights_SR_low_temp_loose,
        )
        return

    def analysis(self, events, output):
        #####################################################################################
        # ---- Trigger event selection
        # Cut based on ak4 jets to replicate the trigger
        #####################################################################################

        # get dataset name
        dataset = events.metadata["dataset"]

        # take care of weights
        weights = self.get_weights(events)

        # Fill the cutflow columns for all
        output[dataset]["cutflow"].fill(len(events) * ["all"], weight=weights)

        # golden jsons for offline data
        if not self.isMC:
            events = golden_json_utils.apply_golden_JSON(events, self.era)

        events = self.eventSelection(events)

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
            weight=weights,
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
            "SR_low_temp_tight": hist.Hist.new.Regular(
                5, 3, 8, name="nMuon", label="nMuon"
            ).Weight(),
            "SR_low_temp_loose": hist.Hist.new.Regular(
                5, 3, 8, name="nMuon", label="nMuon"
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
