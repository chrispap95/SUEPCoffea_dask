import awkward as ak
import fastjet
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

    def apply_SR_low_temp(self, events):
        """
        Apply the SR_low_temp selection to the events.
        """
        muons = events.Muon

        # Filter events with at least one muon and at least one pfcand
        filter_empty_events = (ak.num(events.Muon) > 0) & (ak.num(events.PFCands) > 0)
        events = events[filter_empty_events]
        muons = muons[filter_empty_events]

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

        # Tight SR: selection for muons
        tight_cut = (muons.pt < 35) & (muons.ip3d < 0.007)
        muons_tight_cut = muons[clean_muons & tight_cut]

        # Tight SR: Z mass window cut
        events_tight_cut, muons_tight_cut, Z_cands_tight_cut, _ = (
            self.find_Z_candidates(events, muons_tight_cut)
        )
        mass_cut = Z_cands_tight_cut.mass < 35
        events_tight_cut = events_tight_cut[mass_cut]
        muons_tight_cut = muons_tight_cut[mass_cut]

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
        events_tight_cand_cut = events_tight_cut[found_SUEP_cand_tight_cut]
        muons_tight_cand_cut = muons_tight_cut[found_SUEP_cand_tight_cut]
        tracks_tight_cand_cut = tracks_tight_cut[found_SUEP_cand_tight_cut]
        SUEP_cand_tight_cut = SUEP_cand_tight_cut[found_SUEP_cand_tight_cut]
        SUEP_cluster_tight_cut = SUEP_cluster_tight_cut[found_SUEP_cand_tight_cut]

        # Tight SR: sphericity cut
        sph1_tight_cut = self.S1(SUEP_cand_tight_cut, SUEP_cluster_tight_cut)
        events_tight_cand_cut = events_tight_cand_cut[sph1_tight_cut > 0.7]
        muons_tight_cand_cut = muons_tight_cand_cut[sph1_tight_cut > 0.7]
        tracks_tight_cand_cut = tracks_tight_cand_cut[sph1_tight_cut > 0.7]

        # Loose SR: selection for muons
        loose_cut = (events.Muon.pt < 45) & (events.Muon.ip3d < 0.1)
        muons_loose_cut = muons[clean_muons & loose_cut]

        # Loose SR: Z mass window cut
        events_loose_cut, muons_loose_cut, Z_cands_loose_cut, _ = (
            self.find_Z_candidates(events, muons_loose_cut)
        )
        mass_cut = Z_cands_loose_cut.mass < 45
        events_loose_cut = events_loose_cut[mass_cut]
        muons_loose_cut = muons_loose_cut[mass_cut]

        # Loose SR: at least 3 muons
        select_by_muons_loose_cut = ak.num(muons_loose_cut, axis=-1) > 2
        events_loose_cut = events_loose_cut[select_by_muons_loose_cut]
        muons_loose_cut = muons_loose_cut[select_by_muons_loose_cut]

        # Loose SR: SUEP candidate
        tracks_loose_cut = self.get_clean_tracks(events_loose_cut)
        SUEP_cand_loose_cut, SUEP_cluster_loose_cut = self.find_SUEP_candidate(
            tracks_loose_cut
        )
        found_SUEP_cand_loose_cut = ~ak.is_none(SUEP_cand_loose_cut)
        events_loose_cand_cut = events_loose_cut[found_SUEP_cand_loose_cut]
        muons_loose_cand_cut = muons_loose_cut[found_SUEP_cand_loose_cut]
        tracks_loose_cand_cut = tracks_loose_cut[found_SUEP_cand_loose_cut]
        SUEP_cand_loose_cut = SUEP_cand_loose_cut[found_SUEP_cand_loose_cut]
        SUEP_cluster_loose_cut = SUEP_cluster_loose_cut[found_SUEP_cand_loose_cut]

        # Loose SR: sphericity cut
        sph1_loose_cut = self.S1(SUEP_cand_loose_cut, SUEP_cluster_loose_cut)
        events_loose_cand_cut = events_loose_cand_cut[sph1_loose_cut > 0.2]
        muons_loose_cand_cut = muons_loose_cand_cut[sph1_loose_cut > 0.2]
        tracks_loose_cand_cut = tracks_loose_cand_cut[sph1_loose_cut > 0.2]

        return (
            events_tight_cand_cut,
            events_loose_cand_cut,
            muons_tight_cand_cut,
            muons_loose_cand_cut,
        )

    def apply_SR_high_temp(self, events):
        """
        Apply the SR_high_temp selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        # Apply basic muon cuts
        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dz) < 0.2)
        )

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        # Tight SR selection
        tight_cut = (
            (muons.ip3d < 0.007)
            & (muons.miniPFRelIso_all < 0.65)
            & ((muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 0.5)
        )
        muons_tight_cut = muons[clean_muons & tight_cut]
        events_tight_cut, muons_tight_cut, Z_cands_tight_cut, _ = (
            self.find_Z_candidates(events, muons_tight_cut)
        )
        mass_cut = Z_cands_tight_cut.mass < 70
        events_tight_cut = events_tight_cut[mass_cut]
        muons_tight_cut = muons_tight_cut[mass_cut]

        select_by_muons_tight = ak.num(muons_tight_cut, axis=-1) > 2
        events_tight_cut = events_tight_cut[select_by_muons_tight]
        muons_tight_cut = muons_tight_cut[select_by_muons_tight]

        # Loose SR selection
        loose_cut = (
            (muons.ip3d < 0.1)
            & (muons.miniPFRelIso_all < 5)
            & ((muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 3)
        )
        muons_loose_cut = muons[clean_muons & loose_cut]
        events_loose_cut, muons_loose_cut, Z_cands_loose_cut, _ = (
            self.find_Z_candidates(events, muons_loose_cut)
        )
        mass_cut = Z_cands_loose_cut.mass < 70
        events_loose_cut = events_loose_cut[mass_cut]
        muons_loose_cut = muons_loose_cut[mass_cut]
        select_by_muons_loose_cut = ak.num(muons_loose_cut, axis=-1) > 2
        events_loose_cut = events_loose_cut[select_by_muons_loose_cut]
        muons_loose_cut = muons_loose_cut[select_by_muons_loose_cut]

        return events_tight_cut, events_loose_cut, muons_tight_cut, muons_loose_cut

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
        # Apply tight miniIso id corresponding to miniIso < 0.1
        candidate_muons = muons[candidates_indices]
        prompt_muons = muons[
            (candidate_muons.pt > 25)
            & (candidate_muons.miniIsoId >= 3)
            & (abs(candidate_muons.dxy) < 0.008)
            & (abs(candidate_muons.dz) < 0.01)
            & (abs(candidate_muons.ip3d) < 0.01)
        ]
        muons = muons[ak.num(prompt_muons, axis=-1) > 0]
        events = events[ak.num(prompt_muons, axis=-1) > 0]
        prompt_muons = prompt_muons[ak.num(prompt_muons, axis=-1) > 0]

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
            weights_CR_prompt = self.get_weights(events_CR_prompt, do_vars=True)
            if self.isMC:
                weights_CR_prompt.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_CR_prompt, self.era, syst=""
                        ),
                        axis=-1,
                    ),
                )
            nMuon_CR_prompt = ak.num(muons_CR_prompt, axis=-1)
            nMuon_CR_prompt = ak.where(nMuon_CR_prompt > 6, 6, nMuon_CR_prompt)
            output[dataset]["histograms"]["CR_prompt"].fill(
                ak.flatten(muons_CR_prompt.pt),
                ak.flatten(muons_CR_prompt.eta),
                ak.flatten(ak.broadcast_arrays(muons_CR_prompt.pt, nMuon_CR_prompt)[1]),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_CR_prompt.pt,
                        weights_CR_prompt.weight(),
                    )[1]
                ),
            )

        events_CR_cb, muons_CR_cb = self.apply_CR_cb(events_)
        if len(events_CR_cb) > 0:
            weights_CR_cb = self.get_weights(events_CR_cb, do_vars=True)
            if self.isMC:
                weights_CR_cb.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(muons_CR_cb, self.era, syst=""),
                        axis=-1,
                    ),
                )
            nMuon_CR_cb = ak.num(muons_CR_cb, axis=-1)
            nMuon_CR_cb = ak.where(nMuon_CR_cb > 5, 5, nMuon_CR_cb)
            output[dataset]["histograms"]["CR_cb"].fill(
                ak.flatten(muons_CR_cb.pt),
                ak.flatten(muons_CR_cb.eta),
                ak.flatten(ak.broadcast_arrays(muons_CR_cb.pt, nMuon_CR_cb)[1]),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_CR_cb.pt,
                        weights_CR_cb.weight(),
                    )[1]
                ),
            )

        (
            events_SR_high_temp_tight,
            events_SR_high_temp_loose,
            muons_SR_high_temp_tight,
            muons_SR_high_temp_loose,
        ) = self.apply_SR_high_temp(events_)

        if len(events_SR_high_temp_tight) > 0:
            weights_SR_high_temp_tight = self.get_weights(
                events_SR_high_temp_tight, do_vars=True
            )
            if self.isMC:
                weights_SR_high_temp_tight.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_SR_high_temp_tight, self.era, syst=""
                        ),
                        axis=-1,
                    ),
                )
            nMuon_SR_high_temp_tight = ak.num(muons_SR_high_temp_tight, axis=-1)
            nMuon_SR_high_temp_tight = ak.where(
                nMuon_SR_high_temp_tight > 7, 7, nMuon_SR_high_temp_tight
            )
            output[dataset]["histograms"]["SR_high_temp_tight"].fill(
                ak.flatten(muons_SR_high_temp_tight.pt),
                ak.flatten(muons_SR_high_temp_tight.eta),
                ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_high_temp_tight.pt, nMuon_SR_high_temp_tight
                    )[1]
                ),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_high_temp_tight.pt,
                        weights_SR_high_temp_tight.weight(),
                    )[1]
                ),
            )

        if len(events_SR_high_temp_loose) > 0:
            weights_SR_high_temp_loose = self.get_weights(
                events_SR_high_temp_loose, do_vars=True
            )
            if self.isMC:
                weights_SR_high_temp_loose.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_SR_high_temp_loose, self.era, syst=""
                        ),
                        axis=-1,
                    ),
                )
            nMuon_SR_high_temp_loose = ak.num(muons_SR_high_temp_loose, axis=-1)
            nMuon_SR_high_temp_loose = ak.where(
                nMuon_SR_high_temp_loose > 7, 7, nMuon_SR_high_temp_loose
            )
            output[dataset]["histograms"]["SR_high_temp_loose"].fill(
                ak.flatten(muons_SR_high_temp_loose.pt),
                ak.flatten(muons_SR_high_temp_loose.eta),
                ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_high_temp_loose.pt, nMuon_SR_high_temp_loose
                    )[1]
                ),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_high_temp_loose.pt,
                        weights_SR_high_temp_loose.weight(),
                    )[1]
                ),
            )

        (
            events_SR_low_temp_tight,
            events_SR_low_temp_loose,
            muons_SR_low_temp_tight,
            muons_SR_low_temp_loose,
        ) = self.apply_SR_low_temp(events_)

        if len(events_SR_low_temp_tight) > 0:
            weights_SR_low_temp_tight = self.get_weights(
                events_SR_low_temp_tight, do_vars=True
            )
            if self.isMC:
                weights_SR_low_temp_tight.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_SR_low_temp_tight, self.era, syst=""
                        ),
                        axis=-1,
                    ),
                )
            nMuon_SR_low_temp_tight = ak.num(muons_SR_low_temp_tight, axis=-1)
            nMuon_SR_low_temp_tight = ak.where(
                nMuon_SR_low_temp_tight > 7, 7, nMuon_SR_low_temp_tight
            )
            output[dataset]["histograms"]["SR_low_temp_tight"].fill(
                ak.flatten(muons_SR_low_temp_tight.pt),
                ak.flatten(muons_SR_low_temp_tight.eta),
                ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_low_temp_tight.pt, nMuon_SR_low_temp_tight
                    )[1]
                ),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_low_temp_tight.pt,
                        weights_SR_low_temp_tight.weight(),
                    )[1]
                ),
            )

        if len(events_SR_low_temp_loose) > 0:
            weights_SR_low_temp_loose = self.get_weights(
                events_SR_low_temp_loose, do_vars=True
            )
            if self.isMC:
                weights_SR_low_temp_loose.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_SR_low_temp_loose, self.era, syst=""
                        ),
                        axis=-1,
                    ),
                )
            nMuon_SR_low_temp_loose = ak.num(muons_SR_low_temp_loose, axis=-1)
            nMuon_SR_low_temp_loose = ak.where(
                nMuon_SR_low_temp_loose > 7, 7, nMuon_SR_low_temp_loose
            )
            output[dataset]["histograms"]["SR_low_temp_loose"].fill(
                ak.flatten(muons_SR_low_temp_loose.pt),
                ak.flatten(muons_SR_low_temp_loose.eta),
                ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_low_temp_loose.pt, nMuon_SR_low_temp_loose
                    )[1]
                ),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_low_temp_loose.pt,
                        weights_SR_low_temp_loose.weight(),
                    )[1]
                ),
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
            "CR_cb": hist.Hist.new.Regular(
                30,
                3,
                300,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(20, -2.5, 2.5, name="muon_eta", label="muon_eta")
            .Regular(4, 1, 5, name="nMuon", label="nMuon")
            .Weight(),
            "CR_prompt": hist.Hist.new.Regular(
                30,
                3,
                300,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(20, -2.5, 2.5, name="muon_eta", label="muon_eta")
            .Regular(4, 2, 6, name="nMuon", label="nMuon")
            .Weight(),
            "SR_high_temp_tight": hist.Hist.new.Regular(
                30,
                3,
                300,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(20, -2.5, 2.5, name="muon_eta", label="muon_eta")
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "SR_high_temp_loose": hist.Hist.new.Regular(
                30,
                3,
                300,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(20, -2.5, 2.5, name="muon_eta", label="muon_eta")
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "SR_low_temp_tight": hist.Hist.new.Regular(
                30,
                3,
                300,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(20, -2.5, 2.5, name="muon_eta", label="muon_eta")
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "SR_low_temp_loose": hist.Hist.new.Regular(
                30,
                3,
                300,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(20, -2.5, 2.5, name="muon_eta", label="muon_eta")
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
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
