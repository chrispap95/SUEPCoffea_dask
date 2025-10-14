import awkward as ak
import fastjet
import hist
import numpy as np
import vector  # type: ignore[import]
from coffea import processor

# Importing CMS corrections
import workflows.CMS_corrections.golden_json_utils as golden_json_utils
import workflows.CMS_corrections.muon_sf_utils as muon_sf_utils
import workflows.CMS_corrections.systematics_utils as systematics_utils
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
        events_tight_cut, _, Z_cands_tight_cut, _, muons_tight_cut = (
            self.find_Z_candidates(events, muons, muons_tight_cut)
        )
        mass_cut = ~ak.is_none(Z_cands_tight_cut) & (Z_cands_tight_cut.mass < 70)
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
        events_loose_cut, _, Z_cands_loose_cut, _, muons_loose_cut = (
            self.find_Z_candidates(events, muons, muons_loose_cut)
        )
        mass_cut = ~ak.is_none(Z_cands_loose_cut) & (Z_cands_loose_cut.mass < 70)
        events_loose_cut = events_loose_cut[mass_cut]
        muons_loose_cut = muons_loose_cut[mass_cut]
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
            events_SR_high_temp_tight,
            events_SR_high_temp_loose,
            muons_SR_high_temp_tight,
            muons_SR_high_temp_loose,
        ) = self.apply_SR_high_temp(events_)

        if len(events_SR_high_temp_tight) > 0:
            muon_pairs_0, muon_pairs_1 = self.find_dimuon_pairs(
                muons_SR_high_temp_tight
            )
            dimuon_dr_mask = muon_pairs_0.delta_r(muon_pairs_1) < 0.3
            dimuon_mass = (muon_pairs_0 + muon_pairs_1).mass
            dimuon_mass_mask = ((dimuon_mass > 2.7) & (dimuon_mass < 3.5)) | (
                (dimuon_mass > 8.8) & (dimuon_mass < 11.2)
            )
            events_SR_high_temp_tight = events_SR_high_temp_tight[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muons_SR_high_temp_tight = muons_SR_high_temp_tight[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muon_pairs_0 = muon_pairs_0[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muon_pairs_1 = muon_pairs_1[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]

            if len(events_SR_high_temp_tight) > 0:
                weights_SR_high_temp_tight = self.get_weights(
                    events_SR_high_temp_tight, do_vars=False, apply_lumi_factors=True
                )
                if self.isMC:
                    weights_SR_high_temp_tight.add(
                        "MuonSF",
                        weight=ak.prod(
                            muon_sf_utils.muon_efficiencies(
                                muons_SR_high_temp_tight,
                                era=self.era,
                                region="SR_high_temp_tight",
                                syst="",
                            ),
                            axis=-1,
                        ),
                    )
                nMuon_SR_high_temp_tight = ak.num(muons_SR_high_temp_tight, axis=-1)
                nMuon_SR_high_temp_tight = ak.where(
                    nMuon_SR_high_temp_tight > 7, 7, nMuon_SR_high_temp_tight
                )
                output[dataset]["histograms"]["SR_high_temp_tight_muon_pt"].fill(
                    ak.flatten(muons_SR_high_temp_tight.pt),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_SR_high_temp_tight, muons_SR_high_temp_tight.pt
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_tight.weight(),
                            muons_SR_high_temp_tight.pt,
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["SR_high_temp_tight_muon_eta"].fill(
                    ak.flatten(muons_SR_high_temp_tight.eta),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_SR_high_temp_tight, muons_SR_high_temp_tight.eta
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_tight.weight(),
                            muons_SR_high_temp_tight.pt,
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["SR_high_temp_tight_muon_eta"].fill(
                    ak.flatten(muons_SR_high_temp_tight.eta),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_SR_high_temp_tight, muons_SR_high_temp_tight.eta
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_tight.weight(),
                            muons_SR_high_temp_tight.pt,
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["SR_high_temp_tight_muon_phi"].fill(
                    ak.flatten(muons_SR_high_temp_tight.phi),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_SR_high_temp_tight, muons_SR_high_temp_tight.phi
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_tight.weight(),
                            muons_SR_high_temp_tight.pt,
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["SR_high_temp_tight_muon_iso"].fill(
                    ak.flatten(muons_SR_high_temp_tight.miniPFRelIso_all),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_SR_high_temp_tight,
                            muons_SR_high_temp_tight.miniPFRelIso_all,
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_tight.weight(),
                            muons_SR_high_temp_tight.pt,
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["SR_high_temp_tight_muon_dxy"].fill(
                    ak.flatten(muons_SR_high_temp_tight.dxy),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_SR_high_temp_tight, muons_SR_high_temp_tight.dxy
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_tight.weight(),
                            muons_SR_high_temp_tight.pt,
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["SR_high_temp_tight_muon_dz"].fill(
                    ak.flatten(muons_SR_high_temp_tight.dz),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_SR_high_temp_tight, muons_SR_high_temp_tight.dz
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_tight.weight(),
                            muons_SR_high_temp_tight.pt,
                        )[0]
                    ),
                )
                muon_pairs_dr = muon_pairs_0.delta_r(muon_pairs_1)
                output[dataset]["histograms"]["SR_high_temp_tight_dimuon_dr"].fill(
                    ak.flatten(muon_pairs_dr),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_SR_high_temp_tight, muon_pairs_dr)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_tight.weight(), muon_pairs_dr
                        )[0]
                    ),
                )
                muon_pairs_mass = (muon_pairs_0 + muon_pairs_1).mass
                output[dataset]["histograms"]["SR_high_temp_tight_dimuon_mass"].fill(
                    ak.flatten(muon_pairs_mass),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_SR_high_temp_tight, muon_pairs_mass)[
                            0
                        ]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_tight.weight(), muon_pairs_mass
                        )[0]
                    ),
                )

        if len(events_SR_high_temp_loose) > 0:
            muon_pairs_0, muon_pairs_1 = self.find_dimuon_pairs(
                muons_SR_high_temp_loose
            )
            dimuon_dr_mask = muon_pairs_0.delta_r(muon_pairs_1) < 0.3
            dimuon_mass = (muon_pairs_0 + muon_pairs_1).mass
            dimuon_mass_mask = ((dimuon_mass > 2.7) & (dimuon_mass < 3.5)) | (
                (dimuon_mass > 8.8) & (dimuon_mass < 11.2)
            )
            events_SR_high_temp_loose = events_SR_high_temp_loose[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muons_SR_high_temp_loose = muons_SR_high_temp_loose[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muon_pairs_0 = muon_pairs_0[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muon_pairs_1 = muon_pairs_1[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]

            if len(events_SR_high_temp_loose) > 0:
                weights_SR_high_temp_loose = self.get_weights(
                    events_SR_high_temp_loose, do_vars=False, apply_lumi_factors=True
                )
                if self.isMC:
                    weights_SR_high_temp_loose.add(
                        "MuonSF",
                        weight=ak.prod(
                            muon_sf_utils.muon_efficiencies(
                                muons_SR_high_temp_loose,
                                era=self.era,
                                region="SR_high_temp_loose",
                                syst="",
                            ),
                            axis=-1,
                        ),
                    )
                nMuon_SR_high_temp_loose = ak.num(muons_SR_high_temp_loose, axis=-1)
                nMuon_SR_high_temp_loose = ak.where(
                    nMuon_SR_high_temp_loose > 7, 7, nMuon_SR_high_temp_loose
                )
                output[dataset]["histograms"]["SR_high_temp_loose_muon_pt"].fill(
                    ak.flatten(muons_SR_high_temp_loose.pt),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_SR_high_temp_loose, muons_SR_high_temp_loose.pt
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_loose.weight(),
                            muons_SR_high_temp_loose.pt,
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["SR_high_temp_loose_muon_eta"].fill(
                    ak.flatten(muons_SR_high_temp_loose.eta),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_SR_high_temp_loose, muons_SR_high_temp_loose.eta
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_loose.weight(),
                            muons_SR_high_temp_loose.pt,
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["SR_high_temp_loose_muon_phi"].fill(
                    ak.flatten(muons_SR_high_temp_loose.phi),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_SR_high_temp_loose, muons_SR_high_temp_loose.phi
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_loose.weight(),
                            muons_SR_high_temp_loose.pt,
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["SR_high_temp_loose_muon_iso"].fill(
                    ak.flatten(muons_SR_high_temp_loose.miniPFRelIso_all),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_SR_high_temp_loose,
                            muons_SR_high_temp_loose.miniPFRelIso_all,
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_loose.weight(),
                            muons_SR_high_temp_loose.pt,
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["SR_high_temp_loose_muon_dxy"].fill(
                    ak.flatten(muons_SR_high_temp_loose.dxy),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_SR_high_temp_loose, muons_SR_high_temp_loose.dxy
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_loose.weight(),
                            muons_SR_high_temp_loose.pt,
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["SR_high_temp_loose_muon_dz"].fill(
                    ak.flatten(muons_SR_high_temp_loose.dz),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_SR_high_temp_loose, muons_SR_high_temp_loose.dz
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_loose.weight(),
                            muons_SR_high_temp_loose.pt,
                        )[0]
                    ),
                )
                muon_pairs_dr = muon_pairs_0.delta_r(muon_pairs_1)
                output[dataset]["histograms"]["SR_high_temp_loose_dimuon_dr"].fill(
                    ak.flatten(muon_pairs_dr),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_SR_high_temp_loose, muon_pairs_dr)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_loose.weight(), muon_pairs_dr
                        )[0]
                    ),
                )
                muon_pairs_mass = (muon_pairs_0 + muon_pairs_1).mass
                output[dataset]["histograms"]["SR_high_temp_loose_dimuon_mass"].fill(
                    ak.flatten(muon_pairs_mass),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_SR_high_temp_loose, muon_pairs_mass)[
                            0
                        ]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_SR_high_temp_loose.weight(), muon_pairs_mass
                        )[0]
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
        histograms = {}
        for region in [
            "SR_high_temp_tight",
            "SR_high_temp_loose",
        ]:
            histograms.update(
                {
                    f"{region}_muon_pt": hist.Hist.new.Regular(
                        50,
                        3,
                        300,
                        name="muon_pt",
                        label="muon_pt",
                        transform=hist.axis.transform.log,
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_muon_eta": hist.Hist.new.Regular(
                        50,
                        -3,
                        3,
                        name="muon_eta",
                        label="muon_eta",
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_muon_phi": hist.Hist.new.Regular(
                        50,
                        -np.pi,
                        np.pi,
                        name="muon_phi",
                        label="muon_phi",
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_muon_iso": hist.Hist.new.Regular(
                        50,
                        0.01,
                        10,
                        name="muon_iso",
                        label="muon_iso",
                        transform=hist.axis.transform.log,
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_muon_dxy": hist.Hist.new.Regular(
                        50,
                        1e-4,
                        1,
                        name="muon_dxy",
                        label="muon_dxy",
                        transform=hist.axis.transform.log,
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_muon_dz": hist.Hist.new.Regular(
                        50,
                        1e-4,
                        1,
                        name="muon_dz",
                        label="muon_dz",
                        transform=hist.axis.transform.log,
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_dimuon_dr": hist.Hist.new.Regular(
                        50,
                        1e-2,
                        10,
                        name="dimuon_dr",
                        label="dimuon_dr",
                        transform=hist.axis.transform.log,
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_dimuon_mass": hist.Hist.new.Regular(
                        50,
                        0.1,
                        100,
                        name="dimuon_mass",
                        label="dimuon_mass",
                        transform=hist.axis.transform.log,
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                }
            )

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
