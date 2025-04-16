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

    def apply_SR_low_temp(self, events, ip3d_cut=0.007, sph1_cut=0.7):
        """
        Apply the SR_low_temp tight selection to the events.
        THe ip3d and sph1 cuts can be varied.
        """
        muons = events.Muon

        # Filter events with at least one muon and at least one pfcand
        filter_empty_events = (ak.num(events.Muon) > 0) & (ak.num(events.PFCands) > 0)
        events = events[filter_empty_events]
        muons = muons[filter_empty_events]

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dz) < 0.2)
        )

        # Loose SR selection
        loose_cut = (muons.pt < 35) & (muons.ip3d < ip3d_cut)
        muons_loose_cut = muons[clean_muons & loose_cut]

        # Loose SR: Z mass window cut
        events_loose_cut, muons_loose_cut, Z_cands_loose_cut, _ = (
            self.find_Z_candidates(events, muons_loose_cut)
        )
        mass_cut = Z_cands_loose_cut.mass < 35
        events_loose_cut = events_loose_cut[mass_cut]
        muons_loose_cut = muons_loose_cut[mass_cut]

        # Loose SR: at least 3 muons
        select_by_muons_loose = ak.num(muons_loose_cut, axis=-1) > 2
        events_loose_cut = events_loose_cut[select_by_muons_loose]
        muons_loose_cut = muons_loose_cut[select_by_muons_loose]

        # Loose SR: SUEP candidate
        tracks_loose_cut = self.get_clean_tracks(events_loose_cut)
        SUEP_cand_loose_cut, SUEP_cluster_loose_cut = self.find_SUEP_candidate(
            tracks_loose_cut
        )
        found_SUEP_cand_loose_cut = ~ak.is_none(SUEP_cand_loose_cut)
        events_loose_cut = events_loose_cut[found_SUEP_cand_loose_cut]
        muons_loose_cut = muons_loose_cut[found_SUEP_cand_loose_cut]
        tracks_loose_cut = tracks_loose_cut[found_SUEP_cand_loose_cut]
        SUEP_cand_loose_cut = SUEP_cand_loose_cut[found_SUEP_cand_loose_cut]
        SUEP_cluster_loose_cut = SUEP_cluster_loose_cut[found_SUEP_cand_loose_cut]

        # Loose SR: sphericity cut
        sph1_loose_cut = self.S1(SUEP_cand_loose_cut, SUEP_cluster_loose_cut)
        events_loose_cut = events[sph1_loose_cut > sph1_cut]
        muons_loose_cut = muons_loose_cut[sph1_loose_cut > sph1_cut]

        # Tight SR selection
        tight_cut = (muons.pt < 35) & (muons.ip3d < ip3d_cut)
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
        events_tight_cut = events_tight_cut[found_SUEP_cand_tight_cut]
        muons_tight_cut = muons_tight_cut[found_SUEP_cand_tight_cut]
        tracks_tight_cut = tracks_tight_cut[found_SUEP_cand_tight_cut]
        SUEP_cand_tight_cut = SUEP_cand_tight_cut[found_SUEP_cand_tight_cut]
        SUEP_cluster_tight_cut = SUEP_cluster_tight_cut[found_SUEP_cand_tight_cut]

        # Tight SR: sphericity cut
        sph1_tight_cut = self.S1(SUEP_cand_tight_cut, SUEP_cluster_tight_cut)
        events_tight_cut = events[sph1_tight_cut > sph1_cut]
        muons_tight_cut = muons_tight_cut[sph1_tight_cut > sph1_cut]

        return events_loose_cut, events_tight_cut, muons_loose_cut, muons_tight_cut

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        ip3d_cuts = 1.001 * np.logspace(-3, -1, 11)[:-1]
        for ip3d_cut in ip3d_cuts:
            events_loose_cut, events_tight_cut, muons_loose_cut, muons_tight_cut = (
                self.apply_SR_low_temp(events_, ip3d_cut=ip3d_cut)
            )
            if len(events_loose_cut) > 0:
                weights_loose_cut = self.get_weights(events_loose_cut)
                weights_loose_cut.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_loose_cut, self.era, syst=""
                        ),
                        axis=-1,
                    ),
                )
                nMuon_loose_cut = ak.num(muons_loose_cut, axis=-1)
                output[dataset]["histograms"][
                    "SR_low_temp_loose_ip3d_cut_vs_nMuon"
                ].fill(
                    ip3d_cut=ip3d_cut,
                    nMuon=ak.where(nMuon_loose_cut > 7, 7, nMuon_loose_cut),
                    weight=weights_loose_cut.weight(),
                )
            if len(events_tight_cut) > 0:
                weights_tight_cut = self.get_weights(events_tight_cut)
                weights_tight_cut.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_tight_cut, self.era, syst=""
                        ),
                        axis=-1,
                    ),
                )
                nMuon_tight_cut = ak.num(muons_tight_cut, axis=-1)
                output[dataset]["histograms"][
                    "SR_low_temp_tight_ip3d_cut_vs_nMuon"
                ].fill(
                    ip3d_cut=ip3d_cut,
                    nMuon=ak.where(nMuon_tight_cut > 7, 7, nMuon_tight_cut),
                    weight=weights_tight_cut.weight(),
                )

        sph1_cuts = 1.001 * np.linspace(0.1, 0.9, 11)[:-1]
        for sph1_cut in sph1_cuts:
            events_loose_cut, events_tight_cut, muons_loose_cut, muons_tight_cut = (
                self.apply_SR_low_temp(events_, sph1_cut=sph1_cut)
            )
            if len(events_loose_cut) > 0:
                weights_loose_cut = self.get_weights(events_loose_cut)
                weights_loose_cut.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_loose_cut, self.era, syst=""
                        ),
                        axis=-1,
                    ),
                )
                nMuon_loose_cut = ak.num(muons_loose_cut, axis=-1)
                output[dataset]["histograms"][
                    "SR_low_temp_loose_sph1_cut_vs_nMuon"
                ].fill(
                    sph1_cut=sph1_cut,
                    nMuon=ak.where(nMuon_loose_cut > 7, 7, nMuon_loose_cut),
                    weight=weights_loose_cut.weight(),
                )
            if len(events_tight_cut) > 0:
                weights_tight_cut = self.get_weights(events_tight_cut)
                weights_tight_cut.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_tight_cut, self.era, syst=""
                        ),
                        axis=-1,
                    ),
                )
                nMuon_tight_cut = ak.num(muons_tight_cut, axis=-1)
                output[dataset]["histograms"][
                    "SR_low_temp_tight_sph1_cut_vs_nMuon"
                ].fill(
                    sph1_cut=sph1_cut,
                    nMuon=ak.where(nMuon_tight_cut > 7, 7, nMuon_tight_cut),
                    weight=weights_tight_cut.weight(),
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
            "SR_low_temp_loose_ip3d_cut_vs_nMuon": hist.Hist.new.Reg(
                10,
                1e-3,
                0.1,
                name="ip3d_cut",
                label="ip3d_cut",
                transform=hist.axis.transform.log,
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "SR_low_temp_tight_ip3d_cut_vs_nMuon": hist.Hist.new.Reg(
                10,
                1e-3,
                0.1,
                name="ip3d_cut",
                label="ip3d_cut",
                transform=hist.axis.transform.log,
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "SR_low_temp_loose_sph1_cut_vs_nMuon": hist.Hist.new.Reg(
                10,
                0.1,
                0.9,
                name="sph1_cut",
                label="sph1_cut",
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "SR_low_temp_tight_sph1_cut_vs_nMuon": hist.Hist.new.Reg(
                10,
                0.1,
                0.9,
                name="sph1_cut",
                label="sph1_cut",
            )
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
