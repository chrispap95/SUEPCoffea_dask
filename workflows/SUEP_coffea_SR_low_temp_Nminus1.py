from collections.abc import Callable
from dataclasses import dataclass

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
SR_LOW_TEMP_TIGHT_DIMUON_MASS_MAX = 35
SR_LOW_TEMP_LOOSE_DIMUON_MASS_MAX = 45
SR_LOW_TEMP_TIGHT_SPH1_THRESHOLD = 0.7
SR_LOW_TEMP_LOOSE_SPH1_THRESHOLD = 0.2


@dataclass(slots=True)
class SRLowTempBranchResult:
    events: ak.Array
    muons: ak.Array
    z_cands: ak.Array | None
    sph1: ak.Array


@dataclass(slots=True)
class SRLowTempSelectionResult:
    tight_events: ak.Array
    loose_events: ak.Array
    tight_muons: ak.Array
    loose_muons: ak.Array
    tight_z_cands: ak.Array | None = None
    loose_z_cands: ak.Array | None = None
    tight_sph1: ak.Array | None = None
    loose_sph1: ak.Array | None = None


@dataclass(frozen=True)
class SRLowTempNMinusOneConfig:
    hist_key_tight: str
    hist_key_loose: str
    omit_tight_cuts: tuple[str, ...] = ()
    omit_loose_cuts: tuple[str, ...] = ()
    apply_tight_mass_cut: bool = True
    apply_loose_mass_cut: bool = True
    apply_tight_sph1_cut: bool = True
    apply_loose_sph1_cut: bool = True
    value_attr: str | None = None
    value_fn: Callable[[SRLowTempSelectionResult], tuple[ak.Array, ak.Array]] | None = (
        None
    )
    use_abs: bool = False
    flatten: bool = True
    include_genflav: bool = True


def _z_mass_values(selection: SRLowTempSelectionResult) -> tuple[ak.Array, ak.Array]:
    tight = (
        selection.tight_z_cands.mass
        if selection.tight_z_cands is not None
        else ak.Array([])
    )
    loose = (
        selection.loose_z_cands.mass
        if selection.loose_z_cands is not None
        else ak.Array([])
    )
    return tight, loose


def _sph1_values(selection: SRLowTempSelectionResult) -> tuple[ak.Array, ak.Array]:
    tight = selection.tight_sph1 if selection.tight_sph1 is not None else ak.Array([])
    loose = selection.loose_sph1 if selection.loose_sph1 is not None else ak.Array([])
    return tight, loose


SR_LOW_TEMP_TIGHT_CUTS: dict[str, Callable[[ak.Array], ak.Array]] = {
    "pt": lambda mu: mu.pt < 35,
    "dxy": lambda mu: abs(mu.dxy) < 0.007,
    "dz": lambda mu: abs(mu.dz) < 0.007,
}

SR_LOW_TEMP_LOOSE_CUTS: dict[str, Callable[[ak.Array], ak.Array]] = {
    "pt": lambda mu: mu.pt < 45,
    "dxy": lambda mu: abs(mu.dxy) < 0.1,
    "dz": lambda mu: abs(mu.dz) < 0.1,
}


SR_LOW_TEMP_NMINUS1_CONFIGS: tuple[SRLowTempNMinusOneConfig, ...] = (
    SRLowTempNMinusOneConfig(
        hist_key_tight="SR_low_temp_tight_Nminus1_muon_pt",
        hist_key_loose="SR_low_temp_loose_Nminus1_muon_pt",
        omit_tight_cuts=("pt",),
        omit_loose_cuts=("pt",),
        value_attr="pt",
    ),
    SRLowTempNMinusOneConfig(
        hist_key_tight="SR_low_temp_tight_Nminus1_muon_dxy",
        hist_key_loose="SR_low_temp_loose_Nminus1_muon_dxy",
        omit_tight_cuts=("dxy",),
        omit_loose_cuts=("dxy",),
        value_attr="dxy",
        use_abs=True,
    ),
    SRLowTempNMinusOneConfig(
        hist_key_tight="SR_low_temp_tight_Nminus1_muon_dz",
        hist_key_loose="SR_low_temp_loose_Nminus1_muon_dz",
        omit_tight_cuts=("dz",),
        omit_loose_cuts=("dz",),
        value_attr="dz",
        use_abs=True,
    ),
    SRLowTempNMinusOneConfig(
        hist_key_tight="SR_low_temp_tight_Nminus1_dimuon_mass",
        hist_key_loose="SR_low_temp_loose_Nminus1_dimuon_mass",
        apply_tight_mass_cut=False,
        apply_loose_mass_cut=False,
        value_fn=_z_mass_values,
        flatten=False,
        include_genflav=False,
    ),
    SRLowTempNMinusOneConfig(
        hist_key_tight="SR_low_temp_tight_Nminus1_sph1",
        hist_key_loose="SR_low_temp_loose_Nminus1_sph1",
        apply_tight_sph1_cut=False,
        apply_loose_sph1_cut=False,
        value_fn=_sph1_values,
        flatten=False,
        include_genflav=False,
    ),
)


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

    def _apply_SR_low_temp_template(
        self,
        events,
        *,
        omit_tight_cuts: tuple[str, ...] = (),
        omit_loose_cuts: tuple[str, ...] = (),
        apply_tight_mass_cut: bool = True,
        apply_loose_mass_cut: bool = True,
        apply_tight_sph1_cut: bool = True,
        apply_loose_sph1_cut: bool = True,
    ) -> SRLowTempSelectionResult:
        muons = events.Muon

        filter_empty_events = (ak.num(events.Muon) > 0) & (ak.num(events.PFCands) > 0)
        events = events[filter_empty_events]
        muons = muons[filter_empty_events]

        if len(events) == 0:
            empty = ak.Array([])
            return SRLowTempSelectionResult(
                tight_events=events,
                loose_events=events,
                tight_muons=empty,
                loose_muons=empty,
                tight_z_cands=None,
                loose_z_cands=None,
                tight_sph1=ak.Array([]),
                loose_sph1=ak.Array([]),
            )

        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        tight_branch = self._select_sr_low_temp_branch(
            events,
            muons,
            cut_map=SR_LOW_TEMP_TIGHT_CUTS,
            omit_cuts=omit_tight_cuts,
            mass_threshold=SR_LOW_TEMP_TIGHT_DIMUON_MASS_MAX,
            apply_mass_cut=apply_tight_mass_cut,
            sph1_threshold=SR_LOW_TEMP_TIGHT_SPH1_THRESHOLD,
            apply_sph1_cut=apply_tight_sph1_cut,
        )
        loose_branch = self._select_sr_low_temp_branch(
            events,
            muons,
            cut_map=SR_LOW_TEMP_LOOSE_CUTS,
            omit_cuts=omit_loose_cuts,
            mass_threshold=SR_LOW_TEMP_LOOSE_DIMUON_MASS_MAX,
            apply_mass_cut=apply_loose_mass_cut,
            sph1_threshold=SR_LOW_TEMP_LOOSE_SPH1_THRESHOLD,
            apply_sph1_cut=apply_loose_sph1_cut,
        )

        return SRLowTempSelectionResult(
            tight_events=tight_branch.events,
            loose_events=loose_branch.events,
            tight_muons=tight_branch.muons,
            loose_muons=loose_branch.muons,
            tight_z_cands=tight_branch.z_cands,
            loose_z_cands=loose_branch.z_cands,
            tight_sph1=tight_branch.sph1,
            loose_sph1=loose_branch.sph1,
        )

    def _select_sr_low_temp_branch(
        self,
        events,
        muons,
        *,
        cut_map: dict[str, Callable[[ak.Array], ak.Array]],
        omit_cuts: tuple[str, ...],
        mass_threshold: float,
        apply_mass_cut: bool,
        sph1_threshold: float,
        apply_sph1_cut: bool,
    ) -> SRLowTempBranchResult:
        branch_muons = self._apply_branch_muon_cuts(muons, cut_map, omit_cuts)

        events_sel, branch_muons = self.remove_resonances(  # type: ignore[assignment]
            events, branch_muons, veto_mode=False, return_mask=False
        )

        events_sel, branch_muons, z_cands, _ = self.find_Z_candidates(
            events_sel, branch_muons
        )

        if len(events_sel) == 0:
            return SRLowTempBranchResult(
                events=events_sel,
                muons=branch_muons,
                z_cands=None,
                sph1=ak.Array([]),
            )

        if apply_mass_cut:
            mass_mask = z_cands.mass < mass_threshold
            events_sel = events_sel[mass_mask]
            branch_muons = branch_muons[mass_mask]
            z_cands = z_cands[mass_mask]
            if len(events_sel) == 0:
                return SRLowTempBranchResult(
                    events=events_sel,
                    muons=branch_muons,
                    z_cands=None,
                    sph1=ak.Array([]),
                )

        muon_multiplicity = ak.num(branch_muons, axis=-1) > 2
        events_sel = events_sel[muon_multiplicity]
        branch_muons = branch_muons[muon_multiplicity]
        z_cands = z_cands[muon_multiplicity]

        if len(events_sel) == 0:
            return SRLowTempBranchResult(
                events=events_sel,
                muons=branch_muons,
                z_cands=None,
                sph1=ak.Array([]),
            )

        tracks = self.get_clean_tracks(events_sel)
        suep_cand, suep_cluster = self.find_SUEP_candidate(tracks)
        found_suep = ~ak.is_none(suep_cand)
        events_sel = events_sel[found_suep]
        branch_muons = branch_muons[found_suep]
        z_cands = z_cands[found_suep]
        suep_cand = suep_cand[found_suep]
        suep_cluster = suep_cluster[found_suep]

        if len(events_sel) == 0:
            return SRLowTempBranchResult(
                events=events_sel,
                muons=branch_muons,
                z_cands=None,
                sph1=ak.Array([]),
            )

        sph1_values = self.S1(suep_cand, suep_cluster)
        if apply_sph1_cut:
            sph_mask = sph1_values > sph1_threshold
            events_sel = events_sel[sph_mask]
            branch_muons = branch_muons[sph_mask]
            z_cands = z_cands[sph_mask]
            sph1_values = sph1_values[sph_mask]

        if len(events_sel) == 0:
            return SRLowTempBranchResult(
                events=events_sel,
                muons=branch_muons,
                z_cands=None,
                sph1=ak.Array([]),
            )

        return SRLowTempBranchResult(
            events=events_sel,
            muons=branch_muons,
            z_cands=z_cands,
            sph1=sph1_values,
        )

    def _apply_branch_muon_cuts(
        self,
        muons: ak.Array,
        cut_map: dict[str, Callable[[ak.Array], ak.Array]],
        omit_cuts: tuple[str, ...],
    ) -> ak.Array:
        if len(muons) == 0:
            return muons

        mask = ak.ones_like(muons.pt, dtype=bool)
        for cut_name, cut_fn in cut_map.items():
            if cut_name in omit_cuts:
                continue
            mask = mask & cut_fn(muons)
        return muons[mask]  # type: ignore[type]

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        histograms = output[dataset]["histograms"]

        for config in SR_LOW_TEMP_NMINUS1_CONFIGS:
            selection = self._apply_SR_low_temp_template(
                events_,
                omit_tight_cuts=config.omit_tight_cuts,
                omit_loose_cuts=config.omit_loose_cuts,
                apply_tight_mass_cut=config.apply_tight_mass_cut,
                apply_loose_mass_cut=config.apply_loose_mass_cut,
                apply_tight_sph1_cut=config.apply_tight_sph1_cut,
                apply_loose_sph1_cut=config.apply_loose_sph1_cut,
            )
            if len(selection.tight_events) == 0 and len(selection.loose_events) == 0:
                continue

            weights_tight = None
            weights_loose = None
            if len(selection.tight_events) > 0:
                weights_tight = self.get_weights(selection.tight_events)
            if len(selection.loose_events) > 0:
                weights_loose = self.get_weights(selection.loose_events)

            if self.isMC:
                if weights_tight is not None:
                    weights_tight.add(
                        "MuonSF",
                        weight=self._muon_sf_product(
                            selection.tight_muons,
                            region="SR_low_temp_tight",
                        ),
                    )
                if weights_loose is not None:
                    weights_loose.add(
                        "MuonSF",
                        weight=self._muon_sf_product(
                            selection.loose_muons,
                            region="SR_low_temp_loose",
                        ),
                    )

            (
                tight_values,
                loose_values,
                tight_genflav,
                loose_genflav,
            ) = self._extract_sr_arrays(config, selection)

            self._fill_sr_histogram(
                histograms,
                config.hist_key_tight,
                tight_values,
                tight_genflav,
                weights_tight.weight() if weights_tight is not None else None,
                config.flatten,
            )
            self._fill_sr_histogram(
                histograms,
                config.hist_key_loose,
                loose_values,
                loose_genflav,
                weights_loose.weight() if weights_loose is not None else None,
                config.flatten,
            )

        return

    def _muon_sf_product(self, muons, *, region: str) -> ak.Array:
        if len(muons) == 0:
            return ak.Array([])
        sf = muon_sf_utils.muon_efficiencies(
            muons,
            era=self.era,
            region=region,
            syst="",
        )
        return ak.prod(sf, axis=-1)

    def _extract_sr_arrays(
        self,
        config: SRLowTempNMinusOneConfig,
        selection: SRLowTempSelectionResult,
    ) -> tuple[ak.Array, ak.Array, ak.Array | None, ak.Array | None]:
        if config.value_fn is not None:
            tight_values, loose_values = config.value_fn(selection)
        elif config.value_attr is not None:
            tight_values = getattr(selection.tight_muons, config.value_attr)
            loose_values = getattr(selection.loose_muons, config.value_attr)
            if config.use_abs:
                tight_values = abs(tight_values)
                loose_values = abs(loose_values)
        else:
            raise ValueError(
                f"value_attr must be provided when value_fn is not set for {config.hist_key_tight}"
            )

        tight_genflav: ak.Array | None = None
        loose_genflav: ak.Array | None = None
        if config.include_genflav:
            if self.isMC:
                tight_genflav = selection.tight_muons.genPartFlav
                loose_genflav = selection.loose_muons.genPartFlav
            else:
                tight_genflav = ak.zeros_like(selection.tight_muons.pt, dtype=np.int64)
                loose_genflav = ak.zeros_like(selection.loose_muons.pt, dtype=np.int64)

        return tight_values, loose_values, tight_genflav, loose_genflav

    def _prepare_sr_hist_inputs(
        self,
        values: ak.Array,
        genflav: ak.Array | None,
        event_weights,
        *,
        flatten: bool,
    ) -> tuple[ak.Array, ak.Array, ak.Array | None] | None:
        if len(values) == 0:
            return None

        if event_weights is None:
            if flatten:
                event_weights = ak.ones_like(ak.num(values, axis=-1), dtype=float)
            else:
                event_weights = ak.ones_like(values, dtype=float)

        if flatten:
            flat_values = ak.flatten(values)
            if len(flat_values) == 0:
                return None
            flat_weights = ak.flatten(ak.broadcast_arrays(values, event_weights)[1])
            flat_genflav = ak.flatten(genflav) if genflav is not None else None
            return flat_values, flat_weights, flat_genflav

        genflav_out = genflav if genflav is not None else None
        return values, event_weights, genflav_out

    def _fill_sr_histogram(
        self,
        histograms,
        hist_key: str,
        values: ak.Array,
        genflav: ak.Array | None,
        event_weights,
        flatten: bool,
    ) -> None:
        prepared = self._prepare_sr_hist_inputs(
            values,
            genflav,
            event_weights,
            flatten=flatten,
        )
        if prepared is None:
            return

        values_to_fill, weights_to_fill, genflav_to_fill = prepared
        histogram = histograms[hist_key]
        if genflav_to_fill is not None:
            histogram.fill(values_to_fill, genflav_to_fill, weight=weights_to_fill)
        else:
            histogram.fill(values_to_fill, weight=weights_to_fill)

    def analysis(self, events, output):
        dataset = events.metadata["dataset"]
        weights = self.get_weights(events)

        # Fill the cutflow columns for all
        output[dataset]["cutflow"].fill(len(events) * ["all"], weight=weights.weight())

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
            "SR_low_temp_tight_Nminus1_muon_pt": hist.Hist.new.Regular(
                100, 0, 100, name="muon_pt", label="muon_pt"
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_low_temp_loose_Nminus1_muon_pt": hist.Hist.new.Regular(
                100, 0, 100, name="muon_pt", label="muon_pt"
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_low_temp_tight_Nminus1_muon_dxy": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="muon_absdxy",
                label="muon_absdxy",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_low_temp_loose_Nminus1_muon_dxy": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="muon_absdxy",
                label="muon_absdxy",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_low_temp_tight_Nminus1_muon_dz": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="muon_absdz",
                label="muon_absdz",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_low_temp_loose_Nminus1_muon_dz": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="muon_absdz",
                label="muon_absdz",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_low_temp_tight_Nminus1_dimuon_mass": hist.Hist.new.Regular(
                100, 0, 200, name="dimuon_mass", label="dimuon_mass"
            ).Weight(),
            "SR_low_temp_loose_Nminus1_dimuon_mass": hist.Hist.new.Regular(
                100, 0, 200, name="dimuon_mass", label="dimuon_mass"
            ).Weight(),
            "SR_low_temp_tight_Nminus1_sph1": hist.Hist.new.Regular(
                100, 0, 1, name="sph1", label="sph1"
            ).Weight(),
            "SR_low_temp_loose_Nminus1_sph1": hist.Hist.new.Regular(
                100, 0, 1, name="sph1", label="sph1"
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
