from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass(slots=True)
class SRHighTempSelectionResult:
    tight_events: ak.Array
    loose_events: ak.Array
    tight_muons: ak.Array
    loose_muons: ak.Array
    tight_z_cands: ak.Array | None = None
    loose_z_cands: ak.Array | None = None


@dataclass(frozen=True)
class SRHighTempNMinusOneConfig:
    hist_key_tight: str
    hist_key_loose: str
    omit_tight_cuts: tuple[str, ...] = ()
    omit_loose_cuts: tuple[str, ...] = ()
    apply_mass_cut: bool = True
    value_attr: str | None = None
    value_fn: (
        Callable[[SRHighTempSelectionResult], tuple[ak.Array, ak.Array]] | None
    ) = None
    use_abs: bool = False
    flatten: bool = True
    include_genflav: bool = True


SR_HIGH_TEMP_Z_MASS_THRESHOLD = 70


def _neutral_iso_values(
    selection: SRHighTempSelectionResult,
) -> tuple[ak.Array, ak.Array]:
    tight = (
        selection.tight_muons.miniPFRelIso_all - selection.tight_muons.miniPFRelIso_chg
    )
    loose = (
        selection.loose_muons.miniPFRelIso_all - selection.loose_muons.miniPFRelIso_chg
    )
    return tight, loose


def _dimuon_mass_values(
    selection: SRHighTempSelectionResult,
) -> tuple[ak.Array, ak.Array]:
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


SR_HIGH_TEMP_NMINUS1_CONFIGS: tuple[SRHighTempNMinusOneConfig, ...] = (
    SRHighTempNMinusOneConfig(
        hist_key_tight="SR_high_temp_tight_Nminus1_muon_dxy",
        hist_key_loose="SR_high_temp_loose_Nminus1_muon_dxy",
        omit_tight_cuts=("dxy",),
        omit_loose_cuts=("dxy",),
        value_attr="dxy",
        use_abs=True,
    ),
    SRHighTempNMinusOneConfig(
        hist_key_tight="SR_high_temp_tight_Nminus1_muon_dz",
        hist_key_loose="SR_high_temp_loose_Nminus1_muon_dz",
        omit_tight_cuts=("dz",),
        omit_loose_cuts=("dz",),
        value_attr="dz",
        use_abs=True,
    ),
    SRHighTempNMinusOneConfig(
        hist_key_tight="SR_high_temp_tight_Nminus1_muon_iso",
        hist_key_loose="SR_high_temp_loose_Nminus1_muon_iso",
        omit_tight_cuts=("iso",),
        omit_loose_cuts=("iso",),
        value_attr="miniPFRelIso_all",
    ),
    SRHighTempNMinusOneConfig(
        hist_key_tight="SR_high_temp_tight_Nminus1_muon_neutral_iso",
        hist_key_loose="SR_high_temp_loose_Nminus1_muon_neutral_iso",
        omit_tight_cuts=("neutral_iso",),
        omit_loose_cuts=("neutral_iso",),
        value_fn=_neutral_iso_values,
    ),
    SRHighTempNMinusOneConfig(
        hist_key_tight="SR_high_temp_tight_Nminus1_dimuon_mass",
        hist_key_loose="SR_high_temp_loose_Nminus1_dimuon_mass",
        apply_mass_cut=False,
        value_fn=_dimuon_mass_values,
        flatten=False,
        include_genflav=False,
    ),
)


SR_HIGH_TEMP_TIGHT_CUTS: dict[str, Callable[[ak.Array], ak.Array]] = {
    "dxy": lambda mu: abs(mu.dxy) < 0.007,
    "dz": lambda mu: abs(mu.dz) < 0.007,
    "iso": lambda mu: mu.miniPFRelIso_all < 0.65,
    "neutral_iso": lambda mu: (mu.miniPFRelIso_all - mu.miniPFRelIso_chg) < 0.5,
}

SR_HIGH_TEMP_LOOSE_CUTS: dict[str, Callable[[ak.Array], ak.Array]] = {
    "dxy": lambda mu: abs(mu.dxy) < 0.1,
    "dz": lambda mu: abs(mu.dz) < 0.1,
    "iso": lambda mu: mu.miniPFRelIso_all < 5,
    "neutral_iso": lambda mu: (mu.miniPFRelIso_all - mu.miniPFRelIso_chg) < 3,
}


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

    def _apply_SR_high_temp_template(
        self,
        events,
        omit_tight_cuts: tuple[str, ...] = (),
        omit_loose_cuts: tuple[str, ...] = (),
        apply_mass_cut: bool = True,
    ) -> SRHighTempSelectionResult:
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        tight_mask = ak.ones_like(muons.pt, dtype=bool)
        for name, func in SR_HIGH_TEMP_TIGHT_CUTS.items():
            if name in omit_tight_cuts:
                continue
            tight_mask = tight_mask & func(muons)
        tight_muons = muons[tight_mask]

        loose_mask = ak.ones_like(muons.pt, dtype=bool)
        for name, func in SR_HIGH_TEMP_LOOSE_CUTS.items():
            if name in omit_loose_cuts:
                continue
            loose_mask = loose_mask & func(muons)
        loose_muons = muons[loose_mask]

        tight_events, tight_muons = self.remove_resonances(  # type: ignore[assignment]
            events, tight_muons, veto_mode=False, return_mask=False
        )

        loose_events, loose_muons = self.remove_resonances(  # type: ignore[assignment]
            events, loose_muons, veto_mode=False, return_mask=False
        )

        tight_events, tight_muons, tight_z_cands, _ = self.find_Z_candidates(
            tight_events, tight_muons
        )
        loose_events, loose_muons, loose_z_cands, _ = self.find_Z_candidates(
            loose_events, loose_muons
        )

        if apply_mass_cut:
            tight_mask = tight_z_cands.mass < SR_HIGH_TEMP_Z_MASS_THRESHOLD
            tight_events = tight_events[tight_mask]
            tight_muons = tight_muons[tight_mask]
            tight_z_cands = tight_z_cands[tight_mask]

            loose_mask = loose_z_cands.mass < SR_HIGH_TEMP_Z_MASS_THRESHOLD
            loose_events = loose_events[loose_mask]
            loose_muons = loose_muons[loose_mask]
            loose_z_cands = loose_z_cands[loose_mask]

        tight_select = ak.num(tight_muons, axis=-1) > 2
        tight_events = tight_events[tight_select]
        tight_muons = tight_muons[tight_select]
        tight_z_cands = tight_z_cands[tight_select]

        loose_select = ak.num(loose_muons, axis=-1) > 2
        loose_events = loose_events[loose_select]
        loose_muons = loose_muons[loose_select]
        loose_z_cands = loose_z_cands[loose_select]

        return SRHighTempSelectionResult(
            tight_events=tight_events,
            loose_events=loose_events,
            tight_muons=tight_muons,
            loose_muons=loose_muons,
            tight_z_cands=tight_z_cands,
            loose_z_cands=loose_z_cands,
        )

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        for config in SR_HIGH_TEMP_NMINUS1_CONFIGS:
            selection = self._apply_SR_high_temp_template(
                events_,
                omit_tight_cuts=config.omit_tight_cuts,
                omit_loose_cuts=config.omit_loose_cuts,
                apply_mass_cut=config.apply_mass_cut,
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
                            selection.tight_muons, region="SR_high_temp_tight"
                        ),
                    )
                if weights_loose is not None:
                    weights_loose.add(
                        "MuonSF",
                        weight=self._muon_sf_product(
                            selection.loose_muons, region="SR_high_temp_loose"
                        ),
                    )

            tight_values, loose_values, tight_genflav, loose_genflav = (
                self._extract_sr_arrays(config, selection)
            )
            self._fill_sr_histogram(
                output[dataset]["histograms"],
                config.hist_key_tight,
                tight_values,
                tight_genflav,
                weights_tight.weight() if weights_tight is not None else None,
                config.flatten,
            )
            self._fill_sr_histogram(
                output[dataset]["histograms"],
                config.hist_key_loose,
                loose_values,
                loose_genflav,
                weights_loose.weight() if weights_loose is not None else None,
                config.flatten,
            )

        return

    def _muon_sf_product(self, muons, *, region: str) -> ak.Array:
        if len(muons) == 0:
            return ak.ones_like(muons)
        sf = muon_sf_utils.muon_efficiencies(
            muons,
            era=self.era,
            region=region,
            syst="",
        )
        return ak.prod(sf, axis=-1)

    def _extract_sr_arrays(
        self,
        config: SRHighTempNMinusOneConfig,
        selection: SRHighTempSelectionResult,
    ) -> tuple[ak.Array, ak.Array, ak.Array | None, ak.Array | None]:
        if config.value_fn is not None:
            tight_values, loose_values = config.value_fn(selection)
        else:
            if config.value_attr is None:
                raise ValueError(
                    f"value_attr must be provided when value_fn is not set for {config.hist_key_tight}"
                )
            tight_values = getattr(selection.tight_muons, config.value_attr)
            loose_values = getattr(selection.loose_muons, config.value_attr)
            if config.use_abs:
                tight_values = abs(tight_values)
                loose_values = abs(loose_values)

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
            event_weights = ak.ones_like(ak.num(values, axis=-1), dtype=float)

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
            "SR_high_temp_tight_Nminus1_muon_dxy": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="muon_absdxy",
                label="muon_absdxy",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_loose_Nminus1_muon_dxy": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="muon_absdxy",
                label="muon_absdxy",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_tight_Nminus1_muon_dz": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="muon_absdz",
                label="muon_absdz",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_loose_Nminus1_muon_dz": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="muon_absdz",
                label="muon_absdz",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_tight_Nminus1_muon_iso": hist.Hist.new.Regular(
                100,
                0.02,
                20,
                name="muon_iso",
                label="muon_iso",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_loose_Nminus1_muon_iso": hist.Hist.new.Regular(
                100,
                0.02,
                20,
                name="muon_iso",
                label="muon_iso",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_tight_Nminus1_muon_neutral_iso": hist.Hist.new.Regular(
                100,
                0.02,
                20,
                name="muon_neutral_iso",
                label="muon_neutral_iso",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_loose_Nminus1_muon_neutral_iso": hist.Hist.new.Regular(
                100,
                0.02,
                20,
                name="muon_neutral_iso",
                label="muon_neutral_iso",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_tight_Nminus1_dimuon_mass": hist.Hist.new.Regular(
                100, 0, 200, name="dimuon_mass", label="dimuon_mass"
            ).Weight(),
            "SR_high_temp_loose_Nminus1_dimuon_mass": hist.Hist.new.Regular(
                100, 0, 200, name="dimuon_mass", label="dimuon_mass"
            ).Weight(),
        }

        output = {
            dataset: {
                "cutflow": cutflow,
                "gensumweight": processor.value_accumulator(float, 0),
                "histograms": histograms,
            },
        }

        # gen weights sum
        if self.isMC:
            self.gensumweight = ak.sum(events.genWeight)
            output[dataset]["gensumweight"].add(self.gensumweight)

        # run the analysis
        self.analysis(events, output)

        return output

    def postprocess(self, accumulator):
        pass
