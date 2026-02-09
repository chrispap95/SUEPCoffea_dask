from collections.abc import Callable
from dataclasses import dataclass, field

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
class CRPromptSelectionResult:
    events: ak.Array
    muons: ak.Array
    prompt_muons: ak.Array
    qcd_muons: ak.Array
    z_cands: ak.Array | None = None
    candidates_indices: ak.Array | None = None


@dataclass(frozen=True)
class NMinusOneHistogramConfig:
    hist_key: str
    omit_prompt_cuts: tuple[str, ...] = ()
    omit_qcd_cuts: tuple[str, ...] = ()
    apply_mass_window: bool = True
    collection: str = "prompt"
    value_attr: str | None = None
    value_fn: Callable[[CRPromptSelectionResult], ak.Array] | None = None
    use_abs: bool = False
    flatten: bool = True
    include_genflav: bool = True
    sf_kwargs: dict[str, dict[str, object]] = field(default_factory=dict)


def _dimuon_mass_values(result: CRPromptSelectionResult) -> ak.Array:
    if result.z_cands is None:
        return ak.Array([])
    return result.z_cands.mass


CR_PROMPT_NMINUS1_CONFIGS: tuple[NMinusOneHistogramConfig, ...] = (
    NMinusOneHistogramConfig(
        hist_key="CR_prompt_Nminus1_dimuon_mass",
        apply_mass_window=False,
        value_fn=_dimuon_mass_values,
        include_genflav=False,
        flatten=False,
    ),
    NMinusOneHistogramConfig(
        hist_key="CR_prompt_Nminus1_prompt_muon_pt",
        omit_prompt_cuts=("pt",),
        collection="prompt",
        value_attr="pt",
        sf_kwargs={"prompt": {"override_low_pt_bound": True}},
    ),
    NMinusOneHistogramConfig(
        hist_key="CR_prompt_Nminus1_prompt_muon_iso",
        omit_prompt_cuts=("miniIsoId",),
        collection="prompt",
        value_attr="miniPFRelIso_all",
    ),
    NMinusOneHistogramConfig(
        hist_key="CR_prompt_Nminus1_prompt_muon_dxy",
        omit_prompt_cuts=("dxy",),
        collection="prompt",
        value_attr="dxy",
        use_abs=True,
    ),
    NMinusOneHistogramConfig(
        hist_key="CR_prompt_Nminus1_prompt_muon_dz",
        omit_prompt_cuts=("dz",),
        collection="prompt",
        value_attr="dz",
        use_abs=True,
    ),
    NMinusOneHistogramConfig(
        hist_key="CR_prompt_Nminus1_qcd_muon_dxy",
        omit_qcd_cuts=("dxy",),
        collection="qcd",
        value_attr="dxy",
        use_abs=True,
    ),
    NMinusOneHistogramConfig(
        hist_key="CR_prompt_Nminus1_qcd_muon_dz",
        omit_qcd_cuts=("dz",),
        collection="qcd",
        value_attr="dz",
        use_abs=True,
    ),
    NMinusOneHistogramConfig(
        hist_key="CR_prompt_Nminus1_qcd_muon_iso",
        omit_qcd_cuts=("miniIsoId",),
        collection="qcd",
        value_attr="miniPFRelIso_all",
    ),
)

PROMPT_BASE_CUTS: dict[str, Callable[[ak.Array], ak.Array]] = {
    "pt": lambda muons: muons.pt > 25,
    "miniIsoId": lambda muons: muons.miniIsoId >= 3,
    "dxy": lambda muons: abs(muons.dxy) < 0.01,
    "dz": lambda muons: abs(muons.dz) < 0.01,
}

QCD_BASE_CUTS: dict[str, Callable[[ak.Array], ak.Array]] = {
    "miniIsoId": lambda muons: muons.miniIsoId < 3,
    "dxy": lambda muons: abs(muons.dxy) > 0.01,
    "dz": lambda muons: abs(muons.dz) > 0.01,
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

    def _apply_CR_prompt_template(
        self,
        events,
        omit_prompt_cuts: tuple[str, ...] = (),
        omit_qcd_cuts: tuple[str, ...] = (),
        apply_mass_window: bool = True,
    ) -> CRPromptSelectionResult:
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        prompt_mask = ak.ones_like(muons.pt, dtype=bool)
        for cut_name, cut_fn in PROMPT_BASE_CUTS.items():
            if cut_name in omit_prompt_cuts:
                continue
            prompt_mask = prompt_mask & cut_fn(muons)
        prompt_muons = muons[prompt_mask]

        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        combined_mask = enough_prompt_muons & os_muons_mask
        muons = muons[combined_mask]
        events = events[combined_mask]
        prompt_muons = prompt_muons[combined_mask]

        events, prompt_muons, z_cands, candidates_indices, muons = (
            self.find_Z_candidates(events, prompt_muons, muons, apply_dR_cut=False)
        )

        if apply_mass_window and z_cands is not None:
            inside_mass_window = (
                abs(z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
            )
            events = events[inside_mass_window]
            muons = muons[inside_mass_window]
            prompt_muons = prompt_muons[inside_mass_window]
            z_cands = z_cands[inside_mass_window]
            candidates_indices = candidates_indices[inside_mass_window]

        qcd_mask = ak.ones_like(muons.pt, dtype=bool)
        for cut_name, cut_fn in QCD_BASE_CUTS.items():
            if cut_name in omit_qcd_cuts:
                continue
            qcd_mask = qcd_mask & cut_fn(muons)
        qcd_muons = muons[qcd_mask]
        is_prompt = ak.concatenate(
            [
                ak.ones_like(prompt_muons.pt, dtype=bool),
                ak.zeros_like(qcd_muons.pt, dtype=bool),
            ],
            axis=-1,
        )
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]
        prompt_muons = prompt_muons[select_by_muons_low]
        qcd_muons = qcd_muons[select_by_muons_low]
        is_prompt = is_prompt[select_by_muons_low]
        if z_cands is not None:
            z_cands = z_cands[select_by_muons_low]
        if candidates_indices is not None:
            candidates_indices = candidates_indices[select_by_muons_low]

        events, muons, keep_mask = self.remove_resonances(  # type: ignore[assignment]
            events,
            muons,
            veto_mode=False,
            return_mask=True,
        )
        is_prompt = is_prompt[keep_mask]
        prompt_muons = muons[is_prompt]
        qcd_muons = muons[~is_prompt]

        valid_events = ak.num(muons, axis=-1) > 0
        events = events[valid_events]
        muons = muons[valid_events]
        prompt_muons = prompt_muons[valid_events]
        qcd_muons = qcd_muons[valid_events]
        if z_cands is not None:
            z_cands = z_cands[valid_events]
        if candidates_indices is not None:
            candidates_indices = candidates_indices[valid_events]

        return CRPromptSelectionResult(
            events=events,
            muons=muons,
            prompt_muons=prompt_muons,
            qcd_muons=qcd_muons,
            z_cands=z_cands,
            candidates_indices=candidates_indices,
        )

    def apply_CR_cb_muon_dxy_cut(self, events):
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
        muons = muons[clean_muons]

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

        for config in CR_PROMPT_NMINUS1_CONFIGS:
            selection = self._apply_CR_prompt_template(
                events_,
                omit_prompt_cuts=config.omit_prompt_cuts,
                omit_qcd_cuts=config.omit_qcd_cuts,
                apply_mass_window=config.apply_mass_window,
            )
            if len(selection.events) == 0:
                continue

            weights_CR_prompt = self.get_weights(
                selection.events,
                do_vars=False,
                apply_lumi_factors=True,
            )
            if self.isMC:
                self._apply_muon_scale_factors(config, selection, weights_CR_prompt)

            values, genflav = self._extract_hist_arrays(config, selection)
            prepared_inputs = self._prepare_hist_inputs(
                config, values, genflav, weights_CR_prompt.weight()
            )
            if prepared_inputs is None:
                continue
            values_to_fill, weights_to_fill, genflav_to_fill = prepared_inputs

            histogram = output[dataset]["histograms"][config.hist_key]
            if genflav_to_fill is not None:
                if not self.isMC:
                    genflav_to_fill = np.zeros(len(genflav_to_fill), dtype=np.int64)
                histogram.fill(
                    values_to_fill,
                    genflav_to_fill,
                    weight=weights_to_fill,
                )
            else:
                histogram.fill(values_to_fill, weight=weights_to_fill)

        events_CR_cb, muons_CR_cb = self.apply_CR_cb_muon_dxy_cut(events_)

        events_CR_cb, muons_CR_cb = self.remove_resonances(  # type: ignore[assignment]
            events_CR_cb, muons_CR_cb, veto_mode=False
        )

        if len(events_CR_cb) > 0:
            weights_CR_cb = self.get_weights(
                events_CR_cb, do_vars=False, apply_lumi_factors=True
            )
            muons_CR_cb_genPartFlav = ak.zeros_like(muons_CR_cb.pt, dtype=np.int64)
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
                muons_CR_cb_genPartFlav = muons_CR_cb.genPartFlav
            output[dataset]["histograms"]["CR_cb_Nminus1_muon_dxy"].fill(
                ak.flatten(abs(muons_CR_cb.dxy)),
                ak.flatten(muons_CR_cb_genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_CR_cb.dxy,
                        weights_CR_cb.weight(),
                    )[1]
                ),
            )

        return

    def _apply_muon_scale_factors(
        self,
        config: NMinusOneHistogramConfig,
        selection: CRPromptSelectionResult,
        weights,
    ) -> None:
        prompt_kwargs = config.sf_kwargs.get("prompt", {})
        qcd_kwargs = config.sf_kwargs.get("qcd", {})

        prompt_sf = self._muon_sf_product(
            selection.prompt_muons,
            region="CR_prompt_prompt",
            **prompt_kwargs,
        )
        qcd_sf = self._muon_sf_product(
            selection.qcd_muons,
            region="CR_prompt_qcd",
            **qcd_kwargs,
        )
        weights.add("MuonSF", weight=prompt_sf * qcd_sf)

    def _muon_sf_product(self, muons, *, region: str, **kwargs):
        if len(muons) == 0:
            return ak.ones_like(muons)
        sf = muon_sf_utils.muon_efficiencies(
            muons,
            era=self.era,
            region=region,
            syst="",
            **kwargs,
        )
        return ak.prod(sf, axis=-1)

    def _extract_hist_arrays(
        self,
        config: NMinusOneHistogramConfig,
        selection: CRPromptSelectionResult,
    ) -> tuple[ak.Array, ak.Array | None]:
        if config.value_fn is not None:
            values = config.value_fn(selection)
            genflav = None
        else:
            muon_collection = (
                selection.prompt_muons
                if config.collection == "prompt"
                else selection.qcd_muons
            )
            if config.value_attr is None:
                raise ValueError(
                    f"value_attr must be provided when value_fn is not set for {config.hist_key}"
                )
            values = getattr(muon_collection, config.value_attr)
            if config.use_abs:
                values = abs(values)
            genflav = None
            if config.include_genflav:
                if self.isMC:
                    genflav = muon_collection.genPartFlav
                else:
                    genflav = ak.zeros_like(muon_collection, dtype=np.int64)
        return values, genflav

    def _prepare_hist_inputs(
        self,
        config: NMinusOneHistogramConfig,
        values: ak.Array,
        genflav: ak.Array | None,
        event_weights,
    ) -> tuple[ak.Array, ak.Array, ak.Array | None] | None:
        if len(values) == 0:
            return None

        if config.flatten:
            flat_values = ak.flatten(values)
            if len(flat_values) == 0:
                return None
            weights = ak.flatten(ak.broadcast_arrays(values, event_weights)[1])
            genflav_out = ak.flatten(genflav) if genflav is not None else None
            return flat_values, weights, genflav_out

        genflav_out = genflav if genflav is not None else None
        return values, event_weights, genflav_out

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
            "CR_prompt_Nminus1_dimuon_mass": hist.Hist.new.Regular(
                100,
                0,
                200,
                name="dimuon_mass",
                label="dimuon_mass",
            ).Weight(),
            "CR_prompt_Nminus1_prompt_muon_pt": hist.Hist.new.Regular(
                100,
                0,
                100,
                name="prompt_muon_pt",
                label="prompt_muon_pt",
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_prompt_muon_iso": hist.Hist.new.Regular(
                100,
                0.02,
                20,
                name="prompt_muon_iso",
                label="prompt_muon_iso",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_prompt_muon_dxy": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="prompt_muon_absdxy",
                label="prompt_muon_absdxy",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_prompt_muon_dz": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="prompt_muon_absdz",
                label="prompt_muon_absdz",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_qcd_muon_iso": hist.Hist.new.Regular(
                100,
                0.02,
                20,
                name="qcd_muon_iso",
                label="qcd_muon_iso",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_qcd_muon_dxy": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="qcd_muon_absdxy",
                label="qcd_muon_absdxy",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_qcd_muon_dz": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="qcd_muon_absdz",
                label="qcd_muon_absdz",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_cb_Nminus1_muon_dxy": hist.Hist.new.Regular(
                100,
                2e-4,
                2,
                name="muon_absdxy",
                label="muon_absdxy",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
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
