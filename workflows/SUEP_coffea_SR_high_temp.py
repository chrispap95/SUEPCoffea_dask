import awkward as ak
import hist
import numpy as np
import vector  # type: ignore[import]
from coffea import processor
from coffea.analysis_tools import Weights

# Importing CMS corrections
import workflows.CMS_corrections.golden_json_utils as golden_json_utils
import workflows.CMS_corrections.muon_sf_utils as muon_sf_utils
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
        do_syst: bool = False,
    ) -> None:
        self.isMC = isMC
        self.era = era if isinstance(era, str) else str(era)
        self.do_syst = do_syst
        self.gensumweight = 1.0

    def trigger_selection(self, events):
        """
        Applies trigger, returns events.
        """
        trigger = np.zeros(len(events), dtype=bool)
        if self.era in ["2016", "2016APV"]:
            if "TripleMu_5_3_3" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_5_3_3 == 1)
            if "TripleMu_5_3_3_DZ_Mass3p8" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_5_3_3_DZ_Mass3p8 == 1)
            if "TripleMu_12_10_5" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_12_10_5 == 1)
        elif self.era == "2017":
            if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1)
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_10_5_5_DZ == 1)
            if "TripleMu_12_10_5" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_12_10_5 == 1)
        elif self.era == "2018":
            if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1)
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1)
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_10_5_5_DZ == 1)
            if "TripleMu_12_10_5" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_12_10_5 == 1)
        elif self.era in ["2022", "2023"]:
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1)
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_10_5_5_DZ == 1)
            if "TripleMu_12_10_5" in events.HLT.fields:
                trigger = trigger | (events.HLT.TripleMu_12_10_5 == 1)
        else:
            raise ValueError(f"Invalid era: {self.era}")
        events = events[trigger]
        return events

    def get_weights(self, events):
        weights = Weights(len(events))
        if not self.isMC:
            return weights

        # Generator weights
        weights.add("genWeight", events.genWeight)

        # Pileup weights
        weights.add(
            "PUReweight",
            weight=systematics_utils.pileup_weight(events, self.era),
            weightUp=systematics_utils.pileup_weight(events, self.era, syst="up"),
            weightDown=systematics_utils.pileup_weight(events, self.era, syst="down"),
        )

        # L1 prefire weights
        weights.add(
            "L1PreFire",
            weight=events.L1PreFiringWeight.Nom,
            weightUp=events.L1PreFiringWeight.Up,
            weightDown=events.L1PreFiringWeight.Dn,
        )

        # Parton shower weights
        weights.add(
            "ISR",
            weight=np.ones(len(events)),
            weightUp=systematics_utils.get_PS_weights(events, syst="ISR_up"),
            weightDown=systematics_utils.get_PS_weights(events, syst="ISR_down"),
        )
        weights.add(
            "FSR",
            weight=np.ones(len(events)),
            weightUp=systematics_utils.get_PS_weights(events, syst="FSR_up"),
            weightDown=systematics_utils.get_PS_weights(events, syst="FSR_down"),
        )

        # Matrix element PDF and scale weights
        weights.add(
            "LHEPdf",
            weight=np.ones(len(events)),
            weightUp=systematics_utils.get_pdf_variations(events, syst="up"),
            weightDown=systematics_utils.get_pdf_variations(events, syst="down"),
        )
        weights.add(
            "LHEScaleMuR",
            weight=np.ones(len(events)),
            weightUp=systematics_utils.get_scale_variations(events, syst="MuRUp"),
            weightDown=systematics_utils.get_scale_variations(events, syst="MuRDown"),
        )
        weights.add(
            "LHEScaleMuF",
            weight=np.ones(len(events)),
            weightUp=systematics_utils.get_scale_variations(events, syst="MuFUp"),
            weightDown=systematics_utils.get_scale_variations(events, syst="MuFDown"),
        )

        return weights

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

        # Tight SR selection
        tight_cut = (
            (events.Muon.pt < 45)
            & (events.Muon.ip3d < 0.008)
            & (events.Muon.miniPFRelIso_all < 0.65)
            & ((events.Muon.miniPFRelIso_all - events.Muon.miniPFRelIso_chg) < 0.5)
        )
        muons_tight_cut = muons[clean_muons & tight_cut]
        events_tight_cut, muons_tight_cut, Z_cands_tight_cut = self.find_Z_candidates(
            events, muons_tight_cut
        )
        outside_mass_window_tight_cut = (
            abs(Z_cands_tight_cut.mass - Z_MASS) > 2 * Z_WIDTH
        )
        events_tight_cut = events_tight_cut[outside_mass_window_tight_cut]
        muons_tight_cut = muons_tight_cut[outside_mass_window_tight_cut]

        select_by_muons_tight = ak.num(muons_tight_cut, axis=-1) > 2
        events_tight_cut = events_tight_cut[select_by_muons_tight]
        muons_tight_cut = muons_tight_cut[select_by_muons_tight]

        # Loose SR selection
        loose_cut = (
            (events.Muon.ip3d < 0.1)
            & (events.Muon.miniPFRelIso_all < 10)
            & ((events.Muon.miniPFRelIso_all - events.Muon.miniPFRelIso_chg) < 10)
        )
        muons_loose_cut = muons[clean_muons & loose_cut]
        events_loose_cut, muons_loose_cut, Z_cands_loose_cut = self.find_Z_candidates(
            events, muons_loose_cut
        )
        outside_mass_window_loose_cut = (
            abs(Z_cands_loose_cut.mass - Z_MASS) > 2 * Z_WIDTH
        )
        events_loose_cut = events_loose_cut[outside_mass_window_loose_cut]
        muons_loose_cut = muons_loose_cut[outside_mass_window_loose_cut]
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

        weights_SR_high_temp_tight = self.get_weights(events_SR_high_temp_tight)
        weights_SR_high_temp_tight.add(
            "MuonSF",
            weight=ak.prod(
                muon_sf_utils.muon_scale_factors(muons_SR_high_temp_tight, syst=""),
                axis=-1,
            ),
            weightUp=ak.prod(
                muon_sf_utils.muon_scale_factors(muons_SR_high_temp_tight, syst="up"),
                axis=-1,
            ),
            weightDown=ak.prod(
                muon_sf_utils.muon_scale_factors(muons_SR_high_temp_tight, syst="down"),
                axis=-1,
            ),
        )
        nMuon_SR_high_temp_tight = ak.num(muons_SR_high_temp_tight, axis=-1)
        output[dataset]["histograms"]["SR_high_temp_tight"].fill(
            ak.where(nMuon_SR_high_temp_tight > 7, 7, nMuon_SR_high_temp_tight),
            weight=weights_SR_high_temp_tight.weight(),
        )
        if self.do_syst:
            for syst in weights_SR_high_temp_tight.variations:
                output[dataset]["histograms"][f"SR_high_temp_tight_{syst}"] = (
                    output[dataset]["histograms"]["SR_high_temp_tight"].copy().reset()
                )
                output[dataset]["histograms"][f"SR_high_temp_tight_{syst}"].fill(
                    ak.where(nMuon_SR_high_temp_tight > 7, 7, nMuon_SR_high_temp_tight),
                    weight=weights_SR_high_temp_tight.weight(syst),
                )

        weights_SR_high_temp_loose = self.get_weights(events_SR_high_temp_loose)
        weights_SR_high_temp_loose.add(
            "MuonSF",
            weight=ak.prod(
                muon_sf_utils.muon_scale_factors(muons_SR_high_temp_loose, syst=""),
                axis=-1,
            ),
            weightUp=ak.prod(
                muon_sf_utils.muon_scale_factors(muons_SR_high_temp_loose, syst="up"),
                axis=-1,
            ),
            weightDown=ak.prod(
                muon_sf_utils.muon_scale_factors(muons_SR_high_temp_loose, syst="down"),
                axis=-1,
            ),
        )
        nMuon_SR_high_temp_loose = ak.num(muons_SR_high_temp_loose, axis=-1)
        output[dataset]["histograms"]["SR_high_temp_loose"].fill(
            ak.where(nMuon_SR_high_temp_loose > 7, 7, nMuon_SR_high_temp_loose),
            weight=weights_SR_high_temp_loose.weight(),
        )
        if self.do_syst:
            for syst in weights_SR_high_temp_loose.variations:
                output[dataset]["histograms"][f"SR_high_temp_loose_{syst}"] = (
                    output[dataset]["histograms"]["SR_high_temp_loose"].copy().reset()
                )
                output[dataset]["histograms"][f"SR_high_temp_loose_{syst}"].fill(
                    ak.where(nMuon_SR_high_temp_loose > 7, 7, nMuon_SR_high_temp_loose),
                    weight=weights_SR_high_temp_loose.weight(syst),
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
            "SR_high_temp_tight": hist.Hist.new.Regular(
                5, 3, 8, name="nMuon", label="nMuon"
            ).Weight(),
            "SR_high_temp_loose": hist.Hist.new.Regular(
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

        # gen weights sum
        if self.isMC:
            self.gensumweight = ak.sum(events.genWeight)
            output[dataset]["gensumweight"].add(self.gensumweight)

        # run the analysis
        self.analysis(events, output)

        return output

    def postprocess(self, accumulator):
        pass
