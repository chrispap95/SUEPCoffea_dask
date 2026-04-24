import awkward as ak
import hist
import numpy as np
import vector  # type: ignore[import]
from coffea import processor
from coffea.analysis_tools import Weights

import workflows.CMS_corrections.golden_json_utils as golden_json_utils
import workflows.SUEP_common as SUEP_common
from workflows.CMS_corrections import muon_sf_utils, systematics_utils

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

    def get_efficiency_weights(self, events):
        weights = Weights(len(events))
        if not self.isMC or len(events) == 0:
            return weights.weight()

        weights.add("genWeight", events.genWeight)
        weights.add("PUReweight", systematics_utils.pileup_weight(events, self.era))

        if self.era in ["2016", "2016APV", "2017", "2018"]:
            weights.add("L1PreFire", weight=events.L1PreFiringWeight.Nom)

        return weights.weight()

    def apply_dataset_filters(self, events):
        dataset = events.metadata["dataset"]

        if "WJetsToLNu_HT" in dataset:
            events = events[events.LHE.HT >= 70]
        elif "WJetsToLNu_TuneCP5" in dataset:
            events = events[events.LHE.HT < 70]

        if "DYJetsToLL_LHEFilterPtZ-0_MatchEWPDG20" in dataset:
            events = events[events.LHE.Vpt == 0]

        return events

    def apply_reference_trigger(self, events):
        ref_paths = ["PFHT1050", "PFJet550"]
        available = [p for p in ref_paths if p in events.HLT.fields]
        if not available:
            return events[np.zeros(len(events), dtype=bool)]
        mask = events.HLT[available[0]]
        for path in available[1:]:
            mask = mask | events.HLT[path]
        return events[mask]

    def calculate_HT(self, jets):
        return ak.sum(jets.pt, axis=-1)

    def apply_trigger_plateau(self, events):
        jets = events.Jet
        clean_jets = (
            (jets.pt > 30)
            & (abs(jets.eta) < 2.4)
            & (jets.jetId == 6)
            & (jets.puId == 7)
        )
        jets = jets[clean_jets]

        # # Separate muon veto
        # muons = events.Muon
        # clean_muons = (muons.pt > 10) & (abs(muons.eta) < 2.4) & (abs(muons.dz) < 0.2)
        # muons = muons[clean_muons]

        # jet_muon_dR = jets.metric_table(muons)
        # min_dR = ak.min(jet_muon_dR, axis=-1, mask_identity=False)
        # jets = jets[min_dR > 0.4]

        HT = self.calculate_HT(jets)
        nJet600 = ak.sum(jets.pt >= 600, axis=-1)

        return events[(HT >= 1100) | (nJet600 >= 1)]

    def select_reco_muons(self, events):
        muons = events.Muon
        nonempty = ak.num(muons) > 0
        events = events[nonempty]
        muons = muons[nonempty]

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dz) < 1.0)
        )
        muons = muons[clean_muons]

        has_three_muons = ak.num(muons, axis=-1) > 2
        events = events[has_three_muons]
        muons = muons[has_three_muons]

        muons = muons[ak.argsort(muons.pt, axis=-1, ascending=False)]
        return events, muons

    def get_reco_path_masks(self, muons):
        pt_ge_3 = ak.sum(muons.pt >= 3.0, axis=-1) >= 3
        pt_ge_5 = ak.sum(muons.pt >= 5.0, axis=-1)
        pt_ge_10 = ak.sum(muons.pt >= 10.0, axis=-1)
        pt_ge_12 = ak.sum(muons.pt >= 12.0, axis=-1)

        dimuons = ak.combinations(muons, 2, fields=["mu1", "mu2"])
        dimuon_masses = (dimuons.mu1 + dimuons.mu2).mass
        dimuon_dz = abs(dimuons.mu1.dz - dimuons.mu2.dz)
        # Keep only masses from pairs that pass the dz cut (mirrors mass_cut in HLT_eff.py)
        dz_filtered_masses = dimuon_masses[dimuon_dz < 0.2]

        has_all_dz = ak.sum(dimuon_dz < 0.2, axis=-1) >= 3
        has_mass3p8_dz = ak.sum(dz_filtered_masses > 3.8, axis=-1) >= 3
        has_mass3p8to60_dz = (ak.sum(dz_filtered_masses > 3.8, axis=-1) >= 3) & (
            ak.sum(dz_filtered_masses < 60.0, axis=-1) >= 3
        )

        masks = {
            "HLT_TripleMu_5_3_3": (pt_ge_5 >= 1) & pt_ge_3,
            "HLT_TripleMu_5_3_3_DZ_Mass3p8": (pt_ge_5 >= 1) & pt_ge_3 & has_mass3p8_dz,
            "HLT_TripleMu_5_3_3_Mass3p8_DZ": (pt_ge_5 >= 1) & pt_ge_3 & has_mass3p8_dz,
            "HLT_TripleMu_5_3_3_Mass3p8to60_DZ": (pt_ge_5 >= 1)
            & pt_ge_3
            & has_mass3p8to60_dz,
            "HLT_TripleMu_10_5_5_DZ": (pt_ge_10 >= 1) & (pt_ge_5 >= 3) & has_all_dz,
            "HLT_TripleMu_12_10_5": (pt_ge_12 >= 1) & (pt_ge_10 >= 2) & (pt_ge_5 >= 3),
        }
        return masks

    def fill_path_histograms(
        self,
        output,
        dataset: str,
        path: str,
        mask,
        weights,
        third_muon_pt,
        prefix: str,
    ):
        mask_np = ak.to_numpy(mask)
        if not np.any(mask_np):
            return

        output[dataset]["histograms"][f"{prefix}_{path}"].fill(
            third_muon_pt[mask_np],
            weight=weights[mask_np],
        )
        output[dataset]["histograms"][f"{prefix}_count_{path}"].fill(
            np.zeros(np.count_nonzero(mask_np)),
            weight=weights[mask_np],
        )

    def analysis(self, events, output):
        dataset = events.metadata["dataset"]

        weights = self.get_efficiency_weights(events)
        output[dataset]["cutflow"].fill(len(events) * ["all"], weight=weights)

        if not self.isMC:
            events = golden_json_utils.apply_golden_JSON(events, self.era)

        weights = self.get_efficiency_weights(events)
        output[dataset]["cutflow"].fill(
            len(events) * ["golden_or_input"],
            weight=weights,
        )

        events = self.apply_dataset_filters(events)
        weights = self.get_efficiency_weights(events)
        output[dataset]["cutflow"].fill(
            len(events) * ["dataset_filters"],
            weight=weights,
        )

        events = self.apply_reference_trigger(events)
        events = self.apply_trigger_plateau(events)
        weights = self.get_efficiency_weights(events)
        output[dataset]["cutflow"].fill(
            len(events) * ["reference_trigger"],
            weight=weights,
        )

        events, muons = self.select_reco_muons(events)
        if len(events) == 0:
            return

        weights = self.get_efficiency_weights(events)
        output[dataset]["cutflow"].fill(
            len(events) * ["baseline"],
            weight=weights,
        )

        reco_masks = self.get_reco_path_masks(muons)
        third_muon_pt = ak.to_numpy(muons.pt[:, 2])

        hlt_paths = self.get_triplemu_hlt_paths()
        reco_or_mask = np.zeros(len(events), dtype=bool)
        num_or_mask = np.zeros(len(events), dtype=bool)

        for path in hlt_paths:
            if not self.can_evaluate_triplemu_hlt_path(events, path):
                continue

            path_run_mask = self.get_triplemu_path_run_mask(events, path)
            reco_mask = reco_masks[path] & path_run_mask
            fired_mask = self.get_triplemu_hlt_mask(events, path)
            num_mask = reco_mask & fired_mask

            reco_or_mask = reco_or_mask | ak.to_numpy(reco_mask)
            num_or_mask = num_or_mask | ak.to_numpy(num_mask)

            self.fill_path_histograms(
                output,
                dataset,
                path,
                reco_mask,
                weights,
                third_muon_pt,
                "DEN",
            )
            self.fill_path_histograms(
                output,
                dataset,
                path,
                num_mask,
                weights,
                third_muon_pt,
                "NUM",
            )

        if np.any(reco_or_mask):
            output[dataset]["cutflow"].fill(
                np.count_nonzero(reco_or_mask) * ["reco_or"],
                weight=weights[reco_or_mask],
            )

        self.fill_path_histograms(
            output,
            dataset,
            "HLT_TripleMu_OR",
            reco_or_mask,
            weights,
            third_muon_pt,
            "DEN",
        )
        self.fill_path_histograms(
            output,
            dataset,
            "HLT_TripleMu_OR",
            num_or_mask,
            weights,
            third_muon_pt,
            "NUM",
        )

    def process(self, events):
        dataset = events.metadata["dataset"]
        cutflow = hist.Hist.new.StrCategory(
            [
                "all",
                "golden_or_input",
                "dataset_filters",
                "reference_trigger",
                "baseline",
                "reco_or",
            ],
            name="cutflow",
            label="cutflow",
        ).Weight()

        hlt_paths = self.get_triplemu_hlt_paths() + ["HLT_TripleMu_OR"]
        histograms = {}
        for path in hlt_paths:
            histograms[f"NUM_{path}"] = hist.Hist.new.Regular(
                80,
                0,
                40,
                name="subsubleading_muon_pt",
                label="subsubleading_muon_pt",
            ).Weight()
            histograms[f"DEN_{path}"] = hist.Hist.new.Regular(
                80,
                0,
                40,
                name="subsubleading_muon_pt",
                label="subsubleading_muon_pt",
            ).Weight()
            histograms[f"NUM_count_{path}"] = hist.Hist.new.Regular(
                1,
                0,
                1,
                name="event",
                label="event",
            ).Weight()
            histograms[f"DEN_count_{path}"] = hist.Hist.new.Regular(
                1,
                0,
                1,
                name="event",
                label="event",
            ).Weight()

        output = {
            dataset: {
                "cutflow": cutflow,
                "gensumweight": processor.value_accumulator(float, 0),
                "histograms": histograms,
            },
        }

        if self.isMC:
            self.gensumweight = ak.sum(events.genWeight)
            output[dataset]["gensumweight"].add(self.gensumweight)

        self.analysis(events, output)
        return output

    def postprocess(self, accumulator):
        pass
