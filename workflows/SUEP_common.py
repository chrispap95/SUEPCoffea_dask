import awkward as ak
import numpy as np
from coffea import processor
from coffea.analysis_tools import Weights

from workflows.CMS_corrections import muon_sf_utils, systematics_utils, trigger_sf_utils

Z_MASS = 91.1876
Z_WIDTH = 2.4952


class SUEP_base(processor.ProcessorABC):
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

    hlt_path_lumi = {
        "2016APV": {
            "HLT_TripleMu_5_3_3": {
                "total lumi": 19.501601622,
                "path lumi": 7.657859683,
                "simulated": True,
            },
            "HLT_TripleMu_12_10_5": {
                "total lumi": 19.501601622,
                "path lumi": 19.497897120,
                "simulated": True,
            },
        },
        "2016": {
            "HLT_TripleMu_5_3_3": {
                "total lumi": 16.812151722,
                "path lumi": 0.388200744,
                "simulated": True,
            },
            "HLT_TripleMu_5_3_3_DZ_Mass3p8": {
                "total lumi": 16.812151722,
                "path lumi": 8.740119304,
                "simulated": False,
            },
            "HLT_TripleMu_12_10_5": {
                "total lumi": 16.812151722,
                "path lumi": 16.812151722,
                "simulated": True,
            },
        },
        "2017": {
            "HLT_TripleMu_5_3_3_Mass3p8to60_DZ": {
                "total lumi": 41.479849142,
                "path lumi": 24.259691276,
                "simulated": True,
            },
            "HLT_TripleMu_10_5_5_DZ": {
                "total lumi": 41.479849142,
                "path lumi": 41.478046012,
                "simulated": True,
            },
            "HLT_TripleMu_12_10_5": {
                "total lumi": 41.479849142,
                "path lumi": 41.478046012,
                "simulated": True,
            },
        },
        "2018": {
            "HLT_TripleMu_5_3_3_Mass3p8_DZ": {
                "total lumi": 59.832422397,
                "path lumi": 54.536814521,
                "simulated": True,
            },
            "HLT_TripleMu_5_3_3_Mass3p8to60_DZ": {
                "total lumi": 59.832422397,
                "path lumi": 5.291012014,
                "simulated": False,
            },
            "HLT_TripleMu_10_5_5_DZ": {
                "total lumi": 59.832422397,
                "path lumi": 59.827826535,
                "simulated": True,
            },
            "HLT_TripleMu_12_10_5": {
                "total lumi": 59.832422397,
                "path lumi": 59.827826535,
                "simulated": True,
            },
        },
    }

    def emulate_HLT_TripleMu_5_3_3_DZ_Mass3p8(self, events):
        # Begin from the HLT_TripleMu_5_3_3 trigger
        trigger = events.HLT.TripleMu_5_3_3
        events = ak.mask(events, trigger)

        # Check if there are at least 3 muons
        trigger = trigger & (ak.num(events.Muon) > 2)
        events = ak.mask(events, ak.num(events.Muon) > 2)
        muons = events.Muon

        # Make unique muon pairs
        muon_idx = ak.local_index(muons)
        muon_pairs = ak.unzip(ak.cartesian([muons, muons]))
        muon_pairs_idx = ak.unzip(ak.cartesian([muon_idx, muon_idx]))
        unique_pairs = muon_pairs_idx[0] < muon_pairs_idx[1]  # type: ignore[attr-defined]
        muons1 = ak.mask(muon_pairs[0], unique_pairs)
        muons2 = ak.mask(muon_pairs[1], unique_pairs)  # type: ignore[attr-defined]

        # Reject pairs with large dz
        dz = muons1.dz - muons2.dz
        muons1 = ak.mask(muons1, abs(dz) < 0.2)
        muons2 = ak.mask(muons2, abs(dz) < 0.2)

        # Check if there is at least one pair with mass > 3.8
        os_dimuons = muons1 + muons2
        trigger = trigger & (ak.sum(os_dimuons.mass > 3.8, axis=-1) > 0)
        return ak.fill_none(trigger, False)

    def emulate_HLT_TripleMu_5_3_3_Mass3p8to60_DZ(self, events):
        # Begin from the HLT_TripleMu_5_3_3_Mass3p8 trigger
        trigger = events.HLT.TripleMu_5_3_3_Mass3p8_DZ
        events = ak.mask(events, trigger)

        # Check if there are at least 3 muons
        trigger = trigger & (ak.num(events.Muon) > 2)
        events = ak.mask(events, ak.num(events.Muon) > 2)
        muons = events.Muon

        # Make unique muon pairs
        muon_idx = ak.local_index(muons)
        muon_pairs = ak.unzip(ak.cartesian([muons, muons]))
        muon_pairs_idx = ak.unzip(ak.cartesian([muon_idx, muon_idx]))
        unique_pairs = muon_pairs_idx[0] < muon_pairs_idx[1]  # type: ignore[attr-defined]
        muons1 = ak.mask(muon_pairs[0], unique_pairs)
        muons2 = ak.mask(muon_pairs[1], unique_pairs)  # type: ignore[attr-defined]

        # Reject pairs with large dz
        dz = muons1.dz - muons2.dz
        muons1 = ak.mask(muons1, abs(dz) < 0.2)
        muons2 = ak.mask(muons2, abs(dz) < 0.2)

        # Check if there is at least one pair with mass > 3.8
        os_dimuons = muons1 + muons2
        trigger = (
            trigger
            & (ak.sum(os_dimuons.mass > 3.8, axis=-1) > 0)
            & (ak.sum(os_dimuons.mass < 60, axis=-1) > 0)
        )
        return ak.fill_none(trigger, False)

    def trigger_selection(self, events):
        """
        Applies trigger, returns events.

        Parameters
        ----------
        events : awkward array
            The events to be filtered.
        """
        trigger = np.zeros(len(events), dtype=bool)
        if self.era == "2016APV":
            if "TripleMu_5_3_3" in events.HLT.fields:
                trigger = trigger | events.HLT.TripleMu_5_3_3
            trigger = trigger | events.HLT.TripleMu_12_10_5
        elif self.era == "2016":
            if "TripleMu_5_3_3" in events.HLT.fields:
                trigger = trigger | events.HLT.TripleMu_5_3_3
            # if "TripleMu_5_3_3_DZ_Mass3p8" in events.HLT.fields:
            #     trigger = trigger | events.HLT.TripleMu_5_3_3_DZ_Mass3p8
            # elif "TripleMu_5_3_3" in events.HLT.fields and self.isMC:
            #     missing_path_2016 = self.emulate_HLT_TripleMu_5_3_3_DZ_Mass3p8(events)
            #     trigger = trigger | missing_path_2016
            trigger = trigger | events.HLT.TripleMu_12_10_5
        elif self.era == "2017":
            if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
                trigger = trigger | events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ
            trigger = trigger | events.HLT.TripleMu_10_5_5_DZ
            trigger = trigger | events.HLT.TripleMu_12_10_5
        elif self.era == "2018":
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                trigger = trigger | events.HLT.TripleMu_5_3_3_Mass3p8_DZ
            # if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
            #     trigger = trigger | events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ
            # elif "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields and self.isMC:
            #     missing_path_2018 = self.emulate_HLT_TripleMu_5_3_3_Mass3p8to60_DZ(
            #         events
            #     )
            #     trigger = trigger | missing_path_2018
            trigger = trigger | events.HLT.TripleMu_10_5_5_DZ
            trigger = trigger | events.HLT.TripleMu_12_10_5
        elif self.era in ["2022", "2022EE", "2023", "2023BPix"]:
            trigger = trigger | events.HLT.TripleMu_5_3_3_Mass3p8_DZ
            trigger = trigger | events.HLT.TripleMu_10_5_5_DZ
            trigger = trigger | events.HLT.TripleMu_12_10_5
        else:
            raise ValueError(f"Invalid era: {self.era}")
        events = events[trigger]
        return events

    def apply_trigger_plateau(
        self,
        events,
        pt12_threshold: float = 12,
        pt10_threshold: float = 10,
        pt5_threshold: float = 5,
        pt3_threshold: float = 4,
    ):
        """
        To make sure we are in the trigger plateau, we require that
        there are at least 3 RECO muons with pt greater than the trigger
        threshold of the lowest HLT path +0.5 GeV.
        """

        muons = events.Muon
        muon_cleaning = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[muon_cleaning]

        selection_533 = (ak.sum(muons.pt >= pt5_threshold, axis=-1) >= 1) & (
            ak.sum(muons.pt >= pt3_threshold, axis=-1) >= 3
        )
        selection_1055 = (ak.sum(muons.pt >= pt10_threshold, axis=-1) >= 1) & (
            ak.sum(muons.pt >= pt5_threshold, axis=-1) >= 3
        )
        selection_12105 = (
            (ak.sum(muons.pt >= pt12_threshold, axis=-1) >= 1)
            & (ak.sum(muons.pt >= pt10_threshold, axis=-1) >= 2)
            & (ak.sum(muons.pt >= pt5_threshold, axis=-1) >= 3)
        )

        # Now, blend the selections based on the trigger paths
        trigger_plateau = np.ones(len(events), dtype=bool)
        if self.era in ["2016APV", "2016"]:
            if "TripleMu_5_3_3" in events.HLT.fields:
                trigger_plateau = np.where(
                    events.HLT.TripleMu_5_3_3,
                    trigger_plateau & selection_533,
                    trigger_plateau,
                )
            trigger_plateau = np.where(
                events.HLT.TripleMu_12_10_5,
                trigger_plateau & selection_12105,
                trigger_plateau,
            )
        elif self.era == "2017":
            if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
                trigger_plateau = np.where(
                    events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ,
                    trigger_plateau & selection_533,
                    trigger_plateau,
                )
            trigger_plateau = np.where(
                events.HLT.TripleMu_10_5_5_DZ,
                trigger_plateau & selection_1055,
                trigger_plateau,
            )
            trigger_plateau = np.where(
                events.HLT.TripleMu_12_10_5,
                trigger_plateau & selection_12105,
                trigger_plateau,
            )
        elif self.era == "2018":
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                trigger_plateau = np.where(
                    events.HLT.TripleMu_5_3_3_Mass3p8_DZ,
                    trigger_plateau & selection_533,
                    trigger_plateau,
                )
            trigger_plateau = np.where(
                events.HLT.TripleMu_10_5_5_DZ,
                trigger_plateau & selection_1055,
                trigger_plateau,
            )
            trigger_plateau = np.where(
                events.HLT.TripleMu_12_10_5,
                trigger_plateau & selection_12105,
                trigger_plateau,
            )
        elif self.era in ["2022", "2022EE", "2023", "2023BPix"]:
            trigger_plateau = np.where(
                events.HLT.TripleMu_5_3_3_Mass3p8_DZ,
                trigger_plateau & selection_533,
                trigger_plateau,
            )
            trigger_plateau = np.where(
                events.HLT.TripleMu_10_5_5_DZ,
                trigger_plateau & selection_1055,
                trigger_plateau,
            )
            trigger_plateau = np.where(
                events.HLT.TripleMu_12_10_5,
                trigger_plateau & selection_12105,
                trigger_plateau,
            )

        return trigger_plateau

    def get_lumi_factors(self, events):
        lumi_factors = np.ones(len(events))
        if self.era == "2016APV":
            lumi_factors = np.where(
                events.HLT.TripleMu_5_3_3 & ~events.HLT.TripleMu_12_10_5,
                7.657859683 / 19.497897120,
                lumi_factors,
            )
        if self.era == "2016":
            lumi_factors = np.where(
                events.HLT.TripleMu_5_3_3 & ~events.HLT.TripleMu_12_10_5,
                0.3882007446 / 16.812151722,
                lumi_factors,
            )
            # lumi_factors_2 = np.where(
            #     self.emulate_HLT_TripleMu_5_3_3_DZ_Mass3p8(events)
            #     & ~events.HLT.TripleMu_12_10_5,
            #     8.740119304 / 16.812151722,
            #     0,
            # )
            # lumi_factors = np.where(
            #     (lumi_factors_1 + lumi_factors_2) > 0,
            #     lumi_factors_1 + lumi_factors_2,
            #     lumi_factors,
            # )
        if self.era == "2017":
            lumi_factors = np.where(
                events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ
                & ~events.HLT.TripleMu_10_5_5_DZ
                & ~events.HLT.TripleMu_12_10_5,
                24.259691276 / 41.478046012,
                lumi_factors,
            )
        if self.era == "2018":
            lumi_factors = np.where(
                events.HLT.TripleMu_5_3_3_Mass3p8_DZ
                & ~events.HLT.TripleMu_10_5_5_DZ
                & ~events.HLT.TripleMu_12_10_5,
                54.536814521 / 59.827826535,
                lumi_factors,
            )
            # lumi_factors_2 = np.where(
            #     self.emulate_HLT_TripleMu_5_3_3_Mass3p8to60_DZ(events)
            #     & ~events.HLT.TripleMu_10_5_5_DZ
            #     & ~events.HLT.TripleMu_12_10_5,
            #     5.291012014 / 59.827826535,
            #     0,
            # )
            # lumi_factors = np.where(
            #     (lumi_factors_1 + lumi_factors_2) > 0,
            #     lumi_factors_1 + lumi_factors_2,
            #     lumi_factors,
            # )
        return lumi_factors

    def get_weights(
        self, events, do_vars: bool = False, apply_lumi_factors: bool = False
    ):
        weights = Weights(len(events))
        if not self.isMC or len(events) == 0:
            return weights

        # Generator weights
        weights.add("genWeight", events.genWeight)

        # Lumi weights
        if apply_lumi_factors:
            lumi_factors = self.get_lumi_factors(events)
            # print(f"lumi_factors: {lumi_factors}")
            # print(
            #     f"lumi_factors stats: {np.unique(lumi_factors)}, {np.histogram(lumi_factors, len(np.unique(lumi_factors)))}"
            # )
            weights.add("lumiWeight", lumi_factors)

        # Pileup weights
        weights.add(
            "PUReweight",
            weight=systematics_utils.pileup_weight(events, self.era),
            weightUp=systematics_utils.pileup_weight(events, self.era, syst="up"),
            weightDown=systematics_utils.pileup_weight(events, self.era, syst="down"),
        )

        # L1 prefire weights
        # Reference: https://twiki.cern.ch/twiki/bin/view/CMS/L1PrefiringWeightRecipe
        if self.era in ["2016", "2016APV", "2017", "2018"]:
            weights.add(
                "L1PreFire",
                weight=events.L1PreFiringWeight.Nom,
                weightUp=events.L1PreFiringWeight.Up,
                weightDown=events.L1PreFiringWeight.Dn,
            )

        # Trigger scale factors
        trig_sf_nom, trig_sf_up, trig_sf_down = trigger_sf_utils.trigger_scale_factors(
            events, self.era
        )
        weights.add(
            "TrigSF", weight=trig_sf_nom, weightUp=trig_sf_up, weightDown=trig_sf_down
        )

        # Some of these systematics are CPU intensive, so only compute them when needed
        if do_vars:
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

            # NOTE: This way to calculate PDF variations is deprecated and wrong
            # # Matrix element PDF and scale weights
            # pdf_vars_up, pdf_vars_down = systematics_utils.get_pdf_variations(events)
            # weights.add(
            #     "LHEPdf",
            #     weight=np.ones(len(events)),
            #     weightUp=pdf_vars_up,
            #     weightDown=pdf_vars_down,
            # )

            muRDown, muFDown, muFUp, muRUp = systematics_utils.get_scale_variations(
                events
            )
            weights.add(
                "LHEScaleMuR",
                weight=np.ones(len(events)),
                weightUp=muRUp,
                weightDown=muRDown,
            )
            weights.add(
                "LHEScaleMuF",
                weight=np.ones(len(events)),
                weightUp=muFUp,
                weightDown=muFDown,
            )

        return weights

    def find_dimuon_pairs(self, muons):
        """
        Find recursively all possible pairs of OS muons, starting from the closest in dR.
        Implements a greedy matching algorithm to find the pairs.
        """
        muons1 = muons[muons.charge == 1]
        muons2 = muons[muons.charge == -1]
        muons1_idx = ak.local_index(muons)[muons.charge == 1]
        muons2_idx = ak.local_index(muons)[muons.charge == -1]

        muon_pairs = ak.unzip(ak.cartesian([muons1, muons2], nested=False))
        muon_pairs_idx = ak.unzip(ak.cartesian([muons1_idx, muons2_idx], nested=False))
        muon_pairs_0, muon_pairs_1 = muon_pairs  # type: ignore[index]
        muon_pairs_0_idx, muon_pairs_1_idx = muon_pairs_idx  # type: ignore[index]

        found_0, found_1, idx_0, idx_1 = [], [], [], []

        while ak.any(ak.num(muon_pairs_0) > 0):
            delta_r = muon_pairs_0.delta_r(muon_pairs_1)
            argmin_delta_r = ak.argmin(delta_r, axis=-1, keepdims=True)

            temp_0 = muon_pairs_0[argmin_delta_r]
            temp_1 = muon_pairs_1[argmin_delta_r]
            temp_0_idx = muon_pairs_0_idx[argmin_delta_r]
            temp_1_idx = muon_pairs_1_idx[argmin_delta_r]

            found_0.append(temp_0)
            found_1.append(temp_1)
            idx_0.append(temp_0_idx)
            idx_1.append(temp_1_idx)

            matched_0 = ak.firsts(temp_0_idx)
            matched_1 = ak.firsts(temp_1_idx)

            remove_mask = (muon_pairs_0_idx != matched_0) & (
                muon_pairs_1_idx != matched_1
            )
            muon_pairs_0 = muon_pairs_0[remove_mask]
            muon_pairs_1 = muon_pairs_1[remove_mask]
            muon_pairs_0_idx = muon_pairs_0_idx[remove_mask]
            muon_pairs_1_idx = muon_pairs_1_idx[remove_mask]

            if ak.all(ak.num(muon_pairs_0) == 0):
                break

        # Combine results
        if found_0:
            found_pairs_0 = ak.concatenate(found_0, axis=-1)
            found_pairs_1 = ak.concatenate(found_1, axis=-1)
        else:
            # Handle case with no OS pairs anywhere
            # Type preservation
            found_pairs_0 = muons[[]]
            found_pairs_1 = muons[[]]

        # Ensure the same number of events as input (important!)
        found_pairs_0 = ak.fill_none(found_pairs_0, [], axis=0)
        found_pairs_1 = ak.fill_none(found_pairs_1, [], axis=0)

        found_pairs_0 = found_pairs_0[~ak.is_none(found_pairs_0, axis=-1)]
        found_pairs_1 = found_pairs_1[~ak.is_none(found_pairs_1, axis=-1)]

        return found_pairs_0, found_pairs_1

    # def find_Z_candidates(self, events, muons):
    #     """
    #     Find the Z candidates by forming all possible pairs of OS muons
    #     and selecting the one closest to the Z mass.
    #     """
    #     # Make sure there are at least two muons with opposite charge
    #     muons_idx = ak.local_index(muons)
    #     muons1 = muons[muons.charge == 1]
    #     muons2 = muons[muons.charge == -1]
    #     muons1_idx = muons_idx[muons.charge == 1]
    #     muons2_idx = muons_idx[muons.charge == -1]
    #     enough_muons = (ak.num(muons1) > 0) & (ak.num(muons2) > 0)
    #     muons1 = muons1[enough_muons]
    #     muons2 = muons2[enough_muons]
    #     muons1_idx = muons1_idx[enough_muons]
    #     muons2_idx = muons2_idx[enough_muons]
    #     muons = muons[enough_muons]
    #     events = events[enough_muons]

    #     # Create all possible pairs of OS muons
    #     muon_pairs = ak.unzip(ak.cartesian([muons1, muons2]))
    #     muon_pairs_idx = ak.unzip(ak.cartesian([muons1_idx, muons2_idx]))

    #     # Find the pair closest to the Z mass
    #     Z_cands = muon_pairs[0] + muon_pairs[1]  # type: ignore[attr-defined]
    #     closest_to_peak = ak.argmin(abs(Z_cands.mass - Z_MASS), axis=1)
    #     Z_cands = ak.firsts(Z_cands[ak.singletons(closest_to_peak)])
    #     muon_indices = ak.concatenate(
    #         [
    #             muon_pairs_idx[0][ak.singletons(closest_to_peak)],
    #             muon_pairs_idx[1][ak.singletons(closest_to_peak)],  # type: ignore[index]
    #         ],
    #         axis=-1,
    #     )

    #     return events, muons, Z_cands, muon_indices

    def find_Z_candidates(self, events, muons, *arrays, apply_dR_cut: bool = False):
        """
        Find the Z candidates by forming all possible pairs of OS muons
        and selecting the one closest to the Z mass.
        """
        # Make sure there are at least two muons with opposite charge
        muons_idx = ak.local_index(muons)
        muons1 = muons[muons.charge == 1]
        muons2 = muons[muons.charge == -1]
        muons1_idx = muons_idx[muons.charge == 1]
        muons2_idx = muons_idx[muons.charge == -1]
        enough_muons = (ak.num(muons1) > 0) & (ak.num(muons2) > 0)
        muons1 = muons1[enough_muons]
        muons2 = muons2[enough_muons]
        muons1_idx = muons1_idx[enough_muons]
        muons2_idx = muons2_idx[enough_muons]
        muons = muons[enough_muons]
        events = events[enough_muons]
        arrays = [arr[enough_muons] for arr in arrays]

        # Create all possible pairs of OS muons
        muon_pairs = ak.unzip(ak.cartesian([muons1, muons2]))
        muon_pairs_idx = ak.unzip(ak.cartesian([muons1_idx, muons2_idx]))

        # Place a dR cut on the muon pairs
        if apply_dR_cut:
            dR_cut = muon_pairs[0].delta_r(muon_pairs[1]) > 0.3  # type: ignore[index]
            muon_pairs_0 = muon_pairs[0][dR_cut]  # type: ignore[index]
            muon_pairs_1 = muon_pairs[1][dR_cut]  # type: ignore[index]
            enough_muons = ak.num(muon_pairs[0]) > 0
            muon_pairs_0 = muon_pairs_0[enough_muons]  # type: ignore[index]
            muon_pairs_1 = muon_pairs_1[enough_muons]  # type: ignore[index]
            muon_pairs_idx_0 = muon_pairs_idx[0][enough_muons]  # type: ignore[index]
            muon_pairs_idx_1 = muon_pairs_idx[1][enough_muons]  # type: ignore[index]
            muons = muons[enough_muons]
            events = events[enough_muons]
            arrays = [arr[enough_muons] for arr in arrays]
        else:
            muon_pairs_0 = muon_pairs[0]  # type: ignore[index]
            muon_pairs_1 = muon_pairs[1]  # type: ignore[index]
            muon_pairs_idx_0 = muon_pairs_idx[0]  # type: ignore[index]
            muon_pairs_idx_1 = muon_pairs_idx[1]  # type: ignore[index]

        # Find the pair closest to the Z mass
        Z_cands = muon_pairs_0 + muon_pairs_1  # type: ignore[index]
        closest_to_peak = ak.argmin(abs(Z_cands.mass - Z_MASS), axis=1)
        Z_cands = ak.firsts(Z_cands[ak.singletons(closest_to_peak)])
        muon_indices = ak.concatenate(
            [
                muon_pairs_idx_0[ak.singletons(closest_to_peak)],
                muon_pairs_idx_1[ak.singletons(closest_to_peak)],  # type: ignore[index]
            ],
            axis=-1,
        )

        return events, muons, Z_cands, muon_indices, *arrays

    def muon_filter(self, events):
        """
        Filter events after the TripleMu trigger.
        Cleans muons and electrons.
        Requires at least nMuons with mediumId, pt, dz, and eta cuts.
        """
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
        select_by_muons_low = ak.num(muons, axis=-1) > 2
        events = events[select_by_muons_low]

        return events

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

    def dimuon_mass_range_mask(self, muons, low_range=2.7, high_range=14.0):
        # Remove events with at least on dimuon in the mass range
        muons1 = muons[muons.charge == 1]
        muons2 = muons[muons.charge == -1]
        enough_muons = (ak.num(muons1) > 0) & (ak.num(muons2) > 0)
        muons1 = ak.mask(muons1, enough_muons)
        muons2 = ak.mask(muons2, enough_muons)
        muons = ak.mask(muons, enough_muons)
        muon_pairs = ak.unzip(ak.cartesian([muons1, muons2]))
        muon_pairs_0 = muon_pairs[0]
        muon_pairs_1 = muon_pairs[1]  # type: ignore[index]
        dimuon_masses = (muon_pairs_0 + muon_pairs_1).mass
        mass_mask = (
            ak.sum((dimuon_masses > low_range) & (dimuon_masses < high_range), axis=-1)
            == 0
        )
        return ak.fill_none(mass_mask, False)
