import awkward as ak
import numpy as np
from coffea import processor
from coffea.analysis_tools import Weights

from workflows.CMS_corrections import muon_sf_utils, systematics_utils

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

    def get_weights(self, events, do_vars: bool = False):
        weights = Weights(len(events))
        if not self.isMC or len(events) == 0:
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
        # Reference: https://twiki.cern.ch/twiki/bin/view/CMS/L1PrefiringWeightRecipe
        weights.add(
            "L1PreFire",
            weight=events.L1PreFiringWeight.Nom,
            weightUp=events.L1PreFiringWeight.Up,
            weightDown=events.L1PreFiringWeight.Dn,
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

            # Matrix element PDF and scale weights
            pdf_vars_up, pdf_vars_down = systematics_utils.get_pdf_variations(events)
            weights.add(
                "LHEPdf",
                weight=np.ones(len(events)),
                weightUp=pdf_vars_up,
                weightDown=pdf_vars_down,
            )
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

    def find_Z_candidates(self, events, muons):
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

        # Create all possible pairs of OS muons
        muon_pairs = ak.unzip(ak.cartesian([muons1, muons2]))
        muon_pairs_idx = ak.unzip(ak.cartesian([muons1_idx, muons2_idx]))

        # Find the pair closest to the Z mass
        Z_cands = muon_pairs[0] + muon_pairs[1]  # type: ignore[attr-defined]
        closest_to_peak = ak.argmin(abs(Z_cands.mass - Z_MASS), axis=1)
        Z_cands = ak.firsts(Z_cands[ak.singletons(closest_to_peak)])
        muon_indices = ak.concatenate(
            [
                muon_pairs_idx[0][ak.singletons(closest_to_peak)],
                muon_pairs_idx[1][ak.singletons(closest_to_peak)],  # type: ignore[index]
            ],
            axis=-1,
        )

        return events, muons, Z_cands, muon_indices

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
