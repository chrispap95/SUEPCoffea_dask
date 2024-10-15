"""
SUEP_coffea.py
Coffea producer for SUEP analysis. Uses fastjet package to recluster large jets:
https://github.com/scikit-hep/fastjet
Chad Freer and Luca Lavezzo, 2021
"""
from typing import Optional

import awkward as ak
import hist
import numpy as np
import pandas as pd
import vector
from coffea import processor

# Importing SUEP specific functions
import workflows.SUEP_utils as SUEP_utils

# Importing CMS corrections
from workflows.CMS_corrections.golden_jsons_utils import applyGoldenJSON
from workflows.CMS_corrections.jetmet_utils import apply_jecs
from workflows.CMS_corrections.PartonShower_utils import GetPSWeights
from workflows.CMS_corrections.Prefire_utils import GetPrefireWeights
from workflows.pandas_accumulator import pandas_accumulator

# Set vector behavior
vector.register_awkward()


class SUEP_cluster(processor.ProcessorABC):
    def __init__(
        self,
        isMC: int,
        era: int,
        sample: str,
        do_syst: bool,
        syst_var: str,
        weight_syst: bool,
        flag: bool,
        output_location: Optional[str],
        accum: Optional[bool] = None,
        trigger: Optional[str] = None,
        debug: Optional[bool] = None,
    ) -> None:
        self._flag = flag
        self.output_location = output_location
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.era = int(era)
        self.isMC = bool(isMC)
        self.sample = sample
        self.syst_var, self.syst_suffix = (
            (syst_var, f"_sys_{syst_var}") if do_syst and syst_var else ("", "")
        )
        self.weight_syst = weight_syst
        self.prefixes = {"SUEP": "SUEP"}
        self.doOF = False
        self.accum = accum
        self.trigger = trigger
        self.debug = debug

    def jet_awkward(self, Jets):
        """
        Create awkward array of jets. Applies basic selections.
        Returns: awkward array of dimensions (events x jets x 4 momentum)
        """
        Jets_awk = ak.zip(
            {
                "pt": Jets.pt,
                "eta": Jets.eta,
                "phi": Jets.phi,
                "mass": Jets.mass,
                "btagDeepB": Jets.btagDeepB,
            }
        )
        jet_awk_Cut = (Jets.pt > 30) & (abs(Jets.eta) < 2.4)
        Jets_correct = Jets_awk[jet_awk_Cut]
        return Jets_correct

    def eventSelection(self, events):
        """
        Applies trigger, returns events.
        Default is PFHT triggers. Can use selection variable for customization.
        """
        # NOTE: Might be a good idea to make this a 'match-case' statement
        # once we can move to Python 3.10 for good.
        if self.trigger == "TripleMu":
            if self.era == 2016:
                trigger = events.HLT.TripleMu_5_3_3 == 1
            elif self.era == 2017:
                trigger = events.HLT.TripleMu_5_3_3_Mass3p8to60_DZ == 1
            elif self.era == 2018:
                trigger = events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1
            else:
                raise ValueError("Invalid era")
        events = events[trigger]
        return events

    def muon_filter(self, events, nMuons=None, pt_limit=None, avpt_cut=None):
        """
        Filter events after the TripleMu trigger.
        Cleans muons and electrons.
        Requires at least nMuons with mediumId, pt, dxy, dz, and eta cuts.
        """
        muons = events.Muon
        electrons = events.Electron
        clean_muons = (
            (events.Muon.mediumId)
            & (events.Muon.pt > 3)
            & (abs(events.Muon.dxy) <= 0.02)
            & (abs(events.Muon.dz) <= 0.1)
            & (abs(events.Muon.eta) < 2.4)
        )
        if pt_limit is not None:
            clean_muons = clean_muons & (events.Muon.pt < pt_limit)
        clean_electrons = (
            (events.Electron.mvaFall17V2noIso_WPL)
            & (events.Electron.pt > 3)
            & (
                abs(events.Electron.dxy)
                < 0.05 + 0.05 * (abs(events.Electron.eta) > 1.479)
            )
            & (
                abs(events.Electron.dz)
                < 0.10 + 0.10 * (abs(events.Electron.eta) > 1.479)
            )
            & ((abs(events.Electron.eta) < 1.444) | (abs(events.Electron.eta) > 1.566))
            & (abs(events.Electron.eta) < 2.5)
        )
        muons = muons[clean_muons]
        electrons = electrons[clean_electrons]
        if nMuons is not None:
            select_by_muons = ak.num(muons, axis=-1) >= nMuons
            events = events[select_by_muons]
            muons = muons[select_by_muons]
            electrons = electrons[select_by_muons]
        if avpt_cut is not None:
            avpt = ak.mean(muons.pt, axis=-1)
            events = events[avpt < avpt_cut]
            muons = muons[avpt < avpt_cut]
            electrons = electrons[avpt < avpt_cut]
        return events, electrons, muons

    def getGenTracks(self, events):
        genParts = events.GenPart
        genParts = ak.zip(
            {
                "pt": genParts.pt,
                "eta": genParts.eta,
                "phi": genParts.phi,
                "mass": genParts.mass,
                "pdgID": genParts.pdgId,
            },
            with_name="Momentum4D",
        )
        return genParts

    def storeEventVars(
        self,
        events,
        output,
        electrons,
        muons,
        out_label="",
    ):
        dataset = events.metadata["dataset"]

        # muon inter-isolation
        muonsCollection = ak.zip(
            {
                "pt": muons.pt,
                "eta": muons.eta,
                "phi": muons.phi,
                "mass": muons.mass,
                "charge": muons.pdgId / (-13),
            },
            with_name="Momentum4D",
        )
        electronsCollection = ak.zip(
            {
                "pt": electrons.pt,
                "eta": electrons.eta,
                "phi": electrons.phi,
                "mass": electrons.mass,
                "charge": electrons.pdgId / (-11),
            },
            with_name="Momentum4D",
        )

        # select out ak4jets
        ak4jets = self.jet_awkward(events.Jet)

        # work on JECs and systematics
        prefix = ""
        if self.accum:
            if "casa" in self.accum:
                prefix = "dask-worker-space/"
        jets_c = apply_jecs(
            isMC=self.isMC,
            Sample=self.sample,
            era=self.era,
            events=events,
            prefix=prefix,
        )
        jets_jec = self.jet_awkward(jets_c)
        if self.isMC:
            jets_jec_JERUp = self.jet_awkward(jets_c["JER"].up)
            jets_jec_JERDown = self.jet_awkward(jets_c["JER"].down)
            jets_jec_JESUp = self.jet_awkward(jets_c["JES_jes"].up)
            jets_jec_JESDown = self.jet_awkward(jets_c["JES_jes"].down)
        # For data set these all to nominal so we can plot without switching all of the names
        else:
            jets_jec_JERUp = jets_jec
            jets_jec_JERDown = jets_jec
            jets_jec_JESUp = jets_jec
            jets_jec_JESDown = jets_jec

        # save per event variables to a dataframe
        if out_label == "":
            output[dataset]["vars"]["events" + out_label] = events.event.to_list()
            output[dataset]["vars"][
                "luminosityBlock" + out_label
            ] = events.luminosityBlock.to_list()
            output[dataset]["vars"][
                "genWeight" + out_label
            ] = events.genWeight.to_list()
            output[dataset]["vars"]["ht" + out_label] = ak.sum(
                ak4jets.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label] = ak.sum(
                jets_jec.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label + "_JER_up"] = ak.sum(
                jets_jec_JERUp.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label + "_JER_down"] = ak.sum(
                jets_jec_JERDown.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label + "_JES_up"] = ak.sum(
                jets_jec_JESUp.pt, axis=-1
            ).to_list()
            output[dataset]["vars"]["ht_JEC" + out_label + "_JES_down"] = ak.sum(
                jets_jec_JESDown.pt, axis=-1
            ).to_list()

            if self.era == 2016:
                output[dataset]["vars"]["HLT_PFHT900" + out_label] = events.HLT.PFHT900
            else:
                output[dataset]["vars"][
                    "HLT_PFHT1050" + out_label
                ] = events.HLT.PFHT1050
            output[dataset]["vars"]["ngood_ak4jets" + out_label] = ak.num(
                ak4jets
            ).to_list()
            if self.isMC:
                output[dataset]["vars"][
                    "Pileup_nTrueInt" + out_label
                ] = events.Pileup.nTrueInt
                GetPSWeights(events, output[dataset])  # Parton Shower weights
                GetPrefireWeights(self, events, output[dataset])  # Prefire weights
            output[dataset]["vars"]["PV_npvs" + out_label] = events.PV.npvs
            output[dataset]["vars"]["PV_npvsGood" + out_label] = events.PV.npvsGood

        # get gen SUEP kinematics
        SUEP_genMass = len(events) * [0]
        SUEP_genPt = len(events) * [0]
        SUEP_genEta = len(events) * [0]
        SUEP_genPhi = len(events) * [0]

        if self.isMC:
            genParts = self.getGenTracks(events)
            genSUEP = genParts[(abs(genParts.pdgID) == 25)]

            # we need to grab the last SUEP in the chain for each event
            SUEP_genMass = [g[-1].mass if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genPt = [g[-1].pt if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genPhi = [g[-1].phi if len(g) > 0 else 0 for g in genSUEP]
            SUEP_genEta = [g[-1].eta if len(g) > 0 else 0 for g in genSUEP]

        output[dataset]["vars"]["SUEP_genMass" + out_label] = SUEP_genMass
        output[dataset]["vars"]["SUEP_genPt" + out_label] = SUEP_genPt
        output[dataset]["vars"]["SUEP_genEta" + out_label] = SUEP_genEta
        output[dataset]["vars"]["SUEP_genPhi" + out_label] = SUEP_genPhi

        # MET
        output[dataset]["vars"]["MET_pt" + out_label] = events.MET.pt.to_list()
        output[dataset]["vars"]["MET_phi" + out_label] = events.MET.phi.to_list()
        output[dataset]["vars"]["MET_sumEt" + out_label] = events.MET.sumEt.to_list()

        # BTagging
        btag_scores = ak4jets.btagDeepB
        btag_scores = btag_scores[:, :4]
        btag_scores = ak.pad_none(btag_scores, target=4, axis=-1)
        btag_scores = ak.fill_none(btag_scores, value=-2)
        output[dataset]["vars"]["bTag_score_leading" + out_label] = btag_scores[
            :, 0
        ].to_list()
        output[dataset]["vars"]["bTag_score_subleading" + out_label] = btag_scores[
            :, 1
        ].to_list()
        output[dataset]["vars"]["bTag_score_third" + out_label] = btag_scores[
            :, 2
        ].to_list()
        output[dataset]["vars"]["bTag_score_fourth" + out_label] = btag_scores[
            :, 3
        ].to_list()

        # nMuons
        output[dataset]["vars"]["nMuons" + out_label] = ak.num(events.Muon).to_list()
        output[dataset]["vars"]["nMuons_highPurity" + out_label] = ak.sum(
            muons.highPurity, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_isPFcand" + out_label] = ak.sum(
            muons.isPFcand, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_looseId" + out_label] = ak.sum(
            muons.looseId, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_mediumId" + out_label] = ak.sum(
            muons.mediumId, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_tightId" + out_label] = ak.sum(
            muons.tightId, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_triggerIdLoose" + out_label] = ak.sum(
            muons.triggerIdLoose, axis=-1
        ).to_list()
        output[dataset]["vars"]["nMuons_isTracker" + out_label] = ak.sum(
            muons.isTracker, axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_pt_mean" + out_label] = ak.mean(
            muons.pt, axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_dxy_mean" + out_label] = ak.mean(
            abs(muons.dxy), axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_dz_mean" + out_label] = ak.mean(
            abs(muons.dz), axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_ip3d_mean" + out_label] = ak.mean(
            muons.ip3d, axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_pt_leading" + out_label] = muons.pt[
            :, 0
        ].to_list()
        output[dataset]["vars"]["muon_pt_subleading" + out_label] = muons.pt[
            :, 1
        ].to_list()
        output[dataset]["vars"][
            "muon_miniPFRelIso_all_leading" + out_label
        ] = muons.miniPFRelIso_all[:, 0].to_list()
        output[dataset]["vars"][
            "muon_miniPFRelIso_all_subleading" + out_label
        ] = muons.miniPFRelIso_all[:, 1].to_list()
        output[dataset]["vars"]["muon_miniPFRelIso_all_mean" + out_label] = ak.mean(
            muons.miniPFRelIso_all, axis=-1
        ).to_list()
        output[dataset]["vars"]["muon_l_sl_deltaphi" + out_label] = abs(
            ak.materialized(muonsCollection[:, 0]).deltaphi(
                ak.materialized(muonsCollection[:, 1])
            )
        ).to_list()
        output[dataset]["vars"]["muon_l_sl_deltaeta" + out_label] = abs(
            ak.materialized(muonsCollection[:, 0]).deltaeta(
                ak.materialized(muonsCollection[:, 1])
            )
        ).to_list()
        output[dataset]["vars"]["muon_l_sl_deltaR" + out_label] = (
            ak.materialized(muonsCollection[:, 0])
            .deltaR(ak.materialized(muonsCollection[:, 1]))
            .to_list()
        )
        eigs_muons = SUEP_utils.sphericity(muons, 1.0)
        output[dataset]["vars"]["muon_S1" + out_label] = 1.5 * (
            eigs_muons[:, 1] + eigs_muons[:, 0]
        )

        leptons = ak.concatenate([muonsCollection, electronsCollection], axis=-1)
        leading_muons = leptons[:, 0]
        output[dataset]["vars"][
            "muon_interIsolation_0p2" + out_label
        ] = SUEP_utils.inter_isolation(leading_muons, leptons, dR=0.2).to_list()
        output[dataset]["vars"][
            "muon_interIsolation_0p4" + out_label
        ] = SUEP_utils.inter_isolation(leading_muons, leptons, dR=0.4).to_list()
        output[dataset]["vars"][
            "muon_interIsolation_0p8" + out_label
        ] = SUEP_utils.inter_isolation(leading_muons, leptons, dR=0.8).to_list()
        output[dataset]["vars"][
            "muon_interIsolation_1p6" + out_label
        ] = SUEP_utils.inter_isolation(leading_muons, leptons, dR=1.6).to_list()

        # Eta ring variables
        output[dataset]["vars"][
            "nMuons_eta_ring_0p2" + out_label
        ] = SUEP_utils.n_eta_ring(muonsCollection, 0.2).to_list()
        output[dataset]["vars"][
            "nMuons_eta_ring_0p4" + out_label
        ] = SUEP_utils.n_eta_ring(muonsCollection, 0.4).to_list()
        output[dataset]["vars"][
            "nMuons_eta_ring_0p8" + out_label
        ] = SUEP_utils.n_eta_ring(muonsCollection, 0.8).to_list()
        output[dataset]["vars"][
            "nMuons_eta_ring_1p6" + out_label
        ] = SUEP_utils.n_eta_ring(muonsCollection, 1.6).to_list()

    def count_events(
        self, events, nMuons=None, pt_limit=None, interiso_cut=None, avpt_cut=None
    ):
        # first apply the muon filter
        events, electrons, muons = self.muon_filter(
            events, nMuons=nMuons, pt_limit=pt_limit, avpt_cut=avpt_cut
        )
        # then calculate the inter-isolation
        if interiso_cut is not None:
            muonsCollection = ak.zip(
                {
                    "pt": muons.pt,
                    "eta": muons.eta,
                    "phi": muons.phi,
                    "mass": muons.mass,
                    "charge": muons.pdgId / (-13),
                },
                with_name="Momentum4D",
            )
            electronsCollection = ak.zip(
                {
                    "pt": electrons.pt,
                    "eta": electrons.eta,
                    "phi": electrons.phi,
                    "mass": electrons.mass,
                    "charge": electrons.pdgId / (-11),
                },
                with_name="Momentum4D",
            )
            leptons = ak.concatenate([muonsCollection, electronsCollection], axis=-1)
            leading_muons = leptons[:, 0]
            I15 = SUEP_utils.inter_isolation(leading_muons, leptons, dR=1.6)
            events = events[I15 > interiso_cut]
        return events

    def fill_histograms(self, events, muons, output):
        dataset = events.metadata["dataset"]

        # First, fill plots with event level quantities
        output[dataset]["histograms"]["nMuon"].fill(
            ak.num(muons), weight=events.genWeight
        )

        # These arrays need to be broadcasted to the per muon dims from per event dims
        nMuons = ak.flatten(ak.broadcast_arrays(ak.num(muons), muons.pt)[0])
        weights = ak.flatten(ak.broadcast_arrays(events.genWeight, muons.pt)[0])

        output[dataset]["histograms"]["muon_pt_vs_nMuon"].fill(
            ak.flatten(muons.pt), nMuons, weight=weights
        )

        mT_MET = SUEP_utils.transverse_mass(muons, events.MET)
        mT = np.sqrt(muons.E**2 - muons.pz**2)
        output[dataset]["histograms"]["muon_METmt_vs_nMuon"].fill(
            ak.flatten(mT_MET), nMuons, weight=weights
        )
        output[dataset]["histograms"]["muon_METmt_vs_genflavor_vs_nMuon"].fill(
            ak.flatten(mT_MET), ak.flatten(muons.genPartFlav), nMuons, weight=weights
        )
        output[dataset]["histograms"]["muon_mt_vs_nMuon"].fill(
            ak.flatten(mT), nMuons, weight=weights
        )
        output[dataset]["histograms"]["muon_mt_vs_genflavor_vs_nMuon"].fill(
            ak.flatten(mT), ak.flatten(muons.genPartFlav), nMuons, weight=weights
        )
        output[dataset]["histograms"]["muon_pt_vs_genflavor_vs_nMuon"].fill(
            ak.flatten(muons.pt), ak.flatten(muons.genPartFlav), nMuons, weight=weights
        )
        output[dataset]["histograms"][
            "muon_dxy_vs_miniPFRelIso_vs_genflavor_vs_nMuon"
        ].fill(
            ak.flatten(abs(muons.dxy)),
            ak.flatten(muons.miniPFRelIso_all),
            ak.flatten(muons.genPartFlav),
            nMuons,
            weight=weights,
        )
        output[dataset]["histograms"][
            "muon_pt_vs_miniPFRelIso_vs_genflavor_vs_nMuon"
        ].fill(
            ak.flatten(muons.pt),
            ak.flatten(muons.miniPFRelIso_all),
            ak.flatten(muons.genPartFlav),
            nMuons,
            weight=weights,
        )
        output[dataset]["histograms"]["muon_dz_vs_genflavor_vs_nMuon"].fill(
            ak.flatten(abs(muons.dz)),
            ak.flatten(muons.genPartFlav),
            nMuons,
            weight=weights,
        )
        output[dataset]["histograms"]["muon_ip3d_vs_genflavor_vs_nMuon"].fill(
            ak.flatten(muons.ip3d),
            ak.flatten(muons.genPartFlav),
            nMuons,
            weight=weights,
        )

        # Histograms with genParts
        genParts = events.GenPart[muons.genPartIdx]
        # Get pdgId's of last non-mu parents
        last_parents = SUEP_utils.get_last_parents(genParts)
        output[dataset]["histograms"]["muon_pdgId_parentId_genFlav_nMuon"].fill(
            SUEP_utils.discritize_pdg_codes(
                ak.flatten(ak.fill_none(genParts.pdgId, 0))
            ),
            SUEP_utils.discritize_pdg_codes(
                ak.flatten(ak.fill_none(last_parents.pdgId, 0)), extended=True
            ),
            ak.flatten(muons.genPartFlav),
            nMuons,
            weight=weights,
        )

        # nMuon histograms
        output[dataset]["histograms"]["nMuon_keep_genflavor_all_vs_45_vs_145"].fill(
            ak.num(muons),
            ak.num(muons[(muons.genPartFlav == 4) | (muons.genPartFlav == 5)]),
            ak.num(
                muons[
                    (muons.genPartFlav == 1)
                    | (muons.genPartFlav == 4)
                    | (muons.genPartFlav == 5)
                ]
            ),
            weight=events.genWeight,
        )
        # Get muons associated with pi+/- (211), K0_L (130), K+/- (321)
        matched_LLP_free = SUEP_utils.LLP_free_muons(events, muons)
        # Remove some of the unmatched muons probabilistically
        unmatched_LLP_free = SUEP_utils.probabilistic_removal(muons.genPartFlav)
        output[dataset]["histograms"][
            "nMuon_all_vs_nomatchedLLPs_vs_nounmatchedLLPs_vs_noLLPs"
        ].fill(
            ak.num(muons),
            ak.sum(matched_LLP_free, axis=-1),
            ak.sum(unmatched_LLP_free, axis=-1),
            ak.sum(matched_LLP_free & unmatched_LLP_free, axis=-1),
            weight=events.genWeight,
        )

        return

    def analysis(self, events, output, do_syst=False, col_label=""):
        """
        Main analyzer function.

        Parameters
        ----------
        events : NanoEvents
            NanoEvents object
        output : dict
            accumulator object
        do_syst : bool, optional
            whether to do systematics, by default False
        col_label : str, optional
            label for the collection, by default ""
        """

        # get dataset name
        dataset = events.metadata["dataset"]

        # Fill the cutflow columns for all and trigger
        output[dataset]["cutflow"].fill(len(events) * ["all"], weight=events.genWeight)
        output[dataset]["cutflow"].fill(
            ak.sum(events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1) * ["trigger"],
            weight=events[events.HLT.TripleMu_5_3_3_Mass3p8_DZ == 1].genWeight,
        )

        # golden jsons for offline data
        if not self.isMC:
            events = applyGoldenJSON(self, events)

        events = self.eventSelection(events)

        # Calculate various cutflow variables
        events_n4 = self.count_events(events, nMuons=4)
        output[dataset]["cutflow"].fill(
            len(events_n4) * ["nMu>=4"],
            weight=events_n4.genWeight,
        )
        events_n4_i16 = self.count_events(events, nMuons=4, interiso_cut=1.0)
        output[dataset]["cutflow"].fill(
            len(events_n4_i16) * ["nMu>=4 & I16>1"],
            weight=events_n4_i16.genWeight,
        )
        events_n4_pt10 = self.count_events(events, nMuons=4, pt_limit=10)
        output[dataset]["cutflow"].fill(
            len(events_n4_pt10) * ["nMu>=4 & ptMu<10"],
            weight=events_n4_pt10.genWeight,
        )
        events_n4_pt10_i16 = self.count_events(
            events, nMuons=4, pt_limit=10, interiso_cut=1.0
        )
        output[dataset]["cutflow"].fill(
            len(events_n4_pt10_i16) * ["nMu>=4 & ptMu<10 & I16>1"],
            weight=events_n4_pt10_i16.genWeight,
        )
        events_n4_avpt10 = self.count_events(events, nMuons=4, avpt_cut=10)
        output[dataset]["cutflow"].fill(
            len(events_n4_avpt10) * ["nMu>=4 & avpt<10"],
            weight=events_n4_avpt10.genWeight,
        )
        events_n4_avpt10_i16 = self.count_events(
            events, nMuons=4, avpt_cut=10, interiso_cut=1.0
        )
        output[dataset]["cutflow"].fill(
            len(events_n4_avpt10_i16) * ["nMu>=4 & avpt<10 & I16>1"],
            weight=events_n4_avpt10_i16.genWeight,
        )

        events_n6 = self.count_events(events, nMuons=6)
        output[dataset]["cutflow"].fill(
            len(events_n6) * ["nMu>=6"],
            weight=events_n6.genWeight,
        )
        events_n6_i16 = self.count_events(events, nMuons=6, interiso_cut=1.0)
        output[dataset]["cutflow"].fill(
            len(events_n6_i16) * ["nMu>=6 & I16>1"],
            weight=events_n6_i16.genWeight,
        )
        events_n6_pt10 = self.count_events(events, nMuons=6, pt_limit=10)
        output[dataset]["cutflow"].fill(
            len(events_n6_pt10) * ["nMu>=6 & ptMu<10"],
            weight=events_n6_pt10.genWeight,
        )
        events_n6_pt10_i16 = self.count_events(
            events, nMuons=6, pt_limit=10, interiso_cut=1.0
        )
        output[dataset]["cutflow"].fill(
            len(events_n6_pt10_i16) * ["nMu>=6 & ptMu<10 & I16>1"],
            weight=events_n6_pt10_i16.genWeight,
        )
        events_n6_avpt10 = self.count_events(events, nMuons=6, avpt_cut=10)
        output[dataset]["cutflow"].fill(
            len(events_n6_avpt10) * ["nMu>=6 & avpt<10"],
            weight=events_n6_avpt10.genWeight,
        )
        events_n6_avpt10_i16 = self.count_events(
            events, nMuons=6, avpt_cut=10, interiso_cut=1.0
        )
        output[dataset]["cutflow"].fill(
            len(events_n6_avpt10_i16) * ["nMu>=6 & avpt<10 & I16>1"],
            weight=events_n6_avpt10_i16.genWeight,
        )

        # fill the histograms
        events, electrons, muons = self.muon_filter(events)
        self.fill_histograms(events, muons, output)

        # now actually run the muon filter
        events, electrons, muons = self.muon_filter(events, nMuons=4)

        # output empty dataframe if no events pass trigger
        if len(events) == 0:
            if self.debug:
                print("No events passed trigger. Saving empty outputs.")
            return output

        #####################################################################################
        # ---- Event level information
        #####################################################################################

        self.storeEventVars(
            events,
            output,
            electrons,
            muons,
            out_label=col_label,
        )

    def process(self, events):
        dataset = events.metadata["dataset"]
        cutflow = hist.Hist.new.StrCategory(
            [
                "all",
                "trigger",
                "nMu>=4",
                "nMu>=4 & I16>1",
                "nMu>=4 & ptMu<10",
                "nMu>=4 & ptMu<10 & I16>1",
                "nMu>=4 & avpt<10",
                "nMu>=4 & avpt<10 & I16>1",
                "nMu>=6",
                "nMu>=6 & I16>1",
                "nMu>=6 & ptMu<10",
                "nMu>=6 & ptMu<10 & I16>1",
                "nMu>=6 & avpt<10",
                "nMu>=6 & avpt<10 & I16>1",
                "n_ak15>=1 (30 GeV)",
            ],
            name="cutflow",
            label="cutflow",
        ).Weight()
        histograms = {
            # First plots with event-level quantities
            "nMuon": hist.Hist.new.Reg(10, 0, 10, name="nMuon", label="nMuon").Weight(),
            # Muon pt histogram - per muon entry
            # Add an axis for nMuon
            "muon_pt_vs_nMuon": hist.Hist.new.Reg(
                150, 0, 150, name="muon_pt", label="Muon p_{T} [GeV]"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            # Transverse mass histogram between muons and MET
            "muon_METmt_vs_nMuon": hist.Hist.new.Reg(
                150, 0, 150, name="mt", label="m_{T} [GeV]"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_mt_vs_nMuon": hist.Hist.new.Reg(
                150, 0, 150, name="mt", label="m_{T} [GeV]"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            # comparing stuff with gen flavor
            "muon_pt_vs_genflavor_vs_nMuon": hist.Hist.new.Reg(
                150, 0, 150, name="pt", label="p_{T} [GeV]"
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_METmt_vs_genflavor_vs_nMuon": hist.Hist.new.Reg(
                150, 0, 150, name="mt", label="m_{T} [GeV]"
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_mt_vs_genflavor_vs_nMuon": hist.Hist.new.Reg(
                150, 0, 150, name="mt", label="m_{T} [GeV]"
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_dz_vs_genflavor_vs_nMuon": hist.Hist.new.Regular(
                200, 0, 0.1, name="Muon_dz", label="Muon_dz"
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_ip3d_vs_genflavor_vs_nMuon": hist.Hist.new.Regular(
                200, 0, 0.1, name="Muon_ip3d", label="Muon_ip3d"
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_dxy_vs_miniPFRelIso_vs_genflavor_vs_nMuon": hist.Hist.new.Regular(
                50,
                1e-4,
                1e-1,
                name="Muon_dxy",
                label="Muon_dxy",
                transform=hist.axis.transform.log,
            )
            .Regular(
                50,
                1e-3,
                1e3,
                name="Muon_miniPFRelIso_all",
                label="Muon_miniPFRelIso_all",
                transform=hist.axis.transform.log,
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            "muon_pt_vs_miniPFRelIso_vs_genflavor_vs_nMuon": hist.Hist.new.Regular(
                50,
                1,
                1e3,
                name="pt",
                label="p_{T} [GeV]",
                transform=hist.axis.transform.log,
            )
            .Regular(
                50,
                1e-3,
                1e3,
                name="Muon_miniPFRelIso_all",
                label="Muon_miniPFRelIso_all",
                transform=hist.axis.transform.log,
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            # Histograms to check muon parentage
            # For the GenPart_pdgId: 0 -> 22 are matched exactly, 100 and above include everything
            # for the next 100 codes, 1000 are all codes above 1000, -1 is everything else.
            "muon_pdgId_parentId_genFlav_nMuon": hist.Hist.new.IntCategory(
                [
                    -1,
                    0,
                    11,
                    13,
                    15,
                    22,
                    100,
                    200,
                    300,
                    400,
                    500,
                    1000,
                ],
                name="GenPart_pdgId",
                label="GenPart_pdgId",
            )
            .IntCategory(
                [
                    -1,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    11,
                    13,
                    15,
                    21,
                    22,
                    100,
                    200,
                    300,
                    400,
                    500,
                    1000,
                ],
                name="GenPart_parent_pdgId",
                label="GenPart_parent_pdgId",
            )
            .IntCategory(
                [0, 1, 3, 4, 5, 15], name="Muon_genPartFlav", label="Muon_genPartFlav"
            )
            .Reg(10, 0, 10, name="nMuon", label="nMuon")
            .Weight(),
            # Histograms after muon subtraction
            "nMuon_keep_genflavor_all_vs_45_vs_145": hist.Hist.new.Reg(
                10, 0, 10, name="nMuon", label="nMuon"
            )
            .Reg(10, 0, 10, name="nMuon_45", label="nMuon_45")
            .Reg(10, 0, 10, name="nMuon_145", label="nMuon_145")
            .Weight(),
            "nMuon_all_vs_nomatchedLLPs_vs_nounmatchedLLPs_vs_noLLPs": hist.Hist.new.Reg(
                10, 0, 10, name="nMuon", label="nMuon"
            )
            .Reg(10, 0, 10, name="nMuon_nomatchedLLPs", label="nMuon_nomatchedLLPs")
            .Reg(
                10,
                0,
                10,
                name="nMuon_nounmatchedLLPs",
                label="nMuon_nounmatchedLLPs",
            )
            .Reg(10, 0, 10, name="nMuon_noLLPs", label="nMuon_noLLPs")
            .Weight(),
        }
        output = {
            dataset: {
                "cutflow": cutflow,
                "gensumweight": processor.value_accumulator(float, 0),
                "vars": pandas_accumulator(pd.DataFrame()),
                "histograms": histograms,
            },
        }

        # gen weights
        if self.isMC:
            self.gensumweight = ak.sum(events.genWeight)
            output[dataset]["gensumweight"].add(self.gensumweight)

        # run the analysis with the track systematics applied
        if self.isMC and self.do_syst:
            self.analysis(events, output, do_syst=True, col_label="_track_down")

        # run the analysis
        self.analysis(events, output)

        return output

    def postprocess(self, accumulator):
        pass
