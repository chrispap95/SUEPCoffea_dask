import awkward as ak
import hist
import vector  # type: ignore[import]
from coffea import processor

import workflows.SUEP_common as SUEP_common

# Importing CMS corrections
from workflows.CMS_corrections import golden_json_utils, muon_sf_utils

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

    def apply_VR(self, events):
        """
        Apply the VR selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 1], muons[ak.num(muons) > 1]

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 5)  # I suppose this is to reduce signal contamination
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Form every possible opposite-sign dimuon pair. The shared helper uses
        # greedy one-to-one matching, which would omit pairs that share a muon.
        positive_muons = muons[muons.charge == 1]
        negative_muons = muons[muons.charge == -1]
        muon_indices = ak.local_index(muons)
        positive_indices = muon_indices[muons.charge == 1]
        negative_indices = muon_indices[muons.charge == -1]
        muon_pairs = ak.cartesian(
            {"positive": positive_muons, "negative": negative_muons},
            axis=1,
            nested=False,
        )
        muon_pair_indices = ak.cartesian(
            {"positive": positive_indices, "negative": negative_indices},
            axis=1,
            nested=False,
        )
        muon_pairs_0 = muon_pairs["positive"]
        muon_pairs_1 = muon_pairs["negative"]
        muon_pairs_idx_0 = muon_pair_indices["positive"]
        muon_pairs_idx_1 = muon_pair_indices["negative"]
        if len(muon_pairs_0):
            dimuon_mass = (muon_pairs_0 + muon_pairs_1).mass

            # J/psi resonance window
            jpsi_mask = (dimuon_mass > 2.7) & (dimuon_mass < 3.5)

            # Keep only events with exactly one J/psi among all OS pairs.
            n_jpsi = ak.sum(jpsi_mask, axis=-1)
            has_jpsi = n_jpsi == 1
            events = events[has_jpsi]
            muons = muons[has_jpsi]
            muon_pairs_idx_0 = muon_pairs_idx_0[has_jpsi]
            muon_pairs_idx_1 = muon_pairs_idx_1[has_jpsi]
            jpsi_mask = jpsi_mask[has_jpsi]

            # Remove the two muons forming the unique J/psi candidate.
            jpsi_idx_0 = ak.firsts(muon_pairs_idx_0[jpsi_mask])
            jpsi_idx_1 = ak.firsts(muon_pairs_idx_1[jpsi_mask])
            muon_indices = ak.local_index(muons)
            muons = muons[(muon_indices != jpsi_idx_0) & (muon_indices != jpsi_idx_1)]
        else:
            # No OS dimuon pairs in the batch -> no J/psi -> drop all events
            events = events[:0]
            muons = muons[:0]

        # Form the loose and tight VRs after removing the J/psi muons. The
        # unfiltered ``events`` collection is returned separately so its nJet
        # distribution is not conditioned on either additional-muon selection.
        muons_VR_loose = muons[
            ((abs(muons.dxy) > 0.01) | (abs(muons.dz) > 0.01))
            & (muons.miniPFRelIso_all > 0.2)
        ]
        has_loose_muon = ak.num(muons_VR_loose, axis=-1) > 0
        events_VR_loose = events[has_loose_muon]
        muons_VR_loose = muons_VR_loose[has_loose_muon]

        # # Cut on the max OS dimuon mass
        # muons1 = muons_VR_loose[muons_VR_loose.charge == 1]
        # muons2 = muons_VR_loose[muons_VR_loose.charge == -1]
        # enough_muons = (ak.num(muons1) > 0) & (ak.num(muons2) > 0)
        # muons1 = muons1[enough_muons]
        # muons2 = muons2[enough_muons]
        # muon_pairs = ak.unzip(ak.cartesian([muons1, muons2]))
        # os_dimuons = muon_pairs[0] + muon_pairs[1]  # type: ignore[index]
        # events_VR_loose = events_VR_loose[ak.max(os_dimuons.mass, axis=-1) > 20]  # type: ignore[op_type]
        # muons_VR_loose = muons_VR_loose[ak.max(os_dimuons.mass, axis=-1) > 20]  # type: ignore[op_type]

        muons_VR_tight = muons[
            ((abs(muons.dxy) > 0.02) | (abs(muons.dz) > 0.02))
            & (muons.miniPFRelIso_all > 0.4)
        ]
        has_tight_muon = ak.num(muons_VR_tight, axis=-1) > 0
        events_VR_tight = events[has_tight_muon]
        muons_VR_tight = muons_VR_tight[has_tight_muon]

        # # Cut on the max OS dimuon mass
        # muons1 = muons_VR_tight[muons_VR_tight.charge == 1]
        # muons2 = muons_VR_tight[muons_VR_tight.charge == -1]
        # enough_muons = (ak.num(muons1) > 0) & (ak.num(muons2) > 0)
        # muons1 = muons1[enough_muons]
        # muons2 = muons2[enough_muons]
        # muon_pairs = ak.unzip(ak.cartesian([muons1, muons2]))
        # os_dimuons = muon_pairs[0] + muon_pairs[1]  # type: ignore[index]
        # events_VR_tight = events_VR_tight[ak.max(os_dimuons.mass, axis=-1) > 20]  # type: ignore[op_type]
        # muons_VR_tight = muons_VR_tight[ak.max(os_dimuons.mass, axis=-1) > 20]  # type: ignore[op_type]

        return (
            events,
            events_VR_tight,
            events_VR_loose,
            muons_VR_tight,
            muons_VR_loose,
        )

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        if len(events) == 0:
            return

        (
            events_JPsi,
            events_VR_tight,
            events_VR_loose,
            muons_VR_tight,
            muons_VR_loose,
        ) = self.apply_VR(events)

        # Fill nJet for all selected J/psi events before requiring an additional
        # post-J/psi muon to pass either VR definition. This population is
        # unconditioned on the additional loose/tight muon and is used as input
        # to the muon-multiplicity toy.
        if len(events_JPsi) > 0:
            weights_JPsi = self.get_weights(
                events_JPsi, do_vars=True, apply_lumi_factors=True
            )
            jets_JPsi = events_JPsi.Jet
            jets_JPsi = jets_JPsi[(jets_JPsi.pt > 30) & (abs(jets_JPsi.eta) < 2.4)]
            nJet_JPsi = ak.num(jets_JPsi.pt, axis=-1)
            nJet_JPsi = ak.where(nJet_JPsi > 9, 9, nJet_JPsi)
            output[dataset]["histograms"]["nJet_preMuon"].fill(
                nJet_JPsi,
                weight=weights_JPsi.weight(),
            )

        if len(events_VR_tight) > 0:
            weights_VR_tight = self.get_weights(
                events_VR_tight, do_vars=True, apply_lumi_factors=True
            )
            if self.isMC:
                weights_VR_tight.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_VR_tight, era=self.era, region="VR_tight", syst=""
                        ),
                        axis=-1,
                    ),
                )
            nMuon_VR_tight = ak.num(muons_VR_tight, axis=-1)
            nMuon_VR_tight = ak.where(nMuon_VR_tight > 7, 7, nMuon_VR_tight)
            muon_pairs = ak.combinations(muons_VR_tight, 2, fields=["muon_0", "muon_1"])
            dimuon_dR_VR_tight = muon_pairs.muon_0.delta_r(muon_pairs.muon_1)
            os_pairs = muon_pairs[
                muon_pairs.muon_0.charge * muon_pairs.muon_1.charge < 0
            ]
            dimuon_mass_VR_tight = (os_pairs.muon_0 + os_pairs.muon_1).mass
            jets_VR_tight = events_VR_tight.Jet
            jets_VR_tight = jets_VR_tight[
                (jets_VR_tight.pt > 15) & (abs(jets_VR_tight.eta) < 2.4)
            ]
            nJet_VR_tight = ak.num(jets_VR_tight.pt, axis=-1)
            output[dataset]["histograms"]["VR_tight"].fill(
                nMuon_VR_tight,
                weight=weights_VR_tight.weight(),
            )
            output[dataset]["histograms"]["VR_tight_mass"].fill(
                ak.flatten(dimuon_mass_VR_tight),
                ak.flatten(
                    ak.broadcast_arrays(nMuon_VR_tight, dimuon_mass_VR_tight)[0]
                ),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        weights_VR_tight.weight(), dimuon_mass_VR_tight
                    )[0]
                ),
            )
            output[dataset]["histograms"]["VR_tight_pt"].fill(
                ak.flatten(muons_VR_tight.pt),
                ak.flatten(ak.broadcast_arrays(nMuon_VR_tight, muons_VR_tight.pt)[0]),
                weight=ak.flatten(
                    ak.broadcast_arrays(weights_VR_tight.weight(), muons_VR_tight.pt)[0]
                ),
            )
            output[dataset]["histograms"]["VR_tight_nJet"].fill(
                ak.where(nJet_VR_tight > 9, 9, nJet_VR_tight),
                nMuon_VR_tight,
                weight=weights_VR_tight.weight(),
            )
            output[dataset]["histograms"]["VR_tight_dR"].fill(
                ak.flatten(dimuon_dR_VR_tight),
                ak.flatten(ak.broadcast_arrays(nMuon_VR_tight, dimuon_dR_VR_tight)[0]),
                weight=ak.flatten(
                    ak.broadcast_arrays(weights_VR_tight.weight(), dimuon_dR_VR_tight)[
                        0
                    ]
                ),
            )

        if len(events_VR_loose) > 0:
            weights_VR_loose = self.get_weights(
                events_VR_loose, do_vars=True, apply_lumi_factors=True
            )
            if self.isMC:
                weights_VR_loose.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_VR_loose, era=self.era, region="VR_loose", syst=""
                        ),
                        axis=-1,
                    ),
                )
            nMuon_VR_loose = ak.num(muons_VR_loose, axis=-1)
            nMuon_VR_loose = ak.where(nMuon_VR_loose > 7, 7, nMuon_VR_loose)
            muon_pairs = ak.combinations(muons_VR_loose, 2, fields=["muon_0", "muon_1"])
            dimuon_dR_VR_loose = muon_pairs.muon_0.delta_r(muon_pairs.muon_1)
            os_pairs = muon_pairs[
                muon_pairs.muon_0.charge * muon_pairs.muon_1.charge < 0
            ]
            dimuon_mass_VR_loose = (os_pairs.muon_0 + os_pairs.muon_1).mass
            jets_VR_loose = events_VR_loose.Jet
            jets_VR_loose = jets_VR_loose[
                (jets_VR_loose.pt > 30) & (abs(jets_VR_loose.eta) < 2.4)
            ]
            nJet_VR_loose = ak.num(jets_VR_loose.pt, axis=-1)
            output[dataset]["histograms"]["VR_loose"].fill(
                nMuon_VR_loose,
                weight=weights_VR_loose.weight(),
            )
            output[dataset]["histograms"]["VR_loose_mass"].fill(
                ak.flatten(dimuon_mass_VR_loose),
                ak.flatten(
                    ak.broadcast_arrays(nMuon_VR_loose, dimuon_mass_VR_loose)[0]
                ),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        weights_VR_loose.weight(), dimuon_mass_VR_loose
                    )[0]
                ),
            )
            output[dataset]["histograms"]["VR_loose_pt"].fill(
                ak.flatten(muons_VR_loose.pt),
                ak.flatten(ak.broadcast_arrays(nMuon_VR_loose, muons_VR_loose.pt)[0]),
                weight=ak.flatten(
                    ak.broadcast_arrays(weights_VR_loose.weight(), muons_VR_loose.pt)[0]
                ),
            )
            output[dataset]["histograms"]["VR_loose_nJet"].fill(
                ak.where(nJet_VR_loose > 9, 9, nJet_VR_loose),
                nMuon_VR_loose,
                weight=weights_VR_loose.weight(),
            )
            output[dataset]["histograms"]["VR_loose_dR"].fill(
                ak.flatten(dimuon_dR_VR_loose),
                ak.flatten(ak.broadcast_arrays(nMuon_VR_loose, dimuon_dR_VR_loose)[0]),
                weight=ak.flatten(
                    ak.broadcast_arrays(weights_VR_loose.weight(), dimuon_dR_VR_loose)[
                        0
                    ]
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
        trigger_plateau = self.apply_trigger_plateau(events, pt3_threshold=4)
        events = events[trigger_plateau]

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
            "VR_tight": hist.Hist.new.Regular(
                7, 1, 8, name="nMuon", label="nMuon"
            ).Weight(),
            "VR_loose": hist.Hist.new.Regular(
                7, 1, 8, name="nMuon", label="nMuon"
            ).Weight(),
            "VR_loose_mass": hist.Hist.new.Regular(
                50,
                0.1,
                100,
                name="dimuon_mass",
                label="dimuon_mass",
                transform=hist.axis.transform.log,
            )
            .Regular(7, 1, 8, name="nMuon", label="nMuon")
            .Weight(),
            "VR_tight_mass": hist.Hist.new.Regular(
                50,
                0.1,
                100,
                name="dimuon_mass",
                label="dimuon_mass",
                transform=hist.axis.transform.log,
            )
            .Regular(7, 1, 8, name="nMuon", label="nMuon")
            .Weight(),
            "VR_loose_pt": hist.Hist.new.Regular(
                50,
                1,
                100,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(7, 1, 8, name="nMuon", label="nMuon")
            .Weight(),
            "VR_tight_pt": hist.Hist.new.Regular(
                50,
                1,
                100,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            )
            .Regular(7, 1, 8, name="nMuon", label="nMuon")
            .Weight(),
            "nJet_preMuon": hist.Hist.new.Regular(
                10,
                0,
                10,
                name="nJet",
                label="nJet",
            ).Weight(),
            "VR_loose_nJet": hist.Hist.new.Regular(
                10,
                0,
                10,
                name="nJet",
                label="nJet",
            )
            .Regular(7, 1, 8, name="nMuon", label="nMuon")
            .Weight(),
            "VR_tight_nJet": hist.Hist.new.Regular(
                10,
                0,
                10,
                name="nJet",
                label="nJet",
            )
            .Regular(7, 1, 8, name="nMuon", label="nMuon")
            .Weight(),
            "VR_loose_dR": hist.Hist.new.Regular(
                200,
                0,
                6,
                name="dimuon_dR",
                label="dimuon_dR",
            )
            .Regular(7, 1, 8, name="nMuon", label="nMuon")
            .Weight(),
            "VR_tight_dR": hist.Hist.new.Regular(
                200,
                0,
                6,
                name="dimuon_dR",
                label="dimuon_dR",
            )
            .Regular(7, 1, 8, name="nMuon", label="nMuon")
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
