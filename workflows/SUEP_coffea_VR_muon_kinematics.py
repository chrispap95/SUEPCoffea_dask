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

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 5)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Make sure we are the trigger plateau
        events, muons = events[ak.num(muons) > 2], muons[ak.num(muons) > 2]

        # Form loose VR & make sure there is at least one muon in the event after the cuts
        muons_VR_loose = muons[(muons.ip3d > 0.01) & (muons.miniPFRelIso_all > 0.2)]
        events_VR_loose = events[ak.num(muons_VR_loose, axis=-1) > 0]
        muons_VR_loose = muons_VR_loose[ak.num(muons_VR_loose, axis=-1) > 0]

        # Cut on the max OS dimuon mass
        muons1 = muons_VR_loose[muons_VR_loose.charge == 1]
        muons2 = muons_VR_loose[muons_VR_loose.charge == -1]
        enough_muons = (ak.num(muons1) > 0) & (ak.num(muons2) > 0)
        muons1 = muons1[enough_muons]
        muons2 = muons2[enough_muons]
        muon_pairs = ak.unzip(ak.cartesian([muons1, muons2]))
        os_dimuons = muon_pairs[0] + muon_pairs[1]  # type: ignore[index]
        events_VR_loose = events_VR_loose[ak.max(os_dimuons.mass, axis=-1) > 20]  # type: ignore[op_type]
        muons_VR_loose = muons_VR_loose[ak.max(os_dimuons.mass, axis=-1) > 20]  # type: ignore[op_type]

        # Form tight VR & make sure there is at least one muon in the event after the cuts
        muons_VR_tight = muons[(muons.ip3d > 0.02) & (muons.miniPFRelIso_all > 0.4)]
        events_VR_tight = events[ak.num(muons_VR_tight, axis=-1) > 0]
        muons_VR_tight = muons_VR_tight[ak.num(muons_VR_tight, axis=-1) > 0]

        # Cut on the max OS dimuon mass
        muons1 = muons_VR_tight[muons_VR_tight.charge == 1]
        muons2 = muons_VR_tight[muons_VR_tight.charge == -1]
        enough_muons = (ak.num(muons1) > 0) & (ak.num(muons2) > 0)
        muons1 = muons1[enough_muons]
        muons2 = muons2[enough_muons]
        muon_pairs = ak.unzip(ak.cartesian([muons1, muons2]))
        os_dimuons = muon_pairs[0] + muon_pairs[1]  # type: ignore[index]
        events_VR_tight = events_VR_tight[ak.max(os_dimuons.mass, axis=-1) > 20]  # type: ignore[op_type]
        muons_VR_tight = muons_VR_tight[ak.max(os_dimuons.mass, axis=-1) > 20]  # type: ignore[op_type]

        return events_VR_tight, events_VR_loose, muons_VR_tight, muons_VR_loose

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        events_VR_tight, events_VR_loose, muons_VR_tight, muons_VR_loose = (
            self.apply_VR(events_)
        )
        if len(events_VR_tight) > 0:
            muon_pairs_0, muon_pairs_1 = self.find_dimuon_pairs(muons_VR_tight)
            dimuon_dr_mask = muon_pairs_0.delta_r(muon_pairs_1) < 0.3
            dimuon_mass = (muon_pairs_0 + muon_pairs_1).mass
            dimuon_mass_mask = ((dimuon_mass > 2.7) & (dimuon_mass < 3.5)) | (
                (dimuon_mass > 8.8) & (dimuon_mass < 11.2)
            )
            events_VR_tight = events_VR_tight[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muons_VR_tight = muons_VR_tight[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muon_pairs_0 = muon_pairs_0[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muon_pairs_1 = muon_pairs_1[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]

            if len(events_VR_tight) > 0:
                weights_VR_tight = self.get_weights(
                    events_VR_tight, do_vars=False, apply_lumi_factors=True
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
                nMuon_VR_tight = ak.where(nMuon_VR_tight > 5, 5, nMuon_VR_tight)
                output[dataset]["histograms"]["VR_tight_muon_pt"].fill(
                    ak.flatten(muons_VR_tight.pt),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_VR_tight, muons_VR_tight.pt)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_tight.weight(), muons_VR_tight.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["VR_tight_muon_eta"].fill(
                    ak.flatten(muons_VR_tight.eta),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_VR_tight, muons_VR_tight.eta)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_tight.weight(), muons_VR_tight.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["VR_tight_muon_eta"].fill(
                    ak.flatten(muons_VR_tight.eta),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_VR_tight, muons_VR_tight.eta)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_tight.weight(), muons_VR_tight.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["VR_tight_muon_phi"].fill(
                    ak.flatten(muons_VR_tight.phi),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_VR_tight, muons_VR_tight.phi)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_tight.weight(), muons_VR_tight.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["VR_tight_muon_iso"].fill(
                    ak.flatten(muons_VR_tight.miniPFRelIso_all),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_VR_tight,
                            muons_VR_tight.miniPFRelIso_all,
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_tight.weight(), muons_VR_tight.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["VR_tight_muon_dxy"].fill(
                    ak.flatten(muons_VR_tight.dxy),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_VR_tight, muons_VR_tight.dxy)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_tight.weight(), muons_VR_tight.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["VR_tight_muon_dz"].fill(
                    ak.flatten(muons_VR_tight.dz),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_VR_tight, muons_VR_tight.dz)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_tight.weight(), muons_VR_tight.pt
                        )[0]
                    ),
                )
                muon_pairs_dr = muon_pairs_0.delta_r(muon_pairs_1)
                output[dataset]["histograms"]["VR_tight_dimuon_dr"].fill(
                    ak.flatten(muon_pairs_dr),
                    ak.flatten(ak.broadcast_arrays(nMuon_VR_tight, muon_pairs_dr)[0]),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_VR_tight.weight(), muon_pairs_dr)[0]
                    ),
                )
                muon_pairs_mass = (muon_pairs_0 + muon_pairs_1).mass
                output[dataset]["histograms"]["VR_tight_dimuon_mass"].fill(
                    ak.flatten(muon_pairs_mass),
                    ak.flatten(ak.broadcast_arrays(nMuon_VR_tight, muon_pairs_mass)[0]),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_VR_tight.weight(), muon_pairs_mass)[
                            0
                        ]
                    ),
                )

        if len(events_VR_loose) > 0:
            muon_pairs_0, muon_pairs_1 = self.find_dimuon_pairs(muons_VR_loose)
            dimuon_dr_mask = muon_pairs_0.delta_r(muon_pairs_1) < 0.3
            dimuon_mass = (muon_pairs_0 + muon_pairs_1).mass
            dimuon_mass_mask = ((dimuon_mass > 2.7) & (dimuon_mass < 3.5)) | (
                (dimuon_mass > 8.8) & (dimuon_mass < 11.2)
            )
            events_VR_loose = events_VR_loose[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muons_VR_loose = muons_VR_loose[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muon_pairs_0 = muon_pairs_0[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            muon_pairs_1 = muon_pairs_1[
                ~ak.any(dimuon_dr_mask & dimuon_mass_mask, axis=1)
            ]
            if len(events_VR_loose) > 0:
                weights_VR_loose = self.get_weights(
                    events_VR_loose, do_vars=False, apply_lumi_factors=True
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
                nMuon_VR_loose = ak.where(nMuon_VR_loose > 4, 4, nMuon_VR_loose)
                output[dataset]["histograms"]["VR_loose_muon_pt"].fill(
                    ak.flatten(muons_VR_loose.pt),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_VR_loose, muons_VR_loose.pt)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_loose.weight(), muons_VR_loose.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["VR_loose_muon_eta"].fill(
                    ak.flatten(muons_VR_loose.eta),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_VR_loose, muons_VR_loose.eta)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_loose.weight(), muons_VR_loose.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["VR_loose_muon_eta"].fill(
                    ak.flatten(muons_VR_loose.eta),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_VR_loose, muons_VR_loose.eta)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_loose.weight(), muons_VR_loose.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["VR_loose_muon_phi"].fill(
                    ak.flatten(muons_VR_loose.phi),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_VR_loose, muons_VR_loose.phi)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_loose.weight(), muons_VR_loose.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["VR_loose_muon_iso"].fill(
                    ak.flatten(muons_VR_loose.miniPFRelIso_all),
                    ak.flatten(
                        ak.broadcast_arrays(
                            nMuon_VR_loose,
                            muons_VR_loose.miniPFRelIso_all,
                        )[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_loose.weight(), muons_VR_loose.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["VR_loose_muon_dxy"].fill(
                    ak.flatten(muons_VR_loose.dxy),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_VR_loose, muons_VR_loose.dxy)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_loose.weight(), muons_VR_loose.pt
                        )[0]
                    ),
                )
                output[dataset]["histograms"]["VR_loose_muon_dz"].fill(
                    ak.flatten(muons_VR_loose.dz),
                    ak.flatten(
                        ak.broadcast_arrays(nMuon_VR_loose, muons_VR_loose.dz)[0]
                    ),
                    weight=ak.flatten(
                        ak.broadcast_arrays(
                            weights_VR_loose.weight(), muons_VR_loose.pt
                        )[0]
                    ),
                )
                muon_pairs_dr = muon_pairs_0.delta_r(muon_pairs_1)
                output[dataset]["histograms"]["VR_loose_dimuon_dr"].fill(
                    ak.flatten(muon_pairs_dr),
                    ak.flatten(ak.broadcast_arrays(nMuon_VR_loose, muon_pairs_dr)[0]),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_VR_loose.weight(), muon_pairs_dr)[0]
                    ),
                )
                muon_pairs_mass = (muon_pairs_0 + muon_pairs_1).mass
                output[dataset]["histograms"]["VR_loose_dimuon_mass"].fill(
                    ak.flatten(muon_pairs_mass),
                    ak.flatten(ak.broadcast_arrays(nMuon_VR_loose, muon_pairs_mass)[0]),
                    weight=ak.flatten(
                        ak.broadcast_arrays(weights_VR_loose.weight(), muon_pairs_mass)[
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
        histograms = {}
        for region in [
            "VR_tight",
            "VR_loose",
        ]:
            histograms.update(
                {
                    f"{region}_muon_pt": hist.Hist.new.Regular(
                        50,
                        3,
                        300,
                        name="muon_pt",
                        label="muon_pt",
                        transform=hist.axis.transform.log,
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_muon_eta": hist.Hist.new.Regular(
                        50,
                        -3,
                        3,
                        name="muon_eta",
                        label="muon_eta",
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_muon_phi": hist.Hist.new.Regular(
                        50,
                        -np.pi,
                        np.pi,
                        name="muon_phi",
                        label="muon_phi",
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_muon_iso": hist.Hist.new.Regular(
                        50,
                        0.01,
                        10,
                        name="muon_iso",
                        label="muon_iso",
                        transform=hist.axis.transform.log,
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_muon_dxy": hist.Hist.new.Regular(
                        50,
                        1e-4,
                        1,
                        name="muon_dxy",
                        label="muon_dxy",
                        transform=hist.axis.transform.log,
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_muon_dz": hist.Hist.new.Regular(
                        50,
                        1e-4,
                        1,
                        name="muon_dz",
                        label="muon_dz",
                        transform=hist.axis.transform.log,
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_dimuon_dr": hist.Hist.new.Regular(
                        50,
                        1e-2,
                        10,
                        name="dimuon_dr",
                        label="dimuon_dr",
                        transform=hist.axis.transform.log,
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                    f"{region}_dimuon_mass": hist.Hist.new.Regular(
                        50,
                        0.1,
                        100,
                        name="dimuon_mass",
                        label="dimuon_mass",
                        transform=hist.axis.transform.log,
                    )
                    .Regular(5, 3, 8, name="nMuon", label="nMuon")
                    .Weight(),
                }
            )

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
