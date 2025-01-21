import awkward as ak
import hist
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
        do_rochester: bool = False,
    ) -> None:
        self.isMC = isMC
        self.era = era if isinstance(era, str) else str(era)
        self.gensumweight = 1.0
        self.do_rochester = do_rochester

    def apply_VR(self, events):
        """
        Apply the VR selection to the events.
        """
        muons = events.Muon

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Make sure we are the trigger plateau
        events, muons = events[ak.num(muons) > 2], muons[ak.num(muons) > 2]

        # VR selection
        muons = muons[(muons.ip3d > 0.01) & (muons.miniPFRelIso_all > 0.3)]

        # # Extra selections
        # # add promptness, pt, and tight isolation cuts
        # muons_tight = muons[
        #     muons.mediumPromptId & (muons.pt > 30) & (muons.miniPFRelIso_all < 0.1)
        # ]

        # # Make sure there muons with both charges available
        # # NOTE: I need to reapply this because I want to add more selections to the muons
        # enough_muons = (ak.sum(muons_tight.charge == 1, axis=-1) > 0) & (
        #     ak.sum(muons_tight.charge == -1, axis=-1) > 0
        # )
        # muons_tight = muons_tight[enough_muons]
        # muons = muons[enough_muons]
        # events = events[enough_muons]

        # # VR selection
        # # Select the hardest opposite sign muon pair
        # muons1 = muons_tight[muons_tight.charge == 1]
        # muons2 = muons_tight[muons_tight.charge == -1]
        # muons1 = ak.firsts(muons1)
        # muons2 = ak.firsts(muons2)

        # # Hardest pair
        # Z_cands = muons1 + muons2
        # in_mass_window = abs(Z_cands.mass - Z_MASS) < 2 * Z_WIDTH
        # muons = muons[in_mass_window]
        # events = events[in_mass_window]

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        events_VR, muons_VR = self.apply_VR(events_)
        if len(events_VR) > 0:
            weights_VR = self.get_weights(events_VR).weight()
            nMuon_VR = ak.num(muons_VR, axis=-1)
            nMuon_VR = ak.where(nMuon_VR > 7, 7, nMuon_VR)
            output[dataset]["histograms"]["muon_pt_vs_nMuon"].fill(
                ak.flatten(muons_VR.pt),
                ak.flatten(ak.broadcast_arrays(muons_VR.pt, nMuon_VR)[1]),
                weight=ak.flatten(ak.broadcast_arrays(muons_VR.pt, weights_VR)[1]),
            )
            output[dataset]["histograms"]["muon_ip3d_vs_nMuon"].fill(
                ak.flatten(muons_VR.ip3d),
                ak.flatten(ak.broadcast_arrays(muons_VR.pt, nMuon_VR)[1]),
                weight=ak.flatten(ak.broadcast_arrays(muons_VR.pt, weights_VR)[1]),
            )
            output[dataset]["histograms"]["muon_iso_vs_nMuon"].fill(
                ak.flatten(muons_VR.miniPFRelIso_all),
                ak.flatten(ak.broadcast_arrays(muons_VR.pt, nMuon_VR)[1]),
                weight=ak.flatten(ak.broadcast_arrays(muons_VR.pt, weights_VR)[1]),
            )

            muons1 = muons_VR[muons_VR.charge == 1]
            muons2 = muons_VR[muons_VR.charge == -1]
            enough_muons = (ak.num(muons1) > 0) & (ak.num(muons2) > 0)
            muons1 = ak.firsts(muons1[enough_muons])
            muons2 = ak.firsts(muons2[enough_muons])
            dimuon = muons1 + muons2
            output[dataset]["histograms"]["dimuon_mass_hard_vs_nMuon"].fill(
                dimuon.mass,
                nMuon_VR[enough_muons],
                weight=weights_VR[enough_muons],
            )

            events_VR, muons_VR, Z_cands = self.find_Z_candidates(events_VR, muons_VR)
            output[dataset]["histograms"]["dimuon_mass_best_vs_nMuon"].fill(
                Z_cands.mass,
                ak.num(muons_VR),
                weight=self.get_weights(events_VR).weight(),
            )

        return

    def analysis(self, events, output):
        #######################################################################
        # ---- Trigger event selection
        # Cut based on ak4 jets to replicate the trigger
        #######################################################################

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
            "muon_pt_vs_nMuon": hist.Hist.new.Regular(
                50,
                0,
                100,
                name="muon_pt",
                label="muon_pt",
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "muon_ip3d_vs_nMuon": hist.Hist.new.Regular(
                50,
                0.01,
                1,
                name="muon_ip3d",
                label="muon_ip3d",
                transform=hist.axis.transform.log,
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "muon_iso_vs_nMuon": hist.Hist.new.Regular(
                50,
                1e-3,
                10,
                name="muon_iso",
                label="muon_iso",
                transform=hist.axis.transform.log,
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "dimuon_mass_hard_vs_nMuon": hist.Hist.new.Regular(
                50,
                0,
                200,
                name="dimuon_mass_hardest",
                label="dimuon_mass_hardest",
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "dimuon_mass_best_vs_nMuon": hist.Hist.new.Regular(
                50,
                0,
                200,
                name="dimuon_mass_best",
                label="dimuon_mass_best",
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            # "sph_vs_nMuon": hist.Hist.new.Regular(
            #     50,
            #     0,
            #     1,
            #     name="sph",
            #     label="sph",
            # )
            # .Regular(5, 3, 8, name="nMuon", label="nMuon")
            # .Weight(),
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
