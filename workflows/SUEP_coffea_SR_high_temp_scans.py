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

    def apply_SR_high_temp(self, events, ip3d_cut=None, iso_cut=None):
        """
        Apply the SR_high_temp tight selection to the events.
        THe ip3d and iso cuts can be varied.
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

        # Loose SR selection
        ip3d_cut = ip3d_cut if ip3d_cut is not None else 0.1
        iso_cut = iso_cut if iso_cut is not None else 5
        loose_cut = (
            (muons.ip3d < ip3d_cut)
            & (muons.miniPFRelIso_all < iso_cut)
            & ((muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 3)
        )
        muons_loose_cut = muons[clean_muons & loose_cut]
        events_loose_cut, muons_loose_cut, Z_cands_loose_cut, _ = (
            self.find_Z_candidates(events, muons_loose_cut)
        )
        mass_cut = Z_cands_loose_cut.mass < 70
        events_loose_cut = events_loose_cut[mass_cut]
        muons_loose_cut = muons_loose_cut[mass_cut]

        select_by_muons_loose = ak.num(muons_loose_cut, axis=-1) > 2
        events_loose_cut = events_loose_cut[select_by_muons_loose]
        muons_loose_cut = muons_loose_cut[select_by_muons_loose]

        # Tight SR selection
        ip3d_cut = ip3d_cut if ip3d_cut is not None else 0.007
        iso_cut = iso_cut if iso_cut is not None else 0.65
        tight_cut = (
            (muons.ip3d < ip3d_cut)
            & (muons.miniPFRelIso_all < iso_cut)
            & ((muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 0.5)
        )
        muons_tight_cut = muons[clean_muons & tight_cut]
        events_tight_cut, muons_tight_cut, Z_cands_tight_cut, _ = (
            self.find_Z_candidates(events, muons_tight_cut)
        )
        mass_cut = Z_cands_tight_cut.mass < 70
        events_tight_cut = events_tight_cut[mass_cut]
        muons_tight_cut = muons_tight_cut[mass_cut]

        select_by_muons_tight = ak.num(muons_tight_cut, axis=-1) > 2
        events_tight_cut = events_tight_cut[select_by_muons_tight]
        muons_tight_cut = muons_tight_cut[select_by_muons_tight]

        return events_loose_cut, events_tight_cut, muons_loose_cut, muons_tight_cut

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        ip3d_cuts = 1.001 * np.logspace(-3, -1, 11)[:-1]
        for ip3d_cut in ip3d_cuts:
            events_loose_cut, events_tight_cut, muons_loose_cut, muons_tight_cut = (
                self.apply_SR_high_temp(events_, ip3d_cut=ip3d_cut)
            )
            if len(events_loose_cut) > 0:
                weights_loose_cut = self.get_weights(events_loose_cut)
                weights_loose_cut.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(muons_loose_cut, syst=""),
                        axis=-1,
                    ),
                )
                nMuon_loose_cut = ak.num(muons_loose_cut, axis=-1)
                output[dataset]["histograms"][
                    "SR_high_temp_loose_ip3d_cut_vs_nMuon"
                ].fill(
                    ip3d_cut=ip3d_cut,
                    nMuon=ak.where(nMuon_loose_cut > 7, 7, nMuon_loose_cut),
                    weight=weights_loose_cut.weight(),
                )
            if len(events_tight_cut) > 0:
                weights_tight_cut = self.get_weights(events_tight_cut)
                weights_tight_cut.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(muons_tight_cut, syst=""),
                        axis=-1,
                    ),
                )
                nMuon_tight_cut = ak.num(muons_tight_cut, axis=-1)
                output[dataset]["histograms"][
                    "SR_high_temp_tight_ip3d_cut_vs_nMuon"
                ].fill(
                    ip3d_cut=ip3d_cut,
                    nMuon=ak.where(nMuon_tight_cut > 7, 7, nMuon_tight_cut),
                    weight=weights_tight_cut.weight(),
                )

        iso_cuts = 1.001 * np.logspace(-1, 1, 11)[:-1]
        for iso_cut in iso_cuts:
            events_loose_cut, events_tight_cut, muons_loose_cut, muons_tight_cut = (
                self.apply_SR_high_temp(events_, iso_cut=iso_cut)
            )
            if len(events_loose_cut) > 0:
                weights_loose_cut = self.get_weights(events_loose_cut)
                weights_loose_cut.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(muons_loose_cut, syst=""),
                        axis=-1,
                    ),
                )
                nMuon_loose_cut = ak.num(muons_loose_cut, axis=-1)
                output[dataset]["histograms"][
                    "SR_high_temp_loose_iso_cut_vs_nMuon"
                ].fill(
                    iso_cut=iso_cut,
                    nMuon=ak.where(nMuon_loose_cut > 7, 7, nMuon_loose_cut),
                    weight=weights_loose_cut.weight(),
                )
            if len(events_tight_cut) > 0:
                weights_tight_cut = self.get_weights(events_tight_cut)
                weights_tight_cut.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(muons_tight_cut, syst=""),
                        axis=-1,
                    ),
                )
                nMuon_tight_cut = ak.num(muons_tight_cut, axis=-1)
                output[dataset]["histograms"][
                    "SR_high_temp_tight_iso_cut_vs_nMuon"
                ].fill(
                    iso_cut=iso_cut,
                    nMuon=ak.where(nMuon_tight_cut > 7, 7, nMuon_tight_cut),
                    weight=weights_tight_cut.weight(),
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
            "SR_high_temp_loose_ip3d_cut_vs_nMuon": hist.Hist.new.Reg(
                10,
                1e-3,
                0.1,
                name="ip3d_cut",
                label="ip3d_cut",
                transform=hist.axis.transform.log,
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "SR_high_temp_tight_ip3d_cut_vs_nMuon": hist.Hist.new.Reg(
                10,
                1e-3,
                0.1,
                name="ip3d_cut",
                label="ip3d_cut",
                transform=hist.axis.transform.log,
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "SR_high_temp_loose_iso_cut_vs_nMuon": hist.Hist.new.Reg(
                10,
                0.1,
                10,
                name="iso_cut",
                label="iso_cut",
                transform=hist.axis.transform.log,
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
            .Weight(),
            "SR_high_temp_tight_iso_cut_vs_nMuon": hist.Hist.new.Reg(
                10,
                0.1,
                10,
                name="iso_cut",
                label="iso_cut",
                transform=hist.axis.transform.log,
            )
            .Regular(5, 3, 8, name="nMuon", label="nMuon")
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
