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

    def apply_SR_high_temp(self, events):
        """
        Apply the SR_high_temp selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dz) < 0.2)
        )

        # Tight SR selection
        tight_cut = (
            (muons.ip3d < 0.007)
            & (muons.miniPFRelIso_all < 0.65)
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

        # Loose SR selection
        loose_cut = (
            (muons.ip3d < 0.1)
            & (muons.miniPFRelIso_all < 5)
            & ((muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 3)
        )
        muons_loose_cut = muons[clean_muons & loose_cut]
        events_loose_cut, muons_loose_cut, Z_cands_loose_cut, _ = (
            self.find_Z_candidates(events, muons_loose_cut)
        )
        mass_cut = Z_cands_loose_cut.mass < 70
        events_loose_cut = events_loose_cut[mass_cut]
        muons_loose_cut = muons_loose_cut[mass_cut]

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

        if len(events_SR_high_temp_tight) > 0:
            weights_SR_high_temp_tight = self.get_weights(
                events_SR_high_temp_tight, do_vars=True
            )
            weights_SR_high_temp_tight.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_tight, self.era, syst=""
                    ),
                    axis=-1,
                ),
                weightUp=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_tight, self.era, syst="up"
                    ),
                    axis=-1,
                ),
                weightDown=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_tight, self.era, syst="down"
                    ),
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
                        output[dataset]["histograms"]["SR_high_temp_tight"]
                        .copy()
                        .reset()
                    )
                    output[dataset]["histograms"][f"SR_high_temp_tight_{syst}"].fill(
                        ak.where(
                            nMuon_SR_high_temp_tight > 7, 7, nMuon_SR_high_temp_tight
                        ),
                        weight=weights_SR_high_temp_tight.weight(syst),
                    )

        if len(events_SR_high_temp_loose) > 0:
            weights_SR_high_temp_loose = self.get_weights(
                events_SR_high_temp_loose, do_vars=True
            )
            weights_SR_high_temp_loose.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_loose, self.era, syst=""
                    ),
                    axis=-1,
                ),
                weightUp=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_loose, self.era, syst="up"
                    ),
                    axis=-1,
                ),
                weightDown=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_loose, self.era, syst="down"
                    ),
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
                        output[dataset]["histograms"]["SR_high_temp_loose"]
                        .copy()
                        .reset()
                    )
                    output[dataset]["histograms"][f"SR_high_temp_loose_{syst}"].fill(
                        ak.where(
                            nMuon_SR_high_temp_loose > 7, 7, nMuon_SR_high_temp_loose
                        ),
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
