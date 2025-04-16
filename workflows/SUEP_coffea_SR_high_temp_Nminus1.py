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

    def apply_SR_high_temp_minus_muon_ip3d_cut(self, events):
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
        tight_cut = (muons.miniPFRelIso_all < 0.65) & (
            (muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 0.5
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
        loose_cut = (muons.miniPFRelIso_all < 5) & (
            (muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 3
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

    def apply_SR_high_temp_minus_muon_iso_cut(self, events):
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
        tight_cut = (muons.ip3d < 0.007) & (
            (muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 0.5
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
        loose_cut = (muons.ip3d < 0.1) & (
            (muons.miniPFRelIso_all - muons.miniPFRelIso_chg) < 3
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

    def apply_SR_high_temp_minus_muon_neutral_iso_cut(self, events):
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
        tight_cut = (muons.ip3d < 0.007) & (muons.miniPFRelIso_all < 0.65)
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
        loose_cut = (muons.ip3d < 0.1) & (muons.miniPFRelIso_all < 5)
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

    def apply_SR_high_temp_minus_Z_mass_cut(self, events):
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

        select_by_muons_tight = ak.num(muons_tight_cut, axis=-1) > 2
        events_tight_cut = events_tight_cut[select_by_muons_tight]
        muons_tight_cut = muons_tight_cut[select_by_muons_tight]
        Z_cands_tight_cut = Z_cands_tight_cut[select_by_muons_tight]

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
        select_by_muons_loose_cut = ak.num(muons_loose_cut, axis=-1) > 2
        events_loose_cut = events_loose_cut[select_by_muons_loose_cut]
        muons_loose_cut = muons_loose_cut[select_by_muons_loose_cut]
        Z_cands_loose_cut = Z_cands_loose_cut[select_by_muons_loose_cut]

        return (
            events_tight_cut,
            events_loose_cut,
            muons_tight_cut,
            muons_loose_cut,
            Z_cands_tight_cut,
            Z_cands_loose_cut,
        )

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        # N-1 for muon_ip3d cut
        (
            events_SR_high_temp_tight,
            events_SR_high_temp_loose,
            muons_SR_high_temp_tight,
            muons_SR_high_temp_loose,
        ) = self.apply_SR_high_temp_minus_muon_ip3d_cut(events_)
        if len(events_SR_high_temp_tight) > 0:
            weights_SR_high_temp_tight = self.get_weights(events_SR_high_temp_tight)
            weights_SR_high_temp_tight.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_tight, self.era, syst=""
                    ),
                    axis=-1,
                ),
            )
            output[dataset]["histograms"]["SR_high_temp_tight_Nminus1_muon_ip3d"].fill(
                ak.flatten(muons_SR_high_temp_tight.ip3d),
                ak.flatten(muons_SR_high_temp_tight.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_high_temp_tight.ip3d,
                        weights_SR_high_temp_tight.weight(),
                    )[1]
                ),
            )
        if len(events_SR_high_temp_loose) > 0:
            weights_SR_high_temp_loose = self.get_weights(events_SR_high_temp_loose)
            weights_SR_high_temp_loose.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_loose, self.era, syst=""
                    ),
                    axis=-1,
                ),
            )
            output[dataset]["histograms"]["SR_high_temp_loose_Nminus1_muon_ip3d"].fill(
                ak.flatten(muons_SR_high_temp_loose.ip3d),
                ak.flatten(muons_SR_high_temp_loose.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_high_temp_loose.ip3d,
                        weights_SR_high_temp_loose.weight(),
                    )[1]
                ),
            )

        # N-1 for muon_iso cut
        (
            events_SR_high_temp_tight,
            events_SR_high_temp_loose,
            muons_SR_high_temp_tight,
            muons_SR_high_temp_loose,
        ) = self.apply_SR_high_temp_minus_muon_iso_cut(events_)
        if len(events_SR_high_temp_tight) > 0:
            weights_SR_high_temp_tight = self.get_weights(events_SR_high_temp_tight)
            weights_SR_high_temp_tight.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_tight, self.era, syst=""
                    ),
                    axis=-1,
                ),
            )
            output[dataset]["histograms"]["SR_high_temp_tight_Nminus1_muon_iso"].fill(
                ak.flatten(muons_SR_high_temp_tight.miniPFRelIso_all),
                ak.flatten(muons_SR_high_temp_tight.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_high_temp_tight.miniPFRelIso_all,
                        weights_SR_high_temp_tight.weight(),
                    )[1]
                ),
            )
        if len(events_SR_high_temp_loose) > 0:
            weights_SR_high_temp_loose = self.get_weights(events_SR_high_temp_loose)
            weights_SR_high_temp_loose.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_loose, self.era, syst=""
                    ),
                    axis=-1,
                ),
            )
            output[dataset]["histograms"]["SR_high_temp_loose_Nminus1_muon_iso"].fill(
                ak.flatten(muons_SR_high_temp_loose.miniPFRelIso_all),
                ak.flatten(muons_SR_high_temp_loose.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_high_temp_loose.miniPFRelIso_all,
                        weights_SR_high_temp_loose.weight(),
                    )[1]
                ),
            )

        # N-1 for muon_neutral_iso cut
        (
            events_SR_high_temp_tight,
            events_SR_high_temp_loose,
            muons_SR_high_temp_tight,
            muons_SR_high_temp_loose,
        ) = self.apply_SR_high_temp_minus_muon_neutral_iso_cut(events_)
        if len(events_SR_high_temp_tight) > 0:
            weights_SR_high_temp_tight = self.get_weights(events_SR_high_temp_tight)
            weights_SR_high_temp_tight.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_tight, self.era, syst=""
                    ),
                    axis=-1,
                ),
            )
            output[dataset]["histograms"][
                "SR_high_temp_tight_Nminus1_muon_neutral_iso"
            ].fill(
                ak.flatten(
                    muons_SR_high_temp_tight.miniPFRelIso_all
                    - muons_SR_high_temp_tight.miniPFRelIso_chg
                ),
                ak.flatten(muons_SR_high_temp_tight.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_high_temp_tight.miniPFRelIso_all
                        - muons_SR_high_temp_tight.miniPFRelIso_chg,
                        weights_SR_high_temp_tight.weight(),
                    )[1]
                ),
            )
        if len(events_SR_high_temp_loose) > 0:
            weights_SR_high_temp_loose = self.get_weights(events_SR_high_temp_loose)
            weights_SR_high_temp_loose.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_loose, self.era, syst=""
                    ),
                    axis=-1,
                ),
            )
            output[dataset]["histograms"][
                "SR_high_temp_loose_Nminus1_muon_neutral_iso"
            ].fill(
                ak.flatten(
                    muons_SR_high_temp_loose.miniPFRelIso_all
                    - muons_SR_high_temp_loose.miniPFRelIso_chg
                ),
                ak.flatten(muons_SR_high_temp_loose.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_SR_high_temp_loose.miniPFRelIso_all
                        - muons_SR_high_temp_loose.miniPFRelIso_chg,
                        weights_SR_high_temp_loose.weight(),
                    )[1]
                ),
            )

        # N-1 for Z_mass window cut
        (
            events_SR_high_temp_tight,
            events_SR_high_temp_loose,
            muons_SR_high_temp_tight,
            muons_SR_high_temp_loose,
            Z_cands_tight_cut,
            Z_cands_loose_cut,
        ) = self.apply_SR_high_temp_minus_Z_mass_cut(events_)
        if len(events_SR_high_temp_tight) > 0:
            weights_SR_high_temp_tight = self.get_weights(events_SR_high_temp_tight)
            weights_SR_high_temp_tight.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_tight, self.era, syst=""
                    ),
                    axis=-1,
                ),
            )
            output[dataset]["histograms"][
                "SR_high_temp_tight_Nminus1_dimuon_mass"
            ].fill(Z_cands_tight_cut.mass, weight=weights_SR_high_temp_tight.weight())
        if len(events_SR_high_temp_loose) > 0:
            weights_SR_high_temp_loose = self.get_weights(events_SR_high_temp_loose)
            weights_SR_high_temp_loose.add(
                "MuonSF",
                weight=ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        muons_SR_high_temp_loose, self.era, syst=""
                    ),
                    axis=-1,
                ),
            )
            output[dataset]["histograms"][
                "SR_high_temp_loose_Nminus1_dimuon_mass"
            ].fill(Z_cands_loose_cut.mass, weight=weights_SR_high_temp_loose.weight())

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
            "SR_high_temp_tight_Nminus1_muon_ip3d": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="muon_ip3d",
                label="muon_ip3d",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_loose_Nminus1_muon_ip3d": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="muon_ip3d",
                label="muon_ip3d",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_tight_Nminus1_muon_iso": hist.Hist.new.Regular(
                100,
                0.02,
                20,
                name="muon_iso",
                label="muon_iso",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_loose_Nminus1_muon_iso": hist.Hist.new.Regular(
                100,
                0.02,
                20,
                name="muon_iso",
                label="muon_iso",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_tight_Nminus1_muon_neutral_iso": hist.Hist.new.Regular(
                100,
                0.02,
                20,
                name="muon_neutral_iso",
                label="muon_neutral_iso",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_loose_Nminus1_muon_neutral_iso": hist.Hist.new.Regular(
                100,
                0.02,
                20,
                name="muon_neutral_iso",
                label="muon_neutral_iso",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "SR_high_temp_tight_Nminus1_dimuon_mass": hist.Hist.new.Regular(
                100, 0, 200, name="dimuon_mass", label="dimuon_mass"
            ).Weight(),
            "SR_high_temp_loose_Nminus1_dimuon_mass": hist.Hist.new.Regular(
                100, 0, 200, name="dimuon_mass", label="dimuon_mass"
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
