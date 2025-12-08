import awkward as ak
import hist
import vector  # type: ignore[import]
from coffea import processor

# Importing CMS corrections
import workflows.CMS_corrections.golden_json_utils as golden_json_utils
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

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events = self.muon_filter(events)
        if len(events) == 0:
            return

        muons = events.Muon
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dxy) < 0.01)
            & (abs(muons.dz) < 0.01)
        )
        muons = muons[clean_muons]

        events, muons = events[ak.num(muons) >= 3], muons[ak.num(muons) >= 3]
        weights = self.get_weights(events).weight()

        # All combinations of OS dimuons
        muon_idx = ak.local_index(muons, axis=1)
        muon_pairs = ak.unzip(ak.cartesian([muons, muons], nested=False))
        muon_pairs_idx = ak.unzip(ak.cartesian([muon_idx, muon_idx], nested=False))
        muon_pairs_0, muon_pairs_1 = muon_pairs  # type: ignore[index]
        muon_pairs_idx_0, muon_pairs_idx_1 = muon_pairs_idx  # type: ignore[index]
        delta_r = muon_pairs_0.delta_r(muon_pairs_1)
        delta_r = delta_r[muon_pairs_idx_0 < muon_pairs_idx_1]  # avoid double counting

        output[dataset]["histograms"]["dimuon_dR"].fill(
            ak.flatten(delta_r),
            weight=ak.flatten(ak.broadcast_arrays(weights, delta_r)[0]),
        )

        mass = (muon_pairs_0 + muon_pairs_1).mass
        mass = mass[muon_pairs_idx_0 < muon_pairs_idx_1]  # avoid double counting
        delta_r = delta_r[(mass > 0.4) & (mass < 0.8)]

        output[dataset]["histograms"]["dimuon_dR_mass_cut"].fill(
            ak.flatten(delta_r),
            weight=ak.flatten(ak.broadcast_arrays(weights, delta_r)[0]),
        )

        # All combinations of OS dimuons
        muons1 = muons[muons.charge == 1]
        muons2 = muons[muons.charge == -1]
        muon_pairs = ak.unzip(ak.cartesian([muons1, muons2], nested=False))
        muon_pairs_0, muon_pairs_1 = muon_pairs  # type: ignore[index]
        delta_r = muon_pairs_0.delta_r(muon_pairs_1)

        output[dataset]["histograms"]["OS_dimuon_dR"].fill(
            ak.flatten(delta_r),
            weight=ak.flatten(ak.broadcast_arrays(weights, delta_r)[0]),
        )

        mass = (muon_pairs_0 + muon_pairs_1).mass
        delta_r = delta_r[(mass > 0.4) & (mass < 0.8)]

        output[dataset]["histograms"]["OS_dimuon_dR_mass_cut"].fill(
            ak.flatten(delta_r),
            weight=ak.flatten(ak.broadcast_arrays(weights, delta_r)[0]),
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
        # events = self.trigger_plateau(events)

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
            "dimuon_dR": hist.Hist.new.Regular(
                100,
                0.001,
                10,
                name="dimuon dR",
                label="dimuon dR",
                transform=hist.axis.transform.log,
            ).Weight(),
            "OS_dimuon_dR": hist.Hist.new.Regular(
                100,
                0.001,
                10,
                name="OS dimuon dR",
                label="OS dimuon dR",
                transform=hist.axis.transform.log,
            ).Weight(),
            "dimuon_dR_mass_cut": hist.Hist.new.Regular(
                100,
                0.001,
                10,
                name="dimuon dR",
                label="dimuon dR",
                transform=hist.axis.transform.log,
            ).Weight(),
            "OS_dimuon_dR_mass_cut": hist.Hist.new.Regular(
                100,
                0.001,
                10,
                name="OS dimuon dR",
                label="OS dimuon dR",
                transform=hist.axis.transform.log,
            ).Weight(),
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
