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

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events = self.muon_filter(events)
        if len(events) == 0:
            return

        muons = events.Muon
        events, muons = events[ak.num(muons) >= 3], muons[ak.num(muons) >= 3]
        weights = self.get_weights(events).weight()
        sum_w = ak.sum(weights)
        sum_w2 = ak.sum(weights**2)

        # Cuts:
        #     - muon_pt > 3
        #     - muon_eta < 2.4
        #     - muon_isMediumId == True
        #     - abs(muon_dxy) < 0.2
        #     - abs(muon_dz) < 0.2

        for pt_cut in output[dataset]["histograms"]["muon_pt_NUM"].axes[0].edges[:-1]:
            muons_ = muons[muons.pt > pt_cut]
            weights_ = weights[ak.num(muons_) >= 3]
            output[dataset]["histograms"]["muon_pt_NUM"][1.01 * pt_cut * 1j] = (
                ak.sum(weights_),
                ak.sum(weights_**2),
            )
            output[dataset]["histograms"]["muon_pt_DEN"][1.01 * pt_cut * 1j] = (
                sum_w,
                sum_w2,
            )

        for abseta_cut in (
            output[dataset]["histograms"]["muon_abseta_NUM"].axes[0].edges[:-1]
        ):
            muons_ = muons[abs(muons.eta) < abseta_cut]
            weights_ = weights[ak.num(muons_) >= 3]
            output[dataset]["histograms"]["muon_abseta_NUM"][1.01 * abseta_cut * 1j] = (
                ak.sum(weights_),
                ak.sum(weights_**2),
            )
            output[dataset]["histograms"]["muon_abseta_DEN"][1.01 * abseta_cut * 1j] = (
                sum_w,
                sum_w2,
            )

        muons_ = muons[muons.looseId]
        weights_ = weights[ak.num(muons_) >= 3]
        output[dataset]["histograms"]["muon_id_NUM"]["looseId"] = (
            ak.sum(weights_),
            ak.sum(weights_**2),
        )
        output[dataset]["histograms"]["muon_id_DEN"]["looseId"] = (
            sum_w,
            sum_w2,
        )
        muons_ = muons[muons.mediumId]
        weights_ = weights[ak.num(muons_) >= 3]
        output[dataset]["histograms"]["muon_id_NUM"]["mediumId"] = (
            ak.sum(weights_),
            ak.sum(weights_**2),
        )
        output[dataset]["histograms"]["muon_id_DEN"]["mediumId"] = (
            sum_w,
            sum_w2,
        )
        muons_ = muons[muons.mediumPromptId]
        weights_ = weights[ak.num(muons_) >= 3]
        output[dataset]["histograms"]["muon_id_NUM"]["mediumPromptId"] = (
            ak.sum(weights_),
            ak.sum(weights_**2),
        )
        output[dataset]["histograms"]["muon_id_DEN"]["mediumPromptId"] = (
            sum_w,
            sum_w2,
        )
        muons_ = muons[muons.tightId]
        weights_ = weights[ak.num(muons_) >= 3]
        output[dataset]["histograms"]["muon_id_NUM"]["tightId"] = (
            ak.sum(weights_),
            ak.sum(weights_**2),
        )
        output[dataset]["histograms"]["muon_id_DEN"]["tightId"] = (
            sum_w,
            sum_w2,
        )

        for absdxy_cut in (
            output[dataset]["histograms"]["muon_absdxy_NUM"].axes[0].edges[:-1]
        ):
            muons_ = muons[abs(muons.dxy) < absdxy_cut]
            weights_ = weights[ak.num(muons_) >= 3]
            output[dataset]["histograms"]["muon_absdxy_NUM"][1.01 * absdxy_cut * 1j] = (
                ak.sum(weights_),
                ak.sum(weights_**2),
            )
            output[dataset]["histograms"]["muon_absdxy_DEN"][1.01 * absdxy_cut * 1j] = (
                sum_w,
                sum_w2,
            )

        for absdz_cut in (
            output[dataset]["histograms"]["muon_absdz_NUM"].axes[0].edges[:-1]
        ):
            muons_ = muons[abs(muons.dz) < absdz_cut]
            weights_ = weights[ak.num(muons_) >= 3]
            output[dataset]["histograms"]["muon_absdz_NUM"][1.01 * absdz_cut * 1j] = (
                ak.sum(weights_),
                ak.sum(weights_**2),
            )
            output[dataset]["histograms"]["muon_absdz_DEN"][1.01 * absdz_cut * 1j] = (
                sum_w,
                sum_w2,
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
            "muon_pt_NUM": hist.Hist.new.Regular(
                30,
                3,
                60,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            ).Weight(),
            "muon_pt_DEN": hist.Hist.new.Regular(
                30,
                3,
                60,
                name="muon_pt",
                label="muon_pt",
                transform=hist.axis.transform.log,
            ).Weight(),
            "muon_abseta_NUM": hist.Hist.new.Regular(
                10,
                2,
                3,
                name="muon_abseta",
                label="muon_abseta",
            ).Weight(),
            "muon_abseta_DEN": hist.Hist.new.Regular(
                10,
                2,
                3,
                name="muon_abseta",
                label="muon_abseta",
            ).Weight(),
            "muon_id_NUM": hist.Hist.new.StrCat(
                ["looseId", "mediumId", "mediumPromptId", "tightId"],
                name="muon_id",
                label="muon_id",
            ).Weight(),
            "muon_id_DEN": hist.Hist.new.StrCat(
                ["looseId", "mediumId", "mediumPromptId", "tightId"],
                name="muon_id",
                label="muon_id",
            ).Weight(),
            "muon_absdxy_NUM": hist.Hist.new.Regular(
                30,
                0.001,
                1,
                name="muon_absdxy",
                label="muon_absdxy",
                transform=hist.axis.transform.log,
            ).Weight(),
            "muon_absdxy_DEN": hist.Hist.new.Regular(
                30,
                0.001,
                1,
                name="muon_absdxy",
                label="muon_absdxy",
                transform=hist.axis.transform.log,
            ).Weight(),
            "muon_absdz_NUM": hist.Hist.new.Regular(
                30,
                0.001,
                1,
                name="muon_absdz",
                label="muon_absdz",
                transform=hist.axis.transform.log,
            ).Weight(),
            "muon_absdz_DEN": hist.Hist.new.Regular(
                30,
                0.001,
                1,
                name="muon_absdz",
                label="muon_absdz",
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
