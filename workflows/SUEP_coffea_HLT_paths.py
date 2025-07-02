import awkward as ak
import hist
import numpy as np
import vector  # type: ignore[import]
from coffea import processor

# Importing CMS corrections
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
        self.era = self.era.replace("UL1", "201")
        self.do_syst = do_syst
        self.gensumweight = 1.0
        self.do_rochester = do_rochester

    def analysis(self, events, output):
        #######################################################################
        # ---- Trigger event selection
        # Cut based on ak4 jets to replicate the trigger
        #######################################################################

        dataset = events.metadata["dataset"]

        if len(events) > 0:
            weights = self.get_weights(events)
            output[dataset]["histograms"]["hlt_paths"].fill(
                len(events) * ["all"],
                weight=weights.weight(),
            )
            hlt_paths = [
                "TripleMu_5_3_3",
                "TripleMu_5_3_3_DZ_Mass3p8",
                "TripleMu_5_3_3_Mass3p8_DZ",
                "TripleMu_5_3_3_Mass3p8to60_DZ",
                "TripleMu_10_5_5_DZ",
                "TripleMu_12_10_5",
            ]
            for path in hlt_paths:
                if path in events.HLT.fields:
                    # Check if the path is in the HLT fields
                    hlt_mask = events.HLT[path]
                    output[dataset]["histograms"]["hlt_paths"].fill(
                        ak.sum(hlt_mask) * ["HLT_" + path],
                        weight=weights.weight()[hlt_mask],
                    )

        return

    def process(self, events):
        dataset = events.metadata["dataset"]
        cutflow = hist.Hist.new.StrCategory(
            ["all", "trigger"],
            name="cutflow",
            label="cutflow",
        ).Weight()
        histograms = {
            "hlt_paths": hist.Hist.new.StrCategory(
                [
                    "all",
                    "HLT_TripleMu_5_3_3",
                    "HLT_TripleMu_5_3_3_DZ_Mass3p8",
                    "HLT_TripleMu_5_3_3_Mass3p8_DZ",
                    "HLT_TripleMu_5_3_3_Mass3p8to60_DZ",
                    "HLT_TripleMu_10_5_5_DZ",
                    "HLT_TripleMu_12_10_5",
                ],
                name="hlt_paths",
                label="hlt_paths",
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
