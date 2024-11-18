from typing import Optional
import awkward as ak
from coffea import processor


class GenSumWeightExtractor(processor.ProcessorABC):
    def __init__(self, use_new_format: Optional[bool] = True) -> None:
        self._use_new_format = use_new_format
        self._accumulator = processor.value_accumulator(float, 0)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]
        if self._use_new_format:
            genEventSumw = ak.sum(events.genEventSumwPreSkim)
        else:
            genEventSumw = ak.sum(events.genEventSumw)
        output = processor.value_accumulator(float, genEventSumw)
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
