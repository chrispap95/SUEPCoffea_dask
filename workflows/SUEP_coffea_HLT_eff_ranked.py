"""Conditional triple-muon efficiencies for probes ranked by offline pT.

Each of the three highest-pT selected muons is a probe if at least two OTHER
selected muons match the reference trigger. Each rank contributes at most once
per event. The numerator tests the full target path, not an individual filter.
Reference matching, path-specific cuts, weights and prescale treatment follow
SUEP_coffea_HLT_eff. The original inclusive-probe workflow is unchanged.
"""

import awkward as ak
import hist
import numpy as np
from coffea import processor

from workflows.SUEP_coffea_HLT_eff import SUEP_processor as InclusiveProcessor

# One-GeV bins for the first two ranks; retain the original third-probe binning.
RANK_VARIABLES = {
    1: ("leading_muon_pt", 100, 100),
    2: ("subleading_muon_pt", 60, 60),
    3: ("subsubleading_muon_pt", None, 20),
}


class SUEP_processor(InclusiveProcessor):
    def muon_filter(self, events):
        muons = events.Muon
        clean = muons.looseId & (abs(muons.eta) < 2.4) & (abs(muons.dz) < 0.2)
        muons = muons[clean]
        muons = muons[ak.argsort(muons.pt, axis=-1, ascending=False, stable=True)]
        enough = ak.num(muons, axis=-1) >= 3
        events, muons = events[enough], muons[enough]
        pt_cuts = (muons[:, 0].pt > 17) & (muons[:, 1].pt > 8)
        return events[pt_cuts], muons[pt_cuts]

    @staticmethod
    def select_ranked_probes(muons, matched):
        """Return jagged zero-or-one-probe collections, preserving event alignment.

        Count tags excluding the candidate itself. A matched probe is allowed,
        and additional tag pairs never duplicate a probe. Ranks refer to all
        selected muons in the event, not to individual three-muon combinations.
        """
        other_tags = ak.sum(matched, axis=-1)[:, np.newaxis] - ak.values_astype(
            matched, np.int64
        )
        indices = ak.local_index(muons, axis=-1)
        return {
            rank: muons[(indices == rank - 1) & (other_tags >= 2)]
            for rank in RANK_VARIABLES
        }

    def trigger_matching(self, events, muons, weights):
        is_muon = events.TrigObj.id == 13
        is_iso = (events.TrigObj.filterBits & 1) != 0
        is_dimuon = (events.TrigObj.filterBits & (1 << 4)) != 0
        flags = is_iso if self.era in ("2016APV", "2016") else is_iso & is_dimuon
        trig_muons = events.TrigObj[is_muon & flags]
        pairs = ak.cartesian({"mu": muons, "trig": trig_muons}, axis=1, nested=True)
        matched = ak.any(
            (pairs.mu.delta_r(pairs.trig) < 0.01)
            & (abs(pairs.mu.pt - pairs.trig.pt) / pairs.trig.pt < 0.1),
            axis=-1,
        )
        return events, muons, self.select_ranked_probes(muons, matched), weights

    def target_paths(self):
        if self.era in ("2016APV", "2016"):
            return ["HLT_TripleMu_5_3_3", "HLT_TripleMu_12_10_5"]
        if self.era == "2017":
            mass_path = "HLT_TripleMu_5_3_3_Mass3p8to60_DZ"
        elif self.era == "2018" or self.era.startswith("202"):
            mass_path = "HLT_TripleMu_5_3_3_Mass3p8_DZ"
        else:
            raise RuntimeError(f"Era {self.era} not recognized.")
        return [mass_path, "HLT_TripleMu_10_5_5_DZ", "HLT_TripleMu_12_10_5"]

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]
        fields = events.HLT.fields
        if "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8" in fields:
            reference = events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8
        elif self.era == "2017" and "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8" in fields:
            reference = events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8
        elif (
            self.era in ("2016APV", "2016")
            and "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ" in fields
        ):
            reference = events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ
        else:
            raise RuntimeError("No suitable dimuon trigger found.")
        events = events[reference]
        if len(events) == 0:
            return
        events, muons = self.muon_filter(events)
        if len(events) == 0:
            return
        weights = self.get_weights(
            events, do_vars=False, apply_lumi_factors=False
        ).weight()
        events, muons, ranked_probes, weights = self.trigger_matching(
            events, muons, weights
        )

        if (
            self.isMC
            and self.era == "2017"
            and all(
                field in fields
                for field in (
                    "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
                    "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8",
                )
            )
        ):
            weights = ak.where(
                events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8
                & ~events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8,
                weights * (36.7 / 41.5),
                weights,
            )

        paths = self.target_paths()
        histograms = output[dataset]["histograms"]
        for rank, (variable, _, _) in RANK_VARIABLES.items():
            # Give the inherited path selections references to this rank's own
            # histograms. The histogram objects are shared, not copied.
            rank_output = {
                dataset: {
                    "histograms": {
                        f"{prefix}_{path}": histograms[f"{prefix}_{path}_{variable}"]
                        for path in paths
                        for prefix in ("NUM", "DEN")
                    }
                }
            }
            probes = ranked_probes[rank]
            for path in paths:
                trigger = path.removeprefix("HLT_")
                if trigger not in fields:
                    continue
                fill = getattr(self, f"eff_{trigger}")
                if trigger == "TripleMu_5_3_3":
                    fill(events, probes, weights, rank_output, dataset)
                else:
                    fill(events, muons, probes, weights, rank_output, dataset)

    def process(self, events):
        dataset = events.metadata["dataset"]
        histograms = {}
        for path in self.target_paths():
            for variable, n_bins, pt_max in RANK_VARIABLES.values():
                bins = (
                    n_bins if n_bins is not None else (100 if "5_3_3" in path else 80)
                )
                for prefix in ("NUM", "DEN"):
                    histograms[f"{prefix}_{path}_{variable}"] = hist.Hist.new.Regular(
                        bins, 0, pt_max, name=variable, label=variable
                    ).Weight()
        output = {
            dataset: {
                "cutflow": hist.Hist.new.StrCategory(
                    ["all", "trigger"], name="cutflow", label="cutflow"
                ).Weight(),
                "gensumweight": processor.value_accumulator(float, 0),
                "histograms": histograms,
            }
        }
        if self.isMC:
            self.gensumweight = ak.sum(events.genWeight)
            output[dataset]["gensumweight"].add(self.gensumweight)
        self.analysis(events, output)
        return output
