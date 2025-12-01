import awkward as ak
import hist
import numpy as np
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

    def muon_filter(self, events):
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]
        clean_muons = (muons.mediumId) & (abs(muons.eta) < 2.4) & (abs(muons.dz) < 0.1)
        muons = muons[clean_muons]
        select_by_muons_low = ak.num(muons, axis=-1) > 2
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        # Since the 3rd muon pt cut varies, need to apply it also to the leading and subleading muons
        pt_cuts = (muons[:, 0].pt > 17) & (muons[:, 1].pt > 8)
        return events[pt_cuts], muons[pt_cuts]

    def mass_cut(
        self,
        events,
        muons,
        mu_probes,
        weights,
        mass_min=3.8,
        mass_max=np.inf,
        dz_min=np.inf,
        n_pairs=3,
    ):
        # Make sure at least three dimuons pass the mass cut
        dimuons = ak.combinations(muons, 2, fields=["mu1", "mu2"])
        dimuon_masses = (dimuons.mu1 + dimuons.mu2).mass
        dimuon_dz = abs(dimuons.mu1.dz - dimuons.mu2.dz)
        dimuon_masses = dimuon_masses[dimuon_dz < dz_min]
        pass_mass_cut = (ak.sum(dimuon_masses > mass_min, axis=-1) >= n_pairs) & (
            ak.sum(dimuon_masses < mass_max, axis=-1) >= n_pairs
        )
        events = events[pass_mass_cut]
        muons = muons[pass_mass_cut]
        mu_probes = mu_probes[pass_mass_cut]
        weights = weights[pass_mass_cut]
        return events, muons, mu_probes, weights

    def trigger_matching(self, events, muons, weights):
        is_trig_muuon = events.TrigObj.id == 13
        is_TrkIsoVVL = (events.TrigObj.filterBits >> 0) & 1  # flag for TrkIsoVVL
        is_2mu = (events.TrigObj.filterBits >> 4) & 1  # flag for 2mu
        flags = ak.values_astype(is_TrkIsoVVL & is_2mu, np.bool8)
        if self.era in ["2016APV", "2016"]:
            # no 2mu flag in 2016 for some reason
            flags = ak.values_astype(is_TrkIsoVVL, np.bool8)
        trig_muons = events.TrigObj[is_trig_muuon & flags]

        pairs = ak.cartesian({"mu": muons, "trig": trig_muons}, axis=1, nested=True)
        pair0, pair1 = ak.unzip(pairs)  # type: ignore[index]
        matched = ak.any(
            (pair0.delta_r(pair1) < 0.01) & (abs(pair0.pt - pair1.pt) / pair1.pt < 0.1),
            axis=-1,
        )

        # Form all combinations
        mu1, mu2, mu3 = ak.unzip(ak.combinations(muons, 3, axis=1))  # type: ignore[index]
        mu1_idx, mu2_idx, mu3_idx = ak.unzip(ak.argcombinations(muons, 3, axis=1))  # type: ignore[index]
        mu1_is_matching = matched[mu1_idx]
        mu2_is_matching = matched[mu2_idx]
        mu3_is_matching = matched[mu3_idx]
        mu1_probes = mu1[mu2_is_matching & mu3_is_matching]
        mu2_probes = mu2[mu1_is_matching & mu3_is_matching]
        mu3_probes = mu3[mu1_is_matching & mu2_is_matching]
        mu_probes = ak.concatenate([mu1_probes, mu2_probes, mu3_probes], axis=-1)
        return events, muons, mu_probes, weights

    def eff_TripleMu_5_3_3(self, events, mu_probes, weights, output, dataset):
        prescale = 1.0
        if not self.isMC:
            # make sure trigger path was on
            run_cutoff = (events.run >= 274954) & (events.run <= 281616)  # type: ignore[name-defined]
            events = events[run_cutoff]
            mu_probes = mu_probes[run_cutoff]
            weights = weights[run_cutoff]
            if self.era == "2016APV":
                prescale = 7.657859683 / 16.709824987
            elif self.era == "2016":
                prescale = 0.388200744 / 8.081886970

        if len(weights) == 0:
            return

        output[dataset]["histograms"]["DEN_HLT_TripleMu_5_3_3"].fill(
            ak.flatten(mu_probes.pt),
            weight=prescale * ak.flatten(ak.broadcast_arrays(weights, mu_probes.pt)[0]),
        )
        mu_probes_ = mu_probes[events.HLT.TripleMu_5_3_3]
        weights_ = weights[events.HLT.TripleMu_5_3_3]
        if len(weights_) > 0:
            output[dataset]["histograms"]["NUM_HLT_TripleMu_5_3_3"].fill(
                ak.flatten(mu_probes_.pt),
                weight=ak.flatten(ak.broadcast_arrays(weights_, mu_probes_.pt)[0]),
            )

    # def eff_TripleMu_5_3_3_DZ_Mass3p8(
    #     self, events, muons, mu_probes, weights, output, dataset
    # ):
    #     events_, muons_, mu_probes_, weights_ = self.mass_cut(
    #         events,
    #         muons,
    #         mu_probes,
    #         weights,
    #         mass_min=3.8,
    #         dz_min=0.1,
    #         n_pairs=3,
    #     )

    #     if not self.isMC:
    #         # make sure trigger path was on
    #         run_cutoff = (events_.run >= 281613) & (events_.run <= 284044)  # type: ignore[name-defined]
    #         events_ = events_[run_cutoff]
    #         muons_ = muons_[run_cutoff]
    #         mu_probes_ = mu_probes_[run_cutoff]
    #         weights_ = weights_[run_cutoff]

    #     if len(weights_) == 0:
    #         return

    #     output[dataset]["histograms"]["DEN_HLT_TripleMu_5_3_3_DZ_Mass3p8"].fill(
    #         ak.flatten(mu_probes_.pt),
    #         weight=ak.flatten(ak.broadcast_arrays(weights_, mu_probes_.pt)[0]),
    #     )
    #     mu_probes_ = mu_probes_[events_.HLT.TripleMu_5_3_3_DZ_Mass3p8]
    #     weights_ = weights_[events_.HLT.TripleMu_5_3_3_DZ_Mass3p8]
    #     if len(weights_) > 0:
    #         output[dataset]["histograms"]["NUM_HLT_TripleMu_5_3_3_DZ_Mass3p8"].fill(
    #             ak.flatten(mu_probes_.pt),
    #             weight=ak.flatten(ak.broadcast_arrays(weights_, mu_probes_.pt)[0]),
    #         )

    def eff_TripleMu_5_3_3_Mass3p8to60_DZ(
        self, events, muons, mu_probes, weights, output, dataset
    ):
        events_, muons_, mu_probes_, weights_ = self.mass_cut(
            events,
            muons,
            mu_probes,
            weights,
            mass_min=3.8,
            mass_max=60.0,
            dz_min=0.1,
            n_pairs=3,
        )

        if not self.isMC:
            # make sure trigger path was on
            run_cutoff = (events_.run >= 302509) & (events_.run <= 315973)  # type: ignore[name-defined]
            events_ = events_[run_cutoff]
            muons_ = muons_[run_cutoff]
            mu_probes_ = mu_probes_[run_cutoff]
            weights_ = weights_[run_cutoff]

        if len(weights_) == 0:
            return

        output[dataset]["histograms"]["DEN_HLT_TripleMu_5_3_3_Mass3p8to60_DZ"].fill(
            ak.flatten(mu_probes_.pt),
            weight=ak.flatten(ak.broadcast_arrays(weights_, mu_probes_.pt)[0]),
        )
        mu_probes_ = mu_probes_[events_.HLT.TripleMu_5_3_3_Mass3p8to60_DZ]
        weights_ = weights_[events_.HLT.TripleMu_5_3_3_Mass3p8to60_DZ]
        if len(weights_) > 0:
            output[dataset]["histograms"]["NUM_HLT_TripleMu_5_3_3_Mass3p8to60_DZ"].fill(
                ak.flatten(mu_probes_.pt),
                weight=ak.flatten(ak.broadcast_arrays(weights_, mu_probes_.pt)[0]),
            )

    def eff_TripleMu_5_3_3_Mass3p8_DZ(
        self, events, muons, mu_probes, weights, output, dataset
    ):
        events_, muons_, mu_probes_, weights_ = self.mass_cut(
            events,
            muons,
            mu_probes,
            weights,
            mass_min=3.8,
            dz_min=0.1,
            n_pairs=3,
        )

        if not self.isMC:
            # make sure trigger path was on
            run_cutoff = events_.run >= 315974  # type: ignore[name-defined]
            events_ = events_[run_cutoff]
            muons_ = muons_[run_cutoff]
            mu_probes_ = mu_probes_[run_cutoff]
            weights_ = weights_[run_cutoff]
        if len(weights_) == 0:
            return

        output[dataset]["histograms"]["DEN_HLT_TripleMu_5_3_3_Mass3p8_DZ"].fill(
            ak.flatten(mu_probes_.pt),
            weight=ak.flatten(ak.broadcast_arrays(weights_, mu_probes_.pt)[0]),
        )
        mu_probes_ = mu_probes_[events_.HLT.TripleMu_5_3_3_Mass3p8_DZ]
        weights_ = weights_[events_.HLT.TripleMu_5_3_3_Mass3p8_DZ]
        if len(weights_) > 0:
            output[dataset]["histograms"]["NUM_HLT_TripleMu_5_3_3_Mass3p8_DZ"].fill(
                ak.flatten(mu_probes_.pt),
                weight=ak.flatten(ak.broadcast_arrays(weights_, mu_probes_.pt)[0]),
            )

    def eff_TripleMu_10_5_5_DZ(self, events, mu_probes, weights, output, dataset):
        if len(weights) == 0:
            return

        output[dataset]["histograms"]["DEN_HLT_TripleMu_10_5_5_DZ"].fill(
            ak.flatten(mu_probes.pt),
            weight=ak.flatten(ak.broadcast_arrays(weights, mu_probes.pt)[0]),
        )
        mu_probes_ = mu_probes[events.HLT.TripleMu_10_5_5_DZ]
        weights_ = weights[events.HLT.TripleMu_10_5_5_DZ]
        if len(weights_) > 0:
            output[dataset]["histograms"]["NUM_HLT_TripleMu_10_5_5_DZ"].fill(
                ak.flatten(mu_probes_.pt),
                weight=ak.flatten(ak.broadcast_arrays(weights_, mu_probes_.pt)[0]),
            )

    def eff_TripleMu_12_10_5(self, events, muons, mu_probes, weights, output, dataset):
        if len(weights) == 0:
            return

        events_ = events[muons.pt[:, 1] > 10]
        mu_probes_ = mu_probes[muons.pt[:, 1] > 10]
        weights_ = weights[muons.pt[:, 1] > 10]

        if len(weights_) == 0:
            return

        output[dataset]["histograms"]["DEN_HLT_TripleMu_12_10_5"].fill(
            ak.flatten(mu_probes_.pt),
            weight=ak.flatten(ak.broadcast_arrays(weights_, mu_probes_.pt)[0]),
        )
        mu_probes_ = mu_probes_[events_.HLT.TripleMu_12_10_5]
        weights_ = weights_[events_.HLT.TripleMu_12_10_5]
        if len(weights_) > 0:
            output[dataset]["histograms"]["NUM_HLT_TripleMu_12_10_5"].fill(
                ak.flatten(mu_probes_.pt),
                weight=ak.flatten(ak.broadcast_arrays(weights_, mu_probes_.pt)[0]),
            )

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        # Select events passing the dimuon trigger
        if "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8" in events.HLT.fields:
            events = events[events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8]
        elif ("Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8" in events.HLT.fields) and (
            self.era == "2017"
        ):
            events = events[events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8]
        elif ("Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ" in events.HLT.fields) and (
            self.era in ["2016APV", "2016"]
        ):
            events = events[events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ]
        else:
            raise RuntimeError("No suitable dimuon trigger found.")

        if len(events) == 0:
            return

        # Apply muon selection and trigger matching
        events, muons = self.muon_filter(events)
        weights = self.get_weights(
            events, do_vars=False, apply_lumi_factors=False
        ).weight()
        events, muons, mu_probes, weights = self.trigger_matching(
            events, muons, weights
        )

        # Compensate in MC for different lumi between dimuon paths in 2017
        if self.isMC and self.era == "2017":
            lumi_fraction = 36.7 / 41.5
            weights = ak.where(
                events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8
                & ~events.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8,
                weights * lumi_fraction,
                weights,
            )

        if self.era == "2016APV":
            if "TripleMu_5_3_3" in events.HLT.fields:
                self.eff_TripleMu_5_3_3(events, mu_probes, weights, output, dataset)
            if "TripleMu_12_10_5" in events.HLT.fields:
                self.eff_TripleMu_12_10_5(
                    events, muons, mu_probes, weights, output, dataset
                )
        if self.era == "2016":
            if "TripleMu_5_3_3" in events.HLT.fields:
                self.eff_TripleMu_5_3_3(events, mu_probes, weights, output, dataset)
            # if "TripleMu_5_3_3_DZ_Mass3p8" in events.HLT.fields:
            #     self.eff_TripleMu_5_3_3_DZ_Mass3p8(
            #         events, muons, mu_probes, weights, output, dataset
            #     )
            if "TripleMu_12_10_5" in events.HLT.fields:
                self.eff_TripleMu_12_10_5(
                    events, muons, mu_probes, weights, output, dataset
                )
        elif self.era == "2017":
            if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
                self.eff_TripleMu_5_3_3_Mass3p8to60_DZ(
                    events, muons, mu_probes, weights, output, dataset
                )
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                self.eff_TripleMu_10_5_5_DZ(events, mu_probes, weights, output, dataset)
            if "TripleMu_12_10_5" in events.HLT.fields:
                self.eff_TripleMu_12_10_5(
                    events, muons, mu_probes, weights, output, dataset
                )
        elif self.era == "2018":
            # if "TripleMu_5_3_3_Mass3p8to60_DZ" in events.HLT.fields:
            #     self.eff_TripleMu_5_3_3_Mass3p8to60_DZ(
            #         events, muons, mu_probes, weights, output, dataset
            #     )
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                self.eff_TripleMu_5_3_3_Mass3p8_DZ(
                    events, muons, mu_probes, weights, output, dataset
                )
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                self.eff_TripleMu_10_5_5_DZ(events, mu_probes, weights, output, dataset)
            if "TripleMu_12_10_5" in events.HLT.fields:
                self.eff_TripleMu_12_10_5(
                    events, muons, mu_probes, weights, output, dataset
                )
        elif self.era.startswith("202"):
            if "TripleMu_5_3_3_Mass3p8_DZ" in events.HLT.fields:
                self.eff_TripleMu_5_3_3_Mass3p8_DZ(
                    events, muons, mu_probes, weights, output, dataset
                )
            if "TripleMu_10_5_5_DZ" in events.HLT.fields:
                self.eff_TripleMu_10_5_5_DZ(events, mu_probes, weights, output, dataset)
            if "TripleMu_12_10_5" in events.HLT.fields:
                self.eff_TripleMu_12_10_5(
                    events, muons, mu_probes, weights, output, dataset
                )
        else:
            raise RuntimeError(f"Era {self.era} not recognized.")

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
        hlt_paths = []
        if self.era == "2016APV":
            hlt_paths = [
                "HLT_TripleMu_5_3_3",
                "HLT_TripleMu_12_10_5",
            ]
        elif self.era == "2016":
            hlt_paths = [
                "HLT_TripleMu_5_3_3",
                # "HLT_TripleMu_5_3_3_DZ_Mass3p8",
                "HLT_TripleMu_12_10_5",
            ]
        elif self.era == "2017":
            hlt_paths = [
                "HLT_TripleMu_5_3_3_Mass3p8to60_DZ",
                "HLT_TripleMu_10_5_5_DZ",
                "HLT_TripleMu_12_10_5",
            ]
        elif self.era == "2018":
            hlt_paths = [
                # "HLT_TripleMu_5_3_3_Mass3p8to60_DZ",
                "HLT_TripleMu_5_3_3_Mass3p8_DZ",
                "HLT_TripleMu_10_5_5_DZ",
                "HLT_TripleMu_12_10_5",
            ]
        elif self.era.startswith("202"):
            hlt_paths = [
                "HLT_TripleMu_5_3_3_Mass3p8_DZ",
                "HLT_TripleMu_10_5_5_DZ",
                "HLT_TripleMu_12_10_5",
            ]
        else:
            raise RuntimeError(f"Era {self.era} not recognized.")
        histograms = {}
        for path in hlt_paths:
            histograms[f"NUM_{path}"] = hist.Hist.new.Regular(
                100, 0, 20, name="muon_pt", label="muon_pt"
            ).Weight()
            histograms[f"DEN_{path}"] = hist.Hist.new.Regular(
                100, 0, 20, name="muon_pt", label="muon_pt"
            ).Weight()

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
