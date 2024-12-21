import awkward as ak
import correctionlib
import numpy as np
import uproot


def pileup_weight(events, era, syst=""):
    """
    Function to get the pileup weights for a given era and systematic variation
    The pileup weights are calculated as the ratio of the data distribution to the MC distribution
    The data distribution is normalized to 1

    Parameters:
    era: str
        The year of the data taking
    nTrueInt: array
        The number of true interactions
    sys: str
        The systematic variation to be applied to the pileup weights

    Returns:
    weights: array
        The pileup weights
    """
    if era == "2016APV":
        era = "2016"
    if era not in ["2016", "2017", "2018"]:
        raise ValueError(
            "no pileup weights because no year was selected for function pileup_weight"
        )

    f_MC = uproot.open(f"data/pileup/mcPileupUL{era}.root")
    f_data = uproot.open(f"data/pileup/PileupHistogram-UL{era}-100bins_withVar.root")

    variation = ""
    if "up" in syst:
        variation = "_plus"
    elif "down" in syst:
        variation = "_minus"

    hist_MC = f_MC["pu_mc"].to_numpy()  # type: ignore[no-untyped-call]
    hist_data = f_data["pileup" + variation].to_numpy()  # type: ignore[no-untyped-call]
    hist_data[0].sum()
    norm_data = hist_data[0] / hist_data[0].sum()
    weights = np.divide(
        norm_data, hist_MC[0], out=np.ones_like(norm_data), where=hist_MC[0] != 0
    )

    nTrueInt = ak.values_astype(events.Pileup.nTrueInt, np.int32)

    return weights[nTrueInt]


def get_PS_weights(events, syst):
    """
    Get the parton shower variation weights. Available options are:
        - ISR_up
        - ISR_down
        - FSR_up
        - FSR_down
    """
    PSWeights = np.ones(len(events))
    if len(events.PSWeight[0]) > 3:
        if syst == "ISR_up":
            PSWeights = events.PSWeight[:, 0]
        elif syst == "ISR_down":
            PSWeights = events.PSWeight[:, 2]
        elif syst == "FSR_up":
            PSWeights = events.PSWeight[:, 1]
        elif syst == "FSR_down":
            PSWeights = events.PSWeight[:, 3]
        else:
            raise RuntimeError(f"Unknown PSWeight systematic: {syst}")
    return PSWeights


def get_pdf_variations(events, syst):
    """
    Get the matrix element PDF variations. Only available if there is LHE info.
    """
    pdf_vars = np.ones(len(events))
    if "LHEPdfWeight" not in events.fields:
        return pdf_vars
    if len(events.LHEPdfWeight[0]) == 0:
        return pdf_vars
    if syst == "up":
        pdf_vars = 1 + ak.std(events.LHEPdfWeight, axis=1) / ak.mean(
            events.LHEPdfWeight, axis=1
        )
    elif syst == "down":
        pdf_vars = 1 - ak.std(events.LHEPdfWeight, axis=1) / ak.mean(
            events.LHEPdfWeight, axis=1
        )
    else:
        raise RuntimeError(f"Unknown PDF systematic: {syst}")
    return pdf_vars


def get_scale_variations(events, syst=""):
    """
    Get the variations for scale of renormalization, mu_R, and scale of factorization, mu_F.
    Only available if there is LHE info. Up variation is 2x the nominal value, down is 0.5x.
    Available options are:
        - MuRUp
        - MuRDown
        - MuFUp
        - MuFDown
    """
    pdf_vars = np.ones(len(events))
    if "LHEScaleWeight" not in events.fields:
        return pdf_vars
    if any(ak.num(events.LHEScaleWeight) > 8):
        ones = ak.from_numpy(np.ones((len(events), 9)))
        LHEScaleWeight = ak.where(
            ak.num(events.LHEScaleWeight) == 9, events.LHEScaleWeight, ones
        )
        if syst == "MuRUp":
            pdf_vars = LHEScaleWeight[:, 7]  # type: ignore[index]
        elif syst == "MuRDown":
            pdf_vars = LHEScaleWeight[:, 1]  # type: ignore[index]
        elif syst == "MuFUp":
            pdf_vars = LHEScaleWeight[:, 5]  # type: ignore[index]
        elif syst == "MuFDown":
            pdf_vars = LHEScaleWeight[:, 3]  # type: ignore[index]
        else:
            raise RuntimeError(f"Unknown scale variation systematic: {syst}")
    return pdf_vars


def track_killing(tracks, era):
    """
    Drop 2.7%, 2.2%, and 2.1% of the tracks randomly at reco-level
    for charged-particles with 1 < pT < 20 GeV in simulation for 2016, 2017, and
    2018, respectively when reclustering the constituents.
     For charged-particles with pT > 20 GeV, 1% of the tracks are dropped randomly
    """

    year_percent = {"2018": 0.021, "2017": 0.022, "2016": 0.027}
    block1_percent = year_percent[str(era)]
    block2_percent = 0.01

    block1_indices = (tracks.pt > 1) & (tracks.pt < 20)
    block2_indices = tracks.pt >= 20

    new_indices = []
    for i in range(len(tracks)):
        event_indices = np.arange(len(tracks[i]))
        event_bool = np.array([True] * len(tracks[i]))

        block1_event_indices = event_indices[block1_indices[i]]
        block1_event_indices_drop = np.random.choice(
            block1_event_indices, int((block1_percent) * len(block1_event_indices))
        )
        event_bool[block1_event_indices_drop] = False

        block2_event_indices = event_indices[block2_indices[i]]
        block2_event_indices_drop = np.random.choice(
            block2_event_indices, int((block2_percent) * len(block2_event_indices))
        )
        event_bool[block2_event_indices_drop] = False

        new_indices.append(list(event_bool))

    new_indices = ak.Array(new_indices)
    tracks = tracks[new_indices]
    return tracks


def higgs_reweight(higgs_pt, variation="nominal"):
    json_file = "data/higgs_reweight/higgs_reweight.json"
    higgs_reweight_corrset = correctionlib.CorrectionSet.from_file(json_file)
    higgs_pt_reweight_corr = higgs_reweight_corrset["Higgs_pt_reweighting"]
    return higgs_pt_reweight_corr.evaluate(higgs_pt, variation)
