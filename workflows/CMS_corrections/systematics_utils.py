import awkward as ak
import numpy as np
import uproot


def pileup_weight(era, nTrueInt, sys=""):
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
    if "PU_reweight_up" in sys:
        variation = "_plus"
    elif "PU_reweight_down" in sys:
        variation = "_minus"

    hist_MC = f_MC["pu_mc"].to_numpy()  # type: ignore[no-untyped-call]
    hist_data = f_data["pileup" + variation].to_numpy()  # type: ignore[no-untyped-call]
    hist_data[0].sum()
    norm_data = hist_data[0] / hist_data[0].sum()
    weights = np.divide(
        norm_data, hist_MC[0], out=np.ones_like(norm_data), where=hist_MC[0] != 0
    )

    return weights[nTrueInt]


def get_prefire_weights(events, syst=""):
    if syst == "L1Prefire_up":
        return events.L1PreFiringWeight.Up
    if syst == "L1Prefire_down":
        return events.L1PreFiringWeight.Dn
    return events.L1PreFiringWeight.Nom


def get_PS_weights(events, syst):
    if syst == "ISR_up":
        return events.PSWeight[:, 0]
    elif syst == "ISR_down":
        return events.PSWeight[:, 2]
    elif syst == "FSR_up":
        return events.PSWeight[:, 1]
    elif syst == "FSR_down":
        return events.PSWeight[:, 3]
    else:
        raise RuntimeError(f"Unknown PSWeight systematic: {syst}")


def track_killing(tracks, era, scouting=False):
    """
    Drop 2.7%, 2.2%, and 2.1% of the tracks randomly at reco-level
    for charged-particles with 1 < pT < 20 GeV in simulation for 2016, 2017, and
    2018, respectively when reclustering the constituents.
     For charged-particles with pT > 20 GeV, 1% of the tracks are dropped randomly
    """

    if scouting:
        block1_percent = 0.05
        block2_percent = 0.01
    else:
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
