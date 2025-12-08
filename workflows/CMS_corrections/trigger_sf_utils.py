import awkward as ak
import numpy as np


def trigger_scale_factors(events, era):
    """
    Returns trigger scale factors and their variations based on the era.
    Trigger SFs are stored for each era and each HLT path.
    Additionally, weighted averages of the three paths and the two higher pt
    paths are given as 'w_av_3' and 'w_av_2'. The spread between the paths
    is considered as the dominant uncertainty in the w_av_ SFs.

    Current implementation assigns SFs for 5_3_3 to events with third muon
    pt < 5 GeV and SFs for w_av_2 to events with third muon pt > 5 GeV.

    Parameters:
        events: coffea events object
        era: str - The data-taking year (e.g., "2016", etc.)

    Returns:
        sf_nom: array - Nominal scale factors
        sf_up: array - Scale factors with upward variation
        sf_down: array - Scale factors with downward variation
    """
    trig_sfs = {
        # year 5_3_3 10_5_5 12_10_5 w_av_2 w_av_3
        "2016": {
            "5_3_3": {"val": 0.922, "unc": 0.006},
            "12_10_5": {"val": 0.937, "unc": 0.003},
            "w_av_2": {"val": 0.937, "unc": 0.003, "spread": 0.000},
            "w_av_3": {"val": 0.934, "unc": 0.003, "spread": 0.012},
        },
        "2017": {
            "5_3_3": {"val": 1.001, "unc": 0.007},
            "10_5_5": {"val": 0.992, "unc": 0.003},
            "12_10_5": {"val": 0.985, "unc": 0.002},
            "w_av_2": {"val": 0.989, "unc": 0.002, "spread": 0.004},
            "w_av_3": {"val": 0.989, "unc": 0.002, "spread": 0.012},
        },
        "2018": {
            "5_3_3": {"val": 0.957, "unc": 0.004},
            "10_5_5": {"val": 0.964, "unc": 0.002},
            "12_10_5": {"val": 0.970, "unc": 0.003},
            "w_av_2": {"val": 0.966, "unc": 0.002, "spread": 0.004},
            "w_av_3": {"val": 0.965, "unc": 0.002, "spread": 0.007},
        },
        "2023BPix": {
            "5_3_3": {"val": 0.904, "unc": 0.003},
            "10_5_5": {"val": 0.946, "unc": 0.003},
            "12_10_5": {"val": 0.942, "unc": 0.003},
            "w_av_2": {"val": 0.944, "unc": 0.002, "spread": 0.002},
            "w_av_3": {"val": 0.932, "unc": 0.002, "spread": 0.027},
        },
        "2022": {
            "5_3_3": {"val": 0.943, "unc": 0.009},
            "10_5_5": {"val": 0.956, "unc": 0.005},
            "12_10_5": {"val": 0.964, "unc": 0.005},
            "w_av_2": {"val": 0.960, "unc": 0.004, "spread": 0.004},
            "w_av_3": {"val": 0.957, "unc": 0.003, "spread": 0.014},
        },
        "2022EE": {
            "5_3_3": {"val": 0.902, "unc": 0.002},
            "10_5_5": {"val": 0.948, "unc": 0.002},
            "12_10_5": {"val": 0.947, "unc": 0.002},
            "w_av_2": {"val": 0.948, "unc": 0.002, "spread": 0.001},
            "w_av_3": {"val": 0.931, "unc": 0.001, "spread": 0.029},
        },
        "2023": {
            "5_3_3": {"val": 0.901, "unc": 0.003},
            "10_5_5": {"val": 0.939, "unc": 0.001},
            "12_10_5": {"val": 0.940, "unc": 0.002},
            "w_av_2": {"val": 0.939, "unc": 0.001, "spread": 0.001},
            "w_av_3": {"val": 0.934, "unc": 0.001, "spread": 0.033},
        },
    }

    sf_lowpt = trig_sfs[era]["5_3_3"]["val"] * np.ones(len(events), dtype=float)
    sf_lowpt_unc = trig_sfs[era]["5_3_3"]["unc"] * np.ones(len(events), dtype=float)
    sf_lowpt_up = sf_lowpt + sf_lowpt_unc
    sf_lowpt_down = sf_lowpt - sf_lowpt_unc
    sf_highpt = trig_sfs[era]["w_av_2"]["val"] * np.ones(len(events), dtype=float)
    sf_highpt_unc = max(
        trig_sfs[era]["w_av_2"]["unc"], trig_sfs[era]["w_av_2"]["spread"]
    ) * np.ones(len(events), dtype=float)
    sf_highpt_up = sf_highpt + sf_highpt_unc
    sf_highpt_down = sf_highpt - sf_highpt_unc

    muons = events.Muon
    muon_cleaning = (
        (muons.mediumId)
        & (muons.pt > 3)
        & (abs(muons.eta) < 2.4)
        & (abs(muons.dz) < 0.2)
    )
    muons = muons[muon_cleaning]

    final_SF = ak.where(
        ak.sum(muons.pt >= 5, axis=-1) >= 3,
        sf_highpt,
        sf_lowpt,
    )
    final_SF_up = ak.where(
        ak.sum(muons.pt >= 5, axis=-1) >= 3,
        sf_highpt_up,
        sf_lowpt_up,
    )
    final_SF_down = ak.where(
        ak.sum(muons.pt >= 5, axis=-1) >= 3,
        sf_highpt_down,
        sf_lowpt_down,
    )

    return final_SF, final_SF_up, final_SF_down
