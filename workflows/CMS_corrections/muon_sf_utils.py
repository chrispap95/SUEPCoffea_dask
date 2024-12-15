from pathlib import Path

import awkward as ak
import correctionlib  # type: ignore[import]
import numpy as np


def muon_scale_factors(muons, syst=""):
    var = "nominal"
    if syst == "up":
        var = "systup"
    elif syst == "down":
        var = "systdown"

    muons_flat = ak.flatten(muons)
    n_muons = ak.num(muons)

    # medium pt muons
    low_pt_json_file = (
        Path(__file__).parent.parent.parent
        / "data/muon_corrections/low_pt_muons/muon_JPsi.json"
    )
    low_pt_corrs = correctionlib.CorrectionSet.from_file(str(low_pt_json_file))
    low_pt_muon_corr_id = low_pt_corrs["NUM_MediumID_DEN_TrackerMuons"]
    low_pt_muon_corr_eff = low_pt_corrs["NUM_TrackerMuons_DEN_genTracks"]

    sf_low_pt_muon_id = low_pt_muon_corr_id.evaluate(muons_flat.eta, muons_flat.pt, var)
    sf_low_pt_muon_eff = low_pt_muon_corr_eff.evaluate(
        muons_flat.eta, muons_flat.pt, var
    )

    # medium pt muons
    medium_pt_json_file = (
        Path(__file__).parent.parent.parent
        / "data/muon_corrections/medium_pt_muons/muon_Z.json"
    )
    medium_pt_corrs = correctionlib.CorrectionSet.from_file(str(medium_pt_json_file))
    medium_pt_muon_corr_id = medium_pt_corrs["NUM_MediumID_DEN_TrackerMuons"]
    medium_pt_muon_corr_eff = medium_pt_corrs["NUM_TrackerMuons_DEN_genTracks"]
    # muon_corr_iso = correctionlib.CorrectionSet.from_file(json_file)["NUM_LooseRelIso_DEN_LooseID"]

    sf_medium_pt_muon_id = medium_pt_muon_corr_id.evaluate(muons_flat.eta, 50.0, var)
    sf_medium_pt_muon_eff = medium_pt_muon_corr_eff.evaluate(muons_flat.eta, 50.0, var)

    # Combine low and medium pt SFs
    sf_muon_id = np.where(muons_flat.pt > 15, sf_medium_pt_muon_id, sf_low_pt_muon_id)
    sf_muon_eff = np.where(
        muons_flat.pt > 15, sf_medium_pt_muon_eff, sf_low_pt_muon_eff
    )

    muon_SF = sf_muon_id * sf_muon_eff

    return ak.unflatten(muon_SF, n_muons)
