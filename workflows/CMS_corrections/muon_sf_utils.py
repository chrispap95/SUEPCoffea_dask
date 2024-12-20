import awkward as ak
import correctionlib  # type: ignore[import]
import numpy as np
from coffea.lookup_tools import rochester_lookup, txt_converters


def muon_efficiencies(muons, syst=""):
    var = "nominal"
    if syst == "up":
        var = "systup"
    elif syst == "down":
        var = "systdown"

    muons_flat = ak.flatten(muons)
    n_muons = ak.num(muons)

    # medium pt muons
    low_pt_json_file = "data/muon_corrections/low_pt_muons/muon_JPsi.json"
    low_pt_corrs = correctionlib.CorrectionSet.from_file(low_pt_json_file)
    low_pt_muon_corr_id = low_pt_corrs["NUM_MediumID_DEN_TrackerMuons"]
    low_pt_muon_corr_eff = low_pt_corrs["NUM_TrackerMuons_DEN_genTracks"]

    low_pt_muon_id = low_pt_muon_corr_id.evaluate(muons_flat.eta, muons_flat.pt, var)
    low_pt_muon_eff = low_pt_muon_corr_eff.evaluate(muons_flat.eta, muons_flat.pt, var)

    # medium pt muons
    medium_pt_json_file = "data/muon_corrections/medium_pt_muons/muon_Z.json"
    medium_pt_corrs = correctionlib.CorrectionSet.from_file(medium_pt_json_file)
    medium_pt_muon_corr_id = medium_pt_corrs["NUM_MediumID_DEN_TrackerMuons"]
    medium_pt_muon_corr_eff = medium_pt_corrs["NUM_TrackerMuons_DEN_genTracks"]
    # muon_corr_iso = correctionlib.CorrectionSet.from_file(json_file)["NUM_LooseRelIso_DEN_LooseID"]

    medium_pt_muon_id = medium_pt_muon_corr_id.evaluate(muons_flat.eta, 50.0, var)
    medium_pt_muon_eff = medium_pt_muon_corr_eff.evaluate(muons_flat.eta, 50.0, var)

    # Combine low and medium pt SFs
    sf_muon_id = np.where(muons_flat.pt > 15, medium_pt_muon_id, low_pt_muon_id)
    sf_muon_eff = np.where(muons_flat.pt > 15, medium_pt_muon_eff, low_pt_muon_eff)

    muon_SF = sf_muon_id * sf_muon_eff

    return ak.unflatten(muon_SF, n_muons)


def muon_scale_factors(events, muons, era, is_mc=False, var="nominal"):
    """
    Apply Rochester corrections to muons.
    Use kSpreadMC (or kSmearMC if gen muon is not available) for MC and kScaleDT for data.
    Change the muon pt field in the muons array in place. This will propagate to all other
    muon quantities that depend on pt. (e.g. px, py, pz, energy, etc.)
    """
    if era == "2016APV":
        era = "2016a"
    elif era == "2016":
        era = "2016b"
    rochester_file = f"data/muon_corrections/roccor.Run2.v5/RoccoR{era}UL.txt"
    rochester_data = txt_converters.convert_rochester_file(
        rochester_file, loaduncs=True
    )
    rochester = rochester_lookup.rochester_lookup(rochester_data)

    if not is_mc:
        scale_DT = rochester.kScaleDT(muons.charge, muons.pt, muons.eta, muons.phi)
        scale_DT_error = rochester.kScaleDTerror(
            muons.charge, muons.pt, muons.eta, muons.phi
        )

        if var == "nominal":
            muons["pt"] = muons.pt * scale_DT
        elif var == "up":
            muons["pt"] = muons.pt * (scale_DT + scale_DT_error)
        elif var == "down":
            muons["pt"] = muons.pt * (scale_DT - scale_DT_error)
        else:
            raise ValueError(f"Invalid variation {var}")
        return muons

    # Use kSpreadMC where genPt is available, otherwise use kSmearMC
    muon_gen_pt = events.GenPart.pt[ak.mask(muons, muons.genPartIdx >= 0).genPartIdx]
    spread_MC = rochester.kSpreadMC(
        muons.charge, muons.pt, muons.eta, muons.phi, muon_gen_pt
    )
    random_num_per_muon = ak.unflatten(
        np.random.random(ak.sum(ak.num(muons))), ak.num(muons)
    )
    smear_MC = rochester.kSmearMC(
        muons.charge,
        muons.pt,
        muons.eta,
        muons.phi,
        muons.nTrackerLayers,
        random_num_per_muon,
    )
    mu_SF = ak.where(muons.genPartIdx >= 0, spread_MC, smear_MC)

    # Same for uncertainty
    spread_MC_error = rochester.kSpreadMCerror(
        muons.charge, muons.pt, muons.eta, muons.phi, muon_gen_pt
    )
    smear_MC_error = rochester.kSmearMCerror(
        muons.charge,
        muons.pt,
        muons.eta,
        muons.phi,
        muons.nTrackerLayers,
        random_num_per_muon,
    )
    mu_SF_error = ak.where(muons.genPartIdx >= 0, spread_MC_error, smear_MC_error)

    if var == "nominal":
        muons["pt"] = muons.pt * mu_SF
    elif var == "up":
        muons["pt"] = muons.pt * (mu_SF + mu_SF_error)
    elif var == "down":
        muons["pt"] = muons.pt * (mu_SF - mu_SF_error)  # type: ignore[assignment]
    else:
        raise ValueError(f"Invalid variation {var}")

    return muons
