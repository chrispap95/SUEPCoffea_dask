import awkward as ak
import correctionlib  # type: ignore[import]
import numpy as np
from coffea.lookup_tools import rochester_lookup, txt_converters


def muon_efficiencies(
    muons, era, region, syst, override_low_pt_bound=False, apply_IP_corr=True
):
    """
    This will return the total muon scale factors. The muon scale factors are the product of
    muon RECO efficiency, muon ID efficiency, and muon ISO efficiency. Following the MUO POG
    recommendations:
        - For RECO efficiency, use the JPsi derived scale factors for muons with pt < 10 GeV
        and the Z derived scale factors for muons with pt > 10 GeV. It applies only to Run 2.
        - For ID efficiency, use the JPsi derived scale factors for Medium ID for most regions.
        For CR_prompt, use the Z derived scale factors for Medium ID since it contains medium
        pt muons.
        - For ISO efficiency, only Tight iso WP over MediumId for CR_prompt.

    RECO efficiencies:
        - Run 2:
            - JPsi up to 10 GeV
            - Z above 10 GeV
        - Run 3:
            - Not needed

    ID efficiencies:
        - Run 2:
            - JPsi for most regions
            - Z for CR_prompt
        - Run 3:
            - JPsi for most regions
            - Z for CR_prompt

    ISO efficiencies (TBD):
        - Tight iso WP over MediumId for CR_prompt

    Uncertainties:
        - Run 2:
            - Low pt:
                - RECO: stat only
                - ID: stat only
            - Medium pt:
                - RECO: stat + syst
                - ID: stat + syst
                - ISO: stat + syst
        - Run 3:
            - Low pt:
                - ID: stat + syst
            - Medium pt:
                - ID: stat + syst
                - ISO: stat + syst

    The muon scale factors are calculated for medium pt muons and low pt muons separately.

    Parameters
    ----------
    muons : awkward array
        Muon collection
    era : str
        Era of the data taking. Can be 2016, 2016APV, 2017, 2018, 2022, 2022EE, 2023, 2023BPix
    syst : str
        Systematic variation. Can be "up", "down", or "" (default) for nominal.
    region : str
        Region of the analysis. E.g., "CR_prompt_prompt", "CR_prompt_qcd", "CR_cb".

    Returns
    -------
    muon_SF : awkward array
        Muon scale factors. The shape is the same as the muon collection.
    """
    # NOTE: systup and systdown have the stat + syst uncertainties added in quadrature
    # In the case of Run 2 JPsi efficiencies, the systup and systdown are only the stat uncertainties)
    var = "nominal"
    if syst == "up":
        var = "systup"
    elif syst == "down":
        var = "systdown"

    print(var, era, region)

    muons_flat = ak.flatten(muons)
    n_muons = ak.num(muons)

    # Choose the peak for TnP
    peak = "JPsi"
    if region == "CR_prompt_prompt":
        peak = "Z"

    config = []

    # Add RECO efficiency correction only for Run 2
    if era.startswith("201"):
        config += ["NUM_TrackerMuons_DEN_genTracks"]

    # All muons have medium ID efficiency
    config += ["NUM_MediumID_DEN_TrackerMuons"]

    # Add all other corrections for each region
    match region:
        case "CR_prompt_prompt":
            config += ["NUM_TightMiniIso_DEN_MediumID"]
            if apply_IP_corr:
                config += ["NUM_dxyLT0p01_AND_dzLT0p01_DEN_MediumID"]
        case "CR_prompt_qcd":
            config += ["NUM_MiniIsoGT0p1_DEN_MediumID"]
        case "CR_cb":
            config += []
        case "VR_loose":
            config += ["NUM_MiniIsoGT0p2_DEN_MediumID"]
        case "VR_tight":
            config += ["NUM_MiniIsoGT0p4_DEN_MediumID"]
        case "SR_low_temp_loose":
            if apply_IP_corr:
                config += ["NUM_dxyLT0p1_AND_dzLT0p1_DEN_MediumID"]
        case "SR_low_temp_tight":
            if apply_IP_corr:
                config += ["NUM_dxyLT0p007_AND_dzLT0p007_DEN_MediumID"]
        case "SR_high_temp_loose":
            config += ["NUM_MiniIsoLT0p65_DEN_MediumID"]
            if apply_IP_corr:
                config += ["NUM_dxyLT0p1_AND_dzLT0p1_DEN_MediumID"]
        case "SR_high_temp_tight":
            if apply_IP_corr:
                config += ["NUM_dxyLT0p007_AND_dzLT0p007_DEN_MediumID"]

    json_file_JPsi = f"data/muon_corrections/{era}/muon_JPsi.json"
    corrs_JPsi = correctionlib.CorrectionSet.from_file(json_file_JPsi)

    json_file_Z = f"data/muon_corrections/{era}/muon_Z.json"
    corrs_Z = correctionlib.CorrectionSet.from_file(json_file_Z)

    era_tag = era + "UL" if era.startswith("201") else era
    custom_corrs_file = (
        f"data/muon_corrections/custom_scale_factors/Z_Run{era_tag}_schemaV2.json"
    )
    custom_corrs = correctionlib.CorrectionSet.from_file(custom_corrs_file)

    muon_SF = np.ones_like(muons_flat.pt)

    if var == "nominal":
        print(f"muons.pt: {muons_flat.pt}")
    # Evaluate RECO efficiency
    # Only for Run 2. Muons with pt < 10 GeV use JPsi corrections.
    if "NUM_TrackerMuons_DEN_genTracks" in config:
        low_pt_muon_corr_eff = corrs_JPsi["NUM_TrackerMuons_DEN_genTracks"]
        low_pt_muon_eff = low_pt_muon_corr_eff.evaluate(
            muons_flat.eta, muons_flat.pt, var
        )

        medium_pt_muon_corr_eff = corrs_Z["NUM_TrackerMuons_DEN_genTracks"]
        medium_pt_muon_eff = medium_pt_muon_corr_eff.evaluate(muons_flat.eta, 50.0, var)
        muon_SF = muon_SF * np.where(
            muons_flat.pt > 10, medium_pt_muon_eff, low_pt_muon_eff
        )
        if var == "nominal":
            print(f"tracking eff: {muon_SF}")

    # Evaluate medium ID efficiency
    if peak == "JPsi":
        muon_corr_id = corrs_JPsi["NUM_MediumID_DEN_TrackerMuons"]
        muon_SF = muon_SF * muon_corr_id.evaluate(muons_flat.eta, muons_flat.pt, var)
        if var == "nominal":
            print(
                f"medium eff JPsi: {muon_corr_id.evaluate(muons_flat.eta, muons_flat.pt, var)}"
            )
    elif peak == "Z":
        muon_corr_id = corrs_Z["NUM_MediumID_DEN_TrackerMuons"]
        muon_SF = muon_SF * muon_corr_id.evaluate(muons_flat.eta, 50.0, var)
        if var == "nominal":
            print(f"medium eff Z: {muon_corr_id.evaluate(muons_flat.eta, 50.0, var)}")

    # Evaluate all other efficiencies. They are all in the custom corrections file.
    for corr_name in config:
        if corr_name in [
            "NUM_TrackerMuons_DEN_genTracks",
            "NUM_MediumID_DEN_TrackerMuons",
        ]:
            continue
        corr = custom_corrs[corr_name]
        muon_pt_vals = muons_flat.pt
        if override_low_pt_bound:
            muon_pt_vals = np.where(muon_pt_vals < 10, 10, muon_pt_vals)
        muon_SF = muon_SF * corr.evaluate(muon_pt_vals, var)
        if var == "nominal":
            print(f"{corr_name}: {corr.evaluate(muon_pt_vals, var)}")

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
