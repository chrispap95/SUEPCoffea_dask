import awkward as ak
import correctionlib
import numpy as np
import pythia8  # type: ignore[import]
import uproot

import parton


def pileup_weight(events, era, syst=""):
    """
    Function to get the pileup weights for a given era and systematic variation
    The pileup weights are calculated as the ratio of the data distribution to the MC distribution
    The data distribution is normalized to 1

    Reference: https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData

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
    if era == "2016" or era == "2016APV":
        mc_filename = "mc_pileup_UL16.root"
        data_filename = "PileupHistogram-goldenJSON-13tev-2016-withvars-99bins.root"
    elif era == "2017":
        mc_filename = "mc_pileup_UL17.root"
        data_filename = "PileupHistogram-goldenJSON-13tev-2017-withvars-99bins.root"
    elif era == "2018":
        mc_filename = "mc_pileup_UL18.root"
        data_filename = "PileupHistogram-goldenJSON-13tev-2018-withvars-99bins.root"
    elif era == "2022":
        mc_filename = "mc_pileup_2022.root"
        data_filename = "pileupHistogram-Cert_Collisions2022_355100_357900_eraBCD_GoldenJson-13p6TeV-2022-withvars-99bins.root"
    elif era == "2022EE":
        mc_filename = "mc_pileup_2022EE.root"
        data_filename = "pileupHistogram-Cert_Collisions2022_359022_362760_eraEFG_GoldenJson-13p6TeV-2022EE-withvars-99bins.root"
    elif era == "2023":
        mc_filename = "mc_pileup_2023.root"
        data_filename = "pileupHistogram-Cert_Collisions2023_366403_369802_eraBC_GoldenJson-13p6TeV-2023-withvars-99bins.root"
    elif era == "2023BPix":
        mc_filename = "mc_pileup_2023BPix.root"
        data_filename = "pileupHistogram-Cert_Collisions2023_369803_370790_eraD_GoldenJson-13p6TeV-2023BPix-withvars-99bins.root"
    else:
        raise ValueError(
            "no pileup weights because no year was selected for function pileup_weight"
        )

    f_mc = uproot.open(f"data/pileup/{mc_filename}")
    f_data = uproot.open(f"data/pileup/{data_filename}")

    variation = ""
    if "up" in syst:
        variation = "_up"
    elif "down" in syst:
        variation = "_down"

    hist_mc = f_mc["mc_pileup"].to_numpy()  # type: ignore[no-untyped-call]
    hist_data = f_data["pileup" + variation].to_numpy()  # type: ignore[no-untyped-call]
    normed_mc = hist_mc[0] / hist_mc[0].sum()
    normed_data = hist_data[0] / hist_data[0].sum()
    weights = np.divide(
        normed_data, normed_mc, out=np.ones_like(normed_data), where=normed_mc != 0
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


def manual_pdf_variations(events):
    """
    Get the matrix element PDF variations manually by evaluating all the PDF
    replicas for the given x, id, Q values for the two partons.
    """
    Q = events.Generator.scalePDF
    id1 = np.where(abs(events.Generator.id1) == 21, 0, events.Generator.id1)
    id2 = np.where(abs(events.Generator.id2) == 21, 0, events.Generator.id2)
    x1 = events.Generator.x1
    x2 = events.Generator.x2
    pdfweights = []
    for i in range(103):
        pdf = parton.mkPDF("NNPDF31_nnlo_as_0118_mc_hessian_pdfas", i)
        newpdf1 = pdf.xfxQ(id1, x1, Q, grid=False) / x1
        newpdf2 = pdf.xfxQ(id2, x2, Q, grid=False) / x2
        pdfweights.append(newpdf1 * newpdf2)
    pdfweights = np.array(pdfweights).T
    return pdfweights / pdfweights[:, 0][:, np.newaxis]


def get_pdf_variations(events, allow_manual=True):
    """
    Get the matrix element PDF variations. Only available if there is LHE info.
    If the LHEPdfWeight is not available, the variations are calculated manually.
    This behavior can be disabled by setting allow_manual to False.
    """
    if "LHEPdfWeight" in events.fields:
        if len(events.LHEPdfWeight[0]) > 0:
            mean = np.mean(events.LHEPdfWeight, axis=1)
            std = np.std(events.LHEPdfWeight, axis=1)
            if not all(ak.mean(events.LHEPdfWeight, axis=-1) > 0):
                if allow_manual:
                    pdf_weights = manual_pdf_variations(events)
                    mean_alt = np.mean(pdf_weights, axis=1)
                    std_alt = np.std(pdf_weights, axis=1)
                else:
                    mean_alt, std_alt = 1, 0
                mean = np.where(
                    ak.mean(events.LHEPdfWeight, axis=-1) > 0, mean, mean_alt
                )
                std = np.where(ak.mean(events.LHEPdfWeight, axis=-1) > 0, std, std_alt)
        elif allow_manual:
            pdf_weights = manual_pdf_variations(events)
            mean = np.mean(pdf_weights, axis=1)
            std = np.std(pdf_weights, axis=1)
        else:
            return np.ones(len(events)), np.ones(len(events))
    elif allow_manual:
        pdf_weights = manual_pdf_variations(events)
        mean = np.mean(pdf_weights, axis=1)
        std = np.std(pdf_weights, axis=1)
    else:
        return np.ones(len(events)), np.ones(len(events))
    pdf_vars_up = 1 + std / mean
    pdf_vars_down = 1 - std / mean
    return pdf_vars_up, pdf_vars_down


def matrix_element_scale_variations(events, nEM=0, nQCD=0, kUp=2, kDn=0.5):
    """
    Function to calculate the matrix element scale variations
    """

    # Calculate muR
    pythia = pythia8.Pythia("pythia/pythia8313/share/Pythia8/xmldoc/", False)
    settings = pythia.settings
    # Initialize alphaS with value 0.118 at the Z mass
    alphaS = pythia8.AlphaStrong()
    alphaS.init(valueIn=0.118)
    # Initialize alphaEM with 1st order running
    alphaEM = pythia8.AlphaEM()
    alphaEM.init(1, settings)
    alpS = np.array([alphaS.alphaS(Q**2) for Q in events.Generator.scalePDF])
    alpSup = np.array([alphaS.alphaS(kUp**2 * Q**2) for Q in events.Generator.scalePDF])
    alpSdn = np.array([alphaS.alphaS(kDn**2 * Q**2) for Q in events.Generator.scalePDF])
    alpEM = np.array([alphaEM.alphaEM(Q**2) for Q in events.Generator.scalePDF])
    alpEMup = np.array(
        [alphaEM.alphaEM(kUp**2 * Q**2) for Q in events.Generator.scalePDF]
    )
    alpEMdn = np.array(
        [alphaEM.alphaEM(kDn**2 * Q**2) for Q in events.Generator.scalePDF]
    )
    weightRenUp = (alpEMup / alpEM) ** nEM * (alpSup / alpS) ** nQCD
    weightRenDn = (alpEMdn / alpEM) ** nEM * (alpSdn / alpS) ** nQCD

    # Calculate muF
    Q = events.Generator.scalePDF
    id1 = np.where(abs(events.Generator.id1) == 21, 0, events.Generator.id1)
    id2 = np.where(abs(events.Generator.id2) == 21, 0, events.Generator.id2)
    x1 = events.Generator.x1
    x2 = events.Generator.x2
    nnpdf31 = parton.mkPDF("NNPDF31_nnlo_as_0118", 0)
    pdf1 = nnpdf31.xfxQ(id1, x1, Q, grid=False) / x1
    pdf2 = nnpdf31.xfxQ(id2, x2, Q, grid=False) / x2
    pdf1up = nnpdf31.xfxQ(id1, x1, kUp * Q, grid=False) / x1
    pdf2up = nnpdf31.xfxQ(id2, x2, kUp * Q, grid=False) / x2
    pdf1dn = nnpdf31.xfxQ(id1, x1, kDn * Q, grid=False) / x1
    pdf2dn = nnpdf31.xfxQ(id2, x2, kDn * Q, grid=False) / x2
    weightFacUp = (pdf1up * pdf2up) / (pdf1 * pdf2)
    weightFacDn = (pdf1dn * pdf2dn) / (pdf1 * pdf2)

    # Return an output that resembles the one from NanoAOD
    return np.array(
        [
            weightRenDn * weightFacDn,
            weightRenDn,
            weightRenDn * weightFacUp,
            weightFacDn,
            np.ones_like(weightFacUp),
            weightFacUp,
            weightRenUp * weightFacDn,
            weightRenUp,
            weightRenUp * weightFacUp,
        ]
    ).T


def get_scale_variations(events, allow_manual=True):
    """
    Get the variations for scale of renormalization, mu_R, and scale of factorization, mu_F.
    Only available if there is LHE info. Up variation is 2x the nominal value, down is 0.5x.
    Available options are:
        - MuRUp
        - MuRDown
        - MuFUp
        - MuFDown
    """
    nEM = 0
    nQCD = 0
    if "SUEP" in events.metadata["dataset"]:
        nEM = 0
        nQCD = 2
    if ("QCD" in events.metadata["dataset"]) and (
        "MuEnrichedPt5" in events.metadata["dataset"]
    ):
        nEM = 0
        nQCD = 1

    if "LHEScaleWeight" in events.fields:
        if any(ak.num(events.LHEScaleWeight) > 8):
            if not all(ak.num(events.LHEScaleWeight) == 9) and allow_manual:
                LHEScaleWeight_alt = matrix_element_scale_variations(
                    events, nEM=nEM, nQCD=nQCD, kUp=2, kDn=0.5
                )
            else:
                LHEScaleWeight_alt = ak.from_numpy(np.ones((len(events), 9)))
            LHEScaleWeight = ak.where(
                ak.num(events.LHEScaleWeight) == 9,
                events.LHEScaleWeight,
                LHEScaleWeight_alt,
            )
        elif allow_manual:
            LHEScaleWeight = matrix_element_scale_variations(
                events, nEM=nEM, nQCD=nQCD, kUp=2, kDn=0.5
            )
        else:
            LHEScaleWeight = np.ones((len(events), 9))
    elif allow_manual:
        LHEScaleWeight = matrix_element_scale_variations(
            events, nEM=nEM, nQCD=nQCD, kUp=2, kDn=0.5
        )
    else:
        LHEScaleWeight = np.ones((len(events), 9))
    # The order is: MuRDown, MuFDown, MuFUp, MuRUp
    return (
        LHEScaleWeight[:, 1],  # type: ignore[index]
        LHEScaleWeight[:, 3],  # type: ignore[index]
        LHEScaleWeight[:, 5],  # type: ignore[index]
        LHEScaleWeight[:, 7],  # type: ignore[index]
    )


def track_killing(tracks, era):
    """
    Drop 2.7%, 2.2%, and 2.1% of the tracks randomly at reco-level for
    charged-particles with pT < 20 GeV in simulation for 2016, 2017, and
    2018, respectively when reclustering the constituents. Setting this
    to 3% for Run 3 eras for now. For charged-particles with pT > 20 GeV,
    1% of the tracks are dropped randomly.
    """

    low_pt_tracks = tracks[tracks.pt < 20]
    high_pt_tracks = tracks[tracks.pt >= 20]

    low_pt_trk_cnts = ak.num(low_pt_tracks)
    high_pt_trk_cnts = ak.num(high_pt_tracks)

    rng = np.random.default_rng()
    low_pt_rnd_arr = ak.unflatten(rng.random(ak.sum(low_pt_trk_cnts)), low_pt_trk_cnts)
    high_pt_rnd_arr = ak.unflatten(
        rng.random(ak.sum(high_pt_trk_cnts)), high_pt_trk_cnts
    )

    year_percent = {
        "2016": 0.027,
        "2016APV": 0.027,
        "2017": 0.022,
        "2018": 0.021,
        "2022": 0.03,
        "2022EE": 0.03,
        "2023": 0.03,
        "2023BPix": 0.03,
    }
    low_pt_percent = year_percent[era]
    high_pt_percent = 0.01
    low_pt_tracks_cut = low_pt_tracks[low_pt_rnd_arr > low_pt_percent]
    high_pt_tracks_cut = high_pt_tracks[high_pt_rnd_arr > high_pt_percent]
    return ak.concatenate([high_pt_tracks_cut, low_pt_tracks_cut], axis=1)


def higgs_reweight(higgs_pt, variation="nominal"):
    json_file = "data/higgs_reweight/higgs_reweight.json"
    higgs_reweight_corrset = correctionlib.CorrectionSet.from_file(json_file)
    higgs_pt_reweight_corr = higgs_reweight_corrset["Higgs_pt_reweighting"]
    return higgs_pt_reweight_corr.evaluate(higgs_pt, variation)
