import math

import awkward as ak
import fastjet
import numpy as np
import vector  # type: ignore [import]
from numba import njit  # type: ignore [import]

vector.register_awkward()
ak.numba.register()


def sphericity(particles, r):
    norm = ak.sum(particles.p**r, axis=1, keepdims=True)
    s = np.array(
        [
            [
                ak.sum(
                    particles.px * particles.px * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
                ak.sum(
                    particles.px * particles.py * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
                ak.sum(
                    particles.px * particles.pz * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
            ],
            [
                ak.sum(
                    particles.py * particles.px * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
                ak.sum(
                    particles.py * particles.py * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
                ak.sum(
                    particles.py * particles.pz * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
            ],
            [
                ak.sum(
                    particles.pz * particles.px * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
                ak.sum(
                    particles.pz * particles.py * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
                ak.sum(
                    particles.pz * particles.pz * particles.p ** (r - 2.0),
                    axis=1,
                    keepdims=True,
                )
                / norm,
            ],
        ]
    )
    s = np.squeeze(np.moveaxis(s, 2, 0), axis=3)
    evals = np.sort(np.linalg.eigvalsh(s))
    return evals


def rho(number, jet, tracks, deltaR, dr=0.05):
    r_start = number * dr
    r_end = (number + 1) * dr
    ring = (deltaR > r_start) & (deltaR < r_end)
    rho_values = ak.sum(tracks[ring].pt, axis=1) / (dr * jet.pt)
    return rho_values


def FastJetReclustering(tracks, r, min_pt):
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, r)
    cluster = fastjet.ClusterSequence(tracks, jetdef)
    ak_inclusive_jets = cluster.inclusive_jets(min_pt=min_pt)
    ak_inclusive_cluster = cluster.constituents(min_pt=min_pt)
    return ak_inclusive_jets, ak_inclusive_cluster


def getTopTwoJets(events_, tracks, muons_, ak_inclusive_jets, ak_inclusive_cluster):
    # order the reclustered jets by pT (will take top 2 for ISR removal method)
    highpt_jet = ak.argsort(ak_inclusive_jets.pt, axis=1, ascending=False, stable=True)
    jets_pTsorted = ak_inclusive_jets[highpt_jet]
    clusters_pTsorted = ak_inclusive_cluster[highpt_jet]

    # at least 2 tracks in SUEP and ISR
    singletrackCut = (ak.num(clusters_pTsorted[:, 0]) > 1) & (
        ak.num(clusters_pTsorted[:, 1]) > 1
    )
    jets_pTsorted = jets_pTsorted[singletrackCut]
    clusters_pTsorted = clusters_pTsorted[singletrackCut]
    tracks = tracks[singletrackCut]
    muons_ = muons_[singletrackCut]
    events_ = events_[singletrackCut]

    # number of constituents per jet, sorted by pT
    nconst_pTsorted = ak.num(clusters_pTsorted, axis=-1)

    # Top 2 pT jets. If jet1 has fewer tracks than jet2 then swap
    SUEP_cand = ak.where(
        nconst_pTsorted[:, 1] <= nconst_pTsorted[:, 0],
        jets_pTsorted[:, 0],
        jets_pTsorted[:, 1],
    )
    ISR_cand = ak.where(
        nconst_pTsorted[:, 1] > nconst_pTsorted[:, 0],
        jets_pTsorted[:, 0],
        jets_pTsorted[:, 1],
    )
    SUEP_cluster_tracks = ak.where(
        nconst_pTsorted[:, 1] <= nconst_pTsorted[:, 0],
        clusters_pTsorted[:, 0],
        clusters_pTsorted[:, 1],
    )
    ISR_cluster_tracks = ak.where(
        nconst_pTsorted[:, 1] > nconst_pTsorted[:, 0],
        clusters_pTsorted[:, 0],
        clusters_pTsorted[:, 1],
    )

    return (
        events_,
        tracks,
        muons_,
        (SUEP_cand, ISR_cand, SUEP_cluster_tracks, ISR_cluster_tracks),
    )


def inter_isolation(leptons_1, leptons_2, dR=1.6):
    """
    Compute the inter-isolation of each particle. It is supposed to work for one particle per event. The input is:
    - leptons_1: array of leptons for isolation calculation
    - leptons_2: array of all leptons in the events
    - dR: deltaR cut for isolation calculation
    """
    a, b = ak.unzip(ak.cartesian([leptons_1, leptons_2]))
    deltar_mask = a.deltaR(b) < dR
    return (ak.sum(b[deltar_mask].pt, axis=-1) - leptons_1.pt) / leptons_1.pt


@njit
def interIsolation_old(particles, v_electrons, v_muons, cone_size=0.4):
    """
    Compute the inter-isolation of each particle. It is supposed to work for one particle per event. The input is:
    - particles: array of particles for isolation calculation
    - v_electrons: array of arrays of electrons for each event
    - v_muons: array of arrays of muons for each event
    """
    n_particles = len(particles)
    out = np.zeros(n_particles)
    for i in range(n_particles):
        particle = particles[i]
        muons = v_muons[i]
        electrons = v_electrons[i]
        for j in range(len(muons)):
            dEta = particle.eta - muons[j].eta
            dPhi = particle.phi - muons[j].phi
            if abs(dPhi) > math.pi:
                dPhi = 2 * math.pi - abs(dPhi)
            dR = math.sqrt((dEta) ** 2 + (dPhi) ** 2)
            if dR < 1.6:
                out[i] += muons[j].pt
        for j in range(len(electrons)):
            dEta = particle.eta - electrons[j].eta
            dPhi = particle.phi - electrons[j].phi
            if abs(dPhi) > math.pi:
                dPhi = 2 * math.pi - abs(dPhi)
            dR = math.sqrt((dEta) ** 2 + (dPhi) ** 2)
            if dR < cone_size:
                out[i] += electrons[j].pt
        out[i] -= particle.pt
        out[i] /= particle.pt
    return out


def n_eta_ring(muonsCollection, eta_cutoff):
    """Return the number of muons in the eta ring around the leading muon"""
    leading_muon_eta = muonsCollection[:, 0].eta
    return ak.num(
        muonsCollection[abs(leading_muon_eta - muonsCollection.eta) < eta_cutoff]
    )


def transverse_mass(particles, met):
    """Return the transverse mass of an array of particles and the missing transverse energy"""
    return np.sqrt(2 * particles.pt * met.pt * (1 - np.cos(particles.delta_phi(met))))


def get_last_parents(genParts):
    # Begin with the matched gen muons
    temp_parents = genParts
    # This mask keeps track of which non-muon parents have not appeared already. Initialize to True.
    has_not_appeared = ak.full_like(temp_parents.pt, True, dtype=bool)
    # The useful parents are the last non-muon parents. It should be empty initially.
    last_parents = ak.mask(temp_parents, ~has_not_appeared)
    # Terminate when none of the particles is a muon
    while ak.any(abs(temp_parents.pdgId) == abs(genParts.pdgId)):
        # Create mask of the particles that are not muons and have not already been kept
        mask = (abs(temp_parents.pdgId) != abs(genParts.pdgId)) & has_not_appeared
        # Create the parents collection that includes all non-muon parents
        parents = ak.zip(
            {
                "pt": ak.mask(temp_parents, mask).pt,
                "eta": ak.mask(temp_parents, mask).eta,
                "phi": ak.mask(temp_parents, mask).phi,
                "M": ak.mask(temp_parents, mask).mass,
                "pdgId": ak.mask(temp_parents, mask).pdgId,
                "status": ak.mask(temp_parents, mask).status,
            },
            with_name="Momentum4D",
        )
        # Remove the parents that were found from the mask
        has_not_appeared = has_not_appeared & ~mask
        # Add the parents that are not None
        last_parents = ak.where(ak.fill_none(parents.pt, 0) > 0, parents, last_parents)
        # Get the parents of the particles
        temp_parents = temp_parents.parent
    return last_parents


def probabilistic_removal(muons_genPartFlav):
    """Will return a mask that will remove 7.2% of muons with flavor 0"""
    is_matched = muons_genPartFlav != 0
    rng = np.random.default_rng(12345)
    counts = ak.num(muons_genPartFlav)
    numbers = rng.random(len(ak.flatten(muons_genPartFlav)))
    probs = ak.unflatten(numbers, counts)
    unmatched_muons_passing = probs > 0.072
    return is_matched | unmatched_muons_passing


def LLP_free_muons(events, muons):
    """Will return a mask that will remove the non-0 muons that have an LLP in their gen history"""
    genParts = events.GenPart[muons.genPartIdx]
    is_unmatched = muons.genPartFlav == 0
    temp_parents = genParts
    has_matched = ak.full_like(temp_parents.pt, False, dtype=bool)
    while not ak.all(ak.is_none(temp_parents.pt, axis=-1)):
        mask = ak.fill_none(
            (abs(temp_parents.pdgId) == 130)
            | (abs(temp_parents.pdgId) == 211)
            | (abs(temp_parents.pdgId) == 321),
            False,
        )
        has_matched = ak.where(
            mask, ak.full_like(temp_parents.pt, True, dtype=bool), has_matched
        )
        temp_parents = temp_parents.parent
    return (not has_matched) | is_unmatched


def discritize_pdg_codes(pdf_codes, extended=False):
    discritized_codes = ak.where(pdf_codes == 0, 0, -1)
    if extended:
        discritized_codes = ak.where(pdf_codes == 1, 1, discritized_codes)
        discritized_codes = ak.where(pdf_codes == 2, 2, discritized_codes)
        discritized_codes = ak.where(pdf_codes == 3, 3, discritized_codes)
        discritized_codes = ak.where(pdf_codes == 4, 4, discritized_codes)
        discritized_codes = ak.where(pdf_codes == 5, 5, discritized_codes)
    discritized_codes = ak.where(pdf_codes == 11, 11, discritized_codes)
    discritized_codes = ak.where(pdf_codes == 13, 13, discritized_codes)
    discritized_codes = ak.where(pdf_codes == 15, 15, discritized_codes)
    discritized_codes = ak.where(pdf_codes == 22, 22, discritized_codes)
    discritized_codes = ak.where(
        (100 <= pdf_codes) & (pdf_codes < 200), 100, discritized_codes
    )
    discritized_codes = ak.where(
        (200 <= pdf_codes) & (pdf_codes < 300), 200, discritized_codes
    )
    discritized_codes = ak.where(
        (300 <= pdf_codes) & (pdf_codes < 400), 300, discritized_codes
    )
    discritized_codes = ak.where(
        (400 <= pdf_codes) & (pdf_codes < 500), 400, discritized_codes
    )
    discritized_codes = ak.where(
        (500 <= pdf_codes) & (pdf_codes < 600), 500, discritized_codes
    )
    discritized_codes = ak.where(1000 <= pdf_codes, 1000, discritized_codes)
    return discritized_codes


@njit
def loop_over_arr(arr1, arr2, builder):
    """
    Loop over two arrays and append the elements of arr1 that are not in arr2 to the builder.
    Used to filter the dimuon pairs after some selection (e.g., mass cut).
    arr1: array with indices
    arr2: array with all the indices of the pairs failing the selection
    """
    for i in range(len(arr1)):
        arr1_i = arr1[i]
        builder.begin_list()
        for j in range(len(arr1_i)):
            if arr1[i][j] not in arr2[i]:
                builder.integer(arr1[i][j])
        builder.end_list()
    return builder
