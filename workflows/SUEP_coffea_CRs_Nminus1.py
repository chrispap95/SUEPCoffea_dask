import awkward as ak
import hist
import vector  # type: ignore[import]
from coffea import processor

# Importing CMS corrections
import workflows.CMS_corrections.golden_json_utils as golden_json_utils
import workflows.CMS_corrections.muon_sf_utils as muon_sf_utils
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

    def apply_CR_prompt_dimuon_mass_cut(self, events):
        """
        Apply the CR_prompt selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )

        # Apply tight miniIso id corresponding to miniIso < 0.1
        prompt_muons = muons[
            (muons.pt > 25)
            & (muons.miniIsoId >= 3)
            & (abs(muons.dxy) < 0.01)
            & (abs(muons.dz) < 0.01)
            # & (abs(candidate_muons.ip3d) < 0.01)
        ]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        muons = muons[clean_muons]
        events, prompt_muons, Z_cands, candidates_indices, muons = (
            self.find_Z_candidates(events, prompt_muons, muons, apply_dR_cut=False)
        )

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            (muons.miniIsoId < 3)
            & (abs(muons.dxy) > 0.01)
            & (abs(muons.dz) > 0.01)
            # & (abs(muons.ip3d) > 0.015)
        ]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]
        Z_cands = Z_cands[select_by_muons_low]
        prompt_muons = prompt_muons[select_by_muons_low]
        qcd_muons = qcd_muons[select_by_muons_low]

        return events, muons, Z_cands, prompt_muons, qcd_muons

    def apply_CR_prompt_prompt_muon_pt_cut(self, events):
        """
        Apply the CR_prompt selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Apply tight miniIso id corresponding to miniIso < 0.1
        prompt_muons = muons[
            (muons.miniIsoId >= 3)
            & (abs(muons.dxy) < 0.01)
            & (abs(muons.dz) < 0.01)
            # & (abs(muons.ip3d) < 0.01)
        ]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        events, prompt_muons, Z_cands, candidates_indices, muons = (
            self.find_Z_candidates(events, prompt_muons, muons, apply_dR_cut=False)
        )
        inside_mass_window = (
            abs(Z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
        )
        muons = muons[inside_mass_window]
        events = events[inside_mass_window]
        prompt_muons = prompt_muons[inside_mass_window]
        candidates_indices = candidates_indices[inside_mass_window]

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            (muons.miniIsoId < 3)  # miniIsoId < 3 corresponds to miniIso > 0.1
            & (abs(muons.dxy) > 0.01)
            & (abs(muons.dz) > 0.01)
            # & (abs(muons.ip3d) > 0.015)
        ]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons, prompt_muons, qcd_muons

    def apply_CR_prompt_prompt_muon_iso_cut(self, events):
        """
        Apply the CR_prompt selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Apply tight miniIso id corresponding to miniIso < 0.1
        prompt_muons = muons[
            (muons.pt > 25)
            & (abs(muons.dxy) < 0.01)
            & (abs(muons.dz) < 0.01)
            # & (abs(muons.ip3d) < 0.01)
        ]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        events, prompt_muons, Z_cands, candidates_indices, muons = (
            self.find_Z_candidates(events, prompt_muons, muons, apply_dR_cut=False)
        )
        inside_mass_window = (
            abs(Z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
        )
        muons = muons[inside_mass_window]
        events = events[inside_mass_window]
        prompt_muons = prompt_muons[inside_mass_window]
        candidates_indices = candidates_indices[inside_mass_window]

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            (muons.miniIsoId < 3)  # miniIsoId < 3 corresponds to miniIso > 0.1
            & (abs(muons.dxy) > 0.01)
            & (abs(muons.dz) > 0.01)
            # & (abs(muons.ip3d) > 0.015)
        ]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons, prompt_muons, qcd_muons

    def apply_CR_prompt_prompt_muon_dxy_cut(self, events):
        """
        Apply the CR_prompt selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Apply tight miniIso id corresponding to miniIso < 0.1
        prompt_muons = muons[
            (muons.pt > 25)
            & (muons.miniIsoId >= 3)
            & (abs(muons.dz) < 0.01)
            # & (abs(muons.ip3d) < 0.01)
        ]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        events, prompt_muons, Z_cands, candidates_indices, muons = (
            self.find_Z_candidates(events, prompt_muons, muons, apply_dR_cut=False)
        )
        inside_mass_window = (
            abs(Z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
        )
        muons = muons[inside_mass_window]
        events = events[inside_mass_window]
        prompt_muons = prompt_muons[inside_mass_window]
        candidates_indices = candidates_indices[inside_mass_window]

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            (muons.miniIsoId < 3)  # miniIsoId < 3 corresponds to miniIso > 0.1
            & (abs(muons.dxy) > 0.01)
            & (abs(muons.dz) > 0.01)
            # & (abs(muons.ip3d) > 0.015)
        ]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons, prompt_muons, qcd_muons

    def apply_CR_prompt_prompt_muon_dz_cut(self, events):
        """
        Apply the CR_prompt selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Apply tight miniIso id corresponding to miniIso < 0.1
        prompt_muons = muons[
            (muons.pt > 25)
            & (muons.miniIsoId >= 3)
            & (abs(muons.dxy) < 0.01)
            # & (abs(muons.ip3d) < 0.01)
        ]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        events, prompt_muons, Z_cands, candidates_indices, muons = (
            self.find_Z_candidates(events, prompt_muons, muons, apply_dR_cut=False)
        )
        inside_mass_window = (
            abs(Z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
        )
        muons = muons[inside_mass_window]
        events = events[inside_mass_window]
        prompt_muons = prompt_muons[inside_mass_window]
        candidates_indices = candidates_indices[inside_mass_window]

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            (muons.miniIsoId < 3)  # miniIsoId < 3 corresponds to miniIso > 0.1
            & (abs(muons.dxy) > 0.01)
            & (abs(muons.dz) > 0.01)
            # & (abs(muons.ip3d) > 0.015)
        ]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons, prompt_muons, qcd_muons

    def apply_CR_prompt_prompt_muon_ip3d_cut(self, events):
        """
        Apply the CR_prompt selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Apply tight miniIso id corresponding to miniIso < 0.1
        prompt_muons = muons[
            (muons.pt > 25)
            & (muons.miniIsoId >= 3)
            & (abs(muons.dxy) < 0.01)
            & (abs(muons.dz) < 0.01)
            # & (abs(muons.ip3d) < 0.01)
        ]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        events, prompt_muons, Z_cands, candidates_indices, muons = (
            self.find_Z_candidates(events, prompt_muons, muons, apply_dR_cut=False)
        )
        inside_mass_window = (
            abs(Z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
        )
        muons = muons[inside_mass_window]
        events = events[inside_mass_window]
        prompt_muons = prompt_muons[inside_mass_window]
        candidates_indices = candidates_indices[inside_mass_window]

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            (muons.miniIsoId < 3)  # miniIsoId < 3 corresponds to miniIso > 0.1
            & (abs(muons.dxy) > 0.01)
            & (abs(muons.dz) > 0.01)
            # & (abs(muons.ip3d) > 0.015)
        ]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons, prompt_muons, qcd_muons

    def apply_CR_prompt_qcd_muon_iso_cut(self, events):
        """
        Apply the CR_prompt selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Apply tight miniIso id corresponding to miniIso < 0.1
        prompt_muons = muons[
            (muons.pt > 25)
            & (muons.miniIsoId >= 3)
            & (abs(muons.dxy) < 0.01)
            & (abs(muons.dz) < 0.01)
            # & (abs(muons.ip3d) < 0.01)
        ]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        events, prompt_muons, Z_cands, candidates_indices, muons = (
            self.find_Z_candidates(events, prompt_muons, muons, apply_dR_cut=False)
        )
        inside_mass_window = (
            abs(Z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
        )
        muons = muons[inside_mass_window]
        events = events[inside_mass_window]
        prompt_muons = prompt_muons[inside_mass_window]
        candidates_indices = candidates_indices[inside_mass_window]

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            # (muons.miniIsoId < 3)  # miniIsoId < 3 corresponds to miniIso > 0.1
            (abs(muons.dxy) > 0.01)
            & (abs(muons.dz) > 0.01)
            # & (abs(muons.ip3d) > 0.015)
        ]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons, prompt_muons, qcd_muons

    def apply_CR_prompt_qcd_muon_dxy_cut(self, events):
        """
        Apply the CR_prompt selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Apply tight miniIso id corresponding to miniIso < 0.1
        prompt_muons = muons[
            (muons.pt > 25)
            & (muons.miniIsoId >= 3)
            & (abs(muons.dxy) < 0.01)
            & (abs(muons.dz) < 0.01)
            # & (abs(muons.ip3d) < 0.01)
        ]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        events, prompt_muons, Z_cands, candidates_indices, muons = (
            self.find_Z_candidates(events, prompt_muons, muons, apply_dR_cut=False)
        )
        inside_mass_window = (
            abs(Z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
        )
        muons = muons[inside_mass_window]
        events = events[inside_mass_window]
        prompt_muons = prompt_muons[inside_mass_window]
        candidates_indices = candidates_indices[inside_mass_window]

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            (muons.miniIsoId < 3)  # miniIsoId < 3 corresponds to miniIso > 0.1
            # & (abs(muons.dxy) > 0.01)
            & (abs(muons.dz) > 0.01)
            # & (abs(muons.ip3d) > 0.015)
        ]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons, prompt_muons, qcd_muons

    def apply_CR_prompt_qcd_muon_dz_cut(self, events):
        """
        Apply the CR_prompt selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Apply tight miniIso id corresponding to miniIso < 0.1
        prompt_muons = muons[
            (muons.pt > 25)
            & (muons.miniIsoId >= 3)
            & (abs(muons.dxy) < 0.01)
            & (abs(muons.dz) < 0.01)
            # & (abs(muons.ip3d) < 0.01)
        ]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        events, prompt_muons, Z_cands, candidates_indices, muons = (
            self.find_Z_candidates(events, prompt_muons, muons, apply_dR_cut=False)
        )
        inside_mass_window = (
            abs(Z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
        )
        muons = muons[inside_mass_window]
        events = events[inside_mass_window]
        prompt_muons = prompt_muons[inside_mass_window]
        candidates_indices = candidates_indices[inside_mass_window]

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            (muons.miniIsoId < 3)  # miniIsoId < 3 corresponds to miniIso > 0.1
            & (abs(muons.dxy) > 0.01)
            # & (abs(muons.dz) > 0.01)
            # & (abs(muons.ip3d) > 0.015)
        ]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons, prompt_muons, qcd_muons

    def apply_CR_prompt_qcd_muon_ip3d_cut(self, events):
        """
        Apply the CR_prompt selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        if self.do_rochester:
            muons = muon_sf_utils.muon_scale_factors(
                events, muons, self.era, self.isMC, var="nominal"
            )

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dxy) < 0.2)
            & (abs(muons.dz) < 0.2)
        )
        muons = muons[clean_muons]

        # Apply tight miniIso id corresponding to miniIso < 0.1
        prompt_muons = muons[
            (muons.pt > 25)
            & (muons.miniIsoId >= 3)
            & (abs(muons.dxy) < 0.01)
            & (abs(muons.dz) < 0.01)
            # & (abs(muons.ip3d) < 0.01)
        ]
        enough_prompt_muons = ak.num(prompt_muons, axis=-1) > 1
        os_muons_mask = ak.prod(prompt_muons.charge, axis=-1) < 0
        muons = muons[enough_prompt_muons & os_muons_mask]
        events = events[enough_prompt_muons & os_muons_mask]
        prompt_muons = prompt_muons[enough_prompt_muons & os_muons_mask]

        # Get the Z candidates and make sure they are close to the peak
        events, prompt_muons, Z_cands, candidates_indices, muons = (
            self.find_Z_candidates(events, prompt_muons, muons, apply_dR_cut=False)
        )
        inside_mass_window = (
            abs(Z_cands.mass - SUEP_common.Z_MASS) < 2 * SUEP_common.Z_WIDTH
        )
        muons = muons[inside_mass_window]
        events = events[inside_mass_window]
        prompt_muons = prompt_muons[inside_mass_window]
        candidates_indices = candidates_indices[inside_mass_window]

        # Non prompt muons – these are orthogonal to the previous selection so they can just be added
        qcd_muons = muons[
            (muons.miniIsoId < 3)  # miniIsoId < 3 corresponds to miniIso > 0.1
            & (abs(muons.dxy) > 0.01)
            & (abs(muons.dz) > 0.01)
            # & (abs(muons.ip3d) > 0.015)
        ]
        muons = ak.concatenate([prompt_muons, qcd_muons], axis=-1)

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_low]
        muons = muons[select_by_muons_low]

        return events, muons, prompt_muons, qcd_muons

    def apply_CR_cb_muon_dxy_cut(self, events):
        """
        Apply the CR_cb selection to the events.
        """
        muons = events.Muon
        events, muons = events[ak.num(muons) > 0], muons[ak.num(muons) > 0]

        # Apply basic muon cuts
        clean_muons = (
            (muons.mediumId)
            & (muons.pt > 3)
            & (abs(muons.eta) < 2.4)
            & (abs(muons.dz) < 0.2)
        )

        # Apply extra very tight cuts for CR_cb
        muons = muons[clean_muons]

        # Make sure there is at least one muon in the event after the cuts
        select_by_muons_high = True  # ak.num(muons, axis=-1) < 5
        select_by_muons_low = ak.num(muons, axis=-1) > 0
        events = events[select_by_muons_high & select_by_muons_low]
        muons = muons[select_by_muons_high & select_by_muons_low]

        return events, muons

    def fill_histograms(self, events, output):
        dataset = events.metadata["dataset"]

        events_ = self.muon_filter(events)
        if len(events_) == 0:
            return

        (
            events_CR_prompt,
            muons_CR_prompt,
            dimuons,
            prompt_muons_CR_prompt,
            qcd_muons_CR_prompt,
        ) = self.apply_CR_prompt_dimuon_mass_cut(events_)
        if len(events_CR_prompt) > 0:
            weights_CR_prompt = self.get_weights(
                events_CR_prompt, do_vars=False, apply_lumi_factors=True
            )
            if self.isMC:
                prompt_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        prompt_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_prompt",
                        syst="",
                    ),
                    axis=-1,
                )
                qcd_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        qcd_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_qcd",
                        syst="",
                    ),
                    axis=-1,
                )
                weights_CR_prompt.add("MuonSF", weight=prompt_muon_SFs * qcd_muon_SFs)
            output[dataset]["histograms"]["CR_prompt_Nminus1_dimuon_mass"].fill(
                dimuons.mass,
                weight=weights_CR_prompt.weight(),
            )

        (
            events_CR_prompt,
            muons_CR_prompt,
            prompt_muons_CR_prompt,
            qcd_muons_CR_prompt,
        ) = self.apply_CR_prompt_prompt_muon_pt_cut(events_)
        if len(events_CR_prompt) > 0:
            weights_CR_prompt = self.get_weights(
                events_CR_prompt, do_vars=False, apply_lumi_factors=True
            )
            if self.isMC:
                prompt_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        prompt_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_prompt",
                        syst="",
                    ),
                    axis=-1,
                )
                qcd_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        qcd_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_qcd",
                        syst="",
                    ),
                    axis=-1,
                )
                weights_CR_prompt.add("MuonSF", weight=prompt_muon_SFs * qcd_muon_SFs)
            output[dataset]["histograms"]["CR_prompt_Nminus1_prompt_muon_pt"].fill(
                ak.flatten(prompt_muons_CR_prompt.pt),
                ak.flatten(prompt_muons_CR_prompt.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        prompt_muons_CR_prompt.pt,
                        weights_CR_prompt.weight(),
                    )[1]
                ),
            )

        (
            events_CR_prompt,
            muons_CR_prompt,
            prompt_muons_CR_prompt,
            qcd_muons_CR_prompt,
        ) = self.apply_CR_prompt_prompt_muon_iso_cut(events_)
        if len(events_CR_prompt) > 0:
            weights_CR_prompt = self.get_weights(
                events_CR_prompt, do_vars=False, apply_lumi_factors=True
            )
            if self.isMC:
                prompt_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        prompt_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_prompt",
                        syst="",
                    ),
                    axis=-1,
                )
                qcd_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        qcd_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_qcd",
                        syst="",
                    ),
                    axis=-1,
                )
                weights_CR_prompt.add("MuonSF", weight=prompt_muon_SFs * qcd_muon_SFs)
            output[dataset]["histograms"]["CR_prompt_Nminus1_prompt_muon_iso"].fill(
                ak.flatten(prompt_muons_CR_prompt.miniPFRelIso_all),
                ak.flatten(prompt_muons_CR_prompt.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        prompt_muons_CR_prompt.miniPFRelIso_all,
                        weights_CR_prompt.weight(),
                    )[1]
                ),
            )

        (
            events_CR_prompt,
            muons_CR_prompt,
            prompt_muons_CR_prompt,
            qcd_muons_CR_prompt,
        ) = self.apply_CR_prompt_prompt_muon_dxy_cut(events_)
        if len(events_CR_prompt) > 0:
            weights_CR_prompt = self.get_weights(
                events_CR_prompt, do_vars=False, apply_lumi_factors=True
            )
            if self.isMC:
                prompt_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        prompt_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_prompt",
                        syst="",
                    ),
                    axis=-1,
                )
                qcd_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        qcd_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_qcd",
                        syst="",
                    ),
                    axis=-1,
                )
                weights_CR_prompt.add("MuonSF", weight=prompt_muon_SFs * qcd_muon_SFs)
            output[dataset]["histograms"]["CR_prompt_Nminus1_prompt_muon_dxy"].fill(
                ak.flatten(abs(prompt_muons_CR_prompt.dxy)),
                ak.flatten(prompt_muons_CR_prompt.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        prompt_muons_CR_prompt.dxy,
                        weights_CR_prompt.weight(),
                    )[1]
                ),
            )

        (
            events_CR_prompt,
            muons_CR_prompt,
            prompt_muons_CR_prompt,
            qcd_muons_CR_prompt,
        ) = self.apply_CR_prompt_prompt_muon_dz_cut(events_)
        if len(events_CR_prompt) > 0:
            weights_CR_prompt = self.get_weights(
                events_CR_prompt, do_vars=False, apply_lumi_factors=True
            )
            if self.isMC:
                prompt_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        prompt_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_prompt",
                        syst="",
                    ),
                    axis=-1,
                )
                qcd_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        qcd_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_qcd",
                        syst="",
                    ),
                    axis=-1,
                )
                weights_CR_prompt.add("MuonSF", weight=prompt_muon_SFs * qcd_muon_SFs)
            output[dataset]["histograms"]["CR_prompt_Nminus1_prompt_muon_dz"].fill(
                ak.flatten(abs(prompt_muons_CR_prompt.dz)),
                ak.flatten(prompt_muons_CR_prompt.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        prompt_muons_CR_prompt.dz,
                        weights_CR_prompt.weight(),
                    )[1]
                ),
            )

        (
            events_CR_prompt,
            muons_CR_prompt,
            prompt_muons_CR_prompt,
            qcd_muons_CR_prompt,
        ) = self.apply_CR_prompt_prompt_muon_ip3d_cut(events_)
        if len(events_CR_prompt) > 0:
            weights_CR_prompt = self.get_weights(
                events_CR_prompt, do_vars=False, apply_lumi_factors=True
            )
            if self.isMC:
                prompt_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        prompt_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_prompt",
                        syst="",
                    ),
                    axis=-1,
                )
                qcd_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        qcd_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_qcd",
                        syst="",
                    ),
                    axis=-1,
                )
                weights_CR_prompt.add("MuonSF", weight=prompt_muon_SFs * qcd_muon_SFs)
            output[dataset]["histograms"]["CR_prompt_Nminus1_prompt_muon_ip3d"].fill(
                ak.flatten(abs(prompt_muons_CR_prompt.ip3d)),
                ak.flatten(prompt_muons_CR_prompt.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        prompt_muons_CR_prompt.ip3d,
                        weights_CR_prompt.weight(),
                    )[1]
                ),
            )

        (
            events_CR_prompt,
            muons_CR_prompt,
            prompt_muons_CR_prompt,
            qcd_muons_CR_prompt,
        ) = self.apply_CR_prompt_qcd_muon_dxy_cut(events_)
        if len(events_CR_prompt) > 0:
            weights_CR_prompt = self.get_weights(
                events_CR_prompt, do_vars=False, apply_lumi_factors=True
            )
            if self.isMC:
                prompt_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        prompt_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_prompt",
                        syst="",
                    ),
                    axis=-1,
                )
                qcd_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        qcd_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_qcd",
                        syst="",
                    ),
                    axis=-1,
                )
                weights_CR_prompt.add("MuonSF", weight=prompt_muon_SFs * qcd_muon_SFs)
            output[dataset]["histograms"]["CR_prompt_Nminus1_qcd_muon_dxy"].fill(
                ak.flatten(abs(muons_CR_prompt.dxy)),
                ak.flatten(muons_CR_prompt.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_CR_prompt.dxy,
                        weights_CR_prompt.weight(),
                    )[1]
                ),
            )

        (
            events_CR_prompt,
            muons_CR_prompt,
            prompt_muons_CR_prompt,
            qcd_muons_CR_prompt,
        ) = self.apply_CR_prompt_qcd_muon_dz_cut(events_)
        if len(events_CR_prompt) > 0:
            weights_CR_prompt = self.get_weights(
                events_CR_prompt, do_vars=False, apply_lumi_factors=True
            )
            if self.isMC:
                prompt_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        prompt_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_prompt",
                        syst="",
                    ),
                    axis=-1,
                )
                qcd_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        qcd_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_qcd",
                        syst="",
                    ),
                    axis=-1,
                )
                weights_CR_prompt.add("MuonSF", weight=prompt_muon_SFs * qcd_muon_SFs)
            output[dataset]["histograms"]["CR_prompt_Nminus1_qcd_muon_dz"].fill(
                ak.flatten(abs(muons_CR_prompt.dz)),
                ak.flatten(muons_CR_prompt.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_CR_prompt.dz,
                        weights_CR_prompt.weight(),
                    )[1]
                ),
            )

        (
            events_CR_prompt,
            muons_CR_prompt,
            prompt_muons_CR_prompt,
            qcd_muons_CR_prompt,
        ) = self.apply_CR_prompt_qcd_muon_ip3d_cut(events_)
        if len(events_CR_prompt) > 0:
            weights_CR_prompt = self.get_weights(
                events_CR_prompt, do_vars=False, apply_lumi_factors=True
            )
            if self.isMC:
                prompt_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        prompt_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_prompt",
                        syst="",
                    ),
                    axis=-1,
                )
                qcd_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        qcd_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_qcd",
                        syst="",
                    ),
                    axis=-1,
                )
                weights_CR_prompt.add("MuonSF", weight=prompt_muon_SFs * qcd_muon_SFs)
            output[dataset]["histograms"]["CR_prompt_Nminus1_qcd_muon_ip3d"].fill(
                ak.flatten(muons_CR_prompt.ip3d),
                ak.flatten(muons_CR_prompt.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_CR_prompt.ip3d,
                        weights_CR_prompt.weight(),
                    )[1]
                ),
            )

        (
            events_CR_prompt,
            muons_CR_prompt,
            prompt_muons_CR_prompt,
            qcd_muons_CR_prompt,
        ) = self.apply_CR_prompt_qcd_muon_iso_cut(events_)
        if len(events_CR_prompt) > 0:
            weights_CR_prompt = self.get_weights(
                events_CR_prompt, do_vars=False, apply_lumi_factors=True
            )
            if self.isMC:
                prompt_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        prompt_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_prompt",
                        syst="",
                    ),
                    axis=-1,
                )
                qcd_muon_SFs = ak.prod(
                    muon_sf_utils.muon_efficiencies(
                        qcd_muons_CR_prompt,
                        era=self.era,
                        region="CR_prompt_qcd",
                        syst="",
                    ),
                    axis=-1,
                )
                weights_CR_prompt.add("MuonSF", weight=prompt_muon_SFs * qcd_muon_SFs)
            output[dataset]["histograms"]["CR_prompt_Nminus1_qcd_muon_iso"].fill(
                ak.flatten(muons_CR_prompt.miniPFRelIso_all),
                ak.flatten(muons_CR_prompt.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_CR_prompt.miniPFRelIso_all,
                        weights_CR_prompt.weight(),
                    )[1]
                ),
            )

        events_CR_cb, muons_CR_cb = self.apply_CR_cb_muon_dxy_cut(events_)
        if len(events_CR_cb) > 0:
            weights_CR_cb = self.get_weights(
                events_CR_cb, do_vars=False, apply_lumi_factors=True
            )
            if self.isMC:
                weights_CR_cb.add(
                    "MuonSF",
                    weight=ak.prod(
                        muon_sf_utils.muon_efficiencies(
                            muons_CR_cb, self.era, region="CR_cb", syst=""
                        ),
                        axis=-1,
                    ),
                )
            output[dataset]["histograms"]["CR_cb_Nminus1_muon_dxy"].fill(
                ak.flatten(abs(muons_CR_cb.dxy)),
                ak.flatten(muons_CR_cb.genPartFlav),
                weight=ak.flatten(
                    ak.broadcast_arrays(
                        muons_CR_cb.dxy,
                        weights_CR_cb.weight(),
                    )[1]
                ),
            )

        return

    def analysis(self, events, output):
        #######################################################################
        # ---- Trigger event selection
        # Cut based on ak4 jets to replicate the trigger
        #######################################################################

        # get dataset name
        dataset = events.metadata["dataset"]

        # take care of weights
        weights = self.get_weights(events)

        # Fill the cutflow columns for all
        output[dataset]["cutflow"].fill(len(events) * ["all"], weight=weights.weight())

        # golden jsons for offline data
        if not self.isMC:
            events = golden_json_utils.apply_golden_JSON(events, self.era)

        events = self.trigger_selection(events)

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
        histograms = {
            "CR_prompt_Nminus1_dimuon_mass": hist.Hist.new.Regular(
                100,
                0,
                200,
                name="dimuon_mass",
                label="dimuon_mass",
            ).Weight(),
            "CR_prompt_Nminus1_prompt_muon_pt": hist.Hist.new.Regular(
                100,
                0,
                100,
                name="prompt_muon_pt",
                label="prompt_muon_pt",
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_prompt_muon_iso": hist.Hist.new.Regular(
                100,
                0.02,
                20,
                name="prompt_muon_iso",
                label="prompt_muon_iso",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_prompt_muon_ip3d": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="prompt_muon_ip3d",
                label="prompt_muon_ip3d",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_prompt_muon_dxy": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="prompt_muon_dxy",
                label="prompt_muon_dxy",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_prompt_muon_dz": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="prompt_muon_dz",
                label="prompt_muon_dz",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_qcd_muon_iso": hist.Hist.new.Regular(
                100,
                0.02,
                20,
                name="qcd_muon_iso",
                label="qcd_muon_iso",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_qcd_muon_ip3d": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="qcd_muon_ip3d",
                label="qcd_muon_ip3d",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_qcd_muon_dxy": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="qcd_muon_dxy",
                label="qcd_muon_dxy",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_prompt_Nminus1_qcd_muon_dz": hist.Hist.new.Regular(
                100,
                2e-4,
                0.2,
                name="qcd_muon_dz",
                label="qcd_muon_dz",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
            "CR_cb_Nminus1_muon_dxy": hist.Hist.new.Regular(
                100,
                2e-4,
                2,
                name="muon_dxy",
                label="muon_dxy",
                transform=hist.axis.transform.log,
            )
            .IntCategory([0, 1, 3, 4, 5, 15], name="genPartFlav", label="genPartFlav")
            .Weight(),
        }
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
