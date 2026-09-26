"""Small eta-phi event displays for CMS NanoAOD files.

The public API is intentionally notebook-friendly::

    from plotting.event_display import display_event

    fig, ax, event = display_event("events.root", entry=12)

The module is also a command-line program; run ``python -m
plotting.event_display --help`` for the available options.
"""

from __future__ import annotations

import argparse
import math
import posixpath
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

OBJECTS = ("Jet", "Muon", "Electron", "Photon")
ANALYSIS_SELECTIONS = ("VRloose", "VRtight")
DEFAULT_MIN_PT = {
    "Jet": 20.0,
    "Muon": 5.0,
    "Electron": 5.0,
    "Photon": 10.0,
}

CLEAN_MUON_MIN_PT = 5.0
CLEAN_MUON_ETA_MAX = 2.4
CLEAN_MUON_DZ_MAX = 0.2

_REQUIRED_FIELDS = ("pt", "eta", "phi")
_OPTIONAL_FIELDS = {
    "Jet": ("mass", "jetId", "btagDeepFlavB"),
    "Muon": (
        "mass",
        "charge",
        "mediumId",
        "dxy",
        "dz",
        "miniPFRelIso_all",
    ),
    "Electron": ("mass", "charge"),
    "Photon": ("mass",),
}


@dataclass(frozen=True)
class ObjectCollection:
    """One NanoAOD physics-object collection from a single event."""

    name: str
    pt: np.ndarray
    eta: np.ndarray
    phi: np.ndarray
    extras: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pt)

    def select(
        self, min_pt: float = 0.0, eta_max: float = math.inf
    ) -> ObjectCollection:
        mask = (
            np.isfinite(self.pt)
            & np.isfinite(self.eta)
            & np.isfinite(self.phi)
            & (self.pt >= min_pt)
            & (np.abs(self.eta) <= eta_max)
        )
        return ObjectCollection(
            name=self.name,
            pt=self.pt[mask],
            eta=self.eta[mask],
            phi=self.phi[mask],
            extras={key: values[mask] for key, values in self.extras.items()},
        )

    def masked(self, mask: np.ndarray) -> ObjectCollection:
        """Return the objects selected by a boolean mask."""

        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(self):
            raise ValueError(
                f"mask has {len(mask)} entries for {len(self)} {self.name}s"
            )
        return ObjectCollection(
            name=self.name,
            pt=self.pt[mask],
            eta=self.eta[mask],
            phi=self.phi[mask],
            extras={key: values[mask] for key, values in self.extras.items()},
        )


@dataclass(frozen=True)
class EventData:
    """The displayable content and identifiers for one NanoAOD event."""

    source: str
    tree_name: str
    entry: int
    run: int | None
    luminosity_block: int | None
    event: int | None
    collections: Mapping[str, ObjectCollection]
    warnings: tuple[str, ...] = ()
    selection: str | None = None

    def __getitem__(self, name: str) -> ObjectCollection:
        return self.collections[name]

    @property
    def identifier(self) -> str:
        if (
            self.run is not None
            and self.luminosity_block is not None
            and self.event is not None
        ):
            return f"run {self.run}, lumi {self.luminosity_block}, event {self.event}"
        return f"entry {self.entry}"


class EventSelectionError(ValueError):
    """Raised when an event fails a requested analysis-region selection."""


class EventFilteredOut(EventSelectionError):
    """Control-flow signal for an event rejected by baseline object selection."""


def _normalise_phi(phi: np.ndarray) -> np.ndarray:
    """Map phi to [-pi, pi), as used by the display."""

    return (np.asarray(phi, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _available_branches(tree) -> set:
    return {str(key).split(";", 1)[0] for key in tree.keys()}


def _resolve_entry(
    tree,
    available: set,
    entry: int | None,
    run: int | None,
    luminosity_block: int | None,
    event: int | None,
) -> int:
    if event is None:
        if run is not None or luminosity_block is not None:
            raise ValueError("run/luminosity_block require an event number")
        selected = 0 if entry is None else entry
        if selected < 0:
            selected += tree.num_entries
        if selected < 0 or selected >= tree.num_entries:
            raise IndexError(
                f"entry {entry} is outside the tree with {tree.num_entries} entries"
            )
        return int(selected)

    if entry is not None:
        raise ValueError("choose an entry index or an event number, not both")
    if "event" not in available:
        raise KeyError("the tree has no 'event' branch for event-number lookup")

    id_branches = ["event"]
    if run is not None:
        if "run" not in available:
            raise KeyError("the tree has no 'run' branch")
        id_branches.append("run")
    if luminosity_block is not None:
        if "luminosityBlock" not in available:
            raise KeyError("the tree has no 'luminosityBlock' branch")
        id_branches.append("luminosityBlock")

    identifiers = tree.arrays(id_branches, library="np", how=dict)
    mask = identifiers["event"] == event
    if run is not None:
        mask &= identifiers["run"] == run
    if luminosity_block is not None:
        mask &= identifiers["luminosityBlock"] == luminosity_block
    matches = np.flatnonzero(mask)
    if not len(matches):
        requested = [f"event {event}"]
        if run is not None:
            requested.append(f"run {run}")
        if luminosity_block is not None:
            requested.append(f"lumi {luminosity_block}")
        raise LookupError("no entry matches " + ", ".join(requested))
    return int(matches[0])


def _event_value(arrays: Mapping[str, np.ndarray], branch: str) -> int | None:
    if branch not in arrays:
        return None
    value = np.asarray(arrays[branch])
    return int(value[0]) if len(value) else None


def _object_values(arrays: Mapping[str, np.ndarray], branch: str) -> np.ndarray:
    outer = arrays[branch]
    if not len(outer):
        return np.array([], dtype=float)
    return np.asarray(outer[0]).reshape(-1)


def read_event(
    source: str | Path,
    *,
    entry: int | None = None,
    run: int | None = None,
    luminosity_block: int | None = None,
    event: int | None = None,
    tree_name: str = "Events",
    objects: Sequence[str] = OBJECTS,
) -> EventData:
    """Read one event from a local or XRootD NanoAOD file.

    Select the event either by zero-based ``entry`` (the default is 0), or by
    ``event`` with optional ``run`` and ``luminosity_block`` qualifiers. Only
    branches needed for the requested object collections are read.
    """

    unknown = set(objects) - set(OBJECTS)
    if unknown:
        raise ValueError(f"unknown object collection(s): {', '.join(sorted(unknown))}")

    try:
        import uproot
    except ImportError as exc:  # pragma: no cover - environment-dependent message
        raise ImportError(
            "reading NanoAOD files requires the 'uproot' package"
        ) from exc

    source_text = str(source)
    with uproot.open(source_text) as root_file:
        try:
            tree = root_file[tree_name]
        except KeyError as exc:
            raise KeyError(
                f"tree '{tree_name}' was not found in {source_text}"
            ) from exc

        available = _available_branches(tree)
        selected_entry = _resolve_entry(
            tree, available, entry, run, luminosity_block, event
        )

        branches = [
            branch
            for branch in ("run", "luminosityBlock", "event")
            if branch in available
        ]
        warnings = []
        readable_objects = []
        for name in objects:
            required = [f"{name}_{field}" for field in _REQUIRED_FIELDS]
            missing = [branch for branch in required if branch not in available]
            if missing:
                warnings.append(f"skipped {name}: missing {', '.join(missing)}")
                continue
            readable_objects.append(name)
            branches.extend(required)
            branches.extend(
                f"{name}_{field}"
                for field in _OPTIONAL_FIELDS[name]
                if f"{name}_{field}" in available
            )

        arrays = tree.arrays(
            branches,
            entry_start=selected_entry,
            entry_stop=selected_entry + 1,
            library="np",
            how=dict,
        )

    collections: dict[str, ObjectCollection] = {}
    for name in readable_objects:
        values = {
            field: _object_values(arrays, f"{name}_{field}")
            for field in _REQUIRED_FIELDS
        }
        lengths = {len(value) for value in values.values()}
        if len(lengths) != 1:
            size = min(lengths)
            warnings.append(f"truncated inconsistent {name} branches to {size} objects")
            values = {key: value[:size] for key, value in values.items()}

        extras = {}
        for field_name in _OPTIONAL_FIELDS[name]:
            branch = f"{name}_{field_name}"
            if branch in arrays:
                optional = _object_values(arrays, branch)
                if len(optional) == len(values["pt"]):
                    extras[field_name] = optional
                else:
                    warnings.append(
                        f"ignored {branch}: expected {len(values['pt'])} values, "
                        f"found {len(optional)}"
                    )

        order = np.argsort(values["pt"])[::-1]
        collections[name] = ObjectCollection(
            name=name,
            pt=np.asarray(values["pt"][order], dtype=float),
            eta=np.asarray(values["eta"][order], dtype=float),
            phi=_normalise_phi(values["phi"][order]),
            extras={key: value[order] for key, value in extras.items()},
        )

    return EventData(
        source=source_text,
        tree_name=tree_name,
        entry=selected_entry,
        run=_event_value(arrays, "run"),
        luminosity_block=_event_value(arrays, "luminosityBlock"),
        event=_event_value(arrays, "event"),
        collections=collections,
        warnings=tuple(warnings),
    )


def _dimuon_mass(muons: ObjectCollection, first: int, second: int) -> float:
    """Calculate a dimuon invariant mass from NanoAOD cylindrical coordinates."""

    mass = muons.extras["mass"]
    px1 = muons.pt[first] * math.cos(muons.phi[first])
    py1 = muons.pt[first] * math.sin(muons.phi[first])
    pz1 = muons.pt[first] * math.sinh(muons.eta[first])
    energy1 = math.sqrt(px1**2 + py1**2 + pz1**2 + mass[first] ** 2)
    px2 = muons.pt[second] * math.cos(muons.phi[second])
    py2 = muons.pt[second] * math.sin(muons.phi[second])
    pz2 = muons.pt[second] * math.sinh(muons.eta[second])
    energy2 = math.sqrt(px2**2 + py2**2 + pz2**2 + mass[second] ** 2)
    mass_squared = (energy1 + energy2) ** 2 - (
        (px1 + px2) ** 2 + (py1 + py2) ** 2 + (pz1 + pz2) ** 2
    )
    return math.sqrt(max(mass_squared, 0.0))


def _delta_r(muons: ObjectCollection, first: int, second: int) -> float:
    delta_eta = muons.eta[first] - muons.eta[second]
    delta_phi = _normalise_phi(np.array([muons.phi[first] - muons.phi[second]]))[0]
    return math.hypot(delta_eta, delta_phi)


def _opposite_sign_pairs(muons: ObjectCollection):
    charges = muons.extras["charge"]
    positive = np.flatnonzero(charges == 1)
    negative = np.flatnonzero(charges == -1)
    return [(int(first), int(second)) for first in positive for second in negative]


def _greedy_dimuon_pairs(muons: ObjectCollection):
    """Match OS muons greedily in increasing delta-R, as the VR workflow does."""

    remaining = _opposite_sign_pairs(muons)
    matched = []
    while remaining:
        _, first, second = min(
            (_delta_r(muons, first, second), first, second)
            for first, second in remaining
        )
        matched.append((first, second))
        remaining = [
            (candidate_first, candidate_second)
            for candidate_first, candidate_second in remaining
            if candidate_first != first and candidate_second != second
        ]
    return matched


def _selection_failure(event: EventData, selection: str, reason: str):
    raise EventSelectionError(f"{event.identifier} does not pass {selection}: {reason}")


def apply_clean_muon_selection(event: EventData) -> EventData:
    """Return an event with at least three baseline-cleaned muons."""

    muons = event.collections.get("Muon")
    if muons is None:
        raise EventSelectionError(
            f"{event.identifier} does not pass clean muon selection: "
            "the Muon collection is unavailable"
        )

    required_fields = ("mediumId", "dz")
    missing = [field for field in required_fields if field not in muons.extras]
    if missing:
        branches = ", ".join(f"Muon_{field}" for field in missing)
        raise EventSelectionError(
            f"{event.identifier} cannot apply clean muon selection: "
            f"required branch(es) missing: {branches}"
        )

    clean_mask = (
        np.asarray(muons.extras["mediumId"], dtype=bool)
        & (muons.pt > CLEAN_MUON_MIN_PT)
        & (np.abs(muons.eta) < CLEAN_MUON_ETA_MAX)
        & (np.abs(muons.extras["dz"]) < CLEAN_MUON_DZ_MAX)
    )
    cleaned_muons = muons.masked(clean_mask)
    if len(cleaned_muons) < 3:
        raise EventFilteredOut(
            f"{event.identifier} does not pass clean muon selection: only "
            f"{len(cleaned_muons)} muons pass mediumId, pT > 5 GeV, "
            "|eta| < 2.4, and |dz| < 0.2 cm; at least 3 are required"
        )

    collections = dict(event.collections)
    collections["Muon"] = cleaned_muons
    return replace(event, collections=collections)


def apply_analysis_selection(event: EventData, selection: str) -> EventData:
    """Apply an analysis region and return an event with selected muons only.

    ``VRloose`` and ``VRtight`` reproduce ``apply_VR`` in
    :mod:`workflows.SUEP_coffea_VR` without Rochester corrections. Trigger,
    trigger-plateau, and golden-JSON filtering are intentionally upstream of
    this region definition and are not applied here.
    """

    if selection not in ANALYSIS_SELECTIONS:
        raise ValueError(
            f"unknown analysis selection '{selection}'; choose "
            + ", ".join(ANALYSIS_SELECTIONS)
        )
    muons = event.collections.get("Muon")
    if muons is None:
        _selection_failure(event, selection, "the Muon collection is unavailable")

    required_fields = (
        "mass",
        "charge",
        "mediumId",
        "dxy",
        "dz",
        "miniPFRelIso_all",
    )
    missing = [field for field in required_fields if field not in muons.extras]
    if missing:
        branches = ", ".join(f"Muon_{field}" for field in missing)
        _selection_failure(event, selection, f"required branch(es) missing: {branches}")

    if len(muons) <= 1:
        _selection_failure(event, selection, "fewer than two input muons")

    event = apply_clean_muon_selection(event)
    muons = event["Muon"]

    resonance_indices = set()
    resonance_windows = ((0.4, 0.8), (2.7, 3.5), (8.8, 11.2))
    for first, second in _greedy_dimuon_pairs(muons):
        pair_mass = _dimuon_mass(muons, first, second)
        in_resonance = any(low < pair_mass < high for low, high in resonance_windows)
        if _delta_r(muons, first, second) < 0.3 and in_resonance:
            resonance_indices.update((first, second))
    if resonance_indices:
        keep_mask = np.ones(len(muons), dtype=bool)
        keep_mask[list(resonance_indices)] = False
        muons = muons.masked(keep_mask)

    if selection == "VRloose":
        displacement_threshold = 0.01
        isolation_threshold = 0.2
    else:
        displacement_threshold = 0.02
        isolation_threshold = 0.4
    region_mask = (
        (np.abs(muons.extras["dxy"]) > displacement_threshold)
        | (np.abs(muons.extras["dz"]) > displacement_threshold)
    ) & (muons.extras["miniPFRelIso_all"] > isolation_threshold)
    selected_muons = muons.masked(region_mask)
    if not len(selected_muons):
        _selection_failure(
            event,
            selection,
            f"no muons pass displacement > {displacement_threshold:g} cm "
            f"and miniPFRelIso_all > {isolation_threshold:g}",
        )

    opposite_sign_pairs = _opposite_sign_pairs(selected_muons)
    if not opposite_sign_pairs:
        _selection_failure(event, selection, "no opposite-sign selected-muon pair")
    maximum_mass = max(
        _dimuon_mass(selected_muons, first, second)
        for first, second in opposite_sign_pairs
    )
    if maximum_mass <= 20.0:
        _selection_failure(
            event,
            selection,
            f"maximum opposite-sign dimuon mass is {maximum_mass:.2f} GeV, not > 20 GeV",
        )

    collections = dict(event.collections)
    collections["Muon"] = selected_muons
    return replace(event, collections=collections, selection=selection)


def _point_size(pt: np.ndarray) -> np.ndarray:
    """A restrained marker-area scale that remains useful for soft muons."""

    return np.clip(35.0 + 7.0 * np.sqrt(np.maximum(pt, 0.0)), 40.0, 260.0)


def _annotation(name: str, collection: ObjectCollection, index: int) -> str:
    prefix = {"Jet": "j", "Muon": "μ", "Electron": "e", "Photon": "γ"}[name]
    if name in ("Muon", "Electron") and "charge" in collection.extras:
        charge = collection.extras["charge"][index]
        prefix += "+" if charge > 0 else "−" if charge < 0 else ""
    return f"{prefix} {collection.pt[index]:.0f} GeV"


def _display_source_name(source: str) -> str:
    return posixpath.basename(source.split("?", 1)[0].rstrip("/")) or source


def plot_event(
    event: EventData,
    *,
    objects: Sequence[str] = OBJECTS,
    min_pt: Mapping[str, float] | None = None,
    eta_max: float = 5.0,
    jet_radius: float = 0.4,
    annotate: bool = True,
    title: str | None = None,
    ax=None,
):
    """Plot an :class:`EventData` object and return ``(figure, axes)``.

    Marker area loosely encodes transverse momentum. Jet outlines use
    ``jet_radius`` in eta-phi coordinates and are repeated across the periodic
    phi boundary when needed.
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Patch

    if eta_max <= 0:
        raise ValueError("eta_max must be positive")
    if jet_radius <= 0:
        raise ValueError("jet_radius must be positive")
    unknown = set(objects) - set(OBJECTS)
    if unknown:
        raise ValueError(f"unknown object collection(s): {', '.join(sorted(unknown))}")

    # Keep direct plot_event() calls consistent with display_event(). This is
    # idempotent when an analysis-region selection has already cleaned muons.
    event = apply_clean_muon_selection(event)

    cuts = dict(DEFAULT_MIN_PT)
    if min_pt is not None:
        unknown_cuts = set(min_pt) - set(OBJECTS)
        if unknown_cuts:
            raise ValueError(
                f"unknown min_pt key(s): {', '.join(sorted(unknown_cuts))}"
            )
        cuts.update(min_pt)

    if ax is None:
        figure, ax = plt.subplots(figsize=(11.0, 7.2), constrained_layout=True)
    else:
        figure = ax.figure

    styles = {
        "Jet": {"color": "#3a86ff", "marker": "o", "label": "Jets"},
        "Muon": {"color": "#e63946", "marker": "o", "label": "Muons"},
        "Electron": {"color": "#8338ec", "marker": "^", "label": "Electrons"},
        "Photon": {"color": "#ff9f1c", "marker": "*", "label": "Photons"},
    }
    displayed = {}
    for name in objects:
        collection = event.collections.get(name)
        if collection is None:
            continue
        displayed[name] = collection.select(min_pt=cuts[name], eta_max=eta_max)

    jets = displayed.get("Jet")
    if jets is not None:
        for eta, phi in zip(jets.eta, jets.phi):
            centers = [phi]
            if phi + jet_radius > np.pi:
                centers.append(phi - 2.0 * np.pi)
            if phi - jet_radius < -np.pi:
                centers.append(phi + 2.0 * np.pi)
            for center_phi in centers:
                ax.add_patch(
                    Circle(
                        (eta, center_phi),
                        jet_radius,
                        facecolor=styles["Jet"]["color"],
                        edgecolor=styles["Jet"]["color"],
                        alpha=0.18,
                        linewidth=1.6,
                        zorder=1,
                    )
                )
        if len(jets):
            ax.scatter(
                jets.eta,
                jets.phi,
                s=np.clip(12.0 + 1.5 * np.sqrt(jets.pt), 15.0, 55.0),
                color=styles["Jet"]["color"],
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )

    for name in ("Muon", "Electron", "Photon"):
        collection = displayed.get(name)
        if collection is None or not len(collection):
            continue
        style = styles[name]
        ax.scatter(
            collection.eta,
            collection.phi,
            s=_point_size(collection.pt),
            marker=style["marker"],
            color=style["color"],
            edgecolors="white",
            linewidths=0.8,
            alpha=0.95,
            zorder=5,
        )

    if annotate:
        placed_annotations = {name: [] for name in objects}
        for name, collection in displayed.items():
            for index, (eta, phi) in enumerate(zip(collection.eta, collection.phi)):
                offsets = {
                    "Jet": (6, 7),
                    "Muon": (6, -13),
                    "Electron": (6, 7),
                    "Photon": (6, -13),
                }
                offset_x, offset_y = offsets[name]
                nearby = sum(
                    abs(previous_eta - eta) < 0.3
                    and abs(_normalise_phi(np.array([previous_phi - phi]))[0]) < 0.3
                    for previous_eta, previous_phi in placed_annotations[name]
                )
                if phi > np.pi - 0.25:
                    offset_y = -13 - 14 * nearby
                elif phi < -np.pi + 0.25:
                    offset_y = 7 + 14 * nearby
                elif name in ("Muon", "Photon"):
                    offset_y -= 14 * nearby
                else:
                    offset_y += 14 * nearby
                ax.annotate(
                    _annotation(name, collection, index),
                    (eta, phi),
                    xytext=(offset_x, offset_y),
                    textcoords="offset points",
                    fontsize=8.5,
                    color=styles[name]["color"],
                    weight="semibold",
                    va="top" if offset_y < 0 else "bottom",
                    zorder=7,
                    annotation_clip=True,
                )
                placed_annotations[name].append((eta, phi))

    handles = []
    for name in objects:
        collection = displayed.get(name)
        if collection is None or not len(collection):
            continue
        style = styles[name]
        if name == "Jet":
            handles.append(
                Patch(
                    facecolor=style["color"],
                    edgecolor=style["color"],
                    alpha=0.25,
                    label=f"{style['label']} (R={jet_radius:g})",
                )
            )
        else:
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=style["marker"],
                    markersize=9,
                    markerfacecolor=style["color"],
                    markeredgecolor="white",
                    label=style["label"],
                )
            )
    if handles:
        ax.legend(handles=handles, loc="upper right", framealpha=0.95, ncol=2)

    ax.set_xlim(-eta_max, eta_max)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Pseudorapidity $\eta$")
    ax.set_ylabel(r"Azimuthal angle $\phi$")
    ax.set_yticks([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
    ax.axhline(0.0, color="#aaaaaa", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="#aaaaaa", linewidth=0.8, zorder=0)

    ax.text(
        0.0,
        1.015,
        "CMS",
        transform=ax.transAxes,
        fontsize=19,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        0.103,
        1.018,
        "NanoAOD event display",
        transform=ax.transAxes,
        fontsize=12,
        style="italic",
        va="bottom",
    )
    event_label = event.identifier
    if event.selection is not None:
        selection_label = {"VRloose": "VR loose", "VRtight": "VR tight"}.get(
            event.selection, event.selection
        )
        event_label += f"  •  {selection_label}"
    ax.text(
        1.0,
        1.018,
        event_label,
        transform=ax.transAxes,
        fontsize=10.5,
        ha="right",
        va="bottom",
    )
    if title is None:
        title = _display_source_name(event.source)
    ax.set_title(title, fontsize=10, color="#555555", pad=10)

    summary = []
    for name in objects:
        collection = displayed.get(name)
        count = len(collection) if collection is not None else 0
        if name == "Muon" and cuts[name] <= CLEAN_MUON_MIN_PT:
            threshold = f"$p_T$ > {CLEAN_MUON_MIN_PT:g} GeV"
        else:
            threshold = f"$p_T$ ≥ {cuts[name]:g} GeV"
        summary.append(f"{styles[name]['label']:<9} {count:>2}   {threshold}")
    ax.text(
        0.015,
        0.025,
        "\n".join(summary),
        transform=ax.transAxes,
        fontsize=8.5,
        family="monospace",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#cccccc",
        },
        zorder=10,
    )

    return figure, ax


def display_event(
    source: str | Path,
    *,
    entry: int | None = None,
    run: int | None = None,
    luminosity_block: int | None = None,
    event: int | None = None,
    selection: str | None = None,
    tree_name: str = "Events",
    objects: Sequence[str] = OBJECTS,
    min_pt: Mapping[str, float] | None = None,
    eta_max: float = 5.0,
    jet_radius: float = 0.4,
    annotate: bool = True,
    title: str | None = None,
    ax=None,
    output: str | Path | None = None,
    dpi: int = 160,
):
    """Read and plot an event, returning ``(figure, axes, event_data)``."""

    # Muons are always needed for the baseline clean-muon multiplicity requirement,
    # even when they are not among the requested display objects.
    read_objects = tuple(dict.fromkeys((*objects, "Muon")))
    event_data = read_event(
        source,
        entry=entry,
        run=run,
        luminosity_block=luminosity_block,
        event=event,
        tree_name=tree_name,
        objects=read_objects,
    )
    if selection is not None:
        event_data = apply_analysis_selection(event_data, selection)
    else:
        event_data = apply_clean_muon_selection(event_data)
    figure, axes = plot_event(
        event_data,
        objects=objects,
        min_pt=min_pt,
        eta_max=eta_max,
        jet_radius=jet_radius,
        annotate=annotate,
        title=title,
        ax=ax,
    )
    if output is not None:
        destination = Path(output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    return figure, axes, event_data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an eta-phi event display from a CMS NanoAOD ROOT file."
    )
    parser.add_argument("source", help="local ROOT path or root:// XRootD URL")
    parser.add_argument("--tree", default="Events", help="TTree name (default: Events)")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--entry", type=int, help="zero-based entry index (default: 0)"
    )
    selector.add_argument("--event", type=int, help="NanoAOD event number")
    parser.add_argument("--run", type=int, help="run number qualifier for --event")
    parser.add_argument(
        "--lumi", type=int, help="luminosity-block qualifier for --event"
    )
    region = parser.add_mutually_exclusive_group()
    region.add_argument(
        "--VRloose",
        action="store_const",
        const="VRloose",
        dest="selection",
        help="require the loose validation-region selection",
    )
    region.add_argument(
        "--VRtight",
        action="store_const",
        const="VRtight",
        dest="selection",
        help="require the tight validation-region selection",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        choices=OBJECTS,
        default=list(OBJECTS),
        metavar="OBJECT",
        help="collections to draw (default: Jet Muon Electron Photon)",
    )
    parser.add_argument("--min-jet-pt", type=float, default=DEFAULT_MIN_PT["Jet"])
    parser.add_argument("--min-muon-pt", type=float, default=DEFAULT_MIN_PT["Muon"])
    parser.add_argument(
        "--min-electron-pt", type=float, default=DEFAULT_MIN_PT["Electron"]
    )
    parser.add_argument("--min-photon-pt", type=float, default=DEFAULT_MIN_PT["Photon"])
    parser.add_argument("--eta-max", type=float, default=5.0)
    parser.add_argument("--jet-radius", type=float, default=0.4)
    parser.add_argument("--no-annotations", action="store_true", help="hide pT labels")
    parser.add_argument("--title", help="custom plot title")
    parser.add_argument(
        "-o", "--output", help="output image path (default: event_<entry>.png)"
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--show", action="store_true", help="also open an interactive window"
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (args.run is not None or args.lumi is not None) and args.event is None:
        parser.error("--run and --lumi may only be used with --event")

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    cuts = {
        "Jet": args.min_jet_pt,
        "Muon": args.min_muon_pt,
        "Electron": args.min_electron_pt,
        "Photon": args.min_photon_pt,
    }
    try:
        figure, _, event_data = display_event(
            args.source,
            entry=args.entry,
            run=args.run,
            luminosity_block=args.lumi,
            event=args.event,
            selection=args.selection,
            tree_name=args.tree,
            objects=args.objects,
            min_pt=cuts,
            eta_max=args.eta_max,
            jet_radius=args.jet_radius,
            annotate=not args.no_annotations,
            title=args.title,
        )
    except EventFilteredOut:
        return 0
    except (ImportError, IndexError, KeyError, LookupError, OSError, ValueError) as exc:
        parser.error(str(exc))

    output = args.output
    if output is None:
        suffix = f"_{event_data.selection}" if event_data.selection else ""
        if event_data.event is not None:
            output = f"event_{event_data.event}{suffix}.png"
        else:
            output = f"event_entry_{event_data.entry}{suffix}.png"
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=args.dpi, bbox_inches="tight")

    print(f"Saved {event_data.identifier} to {destination}")
    for warning in event_data.warnings:
        print(f"Warning: {warning}")
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
