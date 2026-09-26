"""Muon-multiplicity toy using the observed pre-muon 2018 nJet distribution.

Each sampled jet produces at most one loose VR muon, independently, with
probability ``P_MUON``. The toy applies the same requirement of at least one
additional loose muon as the data VR before comparing their nMuon shapes.
"""

import argparse
import math
import sys
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

# plot_utils imports other modules from the plotting directory as top-level
# modules, matching how the plotting scripts in that directory are run.
PLOTTING_DIR = Path(__file__).resolve().parent / "plotting"
sys.path.insert(0, str(PLOTTING_DIR))
import plot_utils  # noqa: E402

DEFAULT_N_EVENTS = 100_000
DEFAULT_RNG_SEED = 42

TAG = "VR_JPsi_nJet_study_Aug2026_2018_VR_JPsi_nJet_study"
ERA = "2018"
DATASET = "Data_2018"
NJET_HISTOGRAM = "nJet_preMuon"
NMUON_HISTOGRAM = "VR_loose"
OUTFILE = "toy_muon_multiplicity_realistic_njet.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p-muon",
        type=float,
        nargs="+",
        default=[],
        metavar="P",
        help="One or more per-jet decay probabilities to plot.",
    )
    parser.add_argument(
        "--fit-p-muon",
        choices=("full", "shape", "both"),
        nargs="?",
        const="both",
        help=(
            "Fit p_mu using the full distribution including zero muons, the "
            "selected nMuon shape, or both. With no value, both fits are run."
        ),
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=DEFAULT_N_EVENTS,
        help=f"Number of preselection toy events (default: {DEFAULT_N_EVENTS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RNG_SEED,
        help=f"Random-number seed (default: {DEFAULT_RNG_SEED}).",
    )
    parser.add_argument(
        "--add-exponential",
        action="store_true",
        help=(
            "Fit A*exp(-B*nMuon) to the selected data shape and add it to the "
            "main, ratio, and pull panels."
        ),
    )
    args = parser.parse_args()
    if args.n_events <= 0:
        parser.error("--n-events must be positive")
    if any(not 0 < probability < 1 for probability in args.p_muon):
        parser.error("every --p-muon value must be strictly between 0 and 1")
    return args


def load_data_distributions() -> tuple[np.ndarray, np.ndarray, float, hist.Hist]:
    """Return preselection nJet probabilities and selected data nMuon."""
    plots = plot_utils.loader(tag=TAG, era=ERA, load_data=True)
    data_histograms = plots[DATASET]
    if NJET_HISTOGRAM not in data_histograms:
        raise KeyError(
            f"Missing {NJET_HISTOGRAM!r}; rerun the nJet-study workflow with "
            "the updated histogram definitions"
        )
    if NMUON_HISTOGRAM not in data_histograms:
        raise KeyError(f"Missing selected nMuon histogram {NMUON_HISTOGRAM!r}")
    njet_hist = data_histograms[NJET_HISTOGRAM]
    nmuon_hist = data_histograms[NMUON_HISTOGRAM]

    axis = njet_hist.axes["nJet"]
    edges = np.asarray(axis.edges)
    counts = np.asarray(njet_hist.values(), dtype=float)

    # nJet is filled into unit-width integer bins.  The analysis workflow caps
    # values above 9 at 9, so the last represented value corresponds to 9+.
    if not (np.allclose(np.diff(edges), 1.0) and np.allclose(edges, np.rint(edges))):
        raise ValueError(
            f"Expected unit-width integer nJet bins, but found edges {edges}"
        )
    njet_values = np.rint(edges[:-1]).astype(int)

    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("The nJet histogram contains invalid or negative bin contents")
    if counts.sum() <= 0:
        raise ValueError("The nJet histogram is empty")

    return njet_values, counts / counts.sum(), counts.sum(), nmuon_hist


def binomial_mixture_probabilities(
    njet_values: np.ndarray,
    njet_probabilities: np.ndarray,
    p_muon: float,
    nmuon_cap: int,
) -> np.ndarray:
    """Return probabilities for nMuon = 0, 1, ..., cap-1, cap+."""
    probabilities = np.zeros(nmuon_cap + 1, dtype=float)
    for njet, njet_probability in zip(njet_values, njet_probabilities):
        for nmuon in range(njet + 1):
            probability = (
                math.comb(njet, nmuon) * p_muon**nmuon * (1 - p_muon) ** (njet - nmuon)
            )
            probabilities[min(nmuon, nmuon_cap)] += njet_probability * probability
    return probabilities


def fit_decay_rate(
    njet_values: np.ndarray,
    njet_probabilities: np.ndarray,
    data_preselection_events: float,
    data_selected_counts: np.ndarray,
    nmuon_cap: int,
    *,
    shape_only: bool,
) -> float:
    """Fit p_mu with a binned multinomial likelihood."""
    data_selected_events = data_selected_counts.sum()
    data_zero_muon_events = data_preselection_events - data_selected_events
    if data_zero_muon_events < -1e-9 * data_preselection_events:
        raise ValueError(
            "Selected nMuon yield exceeds the preselection nJet yield; the "
            "histograms do not describe nested event populations"
        )

    if shape_only:
        observed = data_selected_counts
    else:
        observed = np.concatenate(
            ([max(0.0, data_zero_muon_events)], data_selected_counts)
        )

    def negative_log_likelihood(p_muon: float) -> float:
        model = binomial_mixture_probabilities(
            njet_values, njet_probabilities, p_muon, nmuon_cap
        )
        if shape_only:
            model = model[1:] / model[1:].sum()
        return float(-np.dot(observed, np.log(np.clip(model, 1e-300, None))))

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(1e-9, 1 - 1e-9),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"The p_muon fit failed: {result.message}")
    return float(result.x)


def fit_exponential_shape(
    nmuon_values: np.ndarray, data_counts: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Fit A*exp(-B*nMuon), with A fixed by the observed total yield."""
    data_events = data_counts.sum()

    def expected_counts(slope: float) -> tuple[float, np.ndarray]:
        exponentials = np.exp(-slope * nmuon_values)
        amplitude = data_events / exponentials.sum()
        return amplitude, amplitude * exponentials

    def negative_log_likelihood(slope: float) -> float:
        _, model = expected_counts(slope)
        return float(-np.dot(data_counts, np.log(np.clip(model, 1e-300, None))))

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(0, 50),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"The exponential fit failed: {result.message}")

    amplitude, model = expected_counts(float(result.x))
    return amplitude, float(result.x), model


def get_poisson_errors(counts: np.ndarray, alpha: float = 0.6827) -> np.ndarray:
    """Return lower and upper Garwood uncertainties for Poisson counts."""
    upper = stats.gamma.ppf((1 + alpha) / 2, counts + 1) - counts
    lower = counts - stats.gamma.ppf((1 - alpha) / 2, counts)
    return np.asarray((np.nan_to_num(lower), np.nan_to_num(upper)))


def plot_ratio_and_pull(
    data_hist: hist.Hist,
    model_values: np.ndarray,
    model_variances: np.ndarray,
    ax_ratio: plt.Axes,
    ax_pull: plt.Axes,
    *,
    color: str,
) -> None:
    """Plot Data/Model and (Data-Model)/uncertainty for one model."""
    data_values = np.asarray(data_hist.values(), dtype=float)
    data_variances = np.asarray(data_hist.variances(), dtype=float)
    centers = np.asarray(data_hist.axes[0].centers)
    valid_model = model_values > 0

    ratio = np.divide(
        data_values,
        model_values,
        out=np.full_like(data_values, np.nan),
        where=valid_model,
    )
    ratio_variances = np.zeros_like(data_values)
    ratio_variances[valid_model] = (
        data_variances[valid_model] / model_values[valid_model] ** 2
        + data_values[valid_model] ** 2
        * model_variances[valid_model]
        / model_values[valid_model] ** 4
    )
    ax_ratio.errorbar(
        centers[valid_model],
        ratio[valid_model],
        yerr=np.sqrt(ratio_variances[valid_model]),
        color=color,
        fmt="o-",
        markersize=4,
        linewidth=1.5,
    )

    # Match the pull convention used by the repository plotting utilities:
    # choose the Poisson error in the direction of the model prediction and
    # combine it in quadrature with the model statistical uncertainty.
    data_poisson_errors = get_poisson_errors(data_values)
    data_uncertainty = np.where(
        data_values > model_values,
        data_poisson_errors[0],
        data_poisson_errors[1],
    )
    pull_denominator = np.sqrt(model_variances + data_uncertainty**2)
    valid_pull = valid_model & (pull_denominator > 0)
    pulls = np.divide(
        data_values - model_values,
        pull_denominator,
        out=np.full_like(data_values, np.nan),
        where=valid_pull,
    )
    ax_pull.plot(
        centers[valid_pull],
        pulls[valid_pull],
        color=color,
        marker="o",
        markersize=4,
        linewidth=1.5,
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    (
        njet_values,
        njet_probabilities,
        data_preselection_events,
        data_nmuon_hist,
    ) = load_data_distributions()

    nmuon_axis = data_nmuon_hist.axes["nMuon"]
    nmuon_edges = np.asarray(nmuon_axis.edges)
    if not (
        np.allclose(np.diff(nmuon_edges), 1.0)
        and np.allclose(nmuon_edges, np.rint(nmuon_edges))
        and np.isclose(nmuon_edges[0], 1.0)
    ):
        raise ValueError(
            "Expected selected unit-width integer nMuon bins starting at 1, "
            f"but found edges {nmuon_edges}"
        )
    nmuon_cap = int(round(nmuon_edges[-2]))

    data_selected_counts = np.asarray(data_nmuon_hist.values(), dtype=float)
    data_selected_events = data_selected_counts.sum()
    if data_selected_events <= 0:
        raise ValueError("The selected data nMuon histogram is empty")
    data_efficiency = data_selected_events / data_preselection_events

    decay_rates: list[tuple[float, str]] = [
        (probability, "manual") for probability in args.p_muon
    ]
    if args.fit_p_muon in ("full", "both"):
        fitted_probability = fit_decay_rate(
            njet_values,
            njet_probabilities,
            data_preselection_events,
            data_selected_counts,
            nmuon_cap,
            shape_only=False,
        )
        decay_rates.append((fitted_probability, "full fit"))
        print(f"Full-distribution maximum-likelihood p_muon: {fitted_probability:.6f}")
    if args.fit_p_muon in ("shape", "both"):
        fitted_probability = fit_decay_rate(
            njet_values,
            njet_probabilities,
            data_preselection_events,
            data_selected_counts,
            nmuon_cap,
            shape_only=True,
        )
        decay_rates.append((fitted_probability, "shape fit"))
        print(f"Selected-shape maximum-likelihood p_muon: {fitted_probability:.6f}")

    # Avoid plotting the same rate twice if a manual and fitted value coincide.
    unique_decay_rates: list[tuple[float, str]] = []
    for probability, source in decay_rates:
        if not any(
            np.isclose(probability, existing, rtol=0, atol=1e-10)
            for existing, _ in unique_decay_rates
        ):
            unique_decay_rates.append((probability, source))

    # Sample the preselection nJet population once, then decay those jets for
    # each requested probability.
    n_jets = rng.choice(njet_values, size=args.n_events, p=njet_probabilities)

    sampled_counts = np.bincount(n_jets, minlength=int(njet_values.max()) + 1)
    print(
        f"Generated {args.n_events} preselection events using "
        f"{DATASET}/{NJET_HISTOGRAM}"
    )
    print("Sampled nJet populations:")
    for njet, count in zip(njet_values, sampled_counts[njet_values]):
        label = f"{njet}+" if njet == njet_values[-1] else str(njet)
        print(f"  nJet = {label:>2}: {count:5d}")
    print(f"\nMean nJet:  {n_jets.mean():.3f}")
    print(f"Data nMuon >= 1 efficiency: {data_efficiency:.4%}\n")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_results: list[tuple[str, np.ndarray, np.ndarray, str, str]] = []
    for model_index, (probability, source) in enumerate(unique_decay_rates):
        n_muons = rng.binomial(n=n_jets, p=probability)
        selected_n_muons = n_muons[n_muons > 0]
        if selected_n_muons.size == 0:
            print(f"Skipping p_muon={probability:g}: no toy events passed nMuon >= 1")
            continue

        # Match the workflow's inclusive final nMuon bin.
        selected_n_muons = np.minimum(selected_n_muons, nmuon_cap)
        toy_nmuon_hist = data_nmuon_hist.copy().reset()
        toy_nmuon_hist.fill(nMuon=selected_n_muons)
        # Preserve the observed data yield on the plot. Scale each selected toy
        # shape to the integral of the selected VR_loose nMuon distribution.
        toy_nmuon_hist = toy_nmuon_hist * (data_selected_events / selected_n_muons.size)
        toy_efficiency = selected_n_muons.size / args.n_events
        print(
            f"p_muon={probability:.6f} ({source}): mean nMuon before "
            f"selection={n_muons.mean():.4f}, efficiency={toy_efficiency:.4%}"
        )
        model_results.append(
            (
                (
                    f"Toy $p_\\mu={probability:.4g}$ ({source}, "
                    f"eff. {toy_efficiency:.1%})"
                ),
                np.asarray(toy_nmuon_hist.values(), dtype=float),
                np.asarray(toy_nmuon_hist.variances(), dtype=float),
                colors[model_index % len(colors)],
                "histogram",
            )
        )

    nmuon_values = np.rint(nmuon_edges[:-1]).astype(float)
    if args.add_exponential:
        amplitude, slope, exponential_values = fit_exponential_shape(
            nmuon_values, data_selected_counts
        )
        print(f"Exponential shape fit: A={amplitude:.6g}, B={slope:.6f}")
        model_results.append(
            (
                rf"$A e^{{-BN}}$ fit ($B={slope:.3f}$)",
                exponential_values,
                np.zeros_like(exponential_values),
                colors[len(model_results) % len(colors)],
                "exponential",
            )
        )

    hep.style.use("CMS")
    fig, (ax_main, ax_ratio, ax_pull) = plt.subplots(
        3,
        1,
        figsize=(11, 12),
        sharex=True,
        gridspec_kw={"height_ratios": (3.5, 1, 1), "hspace": 0.05},
    )
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.09, top=0.94)

    nmuon_centers = np.asarray(data_nmuon_hist.axes[0].centers)
    for label, values, variances, color, model_kind in model_results:
        if model_kind == "exponential":
            hep.histplot(
                values,
                bins=nmuon_edges,
                color=color,
                histtype="step",
                linestyle="--",
                linewidth=2.5,
                label=label,
                ax=ax_main,
            )
        else:
            hep.histplot(
                values,
                bins=nmuon_edges,
                yerr=np.sqrt(variances),
                color=color,
                histtype="step",
                label=label,
                ax=ax_main,
            )
        plot_ratio_and_pull(
            data_nmuon_hist,
            values,
            variances,
            ax_ratio,
            ax_pull,
            color=color,
        )

    data_nmuon_hist.plot(
        ax=ax_main,
        histtype="errorbar",
        color="black",
        label=f"Data VR loose (efficiency {data_efficiency:.1%})",
    )
    ax_main.set_yscale("log")
    ax_main.set_ylim(0.5, None)
    ax_main.set_ylabel("Events")
    ax_main.set_xlabel("")
    ax_main.legend()
    ax_main.tick_params(axis="x", which="both", labelbottom=False)
    hep.cms.label("Toy", ax=ax_main, data=False, rlabel="")

    ax_ratio.axhline(1, linestyle="--", color="gray")
    ax_ratio.set_ylabel("Data/model", fontsize=18)
    ax_ratio.set_xlabel("")
    ax_ratio.set_ylim(0, 2)
    ax_ratio.set_yticks((0, 1, 2))
    ax_ratio.tick_params(axis="y", labelsize=14)
    ax_ratio.tick_params(axis="x", which="both", labelbottom=False)
    ax_ratio.grid(axis="y", alpha=0.25)

    ax_pull.axhline(0, linestyle="--", color="gray")
    ax_pull.set_ylabel("Pull", fontsize=18)
    ax_pull.set_ylim(-5, 5)
    ax_pull.set_yticks((-5, 0, 5))
    ax_pull.tick_params(axis="y", labelsize=14)
    ax_pull.grid(axis="y", alpha=0.25)

    ax_pull.set_xticks(nmuon_centers)
    nmuon_labels = [str(value) for value in nmuon_values.astype(int)]
    nmuon_labels[-1] += "+"
    ax_pull.set_xticklabels(nmuon_labels)
    ax_pull.set_xlabel("nMuon")

    fig.savefig(OUTFILE, bbox_inches="tight")
    print(f"\nSaved plot to {OUTFILE}")


if __name__ == "__main__":
    main()
