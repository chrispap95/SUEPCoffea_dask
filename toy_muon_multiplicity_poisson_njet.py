"""Muon-multiplicity toy with a Poisson number of muons per observed jet.

Events are sampled from the observed pre-muon 2018 nJet distribution.  Each
jet independently produces ``Poisson(mu)`` loose VR muons, so the total number
of muons in an event with ``nJet`` jets is exactly ``Poisson(nJet * mu)``.  The
toy applies the data VR requirement of at least one additional loose muon and
compares the selected nMuon shape with data.
"""

import argparse

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

from toy_muon_multiplicity_realistic_njet import (
    DATASET,
    DEFAULT_N_EVENTS,
    DEFAULT_RNG_SEED,
    NJET_HISTOGRAM,
    fit_exponential_shape,
    load_data_distributions,
    plot_ratio_and_pull,
)

OUTFILE = "toy_muon_multiplicity_poisson_njet.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mu",
        type=float,
        nargs="+",
        default=[],
        metavar="RATE",
        help="One or more mean muon yields per jet to plot.",
    )
    parser.add_argument(
        "--fit-mu",
        choices=("full", "shape", "both"),
        nargs="?",
        const="both",
        help=(
            "Fit mu using the full distribution including zero muons, the "
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
    parser.add_argument(
        "--output",
        default=OUTFILE,
        help=f"Output plot filename (default: {OUTFILE}).",
    )
    args = parser.parse_args()
    if args.n_events <= 0:
        parser.error("--n-events must be positive")
    if any(not np.isfinite(rate) or rate <= 0 for rate in args.mu):
        parser.error("every --mu value must be finite and strictly positive")
    return args


def poisson_mixture_probabilities(
    njet_values: np.ndarray,
    njet_probabilities: np.ndarray,
    mu: float,
    nmuon_cap: int,
) -> np.ndarray:
    """Return probabilities for nMuon = 0, 1, ..., cap-1, cap+.

    The final element includes the Poisson overflow, matching the inclusive
    final bin in the analysis histogram.
    """
    nmuon_values = np.arange(nmuon_cap)
    probabilities = np.zeros(nmuon_cap + 1, dtype=float)
    for njet, njet_probability in zip(njet_values, njet_probabilities):
        event_rate = float(njet) * mu
        probabilities[:-1] += njet_probability * stats.poisson.pmf(
            nmuon_values, event_rate
        )
        probabilities[-1] += njet_probability * stats.poisson.sf(
            nmuon_cap - 1, event_rate
        )
    return probabilities


def fit_muon_rate(
    njet_values: np.ndarray,
    njet_probabilities: np.ndarray,
    data_preselection_events: float,
    data_selected_counts: np.ndarray,
    nmuon_cap: int,
    *,
    shape_only: bool,
) -> float:
    """Fit the Poisson mean per jet with a binned multinomial likelihood."""
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

    def negative_log_likelihood(log_mu: float) -> float:
        model = poisson_mixture_probabilities(
            njet_values, njet_probabilities, np.exp(log_mu), nmuon_cap
        )
        if shape_only:
            selected_probability = model[1:].sum()
            if selected_probability <= 0:
                return np.inf
            model = model[1:] / selected_probability
        return float(-np.dot(observed, np.log(np.clip(model, 1e-300, None))))

    # Optimize log(mu) so positivity is automatic.  This very broad interval
    # covers rates from effectively zero through 100 muons per jet.
    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(np.log(1e-10), np.log(100.0)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"The mu fit failed: {result.message}")
    return float(np.exp(result.x))


def validate_nmuon_axis(data_nmuon_hist: hist.Hist) -> tuple[np.ndarray, int]:
    """Validate and return the selected nMuon binning and overflow cap."""
    nmuon_edges = np.asarray(data_nmuon_hist.axes["nMuon"].edges)
    if not (
        np.allclose(np.diff(nmuon_edges), 1.0)
        and np.allclose(nmuon_edges, np.rint(nmuon_edges))
        and np.isclose(nmuon_edges[0], 1.0)
    ):
        raise ValueError(
            "Expected selected unit-width integer nMuon bins starting at 1, "
            f"but found edges {nmuon_edges}"
        )
    return nmuon_edges, int(round(nmuon_edges[-2]))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    (
        njet_values,
        njet_probabilities,
        data_preselection_events,
        data_nmuon_hist,
    ) = load_data_distributions()

    nmuon_edges, nmuon_cap = validate_nmuon_axis(data_nmuon_hist)
    data_selected_counts = np.asarray(data_nmuon_hist.values(), dtype=float)
    data_selected_events = data_selected_counts.sum()
    if data_selected_events <= 0:
        raise ValueError("The selected data nMuon histogram is empty")
    data_efficiency = data_selected_events / data_preselection_events

    rates: list[tuple[float, str]] = [(rate, "manual") for rate in args.mu]
    if args.fit_mu in ("full", "both"):
        fitted_rate = fit_muon_rate(
            njet_values,
            njet_probabilities,
            data_preselection_events,
            data_selected_counts,
            nmuon_cap,
            shape_only=False,
        )
        rates.append((fitted_rate, "full fit"))
        print(f"Full-distribution maximum-likelihood mu: {fitted_rate:.6f}")
    if args.fit_mu in ("shape", "both"):
        fitted_rate = fit_muon_rate(
            njet_values,
            njet_probabilities,
            data_preselection_events,
            data_selected_counts,
            nmuon_cap,
            shape_only=True,
        )
        rates.append((fitted_rate, "shape fit"))
        print(f"Selected-shape maximum-likelihood mu: {fitted_rate:.6f}")

    unique_rates: list[tuple[float, str]] = []
    for rate, source in rates:
        if not any(
            np.isclose(rate, existing, rtol=0, atol=1e-10)
            for existing, _ in unique_rates
        ):
            unique_rates.append((rate, source))

    # Sum_j Poisson(mu) is Poisson(nJet*mu), so one vectorized draw per event
    # is statistically identical to explicitly looping over every sampled jet.
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
    print(f"\nMean nJet: {n_jets.mean():.3f}")
    print(f"Data nMuon >= 1 efficiency: {data_efficiency:.4%}\n")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_results: list[tuple[str, np.ndarray, np.ndarray, str, str]] = []
    for model_index, (rate, source) in enumerate(unique_rates):
        n_muons = rng.poisson(n_jets * rate)
        selected_n_muons = n_muons[n_muons > 0]
        if selected_n_muons.size == 0:
            print(f"Skipping mu={rate:g}: no toy events passed nMuon >= 1")
            continue

        selected_n_muons = np.minimum(selected_n_muons, nmuon_cap)
        toy_nmuon_hist = data_nmuon_hist.copy().reset()
        toy_nmuon_hist.fill(nMuon=selected_n_muons)
        toy_nmuon_hist = toy_nmuon_hist * (data_selected_events / selected_n_muons.size)
        toy_efficiency = selected_n_muons.size / args.n_events
        print(
            f"mu={rate:.6f} ({source}): mean nMuon before "
            f"selection={n_muons.mean():.4f}, efficiency={toy_efficiency:.4%}"
        )
        model_results.append(
            (
                (
                    f"Poisson/jet toy $\\mu={rate:.4g}$ ({source}, "
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

    fig.savefig(args.output, bbox_inches="tight")
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
