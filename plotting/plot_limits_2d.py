import os

import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import mplhep as hep
import numpy as np
import uproot

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

input_path = "/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/combine_stuff/Mar2026/CMSSW_14_1_0_pre4/src"
tag = "full_analysis_Feb2026"

scan_points = [
    ("1.000", "0.250"),
    ("1.400", "0.350"),
    ("4.000", "1.000"),
    ("1.400", "1.400"),
    ("2.000", "2.000"),
    ("8.000", "2.000"),
    ("1.400", "2.800"),
    ("6.000", "3.000"),
    ("2.000", "4.000"),
    ("4.000", "4.000"),
    ("8.000", "4.000"),
    ("1.400", "5.600"),
    ("6.000", "6.000"),
    ("2.000", "8.000"),
    ("4.000", "8.000"),
    ("8.000", "8.000"),
    ("6.000", "12.000"),
    ("4.000", "16.000"),
    ("8.000", "16.000"),
    ("6.000", "24.000"),
    ("8.000", "32.000"),
]

suffix = ""

# Fallback used only if no available masses can be read from the combine outputs.
default_mS_targets = [125, 200, 300, 400, 500, 600, 800, 1000]
decay_modes = ("leptonic", "hadronic")
decay_mode_labels = {
    "leptonic": "Leptonic decays",
    "hadronic": "Hadronic decays",
}


def format_param(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def get_median_mu(input_path, scan_point, mS):
    """Return median expected signal strength (mu) at the given mS, or None if unavailable."""
    fname = f"{input_path}/higgsCombine_scale0.0001_{scan_point}{suffix}.AsymptoticLimits.root"
    if not os.path.exists(fname):
        return None
    f = uproot.open(fname)
    tree = f["limit"]
    limits = tree["limit"].array(library="np")
    masses = tree["mh"].array(library="np")
    quantiles = tree["quantileExpected"].array(library="np")

    mask = np.isclose(masses, mS) & np.isclose(quantiles, 0.5)
    if not np.any(mask):
        return None
    # Signal was injected at scale=0.0001, so divide by 1/0.0001=10000 to get true mu
    return float(limits[mask][0]) * 0.0001


def get_available_mS(input_path, channels=("leptonic", "hadronic")):
    """Return all mS values with median expected limits in the available ROOT files."""
    mS_values = set()
    for m_phi, T in scan_points:
        for ch in channels:
            fname = (
                f"{input_path}/higgsCombine_scale0.0001_"
                f"mPhi{m_phi}_T{T}_{ch}{suffix}.AsymptoticLimits.root"
            )
            if not os.path.exists(fname):
                continue

            f = uproot.open(fname)
            tree = f["limit"]
            masses = tree["mh"].array(library="np")
            quantiles = tree["quantileExpected"].array(library="np")
            mS_values.update(float(mS) for mS in masses[np.isclose(quantiles, 0.5)])

    if not mS_values:
        return default_mS_targets

    return sorted(mS_values)


def build_grid(data):
    """Build a 2D grid (T/mPhi vs mPhi) from a list of (mPhi, T, mu) tuples."""
    x_values = sorted({m_phi for m_phi, _, _ in data})
    y_values = sorted({round(T / m_phi, 6) for m_phi, T, _ in data})

    x_index = {v: i for i, v in enumerate(x_values)}
    y_index = {v: i for i, v in enumerate(y_values)}

    grid = np.full((len(y_values), len(x_values)), np.nan)
    for m_phi, T, mu in data:
        ratio = round(T / m_phi, 6)
        grid[y_index[ratio], x_index[m_phi]] = mu

    return grid, x_values, y_values


def get_lumi_label():
    if suffix == "_Run2_13TeV":
        return r"$118\,fb^{-1}$ ($13\,TeV$)"
    if suffix == "_Run3_13p6TeV":
        return r"$62.4\,fb^{-1}$ ($13.6\,TeV$)"
    return r"$118\,fb^{-1}$ ($13\,TeV$) + $62.4\,fb^{-1}$ ($13.6\,TeV$)"


def plot_limits_2d(mS_target, decay_mode, output_dir):
    data = []
    for m_phi, T in scan_points:
        mu = get_median_mu(input_path, f"mPhi{m_phi}_T{T}_{decay_mode}", mS_target)
        if mu is None:
            print(
                f"No {decay_mode} limit found for mPhi={m_phi}, T={T} "
                f"at mS={mS_target}. Skipping."
            )
            continue
        data.append((float(m_phi), float(T), mu))

    if not data:
        print(
            f"No {decay_mode} limits found for mS={format_param(mS_target)}. "
            "Skipping plot."
        )
        return False

    grid, x_values, y_values = build_grid(data)
    log_grid = np.where(np.isfinite(grid), np.log10(grid), np.nan)
    masked_log_grid = np.ma.masked_invalid(log_grid)

    lumi_label = (
        get_lumi_label()
        # + "\n"
        # + decay_mode_labels[decay_mode]
        # + "\n"
        # + r"$m_S = "
        # + format_param(mS_target)
        # + r"\,\mathrm{GeV}$"
    )

    fig, ax = plt.subplots(figsize=(13, 9))

    # Diverging colormap centered at log10(mu)=0, i.e. mu=1
    finite_vals = masked_log_grid.compressed()
    vmax = (
        max(abs(finite_vals.min()), abs(finite_vals.max())) if len(finite_vals) else 1.0
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e5e7eb")
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)

    image = ax.imshow(
        masked_log_grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )

    # Cell grid lines
    ax.set_xticks(np.arange(-0.5, len(x_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_values), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([format_param(v) for v in x_values])
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([format_param(v) for v in y_values])

    # Annotate each cell
    for yi, ratio in enumerate(y_values):
        for xi, m_phi in enumerate(x_values):
            mu = grid[yi, xi]
            if np.isfinite(mu):
                log_mu = np.log10(mu)
                text_color = "white" if abs(log_mu) > 0.4 * vmax else "black"
                ax.text(
                    xi,
                    yi,
                    f"{log_mu:.2f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color=text_color,
                )
            else:
                ax.text(
                    xi, yi, "–", ha="center", va="center", fontsize=14, color="#6b7280"
                )

    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(
        r"95% CL expected upper limit on $\log_{10}(r)$", fontsize=24, labelpad=15
    )
    cbar.ax.axhline(0, color="k", lw=2, ls="--")

    ax.set_xlabel(r"$m_{\phi}$ (GeV)")
    ax.set_ylabel(r"$T\,/\,m_{\phi}$")

    hep.cms.label(
        llabel="Preliminary",
        data=True,
        rlabel=lumi_label,
        ax=ax,
    )

    plt.tight_layout()
    out = f"{output_dir}/limits_2d_{decay_mode}_mS{format_param(mS_target)}.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    return True


if __name__ == "__main__":
    output_dir = f"limit_plots_2d/{tag}"
    os.makedirs(output_dir, exist_ok=True)

    for decay_mode in decay_modes:
        for mS_target in get_available_mS(input_path, channels=(decay_mode,)):
            plot_limits_2d(mS_target, decay_mode, output_dir)
