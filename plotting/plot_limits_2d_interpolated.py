import os

import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import mplhep as hep
import numpy as np
from plot_limits_2d import (
    decay_mode_labels,
    decay_modes,
    format_param,
    get_available_mS,
    get_lumi_label,
    get_median_mu,
    input_path,
    scan_points,
    tag,
)
from scipy.interpolate import griddata  # type: ignore[import]
from scipy.spatial import QhullError  # type: ignore[import]

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

GRID_SIZE = 300


def collect_data(mS_target, decay_mode):
    data = []
    for m_phi, T in scan_points:
        mu = get_median_mu(input_path, f"mPhi{m_phi}_T{T}_{decay_mode}", mS_target)
        if mu is None:
            print(
                f"No {decay_mode} limit found for mPhi={m_phi}, T={T} "
                f"at mS={mS_target}. Skipping."
            )
            continue
        m_phi_float = float(m_phi)
        T_float = float(T)
        data.append((m_phi_float, np.log2(T_float / m_phi_float), mu))

    return data


def interpolate_log_mu(data):
    x = np.array([m_phi for m_phi, _, _ in data])
    y = np.array([ratio for _, ratio, _ in data])
    log_mu = np.log10([mu for _, _, mu in data])

    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), GRID_SIZE),
        np.linspace(y.min(), y.max(), GRID_SIZE),
    )

    try:
        grid_log_mu = griddata((x, y), log_mu, (grid_x, grid_y), method="linear")
    except QhullError:
        grid_log_mu = np.full_like(grid_x, np.nan, dtype=float)

    return x, y, log_mu, grid_x, grid_y, np.ma.masked_invalid(grid_log_mu)


def plot_limits_2d_interpolated(mS_target, decay_mode, output_dir):
    data = collect_data(mS_target, decay_mode)
    if len(data) < 3:
        print(
            f"Need at least three {decay_mode} points to interpolate "
            f"mS={format_param(mS_target)}. Skipping plot."
        )
        return False

    x, y, log_mu, grid_x, grid_y, masked_grid = interpolate_log_mu(data)

    finite_vals = log_mu[np.isfinite(log_mu)]
    vmax = (
        max(abs(finite_vals.min()), abs(finite_vals.max())) if len(finite_vals) else 1.0
    )
    if vmax == 0:
        vmax = 1.0

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e5e7eb")
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
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
    ax.set_facecolor("#e5e7eb")

    image = ax.imshow(
        masked_grid,
        origin="lower",
        extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
    )

    ax.scatter(
        x,
        y,
        c=log_mu,
        cmap=cmap,
        norm=norm,
        s=180,
        edgecolors="black",
        linewidths=1.5,
        zorder=4,
    )

    x_pad = 0.05 * (x.max() - x.min())
    y_pad = 0.05 * (y.max() - y.min())
    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
    ax.set_ylim(y.min() - y_pad, y.max() + y_pad)

    ax.set_xticks(sorted(set(x)))
    ax.set_xticklabels([format_param(v) for v in sorted(set(x))])
    ax.set_yticks(sorted(set(y)))
    ax.set_yticklabels([format_param(v) for v in sorted(set(y))])
    ax.grid(color="white", linewidth=1.0, alpha=0.7)

    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label(
        r"95% CL expected upper limit on $\log_{10}(r)$", fontsize=24, labelpad=15
    )
    cbar.ax.axhline(0, color="k", lw=2, ls="--")

    ax.set_xlabel(r"$m_{\phi}$ (GeV)")
    ax.set_ylabel(r"$\log_{2}(T\,/\,m_{\phi})$")

    hep.cms.label(
        llabel="Preliminary",
        data=True,
        rlabel=lumi_label,
        ax=ax,
    )

    plt.tight_layout()
    out = (
        f"{output_dir}/limits_2d_interpolated_{decay_mode}_"
        f"mS{format_param(mS_target)}.pdf"
    )
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    return True


if __name__ == "__main__":
    output_dir = f"limit_plots_2d_interpolated/{tag}"
    os.makedirs(output_dir, exist_ok=True)

    for decay_mode in decay_modes:
        for mS_target in get_available_mS(input_path, channels=(decay_mode,)):
            plot_limits_2d_interpolated(mS_target, decay_mode, output_dir)
