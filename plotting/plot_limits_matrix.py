import os
from collections import defaultdict
from math import ceil

import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import mplhep as hep
import numpy as np
import uproot
from scipy.interpolate import interp1d  # type: ignore[import]

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

# Definitions
input_path = "/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/combine_stuff/Mar2026/CMSSW_14_1_0_pre4/src"
tag = "full_analysis_Feb2026"

mass_vals = np.linspace(100, 1050, 1000)
masses = np.array([125, 200, 300, 400, 500, 600, 800, 1000])
cross_sections = np.array([45.2, 16.9, 6.59, 3.19, 1.71, 1.0, 0.402, 0.185])

green = "#607641"
yellow = "#F5BB54"

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

# Leave this empty for Run 2 + Run 3
# Otherwise, set to either "_Run2_13TeV" or "_Run3_13p6TeV"
suffix = ""

# Maximum number of columns in the matrix layout
MAX_COLS = 3


def log_interp1d(xx, yy, kind="linear"):
    logx = np.log(xx)
    logy = np.log(yy)
    lin_interp = interp1d(
        logx, logy, bounds_error=False, fill_value="extrapolate", kind=kind
    )
    log_interp = lambda zz: np.power(np.e, lin_interp(np.log(zz)))
    return log_interp


def get_quantiles(tree, scale):
    limits = tree["limit"].array(library="np")
    mS = tree["mh"].array(library="np")
    quantiles = tree["quantileExpected"].array(library="np")

    common_mS = np.intersect1d(mS[quantiles == 0.025], mS[quantiles == 0.16])
    common_mS = np.intersect1d(common_mS, mS[quantiles == 0.5])
    common_mS = np.intersect1d(common_mS, mS[quantiles == 0.84])
    common_mS = np.intersect1d(common_mS, mS[quantiles == 0.975])
    select_mS = np.in1d(mS, common_mS)

    limits = limits[select_mS]
    mS = mS[select_mS]
    quantiles = quantiles[select_mS]

    point_masses = mS[quantiles == 0.025]
    qt_0p025 = limits[quantiles == 0.025]
    qt_0p16 = limits[quantiles == 0.16]
    qt_0p5 = limits[quantiles == 0.5]
    qt_0p84 = limits[quantiles == 0.84]
    qt_0p975 = limits[quantiles == 0.975]

    sort_by_mass = np.argsort(point_masses)
    point_masses = point_masses[sort_by_mass]
    qt_0p025 = qt_0p025[sort_by_mass]
    qt_0p16 = qt_0p16[sort_by_mass]
    qt_0p5 = qt_0p5[sort_by_mass]
    qt_0p84 = qt_0p84[sort_by_mass]
    qt_0p975 = qt_0p975[sort_by_mass]

    xs_map = {
        125: 45.2,
        200: 16.9,
        300: 6.59,
        400: 3.19,
        500: 1.71,
        600: 1.0,
        800: 0.402,
        1000: 0.185,
    }
    for i, mS_val in enumerate(point_masses):
        qt_0p025[i] *= xs_map[mS_val] / scale
        qt_0p16[i] *= xs_map[mS_val] / scale
        qt_0p5[i] *= xs_map[mS_val] / scale
        qt_0p84[i] *= xs_map[mS_val] / scale
        qt_0p975[i] *= xs_map[mS_val] / scale

    cl_0p025_smooth = log_interp1d(point_masses, qt_0p025)
    cl_0p16_smooth = log_interp1d(point_masses, qt_0p16)
    cl_0p5_smooth = log_interp1d(point_masses, qt_0p5)
    cl_0p84_smooth = log_interp1d(point_masses, qt_0p84)
    cl_0p975_smooth = log_interp1d(point_masses, qt_0p975)

    return (
        point_masses,
        qt_0p5,
        (
            cl_0p025_smooth,
            cl_0p16_smooth,
            cl_0p5_smooth,
            cl_0p84_smooth,
            cl_0p975_smooth,
        ),
    )


def plot_limit_on_ax(ax, input_path, scan_point, mass_vals):
    """Draw limit curves for one scan point onto the given Axes. Returns False if file missing."""
    fname = f"{input_path}/higgsCombine_scale0.0001_{scan_point}{suffix}.AsymptoticLimits.root"
    if not os.path.exists(fname):
        print(f"File not found. Skipping {scan_point}")
        return False

    f = uproot.open(fname)
    pt_masses, cl_0p5, cl_smooth = get_quantiles(f["limit"], 10000)

    max_mass = np.max(pt_masses)
    min_mass = np.min(pt_masses)
    mv = mass_vals[(min_mass - 50 <= mass_vals) & (mass_vals <= max_mass + 50)]

    if "leptonic" in scan_point:
        color = "b"
        m_Aprime = 0.5
        br_mumu = 40
    elif "hadronic" in scan_point:
        color = "r"
        m_Aprime = 0.7
        br_mumu = 15
    else:
        raise ValueError("Invalid scan point")

    ax.plot(pt_masses, cl_0p5, color=color, marker="o", ms=8, ls="", zorder=3)
    ax.plot(
        mv,
        cl_smooth[2](mv),
        color=color,
        ls="--",
        lw=3,
        label=r"Median exp. $m_{A'}="
        + str(m_Aprime)
        + r"\,GeV$"
        + "\n"
        + r"$B(A'\rightarrow\mu^+\mu^-)="
        + str(br_mumu)
        + r"\%$",
        zorder=2,
    )
    ax.fill_between(mv, cl_smooth[3](mv), cl_smooth[1](mv), color=green, zorder=1)
    ax.fill_between(mv, cl_smooth[4](mv), cl_smooth[0](mv), color=yellow, zorder=0)
    return True


def decorate_ax(ax, xs_spl, m_phi, T, lumi_label, draw_cms=True):
    ax.plot(
        mass_vals,
        xs_spl(mass_vals),
        color="k",
        ls="-.",
        label="Theory",
        lw=3,
        zorder=4,
    )

    # Dummy patches for legend
    ax.fill_between([], 0, 0, color=green, label="68% expected")
    ax.fill_between([], 0, 0, color=yellow, label="95% expected")

    ax.set_xlim(100, 1050)
    ax.set_ylim(1e-6, 1e3)
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{S}$ (GeV)")
    ax.set_ylabel(r"$\sigma(\sqrt{s}=13\,TeV)$ (pb)")
    ax.legend(loc=1, fontsize=14)
    ax.text(
        320,
        100,
        "GluGluToSUEP\n"
        + r"$m_{\phi}="
        + f"{float(m_phi):g}"
        + r"\,$GeV, $T="
        + f"{float(T):g}"
        + r"\,$GeV",
        ha="center",
        fontsize=16,
    )
    if draw_cms:
        hep.cms.label(llabel="Preliminary", data=True, rlabel=lumi_label, ax=ax)


if __name__ == "__main__":
    os.makedirs(f"limit_plots_matrix/{tag}", exist_ok=True)

    xs_spl = log_interp1d(masses, cross_sections)

    if suffix == "_Run2_13TeV":
        lumi_label = r"$118\,fb^{-1}$ ($13\,TeV$)"
    elif suffix == "_Run3_13p6TeV":
        lumi_label = r"$62.4\,fb^{-1}$ ($13.6\,TeV$)"
    else:
        lumi_label = r"$118\,fb^{-1}$ ($13\,TeV$) + $62.4\,fb^{-1}$ ($13.6\,TeV$)"

    # Group scan points by temperature, preserving insertion order
    by_temp = defaultdict(list)
    for m_phi, T in scan_points:
        by_temp[T].append(m_phi)

    for T, m_phi_list in by_temp.items():
        n = len(m_phi_list)
        ncols = min(n, MAX_COLS)
        nrows = ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(11 * ncols, 10.5 * nrows),
            squeeze=False,
        )

        for idx, m_phi in enumerate(m_phi_list):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]

            plot_limit_on_ax(ax, input_path, f"mPhi{m_phi}_T{T}_leptonic", mass_vals)
            plot_limit_on_ax(ax, input_path, f"mPhi{m_phi}_T{T}_hadronic", mass_vals)
            decorate_ax(ax, xs_spl, m_phi, T, lumi_label, draw_cms=(idx == 0))

        # Hide any unused axes in the last row
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        T_str = T.replace(".", "p")
        plt.tight_layout()
        out = f"limit_plots_matrix/{tag}/limits_matrix_T{T_str}.pdf"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")
