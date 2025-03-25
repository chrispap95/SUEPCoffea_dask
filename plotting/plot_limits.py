import os

import matplotlib as mpl  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import mplhep as hep
import numpy as np
import uproot
from rich.progress import track  # type: ignore[import]
from scipy.interpolate import interp1d  # type: ignore[import]

hep.style.use(hep.style.CMS)
mpl.rcParams["figure.facecolor"] = "white"

# Definitions
input_path = "/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/combine_stuff/Jan2025/CMSSW_11_3_4/src"

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


def log_interp1d(xx, yy, kind="linear"):
    logx = np.log(xx)
    logy = np.log(yy)
    lin_interp = interp1d(
        logx, logy, bounds_error=False, fill_value="extrapolate", kind=kind
    )
    log_interp = lambda zz: np.power(np.e, lin_interp(np.log(zz)))
    return log_interp


def get_quantiles(tree, scale):
    # Import branches
    limits = tree["limit"].array(library="np")
    mS = tree["mh"].array(library="np")
    quantiles = tree["quantileExpected"].array(library="np")

    # Find the masses that are common to all quantiles
    common_mS = np.intersect1d(mS[quantiles == 0.025], mS[quantiles == 0.16])
    common_mS = np.intersect1d(common_mS, mS[quantiles == 0.5])
    common_mS = np.intersect1d(common_mS, mS[quantiles == 0.84])
    common_mS = np.intersect1d(common_mS, mS[quantiles == 0.975])
    select_mS = np.in1d(mS, common_mS)

    # Filter out the masses that are not common to all quantiles
    limits = limits[select_mS]
    mS = mS[select_mS]
    quantiles = quantiles[select_mS]

    # Get the quantiles
    masses = mS[quantiles == 0.025]
    qt_0p025 = limits[quantiles == 0.025]
    qt_0p16 = limits[quantiles == 0.16]
    qt_0p5 = limits[quantiles == 0.5]
    qt_0p84 = limits[quantiles == 0.84]
    qt_0p975 = limits[quantiles == 0.975]

    # Make sure these are sorted by mS
    sort_by_mass = np.argsort(masses)
    masses = masses[sort_by_mass]
    qt_0p025 = qt_0p025[sort_by_mass]
    qt_0p16 = qt_0p16[sort_by_mass]
    qt_0p5 = qt_0p5[sort_by_mass]
    qt_0p84 = qt_0p84[sort_by_mass]
    qt_0p975 = qt_0p975[sort_by_mass]

    # Convert signal strength to CL
    cross_sections = {
        125: 45.2,
        200: 16.9,
        300: 6.59,
        400: 3.19,
        500: 1.71,
        600: 1.0,
        800: 0.402,
        1000: 0.185,
    }
    for i, mS in enumerate(masses):
        qt_0p025[i] *= cross_sections[mS] / scale
        qt_0p16[i] *= cross_sections[mS] / scale
        qt_0p5[i] *= cross_sections[mS] / scale
        qt_0p84[i] *= cross_sections[mS] / scale
        qt_0p975[i] *= cross_sections[mS] / scale

    # Smoothen CLs
    cl_0p025_smooth = log_interp1d(masses, qt_0p025)
    cl_0p16_smooth = log_interp1d(masses, qt_0p16)
    cl_0p5_smooth = log_interp1d(masses, qt_0p5)
    cl_0p84_smooth = log_interp1d(masses, qt_0p84)
    cl_0p975_smooth = log_interp1d(masses, qt_0p975)

    return (
        masses,
        qt_0p5,
        (
            cl_0p025_smooth,
            cl_0p16_smooth,
            cl_0p5_smooth,
            cl_0p84_smooth,
            cl_0p975_smooth,
        ),
    )


def plot_limit(input_path, scan_point, mass_vals, masses):
    if not os.path.exists(
        f"{input_path}/higgsCombine_scale0.0001_{scan_point}.AsymptoticLimits.root"
    ):
        print(f"File not found. Skipping {scan_point}")
        return
    f = uproot.open(
        f"{input_path}/higgsCombine_scale0.0001_{scan_point}.AsymptoticLimits.root"
    )

    limits_tree = f["limit"]
    masses, cl_0p5, cl_smooth = get_quantiles(limits_tree, 10000)

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

    # Plot median expected
    plt.plot(masses, cl_0p5, color=color, marker="o", ms=8, ls="", zorder=3)
    plt.plot(
        mass_vals,
        cl_smooth[2](mass_vals),
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

    # Plot bands
    plt.fill_between(
        mass_vals,
        cl_smooth[3](mass_vals),
        cl_smooth[1](mass_vals),
        color=green,
        zorder=1,
    )
    plt.fill_between(
        mass_vals,
        cl_smooth[4](mass_vals),
        cl_smooth[0](mass_vals),
        color=yellow,
        zorder=0,
    )


if __name__ == "__main__":
    xs_spl = log_interp1d(masses, cross_sections)

    for m_phi, T in track(scan_points):
        fig, ax = plt.subplots(figsize=(10.5, 10.5))

        # Theory
        plt.plot(
            mass_vals,
            xs_spl(mass_vals),
            color="k",
            ls="-.",
            label="Theory",
            lw=3,
            zorder=4,
        )

        # Plot limits
        plot_limit(input_path, f"mPhi{m_phi}_T{T}_leptonic", mass_vals, masses)
        plot_limit(input_path, f"mPhi{m_phi}_T{T}_hadronic", mass_vals, masses)

        # This is just for the legend entries
        plt.fill_between(
            [],
            0,
            0,
            color=green,
            label="68% expected",
            zorder=1,
        )
        plt.fill_between(
            [],
            0,
            0,
            color=yellow,
            label="95% expected",
            zorder=0,
        )

        plt.legend(loc=1)
        plt.xlim(100, 1050)
        plt.ylim(1e-5, 1e3)
        plt.yscale("log")
        ax.set_xlabel(r"$m_{S}$ (GeV)")
        ax.set_ylabel(r"$\sigma$ (pb)")
        hep.cms.label(llabel="Preliminary", data=True, lumi=59.8, ax=ax)
        plt.text(
            320,
            100,
            "GluGluToSUEP\n"
            + r"$m_{\phi}="
            + f"{float(m_phi):g}"
            + r"\,$GeV, $T="
            + f"{float(T):g}"
            + r"\,$GeV",
            ha="center",
            fontsize=22,
        )
        plt.tight_layout()
        plt.savefig(
            f"limit_plots/limits_mPhi{m_phi.replace('.', 'p')}_T{T.replace('.', 'p')}.pdf",
            bbox_inches="tight",
        )
        plt.close()
