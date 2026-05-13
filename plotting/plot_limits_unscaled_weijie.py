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
input_path = "/uscms/home/chpapage/nobackup/SUEPs/MuonTriggers/combine_stuff/Mar2026/CMSSW_14_1_0_pre4/src"
tag = "full_analysis_Feb2026"

input_other = "combine_from_weijie"

mass_vals = np.linspace(100, 1050, 1000)
masses = np.array([125, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

green = "#607641"
yellow = "#F5BB54"

scan_points = [
    ("4.000", "4.000"),
    ("8.000", "8.000"),
]

scale = 10000

# Leave this empty for Run 2 + Run 3
# Otherwise, set to either "_Run2_13TeV" or "_Run3_13p6TeV"
suffix = ""


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

    for i, mS in enumerate(masses):
        qt_0p025[i] /= scale
        qt_0p16[i] /= scale
        qt_0p5[i] /= scale
        qt_0p84[i] /= scale
        qt_0p975[i] /= scale

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
        f"{input_path}/higgsCombine_scale0.0001_{scan_point}{suffix}.AsymptoticLimits.root"
    ):
        print(f"File not found. Skipping {scan_point}")
        return
    f = uproot.open(
        f"{input_path}/higgsCombine_scale0.0001_{scan_point}{suffix}.AsymptoticLimits.root"
    )

    limits_tree = f["limit"]
    masses, cl_0p5, cl_smooth = get_quantiles(limits_tree, scale)

    # Get max and min values of the CLs and mask the mass values
    max_mass = np.max(masses)
    min_mass = np.min(masses)
    mass_vals_new = mass_vals[
        (min_mass - 50 <= mass_vals) & (mass_vals <= max_mass + 50)
    ]

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
        mass_vals_new,
        cl_smooth[2](mass_vals_new),
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
        mass_vals_new,
        cl_smooth[3](mass_vals_new),
        cl_smooth[1](mass_vals_new),
        color=green,
        zorder=1,
    )
    plt.fill_between(
        mass_vals_new,
        cl_smooth[4](mass_vals_new),
        cl_smooth[0](mass_vals_new),
        color=yellow,
        zorder=0,
    )


def plot_legacy(input_path, search, mPhi, T, decay, mass_vals, masses):
    obs_lims = np.array([])

    trig_str = ""
    if search == "scouting":
        trig_str = "HT400"
    if search == "offline":
        trig_str = "HT1000"
    prefix_legacy = f"{input_path}/{search}/higgsCombineGluGluToSUEP_{trig_str}_T{T.replace('.', 'p').replace('000', '00')}"
    suffix_legacy = (
        f"mPhi{mPhi}_T{T}_mode{decay}_TuneCP5_13TeV-pythia8.HybridNew.mH125.root"
    )

    for mass in masses:
        if not os.path.exists(f"{prefix_legacy}_mS{mass:.3f}_{suffix_legacy}"):
            print(
                f"File not found. Skipping {prefix_legacy}_mS{mass:.3f}_{suffix_legacy}"
            )
            return
        f = uproot.open(f"{prefix_legacy}_mS{mass:.3f}_{suffix_legacy}")
        limits_tree = f["limit"]
        limits = limits_tree["limit"].array(library="np")  # type: ignore[array-type]
        obs_lims = np.append(obs_lims, limits)

    # Smoothen CLs
    obs_lims_smooth = log_interp1d(masses, obs_lims)

    # Get max and min values of the CLs and mask the mass values
    max_mass = np.max(masses)
    min_mass = np.min(masses)
    mass_vals_new = mass_vals[
        (min_mass - 50 <= mass_vals) & (mass_vals <= max_mass + 50)
    ]

    if decay == "leptonic":
        color = "b"
        m_Aprime = 0.5
        br_mumu = 40
    elif decay == "hadronic":
        color = "r"
        m_Aprime = 0.7
        br_mumu = 15
    else:
        raise ValueError("Invalid scan point")

    # Plot median expected
    plt.plot(masses, obs_lims, color=color, marker="o", ms=8, ls="", zorder=3)
    plt.plot(
        mass_vals_new,
        obs_lims_smooth(mass_vals_new),
        color=color,
        ls="-" if search == "offline" else ":",
        lw=3,
        label=search
        + r" Obs. $m_{A'}="
        + str(m_Aprime)
        + r"\,GeV$"
        + "\n"
        + r"$B(A'\rightarrow\mu^+\mu^-)="
        + str(br_mumu)
        + r"\%$",
        zorder=2,
    )


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists(f"limit_plots_weijie/{tag}"):
        os.makedirs(f"limit_plots_weijie/{tag}")

    for m_phi, T in track(scan_points):
        fig, ax = plt.subplots(figsize=(11, 10.5))

        # Theory
        plt.hlines(
            1,
            100,
            2050,
            color="k",
            ls="-.",
            label="Theory",
            lw=3,
            zorder=4,
        )

        # Plot limits
        plot_limit(input_path, f"mPhi{m_phi}_T{T}_leptonic", mass_vals, masses)
        plot_limit(input_path, f"mPhi{m_phi}_T{T}_hadronic", mass_vals, masses)

        # # This is just for the legend entries
        # plt.fill_between(
        #     [],
        #     0,
        #     0,
        #     color=green,
        #     label="68% expected",
        #     zorder=1,
        # )
        # plt.fill_between(
        #     [],
        #     0,
        #     0,
        #     color=yellow,
        #     label="95% expected",
        #     zorder=0,
        # )

        plot_legacy(input_other, "offline", m_phi, T, "leptonic", mass_vals, masses)
        plot_legacy(input_other, "offline", m_phi, T, "hadronic", mass_vals, masses)
        plot_legacy(input_other, "scouting", m_phi, T, "leptonic", mass_vals, masses)
        plot_legacy(input_other, "scouting", m_phi, T, "hadronic", mass_vals, masses)

        # handles, labels = plt.gca().get_legend_handles_labels()
        # legend1 = plt.gca().legend(
        #     handles[:1],
        #     labels[:1],
        #     loc="upper left",
        #     frameon=False,
        # )
        # legend2 = plt.gca().legend(
        #     handles[1:],
        #     labels[1:],
        #     loc="upper right",
        #     frameon=False,
        # )
        # plt.gca().add_artist(legend1)

        # Create custom legends
        (plt_theory,) = plt.plot([], [], c="black", ls="-.", lw=3, label="Theory")
        (plt_median_exp,) = plt.plot(
            [], [], c="black", ls="--", lw=3, label="Median expected"
        )
        plt_68exp = plt.fill_between(
            [],
            0,
            0,
            color=green,
            label="68% expected",
            zorder=1,
        )
        plt_95exp = plt.fill_between(
            [],
            0,
            0,
            color=yellow,
            label="95% expected",
            zorder=0,
        )
        (plt_offline,) = plt.plot(
            [], [], c="black", ls="-", lw=3, label="Offline (Run2)"
        )
        (plt_scouting,) = plt.plot(
            [], [], c="black", ls=":", lw=3, label="Scouting (Run2)"
        )
        legend1 = plt.gca().legend(
            handles=[
                plt_theory,
                plt_median_exp,
                plt_68exp,
                plt_95exp,
                plt_offline,
                plt_scouting,
            ],
            loc="upper right",
            frameon=False,
        )
        (plt_leptonic,) = plt.plot(
            [],
            [],
            c="blue",
            ls="-",
            lw=3,
            label=r" Obs. $m_{A'}=0.5\,GeV$"
            + "\n"
            + r"$B(A'\rightarrow\mu^+\mu^-)=40\%$",
        )
        (plt_hadronic,) = plt.plot(
            [],
            [],
            c="red",
            ls="-",
            lw=3,
            label=r" Obs. $m_{A'}=0.7\,GeV$"
            + "\n"
            + r"$B(A'\rightarrow\mu^+\mu^-)=15\%$",
        )
        legend2 = plt.gca().legend(
            handles=[plt_leptonic, plt_hadronic],
            loc="lower right",
            frameon=False,
        )
        plt.gca().add_artist(legend1)

        plt.xlim(100, 1050)
        plt.ylim(1e-6, 1e3)
        plt.yscale("log")
        ax.set_xlabel(r"$m_{S}$ (GeV)")
        ax.set_ylabel(r"signal strength $r$")
        if suffix == "_Run2_13TeV":
            lumi_label = r"$118\,fb^{-1}$ ($13\,TeV$)"
        elif suffix == "_Run3_13p6TeV":
            lumi_label = r"$62.4\,fb^{-1}$ ($13.6\,TeV$)"
        else:
            lumi_label = r"$118\,fb^{-1}$ ($13\,TeV$) + $62.4\,fb^{-1}$ ($13.6\,TeV$)"
        hep.cms.label(
            llabel="Preliminary",
            data=True,
            rlabel=lumi_label,
            ax=ax,
        )
        plt.text(
            900,
            70,
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
            f"limit_plots_weijie/{tag}/limits_unscaled_mPhi{m_phi.replace('.', 'p')}_T{T.replace('.', 'p')}.pdf",
            bbox_inches="tight",
        )
        plt.close()
