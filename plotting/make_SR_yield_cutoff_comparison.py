import argparse
import math

import hist
import plot_utils
from colorama import Fore, Style  # type: ignore[import]
from rich.progress import track  # type: ignore[import]
from tabulate import tabulate  # type: ignore[import]

RUN_PERIODS = {
    # Leaving out 2016APV for now to match the rest of the plotting scripts.
    "Run2": ["2016", "2017", "2018"],
    "Run3": ["2022", "2022EE", "2023", "2023BPix"],
}

SR_KEYS = {
    "high": "SR_high_temp_tight",
    "low": "SR_low_temp_tight",
}

# Background (B, sigmaB) per nMuon cutoff per SR. Used for the upper-limit
# ratio column; only cutoffs 7 and 6 are supported.
BACKGROUNDS = {
    7: {
        "high": (0.21, 0.10),
        "low": (0.066, 0.052),
    },
    6: {
        "high": (10.5, 3.9),
        "low": (2.4, 1.7),
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Feb2026",
        help="Tag to identify the analysis",
    )
    parser.add_argument(
        "--year",
        type=str,
        nargs="*",
        default=["2018"],
        help="Year of the data. Default is 2018. Can be a single year or multiple years.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print the table in LaTeX format",
    )
    parser.add_argument(
        "--nMuon-cutoffs",
        type=int,
        nargs=2,
        default=[7, 6],
        help="The two nMuon cutoffs to compare. Default is 7 and 6. The ratio is computed as second/first.",
    )
    return parser.parse_args()


def expand_years(years):
    expanded_years = []
    for year in years:
        year_list = RUN_PERIODS.get(year, [year])
        for expanded_year in year_list:
            if expanded_year not in expanded_years:
                expanded_years.append(expanded_year)
    return expanded_years


def smart_rounding(value, uncertainty, mode="simple"):
    prefix = suffix = ""
    sep = "±"
    if mode == "latex_raw":
        prefix = "$"
        suffix = "$"
        sep = r"\pm"
    if uncertainty == 0 and value == 0:
        return f"{prefix}0{sep}0{suffix}"
    elif uncertainty < 0.1:
        return f"{prefix}{value:.3f} {sep} {uncertainty:.3f}{suffix}"
    elif uncertainty < 1:
        return f"{prefix}{value:.2f} {sep} {uncertainty:.2f}{suffix}"
    elif uncertainty < 10:
        return f"{prefix}{value:.1f} {sep} {uncertainty:.1f}{suffix}"
    else:
        return f"{prefix}{value:.0f} {sep} {uncertainty:.0f}{suffix}"


def ratio_with_uncertainty(num, den):
    """Ratio of two WeightedSums with propagated uncorrelated uncertainty."""
    if den.value == 0:
        return float("nan"), float("nan")
    r = num.value / den.value
    rel_num_sq = num.variance / num.value**2 if num.value != 0 else 0.0
    rel_den_sq = den.variance / den.value**2
    r_unc = abs(r) * math.sqrt(rel_num_sq + rel_den_sq)
    return r, r_unc


def upper_limit_ratio(y_a, y_b, cutoff_b, label_b):
    """Ratio of upper limits UL_b / UL_a:
        UL_b = 1.96 * sqrt(B + sigmaB^2) / S_b   (B taken at cutoff_b, SR label_b)
        UL_a = 3.0 / S_a                          (Poisson limit, low-B regime)
    so the ratio is 1.96 * sqrt(B + sigmaB^2) * S_a / (3.0 * S_b).
    """
    if label_b is None or y_a.value == 0 or y_b.value == 0:
        return float("nan"), float("nan")
    if cutoff_b not in BACKGROUNDS or label_b not in BACKGROUNDS[cutoff_b]:
        return float("nan"), float("nan")
    B, sigmaB = BACKGROUNDS[cutoff_b][label_b]
    bkg_term = math.sqrt(B + sigmaB**2)
    r = 1.96 * bkg_term * y_a.value / (3.0 * y_b.value)
    rel_sq = y_a.variance / y_a.value**2 + y_b.variance / y_b.value**2
    if bkg_term > 0:
        # Treat sigmaB as the uncertainty on B; d(bkg_term)/dB = 1 / (2*bkg_term).
        rel_sq += (sigmaB / (2 * bkg_term**2)) ** 2
    r_unc = abs(r) * math.sqrt(rel_sq)
    return r, r_unc


def significance_ratio(y_a, y_b, cutoff_a, cutoff_b, label_a, label_b):
    """Ratio of S/sqrt(B + sigmaB^2) between the two cutoffs (b / a)."""
    if label_a is None or label_b is None or y_a.value == 0 or y_b.value == 0:
        return float("nan"), float("nan")
    if cutoff_a not in BACKGROUNDS or label_a not in BACKGROUNDS[cutoff_a]:
        return float("nan"), float("nan")
    if cutoff_b not in BACKGROUNDS or label_b not in BACKGROUNDS[cutoff_b]:
        return float("nan"), float("nan")
    B_a, sigmaB_a = BACKGROUNDS[cutoff_a][label_a]
    B_b, sigmaB_b = BACKGROUNDS[cutoff_b][label_b]
    denom_a = math.sqrt(B_a + sigmaB_a**2)
    denom_b = math.sqrt(B_b + sigmaB_b**2)
    if denom_a == 0 or denom_b == 0:
        return float("nan"), float("nan")
    r = (y_b.value * denom_a) / (y_a.value * denom_b)
    rel_sq = y_a.variance / y_a.value**2 + y_b.variance / y_b.value**2
    # Treat sigmaB as the uncertainty on B; d(denom)/dB = 1 / (2*denom).
    rel_sq += (sigmaB_a / (2 * denom_a**2)) ** 2
    rel_sq += (sigmaB_b / (2 * denom_b**2)) ** 2
    r_unc = abs(r) * math.sqrt(rel_sq)
    return r, r_unc


def asimov_significance(s, b, sigma_b):
    """Asimov significance from Cowan et al., formulae [10] and [20] of
    https://www.pp.rhul.ac.uk/~cowan/stat/notes/medsigNote.pdf.
    """
    if b <= 0 or s <= 0:
        return float("nan")
    sb2 = sigma_b * sigma_b
    if sb2 / b > 1e-12:
        bpsb2 = b + sb2
        b2 = b * b
        spb = s + b
        arg1 = spb * bpsb2 / (b2 + spb * sb2)
        arg2 = 1.0 + sb2 * s / (b * bpsb2)
        if arg1 <= 0 or arg2 <= 0:
            return float("nan")
        za2 = 2.0 * (spb * math.log(arg1) - (b2 / sb2) * math.log(arg2))
    else:
        za2 = 2.0 * ((s + b) * math.log(1.0 + s / b) - s)
    if za2 < 0:
        return float("nan")
    return math.sqrt(za2)


def asimov_significance_ratio(y_a, y_b, cutoff_a, cutoff_b, label_a, label_b):
    """Ratio of Asimov significances Z_b / Z_a between the two cutoffs."""
    if label_a is None or label_b is None or y_a.value <= 0 or y_b.value <= 0:
        return float("nan"), float("nan")
    if cutoff_a not in BACKGROUNDS or label_a not in BACKGROUNDS[cutoff_a]:
        return float("nan"), float("nan")
    if cutoff_b not in BACKGROUNDS or label_b not in BACKGROUNDS[cutoff_b]:
        return float("nan"), float("nan")
    B_a, sigmaB_a = BACKGROUNDS[cutoff_a][label_a]
    B_b, sigmaB_b = BACKGROUNDS[cutoff_b][label_b]
    Z_a = asimov_significance(y_a.value, B_a, sigmaB_a)
    Z_b = asimov_significance(y_b.value, B_b, sigmaB_b)
    if math.isnan(Z_a) or math.isnan(Z_b) or Z_a == 0:
        return float("nan"), float("nan")
    r = Z_b / Z_a

    # Numerical propagation: uncertainties on S (statistical) and on B (sigmaB).
    def var_Z(s_val, var_s, b_val, sigma_b_val):
        eps_s = max(1e-6, abs(s_val) * 1e-4)
        eps_b = max(1e-6, abs(b_val) * 1e-4)
        Z0 = asimov_significance(s_val, b_val, sigma_b_val)
        dZ_dS = (asimov_significance(s_val + eps_s, b_val, sigma_b_val) - Z0) / eps_s
        dZ_dB = (asimov_significance(s_val, b_val + eps_b, sigma_b_val) - Z0) / eps_b
        return dZ_dS**2 * var_s + dZ_dB**2 * sigma_b_val**2

    vZa = var_Z(y_a.value, y_a.variance, B_a, sigmaB_a)
    vZb = var_Z(y_b.value, y_b.variance, B_b, sigmaB_b)
    rel_sq = vZa / Z_a**2 + vZb / Z_b**2
    r_unc = abs(r) * math.sqrt(rel_sq)
    return r, r_unc


def best_sr_yield(plots, sample, years_to_load, cutoff, com_energy):
    """Return (best_sr_label, summed WeightedSum) across years for the SR with the
    larger summed yield at the given nMuon cutoff. Returns (None, None) if no
    matching sample is found in any year."""
    sums = {label: hist.accumulators.WeightedSum() for label in SR_KEYS}
    exists = False
    for year in years_to_load:
        sample_year = f"{sample}_{com_energy(year)}_{year}"
        if sample_year not in plots:
            continue
        exists = True
        for label, key in SR_KEYS.items():
            sums[label] += plots[sample_year][key][(cutoff * 1j) :: sum]
    if not exists:
        return None, None
    best_label = max(sums, key=lambda lab: sums[lab].value)
    return best_label, sums[best_label]


if "__main__" == __name__:
    args = parse_args()
    cutoff_a, cutoff_b = args.nMuon_cutoffs
    years_to_load = expand_years(args.year)
    print(f"Loading plots for years: {', '.join(years_to_load)}")

    plots = {}
    for year in track(years_to_load, description="Loading plots"):
        plots = plots | plot_utils.loader(
            tag=f"{args.tag}_{year}_SRs",
            era=year,
            load_data=False,
        )

    tablefmt = "simple"
    if args.latex:
        tablefmt = "latex_raw"

    header = [
        "SUEP model",
        f"Best SR yield (nMuon ≥ {cutoff_a})",
        f"Best SR yield (nMuon ≥ {cutoff_b})",
        f"Ratio (≥{cutoff_b} / ≥{cutoff_a})",
        f"UL ratio (≥{cutoff_b} / ≥{cutoff_a})",
        f"S/√(B+σB²) ratio (≥{cutoff_b} / ≥{cutoff_a})",
        f"Asimov Z ratio (≥{cutoff_b} / ≥{cutoff_a})",
    ]
    yield_table = []

    masses_S = [125, 200, 300, 400, 500, 600, 800, 1000]
    masses_phi = [1, 1.4, 2, 3, 4, 6, 8]
    T_over_mPhi = [0.25, 0.5, 1, 2, 4]
    modes = ["leptonic", "hadronic"]

    com_energy = lambda year: "13TeV" if year.startswith("201") else "13p6TeV"

    def annotate(label, text):
        """Prefix a cell with the SR label (high/low temp)."""
        if label is None:
            return text
        tag = "H" if label == "high" else "L"
        if tablefmt == "latex_raw":
            return f"[{tag}] {text}"
        color = Fore.GREEN if label == "high" else Fore.CYAN
        return f"{color}[{tag}]{Style.RESET_ALL} {text}"

    for mPhi in masses_phi:
        for r in T_over_mPhi:
            T = r * mPhi
            for mode in modes:
                for mS in masses_S:
                    scan_point = f"mS{mS:.3f}_mPhi{mPhi:.3f}_T{T:.3f}_mode{mode}"
                    sample = f"GluGluToSUEP_{scan_point}"
                    name = f"mS{mS}_mPhi{mPhi}_T{T}_{mode}"
                    if tablefmt == "latex_raw":
                        name = (
                            f"$m_S={mS}$, $m_" + r"\phi" + f"={mPhi}$, $T={T}$, {mode}"
                        )

                    label_a, y_a = best_sr_yield(
                        plots, sample, years_to_load, cutoff_a, com_energy
                    )
                    label_b, y_b = best_sr_yield(
                        plots, sample, years_to_load, cutoff_b, com_energy
                    )

                    if y_a is None or y_b is None:
                        continue

                    cell_a = smart_rounding(
                        y_a.value, math.sqrt(y_a.variance), mode=tablefmt
                    )
                    cell_b = smart_rounding(
                        y_b.value, math.sqrt(y_b.variance), mode=tablefmt
                    )

                    r_val, r_unc = ratio_with_uncertainty(y_b, y_a)
                    if math.isnan(r_val):
                        cell_r = "—" if tablefmt != "latex_raw" else "--"
                    else:
                        cell_r = smart_rounding(r_val, r_unc, mode=tablefmt)

                    ul_val, ul_unc = upper_limit_ratio(y_a, y_b, cutoff_b, label_b)
                    if math.isnan(ul_val):
                        cell_ul = "—" if tablefmt != "latex_raw" else "--"
                    else:
                        cell_ul = smart_rounding(ul_val, ul_unc, mode=tablefmt)

                    sig_val, sig_unc = significance_ratio(
                        y_a, y_b, cutoff_a, cutoff_b, label_a, label_b
                    )
                    if math.isnan(sig_val):
                        cell_sig = "—" if tablefmt != "latex_raw" else "--"
                    else:
                        cell_sig = smart_rounding(sig_val, sig_unc, mode=tablefmt)

                    az_val, az_unc = asimov_significance_ratio(
                        y_a, y_b, cutoff_a, cutoff_b, label_a, label_b
                    )
                    if math.isnan(az_val):
                        cell_az = "—" if tablefmt != "latex_raw" else "--"
                    else:
                        cell_az = smart_rounding(az_val, az_unc, mode=tablefmt)

                    yield_table.append(
                        [
                            name,
                            annotate(label_a, cell_a),
                            annotate(label_b, cell_b),
                            cell_r,
                            cell_ul,
                            cell_sig,
                            cell_az,
                        ]
                    )

    print()
    print(tabulate(yield_table, headers=header, tablefmt=tablefmt))
    print()
