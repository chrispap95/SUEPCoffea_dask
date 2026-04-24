import argparse
import csv
import math
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import uproot

try:
    from rich.progress import track  # type: ignore[import]
except ModuleNotFoundError:

    def track(iterable, description=None):
        return iterable


try:
    from tabulate import tabulate  # type: ignore[import]
except ModuleNotFoundError:

    def tabulate(rows, headers, tablefmt="simple"):
        lines = ["  ".join(str(header) for header in headers)]
        for row in rows:
            lines.append("  ".join(str(value) for value in row))
        return "\n".join(lines)


SAMPLE_DIR_RE = re.compile(
    r"^SUEP_mS(?P<mS>\d+\.\d+)_mPhi(?P<mPhi>\d+\.\d+)_T(?P<T>\d+\.\d+)_mode(?P<mode>[A-Za-z0-9_]+)$"
)
DEFAULT_REDIRECTOR = "root://cmseos.fnal.gov"


@dataclass
class SampleDirectory:
    directory: str
    files: list[str]
    mS: float
    mPhi: float
    T: float
    mode: str


@dataclass
class EfficiencyPoint:
    mS: float
    mPhi: float
    T: float
    ratio: float
    mode: str
    passed: float
    total: float
    efficiency: float
    uncertainty: float
    n_files: int
    directories: list[str] = field(default_factory=list)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate trigger efficiencies for signal MC stored on EOS or on a local "
            "filesystem, then plot the efficiencies across the signal parameter space."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/store/group/lpcsuep/Muon_counting_search/SUEPNano_UL18_May2025_merged",
        help=(
            "Base directory containing the signal samples. This can be a local path, "
            "an EOS /store path, or a full root://...//store/... path."
        ),
    )
    parser.add_argument(
        "--redirector",
        type=str,
        default=DEFAULT_REDIRECTOR,
        help="XRootD redirector used for EOS listing and ROOT file access.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help=(
            "Directory where plots and the CSV summary will be written. "
            "Defaults to trigger_efficiencies/<basename(input-dir)>."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Optional label shown in the plot titles. Defaults to the input directory basename.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum recursion depth used to discover sample directories.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        nargs="*",
        help="Optional list of decay modes to keep, for example --mode leptonic hadronic.",
    )
    parser.add_argument(
        "--mS",
        type=float,
        nargs="*",
        help="Optional list of scalar masses to keep.",
    )
    parser.add_argument(
        "--mPhi",
        type=float,
        nargs="*",
        help="Optional list of dark photon masses to keep.",
    )
    parser.add_argument(
        "--T",
        type=float,
        nargs="*",
        help="Optional list of temperatures to keep.",
    )
    parser.add_argument(
        "--y-axis",
        choices=["ratio", "temperature"],
        default="ratio",
        help="Use either T/mPhi or the absolute temperature on the heatmap y-axis.",
    )
    parser.add_argument(
        "--numerator",
        choices=["entries", "genweight"],
        default="entries",
        help=(
            "Quantity used for the post-trigger count in the Events tree. "
            "'entries' matches the skimmed-tree interpretation described in the request."
        ),
    )
    parser.add_argument(
        "--denominator-branch",
        type=str,
        default="genEventSumwPreSkim",
        help="Branch in the Runs tree that stores the pre-trigger total event sum.",
    )
    parser.add_argument(
        "--plot-format",
        type=str,
        default="pdf",
        help="File format for the produced plots.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only calculate efficiencies and write the CSV summary, without making plots.",
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print the full per-sample efficiency table to stdout.",
    )
    return parser.parse_args()


def format_param(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def float_matches(
    value: float, targets: Optional[list[float]], atol: float = 1e-6
) -> bool:
    if not targets:
        return True
    return any(
        math.isclose(value, target, rel_tol=0.0, abs_tol=atol) for target in targets
    )


def split_remote_input(path: str, redirector: str) -> tuple[str, str]:
    if path.startswith("root://"):
        match = re.match(r"^(root://[^/]+)(/.*)$", path)
        if not match:
            raise ValueError(f"Could not parse remote path: {path}")
        redirector = match.group(1).rstrip("/")
        path = "/" + match.group(2).lstrip("/")
    return redirector.rstrip("/"), path.rstrip("/")


def make_uproot_path(path: str, redirector: str) -> str:
    if path.startswith("root://"):
        return path
    if path.startswith("/store/"):
        return f"{redirector}//{path.lstrip('/')}"
    return path


def is_remote_path(path: str) -> bool:
    return path.startswith("/store/") or path.startswith("root://")


def list_directory(path: str, redirector: str) -> list[str]:
    if is_remote_path(path):
        redirector, store_path = split_remote_input(path, redirector)
        result = subprocess.run(
            ["xrdfs", redirector, "ls", store_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"xrdfs failed for {store_path} with exit code {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        entries = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("root://"):
                entries.append(line.rstrip("/"))
            elif line.startswith("/"):
                entries.append(line.rstrip("/"))
            else:
                entries.append(posixpath.join(store_path, line).rstrip("/"))
        return sorted(entries)

    try:
        return sorted(
            os.path.join(path, entry).rstrip("/")
            for entry in os.listdir(path.rstrip("/"))
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Directory does not exist: {path}") from exc


def discover_samples(
    base_path: str, redirector: str, max_depth: int
) -> list[SampleDirectory]:
    samples: list[SampleDirectory] = []
    visited: set[str] = set()

    def recurse(path: str, depth: int) -> None:
        path = path.rstrip("/")
        if path in visited:
            return
        visited.add(path)

        basename = (
            os.path.basename(path)
            if not is_remote_path(path)
            else posixpath.basename(path)
        )
        match = SAMPLE_DIR_RE.match(basename)

        entries = list_directory(path, redirector)
        root_files = [entry for entry in entries if entry.endswith(".root")]

        if match:
            if not root_files:
                raise RuntimeError(f"Found sample directory without ROOT files: {path}")
            samples.append(
                SampleDirectory(
                    directory=path,
                    files=root_files,
                    mS=float(match.group("mS")),
                    mPhi=float(match.group("mPhi")),
                    T=float(match.group("T")),
                    mode=match.group("mode"),
                )
            )
            return

        if depth == 0:
            return

        for entry in entries:
            if entry.endswith(".root"):
                continue
            recurse(entry, depth - 1)

    recurse(base_path, max_depth)
    return sorted(
        samples, key=lambda sample: (sample.mode, sample.mS, sample.mPhi, sample.T)
    )


def keep_sample(sample: SampleDirectory, args) -> bool:
    if args.mode and sample.mode not in set(args.mode):
        return False
    if not float_matches(sample.mS, args.mS):
        return False
    if not float_matches(sample.mPhi, args.mPhi):
        return False
    if not float_matches(sample.T, args.T):
        return False
    return True


def get_events_passed(events_tree, numerator: str) -> float:
    if numerator == "entries":
        return float(events_tree.num_entries)
    if "genWeight" not in events_tree:
        raise KeyError("Events tree does not contain the requested branch 'genWeight'.")
    return float(np.sum(events_tree["genWeight"].array(library="np")))


def get_total_events(runs_tree, preferred_branch: str) -> float:
    branch_name = preferred_branch
    if branch_name not in runs_tree:
        if preferred_branch == "genEventSumwPreSkim" and "genEventSumw" in runs_tree:
            branch_name = "genEventSumw"
        else:
            raise KeyError(
                f"Runs tree does not contain '{preferred_branch}'"
                + (
                    " or 'genEventSumw'."
                    if preferred_branch == "genEventSumwPreSkim"
                    else "."
                )
            )
    return float(np.sum(runs_tree[branch_name].array(library="np")))


@contextmanager
def open_root_file(path: str, redirector: str):
    uproot_path = make_uproot_path(path, redirector)

    try:
        with uproot.open(uproot_path) as infile:
            yield infile
        return
    except Exception as direct_error:
        if not is_remote_path(path) or shutil.which("xrdcp") is None:
            raise

        with tempfile.TemporaryDirectory(prefix="trigger_eff_") as tmpdir:
            local_copy = os.path.join(tmpdir, os.path.basename(path))
            result = subprocess.run(
                ["xrdcp", "-f", uproot_path, local_copy],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Could not open {path} through uproot and xrdcp fallback failed: "
                    f"{result.stderr.strip()}"
                ) from direct_error

            with uproot.open(local_copy) as infile:
                yield infile


def measure_sample(sample: SampleDirectory, redirector: str, args) -> EfficiencyPoint:
    passed = 0.0
    total = 0.0

    for root_file in sample.files:
        with open_root_file(root_file, redirector) as infile:
            if "Events" not in infile:
                raise KeyError(f"'Events' tree not found in {root_file}")
            if "Runs" not in infile:
                raise KeyError(f"'Runs' tree not found in {root_file}")

            passed += get_events_passed(infile["Events"], args.numerator)
            total += get_total_events(infile["Runs"], args.denominator_branch)

    efficiency = float("nan")
    uncertainty = float("nan")
    if total > 0:
        efficiency = passed / total
        if 0 <= passed <= total:
            uncertainty = math.sqrt(efficiency * (1.0 - efficiency) / total)

    return EfficiencyPoint(
        mS=sample.mS,
        mPhi=sample.mPhi,
        T=sample.T,
        ratio=sample.T / sample.mPhi,
        mode=sample.mode,
        passed=passed,
        total=total,
        efficiency=efficiency,
        uncertainty=uncertainty,
        n_files=len(sample.files),
        directories=[sample.directory],
    )


def aggregate_points(points: list[EfficiencyPoint]) -> list[EfficiencyPoint]:
    grouped: dict[tuple[float, float, float, str], EfficiencyPoint] = {}

    for point in points:
        key = (point.mS, point.mPhi, point.T, point.mode)
        if key not in grouped:
            grouped[key] = EfficiencyPoint(
                mS=point.mS,
                mPhi=point.mPhi,
                T=point.T,
                ratio=point.ratio,
                mode=point.mode,
                passed=point.passed,
                total=point.total,
                efficiency=point.efficiency,
                uncertainty=point.uncertainty,
                n_files=point.n_files,
                directories=list(point.directories),
            )
            continue

        grouped_point = grouped[key]
        grouped_point.passed += point.passed
        grouped_point.total += point.total
        grouped_point.n_files += point.n_files
        grouped_point.directories.extend(point.directories)

    for point in grouped.values():
        point.directories = sorted(set(point.directories))
        point.efficiency = (
            point.passed / point.total if point.total > 0 else float("nan")
        )
        point.uncertainty = float("nan")
        if point.total > 0 and 0 <= point.passed <= point.total:
            point.uncertainty = math.sqrt(
                point.efficiency * (1.0 - point.efficiency) / point.total
            )

    return sorted(
        grouped.values(), key=lambda point: (point.mode, point.mS, point.mPhi, point.T)
    )


def write_summary_csv(points: list[EfficiencyPoint], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "mode",
                "mS",
                "mPhi",
                "T",
                "T_over_mPhi",
                "passed",
                "total",
                "efficiency",
                "uncertainty",
                "n_files",
                "directories",
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "mode": point.mode,
                    "mS": f"{point.mS:.3f}",
                    "mPhi": f"{point.mPhi:.3f}",
                    "T": f"{point.T:.3f}",
                    "T_over_mPhi": f"{point.ratio:.6f}",
                    "passed": f"{point.passed:.6f}",
                    "total": f"{point.total:.6f}",
                    "efficiency": f"{point.efficiency:.8f}",
                    "uncertainty": (
                        f"{point.uncertainty:.8f}"
                        if np.isfinite(point.uncertainty)
                        else ""
                    ),
                    "n_files": point.n_files,
                    "directories": ";".join(point.directories),
                }
            )


def print_summary_table(points: list[EfficiencyPoint]) -> None:
    table = []
    for point in points:
        unc_text = (
            f"{100.0 * point.uncertainty:.2f}"
            if np.isfinite(point.uncertainty)
            else "nan"
        )
        table.append(
            [
                point.mode,
                format_param(point.mS),
                format_param(point.mPhi),
                format_param(point.T),
                format_param(point.ratio),
                f"{100.0 * point.efficiency:.2f}",
                unc_text,
                f"{point.passed:.1f}",
                f"{point.total:.1f}",
                point.n_files,
            ]
        )
    print()
    print(
        tabulate(
            table,
            headers=[
                "mode",
                "mS [GeV]",
                "mPhi [GeV]",
                "T [GeV]",
                "T/mPhi",
                "eff [%]",
                "unc [%]",
                "passed",
                "total",
                "files",
            ],
            tablefmt="simple",
        )
    )
    print()


def build_grid(
    points: list[EfficiencyPoint], y_axis: str
) -> tuple[np.ndarray, list[float], list[float]]:
    x_values = sorted({point.mPhi for point in points})
    y_values = sorted(
        {point.ratio if y_axis == "ratio" else point.T for point in points}
    )
    grid = np.full((len(y_values), len(x_values)), np.nan)

    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}

    for point in points:
        y_value = point.ratio if y_axis == "ratio" else point.T
        grid[y_index[y_value], x_index[point.mPhi]] = point.efficiency

    return grid, x_values, y_values


def plot_mode_overview(
    points: list[EfficiencyPoint],
    mode: str,
    output_dir: str,
    label: str,
    y_axis: str,
    plot_format: str,
) -> None:
    import matplotlib as mpl  # type: ignore[import]
    import matplotlib.pyplot as plt  # type: ignore[import]
    import mplhep as hep

    hep.style.use(hep.style.CMS)
    mpl.rcParams["figure.facecolor"] = "white"

    mode_points = [point for point in points if point.mode == mode]
    if not mode_points:
        return

    masses = sorted({point.mS for point in mode_points})
    ncols = min(4, len(masses))
    nrows = int(math.ceil(len(masses) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.3 * ncols, 3.8 * nrows),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)

    finite_efficiencies = [
        point.efficiency for point in mode_points if np.isfinite(point.efficiency)
    ]
    vmax = max(finite_efficiencies) if finite_efficiencies else 1.0
    vmax = min(1.0, max(0.05, 1.05 * vmax))

    cmap = plt.cm.viridis.copy()  # type: ignore[import]
    cmap.set_bad("#e5e7eb")
    image = None

    for axis in axes_array.flat:
        axis.set_visible(False)

    for mass, axis in zip(masses, axes_array.flat):
        axis.set_visible(True)
        mass_points = [point for point in mode_points if math.isclose(point.mS, mass)]
        grid, x_values, y_values = build_grid(mass_points, y_axis)
        masked_grid = np.ma.masked_invalid(grid)

        image = axis.imshow(
            masked_grid,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )

        axis.set_title(rf"$m_S = {format_param(mass)}$ GeV")
        axis.set_xticks(
            range(len(x_values)), [format_param(value) for value in x_values]
        )
        axis.set_yticks(
            range(len(y_values)), [format_param(value) for value in y_values]
        )
        axis.set_xticks(np.arange(-0.5, len(x_values), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(y_values), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.2)
        axis.tick_params(which="minor", bottom=False, left=False)

        for y_index, y_value in enumerate(y_values):
            for x_index, x_value in enumerate(x_values):
                efficiency = grid[y_index, x_index]
                if np.isfinite(efficiency):
                    text_color = "white" if efficiency > 0.55 * vmax else "black"
                    axis.text(
                        x_index,
                        y_index,
                        f"{100.0 * efficiency:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=text_color,
                    )
                else:
                    axis.text(
                        x_index,
                        y_index,
                        "-",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#6b7280",
                    )

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes_array.ravel().tolist(), shrink=0.92)
        colorbar.set_label("Trigger efficiency")

    fig.supxlabel(r"$m_{\phi}$ [GeV]")
    fig.supylabel(r"$T / m_{\phi}$" if y_axis == "ratio" else r"$T$ [GeV]")

    title = f"Signal trigger efficiency, {mode}"
    if label.strip():
        title = f"{title}\n{label}"
    fig.suptitle(title)
    output_path = os.path.join(
        output_dir,
        f"trigger_efficiency_grid_{mode}.{plot_format}",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def print_compact_report(points: list[EfficiencyPoint]) -> None:
    modes = sorted({point.mode for point in points})
    print()
    print(f"Measured {len(points)} unique signal points.")
    for mode in modes:
        mode_points = [point for point in points if point.mode == mode]
        efficiencies = [
            point.efficiency for point in mode_points if np.isfinite(point.efficiency)
        ]
        if not efficiencies:
            print(f"  {mode}: {len(mode_points)} points")
            continue
        print(
            f"  {mode}: {len(mode_points)} points, "
            f"efficiency range = {100.0 * min(efficiencies):.2f}% to {100.0 * max(efficiencies):.2f}%"
        )
    print()


def main():
    args = parse_args()
    redirector, input_dir = split_remote_input(args.input_dir, args.redirector)

    output_dir = args.output_dir
    if output_dir is None:
        basename = (
            os.path.basename(input_dir.rstrip("/"))
            if not input_dir.startswith("/store/")
            else posixpath.basename(input_dir.rstrip("/"))
        )
        output_dir = os.path.join("trigger_efficiencies", basename)
    os.makedirs(output_dir, exist_ok=True)

    label = args.label
    if label is None:
        label = ""

    discovered_samples = discover_samples(input_dir, redirector, args.max_depth)
    filtered_samples = [
        sample for sample in discovered_samples if keep_sample(sample, args)
    ]

    if not filtered_samples:
        raise SystemExit(
            "No matching SUEP sample directories were found. "
            "Check --input-dir, --max-depth, and any parameter filters."
        )

    points = []
    for sample in track(filtered_samples, description="Measuring trigger efficiencies"):
        points.append(measure_sample(sample, redirector, args))
    points = aggregate_points(points)

    summary_csv = os.path.join(output_dir, "trigger_efficiency_summary.csv")
    write_summary_csv(points, summary_csv)

    if not args.skip_plots:
        for mode in sorted({point.mode for point in points}):
            plot_mode_overview(
                points=points,
                mode=mode,
                output_dir=output_dir,
                label=label,
                y_axis=args.y_axis,
                plot_format=args.plot_format,
            )

    print_compact_report(points)
    print(f"Discovered {len(discovered_samples)} sample directories.")
    print(f"Selected {len(filtered_samples)} sample directories after filtering.")
    print(f"Wrote summary CSV to {summary_csv}")
    if args.skip_plots:
        print("Skipped plot production.")
    else:
        print(f"Wrote plots to {output_dir}")

    if args.print_table:
        print_summary_table(points)


if __name__ == "__main__":
    main()
