import argparse
import time
from pathlib import Path

from rich.console import Console  # type: ignore[import]
from rich.table import Table  # type: ignore[import]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        default="full_analysis_Dec2025",
        help="Tag to identify the analysis",
    )
    return parser.parse_args()


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
YEARS = ["2016", "2017", "2018", "2022", "2022EE", "2023", "2023BPix"]
REGIONS = ["SRs", "CR", "VR"]
CATEGORY_ORDER = ["signal", "background", "data"]

# Use *patterns* instead of full filenames
CATEGORIES = {
    "signal": "GluGluToSUEP*_histograms.pkl",
    "background": "QCD*470*600_MuEnrichedPt5*_histograms.pkl",
    "data": "*Muon*_histograms.pkl",
}

PATH_TEMPLATE = "processor_output_files/{tag}_{year}_{region}_output_histograms"

CHECK_NONEMPTY = True

# Control blinding
BLIND_SR_DATA = True  # If True, SRs data entries are shown as BLINDED

if "__main__" in __name__:
    start_time = time.time()
    args = parse_args()

    console = Console()

    # -------------------------------------------------------------------
    # Compute statuses
    # -------------------------------------------------------------------
    # Store status text per (year, category, region)
    statuses = {}
    missing_files = []

    for year in YEARS:
        for region in REGIONS:
            dir_path = Path(
                PATH_TEMPLATE.format(tag=args.tag, year=year, region=region)
            )

            for category in CATEGORY_ORDER:
                pattern = CATEGORIES[category]

                # Handle blinding: SRs data
                if category == "data" and region == "SRs" and BLIND_SR_DATA:
                    status_text = "[bold blue]BLINDED[/bold blue]"
                    statuses[(year, category, region)] = status_text
                    # Do NOT count as missing/empty, and don't check the files
                    continue

                if not dir_path.exists():
                    status_text = "[bold red]DIR MISSING[/bold red]"
                    missing_files.append(dir_path / pattern)
                else:
                    matches = list(dir_path.glob(pattern))

                    if not matches:
                        status_text = "[bold red]MISSING[/bold red]"
                        missing_files.append(dir_path / pattern)
                    else:
                        if CHECK_NONEMPTY:
                            nonempty = [p for p in matches if p.stat().st_size > 0]
                            if nonempty:
                                status_text = "[bold green]OK[/bold green]"
                            else:
                                status_text = "[bold yellow]EMPTY[/bold yellow]"
                                missing_files.extend(matches)
                        else:
                            status_text = "[bold green]OK[/bold green]"

                statuses[(year, category, region)] = status_text

    # -------------------------------------------------------------------
    # Build table: 1 row per year, nested-like columns
    # -------------------------------------------------------------------
    table = Table(title="Output File Status")

    table.add_column("Year", style="cyan", no_wrap=True)

    # Visually nested headers: "Signal\nSRs", "Signal\nCR", ...
    for category in CATEGORY_ORDER:
        for region in REGIONS:
            header = f"{category.capitalize()}\n[dim]{region}[/dim]"
            table.add_column(header, justify="center")

    # Fill rows
    for year in YEARS:
        row = [year]
        for category in CATEGORY_ORDER:
            for region in REGIONS:
                row.append(statuses[(year, category, region)])
        table.add_row(*row)

    # -------------------------------------------------------------------
    # Print full report
    # -------------------------------------------------------------------
    console.print(table)

    console.print()
    entries_checked = len(YEARS) * len(REGIONS) * len(CATEGORY_ORDER)
    console.print(
        f"[bold]Total entries (year/region/category):[/bold] {entries_checked}"
    )
    console.print(
        f"[bold]Missing/Empty entries (excluding BLINDED):[/bold] {len(missing_files)}\n"
    )

    # if missing_files:
    #     console.print("[bold red]Files/patterns needing attention:[/bold red]")
    #     for p in missing_files:
    #         console.print(f"  - {p}")
    # console.print()
