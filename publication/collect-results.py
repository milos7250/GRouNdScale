#!/usr/bin/env python3
"""Collect the summary.json of every publication run into one CSV, one row per (experiment, run, mode).

Usage:
    python publication/collect-results.py [--output publication/results.csv]
"""

import csv
import json
import re
from pathlib import Path

try:
    import rich_click as click
except ImportError:
    import click

HERE = Path(__file__).resolve().parent
HYPERPARAMETERS = ("B", "F", "G", "k", "d", "w", "n")
MEASUREMENTS = (
    "average_step_time_ms",
    "peak_torch_allocated_mib",
    "peak_torch_reserved_mib",
    "peak_nvidia_smi_mib",
)
STEADY_STATE = ("peak_torch_allocated_mib", "peak_torch_reserved_mib")
PHASES = ("controller", "generator")


def natural_key(text: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def flatten(summary: dict) -> dict:
    row = {
        "experiment": summary["experiment"],
        "run": summary["run"],
        "mode": summary["mode"],
        "status": summary["status"],
        **{name: summary["hyperparameters"][name] for name in HYPERPARAMETERS},
        "total_genes": summary["graph"]["total_genes"],
        "k_min": summary["graph"]["k_min"],
        "k_mean": summary["graph"]["k_mean"],
    }
    for phase in PHASES:
        data = summary[phase]
        row[f"{phase}_complete"] = data["complete"]
        for name in MEASUREMENTS:
            row[f"{phase}_{name}"] = data.get(name)
        for name in STEADY_STATE:
            row[f"{phase}_steady_{name}"] = data.get("steady_state", {}).get(name)
    row["generator_steady_stage_peak_torch_allocated_mib"] = json.dumps(
        summary["generator"].get("steady_state", {}).get("stage_peak_torch_allocated_mib", {})
    )
    row["git_commit"] = summary["settings"]["git"]["commit"]
    row["gpu"] = summary["settings"]["gpu"]
    return row


@click.command(context_settings={"show_default": True, "help_option_names": ["-h", "--help"]})
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=HERE / "results.csv")
def cli(output: Path) -> None:
    """Collects all results/*/*/summary.json below the experiment directories into a CSV."""
    paths = sorted(HERE.glob("*/results/*/*/summary.json"))
    if not paths:
        raise click.ClickException(f"No summary.json found below {HERE}")
    rows = [flatten(json.loads(path.read_text())) for path in paths]
    rows.sort(key=lambda row: (row["experiment"], natural_key(row["run"]), row["mode"]))

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    click.echo(f"Wrote {len(rows)} runs to {output} ({', '.join(f'{n} {s}' for s, n in sorted(statuses.items()))})")


if __name__ == "__main__":
    cli()
