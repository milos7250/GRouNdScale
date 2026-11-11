#!/usr/bin/env python3
"""
Repeat ``src/main.py --generate --evaluate`` with separate output directories, parse the printed metrics, and report the
mean and sample standard deviation (ddof=1) of each metric.

For run ``i`` the script writes a copy of the configuration whose output directory, generation path, and simulated data
path point to ``<output directory>/evaluate_range/run_NNN/``, and runs the upstream CLI with it. The upstream CLI prints
its metrics, which are parsed from its stdout.

The summary is written to ``<output directory>/evaluation_range_summary.json``.

The paths in the configuration are relative to its directory, so the upstream CLI (``upstream/src/main.py`` of the
repository) is run from there. Run this script from its directory, for example::

    python evaluate_range.py --config causal_gan.cfg --runs 3

The random seed is not set, so runs are not reproducible.
"""

import argparse
import json
import logging
import os
import re
import statistics
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path

MAIN = Path(__file__).resolve().parents[2] / "upstream" / "src" / "main.py"  # publication/<this directory>/ -> repository root

PAIR_LINE = re.compile(r"^(Euclidean distance|Cosine distance|MMD|miLISI) \((real vs fake|control)\): (.+)$")
RF_LINE = re.compile(r"^RF AUROC: (.+)$")
METRIC_NAMES = {"Euclidean distance": "euclidean_distance", "Cosine distance": "cosine_distance", "MMD": "mmd", "miLISI": "miLISI"}
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("evaluate_range")


def read_config(path: Path) -> ConfigParser:
    cfg = ConfigParser(empty_lines_in_values=False, allow_no_value=True, inline_comment_prefixes=";", interpolation=None)
    cfg.read(path)
    return cfg


def parse_metrics(stdout: str) -> dict[str, float]:
    """Extract metric values from the upstream CLI output, keyed like the GRouNdScale evaluate_range summary."""
    results: dict[str, float] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if match := PAIR_LINE.match(line):
            name, kind, value = match.groups()
            key = f"{METRIC_NAMES[name]}_{kind.replace(' ', '_')}"
        elif match := RF_LINE.match(line):
            key, value = "rf_auroc", match.group(1)
        else:
            continue
        # MMD is printed as a TensorFlow tensor, e.g. "tf.Tensor(0.0145, shape=(), dtype=float32)"
        results[key] = float(NUMBER.search(value).group())  # pyright: ignore[reportOptionalMemberAccess]
    return results


def main(config: Path, runs: int) -> None:
    config = config.resolve()
    cwd = config.parent
    base = read_config(config)
    base_output = Path(base.get("EXPERIMENT", "output directory"))

    run_results: list[dict[str, float]] = []
    for run in range(1, runs + 1):
        run_output = base_output / "evaluate_range" / f"run_{run:03d}"
        run_dir = cwd / run_output
        run_dir.mkdir(parents=True, exist_ok=True)

        # The upstream factory loads the causal controller from "<output directory>_CC", so each run's output directory
        # gets a link to the shared controller directory.
        cc_link = cwd / f"{run_output}_CC"
        if not cc_link.exists():
            os.symlink(os.path.relpath(cwd / f"{base_output}_CC", cc_link.parent), cc_link)

        run_cfg = base
        run_cfg.set("EXPERIMENT", "output directory", str(run_output))
        run_cfg.set("Generation", "generation path", str(run_output / "simulated.h5ad"))
        run_cfg.set("Evaluation", "simulated data path", str(run_output / "simulated.h5ad"))
        run_config_path = run_dir / config.name
        with run_config_path.open("w") as f:
            run_cfg.write(f)

        logger.info(f"Evaluation range run {run}/{runs}: running upstream --generate --evaluate...")
        proc = subprocess.run(
            [sys.executable, str(MAIN), "--config", str(run_config_path.resolve()), "--generate", "--evaluate"],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        (run_dir / "upstream.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"Upstream run {run} failed with exit code {proc.returncode}; see {run_dir / 'upstream.log'}")

        metrics = parse_metrics(proc.stdout)
        (run_dir / "evaluation_results.json").write_text(json.dumps(metrics, indent=2))
        run_results.append(metrics)
        logger.info(f"Evaluation range run {run}/{runs}: {len(metrics)} metrics parsed")

    keys = set(run_results[0])
    if any(set(r) != keys for r in run_results):
        raise RuntimeError("Runs produced different metric sets; check the upstream logs in each run directory")

    summary: dict[str, dict[str, float | list[float]]] = {}
    for key in run_results[0]:
        values = [r[key] for r in run_results]
        mean, std = statistics.mean(values), statistics.stdev(values)
        summary[key] = {"mean": mean, "std": std, "values": values}
        logger.info(f"{key}: {mean:.6g} +/- {std:.6g}")

    summary_path = cwd / base_output / "evaluation_range_summary.json"
    summary_path.write_text(json.dumps({"n_runs": runs, "metrics": summary}, indent=2))
    logger.info(f"Evaluation range summary saved to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat upstream generation and evaluation and report mean and SD of each metric.")
    parser.add_argument("--config", required=True, type=Path, help="Path to the configuration file")
    parser.add_argument("--runs", type=int, default=3, help="Number of repeated runs (>= 2)")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be an integer of at least 2")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.runs)
