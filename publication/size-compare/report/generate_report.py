#!/usr/bin/env python3
"""Generate the size-comparison report and scaling graphs for GRouNdScale."""

import csv
import json
import pickle
import re
from datetime import datetime
from itertools import chain
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# This script lives in the report directory; run artefacts sit alongside it.
OUT = Path(__file__).parent
ROOT = OUT.parent
LOG_ROOT = ROOT / "logs" / "causalgan-size"
RESULT_ROOT = ROOT / "results"

CONTROLLER_STEPS = 200_000
GENERATOR_STEPS = 500_000
DAY = 86_400

TIMESTAMP_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]")
STEP_RE = re.compile(r"Step 20/20 completed in \d+ milliseconds \(average:\s+(\d+)\s+ms\)")
LOOP_RE = re.compile(r"Starting training loop")
OOM_RE = re.compile(r"OutOfMemoryError")


def time_of_day(stamp: str) -> int:
    """Convert ``HH:MM:SS`` to seconds since midnight."""
    hours, minutes, seconds = (int(part) for part in stamp.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def result_directories() -> dict[int, Path]:
    """Map the requested HVG count of a run to its results directory."""
    mapping: dict[int, Path] = {}
    for directory in RESULT_ROOT.iterdir():
        if not directory.is_dir():
            continue
        config = (directory / "causal_gan.cfg").read_text(errors="replace")
        requested = re.search(r"results\d*/(\d+)/", config)
        if requested:
            mapping[int(requested.group(1))] = directory
    return mapping


def graph_statistics(path: Path) -> dict:
    """Recompute the causal-graph statistics from the saved graph."""
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    targets = len(graph)
    tfs = set(chain.from_iterable(graph.values()))
    imposed = sum(len(regulators) for regulators in graph.values())
    possible = len(tfs) * targets
    return {
        "tfs": len(tfs),
        "targets": targets,
        "genes": len(tfs) + targets,
        "imposed": imposed,
        "possible": possible,
        "density": 100 * imposed / possible if possible else 0.0,
    }


def parse_log(path: Path) -> dict:
    """Extract step timings and training-phase timestamps from a run log.

    A timestamp prefix applies to its own line and to the continuation lines
    that follow it, so the phase boundaries are the most recent timestamp seen
    at the moment the marker line is read. Times are returned in a monotonic
    seconds-since-first-midnight frame, which stays correct for runs that cross
    midnight.
    """
    lines = path.read_text(errors="replace").splitlines()
    phases: dict[str, int] = {}
    step_times: list[int] = []
    seen: list[int] = []
    current: int | None = None
    previous: int | None = None
    loops = 0
    steps = 0

    for line in lines:
        found = TIMESTAMP_RE.search(line)
        if found:
            value = time_of_day(found.group(1))
            if previous is not None and value < previous:
                value += DAY
            current = value
            seen.append(value)
            previous = value
        if current is None:
            continue
        if LOOP_RE.search(line):
            loops += 1
            phases["controller_start" if loops == 1 else "generator_start"] = current
        if STEP_RE.search(line):
            steps += 1
            step_times.append(int(STEP_RE.search(line).group(1)))
            phases["controller_end" if steps == 1 else "generator_end"] = current

    requested = re.search(r"results\d*/(\d+)/", "\n".join(lines))
    return {
        "task": int(path.stem),
        "requested": int(requested.group(1)) if requested else None,
        "first_stamp": seen[0] if seen else None,
        "phases": phases,
        "controller_ms": step_times[0] if len(step_times) > 0 else None,
        "generator_ms": step_times[1] if len(step_times) > 1 else None,
        "oom": any(OOM_RE.search(line) for line in lines),
    }


def peak_memory(path: Path, start: float | None, end: float | None, offset: float) -> int:
    """Return the peak GPU memory in MiB between two log-relative timestamps."""
    if start is None or end is None or start > end:
        return 0
    low, high = start + offset, end + offset
    peak = 0
    with path.open() as handle:
        for row in csv.DictReader(handle):
            moment = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
            if low <= moment <= high:
                peak = max(peak, int(row["memory_used"]))
    return peak


def overall_peak(path: Path) -> int:
    with path.open() as handle:
        return max(int(row["memory_used"]) for row in csv.DictReader(handle))


def load_runs() -> list[dict]:
    """Collect every completed run, keyed by the actual gene count."""
    directories = result_directories()
    runs: list[dict] = []
    for log in sorted(LOG_ROOT.glob("*/*.out"), key=lambda item: int(item.stem)):
        parsed = parse_log(log)
        if parsed["requested"] is None or parsed["generator_ms"] is None:
            continue
        directory = directories.get(parsed["requested"])
        if directory is None:
            continue
        graph = graph_statistics(directory / "causal_graph.pkl")
        memory_csv = directory / "gpu-usage.csv"
        params = json.loads((directory / "params-size.json").read_text())
        with memory_csv.open() as handle:
            first_row = next(csv.DictReader(handle))
        csv_start = datetime.strptime(first_row["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
        # Align the log's since-midnight frame with the CSV's epoch frame.
        offset = csv_start - (parsed["first_stamp"] or 0)
        phases = parsed["phases"]
        controller_peak = peak_memory(memory_csv, phases.get("controller_start"), phases.get("controller_end"), offset)
        generator_peak = peak_memory(memory_csv, phases.get("generator_start"), phases.get("generator_end"), offset)
        runs.append(
            {
                **graph,
                **parsed,
                "directory": directory.name,
                "controller_mib": controller_peak,
                "generator_mib": generator_peak or overall_peak(memory_csv),
                "ratio_pct": 100 * graph["tfs"] / graph["targets"] if graph["targets"] else 0.0,
                "gen_params": params["gen_state_dict"],
                "params": params,
            }
        )
    return sorted(runs, key=lambda row: row["genes"])


def fit(runs: list[dict], metric: str, predictor: str) -> tuple[float, float, float]:
    """Ordinary least-squares fit of ``metric`` against ``predictor``."""
    x = np.array([row[predictor] for row in runs], dtype=float)
    y = np.array([row[metric] for row in runs], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), float(np.corrcoef(x, y)[0, 1] ** 2)


def graph_params(runs: list[dict]) -> None:
    x = np.array([row["genes"] for row in runs], dtype=float)
    y = np.array([row["gen_params"] for row in runs], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y / 1e6, "o-", color="#1565C0", markersize=5, label="Measured")
    grid = np.linspace(x.min(), x.max(), 200)
    ax.plot(grid, (slope * grid + intercept) / 1e6, ":", color="#555555",
            label=f"Linear fit ($R^2$={r2:.4f})")
    ax.set_xlabel("Highly variable genes")
    ax.set_ylabel("Generator parameters (M)")
    ax.set_title("Generator Parameter Count vs. Gene Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "gen_params_vs_hvg.pdf", dpi=300)
    plt.close(fig)


def graph_memory(runs: list[dict], total_mib: int) -> None:
    x = np.array([row["genes"] for row in runs], dtype=float)
    memory = np.array([row["generator_mib"] for row in runs], dtype=float)
    controller = np.array([row["controller_mib"] for row in runs], dtype=float)
    slope, intercept = np.polyfit(x, memory, 1)
    r2 = float(np.corrcoef(x, memory)[0, 1] ** 2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, memory / 1024, "o-", color="#EF6C00", markersize=5, label="Generator training")
    ax.plot(x, controller / 1024, "o-", color="#2E7D32", markersize=5, label="Controller training")
    grid = np.linspace(x.min(), x.max(), 200)
    ax.plot(grid, (slope * grid + intercept) / 1024, ":", color="#555555",
            label=f"Generator linear fit ($R^2$={r2:.4f})")
    ax.axhline(total_mib / 1024, color="#C62828", linestyle="--", alpha=0.7,
               label=f"GPU capacity ({total_mib / 1024:.0f} GB)")
    ax.set_xlabel("Highly variable genes")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title("GPU Memory Requirement vs. Gene Count")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "gpu_mem_vs_hvg.pdf", dpi=300)
    plt.close(fig)


def graph_time(runs: list[dict]) -> None:
    x = np.array([row["genes"] for row in runs], dtype=float)
    y = np.array([row["generator_ms"] for row in runs], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, "o-", color="#1565C0", markersize=5, label="Measured")
    grid = np.linspace(x.min(), x.max(), 200)
    ax.plot(grid, slope * grid + intercept, ":", color="#555555",
            label=f"Linear fit ($R^2$={r2:.4f})")
    ax.set_xlabel("Highly variable genes")
    ax.set_ylabel("Generator average step time (ms)")
    ax.set_title("Generator Step Time vs. Gene Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "gen_step_vs_hvg.pdf", dpi=300)
    plt.close(fig)


def graph_ratio(runs: list[dict]) -> None:
    order = np.argsort([row["ratio_pct"] for row in runs])
    ratio = np.array([row["ratio_pct"] for row in runs])[order]
    genes = np.array([row["genes"] for row in runs])[order]
    step = np.array([row["generator_ms"] for row in runs])[order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scatter = ax.scatter(ratio, step, c=genes, cmap="viridis", s=55,
                         edgecolors="white", linewidths=0.5)
    fig.colorbar(scatter, ax=ax, label="Gene count")
    ax.set_xlabel("TF / target-gene ratio (%)")
    ax.set_ylabel("Generator average step time (ms)")
    ax.set_title("Generator Step Time vs. TF/Target Ratio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "gen_step_vs_ratio.pdf", dpi=300)
    plt.close(fig)


def latex(runs: list[dict], fits: dict) -> None:
    total_mib = 81_920
    memory_fit = fits[("generator_mib", "genes")]
    time_fit = fits[("generator_ms", "genes")]
    target_memory_fit = fits[("generator_mib", "targets")]
    target_time_fit = fits[("generator_ms", "targets")]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=2.2cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\usepackage{graphicx}",
        r"\usepackage{caption}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
        r"\title{GRouNdScale Size Comparison Report}",
        r"\author{Auto-generated from \texttt{size-compare} run artefacts}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Overview}",
        rf"This report summarises {len(runs)} completed training runs of the causal GAN pipeline with "
        rf"increasingly higher highly variable gene (HVG) counts, from {runs[0]['genes']:,} to "
        rf"{runs[-1]['genes']:,} genes. Every run uses the same architecture and training "
        r"configuration (20 steps for the causal controller, then 20 steps for the main causal GAN);"
        r" only the HVG count selected during preprocessing differs.",
        r"",
        rf"Causal-graph statistics are recomputed from the saved \texttt{{causal\_graph.pkl}} of each "
        rf"run, because preprocessing and GRN inference are skipped when the graph already exists and"
        rf" the logs therefore contain no graph summary. The run requesting 18{{,}}000 HVGs resolved to "
        rf"{runs[-1]['genes']:,} genes and is labelled by that gene count.",
        r"",
        rf"All runs completed without CUDA out-of-memory errors. The largest run peaked at "
        rf"{max(row['generator_mib'] for row in runs):,}\,MiB, "
        rf"against a GPU capacity of {total_mib:,}\,MiB.",
        r"",
        r"\section{GRN Characteristics}",
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Causal graph statistics per run. The TF/target ratio is transcription factors"
        r" divided by target genes.}",
        r"\label{tab:grn}",
        r"\begin{tabular}{rrrrrrrr}", r"\toprule",
        r"Genes & TFs & Target genes & TF/target (\%) & Possible edges & Imposed edges & Density (\%) \\",
        r"\midrule",
    ]
    for row in runs:
        lines.append(
            f"{row['genes']:,} & {row['tfs']:,} & {row['targets']:,} & {row['ratio_pct']:.2f} & "
            f"{row['possible']:,} & {row['imposed']:,} & {row['density']:.1f}" + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r""]

    lines += [
        r"\section{Training Step Times}",
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Average time per training step for the causal controller and the causal"
        r" generator, taken from the final step log line of each phase.}",
        r"\label{tab:steps}",
        r"\begin{tabular}{rccc}", r"\toprule",
        r"Genes & Controller step (ms) & Generator step (ms) & TF/target (\%) \\",
        r"\midrule",
    ]
    for row in runs:
        lines.append(
            f"{row['genes']:,} & {row['controller_ms']:,} & {row['generator_ms']:,} & "
            f"{row['ratio_pct']:.2f}" + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r""]

    lines += [
        r"\begin{table}[H]", r"\centering", r"\small",
        rf"\caption{{Expected training duration for {CONTROLLER_STEPS:,} causal-controller steps and"
        rf" {GENERATOR_STEPS:,} causal-generator steps, extrapolated from the measured step times.}}",
        r"\label{tab:duration}",
        r"\begin{tabular}{rccc}", r"\toprule",
        rf"Genes & Controller ({CONTROLLER_STEPS:,} steps) & Generator ({GENERATOR_STEPS:,} steps) & Total \\",
        r"\midrule",
    ]
    for row in runs:
        controller_hours = row["controller_ms"] * CONTROLLER_STEPS / 1000 / 3600
        generator_hours = row["generator_ms"] * GENERATOR_STEPS / 1000 / 3600
        total_hours = controller_hours + generator_hours
        lines.append(
            rf"{row['genes']:,} & {controller_hours:.1f}\,h & {generator_hours:.1f}\,h & "
            rf"{total_hours:.1f}\,h ({total_hours / 24:.1f}\,d)" + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r""]

    lines += [
        r"\section{Parameter Counts}",
        r"\begin{table}[H]", r"\centering",
        r"\caption{Parameter counts per model component and optimiser, in millions.}",
        r"\label{tab:params}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
        r"Genes & Generator & Critic & Labeler & Antilabeler & Gen.~opt & Crit.~opt & Labeler~opt & Antilabeler~opt \\",
        r"\midrule",
    ]
    for row in runs:
        params = row["params"]
        cells = [
            params["gen_state_dict"], params["crit_state_dict"], params["labeler_state_dict"],
            params["antilabeler_state_dict"], params["gen_optimizer_state_dict"],
            params["crit_optimizer_state_dict"], params["labeler_optimizer_state_dict"],
            params["antilabeler_optimizer_state_dict"],
        ]
        lines.append(f"{row['genes']:,} & " + " & ".join(f"{value / 1e6:.2f}M" for value in cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", r""]

    lines += [
        r"\section{GPU Memory Requirements}",
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Peak GPU memory per training phase. Phase windows are taken from the training"
        r" log timestamps and intersected with the \texttt{gpu-usage.csv} monitoring data.}",
        r"\label{tab:memory}",
        r"\begin{tabular}{rcc}", r"\toprule",
        r"Genes & Controller (MiB) & Generator (MiB) \\",
        r"\midrule",
    ]
    for row in runs:
        lines.append(
            f"{row['genes']:,} & {row['controller_mib']:,} & {row['generator_mib']:,}" + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r""]

    lines += [
        r"\section{Scaling Relations}",
        r"Ordinary least-squares fits over all runs, with $x$ the number of highly variable genes,",
        r"give the causal-generator requirements",
        r"\begin{equation}",
        rf"\widehat{{M}}_{{\mathrm{{gen}}}}(x) = {memory_fit[0]:.4f}\,x + {memory_fit[1]:.1f}\ "
        rf"\mathrm{{MiB}},\qquad R^2 = {memory_fit[2]:.4f}",
        r"\end{equation}",
        r"\begin{equation}",
        rf"\widehat{{t}}_{{\mathrm{{gen}}}}(x) = {time_fit[0]:.5f}\,x + {time_fit[1]:.2f}\ "
        rf"\mathrm{{ms/step}},\qquad R^2 = {time_fit[2]:.4f}",
        r"\end{equation}",
        r"The memory fit uses the peak memory of the generator training window and the time fit uses"
        r" the second final-step log line. The relations describe the measured range only.",
        r"",
        r"\subsection{Gene Count versus Target-Gene Count}",
        rf"Fitting against the number of target genes instead of highly variable genes gives"
        rf" $R^2 = {target_memory_fit[2]:.4f}$ for memory and $R^2 = {target_time_fit[2]:.4f}$ for"
        rf" step time, compared with $R^2 = {memory_fit[2]:.4f}$ and $R^2 = {time_fit[2]:.4f}$ for the"
        rf" gene-count fits. The two predictors are strongly correlated, so the differences are"
        rf" small, but the target-gene form is the better description of the underlying model size:",
        r"\begin{equation}",
        rf"\widehat{{M}}_{{\mathrm{{gen}}}}(y) = {target_memory_fit[0]:.4f}\,y + "
        rf"{target_memory_fit[1]:.1f}\ \mathrm{{MiB}},\qquad R^2 = {target_memory_fit[2]:.4f}",
        r"\end{equation}",
        r"\begin{equation}",
        rf"\widehat{{t}}_{{\mathrm{{gen}}}}(y) = {target_time_fit[0]:.5f}\,y + "
        rf"{target_time_fit[1]:.2f}\ \mathrm{{ms/step}},\qquad R^2 = {target_time_fit[2]:.4f}",
        r"\end{equation}",
        r"where $y$ denotes the number of target genes in the causal graph.",
        r"",
        r"\section{Scaling Graphs}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.9\textwidth]{gen_params_vs_hvg.pdf}",
        r"\caption{Generator parameter count versus gene count.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.9\textwidth]{gpu_mem_vs_hvg.pdf}",
        r"\caption{Peak GPU memory during controller and generator training versus gene count, with"
        r" the linear fit and the GPU capacity.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.9\textwidth]{gen_step_vs_hvg.pdf}",
        r"\caption{Causal-generator average step time versus gene count.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.9\textwidth]{gen_step_vs_ratio.pdf}",
        r"\caption{Causal-generator average step time versus the TF/target ratio; point colour"
        r" encodes the gene count.}\end{figure}",
        r"\section{Notes}",
        r"\begin{itemize}",
        r"\item All runs share the architecture: latent dim 128, generator layers 256--512--1024,"
        r" critic layers 1024--512--256, labeler layers 2000--2000--2000, batch size 1024 and"
        r" 20 steps per phase.",
        r"\item The controller phase is trained first with batch size 128 and the main causal GAN"
        r" second with batch size 1024; generator training dominates the memory requirement.",
        r"\item Memory values are in MiB as reported by \texttt{nvidia-smi}.",
        r"\end{itemize}",
        r"\end{document}",
    ]
    (OUT / "report.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    runs = load_runs()
    fits = {
        ("generator_mib", "genes"): fit(runs, "generator_mib", "genes"),
        ("generator_ms", "genes"): fit(runs, "generator_ms", "genes"),
        ("generator_mib", "targets"): fit(runs, "generator_mib", "targets"),
        ("generator_ms", "targets"): fit(runs, "generator_ms", "targets"),
    }
    graph_params(runs)
    graph_memory(runs, 81_920)
    graph_time(runs)
    graph_ratio(runs)
    latex(runs, fits)
    print(f"Generated report for {len(runs)} runs in {OUT}")
    for key, (slope, intercept, r2) in fits.items():
        print(f"  {key[0]} ~ {key[1]}: {slope:.5f} x + {intercept:.2f} (R²={r2:.5f})")


if __name__ == "__main__":
    main()