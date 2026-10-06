#!/usr/bin/env python3
"""Generate the TF-number experiment report and scaling graphs."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
LOG_ROOT = ROOT / "logs" / "causalgan-tf-num"
RESULT_ROOT = ROOT / "results"
OUT = ROOT / "report"


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    selected_match = re.search(r"of the\s+(\d+)\s+selected TFs", text, re.S)
    retained_match = re.search(r"retained\s+(\d+)\s+limit-tfs\.py", text)
    if retained_match:
        retained = int(retained_match.group(1))
    else:
        retained_line = next((line for line in text.splitlines() if "of the" in line and "selected TFs" in line), "")
        retained_match = re.search(r"^\s*(\d+)\s+of the", retained_line)
        retained = int(retained_match.group(1)) if retained_match else None
    graph = {}
    patterns = {
        "tfs": r"``TFs``\s+(\d+)",
        "targets": r"``Targets``\s+(\d+)",
        "genes": r"^\s*Genes\s+(\d+)",
        "possible": r"Possible Edges\s+(\d+)",
        "imposed": r"Imposed Edges\s+(\d+)",
        "density": r"GRN density Edges\s+([\d.]+)%",
    }
    for key, pattern in patterns.items():
        found = re.findall(pattern, text, re.MULTILINE)
        if found:
            graph[key] = float(found[-1]) if key == "density" else int(found[-1])
    steps = [int(x) for x in re.findall(r"Step 20/20 completed in \d+ milliseconds \(average:\s+(\d+)\s+ms\)", text)]
    return {
        "bipartite": retained,
        "selected": int(selected_match.group(1)) if selected_match else None,
        "graph": graph,
        "step_ms": steps[1] if len(steps) > 1 else None,
    }


def load_runs() -> list[dict]:
    runs = []
    for log in LOG_ROOT.glob("*/*.out"):
        parsed = parse_log(log)
        result_match = re.search(r"results/(\d+)/", log.read_text(errors="replace"))
        if not result_match:
            continue
        requested = int(result_match.group(1))
        gpu_path = RESULT_ROOT / str(requested) / "gpu-usage.csv"
        if not gpu_path.exists():
            continue
        with gpu_path.open() as f:
            memory = max(int(row["memory_used"]) for row in csv.DictReader(f))
        parsed.update(requested=requested, memory_mib=memory)
        runs.append(parsed)
    return sorted(runs, key=lambda row: row["requested"])


def fit(runs: list[dict], key: str) -> tuple[float, float, float]:
    x = np.array([row["requested"] for row in runs], dtype=float)
    y = np.array([row[key] for row in runs], dtype=float)
    coeff = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return float(coeff[0]), float(coeff[1]), r2


def fit_predictor(runs: list[dict], metric: str, predictor: str) -> tuple[float, float, float]:
    x = np.array([row[predictor] for row in runs], dtype=float)
    y = np.array([row[metric] for row in runs], dtype=float)
    coeff = np.polyfit(x, y, 1)
    return float(coeff[0]), float(coeff[1]), float(np.corrcoef(x, y)[0, 1] ** 2)


def graph(runs: list[dict], key: str, ylabel: str, filename: str, title: str, unit_scale: float = 1.0) -> None:
    x = np.array([row["requested"] for row in runs], dtype=float)
    y = np.array([row[key] for row in runs], dtype=float) / unit_scale
    slope, intercept, r2 = fit(runs, key)
    fit_x = np.linspace(x.min(), x.max(), 200)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, "o-", color="#1565C0", markersize=5, label="Measured")
    ax.plot(fit_x, (slope * fit_x + intercept) / unit_scale, ":", color="#555555", label=f"Linear fit ($R^2$={r2:.4f})")
    ax.set_xlabel("Requested number of TFs")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=300)
    plt.close(fig)


def latex(
    runs: list[dict],
    memory_fit: tuple[float, float, float],
    time_fit: tuple[float, float, float],
    target_memory_fit: tuple[float, float, float],
    target_time_fit: tuple[float, float, float],
) -> None:
    ms, mi, mr2 = memory_fit
    ts, ti, tr2 = time_fit
    tms, tmi, tmr2 = target_memory_fit
    tts, tti, ttr2 = target_time_fit
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=2.2cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\usepackage{graphicx}",
        r"\title{TF-Number Experiment 2}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"This experiment holds the input at 3,000 highly variable genes and varies the requested TF count.",
        r"The bipartite-retained count is measured by \texttt{limit-tfs.py}; the causal-graph count is measured after top-15 TF subsetting by \texttt{grn\_creation.py}.",
        r"\begin{equation}",
        rf"\widehat{{M}}(x) = {ms:.4f}x + {mi:.1f}\ \mathrm{{MiB}},\qquad R^2={mr2:.4f}",
        r"\end{equation}",
        r"\begin{equation}",
        rf"\widehat{{t}}(x) = {ts:.5f}x + {ti:.2f}\ \mathrm{{ms/step}},\qquad R^2={tr2:.4f}",
        r"\end{equation}",
        r"The fits are ordinary least-squares fits over the completed runs and use requested TFs as $x$.",
        r"\subsection{Requested TFs versus Target Genes}",
        rf"Target-gene count is not a better predictor in this experiment: its linear fits have "
        rf"$R^2={tmr2:.4f}$ for memory and $R^2={ttr2:.4f}$ for time, identical to the requested-TF fits. "
        r"This is expected because the runs use a fixed 3,000-gene input and satisfy "
        r"$\mathrm{targets}=3{,}000-\mathrm{requested\ TFs}$ (apart from graph genes excluded for lack of regulators).",
        r"The equivalent target-based fits are:",
        r"\begin{equation}",
        rf"\widehat{{M}}(y) = {tms:.4f}y + {tmi:.1f}\ \mathrm{{MiB}},\qquad R^2={tmr2:.4f}",
        r"\end{equation}",
        r"\begin{equation}",
        rf"\widehat{{t}}(y) = {tts:.5f}y + {tti:.2f}\ \mathrm{{ms/step}},\qquad R^2={ttr2:.4f}",
        r"\end{equation}",
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Requested and effective TF counts, plus final causal-graph size.}",
        r"\begin{tabular}{rrrrrrrr}", r"\toprule",
        r"Requested & Bipartite retained & Graph TFs & Targets & Genes & Possible edges & Imposed edges & Density\\",
        r"\midrule",
    ]
    for row in runs:
        g = row["graph"]
        lines.append(
            f"{row['requested']:,} & {row['bipartite']:,} & {g['tfs']:,} & {g['targets']:,} & {g['genes']:,} & "
            f"{g['possible']:,} & {g['imposed']:,} & {g['density']:.1f}" + r"\%\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
        r"\begin{table}[H]", r"\centering", r"\small",
        r"\caption{Measured causal-generator time and memory requirements.}",
        r"\begin{tabular}{rrrr}", r"\toprule",
        r"Requested TFs & Target genes & Avg step time (ms) & Peak memory (MiB)\\",
        r"\midrule",
    ]
    for row in runs:
        lines.append(
            f"{row['requested']:,} & {row['graph']['targets']:,} & {row['step_ms']:,} & {row['memory_mib']:,}" + r"\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.9\textwidth]{memory_vs_requested_tfs.pdf}",
        r"\caption{Peak GPU memory requirement versus requested TF count.}\end{figure}",
        r"\begin{figure}[H]\centering\includegraphics[width=0.9\textwidth]{time_vs_requested_tfs.pdf}",
        r"\caption{Causal-generator average step time versus requested TF count.}\end{figure}",
        r"\end{document}",
    ]
    (OUT / "report.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    runs = [row for row in load_runs() if row["bipartite"] is not None and row["step_ms"] is not None]
    memory_fit = fit(runs, "memory_mib")
    time_fit = fit(runs, "step_ms")
    for row in runs:
        row["target_count"] = row["graph"]["targets"]
    target_memory_fit = fit_predictor(runs, "memory_mib", "target_count")
    target_time_fit = fit_predictor(runs, "step_ms", "target_count")
    graph(runs, "memory_mib", "Peak GPU memory (GB)", "memory_vs_requested_tfs.pdf", "GPU Memory vs. Requested TF Count", 1024)
    graph(runs, "step_ms", "Generator average step time (ms)", "time_vs_requested_tfs.pdf", "Generator Time vs. Requested TF Count")
    latex(runs, memory_fit, time_fit, target_memory_fit, target_time_fit)
    print(f"Generated report for {len(runs)} runs in {OUT}")
    print(f"Memory fit: {memory_fit[0]:.4f} x + {memory_fit[1]:.1f} MiB (R²={memory_fit[2]:.4f})")
    print(f"Time fit: {time_fit[0]:.5f} x + {time_fit[1]:.2f} ms/step (R²={time_fit[2]:.4f})")


if __name__ == "__main__":
    main()
