#!/usr/bin/env python3
"""Generate the TF-count scaling report for publication/tf-num-experiment, indexed by target-gene count."""

import csv
import importlib.util
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale/publication/tf-num-experiment"
LOG_JOB = "35274208"  # job whose logs match the current results/ directory (sizes 200..2800)
LOG_DIR = os.path.join(EXP, "logs", "causalgan-tf-num", LOG_JOB)
RESULTS = os.path.join(EXP, "results")
OUT = os.path.join(EXP, "report")
SIZES = list(range(200, 2801, 200))  # requested TF counts; logs are task 1..14
N_GENES_DATASET = 3000
MAX_DISCARDED = 100  # subset criterion: fewer than 100 genes discarded from the dataset

MODEL_PATH = "/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale/publication/theoretical-size/gpu_memory_model.py"
_spec = importlib.util.spec_from_file_location("gpu_memory_model", MODEL_PATH)
gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gm)
# Controller state-dict count from the analytic model. Validated against the controller's own
# checkpoint (results/200/CC/checkpoints/step_20.pth): model 3,772,094 vs stored 3,772,092, i.e. the
# model counts the 3 int64 num_batches_tracked entries as 6 elements rather than 3.
CC_COUNT_OFFSET = 2

TS_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]")
STEP20_RE = re.compile(r"Step 20/20 completed in \d+ milliseconds \(average:\s+(\d+)\s+ms\)")
GRN_PATTERNS = {
    "tfs": r"``TFs``\s+(\d+)",
    "targets": r"``Targets``\s+(\d+)",
    "genes": r"^\s*Genes\s+(\d+)",
    "possible_edges": r"Possible Edges\s+(\d+)",
    "imposed_edges": r"Imposed Edges\s+(\d+)",
}
DENSITY_RE = re.compile(r"GRN density Edges\s+([\d.]+)%")


def controller_params(tfs, targets):
    """Parameters in the causal controller (genes = TFs + targets), excluded from the generator."""
    return int(gm.compute_cc(tfs, targets, gm.BATCH_SIZE)["model"] / 4) - CC_COUNT_OFFSET


def backward_ts(lines, idx):
    for i in range(idx, -1, -1):
        m = TS_RE.search(lines[i])
        if m:
            return m.group(1)
    return None


def forward_ts(lines, idx):
    for j in range(idx, min(idx + 50, len(lines))):
        m = TS_RE.search(lines[j])
        if m:
            return m.group(1)
    return None


def secs(ts):
    h, m, s = map(int, ts.strip().split(" ")[-1].split(":"))
    return h * 3600 + m * 60 + s


def parse_log(path):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    info = {}
    # The final causal-graph block is the one used for training.
    graph_idx = [i for i, l in enumerate(lines) if "Causal graph info" in l]
    scan = lines[graph_idx[-1]:graph_idx[-1] + 12] if graph_idx else []
    for line in scan:
        for key, pattern in GRN_PATTERNS.items():
            m = re.search(pattern, line)
            if m and key not in info:
                info[key] = int(m.group(1))
        m = DENSITY_RE.search(line)
        if m and "density" not in info:
            info["density"] = float(m.group(1))

    step_idx, step_vals = [], []
    for i, line in enumerate(lines):
        m = STEP20_RE.search(line)
        if m:
            step_idx.append(i)
            step_vals.append(int(m.group(1)))
    info["controller_step_avg"] = step_vals[0] if step_vals else None
    info["generator_step_avg"] = step_vals[1] if len(step_vals) > 1 else None

    starts = [backward_ts(lines, i) for i, l in enumerate(lines) if "Starting training loop" in l]
    inits = [backward_ts(lines, i) for i, l in enumerate(lines)
             if "Initializing training" in l and "libraries" not in l.lower()]
    info["ctrl_window"] = (starts[0] if starts else None,
                           forward_ts(lines, step_idx[0]) if step_idx else None)
    gen_start = inits[1] if len(inits) > 1 else (starts[1] if len(starts) > 1 else None)
    gen_end = forward_ts(lines, step_idx[1]) if len(step_idx) > 1 else None
    info["gen_window"] = (gen_start, gen_end)
    info["oom"] = any("OutOfMemoryError" in l for l in lines)
    return info


def parse_gpu(path, ctrl_window, gen_window):
    rows = list(csv.DictReader(open(path)))
    mem = [int(r["memory_used"]) for r in rows]
    times = [secs(r["timestamp"]) for r in rows]

    def window_max(window):
        if not window[0] or not window[1]:
            return None
        a, b = secs(window[0]), secs(window[1])
        vals = [m for t, m in zip(times, mem) if a <= t <= b]
        return max(vals) if vals else None

    ctrl = window_max(ctrl_window)
    gen = window_max(gen_window)
    threshold_idx = next((i for i, v in enumerate(mem) if v > 5000), None)
    if ctrl is None:
        ctrl = max(mem[:threshold_idx]) if threshold_idx else max(mem)
    if gen is None:
        gen = max(mem[threshold_idx:]) if threshold_idx is not None else 0
    return {
        "controller_max": ctrl,
        "generator_max": gen,
        "overall_max": max(mem),
        "memory_total": int(rows[0]["memory_total"]),
    }


def load_runs():
    runs = {}
    for idx, requested in enumerate(SIZES, start=1):
        info = parse_log(os.path.join(LOG_DIR, f"{idx}.out"))
        with open(os.path.join(RESULTS, str(requested), "params-size.json")) as f:
            params = json.load(f)
        gpu = parse_gpu(os.path.join(RESULTS, str(requested), "gpu-usage.csv"),
                        info["ctrl_window"], info["gen_window"])
        info["requested_tfs"] = requested
        info["tf_target_ratio"] = info["tfs"] / info["targets"]
        info["genes_discarded"] = N_GENES_DATASET - info["genes"]
        info["controller_params"] = controller_params(info["tfs"], info["targets"])
        info["gen_only_params"] = params["gen_state_dict"] - info["controller_params"]
        runs[requested] = {**info, "params": params, "gpu": gpu}
    return runs


def fit(x, y):
    """Ordinary least-squares line y = c1*x + c0."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = np.polyfit(x, y, 1)
    res = y - np.polyval(c, x)
    r2 = 1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)
    return {"c": c, "r2": r2, "rmse": float(np.sqrt(np.mean(res ** 2)))}


def tex_num(c, sig=4):
    if c == 0:
        return "0"
    e = int(math.floor(math.log10(abs(c))))
    if -2 <= e <= 3:
        return f"{c:.{max(0, sig - 1 - e)}f}"
    return f"{c / 10 ** e:.{sig - 1}f}\\times10^{{{e}}}"


def tex_line(c, var, unit):
    """Linear equation text: slope * var + intercept, with unit."""
    slope, intercept = c
    sign = "-" if intercept < 0 else "+"
    slope_s = tex_num(abs(slope)) if abs(abs(slope) - 1) > 1e-12 else ""
    return f"{'-' if slope < 0 else ''}{slope_s}\\,{var} {sign} {tex_num(abs(intercept))}\\ \\mathrm{{{unit}}}"


def make_graphs(runs, fits, out, targets, grid, subset):
    """targets: measured target counts for all runs; grid: smooth x-grid; subset: booleans per run."""
    os.makedirs(out, exist_ok=True)
    params = np.array([runs[s]["gen_only_params"] for s in SIZES], float)
    memory = np.array([runs[s]["gpu"]["generator_max"] for s in SIZES], float)
    times = np.array([runs[s]["generator_step_avg"] for s in SIZES], float)
    inc = np.array(subset, bool)

    def measured(ax, y, color, scale=1.0):
        ax.plot(targets[inc], y[inc] / scale, "o", color=color, markersize=6, label="Measured (fit subset)", zorder=5)
        ax.plot(targets[~inc], y[~inc] / scale, "o", mfc="white", mec=color, markersize=6,
                label="Measured (excluded)", zorder=5)

    def lines(ax, key, scale=1.0):
        f_all, f_sub = fits[key + "_all"], fits[key + "_sub"]
        ax.plot(grid, np.polyval(f_all["c"], grid) / scale, ":", color="#666",
                label=f"Linear fit, all runs ($R^2$={f_all['r2']:.4f})")
        ax.plot(grid, np.polyval(f_sub["c"], grid) / scale, "--", color="#8E44AD",
                label=f"Linear fit, fit subset ($R^2$={f_sub['r2']:.4f})")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    measured(ax, params, "#2196F3")
    lines(ax, "param")
    ax.set(xlabel="Target genes", ylabel="Generator Parameter Count (excl. controller)",
           title="Generator Parameter Count vs. Target Genes")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gen_params_vs_targets.pdf"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    measured(ax, memory, "#FF9800", scale=1024)
    lines(ax, "mem", scale=1024)
    ax.set(xlabel="Target genes", ylabel="Peak GPU Memory (GB)",
           title="GPU Memory Requirement vs. Target Genes")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gpu_mem_vs_targets.pdf"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    measured(ax, times, "#2196F3")
    lines(ax, "time")
    ax.set(xlabel="Target genes", ylabel="Generator Avg Step Time (ms)",
           title="Generator Step Time vs. Target Genes")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gen_step_vs_targets.pdf"), dpi=300)
    plt.close(fig)


def figure_env(filename, caption, label):
    return (r"\begin{figure}[H]\centering" + "\n"
            + rf"\includegraphics[width=0.95\textwidth]{{{filename}}}" + "\n"
            + rf"\caption{{{caption}}}\label{{{label}}}" + "\n"
            + r"\end{figure}")


def write_latex(runs, fits, out, subset_sizes):
    ORDER = sorted(SIZES, key=lambda s: runs[s]["targets"])  # tables ordered by target genes, ascending
    n_oom = sum(runs[s]["oom"] for s in SIZES)
    lo, hi = min(subset_sizes), max(subset_sizes)
    max_disc_sub = max(runs[s]["genes_discarded"] for s in subset_sizes)
    min_disc_excl = min(runs[s]["genes_discarded"] for s in SIZES if s not in subset_sizes)
    a = []
    a.append(r"""\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{float}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\title{GRouNdScale TF-Count Scaling Report}
\author{Auto-generated from \texttt{publication/tf-num-experiment} run logs}
\date{\today}
\begin{document}
\maketitle
\section{Overview}""")
    a.append(r"""This report analyses the 14 completed runs of the publication TF-count experiment. The number of TFs was swept from 200 to 2,800 in steps of 200 while the GRN parameter $k$ was held fixed at 15. Results are indexed by the number of \emph{target genes} in the causal graph, as requested. Logs are taken from Slurm job \texttt{35274208}, which matches the current \texttt{results/} directory. The older sweep (job \texttt{33898186}/\texttt{33898382}, sizes 100 to 2,900) and \texttt{results-old/} are excluded.""")
    a.append(r"""\paragraph{Requested versus realised TFs.} The sweep requests a number of TFs, but the causal graph that is actually trained on retains a different number once the bipartite filtering removes TFs that regulate only other selected TFs. The realised count matches the request up to 1,200 TFs, is nearly equal at 1,400 and 1,600, and then falls below it (1,155 realised for 2,800 requested). This report uses the \emph{realised} counts from the causal-graph block of each log, and Table~\ref{tab:grn} lists both.""")
    a.append(r"""\paragraph{Genes discarded and the fit subset.} The dataset has 3,000 genes. As the TF count grows, some genes are discarded from the causal graph; the number discarded is listed in Table~\ref{tab:grn}. Runs with fewer than 100 genes discarded (requested TFs """ + f"{lo}--{hi}" + r""", """ + f"{len(subset_sizes)}" + r""" runs) form a \emph{fit subset}. Beyond that, discarding grows quickly, so the subset fit tests whether the discarding is what drives the scaling.""")
    a.append(r"""\paragraph{Missing run.} There is no 3,000-TF run in either \texttt{results/} or the logs. With 3,000 genes in the dataset, 3,000 TFs would leave no target genes, which is the likely reason it was not formed; no log confirms this directly.""")

    a.append(r"\section{GRN Characteristics}")
    a.append(r"\begin{table}[H]\centering")
    a.append(r"\caption{GRN statistics per run, ordered by target genes. \emph{Requested} is the swept TF count; the remaining columns are read from the causal-graph block of each log. Genes discarded is $3{,}000$ minus the genes retained in the causal graph.}")
    a.append(r"\label{tab:grn}")
    a.append(r"\resizebox{\textwidth}{!}{%")
    a.append(r"\begin{tabular}{rrrrrrrrrr}\toprule")
    a.append(r"Target genes & Requested TFs & TFs & Genes & Genes discarded & TF/Target (\%) & Imposed Edges & Possible Edges & Density (\%) \\ \midrule")
    for s in ORDER:
        r = runs[s]
        a.append(f"{r['targets']} & {s} & {r['tfs']} & {r['genes']} & {r['genes_discarded']} & {100 * r['tf_target_ratio']:.2f} & "
                 f"{r['imposed_edges']:,} & {r['possible_edges']:,} & {r['density']} \\\\")
    a.append(r"\bottomrule\end{tabular}}\end{table}")

    a.append(r"\section{Training Step Times}")
    a.append(r"\begin{table}[H]\centering")
    a.append(r"\caption{Average final-step times (ms) for the causal controller (first phase) and causal generator (second phase).}")
    a.append(r"\label{tab:step}")
    a.append(r"\begin{tabular}{rrrr}\toprule Target genes & Requested TFs & Controller (ms) & Generator (ms) \\ \midrule")
    for s in ORDER:
        r = runs[s]
        a.append(f"{r['targets']} & {s} & {r['controller_step_avg']} & {r['generator_step_avg']} \\\\")
    a.append(r"\bottomrule\end{tabular}\end{table}")

    a.append(r"\begin{table}[H]\centering")
    a.append(r"\caption{Expected duration for 200,000 controller steps and 500,000 generator steps.}")
    a.append(r"\label{tab:duration}")
    a.append(r"\begin{tabular}{rrrrr}\toprule Target genes & Requested TFs & Controller & Generator & Total \\ \midrule")
    for s in ORDER:
        r = runs[s]
        c = r["controller_step_avg"] * 200000 / 3.6e6
        g = r["generator_step_avg"] * 500000 / 3.6e6
        a.append(f"{r['targets']} & {s} & {c:.1f} h & {g:.1f} h & {c + g:.1f} h ({(c + g) / 24:.2f} d) \\\\")
    a.append(r"\bottomrule\end{tabular}\end{table}")

    a.append(r"\section{Parameter Counts}")
    a.append(r"\begin{table}[H]\centering\resizebox{\textwidth}{!}{%")
    a.append(r"\begin{tabular}{rrrrrrrrrr}\toprule Target genes & Generator (excl. CC) & Controller & Critic & Labeler & Anti-Lab. & Gen. Opt. & Crit. Opt. & Lab. Opt. & Anti-Lab. Opt. \\ \midrule")
    fields = ["crit_state_dict", "labeler_state_dict", "antilabeler_state_dict",
              "gen_optimizer_state_dict", "crit_optimizer_state_dict",
              "labeler_optimizer_state_dict", "antilabeler_optimizer_state_dict"]
    for s in ORDER:
        r = runs[s]
        p = r["params"]
        a.append(f"{r['targets']} & {r['gen_only_params'] / 1e6:.2f}M & {r['controller_params'] / 1e6:.2f}M & "
                 + " & ".join(f"{p[f] / 1e6:.2f}M" for f in fields) + r" \\")
    a.append(r"\bottomrule\end{tabular}}")
    a.append(r"\caption{Parameter counts for each component and optimiser, from \texttt{params-size.json}. The generator column excludes the causal controller, which is a submodule of the stored \texttt{gen\_state\_dict} and is shown separately.}")
    a.append(r"\end{table}")

    a.append(r"\section{GPU Memory}")
    a.append(r"\begin{table}[H]\centering")
    a.append(r"\caption{Peak GPU memory (MiB) by training phase, from the monitoring CSV using the logged phase windows.}")
    a.append(r"\label{tab:gpu}")
    a.append(r"\begin{tabular}{rrrr}\toprule Target genes & Requested TFs & Controller (MiB) & Generator (MiB) \\ \midrule")
    for s in ORDER:
        r = runs[s]
        g = r["gpu"]
        a.append(f"{r['targets']} & {s} & {g['controller_max']:,} & {g['generator_max']:,} \\\\")
    a.append(r"\bottomrule\end{tabular}\end{table}")

    a.append(r"\section{Linear Fits in Target Genes}\label{sec:fits}")
    a.append(r"Let $T$ denote the number of target genes. Two ordinary least-squares lines are reported for each quantity: one fitted to all 14 runs, and one fitted to the fit subset (fewer than 100 genes discarded). Quadratic fits are not reported, since the theoretical model does not support curvature in these quantities.")
    a.append(r"\begin{align}")
    a.append(r"\widehat{P}_{\mathrm{gen}}(T) &= " + tex_line(fits["param_all"]["c"], "T", "parameters")
             + rf",\quad R^2={fits['param_all']['r2']:.4f} \quad\text{{(all runs)}}, \\")
    a.append(r"\widehat{P}_{\mathrm{gen}}^{\mathrm{sub}}(T) &= " + tex_line(fits["param_sub"]["c"], "T", "parameters")
             + rf",\quad R^2={fits['param_sub']['r2']:.4f} \quad\text{{(fit subset)}}, \\")
    a.append(r"\widehat{M}_{\mathrm{gen}}(T) &= " + tex_line(fits["mem_all"]["c"], "T", "MiB")
             + rf",\quad R^2={fits['mem_all']['r2']:.4f} \quad\text{{(all runs)}}, \\")
    a.append(r"\widehat{M}_{\mathrm{gen}}^{\mathrm{sub}}(T) &= " + tex_line(fits["mem_sub"]["c"], "T", "MiB")
             + rf",\quad R^2={fits['mem_sub']['r2']:.4f} \quad\text{{(fit subset)}}, \\")
    a.append(r"\widehat{t}_{\mathrm{gen}}(T) &= " + tex_line(fits["time_all"]["c"], "T", "ms/step")
             + rf",\quad R^2={fits['time_all']['r2']:.4f} \quad\text{{(all runs)}}, \\")
    a.append(r"\widehat{t}_{\mathrm{gen}}^{\mathrm{sub}}(T) &= " + tex_line(fits["time_sub"]["c"], "T", "ms/step")
             + rf",\quad R^2={fits['time_sub']['r2']:.4f} \quad\text{{(fit subset)}}.")
    a.append(r"\end{align}")

    a.append(r"\begin{table}[H]\centering")
    a.append(r"\caption{Goodness of fit in target genes, for all runs and for the fit subset.}")
    a.append(r"\label{tab:fits}")
    a.append(r"\begin{tabular}{lrrrr}\toprule Quantity & $R^2$ (all) & RMSE (all) & $R^2$ (subset) & RMSE (subset) \\ \midrule")
    for label, key in [("Generator parameters", "param"), ("Generator peak memory (MiB)", "mem"),
                       ("Generator step time (ms)", "time")]:
        a.append(f"{label} & {fits[key + '_all']['r2']:.4f} & {fits[key + '_all']['rmse']:,.2f} & "
                 f"{fits[key + '_sub']['r2']:.4f} & {fits[key + '_sub']['rmse']:,.2f} \\\\")
    a.append(r"\bottomrule\end{tabular}\end{table}")

    a.append(r"\section{Graphs}")
    a.append(figure_env("gen_params_vs_targets.pdf", "Generator parameter count (excluding the controller) versus target genes. Filled points are the fit subset; hollow points are excluded runs. Dotted: linear fit to all runs; dashed: linear fit to the subset.", "fig:params"))
    a.append(figure_env("gpu_mem_vs_targets.pdf", "Peak generator GPU memory versus target genes, with the same two linear fits.", "fig:memory"))
    a.append(figure_env("gen_step_vs_targets.pdf", "Causal-generator average step time versus target genes, with the same two linear fits.", "fig:time"))

    a.append(r"\section{Notes}")
    a.append(r"\begin{itemize}")
    excl = [s for s in SIZES if s not in subset_sizes]
    t_excl = np.array([runs[s]["targets"] for s in excl], float)
    comp = []
    for key, get, label in [("param", lambda s: runs[s]["gen_only_params"], "parameters"),
                            ("mem", lambda s: runs[s]["gpu"]["generator_max"], "peak memory"),
                            ("time", lambda s: runs[s]["generator_step_avg"], "step time")]:
        slope_change = 100 * (fits[key + "_sub"]["c"][0] / fits[key + "_all"]["c"][0] - 1)
        actual = np.array([get(s) for s in excl], float)
        pred = np.polyval(fits[key + "_sub"]["c"], t_excl)
        err = 100 * np.max(np.abs(pred / actual - 1))
        comp.append(f"{label}: slope {slope_change:+.1f}\\%, subset line misses excluded runs by up to {err:.1f}\\%")
    a.append(rf"\item Fit subset: the {len(subset_sizes)} runs with fewer than {MAX_DISCARDED} genes discarded (largest discard in the subset: {max_disc_sub}; smallest discard among excluded runs: {min_disc_excl}). Comparing the subset line with the all-run line: " + "; ".join(comp) + ". The subset covers a narrower target range (1,200 to 2,800), so these slope changes carry more uncertainty than the R\\textsuperscript{2} values.")
    a.append(r"\item Scope of each input. Generator parameters are \texttt{gen\_state\_dict} minus the controller's parameters, so they describe the causal generator alone. Optimiser counts are already generator-only: the generator optimiser holds about three times the sparse-generator parameter count, far below what it would be if the controller were included. Peak GPU memory is the maximum inside the generator-training window, which starts after the controller window closes. The controller's resident weights are about 14 MiB, under 0.2\% of the generator peak. Step time is the generator-phase step logged by the trainer; the causal generator's forward pass calls the controller, so that time is not separable from the logs.")
    a.append(r"\item TF/Target exceeds 100\% from 1{,}600 requested TFs onward, meaning the graph has more TFs than target genes; the ratio is reported as a plain percentage for consistency with the other experiments.")
    a.append(r"\item Imposed edges fall from 41{,}610 to 3{,}000 across the sweep, and the generator's parameter count falls with them. Target genes therefore track the sparse-edge budget as well as the output dimension, so the fits in $T$ describe this sweep rather than a pure dependence on target count.")
    a.append(rf"\item Runs with an out-of-memory error: {n_oom} of {len(SIZES)}.")
    a.append(r"\end{itemize}")
    a.append(r"\end{document}")
    with open(os.path.join(out, "report.tex"), "w") as f:
        f.write("\n".join(a))


def main():
    runs = load_runs()
    targets = np.array([runs[s]["targets"] for s in SIZES], float)
    grid = np.linspace(targets.min() - 100, targets.max() + 100, 300)
    subset_sizes = [s for s in SIZES if runs[s]["genes_discarded"] < MAX_DISCARDED]
    inc = [s in subset_sizes for s in SIZES]
    t_sub = np.array([runs[s]["targets"] for s in subset_sizes], float)

    getters = {
        "param": lambda s: runs[s]["gen_only_params"],
        "mem": lambda s: runs[s]["gpu"]["generator_max"],
        "time": lambda s: runs[s]["generator_step_avg"],
    }
    fits = {}
    for key, get in getters.items():
        fits[key + "_all"] = fit(targets, [get(s) for s in SIZES])
        fits[key + "_sub"] = fit(t_sub, [get(s) for s in subset_sizes])

    for s in SIZES:
        r = runs[s]
        print(f"req={s:4d} tfs={r['tfs']:4d} targets={r['targets']:4d} genes={r['genes']:4d} "
              f"discarded={r['genes_discarded']:3d} gen_only={r['gen_only_params']:,} "
              f"gen={r['generator_step_avg']}ms gpu_g={r['gpu']['generator_max']} subset={s in subset_sizes}")
    print("subset:", subset_sizes)
    for key in ["param", "mem", "time"]:
        print(key, "all R2=%.6f  subset R2=%.6f" % (fits[key + "_all"]["r2"], fits[key + "_sub"]["r2"]))

    os.makedirs(OUT, exist_ok=True)
    make_graphs(runs, fits, OUT, targets, grid, inc)
    write_latex(runs, fits, OUT, subset_sizes)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
