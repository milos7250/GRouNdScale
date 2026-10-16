#!/usr/bin/env python3
"""Generate the top-k (TFs per gene) scaling report for publication/tf-k-experiment."""

import csv
import json
import math
import os
import re

import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale/publication/tf-k-experiment"
LOG_JOB = "35270928"  # job whose logs match the current results/ directory
LOG_DIR = os.path.join(EXP, "logs", "causalgan-tf-k", LOG_JOB)
RESULTS = os.path.join(EXP, "results")
OUT = os.path.join(EXP, "report")
KS = list(range(1, 16))
MODEL_PATH = "/mnt/shared/scratch/mmicik/private/Geneformer/simulation/GRouNdScale/publication/theoretical-size/gpu_memory_model.py"
_spec = importlib.util.spec_from_file_location("gpu_memory_model", MODEL_PATH)
gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gm)
CC_COUNT_OFFSET = 2  # model counts 3 int64 num_batches_tracked as 6 elements; checkpoint counts 3


def controller_params(tfs, targets):
    """Causal-controller parameters (genes = TFs + targets), excluded from the generator count."""
    return int(gm.compute_cc(tfs, targets, gm.BATCH_SIZE)["model"] / 4) - CC_COUNT_OFFSET

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
    for line in lines:
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
    for k in KS:
        info = parse_log(os.path.join(LOG_DIR, f"{k}.out"))
        with open(os.path.join(RESULTS, str(k), "params-size.json")) as f:
            params = json.load(f)
        gpu = parse_gpu(os.path.join(RESULTS, str(k), "gpu-usage.csv"),
                        info["ctrl_window"], info["gen_window"])
        info["tf_target_ratio"] = info["tfs"] / info["targets"]
        info["controller_params"] = controller_params(info["tfs"], info["targets"])
        info["gen_only_params"] = params["gen_state_dict"] - info["controller_params"]
        runs[k] = {**info, "params": params, "gpu": gpu}
    return runs


def fit(x, y, deg):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = np.polyfit(x, y, deg)
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


def tex_poly(c, var, unit):
    deg = len(c) - 1
    out = ""
    for i, coef in enumerate(c):
        power = deg - i
        sign = "-" if coef < 0 else "+"
        mag = abs(coef)
        if power == 0:
            body = tex_num(mag)
        else:
            var_s = var if power == 1 else f"{var}^{power}"
            body = var_s if abs(mag - 1) < 1e-12 else f"{tex_num(mag)}\\,{var_s}"
        if i == 0:
            out += ("-" if sign == "-" else "") + body
        else:
            out += f" {sign} {body}"
    return f"{out}\\ \\mathrm{{{unit}}}"


def plot_series(ax, x, y, label, **kw):
    ax.plot(x, y, "o", color=kw.get("color", "#2196F3"), markersize=6, label=label, zorder=5)


def make_graphs(runs, fits, out):
    ks = np.array(KS, float)
    xx = np.linspace(0.5, 15.5, 300)
    os.makedirs(out, exist_ok=True)

    params = np.array([runs[k]["gen_only_params"] for k in KS], float)
    memory = np.array([runs[k]["gpu"]["generator_max"] for k in KS], float)
    times = np.array([runs[k]["generator_step_avg"] for k in KS], float)
    total = runs[1]["gpu"]["memory_total"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, params, "o", color="#2196F3", markersize=6, label="Measured", zorder=5)
    ax.plot(xx, np.polyval(fits["param1"]["c"], xx), ":", color="#666",
            label=f"Linear fit ($R^2$={fits['param1']['r2']:.4f})")
    ax.plot(xx, np.polyval(fits["param2"]["c"], xx), "--", color="#8E44AD",
            label=f"Quadratic fit ($R^2$={fits['param2']['r2']:.4f})")
    ax.set(xlabel="$k$ (TFs per gene)", ylabel="Generator Parameter Count (excl. controller)",
           title="Generator Parameter Count vs. $k$")
    ax.set_xticks(KS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gen_params_vs_k.pdf"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, memory / 1024, "o", color="#FF9800", markersize=6, label="Generator training", zorder=5)
    ax.plot(xx, np.polyval(fits["mem1"]["c"], xx) / 1024, ":", color="#666",
            label=f"Generator linear fit ($R^2$={fits['mem1']['r2']:.4f})")
    ax.plot(xx, np.polyval(fits["mem2"]["c"], xx) / 1024, "--", color="#8E44AD",
            label=f"Generator quadratic fit ($R^2$={fits['mem2']['r2']:.4f})")
    ax.set(xlabel="$k$ (TFs per gene)", ylabel="Peak GPU Memory (GB)",
           title="GPU Memory Requirement vs. $k$")
    ax.set_xticks(KS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gpu_mem_vs_k.pdf"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, times, "o", color="#2196F3", markersize=6, label="Measured", zorder=5)
    ax.plot(xx, np.polyval(fits["time1"]["c"], xx), ":", color="#666",
            label=f"Linear fit ($R^2$={fits['time1']['r2']:.4f})")
    ax.plot(xx, np.polyval(fits["time2"]["c"], xx), "--", color="#8E44AD",
            label=f"Quadratic fit ($R^2$={fits['time2']['r2']:.4f})")
    ax.set(xlabel="$k$ (TFs per gene)", ylabel="Generator Avg Step Time (ms)",
           title="Generator Step Time vs. $k$")
    ax.set_xticks(KS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gen_step_vs_k.pdf"), dpi=300)
    plt.close(fig)


def figure_env(filename, caption, label):
    return (r"\begin{figure}[H]\centering" + "\n"
            + rf"\includegraphics[width=0.95\textwidth]{{{filename}}}" + "\n"
            + rf"\caption{{{caption}}}\label{{{label}}}" + "\n"
            + r"\end{figure}")


def write_latex(runs, fits, out):
    total = runs[1]["gpu"]["memory_total"]
    n_oom = sum(runs[k]["oom"] for k in KS)
    a = []
    a.append(r"""\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{float}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\title{GRouNdScale Top-$k$ Causal-Graph Scaling Report}
\author{Auto-generated from \texttt{publication/tf-k-experiment} run logs}
\date{\today}
\begin{document}
\maketitle
\section{Overview}""")
    a.append(r"""This report analyses the 15 completed runs of the publication top-$k$ experiment. The GRN parameter $k$ is the number of top-ranked transcription factors retained per gene (TFs per gene), and $k$ is the independent variable throughout the scaling analysis, replacing imposed-edge count. Imposed edges are shown only as a descriptive column in Table~\ref{tab:grn}.""")
    a.append(r"""Logs are taken from Slurm job \texttt{35270928}, which matches the current \texttt{results/} directory. The older job \texttt{34606270} and \texttt{results-old/} are excluded.""")
    a.append(r"""\paragraph{Observed dimensions.} The HVG and target-gene counts are constant across runs, but the TF and gene counts are not perfectly fixed: the $k=1$ run has 194 TFs and 2,997 genes, while all other runs have 197 TFs and 3,000 genes. This is reported as observed; the Notes section includes a sensitivity check that excludes $k=1$.""")

    a.append(r"\section{GRN Characteristics}")
    a.append(r"\begin{table}[H]\centering")
    a.append(r"\caption{GRN statistics for each top-$k$ setting. Imposed edges are a derived descriptive quantity and are not used as the independent variable.}")
    a.append(r"\label{tab:grn}")
    a.append(r"\begin{tabular}{rrrrrrrr}\toprule")
    a.append(r"$k$ & TFs & Targets & Genes & TF/Target (\%) & Imposed Edges & Possible Edges & Density (\%) \\ \midrule")
    for k in KS:
        r = runs[k]
        a.append(f"{k} & {r['tfs']} & {r['targets']} & {r['genes']} & {100 * r['tf_target_ratio']:.2f} & "
                 f"{r['imposed_edges']:,} & {r['possible_edges']:,} & {r['density']} \\\\")
    a.append(r"\bottomrule\end{tabular}\end{table}")

    a.append(r"\section{Training Step Times}")
    a.append(r"\begin{table}[H]\centering")
    a.append(r"\caption{Average final-step times (ms) for the causal controller (first phase) and causal generator (second phase).}")
    a.append(r"\label{tab:step}")
    a.append(r"\begin{tabular}{rrr}\toprule $k$ & Controller (ms) & Generator (ms) \\ \midrule")
    for k in KS:
        r = runs[k]
        a.append(f"{k} & {r['controller_step_avg']} & {r['generator_step_avg']} \\\\")
    a.append(r"\bottomrule\end{tabular}\end{table}")

    a.append(r"\begin{table}[H]\centering")
    a.append(r"\caption{Expected duration for 200,000 controller steps and 500,000 generator steps.}")
    a.append(r"\label{tab:duration}")
    a.append(r"\begin{tabular}{rrrr}\toprule $k$ & Controller & Generator & Total \\ \midrule")
    for k in KS:
        r = runs[k]
        c = r["controller_step_avg"] * 200000 / 3.6e6
        g = r["generator_step_avg"] * 500000 / 3.6e6
        a.append(f"{k} & {c:.1f} h & {g:.1f} h & {c + g:.1f} h ({(c + g) / 24:.2f} d) \\\\")
    a.append(r"\bottomrule\end{tabular}\end{table}")

    a.append(r"\section{Parameter Counts}")
    a.append(r"\begin{table}[H]\centering\resizebox{\textwidth}{!}{%")
    a.append(r"\begin{tabular}{rrrrrrrrrr}\toprule $k$ & Generator (excl. CC) & Controller & Critic & Labeler & Anti-Lab. & Gen. Opt. & Crit. Opt. & Lab. Opt. & Anti-Lab. Opt. \\ \midrule")
    fields = ["gen_state_dict", "crit_state_dict", "labeler_state_dict", "antilabeler_state_dict",
              "gen_optimizer_state_dict", "crit_optimizer_state_dict",
              "labeler_optimizer_state_dict", "antilabeler_optimizer_state_dict"]
    for k in KS:
        p = runs[k]["params"]
        a.append(f"{k} & {runs[k]['gen_only_params'] / 1e6:.2f}M & {runs[k]['controller_params'] / 1e6:.2f}M & "
                 + " & ".join(f"{p[f] / 1e6:.2f}M" for f in fields[1:]) + r" \\")
    a.append(r"\bottomrule\end{tabular}}")
    a.append(r"\caption{Parameter counts for each component and optimiser, from \texttt{params-size.json}. The generator column excludes the causal controller, which is a submodule of the stored \texttt{gen\_state\_dict} and is shown separately.}")
    a.append(r"\end{table}")

    a.append(r"\section{GPU Memory}")
    a.append(r"\begin{table}[H]\centering")
    a.append(r"\caption{Peak GPU memory (MiB) by training phase, from the monitoring CSV using the logged phase windows.}")
    a.append(r"\label{tab:gpu}")
    a.append(r"\begin{tabular}{rrr}\toprule $k$ & Controller (MiB) & Generator (MiB) \\ \midrule")
    for k in KS:
        g = runs[k]["gpu"]
        a.append(f"{k} & {g['controller_max']:,} & {g['generator_max']:,} \\\\")
    a.append(r"\bottomrule\end{tabular}\end{table}")

    a.append(r"\section{Fits in $k$}\label{sec:fits}")
    a.append(r"Let $k$ denote the number of TFs per gene. Ordinary least-squares fits over all 15 runs give:")
    a.append(r"\begin{align}")
    a.append(r"\widehat{P}_{\mathrm{gen}}(k) &= " + tex_poly(fits["param1"]["c"], "k", "parameters")
             + rf",\quad R^2={fits['param1']['r2']:.4f}, \\")
    a.append(r"\widehat{M}_{\mathrm{gen}}(k) &= " + tex_poly(fits["mem1"]["c"], "k", "MiB")
             + rf",\quad R^2={fits['mem1']['r2']:.4f}, \\")
    a.append(r"\widehat{t}_{\mathrm{gen}}(k) &= " + tex_poly(fits["time1"]["c"], "k", "ms/step")
             + rf",\quad R^2={fits['time1']['r2']:.4f}.")
    a.append(r"\end{align}")
    a.append(r"Quadratic fits, which capture the curvature visible in the data, are:")
    a.append(r"\begin{align}")
    a.append(r"\widehat{P}_{\mathrm{gen},2}(k) &= " + tex_poly(fits["param2"]["c"], "k", "parameters")
             + rf",\quad R^2={fits['param2']['r2']:.4f}, \\")
    a.append(r"\widehat{M}_{\mathrm{gen},2}(k) &= " + tex_poly(fits["mem2"]["c"], "k", "MiB")
             + rf",\quad R^2={fits['mem2']['r2']:.4f}, \\")
    a.append(r"\widehat{t}_{\mathrm{gen},2}(k) &= " + tex_poly(fits["time2"]["c"], "k", "ms/step")
             + rf",\quad R^2={fits['time2']['r2']:.4f}.")
    a.append(r"\end{align}")

    a.append(r"\begin{table}[H]\centering")
    a.append(r"\caption{Goodness of fit in $k$ (in-sample, all 15 runs).}")
    a.append(r"\label{tab:fits}")
    a.append(r"\begin{tabular}{lrrrr}\toprule Quantity & Linear $R^2$ & Quadratic $R^2$ & Linear RMSE & Quadratic RMSE \\ \midrule")
    rows = [("Generator parameters", "param"), ("Generator peak memory (MiB)", "mem"),
            ("Generator step time (ms)", "time")]
    for label, key in rows:
        a.append(f"{label} & {fits[key + '1']['r2']:.4f} & {fits[key + '2']['r2']:.4f} & "
                 f"{fits[key + '1']['rmse']:,.2f} & {fits[key + '2']['rmse']:,.2f} \\\\")
    a.append(r"\bottomrule\end{tabular}\end{table}")

    a.append(r"\section{Graphs}")
    a.append(figure_env("gen_params_vs_k.pdf", "Generator parameter count versus $k$, with linear and quadratic fits.", "fig:params"))
    a.append(figure_env("gpu_mem_vs_k.pdf", "Peak GPU memory versus $k$, with the generator linear and quadratic fits.", "fig:memory"))
    a.append(figure_env("gen_step_vs_k.pdf", "Causal-generator average step time versus $k$, with linear and quadratic fits.", "fig:time"))

    ks2 = np.array(KS[1:], float)
    sens = {}
    for key, getter in [("param", lambda k: runs[k]["gen_only_params"]),
                        ("mem", lambda k: runs[k]["gpu"]["generator_max"]),
                        ("time", lambda k: runs[k]["generator_step_avg"])]:
        y = [getter(k) for k in KS[1:]]
        sens[key] = (fit(ks2, y, 1)["r2"], fit(ks2, y, 2)["r2"])

    a.append(r"\section{Notes}")
    a.append(r"\begin{itemize}")
    a.append(rf"\item Sensitivity to the $k=1$ point (which has different TF and gene counts): excluding it, the linear/quadratic $R^2$ are "
             rf"{sens['param'][0]:.4f}/{sens['param'][1]:.4f} for parameters, "
             rf"{sens['mem'][0]:.4f}/{sens['mem'][1]:.4f} for peak memory, and "
             rf"{sens['time'][0]:.4f}/{sens['time'][1]:.4f} for step time.")
    a.append(rf"\item Runs with an out-of-memory error: {n_oom} of 15.")
    c1 = fits["param1"]["c"]
    k0 = -c1[1] / c1[0]
    a.append(rf"\item The linear parameter fit crosses zero at $k\approx{k0:.1f}$, so it predicts a negative generator parameter count below that value; the quadratic fit reproduces the measured counts and should be used at small $k$.")
    a.append(r"\item The TF/target ratio is essentially constant (6.92\% at $k=1$; 7.03\% for $k\ge2$), because the target count is fixed. A ratio-based figure would therefore carry no information and is omitted; the scaling is attributed to $k$.")
    a.append(rf"\item The controller peak memory is flat across $k$ (from {min(runs[k]['gpu']['controller_max'] for k in KS):,} to {max(runs[k]['gpu']['controller_max'] for k in KS):,} MiB), as expected because the controller does not depend on $k$. This supports attributing the generator's growth to $k$ rather than to a phase-window artefact.")
    a.append(rf"\item GPU capacity is {total:,} MiB; the largest generator peak is {max(runs[k]['gpu']['generator_max'] for k in KS):,} MiB at $k={max(KS, key=lambda k: runs[k]['gpu']['generator_max'])}$.")
    a.append(r"\end{itemize}")
    a.append(r"\end{document}")
    with open(os.path.join(out, "report.tex"), "w") as f:
        f.write("\n".join(a))


def main():
    runs = load_runs()
    ks = np.array(KS, float)
    params = [runs[k]["gen_only_params"] for k in KS]
    memory = [runs[k]["gpu"]["generator_max"] for k in KS]
    times = [runs[k]["generator_step_avg"] for k in KS]
    fits = {}
    for key, y in [("param", params), ("mem", memory), ("time", times)]:
        fits[key + "1"] = fit(ks, y, 1)
        fits[key + "2"] = fit(ks, y, 2)

    for k in KS:
        r = runs[k]
        print(f"k={k:2d} tfs={r['tfs']} genes={r['genes']} imposed={r['imposed_edges']} "
              f"ctrl={r['controller_step_avg']}ms gen={r['generator_step_avg']}ms "
              f"gpu_c={r['gpu']['controller_max']} gpu_g={r['gpu']['generator_max']} "
              f"(overall={r['gpu']['overall_max']}) oom={r['oom']} "
              f"windows={r['ctrl_window']} {r['gen_window']}")
    for key in ["param", "mem", "time"]:
        print(key, "linear R2=%.6f quad R2=%.6f" % (fits[key + "1"]["r2"], fits[key + "2"]["r2"]))

    os.makedirs(OUT, exist_ok=True)
    make_graphs(runs, fits, OUT)
    write_latex(runs, fits, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
