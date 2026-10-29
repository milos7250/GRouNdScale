#!/usr/bin/env python3
"""Collect the summary.json of every arabidopsis-ablation variant into two CSVs, see README.md.

- results.csv           one row per (variant, phase): status, step time, peaks, settings
- results-summary.csv   one row per (comparison, phase): speed-up and memory ratios of the comparisons of the README,
                        with the extrapolated training time of both variants for the paper's step counts

Usage:
    python collect.py
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = (
    "a-upstream",
    "a2-upstream-newenv",
    "dense-eager-fp32",
    "dense-eager-tf32",
    "dense-compiled-fp32",
    "dense-compiled-tf32",
    "sparse-eager-fp32",
    "sparse-eager-tf32",
    "sparse-compiled-fp32",
    "sparse-compiled-tf32",
)
PHASES = ("controller", "generator")
PAPER_STEPS = {"controller": 200_000, "generator": 500_000}
MEASUREMENTS = ("average_step_time_ms", "peak_torch_allocated_mib", "peak_torch_reserved_mib", "peak_nvidia_smi_mib")
STEADY_STATE = ("peak_torch_allocated_mib", "peak_torch_reserved_mib")

# (what the comparison isolates, baseline, variant)
COMPARISONS = [
    ("environment", "a-upstream", "a2-upstream-newenv"),
    ("code update", "a2-upstream-newenv", "dense-eager-fp32"),
    ("environment + code update", "a-upstream", "dense-eager-fp32"),
    *[
        (what, f"{base}", f"{variant}")
        for what, pairs in {
            "sparse layers": [
                (f"dense-{c}-{m}", f"sparse-{c}-{m}") for c in ("eager", "compiled") for m in ("fp32", "tf32")
            ],
            "compilation": [
                (f"{l}-eager-{m}", f"{l}-compiled-{m}") for l in ("dense", "sparse") for m in ("fp32", "tf32")
            ],
            "TF32": [(f"{l}-{c}-fp32", f"{l}-{c}-tf32") for l in ("dense", "sparse") for c in ("eager", "compiled")],
        }.items()
        for base, variant in pairs
    ],
    ("paper: environment + code update + TF32", "a-upstream", "dense-eager-tf32"),
    ("paper: sparse layers", "dense-eager-tf32", "sparse-eager-tf32"),
    ("paper: compilation", "sparse-eager-tf32", "sparse-compiled-tf32"),
    ("all", "a-upstream", "sparse-compiled-tf32"),
]


def load(variant: str) -> dict | None:
    path = HERE / "results" / variant / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def phase_row(variant: str, summary: dict | None, phase: str) -> dict:
    row: dict = {"variant": variant, "phase": phase}
    if summary is None:
        return row | {"status": "missing"}
    settings = summary["settings"]
    data = summary.get(phase) or {}
    row |= {
        "status": summary["status"],
        "error": summary["error"],
        "complete": data.get("complete"),
        "steps_recorded": data.get("steps_recorded"),
        **{name: data.get(name) for name in MEASUREMENTS},
        **{f"steady_{name}": data.get("steady_state", {}).get(name) for name in STEADY_STATE},
        "python": settings.get("python"),
        "torch": settings.get("torch"),
        "generator_layers": settings.get("generator_layers", "dense" if variant.startswith("a") else None),
        "float32_matmul_precision": settings.get("float32_matmul_precision"),
        "gpu": settings.get("gpu"),
        "host": settings.get("host"),
        "git_commit": settings["git"]["commit"],
        "src_modified": settings["git"]["src_modified"],
    }
    if row["average_step_time_ms"] is not None and row["complete"]:
        row["extrapolated_hours"] = round(row["average_step_time_ms"] * PAPER_STEPS[phase] / 3.6e6, 2)
    return row


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if not numerator or not denominator:
        return None
    return round(numerator / denominator, 3)


def main() -> None:
    rows = {(v, p): phase_row(v, load(v), p) for v in VARIANTS for p in PHASES}
    with (HERE / "results.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(key for row in rows.values() for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows.values())

    summary_rows = []
    for what, base, variant in COMPARISONS:
        for phase in PHASES:
            b, v = rows[base, phase], rows[variant, phase]
            usable = b.get("status") == "ok" and v.get("status") == "ok"
            summary_rows.append({
                "comparison": what,
                "baseline": base,
                "variant": variant,
                "phase": phase,
                "baseline_status": b.get("status"),
                "variant_status": v.get("status"),
                "baseline_step_ms": b.get("average_step_time_ms"),
                "variant_step_ms": v.get("average_step_time_ms"),
                "speed_up": ratio(b.get("average_step_time_ms"), v.get("average_step_time_ms")) if usable else None,
                "torch_allocated_ratio": (
                    ratio(v.get("peak_torch_allocated_mib"), b.get("peak_torch_allocated_mib")) if usable else None
                ),
                "steady_torch_allocated_ratio": (
                    ratio(v.get("steady_peak_torch_allocated_mib"), b.get("steady_peak_torch_allocated_mib"))
                    if usable
                    else None
                ),
                "nvidia_smi_ratio": ratio(v.get("peak_nvidia_smi_mib"), b.get("peak_nvidia_smi_mib"))
                if usable
                else None,
                "baseline_extrapolated_hours": b.get("extrapolated_hours"),
                "variant_extrapolated_hours": v.get("extrapolated_hours"),
            })
    with (HERE / "results-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {HERE / 'results.csv'} and {HERE / 'results-summary.csv'}")


if __name__ == "__main__":
    main()
