#!/usr/bin/env python3
"""Prints one line per compile-overhead run: status, where it failed, the generator peaks and the stack limit."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

for summary_path in sorted(
    HERE.glob("results/*/*/compiled/summary.json"), key=lambda p: (p.parts[-4], int(p.parts[-3][1:]))
):
    run = summary_path.parent
    summary = json.loads(summary_path.read_text())
    phases = json.loads((run / "trace.json").read_text())["phases"]
    ballast = run / "ballast.json"
    free = round(json.loads(ballast.read_text())["free_mib"]) if ballast.exists() else "-"
    generator = summary.get("generator") or {}
    steps = phases.get("generator", {}).get("steps", [])
    failed = next((step["step"] for step in steps if step["failed"]), None)
    stacks = [step["after"].get("stack_size_bytes") for step in steps]
    smi = [step["peak_nvidia_smi_mib"] for step in steps if step["peak_nvidia_smi_mib"] is not None]
    print(
        f"{summary_path.parts[-4]:28} {summary_path.parts[-3]:7} {summary['status']:5} free={free!s:5}"
        f" failed_step={failed!s:4} res={generator.get('peak_torch_reserved_mib')!s:8}"
        f" smi_peak={max(smi) if smi else '-'!s:6} smi_last={smi[-1] if smi else '-'!s:6}"
        f" stack_max={max((s for s in stacks if s), default='-')!s:6}"
        f" step_ms={generator.get('average_step_time_ms')}"
    )
    if summary["error"] and summary["status"] != "ok":
        print("    ", summary["error"].splitlines()[0][:160])
