#!/usr/bin/env python3
"""Export and plot Optuna intermediate values (RF AUROC) vs training step for gan_hyperopt."""

import sys
from pathlib import Path

# Add parent folder to pythonpath
sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())
from src.loggers import setup_logger

logger = setup_logger("common_scatterfig")

import csv
import os

import matplotlib.pyplot as plt
import optuna

STUDY_NAME = "gan_hyperopt"
ROOT = Path("publication/paper-arabidopsis")
ENV_FILE = ROOT / "postgres" / ".env"
CSV_PATH = ROOT / "gan_hyperopt_intermediate.csv"
PLOT_PATH = ROOT / "gan_hyperopt_intermediate.pdf"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    load_env(ENV_FILE)
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['POSTGRES_DB']}"
    )
    # load_study only issues SELECTs when reading; no heartbeat is started.
    study = optuna.load_study(study_name=STUDY_NAME, storage=url)

    rows = []
    series = []  # (trial_number, state, steps, values)
    for trial in study.trials:
        iv = trial.intermediate_values
        if not iv:
            continue
        steps = sorted(iv)
        vals = [iv[s] for s in steps]
        series.append((trial.number, str(trial.state), steps, vals))
        for s, v in zip(steps, vals):
            rows.append((trial.number, str(trial.state), s, v))

    with CSV_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "state", "step", "rf_auroc"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} intermediate values for {len(series)} trials -> {CSV_PATH}")

    if not series:
        print("No intermediate values found; nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.turbo
    norm = plt.Normalize(vmin=0, vmax=max(n for n, _, _, _ in series))
    complete = optuna.trial.TrialState.COMPLETE
    for number, state, steps, vals in series:
        is_complete = state == str(complete)
        color = cmap(norm(number))
        ax.plot(
            steps,
            vals,
            color=color,
            linestyle="-" if is_complete else "--",
            alpha=0.8 if is_complete else 0.5,
            linewidth=1.2,
        )
        ax.scatter(
            steps,
            vals,
            color=color,
            s=12,
            alpha=0.9 if is_complete else 0.6,
            zorder=3,
        )

    def _k_fmt(value, _pos):
        if value == 0:
            return "0"
        return f"{int(value) // 1000}k"

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Trial Number")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("RF AUROC (intermediate)")
    ax.set_title(f"Optuna Intermediate Values: {STUDY_NAME}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_k_fmt))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=600)
    print(f"Saved plot -> {PLOT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
