#!/usr/bin/env python3
"""Peak memory and step time of the compiled causal generator only, vs G (target-gene-num) and vs k (tf-k).

Reuses the plotting code of plot-results.py (same styling, metrics and analytical estimate) but keeps only the
`compiled` runs, without the un-compiled and original (dense) series. For each experiment this writes, as PNG and PDF,
plots/memory-vs-<x>-compiled and plots/time-vs-<x>-compiled.

Without the dense series every compiled run fits far below the GPU memory, so the GPU memory line is off by default.

Usage::

    python plot-compiled.py                 # figures in publication/plots
    python plot-compiled.py --log-y --steady
    python plot-compiled.py --gpu-limit-gib 80  # with the GPU memory line
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = ("target-gene-num", "tf-k")
MODE = "compiled"

spec = importlib.util.spec_from_file_location("plot_results", HERE / "plot-results.py")
assert spec is not None and spec.loader is not None
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

_save = pr.save
FAILED_NOTE = " and is drawn beyond the GPU memory, where runs failed."
drop_failed_note = True  # set per experiment: the note stays when some run ran out of memory


def save(figure, path: Path) -> None:
    """Saves under a -compiled name, without the texts that only make sense with several series or failed runs."""
    for axes in figure.axes:
        for text in list(axes.texts):
            if text.get_text() == pr.MODES[MODE][0]:  # end-of-series label, redundant with a single series
                text.remove()
    if drop_failed_note:
        for text in figure.texts:
            text.set_text(text.get_text().replace(FAILED_NOTE, "."))
    _save(figure, path.with_name(f"{path.name}-{MODE}"))


pr.save = save


def main(argv: list[str] | None = None) -> int:
    global drop_failed_note
    args = pr.parse_args(argv)
    experiments = pr.load_summaries(HERE)
    theory = {} if args.no_theory else {MODE: pr.load_theory()[MODE]}
    limit = args.gpu_limit_gib or 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    for name in EXPERIMENTS:
        runs = [run for run in experiments.get(name, []) if run["mode"] == MODE]
        if not runs:
            print(f"no {MODE} runs of {name} found below {HERE}, skipped", file=sys.stderr)
            continue
        x = pr.varying_hyperparameter(name, runs)
        if x is None:
            continue
        drop_failed_note = all(run["generator"]["complete"] for run in runs)
        pr.print_table(runs, x, theory)
        pr.plot_memory(runs, x, args.out_dir, args.log_y, args.steady, limit or None, theory)
        pr.plot_time(runs, x, args.out_dir, args.log_y)
        plotted += 1
    return 0 if plotted else 1


if __name__ == "__main__":
    sys.exit(main())
