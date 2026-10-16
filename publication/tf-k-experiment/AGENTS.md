# Agents: tf-k-experiment report

Single-variable experiment: hold the HVG / target / TF counts roughly fixed and sweep the
top-`k` GRN parameter (`k` = number of top-ranked TFs retained per target gene). Slurm
task id `i` -> `k=i` (1..15), with `results/<k>/` as the output directory.

## Artifacts
- `run.sh` — Slurm array driver (@ array = 1..15).
- `causal_gan.cfg` — has to be re-interpolated for each `k`.
- `generate_report.py` — single self-contained Python script using the standard stack:
  parses logs, measures GPU memory, fits lines/quadratics in `k`, writes figures + `report.tex`.
- `report/` — generated outputs: `report.tex`, `report.pdf`, four per-measurement PDF plots.

## Recipe (how to reproduce / regenerate)
```bash
cd publication/tf-k-experiment
python3 generate_report.py
cd report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex
```
Dependencies: Python 3.11 with numpy + matplotlib, plus pdflatex (TinyTeX).

## When changing the script
- `k` is the independent variable in figures, tables, fits and equations. Imposed edges are
  reported only as a descending column (density is derived from them).
- GPU memory: log timestamps identify the controller-vs-generator window; nvidia-smi values
  are in MiB, with the 80 GB (81,920 MiB) capacity line overlaid. Do not fabricate the
  offset: the controller plateau is ~6.1 GB.
- Figures are emitted one plot per PDF with `[H]` floating. ruff/pyright CI rules do not
  apply to these standalone scripts, but keep them working end-to-end if touched.
- Fit scope: linear fits and quadratic fits are both reported in LaTeX (`\hat P, \hat M, \hat t`).

## Note
The controller's resident parameters (≈ 3.77 M, 2-element mismatch vs. the model formula)
are shown separately in the parameter table; do not double-count them into the generator column.