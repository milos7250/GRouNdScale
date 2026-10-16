# Agents: tf-num-experiment report

Single-variable experiment: hold HVG / total gene count roughly constant at ≈ 3,000 and
sweep the number of top TFs retained per gene (`num-tfs`, the GRN parameter). Slurm
task id `i` -> `SIZE = 200 * i` TFs requested. 14 runs (200..2,800 in steps of 200);
run 3,000 is not feasible (3,000 TFs would leave no target genes).

## Artifacts
- `run.sh`, `limit-tfs.py` — Slurm driver and top-TF filtering.
- `causal_gan.cfg` — interpolated per run after GRN creation (`TFs = ` re-injected).
- `generate_report.py` — self-contained Python: parses logs, measures GPU memory,
  fits in **target-gene count**, writes figures + `report.tex`.
- `report/` — outputs: `report.tex`, `report.pdf` (7 pp), three `gen_*_vs_targets.pdf` plots.

## Recipe (how to reproduce / regenerate)
```bash
cd publication/tf-num-experiment
python3 generate_report.py          # writes to report/
cd report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex
```
Needs Python (numpy + matplotlib) and pdflatex.

## When changing the script
- The independent variable asked for by this user is **target genes** (3000 minus realised TFs).
  Requested TFs are 200, 400, ..., 2,800; realized counts match up to 1,200 then diverge.
  This is documented as a caveat in the report.
- Tables and plots are ordered by target genes ascending; controller column stays as-is.
- Generator parameters = `gen_state_dict` minus the controller's state size (~3.77 M).
  Controller parameters are already listed separately in the table.
- Fits are linear only (no quadratic); the theoretical memory model does not support curvature.
  Two fits are reported: all runs and a "fit subset" of the 9 runs with <100 genes discarded.
  The second test is whether discarding drives the scaling; see the report's fit table.
- GPU figures show only the generator phase (controller is excluded).
- No OOM in this experiment; GPU peaks 7,261–8,959 MiB.

## Note
Files under results/100 .. results/1900 and `limit-tfs.py`, `results-old/` were already
modified/deleted in the tree before the report was generated — do not restore them.