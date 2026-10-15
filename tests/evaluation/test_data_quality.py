import json
from pathlib import Path

import numpy as np
import pytest

from tests.resources.constants import TRAIN_FILE

from .. import pytestmark  # pyright: ignore[reportUnusedImport]  # noqa: F401


class TestAggregateEvaluations:
    def test_mean_and_sample_std(self, tmp_path: Path) -> None:
        from evaluation.data_quality import aggregate_evaluations

        runs = [{"a": 1.0, "b": 10.0}, {"a": 2.0, "b": 10.0}, {"a": 4.0, "b": 10.0}]
        aggregate_evaluations(runs, tmp_path / "summary.json")

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["n_runs"] == 3
        assert summary["metrics"]["a"]["mean"] == pytest.approx(7 / 3)
        assert summary["metrics"]["a"]["std"] == pytest.approx(np.std([1.0, 2.0, 4.0], ddof=1))
        assert summary["metrics"]["b"]["mean"] == pytest.approx(10.0)
        assert summary["metrics"]["b"]["std"] == pytest.approx(0.0)

    def test_summary_keeps_run_order(self, tmp_path: Path) -> None:
        from evaluation.data_quality import aggregate_evaluations

        runs = [{"m": 3.0}, {"m": 1.0}, {"m": 2.0}]
        aggregate_evaluations(runs, tmp_path / "summary.json")

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["metrics"]["m"]["values"] == [3.0, 1.0, 2.0]

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        from evaluation.data_quality import aggregate_evaluations

        output_path = tmp_path / "nested" / "dir" / "summary.json"
        aggregate_evaluations([{"m": 1.0}, {"m": 2.0}], output_path)

        assert output_path.exists()

    def test_empty_metrics(self, tmp_path: Path) -> None:
        from evaluation.data_quality import aggregate_evaluations

        aggregate_evaluations([{}, {}], tmp_path / "summary.json")

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary == {"n_runs": 2, "metrics": {}}


class TestEvaluateReturnsResults:
    def test_returns_and_writes_same_results(self, tmp_path: Path) -> None:
        from custom_parser import get_configparser
        from evaluation.data_quality import evaluate

        cfg = get_configparser()
        cfg.read_dict({
            "EXPERIMENT": {"output directory": str(tmp_path)},
            "Data": {"test": str(TRAIN_FILE)},
            "Evaluation": {
                "simulated data path": str(TRAIN_FILE),
                "plot umap": "False",
                "compute euclidean distance": "True",
                "compute cosine distance": "False",
                "compute rf auroc": "False",
                "compute MMD": "False",
                "compute miLISI": "False",
            },
        })

        results = evaluate(cfg)

        # Euclidean and cosine distances are computed together, so enabling one returns both
        assert set(results) == {
            "euclidean_distance_real_vs_fake",
            "euclidean_distance_control",
            "cosine_distance_real_vs_fake",
            "cosine_distance_control",
        }
        # Real and simulated data are the same file, so the real-vs-fake distance is zero
        assert results["euclidean_distance_real_vs_fake"] == pytest.approx(0.0, abs=1e-6)
        written = json.loads((tmp_path / "evaluation_results.json").read_text())
        assert written == pytest.approx(results)
