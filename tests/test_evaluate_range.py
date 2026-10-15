import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner

from tests.resources.constants import CAUSAL_GRAPH_FILE, TRAIN_FILE, VALID_FILE

from . import pytestmark  # pyright: ignore[reportUnusedImport]  # noqa: F401

CONFIG_PATH = Path("tests/resources/configs/gan_train.cfg")


@pytest.fixture
def pytest_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output_dir = tmp_path / "results/GAN"
    monkeypatch.setenv("PYTEST_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("PYTEST_TRAIN_FILE", str(TRAIN_FILE))
    monkeypatch.setenv("PYTEST_VALID_FILE", str(VALID_FILE))
    monkeypatch.setenv("PYTEST_TEST_FILE", str(VALID_FILE))
    monkeypatch.setenv("PYTEST_CAUSAL_GRAPH_FILE", str(CAUSAL_GRAPH_FILE))
    return output_dir


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    Replaces the GAN factory and the per-run evaluation with fakes. Each fake evaluation returns the run index as the
    value of metric "m", so the summary can be checked exactly.
    """
    import factory
    from evaluation import data_quality

    generated: list[dict[str, Any]] = []
    evaluated_outputs: list[Path] = []

    def fake_generate_h5ad(cells_no: int, save_path: Path, *, reference_dataset: Path, checkpoint: Any) -> None:
        generated.append({"cells_no": cells_no, "save_path": save_path, "reference_dataset": reference_dataset})
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.touch()

    fake_gan = MagicMock()
    fake_gan.generate_h5ad.side_effect = fake_generate_h5ad
    fake_factory = MagicMock()
    fake_factory.get_gan.return_value = fake_gan
    monkeypatch.setattr(factory, "get_factory", MagicMock(return_value=fake_factory))

    def fake_evaluate(cfg: Any) -> dict[str, float]:
        out = Path(cfg.get("EXPERIMENT", "output directory"))
        evaluated_outputs.append(out)
        return {"m": float(len(evaluated_outputs))}

    monkeypatch.setattr(data_quality, "evaluate", fake_evaluate)

    return {"generated": generated, "evaluated_outputs": evaluated_outputs}


class TestEvaluateRangeLoop:
    def test_runs_generation_and_evaluation_per_iteration(self, pytest_env: Path, fakes: dict[str, Any]) -> None:
        from main import main

        main(CONFIG_PATH, evaluate_range=3)

        base = pytest_env
        expected_paths = [base / "evaluate_range" / f"run_{i:03d}" / "simulated.h5ad" for i in (1, 2, 3)]
        assert [g["save_path"] for g in fakes["generated"]] == expected_paths
        assert all(g["cells_no"] == 10000 for g in fakes["generated"])
        assert fakes["evaluated_outputs"] == [p.parent for p in expected_paths]

    def test_summary_written_with_mean_and_std(self, pytest_env: Path, fakes: dict[str, Any]) -> None:
        from main import main

        main(CONFIG_PATH, evaluate_range=3)

        summary = json.loads((pytest_env / "evaluation_range_summary.json").read_text())
        assert summary["n_runs"] == 3
        assert summary["metrics"]["m"]["values"] == [1.0, 2.0, 3.0]
        assert summary["metrics"]["m"]["mean"] == pytest.approx(2.0)
        assert summary["metrics"]["m"]["std"] == pytest.approx(1.0)

    def test_run_configs_have_overridden_paths(self, pytest_env: Path, fakes: dict[str, Any]) -> None:
        from custom_parser import get_configparser
        from main import main

        main(CONFIG_PATH, evaluate_range=2)

        for run in (1, 2):
            run_dir = pytest_env / "evaluate_range" / f"run_{run:03d}"
            run_cfg = get_configparser()
            run_cfg.read(run_dir / CONFIG_PATH.name)
            generation_path = run_dir / "simulated.h5ad"

            assert Path(run_cfg.get("EXPERIMENT", "output directory")) == run_dir
            assert Path(run_cfg.get("Generation", "generation path")) == generation_path
            assert Path(run_cfg.get("Evaluation", "simulated data path")) == generation_path

    def test_overrides_explicit_generation_and_evaluation_paths(
        self, pytest_env: Path, fakes: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from custom_parser import get_configparser
        from main import main

        # Set the paths explicitly to a location the loop must not use
        explicit = tmp_path / "explicit.h5ad"
        cfg = get_configparser()
        cfg.read(CONFIG_PATH)
        cfg.set("Generation", "generation path", str(explicit))
        cfg.add_section("Evaluation")
        cfg.set("Evaluation", "simulated data path", str(explicit))
        config = tmp_path / "explicit.cfg"
        with config.open("w") as f:
            cfg.write(f)

        main(config, evaluate_range=2)

        for g in fakes["generated"]:
            assert g["save_path"] != explicit
            assert g["save_path"].name == "simulated.h5ad"

    def test_original_config_output_directory_unchanged(self, pytest_env: Path, fakes: dict[str, Any]) -> None:
        from custom_parser import get_configparser
        from main import main

        main(CONFIG_PATH, evaluate_range=2)

        saved = get_configparser()
        saved.read(pytest_env / CONFIG_PATH.name)
        assert Path(saved.get("EXPERIMENT", "output directory")) == pytest_env

    def test_not_run_when_option_absent(self, pytest_env: Path, fakes: dict[str, Any]) -> None:
        from main import main

        main(CONFIG_PATH)

        assert fakes["generated"] == []
        assert fakes["evaluated_outputs"] == []
        assert not (pytest_env / "evaluation_range_summary.json").exists()


class TestEvaluateRangeOption:
    @pytest.fixture
    def command(self) -> click.Command:
        from custom_parser import click_options

        @click.command()
        @click_options
        def _cmd(
            config: Path,
            preprocess: bool,
            create_grn: bool,
            train: bool,
            optimize_hyperparameters: bool,
            generate: bool,
            evaluate: bool,
            evaluate_range: int | None,
            benchmark_grn: bool,
            perturb: bool,
        ) -> None:
            click.echo(f"evaluate_range={evaluate_range}")

        return _cmd

    @pytest.fixture
    def config_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "config.cfg"
        path.write_text("[EXPERIMENT]\n")
        return path

    @pytest.mark.parametrize("value", ["2", "3", "10"])
    def test_accepts_integers_of_at_least_two(self, command: click.Command, config_file: Path, value: str) -> None:
        result = CliRunner().invoke(command, ["--config", str(config_file), "--evaluate-range", value])

        assert result.exit_code == 0
        assert result.output.strip() == f"evaluate_range={value}"

    @pytest.mark.parametrize("value", ["1", "0", "-3", "2.5", "abc"])
    def test_rejects_invalid_values(self, command: click.Command, config_file: Path, value: str) -> None:
        result = CliRunner().invoke(command, ["--config", str(config_file), "--evaluate-range", value])

        assert result.exit_code == 2

    def test_defaults_to_none(self, command: click.Command, config_file: Path) -> None:
        result = CliRunner().invoke(command, ["--config", str(config_file)])

        assert result.exit_code == 0
        assert result.output.strip() == "evaluate_range=None"
