import pytest


def get_rates(decay_type: str) -> list[float]:
    import torch

    from training.helpers import set_learning_rate_scheduler

    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = set_learning_rate_scheduler(
        optimizer,
        alpha_0=1.0,
        alpha_final=0.1,
        max_steps=100,
        decay_type=decay_type,
        warmup_percent=0.1,
        holding_percent=0.2,
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    return rates


@pytest.mark.parametrize("decay_type", ["linear", "cosine", "exponential"])
def test_learning_rate_scheduler_phases(decay_type: str) -> None:
    rates = get_rates(decay_type)

    assert rates[0] == pytest.approx(0.01)
    assert rates[10] == pytest.approx(1.0)
    assert rates[20] == pytest.approx(1.0)
    assert rates[21] < 1.0
    assert rates[-1] == pytest.approx(0.1)
    assert all(left >= right for left, right in zip(rates[20:], rates[21:]))


def test_learning_rate_scheduler_rejects_invalid_configuration() -> None:
    import torch

    from training.helpers import set_learning_rate_scheduler

    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)

    with pytest.raises(ValueError, match="Unknown learning rate decay type"):
        set_learning_rate_scheduler(optimizer, 1.0, 0.1, 10, decay_type="quadratic")
    with pytest.raises(ValueError, match="must be non-negative"):
        set_learning_rate_scheduler(optimizer, 1.0, 0.1, 10, warmup_percent=-0.1)
    with pytest.raises(ValueError, match="at most 1"):
        set_learning_rate_scheduler(optimizer, 1.0, 0.1, 10, holding_percent=1.1)
    with pytest.raises(ValueError, match="warmup at most holding"):
        set_learning_rate_scheduler(optimizer, 1.0, 0.1, 10, holding_percent=0.1, warmup_percent=0.9)


@pytest.mark.parametrize("warmup_percent", [0.0, 0.05])
def test_holding_percent_is_independent_of_warmup(warmup_percent: float) -> None:
    import torch

    from training.helpers import set_learning_rate_scheduler

    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = set_learning_rate_scheduler(
        optimizer,
        alpha_0=1.0,
        alpha_final=0.1,
        max_steps=100,
        decay_type="linear",
        warmup_percent=warmup_percent,
        holding_percent=0.60,
    )

    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates[59] == pytest.approx(1.0)
    assert rates[60] == pytest.approx(1.0)
    assert rates[61] < 1.0


def test_holding_percent_one_disables_decay() -> None:
    import torch

    from training.helpers import set_learning_rate_scheduler

    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = set_learning_rate_scheduler(
        optimizer,
        alpha_0=1.0,
        alpha_final=0.1,
        max_steps=100,
        warmup_percent=0.05,
        holding_percent=1.0,
    )

    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])

    assert rates[0] == pytest.approx(0.01)
    assert all(rate == pytest.approx(1.0) for rate in rates[5:])


def make_trainer(error: Exception):
    """A trainer whose train method, decorated like GANTrainer.train, raises `error`."""
    from unittest.mock import MagicMock

    from training.helpers import hint_on_out_of_memory

    class Trainer:
        logger = MagicMock()

        @hint_on_out_of_memory
        def train(self) -> None:
            raise error

    return Trainer()


def raised_from(error: Exception) -> Exception:
    """A RuntimeError raised from `error`, as torch.compile wraps the errors of compilation."""
    try:
        raise RuntimeError("compilation failed") from error
    except RuntimeError as wrapped:
        return wrapped


@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("error_type", ["oom", "no_valid_choices"])
def test_out_of_memory_hint_is_logged(monkeypatch: pytest.MonkeyPatch, error_type: str, wrap: bool) -> None:
    import torch
    from torch._inductor.select_algorithm import NoValidChoicesError

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", raising=False)
    error: Exception = torch.OutOfMemoryError("CUDA out of memory") if error_type == "oom" else NoValidChoicesError()
    raised = raised_from(error) if wrap else error
    trainer = make_trainer(raised)

    with pytest.raises(type(raised)):
        trainer.train()

    message = trainer.logger.error.call_args.args[0]
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in message
    assert "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN" in message


def test_out_of_memory_hint_skips_settings_already_made(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
    monkeypatch.delenv("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", raising=False)
    trainer = make_trainer(torch.OutOfMemoryError("CUDA out of memory"))

    with pytest.raises(torch.OutOfMemoryError):
        trainer.train()

    message = trainer.logger.error.call_args.args[0]
    assert "PYTORCH_CUDA_ALLOC_CONF" not in message
    assert "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN" in message

    monkeypatch.setenv("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN")
    trainer.logger.reset_mock()
    with pytest.raises(torch.OutOfMemoryError):
        trainer.train()
    trainer.logger.error.assert_not_called()


def test_other_errors_are_not_reported_as_out_of_memory() -> None:
    trainer = make_trainer(ValueError("not a memory problem"))

    with pytest.raises(ValueError, match="not a memory problem"):
        trainer.train()

    trainer.logger.error.assert_not_called()
