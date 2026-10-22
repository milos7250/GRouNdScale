import functools
import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from torch import OutOfMemoryError
from torch._inductor.select_algorithm import NoValidChoicesError
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer

T = TypeVar("T")

# Environment variables that lower the GPU memory needed for training, with the value to set and what it does. Both are
# read when torch starts, so they have to be set before the process starts.
LOW_MEMORY_ENVIRONMENT = {
    "PYTORCH_CUDA_ALLOC_CONF": (
        "expandable_segments:True",
        "reduces the fragmentation of the GPU memory held by PyTorch",
    ),
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": (
        "ATEN",
        (
            "autotunes compiled matrix multiplications over cuBLAS only; the Triton kernels it skips can reserve "
            "several GiB of GPU memory while they are benchmarked"
        ),
    ),
}


def set_learning_rate_scheduler(
    optimizer: "Optimizer",
    alpha_0: float,
    alpha_final: float,
    max_steps: int,
    decay_type: str = "cosine",
    warmup_percent: float = 0.02,
    holding_percent: float = 0.60,
) -> LRScheduler:
    """
    Set up a warmup, holding, and decay learning rate schedule.

    Parameters
    ----------
    optimizer
        Optimizer for which to create a learning rate scheduler.
    alpha_0
        Initial learning rate.
    alpha_final
        Final learning rate.
    max_steps
        Total number of training steps. When current_step=max_steps, alpha_final
        will be set as the learning rate.
    decay_type
        Decay curve to use: ``linear``, ``cosine``, or ``exponential``.
    warmup_percent
        Percentage of total steps to use for learning rate warmup (default 0.02).
    holding_percent
        Percentage of total steps at which the holding phase ends, including warmup
        (default 0.60).

    Returns
    -------
    LRScheduler
        Learning rate scheduler. Call the step() function on this
        scheduler in the training loop.
    """
    if decay_type not in {"linear", "cosine", "exponential"}:
        raise ValueError(f"Unknown learning rate decay type: {decay_type!r}")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if alpha_0 <= 0 or alpha_final < 0:
        raise ValueError("alpha_0 must be positive and alpha_final must be non-negative")
    if warmup_percent < 0 or holding_percent < 0 or warmup_percent > holding_percent:
        raise ValueError("warmup_percent and holding_percent must be non-negative, with warmup at most holding")
    if holding_percent > 1:
        raise ValueError("holding_percent must be at most 1")

    warmup_steps = int(max_steps * warmup_percent)
    holding_end_step = int(max_steps * holding_percent)
    decay_steps = max_steps - holding_end_step
    final_factor = alpha_final / alpha_0

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return 0.01 + 0.99 * step / warmup_steps
        if step < holding_end_step or decay_steps == 0:
            return 1.0

        progress = min(step - holding_end_step, decay_steps) / decay_steps
        if decay_type == "linear":
            return 1.0 + (final_factor - 1.0) * progress
        if decay_type == "cosine":
            return final_factor + (1.0 - final_factor) * (1.0 + math.cos(math.pi * progress)) / 2
        return final_factor**progress

    return LambdaLR(optimizer, lr_lambda=lr_factor)


class RunningAverage:
    """
    Class for computing a running average of a metric.
    """

    def __init__(self, ignore_first: int = 0) -> None:
        self.total = 0.0
        self.count = 0
        self.ignore_next = ignore_first

    def update(self, value: float, n: int = 1) -> None:
        """
        Updates the running average with a new value.

        Parameters
        ----------
        value
            New value to update the running average with.
        n
            Number of samples that the value corresponds to (default
        """
        if self.ignore_next > 0:
            self.ignore_next -= n
            if self.ignore_next < 0:
                n = -self.ignore_next
                self.ignore_next = 0
            else:
                return
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        """
        Returns the current running average.

        Returns
        -------
        float
            Current running average. Returns nan if no values have been added yet.
        """
        return self.total / self.count if self.count > 0 else float("nan")


def _is_out_of_memory(error: BaseException) -> bool:
    """Whether the error, or an error it was raised from, is the GPU running out of memory.

    Autotuning only logs the out-of-memory errors of the kernels it benchmarks and raises NoValidChoicesError when all
    of them failed, so that error counts as running out of memory too.
    """
    seen: set[int] = set()
    pending: list[BaseException | None] = [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (OutOfMemoryError, NoValidChoicesError)):
            return True
        pending += [current.__cause__, current.__context__, getattr(current, "inner_exception", None)]
    return False


def hint_on_out_of_memory(train: Callable[..., T]) -> Callable[..., T]:
    """Decorates a trainer method to log, when the GPU runs out of memory, which environment variables can help.

    Only the settings of `LOW_MEMORY_ENVIRONMENT` that are not set already are suggested. The error is re-raised.
    """

    @functools.wraps(train)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
        try:
            return train(self, *args, **kwargs)
        except Exception as error:
            if _is_out_of_memory(error):
                missing = [
                    f"{name}={value} ({effect})"
                    for name, (value, effect) in LOW_MEMORY_ENVIRONMENT.items()
                    if value not in os.environ.get(name, "")
                ]
                if missing:
                    self.logger.error(
                        "Training ran out of GPU memory. Setting these environment variables before starting can "
                        "help, at little or no cost in speed: " + "; ".join(missing)
                    )
            raise

    return wrapper
