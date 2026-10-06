import logging
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from logging import LogRecord
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from rich.console import Console
from rich.logging import RichHandler
from tqdm import TqdmExperimentalWarning
from tqdm import tqdm as std_tqdm

if TYPE_CHECKING:
    from typing import Any

FORMAT = "([green]%(name)s[/green]) %(message)s "

__set_up = False


def __get_handler(rank: str | None = None) -> RichHandler:
    """
    Get a custom RichHandler for logging.

    Parameters
    ----------
    rank
        Rank of the process in distributed training. If None, no rank is displayed.

    Returns
    -------
    RichHandler
        Configured RichHandler instance.
    """
    handler = RichHandler(
        level=logging.NOTSET,
        # console=Console(width=200, force_terminal=True, no_color=False),
        console=Console(no_color=False),
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        log_time_format="[%X]",
    )
    handler.formatter = logging.Formatter(
        (rf"[orange1]\[RANK {rank}][/orange1] " if rank else "") + FORMAT, datefmt="[%X]"
    )
    return handler


@contextmanager
def with_log_level(logger: logging.Logger, level: int) -> Iterator[None]:
    """
    Context manager to temporarily set the log level of a logger.

    Parameters
    ----------
    logger
        The logger whose level is to be set.
    level
        The log level to set temporarily.
    """
    old_level = logger.getEffectiveLevel()
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(old_level)


def setup_logger(name: str | None = None) -> logging.Logger:
    """
    Custom function that initializes and returns a logger with a RichHandler.

    Parameters
    ----------
    name
        Name of the logger. If None, the root logger is used.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    logger.propagate = True  # False messes up pytest caplog
    logger.setLevel(os.environ.get("GROUNDSCALE_LOGLEVEL", "INFO"))

    if name is not None:
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

    if os.environ.get("RANK", "0") != "0" and logger.level > logging.DEBUG:
        logger.propagate = False

    if logger.level <= logging.DEBUG:
        logger.debug(f"Logger {name} initialized.")

    return logger


@contextmanager
def tqdm_logging_redirect(
    loggers: "list[logging.Logger] | None" = None,
    tqdm_class: "type[std_tqdm[Any]] | None" = None,
    *tqdm_args: "Any",
    **tqdm_kwargs: "Any",
) -> "Iterator[std_tqdm[Any]]":
    """
    Context manager to allow logging output to be printed out without interfering with `tqdm` progress bars.

    Parameters
    ----------
    loggers
        List of loggers to redirect. If None, defaults to [logging.root].
    tqdm_class
        The `tqdm` class to use for progress bars. If None, defaults to standard `tqdm`.
    *tqdm_args
        Additional arguments and keyword arguments to pass to the `tqdm` class.

    Yields
    ------
    Iterator[std_tqdm]
        The progress bar instance created by `tqdm_class`.
    """
    # TODO: currently does not support tqdm.rich, is only tested with standard tqdm
    if loggers is None:
        loggers = [logging.root]
    if tqdm_class is None:
        tqdm_class = std_tqdm
    try:
        tqdm_kwargs["disable"] = os.environ.get("RANK", "0") != "0" or os.environ.get("GROUNDSCALE_NO_TQDM", "0") == "1"
        with tqdm_class(
            *tqdm_args,
            **tqdm_kwargs,
        ) as pbar:
            for logger in loggers:
                for handler in logger.handlers:
                    old_emit = handler.emit

                    def new_emit(record: "LogRecord") -> None:
                        with pbar.external_write_mode():
                            old_emit(record)

                    handler.old_emit = old_emit  # type: ignore
                    handler.emit = new_emit
            yield pbar
    finally:
        for logger in loggers:
            for handler in logger.handlers:
                if handler.emit.__name__ == "new_emit":
                    handler.emit = handler.old_emit  # type: ignore


def setup_logging() -> None:
    # Set up basic configuration for unmanaged loggers
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "WARNING"),
        force=True,  # True messes up pytest caplog
        format=FORMAT,
        datefmt="[%X]",
        handlers=[__get_handler(rank=os.environ.get("RANK", None))],
    )


if not __set_up:
    setup_logging()

    # Suppress specific warnings
    warnings.filterwarnings(
        "ignore", message=r".*pkg_resources is deprecated as an API.*", category=UserWarning, module="louvain"
    )
    warnings.filterwarnings("ignore", message=r".*GPSampler is experimental.*")
    warnings.filterwarnings(
        "ignore", message=r".*dynamo_pgo force disabled by torch.compiler.config.force_disable_caches*"
    )
    warnings.filterwarnings("ignore", message=r".*Using an existing study with name .* instead of creating a new one.*")
    warnings.filterwarnings("ignore", message=r".*Trial [0-9]+ pruned.*")
    warnings.filterwarnings("ignore", message=r".*Rich is experimental/alpha.*", category=TqdmExperimentalWarning)

    # Setup plotting variables
    plt.rcParams.update({
        "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
        "axes.facecolor": (0.0, 0.0, 0.0, 0.0),
        "legend.facecolor": (1.0, 1.0, 1.0, 0.7),
        "savefig.transparent": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        "savefig.dpi": 300,
    })

    # Set up specific loggers
    setup_logger("optuna")
    logging.getLogger("dotenv.main").setLevel(logging.CRITICAL)

    __set_up = True
