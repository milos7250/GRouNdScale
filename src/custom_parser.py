from collections.abc import Callable
from configparser import _UNSET, ConfigParser, ExtendedInterpolation  # pyright: ignore[reportAttributeAccessIssue]
from copy import deepcopy
from os import environ
from pathlib import Path
from typing import Any, overload

import click


class MyConfigParser(ConfigParser):
    """
    Custom ConfigParser that adds an option to save interpolated version.
    """

    def save_interpolated(self, file_path: Path) -> None:
        """
        Save the interpolated version of the configuration to a file.

        Parameters
        ----------
        file_path
            The path to the file where the interpolated configuration will be saved.
        """
        cfg_parser = deepcopy(self)
        default_keys = list(cfg_parser.defaults().keys())
        for section in cfg_parser.sections():
            for key, value in cfg_parser.items(section):
                if key not in default_keys:
                    cfg_parser.set(section, key, value)
                if value == "":
                    cfg_parser.remove_option(section, key)
        for option in list(cfg_parser.defaults().keys()):
            cfg_parser.remove_option("DEFAULT", option)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as cfg_file:
            cfg_parser.write(cfg_file)

    @overload
    def getpath(self, section: str, option: str) -> Path: ...
    @overload
    def getpath(self, section: str, option: str, fallback: Path | str) -> Path: ...
    @overload
    def getpath(self, section: str, option: str, fallback: None) -> Path | None: ...
    def getpath(self, section: str, option: str, fallback: Path | None | str | object = _UNSET) -> Path | None:
        """
        Get a configuration value as a Path object, with an optional fallback.

        Parameters
        ----------
        section
            The section of the configuration file.
        option
            The option within the section to retrieve.
        fallback
            The fallback value to return if the option is not found. If not provided, a KeyError will be raised if the
            option is not found.

        Returns
        -------
        Path | None
            The configuration value as a Path object, or None if the option is not found and fallback is None.
        """
        value = self.get(section, option, fallback=fallback)
        return Path(value) if value is not None else None  # pyright: ignore[reportArgumentType]


def get_configparser() -> MyConfigParser:
    """
    Configure and read config file .cfg .ini parser.

    Returns
    -------
    MyConfigParser
    """
    return MyConfigParser(
        defaults=environ,
        empty_lines_in_values=False,
        allow_no_value=True,
        inline_comment_prefixes=";",
        interpolation=ExtendedInterpolation(),
    )


def click_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that applies Click CLI options to a function.

    The resulting function signature will receive the following parameters:
    `config, preprocess, create_grn, train, optimize_hyperparameters, generate, evaluate, benchmark_grn, perturb`.
    """
    options = [
        click.option(
            "--config",
            required=True,
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="Path to the configuration file",
        ),
        click.option("--preprocess", is_flag=True, default=False, help="Preprocess raw data for GAN training"),
        click.option(
            "--create-grn",
            "create_grn",
            is_flag=True,
            default=False,
            help="Infer a GRN from preprocessed data using GRNBoost2 or RustScenic and appropriately format as causal graph",
        ),
        click.option("--train", is_flag=True, default=False, help="Start or resume model training"),
        click.option(
            "--optimize-hyperparameters",
            "optimize_hyperparameters",
            is_flag=True,
            default=False,
            help="Start or resume hyperparameter optimization using Optuna",
        ),
        click.option("--generate", is_flag=True, default=False, help="Simulate single-cells RNA-seq data in-silico"),
        click.option(
            "--evaluate", is_flag=True, default=False, help="Evaluate the data quality of the simulated dataset"
        ),
        click.option(
            "--benchmark-grn",
            "benchmark_grn",
            is_flag=True,
            default=False,
            help="Evaluate the performance of a GRN inference method in inferring the ground truth GRN",
        ),
        click.option(
            "--perturb",
            is_flag=True,
            default=False,
            help="Perform a perturbation experiment using a trained GRouNdScale model",
        ),
    ]

    for opt in reversed(options):
        func = opt(func)
    return func


__all__ = ["MyConfigParser", "click_options", "get_configparser"]
