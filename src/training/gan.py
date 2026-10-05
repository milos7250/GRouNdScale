import os
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd
import torch
import torch._inductor.config
import torch._inductor.select_algorithm
from optuna import TrialPruned
from torch.distributed import barrier
from torch.distributed import is_initialized as is_ddp_initialized
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.rich import tqdm
from umap import UMAP

from evaluation.data_quality import compute_RF_AUROC, plot_UMAP
from loggers import setup_logger, tqdm_logging_redirect
from sc_dataset import get_loader

from .dicts import (
    GANCritLosses,
    GANGenLosses,
    GANLosses,
    LossList,
)
from .helpers import RunningAverage, set_learning_rate_scheduler

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from optuna import Trial
    from torch import Tensor
    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from gans import GAN
    from sc_dataset import SCDataLoader

    from .dicts import GANTrainingArgs, SummaryArgs


class GANTrainer:
    class HookResults(TypedDict, total=False):
        rf_auroc: float

    def __init__(
        self,
        gan: "GAN",
        train_file: Path,
        valid_file: Path,
        training_args: "GANTrainingArgs",
        summary_args: "SummaryArgs",
        output_dir: Path,
    ) -> None:
        self.gan = gan
        self.train_file = train_file
        self.valid_file = valid_file
        self.training_args = training_args
        self.summary_args = summary_args
        self.output_dir = output_dir

        # self.modules keys need to match attribute names of the GAN class
        self.modules: dict[str, Module] = {"gen": self.gan.gen, "crit": self.gan.crit}
        self.loaders: dict[str, SCDataLoader] = {}
        self.optimizers: dict[str, Optimizer] = {}
        self.schedulers: dict[str, LRScheduler] = {}
        self.summary_writers: dict[str, SummaryWriter] = (
            {
                "stats": SummaryWriter(output_dir / "TensorBoard/"),
                "umap": SummaryWriter(output_dir / "TensorBoard/", filename_suffix=".UMAP"),
                "rf_auroc": SummaryWriter(output_dir / "TensorBoard/", filename_suffix=".RF_AUROC"),
            }
            if not is_ddp_initialized() or os.environ.get("RANK") == "0"
            else {}
        )  # Only initialize summary writers on rank 0 to avoid conflicts in DDP

        self.step = 0
        self.compiled = False
        self.logger = setup_logger(__name__)

    def _init_loaders(self) -> None:
        self.logger.debug("Initializing data loaders...")
        self.loaders["train"] = get_loader(
            self.train_file, batch_size=self.gan.batch_size, shuffle=True, drop_last=True
        )
        self.loaders["valid"] = get_loader(self.valid_file, batch_size=2000, shuffle=False, drop_last=False)

    def _init_optimizers(self) -> None:
        self.logger.debug("Initializing optimizers...")
        self.optimizers["gen"] = AdamW(
            filter(lambda p: p.requires_grad, self.gan.gen.parameters()),
            lr=torch.tensor(self.training_args["gen_alpha_0"], device=self.gan.device),
            betas=(self.training_args["beta1"], self.training_args["beta2"]),
            amsgrad=True,
            fused=True,
        )

        self.optimizers["crit"] = AdamW(
            filter(lambda p: p.requires_grad, self.gan.crit.parameters()),
            lr=torch.tensor(self.training_args["crit_alpha_0"], device=self.gan.device),
            betas=(self.training_args["beta1"], self.training_args["beta2"]),
            amsgrad=True,
            fused=True,
        )

    def _init_schedulers(self) -> None:
        self.logger.debug("Initializing schedulers...")

        def scheduler_args(comp: str):
            return {
                "optimizer": self.optimizers[comp],
                "alpha_0": self.training_args[f"{comp}_alpha_0"],
                "alpha_final": self.training_args[f"{comp}_alpha_final"],
                "max_steps": self.training_args["max_steps"],
                "decay_type": self.training_args.get("lr_decay_type", "cosine"),
                "warmup_percent": self.training_args.get("lr_warmup_percent", 0.02),
                "holding_percent": self.training_args.get("lr_holding_percent", 0.60),
            }

        self.schedulers["gen"] = set_learning_rate_scheduler(**scheduler_args("gen"))
        self.schedulers["crit"] = set_learning_rate_scheduler(**scheduler_args("crit"))

    def _init_umap(self) -> None:
        """Precompute UMAP embeddings for the validation set to speed up UMAP plotting during training."""
        self.logger.debug("Initializing UMAP...")
        self.umap = UMAP(random_state=42, min_dist=0.0, n_jobs=1)
        self.umap.fit(self.loaders["valid"].dataset.cells)  # ensure UMAP is fitted only once to preserve comparability
        self.real_embedding = np.array(self.umap.transform(self.loaders["valid"].dataset.cells))

    def _compile_and_ddp(self, compile_modules: bool) -> None:
        if not self.compiled:
            if compile_modules:
                torch._inductor.select_algorithm.PRINT_AUTOTUNE = False  # to suppress autotune printing
                torch._inductor.config.max_autotune_report_choices_stats = 0  # to suppress autotune stats printing
            if compile_modules and not is_ddp_initialized():
                self._critic_step = torch.compile(self._critic_step, fullgraph=True, mode="max-autotune-no-cudagraphs")
                self._generator_step = torch.compile(
                    self._generator_step, fullgraph=True, mode="max-autotune-no-cudagraphs"
                )
            if compile_modules and is_ddp_initialized():
                for key, module in self.modules.items():
                    self.modules[key] = DDP(
                        torch.compile(module, fullgraph=True, mode="max-autotune-no-cudagraphs"),
                        device_ids=[int(os.environ["LOCAL_RANK"])],
                    )
            if not compile_modules and is_ddp_initialized():
                for key, module in self.modules.items():
                    self.modules[key] = DDP(module, device_ids=[int(os.environ["LOCAL_RANK"])])
            if is_ddp_initialized():
                for key in self.modules:
                    self.gan.__dict__[key] = self.modules[
                        key
                    ]  # Update the module reference in the GAN to the DDP module
        else:
            self.logger.warning("Attempted to compile already compiled trainer. This should not have any effect.")
        self.compiled = True

    @property
    def training_mode(self) -> list[bool]:
        return [module.training for module in self.modules.values()]

    @training_mode.setter
    def training_mode(self, training: list[bool] | bool) -> None:
        if isinstance(training, bool):
            training = [training] * len(self.modules)
        for module, is_training in zip(self.modules.values(), training):
            module.train(is_training)

    def _load_checkpoint(self, checkpoint_path: Path, model_only: bool = False) -> None:
        self.logger.debug(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.gan.device)
        for key, module in self.modules.items():
            if is_ddp_initialized():
                # DDP wraps the module, so we need to load the state dict into the wrapped module
                module.module.load_state_dict(checkpoint[f"{key}_state_dict"])  # pyright: ignore[reportAttributeAccessIssue]
            else:
                module.load_state_dict(checkpoint[f"{key}_state_dict"])
        if not model_only:
            if any(
                key not in checkpoint
                for key in ["step"]
                + [f"{key}_optimizer_state_dict" for key in self.optimizers]
                + [f"{key}_scheduler_state_dict" for key in self.schedulers]
            ):
                self.logger.warning(
                    f"Checkpoint at {checkpoint_path} is missing optimizer/scheduler state dicts or step count. Loading model weights only."
                )
            self.step = checkpoint["step"] + 1
            for key, optimizer in self.optimizers.items():
                optimizer.load_state_dict(checkpoint[f"{key}_optimizer_state_dict"])
            for key, scheduler in self.schedulers.items():
                scheduler.load_state_dict(checkpoint[f"{key}_scheduler_state_dict"])

    def _save_checkpoint(self, output_path: Path, model_only: bool = False) -> None:
        checkpoint = {}
        for key, module in self.modules.items():
            if is_ddp_initialized():
                # DDP wraps the module, so we need to save the state dict of the wrapped module
                checkpoint[f"{key}_state_dict"] = module.module.state_dict()  # pyright: ignore[reportAttributeAccessIssue]
            else:
                checkpoint[f"{key}_state_dict"] = module.state_dict()
        if not model_only:
            checkpoint["step"] = self.step
            for key, optimizer in self.optimizers.items():
                checkpoint[f"{key}_optimizer_state_dict"] = optimizer.state_dict()
            for key, scheduler in self.schedulers.items():
                checkpoint[f"{key}_scheduler_state_dict"] = scheduler.state_dict()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_path.with_suffix(".pth"))

    def _should_run(self, freq: int, root_thread_only: bool) -> bool:
        if root_thread_only and is_ddp_initialized() and os.environ.get("RANK") != "0":
            return False
        return (freq > 0 and self.step % freq == 0 and self.step > 0) or (self.step == self.training_args["max_steps"])

    @staticmethod
    def _generator_loss(crit_fake_pred: "Tensor") -> "Tensor":
        """
        Compute the generator loss from the critic's score of the generated cells.

        Parameters
        ----------
        crit_fake_pred
            The critic's score on fake generated cells.

        Returns
        -------
        Tensor
            Generator's loss value for the current batch.
        """
        return -1.0 * torch.mean(crit_fake_pred)

    @staticmethod
    def _critic_loss(
        crit_fake_pred: "Tensor",
        crit_real_pred: "Tensor",
        gp: "Tensor",
        c_lambda: float,
    ) -> "Tensor":
        """
        Compute critic's loss given the its scores on real and fake cells,
        the gradient penalty, and gradient penalty regularization hyper-parameter.

        Parameters
        ----------
        crit_fake_pred
            Critic's score on fake cells.
        crit_real_pred
            Critic's score on real cells.
        gp
            Unweighted gradient penalty
        c_lambda
            Regularization hyper-parameter to be used with the gradient penalty
            in the WGAN loss.

        Returns
        -------
        Tensor
            Critic's loss for the current batch.
        """
        return torch.nansum(torch.stack([torch.mean(crit_fake_pred) - torch.mean(crit_real_pred), c_lambda * gp]))

    @torch.compiler.set_stance("force_eager")  # gradients not supported in compiled mode
    def _get_gradient(
        self,
        real: "Tensor",
        fake: "Tensor",
        real_labels: "Tensor",
    ) -> "Tensor":
        """
        Compute the gradient of the critic's scores with respect to interpolations
        of real and fake cells.

        Parameters
        ----------
        real
            A batch of real cells.
        fake
            A batch of fake cells.
        real_labels
            A batch of real labels (corresponding to real_cells). Not used in non-conditional GAN.

        Returns
        -------
        Tensor
            Gradient of the critic's score with respect to interpolated data.
        """

        # Mix real and fake cells together
        epsilon = torch.rand(len(real), 1, device=self.gan.device)
        interpolates = real * epsilon + fake * (1 - epsilon)
        interpolates.requires_grad_(True)

        # Calculate the critic's scores on the mixed data
        critic_interpolates = self.modules["crit"](interpolates)

        # Take the gradient of the scores with respect to the data
        gradient = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates, device=self.gan.device),
            create_graph=True,
        )[0]
        return gradient  # noqa: RET504

    @staticmethod
    def _gradient_penalty(gradient: "Tensor") -> "Tensor":
        """
        Compute the gradient penalty given a gradient.

        Parameters
        ----------
        gradient
            The gradient of the critic's score with respect to
            the interpolated data.

        Returns
        -------
        Tensor
            Gradient penalty of the given gradient.
        """
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1, dtype=gradient.dtype)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        return torch.mean((gradient_norm - 1) ** 2)  # pyright: ignore[reportUnknownArgumentType]

    def _critic_step(self, real_cells: "Tensor", real_labels: "Tensor") -> tuple["Tensor", "Tensor", "Tensor"]:
        """
        Performs a forward pass of the critic on real and fake cells.

        Parameters
        ----------
        real_cells
            Tensor containing a batch of real cells.
        real_labels
            Tensor containing a batch of real labels (corresponding to real_cells). Not used in non-conditional GAN.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Critic's scores on fake cells, critic's scores on real cells, and the generated fake cells (generated without tracking gradients).
        """
        with torch.no_grad():
            fake_noise = self.gan.generate_noise(len(real_cells), self.gan.latent_dim, self.gan.device)
            fake_cells = self.modules["gen"](fake_noise)

        crit_fake_pred = self.modules["crit"](fake_cells)
        crit_real_pred = self.modules["crit"](real_cells)

        return crit_fake_pred, crit_real_pred, fake_cells

    def _train_critic(self, real_cells: "Tensor", real_labels: "Tensor", c_lambda: float) -> GANCritLosses:
        """
        Trains the critic for one iteration.

        Parameters
        ----------
        real_cells
            Tensor containing a batch of real cells.
        real_labels
            Tensor containing a batch of real labels (corresponding to real_cells).
        c_lambda
            Regularization hyper-parameter for gradient penalty.

        Returns
        -------
        dict[str, float]
            The computed critic loss and gradient penalty.

        Raises
        ------
        RuntimeError
            If the critic optimizer is not initialized.
        """
        self.optimizers["crit"].zero_grad(set_to_none=True)

        crit_fake_pred, crit_real_pred, fake_cells = self._critic_step(real_cells, real_labels)
        gradient = self._get_gradient(real_cells, fake_cells.clone(), real_labels)
        gp = self._gradient_penalty(gradient)

        total_loss = self._critic_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)
        losses = GANCritLosses({
            "Critic Fake Loss": crit_fake_pred.mean().item(),
            "Critic Real Loss": -crit_real_pred.mean().item(),
            "Critic Gradient Penalty Loss": c_lambda * gp.item(),
            "Critic Total Loss": total_loss.item(),
        })

        # Update gradients
        total_loss.backward()

        # Update optimizer
        self.optimizers["crit"].step()

        return losses

    def _generator_step(self) -> tuple["Tensor", "Tensor", "Tensor"]:
        """
        Performs a forward pass of the generator and critic and computes the generator loss.

        Returns
        -------
        Tensor
            Generator's loss for the current batch, critic's scores on fake cells, and the generated fake cells.
        """
        fake_noise = self.gan.generate_noise(self.gan.batch_size, self.gan.latent_dim, device=self.gan.device)

        fake = self.modules["gen"](fake_noise)
        crit_fake_pred = self.modules["crit"](fake)

        gen_loss = self._generator_loss(crit_fake_pred)

        return gen_loss, crit_fake_pred, fake

    def _train_generator(self) -> GANGenLosses:
        """
        Trains the generator for one iteration.

        Returns
        -------
        dict[str, float]
            Dictionary containing only 1 item, the generator loss.

        Raises
        ------
        RuntimeError
            If the generator optimizer is not initialized.
        """
        self.optimizers["gen"].zero_grad(set_to_none=True)

        gen_loss, _, _ = self._generator_step()
        losses = GANGenLosses({
            "Generator Loss": gen_loss.item(),
            "Generator Total Loss": gen_loss.item(),
        })
        gen_loss.backward()

        # Update weights
        self.optimizers["gen"].step()

        return losses

    def _training_step(
        self, real_cells: "Tensor", real_labels: "Tensor", critic_iter: int, c_lambda: float
    ) -> GANLosses:
        """
        Performs one training step

        Parameters
        ----------
        real_cells
            Tensor containing a batch of real cells.
        real_labels
            Tensor containing a batch of real labels (corresponding to real_cells). Not used in non-conditional GAN.
        critic_iter
            Number of training iterations of the critic for each iteration on the generator.
        c_lambda
            Regularization hyper-parameter for gradient penalty.

        Returns
        -------
        dict[str, float]
            Dictionary containing the computed losses.

        Raises
        ------
        RuntimeError
            If the critic or generator learning rate schedulers are not initialized.
        """
        if self.step != 0:
            crit_losses = LossList[GANCritLosses]()
            for _ in range(critic_iter):
                torch.compiler.cudagraph_mark_step_begin()
                crit_losses.append(self._train_critic(real_cells, real_labels, c_lambda))

            crit_losses = crit_losses.avg()
            # Update learning rate
            self.schedulers["crit"].step()
        else:
            crit_losses = GANCritLosses({
                "Critic Total Loss": np.nan,
                "Critic Fake Loss": np.nan,
                "Critic Real Loss": np.nan,
                "Critic Gradient Penalty Loss": np.nan,
            })

        torch.compiler.cudagraph_mark_step_begin()
        gen_losses = self._train_generator()
        losses = GANLosses(
            gen_losses
            | crit_losses  # pyright: ignore[reportOperatorIssue]
            | {
                "Total Loss": np.nansum([crit_losses["Critic Total Loss"], gen_losses["Generator Total Loss"]]).item(),
            }
        )
        self.schedulers["gen"].step()

        return losses

    def _get_validation_loss(self) -> dict[str, float]:
        """
        Computes the validation loss over the entire validation set.

        Returns
        -------
        dict[str, float]
            Dictionary containing average generator loss, critic loss and gradient penalty
            over the entire validation set.
        """
        self.logger.debug("Computing validation loss...")
        was_training = self.training_mode
        self.training_mode = False  # Set modules to eval mode for validation

        losses = LossList[GANLosses]()
        for real_cells, real_labels in self.loaders["valid"]:
            real_cells = real_cells.to(self.gan.device)
            real_labels = real_labels.flatten().to(self.gan.device)
            with torch.no_grad():
                crit_fake_pred, crit_real_pred, fake_cells = self._critic_step(real_cells, real_labels)

            gradient = self._get_gradient(real_cells, fake_cells, real_labels)
            gp = self._gradient_penalty(gradient)

            with torch.inference_mode():
                gen_loss = self._generator_loss(crit_fake_pred)
                crit_loss = self._critic_loss(
                    crit_fake_pred, crit_real_pred, gp, c_lambda=self.training_args["c_lambda"]
                )

                losses.append(
                    GANLosses({
                        "Generator Loss": gen_loss.item(),
                        "Generator Total Loss": gen_loss.item(),
                        "Critic Fake Loss": crit_fake_pred.mean().item(),
                        "Critic Real Loss": -crit_real_pred.mean().item(),
                        "Critic Gradient Penalty Loss": self.training_args["c_lambda"] * gp.item(),
                        "Critic Total Loss": crit_loss.item(),
                        "Total Loss": torch.nansum(torch.tensor([gen_loss.item(), crit_loss.item()])).item(),
                    })
                )

        losses = losses.avg()

        self.training_mode = was_training  # Restore original training mode
        return {f"Validation {key}": value for key, value in losses.items()}  # pyright: ignore[reportReturnType]

    def _get_learning_rates_dict(self) -> dict[str, float]:
        return {
            "Generator LR": self.schedulers["gen"].get_last_lr()[0].item(),  # pyright: ignore[reportAttributeAccessIssue]
            "Critic LR": self.schedulers["crit"].get_last_lr()[0].item(),  # pyright: ignore[reportAttributeAccessIssue]
        }

    def _log_stats(self, loss_list: LossList[GANLosses]) -> None:
        losses = loss_list.avg(last_n=self.summary_args["summary_freq"])
        losses |= self._get_validation_loss()

        if is_ddp_initialized() and os.environ.get("RANK", "0") != "0":
            return  # Only log stats on rank 0 in DDP to avoid conflicts

        gen_params = [v for k, v in self.gan.gen.named_parameters() if "_lsn" not in k]
        gen_total_sum = sum(p.detach().abs().sum().item() for p in gen_params)
        gen_total_n = sum(p.numel() for p in gen_params)
        gen_mean_abs_weight = gen_total_sum / gen_total_n if gen_total_n else 0.0

        crit_params = list(self.gan.crit.parameters())
        crit_total_sum = sum(p.detach().abs().sum().item() for p in crit_params)
        crit_total_n = sum(p.numel() for p in crit_params)
        crit_mean_abs_weight = crit_total_sum / crit_total_n if crit_total_n else 0.0

        learning_rates_dict = self._get_learning_rates_dict() | {
            "Generator Avg Abs Weight": gen_mean_abs_weight,
            "Critic Avg Abs Weight": crit_mean_abs_weight,
        }

        self.logger.info(f"Step {self.step}:\n" + pd.Series(losses).to_string(float_format="{:.2g}".format))
        self.logger.debug(
            f"Step {self.step}:\n" + pd.Series(learning_rates_dict).to_string(float_format="{:.2g}".format)
        )
        for key, value in (losses | learning_rates_dict).items():
            self.summary_writers["stats"].add_scalar(key, value, self.step)
        self.summary_writers["stats"].flush()

    def _generate_umap_figures(
        self, fake_embedding: np.ndarray, fake_labels: np.ndarray | None
    ) -> tuple["Figure", "Figure", "Figure"]:
        """
        Generates UMAP figures comparing the real and fake cells in the validation set. This class uses the same
        method as the evaluation code.

        Parameters
        ----------
        fake_embedding
            The generated fake cell embeddings to be compared against the real cell embeddings from the validation set.
        fake_labels
            The generated fake cell labels corresponding to `fake_embedding`. (not used in non-conditional GAN)

        Returns
        -------
        scatter_fig
            Generated UMAP scatter plot figure comparing real and fake cells in the validation set.
        hexbin_fig
            Generated UMAP hexbin plot figure comparing real and fake cells in the validation set.
        hist_rel_abun_fig
            Generated histogram figure of the relative abundance of real cells in the validation set compared to fake cells in the UMAP space.
        """
        scatter_fig, hexbin_fig, hist_rel_abun_fig = plot_UMAP(self.real_embedding, fake_embedding)
        return scatter_fig, hexbin_fig, hist_rel_abun_fig

    def _log_umap_plots(self, fake_cells: np.ndarray, fake_labels: np.ndarray | None) -> None:
        """Generates and saves UMAP plots comparing the real and fake cells in the validation set.

        Parameters
        ----------
        fake_cells
            The generated fake cells to be compared against the real cells from the validation set.
        fake_labels
            The generated fake cell labels corresponding to fake_cells.
        """
        self.logger.debug("Generating UMAP plots...")
        fake_embedding = np.array(self.umap.transform(fake_cells))
        scatter_fig, hexbin_fig, hist_rel_abun_fig = self._generate_umap_figures(fake_embedding, fake_labels)

        scatter_fig.suptitle(f"UMAP Scatter Plot at Step {self.step}")
        hexbin_fig.suptitle(f"UMAP Hexbin Plot at Step {self.step}")
        hist_rel_abun_fig.suptitle(f"UMAP Relative Abundance of Real Cells at Step {self.step}")

        umap_dir = self.output_dir / "UMAP"
        umap_dir.mkdir(parents=True, exist_ok=True)
        scatter_fig.savefig(umap_dir / f"step_{self.step}_scatter.jpg")
        hexbin_fig.savefig(umap_dir / f"step_{self.step}_hexbin.jpg")
        hist_rel_abun_fig.savefig(umap_dir / f"step_{self.step}_hist_real_relative.jpg")

        self.summary_writers["umap"].add_figure("UMAP Scatter Plot", scatter_fig, self.step)
        self.summary_writers["umap"].add_figure("UMAP Hexbin Plot", hexbin_fig, self.step)
        self.summary_writers["umap"].add_figure(
            "UMAP istogram Relative Abundance of Real Cells", hist_rel_abun_fig, self.step
        )
        self.summary_writers["umap"].flush()

    def _rf_auroc(self, fake_cells: np.ndarray) -> float:
        """
        Computes the AUROC of a random forest classifier trained to distinguish real from fake cells on the validation set.

        Parameters
        ----------
        fake_cells
            The generated fake cells to be compared against the real cells from the validation set.

        Returns
        -------
        float
            AUROC of the random forest classifier.
        """
        self.logger.debug(f"Step {self.step}: Computing Random Forest AUROC...")

        rf_auroc, fig = compute_RF_AUROC(self.loaders["valid"].dataset.cells, fake_cells)
        rf_auroc_dir = self.output_dir / "RF_AUROC"
        rf_auroc_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(rf_auroc_dir / f"step_{self.step}.jpg")

        self.summary_writers["rf_auroc"].add_figure("Random Forest AUROC", fig, self.step)
        self.summary_writers["stats"].add_scalar("AUROC", rf_auroc, self.step)
        self.summary_writers["rf_auroc"].flush()
        self.summary_writers["stats"].flush()

        self.logger.info(f"Step {self.step}: Computed Random Forest AUROC: {rf_auroc:.3f}")
        return rf_auroc

    def _ignore_first_n_in_running_average(self) -> int:
        """
        Determines the number of initial steps to ignore when computing the running average time per step.

        Returns
        -------
        int
            The number of initial steps to ignore in the running average calculation.
        """
        # By default, ignore the first 2 iterations in the running average to allow both generator and critic to warm up
        return 2

    def eval_hooks(self, loss_list: LossList[GANLosses] | None, force: bool = False) -> "GANTrainer.HookResults":
        hook_results = self.__class__.HookResults()
        run_save = self._should_run(self.summary_args["save_freq"], root_thread_only=True) or force
        run_summary = (
            self._should_run(self.summary_args["summary_freq"], root_thread_only=False) or force
        ) and loss_list
        run_umap = self._should_run(self.summary_args["plt_freq"], root_thread_only=True) or force
        run_rf_auroc = self._should_run(self.summary_args["rf_auroc_freq"], root_thread_only=True) or force

        try:
            if run_save:
                self._save_checkpoint(self.output_dir / f"checkpoints/step_{self.step}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint at step {self.step}: {e}")

        try:
            if run_summary:
                self._log_stats(loss_list)
        except Exception as e:
            self.logger.error(f"Error saving summary at step {self.step}: {e}")

        if any([run_umap, run_rf_auroc]):
            fake_cells, fake_labels = self.gan.generate_cells(len(self.loaders["valid"].dataset))
            try:
                if run_umap:
                    self._log_umap_plots(fake_cells, fake_labels=fake_labels)
            except Exception as e:
                self.logger.error(f"Error saving plots at step {self.step}: {e}")

            try:
                if run_rf_auroc:
                    hook_results["rf_auroc"] = self._rf_auroc(fake_cells)
            except Exception as e:
                self.logger.error(f"Error saving RF AUROC at step {self.step}: {e}")

        return hook_results

    def train(
        self, checkpoint_path: "Path | None" = None, compile_modules: bool = True, trial: "Trial | None" = None
    ) -> float:
        self.logger.info("Initializing training...")
        self._init_loaders()
        self._init_optimizers()
        self._init_schedulers()

        # Save the model graph to TensorBoard for visualization.
        # This is done only on rank 0 in DDP or when not using DDP.
        if (is_ddp_initialized() and os.environ.get("RANK") == "0") or not is_ddp_initialized():
            self._init_umap()
            # Skipped if modules are compiled to avoid OOM errors with large models,
            # as eager execution is required for graph logging, and requires much
            # more memory than compiled execution.
            if not compile_modules:
                self.logger.debug("Saving model graph...")
                self.gan.log_tensorboard_graph(self.output_dir)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        # Compile modules for faster training and wrap in DDP for distributed training if applicable.
        self._compile_and_ddp(compile_modules)

        self.training_mode = True
        torch.set_float32_matmul_precision("high")  # to speed up training with minimal effect on model performance
        loader_gen = iter(self.loaders["train"])
        losses = LossList[GANLosses]()
        average_time_per_step = RunningAverage(ignore_first=self._ignore_first_n_in_running_average())
        eval_hook_results = {}
        self.logger.info("Starting training loop...")
        with (
            tqdm_logging_redirect(
                loggers=[self.logger], tqdm_class=tqdm, desc="Training GAN", total=self.training_args["max_steps"]
            ) as pbar,
        ):
            try:
                while self.step <= self.training_args["max_steps"]:
                    time_start = time_ns()
                    try:
                        real_cells, real_labels = next(loader_gen)
                    except StopIteration:
                        loader_gen = iter(self.loaders["train"])
                        real_cells, real_labels = next(loader_gen)

                    real_cells = real_cells.to(self.gan.device)
                    real_labels = real_labels.flatten().to(self.gan.device)

                    iter_losses = self._training_step(
                        real_cells, real_labels, self.training_args["crit_iter"], self.training_args["c_lambda"]
                    )
                    losses.append(iter_losses)

                    if is_ddp_initialized():
                        barrier()
                    time_end = time_ns()
                    average_time_per_step.update(
                        (time_end - time_start) / 1_000_000
                    )  # update average time per step in milliseconds
                    self.logger.debug(
                        f"Step {self.step}/{self.training_args['max_steps']} completed in {(time_end - time_start) // 1_000_000:.0f} milliseconds (average: {average_time_per_step.average:.0f} ms)"
                    )

                    eval_hook_results = self.eval_hooks(losses)

                    # Allow trial pruning before reaching the end of training based on rf_auroc values
                    if trial is not None and "rf_auroc" in eval_hook_results:
                        trial.report(eval_hook_results["rf_auroc"], step=self.step)
                        if (
                            eval_hook_results["rf_auroc"] > 0.99 and self.step > 10_000
                        ):  # If the RF AUROC is very high after a considerable number of steps, the generator is likely not learning and we can stop early
                            self.logger.error(
                                f"Step {self.step}: RF AUROC is very high ({eval_hook_results['rf_auroc']:.3f}), stopping early."
                            )
                            raise TrialPruned()
                        if trial.should_prune():
                            raise TrialPruned()

                    if is_ddp_initialized():
                        barrier()
                    pbar.update(self.step > 0)  # Update progress bar only after the first step
                    self.step += 1

                pbar.set_description(f"Finished training after {self.step - 1} steps.")
            except TrialPruned:
                self.logger.warning(f"Trial pruned at step {self.step}.")
                self.eval_hooks(None, force=True)  # Run hooks one last time to save final checkpoint and plots
                raise
            except Exception as e:
                self.logger.error(f"Error during training at step {self.step}: {e}")
                self.eval_hooks(None, force=True)  # Run hooks one last time to save final checkpoint and plots
                raise

        return eval_hook_results.get("rf_auroc", np.inf)
