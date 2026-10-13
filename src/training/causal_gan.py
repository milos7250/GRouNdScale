from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.distributed import is_initialized as is_ddp_initialized
from torch.optim import AdamW

from .dicts import CausalGANGenLosses, CausalGANLabelerLosses, CausalGANLosses
from .gan import GANTrainer

if TYPE_CHECKING:
    from torch import Tensor

    from gans import CausalGAN

    from .dicts import CausalGANTrainingArgs, SummaryArgs


class CausalGANTrainer(GANTrainer):
    def __init__(
        self,
        gan: "CausalGAN",
        train_file: Path,
        valid_file: Path,
        training_args: "CausalGANTrainingArgs",
        summary_args: "SummaryArgs",
        output_dir: Path,
    ) -> None:
        super().__init__(gan, train_file, valid_file, training_args, summary_args, output_dir)
        self.gan = gan
        self.training_args = training_args

        self.mse = torch.nn.MSELoss()
        self.modules["labeler"] = self.gan.labeler
        self.modules["antilabeler"] = self.gan.antilabeler
        self.genes = self.gan.gen.genes_tensor
        self.tfs = self.gan.gen.tfs_tensor

        if gan.device == "cpu":
            self.logger.warning("Training on CPU may be very slow. Consider using a GPU if possible.")

    def _init_optimizers(self) -> None:
        super()._init_optimizers()
        self.optimizers["labeler"] = AdamW(
            filter(lambda p: p.requires_grad, self.modules["labeler"].parameters()),
            lr=torch.tensor(self.training_args["labeler_alpha"], device=self.gan.device),
            betas=(self.training_args["beta1"], self.training_args["beta2"]),
            amsgrad=True,
            fused=True,
        )
        self.optimizers["antilabeler"] = AdamW(
            filter(lambda p: p.requires_grad, self.modules["antilabeler"].parameters()),
            lr=torch.tensor(self.training_args["antilabeler_alpha"], device=self.gan.device),
            betas=(self.training_args["beta1"], self.training_args["beta2"]),
            amsgrad=True,
            fused=True,
        )

    def _compile_and_ddp(self, compile_modules: bool) -> None:
        if not self.compiled:
            if compile_modules and not is_ddp_initialized():
                self._labeler_step = torch.compile(
                    self._labeler_step, fullgraph=True, mode="max-autotune-no-cudagraphs"
                )
                self._antilabeler_step = torch.compile(
                    self._antilabeler_step, fullgraph=True, mode="max-autotune-no-cudagraphs"
                )
            super()._compile_and_ddp(compile_modules)
        else:
            self.logger.warning("Attempted to compile already compiled trainer. This should not have any effect.")

    def _antilabeler_step(self, cells: "Tensor", genes: "Tensor", tfs: "Tensor") -> tuple["Tensor", "Tensor"]:
        """
        Performs a forward pass of the antilabeler and computes the antilabeler loss.

        Parameters
        ----------
        cells
            Tensor containing a batch of cells.
        genes
            Tensor containing the indices of the genes in the causal graph.
        tfs
            Tensor containing the indices of the TFs in the causal graph.

        Returns
        -------
        tuple[Tensor, Tensor]
            Antilabeler's loss for the current batch and the predicted TFs.
        """
        predicted_tfs = self.modules["antilabeler"](cells[:, genes])
        antilabeler_loss = self.mse(predicted_tfs, cells[:, tfs])

        return antilabeler_loss, predicted_tfs

    def _labeler_step(self, cells: "Tensor", genes: "Tensor", tfs: "Tensor") -> tuple["Tensor", "Tensor"]:
        """
        Performs a forward pass of the labeler and computes the labeler loss.

        Parameters
        ----------
        cells
            Tensor containing a batch of cells.
        genes
            Tensor containing the indices of the genes in the causal graph.
        tfs
            Tensor containing the indices of the TFs in the causal graph.

        Returns
        -------
        Tensor
            Labeler's loss for the current batch and the predicted TFs.
        """
        predicted_tfs = self.modules["labeler"](cells[:, genes])
        labeler_loss = self.mse(predicted_tfs, cells[:, tfs])

        return labeler_loss, predicted_tfs

    def _train_labelers(self, real_cells: "Tensor") -> CausalGANLabelerLosses:
        """
        Trains the labeler (on real and fake) and anti-labeler (on fake only).

        Parameters
        ----------
        real_cells
            Tensor containing a batch of real cells.

        Returns
        -------
        CausalGANLabelerLosses
            CausalGANLabelerLosses object containing the labeler and anti-labeler losses.
        """
        torch.compiler.cudagraph_mark_step_begin()

        with torch.no_grad():
            _, _, fake = self._generator_step()

        losses: dict[str, float] = {}
        # train anti-labeler
        self.optimizers["antilabeler"].zero_grad(set_to_none=True)
        antilabeler_loss, _ = self._antilabeler_step(fake, self.genes, self.tfs)
        losses["Antilabeler Loss"] = antilabeler_loss.item()
        antilabeler_loss.backward()
        self.optimizers["antilabeler"].step()

        # train labeler on fake data
        self.optimizers["labeler"].zero_grad(set_to_none=True)
        labeler_fake_loss, _ = self._labeler_step(fake, self.genes, self.tfs)
        losses["Labeler Fake Loss"] = labeler_fake_loss.item()
        labeler_fake_loss.backward()
        self.optimizers["labeler"].step()

        # train labeler on real data
        self.optimizers["labeler"].zero_grad(set_to_none=True)
        labeler_real_loss, _ = self._labeler_step(real_cells, self.genes, self.tfs)
        losses["Labeler Real Loss"] = labeler_real_loss.item()
        labeler_real_loss.backward()
        self.optimizers["labeler"].step()

        return CausalGANLabelerLosses(**losses)

    def _train_generator(self) -> CausalGANGenLosses:
        """
        Trains the causal generator for one iteration.

        Returns
        -------
        CausalGANGenLosses
            CausalGANGenLosses object containing the generator, labeler and anti-labeler losses.
        """
        self.optimizers["gen"].zero_grad(set_to_none=True)

        gen_loss, _, fake = self._generator_step()

        labeler_loss, _ = self._labeler_step(fake, self.genes, self.tfs)
        antilabeler_loss, _ = self._antilabeler_step(fake, self.genes, self.tfs)

        total_loss = torch.nansum(torch.stack([gen_loss, labeler_loss, antilabeler_loss]))
        losses = CausalGANGenLosses({
            "Generator Loss": gen_loss.item(),
            "Generator Labeler Loss": labeler_loss.item(),
            "Generator Antilabeler Loss": antilabeler_loss.item(),
            "Generator Total Loss": total_loss.item(),
        })

        total_loss.backward()  # only total_loss needs to be backpropagated as it includes labeler and anti-labeler losses

        # Update weights
        self.optimizers["gen"].step()

        return losses

    def _training_step(
        self, real_cells: "Tensor", real_labels: "Tensor", critic_iter: int, c_lambda: float
    ) -> CausalGANLosses:
        # Train the generator and critic as in the base GAN trainer
        losses = super()._training_step(real_cells, real_labels, critic_iter, c_lambda)

        if self._should_run(self.training_args["labeler_training_interval"], root_thread_only=False):
            labeler_losses = self._train_labelers(real_cells)
        else:
            labeler_losses = CausalGANLabelerLosses({
                "Labeler Real Loss": np.nan,
                "Labeler Fake Loss": np.nan,
                "Antilabeler Loss": np.nan,
            })

        return CausalGANLosses(losses | labeler_losses)  # pyright: ignore[reportOperatorIssue]

    def _get_learning_rates_dict(self) -> dict[str, float]:
        return super()._get_learning_rates_dict() | {
            "Labeler LR": self.training_args["labeler_alpha"],  # pyright: ignore[reportAttributeAccessIssue]
            "Antilabeler LR": self.training_args["antilabeler_alpha"],  # pyright: ignore[reportAttributeAccessIssue]
        }

    def _ignore_first_n_in_running_average(self) -> int:
        """
        Determines the number of initial steps to ignore when computing the running average time per step.

        Returns
        -------
        int
            The number of initial steps to ignore in the running average calculation.
        """
        # Also ignore all iterations before labeler and antilabeler is warmed up
        return max(super()._ignore_first_n_in_running_average(), self.training_args["labeler_training_interval"] + 1)
