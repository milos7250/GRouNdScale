"""The dense masked layers of GRouNdGAN in the causal generator of the current code, for arabidopsis-ablation.

Importing this module replaces `CausalGenerator._create_generator` and `_create_generator_block`, so that every block
of the causal generator uses `layers.masked_linear.MaskedLinear` (the dense masked linear layer of GRouNdGAN, still kept
in src/ for reference) instead of `sparselinear.SparseLinear`. As in GRouNdGAN (upstream commit 2df087f):

- the masks are dense 32-bit float tensors on the device, and the hidden layers share one mask tensor,
- the weights are initialised with Xavier uniform (input and hidden blocks) or Kaiming normal (output block) and then
  masked with `reapply_mask()`.

Everything else (data loading, training loop, logging, environment) is the current code. launch.py imports this module
before it runs ../profile-memory.py, so the patch is in place when the model is built.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # repository root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))  # publication/, for memory_tools when profile-memory.py runs

from layers.lsn import LSN
from layers.masked_linear import MaskedLinear
from networks import masked_causal_generator as mcg

_original_create_generator = mcg.CausalGenerator._create_generator


def _create_generator(self: mcg.CausalGenerator) -> None:
    # _create_generator builds the three CSR masks and calls _create_generator_block once per block, with the same
    # hidden mask object for every hidden layer. The cache converts each mask to a dense tensor once, so that the hidden
    # layers share it, as in GRouNdGAN, and is dropped afterwards.
    self._dense_masks = {}
    try:
        _original_create_generator(self)
    finally:
        del self._dense_masks


def _create_generator_block(
    self: mcg.CausalGenerator,
    mask,  # scipy.sparse.csr_matrix (bool), shape (n_input_feature, n_output_feature)
    library_size: int | None = None,
    final_layer: bool | None = False,
) -> nn.Sequential:
    dense = self._dense_masks.get(id(mask))
    if dense is None:
        dense = torch.as_tensor(mask.toarray(), dtype=torch.float32, device=self.device)
        self._dense_masks[id(mask)] = dense

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        masked_linear = MaskedLinear(dense, device=self.device)

    if not final_layer:
        nn.init.xavier_uniform_(masked_linear.weight)
        masked_linear.reapply_mask()
        return nn.Sequential(masked_linear, nn.BatchNorm1d(mask.shape[1]), nn.ReLU(inplace=True))

    nn.init.kaiming_normal_(masked_linear.weight, mode="fan_in", nonlinearity="relu")
    masked_linear.reapply_mask()
    torch.nn.init.zeros_(masked_linear.bias)
    if library_size is not None:
        return nn.Sequential(masked_linear, mcg.ReLU(inplace=False), LSN(library_size))
    return nn.Sequential(masked_linear, mcg.ReLU(inplace=False))


mcg.CausalGenerator._create_generator = _create_generator  # type: ignore[method-assign]
mcg.CausalGenerator._create_generator_block = _create_generator_block  # type: ignore[method-assign]

