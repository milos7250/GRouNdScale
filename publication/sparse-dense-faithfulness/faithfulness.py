"""Numerical faithfulness of GRouNdScale's sparse masked layers to GRouNdGAN's dense masked layers.

Each trial draws a random bipartite GRN with G target genes, F candidate TFs and k regulators per gene, and builds
GRouNdScale's CausalGenerator for it (sparse layers from ``sparselinear``). A dense twin of its masked network is then
built from GRouNdGAN's own ``MaskedLinear`` (loaded from the pinned upstream checkout), with the masks taken from the
sparse connectivity and identical weights, biases and batch normalisation. Both networks are run on the same inputs in
the precision under test, and compared

* layer by layer: every masked layer gets the same input (the activation of the sparse network at that layer) and the
  same upstream gradient, and its output, input gradient, weight gradient and bias gradient are compared;
* end to end: the whole masked network (masked layers, batch normalisation, ReLU) is run on the same TF expressions and
  noise, and its output and the gradients of a fixed random linear loss with respect to the inputs and to every
  parameter are compared.

Both are also compared with an FP64 reference (the sparse network in float64; for small networks the dense network in
float64 is also checked against it). Masked-out weight positions are checked to receive exactly zero gradient: for the
dense layer by counting non-zeros of the full gradient outside the mask, for the sparse layer by checking that its
parameters cover exactly the masked-in positions implied by the GRN, so that no masked-out position has a parameter.

FP32 runs with ``torch.set_float32_matmul_precision("highest")``, TF32 with ``"high"`` (the setting used for training in
src/training/gan.py). A calibration matmul is recorded for each mode to confirm that TF32 was in effect.

Usage (from the repository root or anywhere, the script locates src/ itself):
    python faithfulness.py run --out results [--device cuda] [--sizes 32 --min-genes 10 --max-genes 2500 --seeds 4]
    python faithfulness.py summarize --out results
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
PRECISIONS = {"fp32": "highest", "tf32": "high"}
LAYER_QUANTITIES = ("output", "grad_input", "grad_weight", "grad_bias")


# ----------------------------------------------------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------------------------------------------------


def load_masked_linear(upstream: Path):
    """Loads GRouNdGAN's MaskedLinear from the upstream checkout without putting upstream's src/ on sys.path."""
    path = upstream / "src" / "layers" / "masked_linear.py"
    spec = importlib.util.spec_from_file_location("upstream_masked_linear", path)
    assert spec is not None and spec.loader is not None, f"Cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MaskedLinear, path


def random_grn(num_genes: int, num_tfs: int, k: int, rng: np.random.Generator) -> dict[int, set[int]]:
    """Bipartite GRN {target gene: {TF, ...}}: TFs are nodes 0..F-1, targets F..F+G-1, k distinct TFs per target."""
    k = min(k, num_tfs)
    return {num_tfs + gene: set(rng.choice(num_tfs, size=k, replace=False).tolist()) for gene in range(num_genes)}


def expected_nnz(graph: dict[int, set[int]], noise_per_gene: int, width: int, depth: int) -> list[int]:
    """Number of connections of each masked layer implied by the GRN, computed independently of the generator."""
    blocks = [width * (len(regulators) + noise_per_gene) for regulators in graph.values()]
    input_nnz = sum(block * (len(regulators) + noise_per_gene) for block, regulators in zip(blocks, graph.values()))
    hidden_nnz = sum(block * block for block in blocks)
    output_nnz = sum(blocks)
    return [input_nnz, *([hidden_nnz] * depth), output_nnz]


def build_sparse(graph, noise_per_gene: int, depth: int, width: int, device: str) -> tuple[nn.Module, nn.Sequential]:
    from networks.masked_causal_generator import CausalGenerator

    generator = CausalGenerator(
        z_input=1,
        noise_per_gene=noise_per_gene,
        depth_per_gene=depth,
        width_scale_per_gene=width,
        causal_controller=nn.Identity(),
        causal_graph=graph,
        library_size=None,
        device=device,
    ).to(device)
    return generator, generator._generator


def is_sparse_linear(module: nn.Module) -> bool:
    return hasattr(module, "weights") and hasattr(module, "indices")


def masked_layers(network: nn.Sequential) -> list[nn.Module]:
    return [block[0] for block in network]


def build_dense_twin(sparse_network: nn.Sequential, MaskedLinear, device: str) -> nn.Sequential:
    """Dense copy of the sparse masked network built from upstream's MaskedLinear with identical parameters.

    Hidden layers share one connectivity, so they share one mask tensor (MaskedLinear keeps a transposed view of the
    mask it is given), which saves one dense matrix per extra hidden layer without changing any computation.
    """
    blocks = []
    previous: tuple[torch.Tensor, torch.Tensor] | None = None
    for block in sparse_network:
        modules = []
        for module in block:
            if not is_sparse_linear(module):
                modules.append(copy.deepcopy(module))
                continue
            indices = module.indices.to(device)  # row 0: output feature, row 1: input feature
            if previous is not None and previous[0].shape == indices.shape and torch.equal(previous[0], indices):
                mask = previous[1]
            else:
                mask = torch.zeros(module.in_features, module.out_features, device=device)
                mask[indices[1], indices[0]] = 1.0
                previous = (indices, mask)
            dense = MaskedLinear(mask, device=device)
            with torch.no_grad():
                dense.weight.zero_()
                dense.weight[indices[0], indices[1]] = module.weights.detach().to(device)
                dense.bias.copy_(module.bias.detach())
            modules.append(dense)
        blocks.append(nn.Sequential(*modules))
    return nn.Sequential(*blocks).to(device)


# ----------------------------------------------------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------------------------------------------------


def deviation(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Maximum absolute deviation, and relative deviations scaled by the largest |expected| and by ||expected||_2."""
    actual = actual.detach().double()
    expected = expected.detach().double()
    difference = (actual - expected).abs()
    max_abs = difference.max().item() if difference.numel() else 0.0
    scale = expected.abs().max().item() if expected.numel() else 0.0
    norm = torch.linalg.vector_norm(expected).item()
    return {
        "max_abs": max_abs,
        "max_rel": max_abs / scale if scale > 0 else (0.0 if max_abs == 0 else math.inf),
        "rel_l2": torch.linalg.vector_norm(actual - expected).item() / norm if norm > 0 else 0.0,
        "scale": scale,
        "exact": bool(max_abs == 0.0),
    }


def sparse_weight_grad(layer: nn.Module) -> torch.Tensor:
    return layer.weights.grad


def dense_weight_grad_at(layer: nn.Module, indices: torch.Tensor) -> torch.Tensor:
    return layer.weight.grad[indices[0], indices[1]]


def masked_out_nonzeros(layer: nn.Module, indices: torch.Tensor) -> int:
    """Non-zero entries of the dense layer's full weight gradient outside the mask (should be exactly 0)."""
    grad = layer.weight.grad
    total = int(torch.count_nonzero(grad).item())
    inside = int(torch.count_nonzero(grad[indices[0], indices[1]]).item())
    return total - inside


def masked_out_weight_nonzeros(layer: nn.Module, indices: torch.Tensor) -> int:
    """Non-zero dense weights outside the mask (should be exactly 0)."""
    weight = layer.weight.detach()
    return int(torch.count_nonzero(weight).item()) - int(torch.count_nonzero(weight[indices[0], indices[1]]).item())


def run_layer(layer: nn.Module, x: torch.Tensor, grad_output: torch.Tensor) -> dict[str, torch.Tensor]:
    layer.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    y = layer(x)
    y.backward(grad_output)
    return {"output": y.detach(), "grad_input": x.grad.detach()}


class ReluPatterns:
    """Records which units each ReLU of a network lets through (output > 0) during a forward pass.

    End to end, a pre-activation within rounding error of zero can land on different sides of the ReLU kink in two
    implementations. The forward output barely changes, but the gradient of that unit switches between its full value
    and zero, which then dominates the maximum gradient deviation. Counting these flips separates that effect, which
    occurs between any two floating-point evaluations, from a genuine difference between the layers.
    """

    def __init__(self, network: nn.Module) -> None:
        self.patterns: list[torch.Tensor] = []
        self.handles = [
            module.register_forward_hook(self._hook) for module in network.modules() if isinstance(module, nn.ReLU)
        ]

    def _hook(self, module, inputs, output) -> None:
        self.patterns.append(output.detach() > 0)

    def remove(self) -> list[torch.Tensor]:
        for handle in self.handles:
            handle.remove()
        return self.patterns


def flips(a: list[torch.Tensor], b: list[torch.Tensor]) -> int:
    return int(sum(torch.count_nonzero(x != y).item() for x, y in zip(a, b)))


def calibrate(device: str) -> float:
    """Relative error of a 4096^3 float32 matmul against float64, to confirm which matmul precision is in effect."""
    generator = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(1024, 4096, device=device, generator=generator)
    b = torch.randn(4096, 1024, device=device, generator=generator)
    exact = a.double() @ b.double()
    return (torch.linalg.vector_norm((a @ b).double() - exact) / torch.linalg.vector_norm(exact)).item()


def environment(args: argparse.Namespace, masked_linear_path: Path) -> dict[str, Any]:
    import sparselinear
    import torch_sparse

    try:
        upstream_commit = subprocess.run(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001
        upstream_commit = f"unknown ({error})"
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "torch_sparse": getattr(torch_sparse, "__version__", "unknown"),
        "sparselinear": getattr(sparselinear, "__version__", "unknown"),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30 if args.device == "cuda" else None,
        "upstream": str(args.upstream),
        "upstream_commit": upstream_commit,
        "masked_linear_md5": hashlib.md5(masked_linear_path.read_bytes()).hexdigest(),
        "calibration_rel_l2": {},
    }
    if args.device == "cuda":
        for name, setting in PRECISIONS.items():
            torch.set_float32_matmul_precision(setting)
            info["calibration_rel_l2"][name] = calibrate(args.device)
    return info


def run_trial(args, MaskedLinear, num_genes: int, seed: int, records: list[dict[str, Any]]) -> None:
    device = args.device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    graph = random_grn(num_genes, args.tfs, args.k, rng)

    generator, sparse = build_sparse(graph, args.noise_per_gene, args.depth, args.width, device)
    sparse.train()
    layers_s = masked_layers(sparse)
    num_inputs = layers_s[0].in_features
    hidden_dims = layers_s[0].out_features
    batch = args.batch_size

    trial = {
        "genes": num_genes,
        "seed": seed,
        "tfs_used": generator.num_tfs,
        "hidden_dims": hidden_dims,
        "layers": [
            {
                "in": layer.in_features,
                "out": layer.out_features,
                "nnz": int(layer.indices.shape[1]),
                "density": layer.indices.shape[1] / (layer.in_features * layer.out_features),
            }
            for layer in layers_s
        ],
    }

    # Sparse connectivity must be exactly the masked-in positions implied by the GRN, with no duplicates.
    nnz_expected = expected_nnz(graph, args.noise_per_gene, args.width, args.depth)
    nnz_actual = [entry["nnz"] for entry in trial["layers"]]
    unique_actual = [
        int(torch.unique(layer.indices[0] * layer.in_features + layer.indices[1]).numel()) for layer in layers_s
    ]
    trial["sparse_connectivity_ok"] = nnz_expected == nnz_actual == unique_actual

    # Inputs: non-negative TF expressions and standard normal noise, as fed to the masked network during training.
    tf_expressions = torch.randn(batch, generator.num_tfs, device=device).abs()
    noise = torch.randn(batch, generator.num_noises, device=device)
    regulators = torch.cat([tf_expressions, noise], dim=1)
    assert regulators.shape[1] == num_inputs
    loss_weights = torch.randn(batch, num_genes, device=device)

    # Inputs and upstream gradients of each masked layer for the layer-by-layer comparison.
    layer_inputs, layer_grads = [], []
    with torch.no_grad():
        x = regulators
        for block in sparse:
            layer_inputs.append(x.clone())
            layer_grads.append(torch.randn(batch, block[0].out_features, device=device))
            x = block(x)

    # FP64 reference: the sparse network in float64.
    sparse64 = copy.deepcopy(sparse).double().train()
    ref_layers = []
    for layer, x, g in zip(masked_layers(sparse64), layer_inputs, layer_grads):
        result = run_layer(layer, x.double(), g.double())
        result["grad_weight"] = sparse_weight_grad(layer).detach()
        result["grad_bias"] = layer.bias.grad.detach()
        ref_layers.append(result)
    sparse64.zero_grad(set_to_none=True)
    r64 = regulators.double().requires_grad_(True)
    recorder = ReluPatterns(sparse64)
    out64 = sparse64(r64)
    patterns64 = recorder.remove()
    (out64 * loss_weights.double()).sum().backward()
    ref_e2e = {
        "output": out64.detach(),
        "grad_input": r64.grad.detach(),
        "grad_weight": [sparse_weight_grad(layer).detach() for layer in masked_layers(sparse64)],
        "grad_bias": [layer.bias.grad.detach() for layer in masked_layers(sparse64)],
    }
    del sparse64, out64, r64

    dense = build_dense_twin(sparse, MaskedLinear, device).train()
    layers_d = masked_layers(dense)
    indices = [layer.indices.to(device) for layer in layers_s]
    trial["dense_weights_masked_ok"] = all(
        masked_out_weight_nonzeros(layer, idx) == 0 for layer, idx in zip(layers_d, indices)
    )

    # Optional FP64 check of the dense implementation against the sparse FP64 reference, for small networks.
    if hidden_dims <= args.dense64_max_hidden:
        dense64 = copy.deepcopy(dense).double().train()
        r = regulators.double().requires_grad_(True)
        out = dense64(r)
        (out * loss_weights.double()).sum().backward()
        trial["dense64_vs_sparse64"] = {
            "output": deviation(out, ref_e2e["output"]),
            "grad_input": deviation(r.grad, ref_e2e["grad_input"]),
            "grad_weight_max_abs": max(
                deviation(dense_weight_grad_at(layer, idx), ref)["max_abs"]
                for layer, idx, ref in zip(masked_layers(dense64), indices, ref_e2e["grad_weight"])
            ),
        }
        del dense64, out, r

    for precision, setting in PRECISIONS.items():
        torch.set_float32_matmul_precision(setting)
        base = {"precision": precision, "genes": num_genes, "seed": seed, "hidden_dims": hidden_dims}

        # Layer by layer.
        for index, (layer_s, layer_d, idx, x, g, ref) in enumerate(
            zip(layers_s, layers_d, indices, layer_inputs, layer_grads, ref_layers)
        ):
            res_s = run_layer(layer_s, x, g)
            res_s["grad_weight"] = sparse_weight_grad(layer_s).detach()
            res_s["grad_bias"] = layer_s.bias.grad.detach()
            res_d = run_layer(layer_d, x, g)
            res_d["grad_weight"] = dense_weight_grad_at(layer_d, idx).detach()
            res_d["grad_bias"] = layer_d.bias.grad.detach()
            masked_out = masked_out_nonzeros(layer_d, idx)
            layer_d.zero_grad(set_to_none=True)
            layer_s.zero_grad(set_to_none=True)
            kind = "input" if index == 0 else ("output" if index == len(layers_s) - 1 else "hidden")
            for quantity in LAYER_QUANTITIES:
                records.append({
                    **base,
                    "scope": "layer",
                    "layer": index,
                    "kind": kind,
                    "density": trial["layers"][index]["density"],
                    "quantity": quantity,
                    "sparse_vs_dense": deviation(res_s[quantity], res_d[quantity]),
                    "sparse_vs_fp64": deviation(res_s[quantity], ref[quantity]),
                    "dense_vs_fp64": deviation(res_d[quantity], ref[quantity]),
                    "dense_masked_out_grad_nonzeros": masked_out if quantity == "grad_weight" else None,
                })
            del res_s, res_d

        # End to end.
        results = {}
        for name, network, layers in (("sparse", sparse, layers_s), ("dense", dense, layers_d)):
            network.zero_grad(set_to_none=True)
            r = regulators.clone().requires_grad_(True)
            recorder = ReluPatterns(network)
            out = network(r)
            patterns = recorder.remove()
            (out * loss_weights).sum().backward()
            results[name] = {
                "patterns": patterns,
                "output": out.detach(),
                "grad_input": r.grad.detach(),
                "grad_weight": [
                    sparse_weight_grad(layer).detach()
                    if name == "sparse"
                    else dense_weight_grad_at(layer, idx).detach()
                    for layer, idx in zip(layers, indices)
                ],
                "grad_bias": [layer.bias.grad.detach() for layer in layers],
                # Batch normalisation weight and bias: module 1 of each hidden block ("<block>.1.weight").
                "grad_other": {
                    key: param.grad.detach()
                    for key, param in network.named_parameters()
                    if key.split(".")[1] == "1" and param.grad is not None
                },
            }
            if name == "dense":
                results[name]["masked_out"] = [masked_out_nonzeros(layer, idx) for layer, idx in zip(layers, indices)]
            network.zero_grad(set_to_none=True)
            del out, r

        s, d = results["sparse"], results["dense"]
        records.append({
            **base,
            "scope": "end_to_end",
            "quantity": "relu_flips",
            "units": int(sum(pattern.numel() for pattern in s["patterns"])),
            "sparse_vs_dense": flips(s["patterns"], d["patterns"]),
            "sparse_vs_fp64": flips(s["patterns"], patterns64),
            "dense_vs_fp64": flips(d["patterns"], patterns64),
        })
        for quantity in ("output", "grad_input"):
            records.append({
                **base,
                "scope": "end_to_end",
                "quantity": quantity,
                "sparse_vs_dense": deviation(s[quantity], d[quantity]),
                "sparse_vs_fp64": deviation(s[quantity], ref_e2e[quantity]),
                "dense_vs_fp64": deviation(d[quantity], ref_e2e[quantity]),
            })
        for quantity in ("grad_weight", "grad_bias"):
            for index in range(len(layers_s)):
                # In training mode, batch normalisation removes the batch mean, so the loss does not depend on the
                # biases of the layers it follows: their exact gradient is zero and both implementations return
                # rounding noise, which is reported separately rather than as a relative deviation.
                before_bn = quantity == "grad_bias" and index < len(layers_s) - 1
                records.append({
                    **base,
                    "scope": "end_to_end",
                    "quantity": "grad_bias_before_bn" if before_bn else quantity,
                    "layer": index,
                    "density": trial["layers"][index]["density"],
                    "sparse_vs_dense": deviation(s[quantity][index], d[quantity][index]),
                    "sparse_vs_fp64": deviation(s[quantity][index], ref_e2e[quantity][index]),
                    "dense_vs_fp64": deviation(d[quantity][index], ref_e2e[quantity][index]),
                    "dense_masked_out_grad_nonzeros": d["masked_out"][index] if quantity == "grad_weight" else None,
                })
        common = sorted(set(s["grad_other"]) & set(d["grad_other"]))
        for key in common:
            records.append({
                **base,
                "scope": "end_to_end",
                "quantity": "grad_batchnorm",
                "parameter": key,
                "sparse_vs_dense": deviation(s["grad_other"][key], d["grad_other"][key]),
            })
        del results, s, d

    records.append({"scope": "trial", **trial})
    del dense, layers_d, sparse, generator, layers_s
    if device == "cuda":
        torch.cuda.empty_cache()


def command_run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO / "src"))
    MaskedLinear, masked_linear_path = load_masked_linear(args.upstream)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = environment(args, masked_linear_path)
    sizes = sorted({int(round(value)) for value in np.geomspace(args.min_genes, args.max_genes, args.sizes)})
    env["sizes"] = sizes
    env["config"] = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2), flush=True)
    if len(sizes) != args.sizes:
        print(f"Warning: only {len(sizes)} distinct sizes after rounding", flush=True)

    records_path = out / "records.jsonl"
    done = set()
    if records_path.exists() and not args.overwrite:
        for line in records_path.read_text().splitlines():
            record = json.loads(line)
            if record.get("scope") in ("trial", "failed"):
                done.add((record["genes"], record["seed"]))
    elif records_path.exists():
        records_path.unlink()

    # Every GRN of a given size has the same layer shapes (each target has exactly k regulators), so an out-of-memory
    # error at one size means every larger size would fail too; those are recorded as skipped.
    out_of_memory_at = None
    for num_genes in sizes:
        for replicate in range(args.seeds):
            seed = args.base_seed + 1000 * num_genes + replicate
            if (num_genes, seed) in done:
                continue
            start = time.perf_counter()
            records: list[dict[str, Any]] = []
            if out_of_memory_at is not None:
                reason = f"skipped: oom at G={out_of_memory_at}"
                records = [{"scope": "failed", "genes": num_genes, "seed": seed, "reason": reason}]
                status = "skipped"
            else:
                try:
                    run_trial(args, MaskedLinear, num_genes, seed, records)
                    status = "ok"
                except torch.OutOfMemoryError as error:
                    records = [{"scope": "failed", "genes": num_genes, "seed": seed, "reason": f"oom: {error}"[:500]}]
                    status = "oom"
                    out_of_memory_at = num_genes
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            with records_path.open("a") as file:
                for record in records:
                    file.write(json.dumps(record) + "\n")
            peak = torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else 0.0
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            print(
                f"G={num_genes:5d} seed={seed} {status} {time.perf_counter() - start:7.1f} s peak {peak:6.1f} GiB",
                flush=True,
            )


# ----------------------------------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------------------------------


def command_summarize(args: argparse.Namespace) -> None:
    out = Path(args.out)
    env = json.loads((out / "environment.json").read_text())
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    trials = [record for record in records if record["scope"] == "trial"]
    failed = [record for record in records if record["scope"] == "failed"]

    def agg(rows, comparison="sparse_vs_dense"):
        rows = [row for row in rows if comparison in row]
        if not rows:
            return None
        worst_abs = max(rows, key=lambda row: row[comparison]["max_abs"])
        worst_rel = max(rows, key=lambda row: row[comparison]["max_rel"])
        return {
            "max_abs": worst_abs[comparison]["max_abs"],
            "max_abs_at": {"genes": worst_abs["genes"], "seed": worst_abs["seed"]},
            "max_rel": worst_rel[comparison]["max_rel"],
            "max_rel_at": {"genes": worst_rel["genes"], "seed": worst_rel["seed"]},
            "max_rel_l2": max(row[comparison]["rel_l2"] for row in rows),
            "median_max_rel": float(np.median([row[comparison]["max_rel"] for row in rows])),
            "n": len(rows),
        }

    densities = [layer["density"] for trial in trials for layer in trial["layers"]]
    summary: dict[str, Any] = {
        "environment": {key: env[key] for key in ("torch", "cuda", "gpu", "gpu_memory_gib", "upstream_commit")},
        "calibration_rel_l2": env.get("calibration_rel_l2"),
        "trials_ok": len(trials),
        "trials_failed": [(record["genes"], record["seed"], record["reason"][:40]) for record in failed],
        "sizes_ok": sorted({trial["genes"] for trial in trials}),
        "genes_range": [min(t["genes"] for t in trials), max(t["genes"] for t in trials)] if trials else None,
        "hidden_dims_range": [min(t["hidden_dims"] for t in trials), max(t["hidden_dims"] for t in trials)]
        if trials
        else None,
        "density_range": [min(densities), max(densities)] if densities else None,
        "sparse_connectivity_ok": all(trial["sparse_connectivity_ok"] for trial in trials),
        "dense_weights_masked_ok": all(trial["dense_weights_masked_ok"] for trial in trials),
        "dense64_vs_sparse64": None,
        "precisions": {},
    }
    dense64 = [trial["dense64_vs_sparse64"] for trial in trials if "dense64_vs_sparse64" in trial]
    if dense64:
        summary["dense64_vs_sparse64"] = {
            "trials": len(dense64),
            "output_max_rel": max(entry["output"]["max_rel"] for entry in dense64),
            "grad_input_max_rel": max(entry["grad_input"]["max_rel"] for entry in dense64),
            "grad_weight_max_abs": max(entry["grad_weight_max_abs"] for entry in dense64),
        }

    for precision in PRECISIONS:
        rows = [record for record in records if record.get("precision") == precision]
        result: dict[str, Any] = {}
        for scope in ("layer", "end_to_end"):
            result[scope] = {}
            for quantity in (*LAYER_QUANTITIES, "grad_batchnorm", "grad_bias_before_bn"):
                selected = [row for row in rows if row["scope"] == scope and row["quantity"] == quantity]
                if not selected:
                    continue
                result[scope][quantity] = {
                    "sparse_vs_dense": agg(selected),
                    "sparse_vs_fp64": agg(selected, "sparse_vs_fp64"),
                    "dense_vs_fp64": agg(selected, "dense_vs_fp64"),
                }
            if scope == "layer":
                result[scope]["by_kind"] = {
                    kind: agg([
                        row
                        for row in rows
                        if row["scope"] == "layer" and row.get("kind") == kind and row["quantity"] == "output"
                    ])
                    for kind in ("input", "hidden", "output")
                }
        flip_rows = [row for row in rows if row["quantity"] == "relu_flips"]
        no_flip = {(row["genes"], row["seed"]) for row in flip_rows if row["sparse_vs_dense"] == 0}
        result["relu_flips"] = {
            "trials": len(flip_rows),
            "units_per_comparison": sum(row["units"] for row in flip_rows),
            "sparse_vs_dense": sum(row["sparse_vs_dense"] for row in flip_rows),
            "sparse_vs_fp64": sum(row["sparse_vs_fp64"] for row in flip_rows),
            "dense_vs_fp64": sum(row["dense_vs_fp64"] for row in flip_rows),
            "trials_without_sparse_dense_flip": len(no_flip),
        }
        # End-to-end deviations over the trials in which both implementations took the same side of every ReLU kink.
        result["end_to_end_same_relu_pattern"] = {
            quantity: agg([
                row
                for row in rows
                if row["scope"] == "end_to_end"
                and row["quantity"] == quantity
                and (row["genes"], row["seed"]) in no_flip
            ])
            for quantity in ("output", "grad_input", "grad_weight", "grad_bias", "grad_batchnorm")
        }
        # Each masked-out position is checked twice per precision: in the layer-by-layer and the end-to-end backward.
        result["masked_out_positions_checked"] = 2 * sum(
            layer["in"] * layer["out"] - layer["nnz"] for trial in trials for layer in trial["layers"]
        )
        result["masked_out_grad_nonzeros"] = sum(
            row["dense_masked_out_grad_nonzeros"] or 0 for row in rows if row["quantity"] == "grad_weight"
        )
        summary["precisions"][precision] = result

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", type=Path, default=Path("results"))
    run.add_argument("--upstream", type=Path, default=REPO / "upstream", help="GRouNdGAN checkout (pinned 2df087f)")
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--sizes", type=int, default=32, help="number of log-spaced GRN sizes")
    run.add_argument("--min-genes", type=int, default=10)
    run.add_argument("--max-genes", type=int, default=2500)
    run.add_argument("--seeds", type=int, default=4, help="random GRNs, weights and inputs per size")
    run.add_argument("--base-seed", type=int, default=20261110)
    run.add_argument("--tfs", type=int, default=63, help="candidate TFs F (63 in the PBMC GRN)")
    run.add_argument("--k", type=int, default=15, help="regulators per target gene")
    run.add_argument("--noise-per-gene", type=int, default=1)
    run.add_argument("--depth", type=int, default=2)
    run.add_argument("--width", type=int, default=1)
    run.add_argument("--batch-size", type=int, default=1024)
    run.add_argument("--dense64-max-hidden", type=int, default=8000, help="also run dense FP64 up to this width")
    run.add_argument("--overwrite", action="store_true", help="start again instead of resuming records.jsonl")
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.command == "run":
        command_run(args)
    else:
        command_summarize(args)


if __name__ == "__main__":
    main()
