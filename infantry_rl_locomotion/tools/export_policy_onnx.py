#!/usr/bin/env python
"""Export an rsl_rl policy checkpoint to ONNX for STM32Cube.AI / X-CUBE-AI.

The checkpoint stores the actor MLP *and* its observation normaliser separately::

    actor_state_dict:
        obs_normalizer._mean (1, obs_dim)
        obs_normalizer._std  (1, obs_dim)
        mlp.0.weight / mlp.0.bias ... mlp.<2k>.weight / bias
        distribution.std_param (action_dim,)      # sampling only, not exported

Deployment (deterministic inference) is ``mlp((obs - mean) / (std + eps))`` with
``eps = 1e-2`` (rsl_rl ``EmpiricalNormalization``).  By default the normaliser is
FOLDED into the first Linear layer, so the ONNX graph contains only
``MatMul/Add/Elu`` -- the easiest form for Cube.AI -- and is numerically identical.

The ONNX output is the raw policy mean.  It does NOT include the environment's
post-processing:

    squat task:  wheel_target = 5.0 * clip(action, -12, 12)   [rad/s]
                 hips are a constant locked pose
                 knees are NOT in the policy: solve the five-bar closure at 50 Hz
                 obs scaling: base_ang_vel * 0.25, wheel_vel * 0.1
                 control rate: 50 Hz (decimation 4 x 200 Hz physics)

Examples::

    C:/Users/32560/.conda/envs/isaaclab/python.exe tools/export_policy_onnx.py ^
      --checkpoint logs/rsl_rl/infantry_rl_squat/2026-11-03_23-02-16_chassis_squat_s1_cont/model_598.pt ^
      --output exports/chassis_squat_model_598.onnx

    # keep the normaliser as explicit Sub/Div ops (for inspection)
    ... --no-fold-normalizer --output policy_raw.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


def infer_mlp(state_dict: dict) -> tuple[int, int, list[int]]:
    """Infer (obs_dim, action_dim, hidden_dims) from ``mlp.<k>.weight`` keys."""
    weights = []
    for key, value in state_dict.items():
        if key.startswith("mlp.") and key.endswith(".weight"):
            weights.append((int(key.split(".")[1]), value))
    if not weights:
        raise KeyError("no 'mlp.*.weight' tensors found in actor_state_dict")
    weights.sort(key=lambda item: item[0])
    shapes = [tuple(w.shape) for _, w in weights]
    obs_dim = shapes[0][1]
    action_dim = shapes[-1][0]
    hidden_dims = [shape[0] for shape in shapes[:-1]]
    return obs_dim, action_dim, hidden_dims


def build_mlp(obs_dim: int, action_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
    act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers: list[nn.Module] = []
    dims = [obs_dim, *hidden_dims, action_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def load_mlp_weights(mlp: nn.Sequential, state_dict: dict) -> None:
    """Copy ``mlp.<layer>.weight/bias`` into the Sequential's Linear layers."""
    linears = [m for m in mlp if isinstance(m, nn.Linear)]
    for index, linear in enumerate(linears):
        linear.weight.data.copy_(state_dict[f"mlp.{2 * index}.weight"])
        linear.bias.data.copy_(state_dict[f"mlp.{2 * index}.bias"])


def fold_normalizer(mlp: nn.Sequential, mean: torch.Tensor, denom: torch.Tensor) -> None:
    """Fold ``(x - mean) / denom`` into the first Linear layer.

    y     = (x - mean) / denom
    h     = W y + b = (W / denom) x + b - W (mean / denom)
    """
    first = next(m for m in mlp if isinstance(m, nn.Linear))
    scale = 1.0 / denom  # (1, obs_dim)
    with torch.no_grad():
        new_w = first.weight.data * scale
        new_b = first.bias.data - first.weight.data @ (mean.squeeze(0) * scale.squeeze(0))
        first.weight.data.copy_(new_w)
        first.bias.data.copy_(new_b)


class Policy(nn.Module):
    """Normaliser (optional) + actor MLP; output is the deterministic action mean."""

    def __init__(
        self,
        mlp: nn.Sequential,
        mean: torch.Tensor | None = None,
        denom: torch.Tensor | None = None,
        folded: bool = False,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.obs_dim = mlp[0].in_features
        #: True when the normaliser has been folded into ``mlp[0]``: skip the
        #: (x - 0) / 1 ops entirely so the ONNX graph is only Gemm/Elu.
        self.folded = folded
        self.register_buffer("_mean", mean if mean is not None else torch.zeros(1, self.obs_dim))
        self.register_buffer("_denom", denom if denom is not None else torch.ones(1, self.obs_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.folded:
            return self.mlp(obs)
        return self.mlp((obs - self._mean) / self._denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <checkpoint dir>/policy_<iter>.onnx")
    parser.add_argument("--activation", choices=("elu", "relu", "tanh"), default="elu")
    parser.add_argument("--eps", type=float, default=1e-2,
                        help="rsl_rl EmpiricalNormalization eps (default 1e-2)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-fold-normalizer", action="store_true",
                        help="keep Sub/Div ops instead of folding them into layer 0")
    parser.add_argument("--skip-check", action="store_true",
                        help="skip the onnx.checker + ReferenceEvaluator comparison")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor_sd = ckpt.get("actor_state_dict")
    if actor_sd is None:
        raise KeyError(f"{args.checkpoint} has no 'actor_state_dict' (keys: {list(ckpt)})")
    iteration = int(ckpt.get("iter", -1))

    obs_dim, action_dim, hidden_dims = infer_mlp(actor_sd)
    mean = actor_sd["obs_normalizer._mean"].float().cpu()
    std = actor_sd["obs_normalizer._std"].float().cpu()
    denom = std + float(args.eps)
    print(f"[onnx] checkpoint     : {args.checkpoint}")
    print(f"[onnx] iter           : {iteration}")
    print(f"[onnx] actor MLP      : {obs_dim} -> {hidden_dims} -> {action_dim} ({args.activation})")
    print(f"[onnx] normalizer     : mean/std from checkpoint, eps={args.eps}")

    # -- reference model (normaliser kept explicit) -------------------------- #
    reference = Policy(
        build_mlp(obs_dim, action_dim, hidden_dims, args.activation),
        mean.clone(),
        denom.clone(),
    )
    load_mlp_weights(reference.mlp, actor_sd)
    reference.eval()

    # -- exported model ------------------------------------------------------ #
    fold = not args.no_fold_normalizer
    exported = Policy(
        build_mlp(obs_dim, action_dim, hidden_dims, args.activation),
        mean.clone(),
        denom.clone(),
        folded=fold,
    )
    load_mlp_weights(exported.mlp, actor_sd)
    if fold:
        fold_normalizer(exported.mlp, mean, denom)
        print("[onnx] normalizer      : folded into mlp.0 (graph is Gemm/Elu only)")
    else:
        print("[onnx] normalizer      : kept as explicit Sub/Div ops")
    exported.eval()

    output = args.output
    if output is None:
        output = args.checkpoint.parent / f"policy_{iteration}.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, obs_dim)
    with torch.no_grad():
        torch.onnx.export(
            exported,
            dummy,
            str(output),
            input_names=["obs"],
            output_names=["action"],
            opset_version=int(args.opset),
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"[onnx] wrote          : {output} ({output.stat().st_size / 1024:.1f} KiB)")

    # -- validation ---------------------------------------------------------- #
    if not args.skip_check:
        import numpy as np
        import onnx
        from onnx.reference import ReferenceEvaluator

        model = onnx.load(str(output))
        onnx.checker.check_model(model)
        ops = sorted({node.op_type for node in model.graph.node})
        print(f"[onnx] checker         : OK; ops = {ops}")

        torch.manual_seed(0)
        x = torch.randn(64, obs_dim)
        with torch.no_grad():
            y_ref = reference(x)
            y_pt = exported(x)
        max_diff_pt = float((y_ref - y_pt).abs().max())
        session = ReferenceEvaluator(model)
        y_onnx = torch.from_numpy(np.asarray(session.run(None, {"obs": x.numpy()})[0]))
        max_diff_onnx = float((y_ref - y_onnx).abs().max())
        print(f"[onnx] torch ref vs folded : max |diff| = {max_diff_pt:.3e}")
        print(f"[onnx] torch ref vs ONNX   : max |diff| = {max_diff_onnx:.3e}")
        if max_diff_pt > 1e-4 or max_diff_onnx > 1e-4:
            raise RuntimeError("exported model does not match the checkpoint")

    # -- deployment sidecar -------------------------------------------------- #
    meta = {
        "checkpoint": str(args.checkpoint),
        "iteration": iteration,
        "onnx": str(output),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dims": hidden_dims,
        "activation": args.activation,
        "normalizer_folded": not args.no_fold_normalizer,
        "normalizer_eps": args.eps,
        "opset": args.opset,
        "input_names": ["obs"],
        "output_names": ["action"],
        "output_is": "deterministic policy mean (raw action, before clip/scale)",
        "squat_postprocess": {
            "clip_actions": 12.0,
            "wheel_scale": 5.0,
            "wheel_target_rad_s": "5.0 * clip(action, -12, 12)",
            "hips": "constant locked pose",
            "knees": "five-bar closure solved at 50 Hz, not in ONNX",
            "control_rate_hz": 50.0,
            "obs_scales": {"base_ang_vel": 0.25, "wheel_vel": 0.1},
        },
    }
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[onnx] metadata       : {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
