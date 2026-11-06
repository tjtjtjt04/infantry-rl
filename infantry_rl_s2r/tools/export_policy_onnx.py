#!/usr/bin/env python
"""Export the history-encoder torque policy to ONNX for STM32Cube.AI.

Checkpoint layout (rsl_rl 5.x)::

    actor_state_dict:
        obs_normalizer._mean / _std   (1, 125)
        encoder.<2i>.weight / bias   125 -> 128 -> 64 -> 3 (latent)
        mlp.<2i>.weight / bias       28 (= 25 current + 3 latent) -> ... -> 6
        distribution.std_param       (sampling only, not exported)

The ONNX graph computes::

    x       = (history - mean) / (std + eps)          # explicit Sub/Div
    current = x[..., -25:]                            # o_t
    latent  = encoder(x)                              # 3
    action  = actor([current, latent])                # 28 -> 6

``history`` is the flattened 5-frame observation, oldest to newest::

    [o_{t-4}, o_{t-3}, o_{t-2}, o_{t-1}, o_t]         # 5 * 25 = 125

with one frame being::

    [base_ang_vel(3)*0.25, projected_gravity(3), command(3),
     hip_pos_rel_locked(4), joint_vel(4 hips + 2 wheels)(6)*0.05, last_action(6)]

The ONNX output is the raw target mean (4 hip position offsets + 2 wheel
velocity targets).  Deployment post-processing::

    q_ref   = squat_hips + clip(action[:4], -4, 4) * 0.5     [rad]
    v_ref   = clip(action[4:6], -4, 4) * 10.0                [rad/s]
    tau_hip = clamp(kp*(q_ref - q) - kd*qdot, -40, 40)       [Nm]
    tau_w   = clamp(0.5*(v_ref - v), -3.69, 3.69)            [Nm]

Examples::

    C:/Users/32560/.conda/envs/isaaclab/python.exe tools/export_policy_onnx.py ^
      --checkpoint logs/rsl_rl/infantry_rl_squat_torque/<run>/model_600.pt ^
      --output exports/chassis_squat_torque.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


def _widths(state_dict: dict, prefix: str) -> list[tuple[int, torch.Tensor]]:
    entries = []
    for key, value in state_dict.items():
        if key.startswith(f"{prefix}.") and key.endswith(".weight"):
            entries.append((int(key.split(".")[1]), value))
    entries.sort(key=lambda item: item[0])
    return entries


def infer_mlp(state_dict: dict, prefix: str = "mlp") -> tuple[int, int, list[int]]:
    """Infer ``(input_dim, output_dim, hidden_dims)`` for one MLP."""
    entries = _widths(state_dict, prefix)
    if not entries:
        raise KeyError(f"no '{prefix}.*.weight' tensors found in actor_state_dict")
    shapes = [tuple(w.shape) for _, w in entries]
    return int(shapes[0][1]), int(shapes[-1][0]), [int(s[0]) for s in shapes[:-1]]


def build_mlp(input_dim: int, output_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
    act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers: list[nn.Module] = []
    dims = [input_dim, *hidden_dims, output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def load_mlp_weights(mlp: nn.Sequential, state_dict: dict, prefix: str = "mlp") -> None:
    """Copy ``<prefix>.<2i>.weight/bias`` into the Sequential's Linear layers."""
    linears = [m for m in mlp if isinstance(m, nn.Linear)]
    for index, linear in enumerate(linears):
        linear.weight.data.copy_(state_dict[f"{prefix}.{2 * index}.weight"])
        linear.bias.data.copy_(state_dict[f"{prefix}.{2 * index}.bias"])


class Policy(nn.Module):
    """History normaliser + encoder + actor; output is the deterministic mean."""

    def __init__(
        self,
        encoder: nn.Sequential,
        actor: nn.Sequential,
        mean: torch.Tensor,
        denom: torch.Tensor,
        current_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.current_dim = int(current_dim)
        self.register_buffer("_mean", mean)
        self.register_buffer("_denom", denom)
        self.obs_dim = int(mean.shape[-1])
        if self.obs_dim % self.current_dim != 0:
            raise ValueError(
                f"history dim {self.obs_dim} is not a multiple of current_dim {self.current_dim}"
            )
        self.history_length = self.obs_dim // self.current_dim

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = (history - self._mean) / self._denom
        current = x[..., -self.current_dim :]
        latent = self.encoder(x)
        return self.actor(torch.cat((current, latent), dim=-1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <checkpoint dir>/policy_<iter>.onnx")
    parser.add_argument("--activation", choices=("elu", "relu", "tanh"), default="elu")
    parser.add_argument("--eps", type=float, default=1e-2,
                        help="rsl_rl EmpiricalNormalization eps (default 1e-2)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--clip-actions", type=float, default=4.0)
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

    # -- architecture ------------------------------------------------------ #
    mean = actor_sd["obs_normalizer._mean"].float().cpu()
    std = actor_sd["obs_normalizer._std"].float().cpu()
    obs_dim = int(mean.shape[-1])
    denom = std + float(args.eps)

    actor_in, action_dim, hidden_dims = infer_mlp(actor_sd, "mlp")
    if action_dim != 6:
        raise ValueError(f"the torque policy must output 6 dims, got {action_dim}")
    encoder_in, latent_dim, encoder_hidden_dims = infer_mlp(actor_sd, "encoder")
    if encoder_in != obs_dim:
        raise ValueError(f"encoder input {encoder_in} != observation dim {obs_dim}")
    current_dim = actor_in - latent_dim
    if current_dim <= 0 or obs_dim % current_dim != 0:
        raise ValueError(
            f"cannot split actor input {actor_in} into current + latent({latent_dim}) "
            f"with history dim {obs_dim}"
        )
    history_length = obs_dim // current_dim
    frame_dim = current_dim
    if history_length != 5:
        print(f"[onnx] WARNING: inferred history_length={history_length}, expected 5")

    clip_actions = float(args.clip_actions)
    print(f"[onnx] checkpoint      : {args.checkpoint}")
    print(f"[onnx] iter            : {iteration}")
    print(f"[onnx] history         : {obs_dim} = {history_length} x {frame_dim}")
    print(f"[onnx] encoder         : {obs_dim} -> {encoder_hidden_dims} -> {latent_dim}")
    print(f"[onnx] actor           : {current_dim}+{latent_dim} = {actor_in} -> {hidden_dims} -> {action_dim}")
    print(f"[onnx] normalizer      : mean/std from checkpoint, eps={args.eps} (explicit Sub/Div)")

    encoder = build_mlp(obs_dim, latent_dim, encoder_hidden_dims, args.activation)
    actor = build_mlp(actor_in, action_dim, hidden_dims, args.activation)
    load_mlp_weights(encoder, actor_sd, "encoder")
    load_mlp_weights(actor, actor_sd, "mlp")
    exported = Policy(encoder, actor, mean.clone(), denom.clone(), current_dim)
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
            input_names=["history"],
            output_names=["action"],
            opset_version=int(args.opset),
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"[onnx] wrote           : {output} ({output.stat().st_size / 1024:.1f} KiB)")

    # -- validation -------------------------------------------------------- #
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
            y_pt = exported(x)
        session = ReferenceEvaluator(model)
        y_onnx = torch.from_numpy(np.asarray(session.run(None, {"history": x.numpy()})[0]))
        max_diff = float((y_pt - y_onnx).abs().max())
        print(f"[onnx] torch vs ONNX   : max |diff| = {max_diff:.3e}")
        if max_diff > 1e-4:
            raise RuntimeError("exported model does not match the checkpoint")

    # -- deployment sidecar ------------------------------------------------ #
    meta = {
        "checkpoint": str(args.checkpoint),
        "iteration": iteration,
        "onnx": str(output),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "history_length": history_length,
        "frame_dim": frame_dim,
        "current_dim": current_dim,
        "latent_dim": latent_dim,
        "actor_input": actor_in,
        "encoder_hidden_dims": encoder_hidden_dims,
        "hidden_dims": hidden_dims,
        "activation": args.activation,
        "normalizer_folded": False,
        "normalizer_eps": args.eps,
        "opset": args.opset,
        "input_names": ["history"],
        "output_names": ["action"],
        "output_is": "deterministic policy mean (raw action, before clip/scale)",
        "history_layout": (
            "oldest to newest: [o_{t-4}, o_{t-3}, o_{t-2}, o_{t-1}, o_t], "
            f"each frame {frame_dim} dims"
        ),
        "obs_layout": [
            {"name": "base_ang_vel", "slice": [0, 3], "scale": 0.25},
            {"name": "projected_gravity", "slice": [3, 6], "scale": 1.0},
            {"name": "command", "slice": [6, 9], "scale": 1.0},
            {"name": "hip_pos_rel_locked", "slice": [9, 13], "scale": 1.0},
            {"name": "joint_vel_hips_wheels", "slice": [13, 19], "scale": 0.1},
            {"name": "last_action", "slice": [19, 25], "scale": 1.0},
        ],
        "action_layout": {
            "0:4": "hip position target offset [rad]",
            "4:6": "wheel velocity target [rad/s]",
        },
        "action_postprocess": {
            "hip_position_target_rad": (
                f"squat_hips + clip(action[:4], {-clip_actions:g}, {clip_actions:g}) * 0.5"
            ),
            "wheel_velocity_target_rad_s": (
                f"clip(action[4:6], {-clip_actions:g}, {clip_actions:g}) * 10.0"
            ),
            "low_level_pd": {
                "hip_kp": 40.0,
                "hip_kd": 1.0,
                "wheel_kd": 0.5,
                "tau_hip": "kp*(q_ref - q) - kd*qdot  [Nm], clamp ±40",
                "tau_wheel": "wheel_kd*(v_ref - v)  [Nm], clamp ±3.69",
            },
            "hips": "position target -> PD -> torque",
            "wheels": "velocity target -> damping -> torque",
        },
        "knees": "position-controlled five-bar closure at 50 Hz, not in ONNX",
        "control_rate_hz": 100.0,
        "obs_scales": {
            "base_ang_vel": 0.25,
            "joint_vel_hips_wheels": 0.1,
        },
    }
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[onnx] metadata        : {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
