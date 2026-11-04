"""Training-only teacher for the pure-torque squat.

A pure-torque hip has no PD to catch the initial fall, and pure-torque wheels have
no velocity servo, so a from-scratch run starts with ~0.4 s episodes and PPO never
sees enough of the balance problem.  The existing position/velocity squat policy
(``--task squat``) already balances on the same chassis; this module runs that
checkpoint as a **wheel-velocity teacher** and the action term converts its
velocity targets into a wheel torque with the same damping law the original
actuator used::

    v* = 5.0 * clip(teacher_action, -6, 6)
    tau_teacher = kd * (v* - wheel_vel)

The teacher torque is decayed to zero together with the hip PD assist, so the
final policy is pure torque.  It is never used during play.
"""

from __future__ import annotations

import torch
from torch import nn

from .observations import heading_sin_cos, position_error


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [input_dim, *hidden_dims, output_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class SquatTeacherPolicy:
    """Deterministic actor from a ``--task squat`` rsl_rl checkpoint."""

    def __init__(self, checkpoint: str, num_envs: int, device: torch.device | str) -> None:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("actor_state_dict")
        if sd is None:
            raise KeyError(f"{checkpoint} has no actor_state_dict")
        weights = sorted(
            (int(k.split(".")[1]), v)
            for k, v in sd.items()
            if k.startswith("mlp.") and k.endswith(".weight")
        )
        shapes = [tuple(w.shape) for _, w in weights]
        input_dim = shapes[0][1]
        output_dim = shapes[-1][0]
        hidden_dims = [s[0] for s in shapes[:-1]]
        if input_dim != 30 or output_dim != 2:
            raise ValueError(
                f"expected a 30-D squat teacher with 2 wheel actions, got {input_dim}->{output_dim}"
            )
        self.device = torch.device(device)
        self.mlp = _build_mlp(input_dim, output_dim, hidden_dims).to(self.device)
        linears = [m for m in self.mlp if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linears):
            layer.weight.data.copy_(sd[f"mlp.{2 * i}.weight"].to(self.device))
            layer.bias.data.copy_(sd[f"mlp.{2 * i}.bias"].to(self.device))
        self.mlp.eval()
        self.mean = sd["obs_normalizer._mean"].to(self.device)
        self.denom = sd["obs_normalizer._std"].to(self.device) + 1e-2
        self.prev_action = torch.zeros(num_envs, 2, device=self.device)

    def reset(self, env_ids) -> None:
        self.prev_action[env_ids] = 0.0

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        a = self.mlp((obs - self.mean) / self.denom)
        self.prev_action[:] = a
        return a


def build_squat_teacher_obs(env, term, base_height_target: float) -> torch.Tensor:
    """The 30-D observation the ``--task squat`` teacher was trained on.

    Order matches ``env_squat.SquatObservationsCfg`` exactly.
    """
    robot = term._robot
    roll_terms = [
        robot.data.projected_gravity_b,
        robot.data.root_ang_vel_b * 0.25,
        robot.data.root_lin_vel_b,
        heading_sin_cos(env, target_heading=0.0),
        position_error(env, target_x=0.0, target_y=0.0),
        robot.data.joint_pos[:, term.hip_ids] - term.nominal_hips,
        robot.data.joint_pos[:, term.knee_ids] - term.knee_targets,
        term.solved_leg_state(),
        robot.data.joint_vel[:, term.wheel_ids] * 0.1,
        term._teacher.prev_action,
        (robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - base_height_target).unsqueeze(-1),
    ]
    return torch.cat(roll_terms, dim=-1)
