"""Reference-style joint-space action term for the five-bar wheel-leg.

This mirrors the 25-dim Wheel-Legged-Gym ``wheel_legged`` task: the policy
commands **hip position targets** and **wheel velocity targets**, and an explicit
low-level PD produces the motor torques::

    hip_ref   = squat_pose + clip(action[:4]) * 0.5            [rad]
    wheel_ref = clip(action[4:6]) * 10.0                       [rad/s]

    tau_hip   = kp_hip*(hip_ref - q) - kd_hip*qdot             [Nm]
    tau_wheel = kd_wheel*(wheel_ref - wheel_vel)               [Nm]

The motors are torque driven; the PD is part of the low-level controller (in
simulation here, on the real robot in firmware).  The four passive knees stay
position-controlled closed-chain followers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from robot_model import (
    HIP_ORDER,
    KNEE_ORDER,
    WHEEL_ORDER,
    FiveBarModel,
    joint_limits,
    nominal_pose,
)


class FiveBarJointAction(ActionTerm):
    """Hip position targets + wheel velocity targets, with loop-closing knees."""

    cfg: FiveBarJointActionCfg

    def __init__(self, cfg: FiveBarJointActionCfg, env):
        super().__init__(cfg, env)
        self._robot = self._asset

        self._hip_ids, self._hip_names = self._robot.find_joints(list(HIP_ORDER), preserve_order=True)
        self._knee_ids, self._knee_names = self._robot.find_joints(list(KNEE_ORDER), preserve_order=True)
        self._wheel_ids, self._wheel_names = self._robot.find_joints(list(WHEEL_ORDER), preserve_order=True)
        if list(self._hip_names) != list(HIP_ORDER) or list(self._knee_names) != list(KNEE_ORDER):
            raise RuntimeError(
                f"joint name resolution mismatch: hips={self._hip_names} knees={self._knee_names}"
            )

        self.model = FiveBarModel(device=self.device, iters=cfg.solver_iters)
        self._num_hips = len(self._hip_ids)
        self._num_wheels = len(self._wheel_ids)
        self._action_dim = self._num_hips + self._num_wheels

        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        #: processed targets: hip offsets (rad) then wheel velocities (rad/s)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._hip_targets = torch.zeros(self.num_envs, self._num_hips, device=self.device)
        self._wheel_targets = torch.zeros(self.num_envs, self._num_wheels, device=self.device)
        self._hip_torques = torch.zeros_like(self._hip_targets)
        self._wheel_torques = torch.zeros_like(self._wheel_targets)
        self._hip_torques_applied = self._hip_torques.clone()
        self._wheel_torques_applied = self._wheel_torques.clone()

        self._nominal_hips = self.model.nominal_hips.unsqueeze(0).repeat(self.num_envs, 1)
        self._nominal_knees = self.model.nominal_knees.unsqueeze(0).repeat(self.num_envs, 1)
        self._hip_bias = torch.tensor(cfg.hip_bias, device=self.device, dtype=torch.float32).view(1, -1)
        if self._hip_bias.shape[1] != self._num_hips:
            raise ValueError(f"hip_bias must have {self._num_hips} entries, got {cfg.hip_bias}")

        self._scale = torch.tensor(
            [cfg.hip_pos_scale] * self._num_hips + [cfg.wheel_vel_scale] * self._num_wheels,
            device=self.device,
            dtype=torch.float32,
        )
        motor_lo, motor_hi = (float(v) for v in cfg.motor_torque_scale_range)
        if motor_lo <= 0.0 or motor_hi < motor_lo:
            raise ValueError(
                f"motor_torque_scale_range must satisfy 0 < lo <= hi, got {cfg.motor_torque_scale_range}"
            )
        self._motor_scale = torch.empty(
            self.num_envs, self._action_dim, device=self.device
        ).uniform_(motor_lo, motor_hi)

        limits = joint_limits()
        self._hip_lower = torch.tensor(
            [limits[n][0] + cfg.limit_margin for n in HIP_ORDER], device=self.device
        )
        self._hip_upper = torch.tensor(
            [limits[n][1] - cfg.limit_margin for n in HIP_ORDER], device=self.device
        )
        self._knee_lower = torch.tensor(
            [limits[n][0] + cfg.limit_margin for n in KNEE_ORDER], device=self.device
        )
        self._knee_upper = torch.tensor(
            [limits[n][1] - cfg.limit_margin for n in KNEE_ORDER], device=self.device
        )
        #: absolute squat pose used as the hip target at zero action
        self._squat_hips = torch.clamp(
            self._nominal_hips + self._hip_bias, self._hip_lower, self._hip_upper
        )
        self._hip_targets = self._squat_hips.clone()

        # knee pair that closes the loop at the squat pose
        self._locked_knees = self._nominal_knees[0].clone()
        self._locked_knee_residual = 0.0
        locked_hips = (self._nominal_hips[0] + self._hip_bias[0]).unsqueeze(0)
        seed5 = self._nominal_knees[0, [0, 2]].view(1, -1)
        seed6 = self._nominal_knees[0, [1, 3]].view(1, -1)
        fallback = self._nominal_knees[0].unsqueeze(0)
        knees, residual = fallback, torch.zeros(1, 2, device=self.device)
        for _ in range(12):
            knees, residual = self.model.solve_knees(locked_hips, seed5, seed6)
            seed5 = knees[:, [0, 2]].clone()
            seed6 = knees[:, [1, 3]].clone()
        knees = torch.where(torch.isfinite(knees), knees, fallback)
        self._locked_knees = torch.clamp(knees[0], self._knee_lower, self._knee_upper)
        self._locked_knee_residual = float(
            torch.nan_to_num(residual, nan=1.0, posinf=1.0, neginf=1.0).max().item()
        )

        seeds = []
        for side in ("left", "right"):
            col = nominal_pose()[side]
            seeds.append(torch.tensor([col["front_knee"], col["back_knee"]], device=self.device))
        self._knee_seed = torch.stack(seeds, dim=0).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._knee_targets = self._nominal_knees.clone()
        self.solver_residual = torch.zeros(self.num_envs, 2, device=self.device)
        self.measured_residual = torch.zeros(self.num_envs, 2, device=self.device)
        self.knee_target_clamped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._knee_clamped_total = torch.zeros((), dtype=torch.long, device=self.device)
        self._closure_cache: torch.Tensor | None = None

        self._action_prev1 = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._action_prev2 = torch.zeros(self.num_envs, self._action_dim, device=self.device)

        if cfg.vel_source not in ("sim", "position_diff"):
            raise ValueError(f"vel_source must be 'sim' or 'position_diff', got {cfg.vel_source!r}")
        if not 0.0 <= float(cfg.vel_filter_alpha) < 1.0:
            raise ValueError(f"vel_filter_alpha must be in [0, 1), got {cfg.vel_filter_alpha}")
        self._vel_source = str(cfg.vel_source)
        self._vel_filter_alpha = float(cfg.vel_filter_alpha)
        self._vel_pos_prev = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._vel_prev_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._vel_filt = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._vel_step = -1

        self._hist_buf: torch.Tensor | None = None
        self._hist_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hist_cache: torch.Tensor | None = None
        self._hist_step = -1

        delay_lo, delay_hi = (int(v) for v in cfg.action_delay_steps_range)
        if delay_lo < 0 or delay_hi < delay_lo:
            raise ValueError(
                f"action_delay_steps_range must satisfy 0 <= lo <= hi, got {cfg.action_delay_steps_range}"
            )
        self._delay_max = delay_hi
        self._delay_idx = torch.randint(delay_lo, delay_hi + 1, (self.num_envs,), device=self.device)
        self._action_fifo = torch.zeros(
            self.num_envs, delay_hi + 1, self._action_dim, device=self.device
        )

    # ------------------------------------------------------------------ #
    # properties
    # ------------------------------------------------------------------ #
    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Targets: hip offsets (rad) then wheel velocities (rad/s)."""
        return self._processed_actions

    @property
    def hip_torques(self) -> torch.Tensor:
        return self._hip_torques

    @property
    def wheel_torques(self) -> torch.Tensor:
        return self._wheel_torques

    @property
    def hip_torques_applied(self) -> torch.Tensor:
        return self._hip_torques_applied

    @property
    def wheel_torques_applied(self) -> torch.Tensor:
        return self._wheel_torques_applied

    @property
    def hip_targets(self) -> torch.Tensor:
        """Absolute hip position targets, ``(N, 4)`` rad."""
        return self._hip_targets

    @property
    def wheel_targets(self) -> torch.Tensor:
        """Wheel velocity targets, ``(N, 2)`` rad/s."""
        return self._wheel_targets

    @property
    def hip_ids(self) -> list[int]:
        return self._hip_ids

    @property
    def knee_ids(self) -> list[int]:
        return self._knee_ids

    @property
    def wheel_ids(self) -> list[int]:
        return self._wheel_ids

    @property
    def knee_targets(self) -> torch.Tensor:
        return self._knee_targets

    @property
    def nominal_hips(self) -> torch.Tensor:
        return self._nominal_hips

    @property
    def nominal_knees(self) -> torch.Tensor:
        return self._nominal_knees

    @property
    def locked_hips(self) -> torch.Tensor:
        return self._nominal_hips[0] + self._hip_bias[0]

    @property
    def reset_knees(self) -> torch.Tensor:
        return self._locked_knees

    # ------------------------------------------------------------------ #
    # action pipeline
    # ------------------------------------------------------------------ #
    def process_actions(self, actions: torch.Tensor):
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        self._action_prev2 = self._action_prev1
        self._action_prev1 = self._raw_actions.clone()
        self._raw_actions[:] = actions

        if self._delay_max > 0:
            self._action_fifo = torch.cat(
                (self._raw_actions.unsqueeze(1), self._action_fifo[:, :-1, :]), dim=1
            )
            delayed = self._action_fifo[
                torch.arange(self.num_envs, device=self.device), self._delay_idx
            ]
        else:
            delayed = self._raw_actions
        self._processed_actions = delayed * self._scale

        hip_offset = self._processed_actions[:, : self._num_hips]
        self._hip_targets = torch.clamp(
            self._squat_hips + hip_offset, self._hip_lower, self._hip_upper
        )
        self._wheel_targets = self._processed_actions[:, self._num_hips :]

        q_hip = self._robot.data.joint_pos[:, self._hip_ids]
        qd_hip = self._robot.data.joint_vel[:, self._hip_ids]
        self._hip_torques = (
            float(self.cfg.hip_kp) * (self._hip_targets - q_hip)
            - float(self.cfg.hip_kd) * qd_hip
        )
        qd_wheel = self._robot.data.joint_vel[:, self._wheel_ids]
        self._wheel_torques = float(self.cfg.wheel_kd) * (
            self._wheel_targets - qd_wheel
        )

        hip_tau = torch.clamp(
            self._hip_torques,
            -float(self.cfg.hip_effort_limit),
            float(self.cfg.hip_effort_limit),
        )
        wheel_tau = torch.clamp(
            self._wheel_torques,
            -float(self.cfg.wheel_effort_limit),
            float(self.cfg.wheel_effort_limit),
        )
        self._hip_torques_applied = torch.clamp(
            hip_tau * self._motor_scale[:, : self._num_hips],
            -float(self.cfg.hip_effort_limit),
            float(self.cfg.hip_effort_limit),
        )
        self._wheel_torques_applied = torch.clamp(
            wheel_tau * self._motor_scale[:, self._num_hips :],
            -float(self.cfg.wheel_effort_limit),
            float(self.cfg.wheel_effort_limit),
        )
        self._hip_torques = hip_tau
        self._wheel_torques = wheel_tau

    def action_jerk(self) -> torch.Tensor:
        return self._raw_actions - 2.0 * self._action_prev1 + self._action_prev2

    def apply_actions(self):
        self._robot.set_joint_effort_target(self._hip_torques_applied, joint_ids=self._hip_ids)
        self._robot.set_joint_effort_target(self._wheel_torques_applied, joint_ids=self._wheel_ids)

    # ------------------------------------------------------------------ #
    # test helpers
    # ------------------------------------------------------------------ #
    def set_action_delay(self, steps: int = 0) -> None:
        steps = int(steps)
        if steps < 0 or steps > self._delay_max:
            raise ValueError(f"action delay {steps} outside [0, {self._delay_max}]")
        self._delay_idx[:] = steps
        self._action_fifo.zero_()

    def set_motor_torque_scale(self, scale: float = 1.0) -> None:
        self._motor_scale.fill_(float(scale))

    # ------------------------------------------------------------------ #
    # observation history / measured velocity
    # ------------------------------------------------------------------ #
    def push_history(
        self,
        frame: torch.Tensor,
        length: int = 5,
        noise_vec: tuple[float, ...] | None = None,
    ) -> torch.Tensor:
        step = int(getattr(self._env, "common_step_counter", -1))
        if step == self._hist_step and self._hist_cache is not None:
            return self._hist_cache

        frame = torch.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
        if noise_vec is not None:
            nv = torch.as_tensor(noise_vec, device=self.device, dtype=frame.dtype)
            frame = frame + (2.0 * torch.rand_like(frame) - 1.0) * nv

        length = int(length)
        if (
            self._hist_buf is None
            or self._hist_buf.shape[1] != length
            or self._hist_buf.shape[2] != frame.shape[1]
        ):
            self._hist_buf = torch.zeros(
                self.num_envs, length, frame.shape[1], device=self.device, dtype=frame.dtype
            )
            self._hist_valid[:] = False

        invalid = ~self._hist_valid
        if bool(invalid.any()):
            self._hist_buf[invalid] = frame[invalid].unsqueeze(1).expand(-1, length, -1)
            self._hist_valid[invalid] = True

        self._hist_buf = torch.cat((self._hist_buf[:, 1:], frame.unsqueeze(1)), dim=1)
        self._hist_cache = self._hist_buf.reshape(self.num_envs, -1)
        self._hist_step = step
        return self._hist_cache

    def measured_joint_vel(self) -> torch.Tensor:
        step = int(getattr(self._env, "common_step_counter", -1))
        if step == self._vel_step:
            return self._vel_filt

        if self._vel_source == "sim":
            vel = torch.cat(
                (
                    self._robot.data.joint_vel[:, self._hip_ids],
                    self._robot.data.joint_vel[:, self._wheel_ids],
                ),
                dim=-1,
            )
        else:
            pos = torch.cat(
                (
                    self._robot.data.joint_pos[:, self._hip_ids],
                    self._robot.data.joint_pos[:, self._wheel_ids],
                ),
                dim=-1,
            )
            dt = float(self._env.step_dt)
            delta = pos - self._vel_pos_prev
            delta = torch.remainder(delta + math.pi, 2.0 * math.pi) - math.pi
            diff = delta / dt
            diff = torch.where(
                self._vel_prev_valid.unsqueeze(-1), diff, torch.zeros_like(diff)
            )
            diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
            self._vel_pos_prev = pos.clone()
            self._vel_prev_valid[:] = True
            alpha = self._vel_filter_alpha
            self._vel_filt = alpha * self._vel_filt + (1.0 - alpha) * diff
            vel = self._vel_filt

        self._vel_step = step
        return vel

    def measured_leg_state(self) -> torch.Tensor:
        hip_pos = self._robot.data.joint_pos[:, self._hip_ids]
        knee_pos = self._robot.data.joint_pos[:, self._knee_ids]
        return self.model.measured_leg_state(hip_pos, knee_pos)

    # ------------------------------------------------------------------ #
    # loop closure
    # ------------------------------------------------------------------ #
    def update_knee_targets(self) -> torch.Tensor:
        hip_pos = self._robot.data.joint_pos[:, self._hip_ids]
        knees, residual = self.model.solve_knees(
            hip_pos, self._knee_seed[:, 0, :], self._knee_seed[:, 1, :]
        )
        finite = torch.isfinite(knees).all(dim=-1, keepdim=True)
        knees = torch.where(finite, knees, self._nominal_knees)
        clamped = torch.clamp(knees, self._knee_lower, self._knee_upper)
        self.knee_target_clamped = (clamped - knees).abs() > 1e-9
        self._knee_clamped_total += self.knee_target_clamped.sum()
        self._knee_targets = clamped
        self.solver_residual = torch.nan_to_num(residual, nan=0.0, posinf=1.0, neginf=1.0)
        self._closure_cache = None
        self._knee_seed = torch.stack((clamped[:, 0:2], clamped[:, 2:4]), dim=1).detach()
        return self._knee_targets

    def measured_closure_norm(self, reuse_cache: bool = False) -> torch.Tensor:
        if reuse_cache and self._closure_cache is not None:
            return self._closure_cache
        hip_pos = self._robot.data.joint_pos[:, self._hip_ids]
        knee_pos = self._robot.data.joint_pos[:, self._knee_ids]
        self.measured_residual = self.model.measured_closure(hip_pos, knee_pos)
        self._closure_cache = self.measured_residual
        return self.measured_residual

    @property
    def knee_clamped_total(self) -> int:
        return int(self._knee_clamped_total.item())

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._action_fifo[env_ids] = 0.0
        self._hip_targets[env_ids] = self._squat_hips[env_ids]
        self._wheel_targets[env_ids] = 0.0
        self._hip_torques[env_ids] = 0.0
        self._wheel_torques[env_ids] = 0.0
        self._hip_torques_applied[env_ids] = 0.0
        self._wheel_torques_applied[env_ids] = 0.0
        self._action_prev1[env_ids] = 0.0
        self._action_prev2[env_ids] = 0.0
        self._vel_prev_valid[env_ids] = False
        self._vel_filt[env_ids] = 0.0
        self._vel_step = -1
        self._hist_valid[env_ids] = False
        self._hist_cache = None
        self._hist_step = -1
        if isinstance(env_ids, slice):
            self._knee_seed[:] = self.model.nominal_knees.view(1, 2, 2)
            self._knee_targets[:] = self._nominal_knees
        else:
            self._knee_seed[env_ids] = self.model.nominal_knees.view(1, 2, 2)
            self._knee_targets[env_ids] = self._nominal_knees[env_ids]


@configclass
class FiveBarJointActionCfg(ActionTermCfg):
    """Configuration for the reference-style joint-space action."""

    class_type: type[ActionTerm] = FiveBarJointAction
    hip_bias: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    #: rad of hip position target per action unit
    hip_pos_scale: float = 0.5
    #: rad/s of wheel velocity target per action unit
    wheel_vel_scale: float = 10.0
    #: low-level PD gains (reference: 40/1 for hips, 0.5 for wheels)
    hip_kp: float = 40.0
    hip_kd: float = 1.0
    wheel_kd: float = 0.5
    hip_effort_limit: float = 40.0
    wheel_effort_limit: float = 3.69
    motor_torque_scale_range: tuple[float, float] = (1.0, 1.0)
    action_delay_steps_range: tuple[int, int] = (0, 0)
    vel_source: str = "position_diff"
    vel_filter_alpha: float = 0.0
    limit_margin: float = 0.01
    solver_iters: int = 2


__all__ = ["FiveBarJointAction", "FiveBarJointActionCfg"]
