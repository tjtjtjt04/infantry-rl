"""Action terms for the infantry-rl five-bar balance task.

PhysX does **not** enforce the closed chain of this robot (the articulation has 10
joints; ``loop_joints/*_calf_axle`` are not DOFs), so the two knees of each leg are
passive followers whose position targets are recomputed from the verified planar
loop solver on every control step, warm-started from the previous solution.  That
duty lives in :class:`FiveBarJointAction` so that the solved values, the solver
residual and the *measured* closure error are all reachable for logging, rewards
and the smoke test.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..model import (
    HIP_ORDER,
    KNEE_ORDER,
    WHEEL_ORDER,
    FiveBarModel,
    joint_limits,
    nominal_pose,
)


class FiveBarJointAction(ActionTerm):
    """Hip position offsets + wheel velocity targets, with loop-closing knees.

    Action vector (6 dims)::

        [0:4] hip position offset from the nominal stance, in :data:`HIP_ORDER`
        [4:6] wheel velocity target (rad/s), in :data:`WHEEL_ORDER`

    The four knees are *not* part of the action: they are solved from the measured
    hip angles at every control step (see :meth:`update_knee_targets`).
    """

    cfg: FiveBarJointActionCfg

    def __init__(self, cfg: FiveBarJointActionCfg, env):
        super().__init__(cfg, env)
        self._robot = self._asset  # :class:`Articulation`

        # resolve every joint by NAME (the articulation's own order is never assumed)
        self._hip_ids, self._hip_names = self._robot.find_joints(list(HIP_ORDER), preserve_order=True)
        self._knee_ids, self._knee_names = self._robot.find_joints(list(KNEE_ORDER), preserve_order=True)
        self._wheel_ids, self._wheel_names = self._robot.find_joints(list(WHEEL_ORDER), preserve_order=True)
        if list(self._hip_names) != list(HIP_ORDER) or list(self._knee_names) != list(KNEE_ORDER):
            raise RuntimeError(
                f"joint name resolution mismatch: hips={self._hip_names} knees={self._knee_names}"
            )

        # batched model + solvers, on the environment device
        self.model = FiveBarModel(device=self.device, iters=cfg.solver_iters)

        self._num_hips = len(self._hip_ids)
        self._num_wheels = len(self._wheel_ids)
        self._action_dim = self._num_hips + self._num_wheels

        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._hip_targets = torch.zeros(self.num_envs, self._num_hips, device=self.device)
        self._wheel_targets = torch.zeros(self.num_envs, self._num_wheels, device=self.device)

        # nominal stance (broadcast over environments)
        self._nominal_hips = self.model.nominal_hips.unsqueeze(0).repeat(self.num_envs, 1)
        self._nominal_knees = self.model.nominal_knees.unsqueeze(0).repeat(self.num_envs, 1)
        self._scale = torch.tensor(
            [cfg.hip_scale] * self._num_hips + [cfg.wheel_scale] * self._num_wheels,
            device=self.device,
            dtype=torch.float32,
        )

        # joint safety envelope: keep every command inside the measured feasible set
        limits = joint_limits()
        self._hip_lower = torch.tensor([limits[n][0] + cfg.limit_margin for n in HIP_ORDER], device=self.device)
        self._hip_upper = torch.tensor([limits[n][1] - cfg.limit_margin for n in HIP_ORDER], device=self.device)
        self._hip_lower = torch.maximum(self._hip_lower, self._nominal_hips[0] - cfg.hip_range)
        self._hip_upper = torch.minimum(self._hip_upper, self._nominal_hips[0] + cfg.hip_range)
        self._knee_lower = torch.tensor([limits[n][0] + cfg.limit_margin for n in KNEE_ORDER], device=self.device)
        self._knee_upper = torch.tensor([limits[n][1] - cfg.limit_margin for n in KNEE_ORDER], device=self.device)

        # loop-closure state
        seeds = []
        for side in ("left", "right"):
            col = nominal_pose()[side]
            seeds.append(torch.tensor([col["front_knee"], col["back_knee"]], device=self.device))
        self._knee_seed = torch.stack(seeds, dim=0).unsqueeze(0).repeat(self.num_envs, 1, 1)  # (N, side, 2)
        self._knee_targets = self._nominal_knees.clone()
        #: last differential wheel correction applied by the optional heading loop
        self.heading_delta = torch.zeros(self.num_envs, device=self.device)
        self.solver_residual = torch.zeros(self.num_envs, 2, device=self.device)
        self.measured_residual = torch.zeros(self.num_envs, 2, device=self.device)
        self.knee_target_clamped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # NOTE: kept as a GPU scalar on purpose.  Accumulating it with
        # ``int(...sum().item())`` forces a CUDA->CPU sync EVERY control step, which
        # stalls the pipeline (the number is only ever read by the smoke test).
        self._knee_clamped_total = torch.zeros((), dtype=torch.long, device=self.device)
        #: cache for :meth:`measured_closure_norm`, invalidated once per control step
        self._closure_cache: torch.Tensor | None = None

        # --- leg-rate estimation (dL0, dphi) -----------------------------
        # The firmware LQR state is [theta, dtheta, xb, dxb, phi, dphi], i.e. the leg
        # angle RATE is a first-class control signal (it is what damped the phi loop;
        # the Tp row of K has a gain of ~5.9 on dphi).  Our MLP policy has no memory,
        # so unless dL0/dphi are fed in explicitly the legs can only use position
        # feedback and will lag / oscillate when they start carrying the attitude loop.
        #
        # These are obtained by finite-differencing the MEASURED (L0, phi) over one
        # control step -- one extra FK evaluation per step.  Computing the analytic
        # Jacobian J = d(L0, phi)/d(q1, q4) instead would cost 4 more Newton solves per
        # step, and this environment is already launch-bound on the CPU.
        #: previous measured (L0_l, phi_l, L0_r, phi_r)
        self._leg_state_prev = torch.zeros(self.num_envs, 4, device=self.device)
        #: which envs have a valid previous sample (cleared on reset to avoid the
        #: huge fake rate that a reset pose jump would otherwise produce)
        self._leg_prev_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        #: last estimated (dL0_l, dphi_l, dL0_r, dphi_r)
        self.leg_rate = torch.zeros(self.num_envs, 4, device=self.device)

    # ------------------------------------------------------------------ #
    # properties required by ActionTerm
    # ------------------------------------------------------------------ #
    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """The 6-dim action exactly as sent by the policy (before scaling)."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Scaled action: hip offsets (rad) followed by wheel velocity targets (rad/s)."""
        return self._processed_actions

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
        """Current solved knee position targets, ``(N, 4)`` in :data:`KNEE_ORDER`."""
        return self._knee_targets

    @property
    def hip_targets(self) -> torch.Tensor:
        """Current (clamped) hip position targets, ``(N, 4)`` in :data:`HIP_ORDER`."""
        return self._hip_targets

    @property
    def nominal_hips(self) -> torch.Tensor:
        return self._nominal_hips

    @property
    def nominal_knees(self) -> torch.Tensor:
        return self._nominal_knees

    # ------------------------------------------------------------------ #
    # action pipeline
    # ------------------------------------------------------------------ #
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale
        self._hip_targets = torch.clamp(
            self._nominal_hips + self._processed_actions[:, : self._num_hips], self._hip_lower, self._hip_upper
        )
        self._wheel_targets = self._processed_actions[:, self._num_hips :]

    def apply_actions(self):
        # hips: position targets relative to the nominal stance
        self._robot.set_joint_position_target(self._hip_targets, joint_ids=self._hip_ids)
        # wheels: velocity targets (+ optional heading-hold correction, see below)
        self._robot.set_joint_velocity_target(self._effective_wheel_targets(), joint_ids=self._wheel_ids)

    # ------------------------------------------------------------------ #
    # heading hold (outer loop, optional)
    # ------------------------------------------------------------------ #
    def _heading_delta(self) -> torch.Tensor:
        """Differential wheel speed (rad/s) that drives the heading error to zero."""
        quat = self._robot.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        err = self.cfg.target_heading - yaw
        # wrap to (-pi, pi]: always correct the short way round
        err = torch.atan2(torch.sin(err), torch.cos(err))

        yaw_rate = self._robot.data.root_ang_vel_w[:, 2]
        omega_des = self.cfg.heading_kp * err - self.cfg.heading_kd * yaw_rate
        omega_des = torch.clamp(omega_des, -self.cfg.heading_max_yaw_rate, self.cfg.heading_max_yaw_rate)

        # yaw rate produced by a wheel-speed differential:
        #   omega_z = 2 * R * delta / track   ->   delta = omega_des * track / (2 R)
        # The sign was measured in simulation, not assumed: with BOTH wheel axes
        # pointing -Y in the body frame a POSITIVE joint velocity rolls that wheel
        # BACKWARD, so `left joint vel > right joint vel` means "left wheel back,
        # right wheel forward" = a +yaw (CCW) turn.  Hence left += delta, right -= delta.
        gain = self.cfg.track_width / (2.0 * self.cfg.wheel_radius)
        return omega_des * gain

    def _effective_wheel_targets(self) -> torch.Tensor:
        """Wheel velocity targets after the optional heading / position corrections.

        WHY THIS EXISTS
        ---------------
        The first policy holds the robot up but does not care about yaw or about
        where it goes: ``flat_orientation_l2`` only looks at the x/y part of the
        projected gravity (roll and pitch), and neither the heading nor the planar
        position appeared in the reward or in the observation.  So the robot was free
        to spin on the spot, and later to balance perfectly while drifting backwards.
        Two independent fixes are provided for each of the two problems:

        * retrain with the ``heading`` / ``position`` observations and the matching
          rewards (the policy then holds both by itself); or
        * keep an existing policy and close these loops HERE, on top of its wheel
          commands.  This needs no retraining.

        Heading is handled as a DIFFERENTIAL (yaw torque needs unequal wheels);
        position is handled as a COMMON MODE (translating needs equal wheels).  The
        two corrections are therefore orthogonal and can be enabled independently.
        """
        needs_yaw = self.cfg.heading_control
        needs_pos = self.cfg.position_control
        if not (needs_yaw or needs_pos):
            return self._wheel_targets

        # --- common mode: position / velocity along the ground ---
        mean = 0.5 * (self._wheel_targets[:, 0] + self._wheel_targets[:, 1])
        if needs_pos:
            mean = mean + self._position_speed_correction()

        # --- differential: yaw rate ---
        if needs_yaw:
            delta = self._heading_delta()
        else:
            delta = 0.5 * (self._wheel_targets[:, 0] - self._wheel_targets[:, 1])

        out = self._wheel_targets.clone()
        if self.cfg.heading_mode == "override" or needs_pos:
            # rebuild from (common mode) +/- (differential)
            out[:, 0] = mean + delta
            out[:, 1] = mean - delta
        else:  # additive heading only
            out[:, 0] = self._wheel_targets[:, 0] + delta
            out[:, 1] = self._wheel_targets[:, 1] - delta

        self.heading_delta = delta
        return out

    def _position_speed_correction(self) -> torch.Tensor:
        """Extra COMMON-MODE wheel speed (rad/s) that drives the robot back to target.

        Sign chain, measured in simulation rather than assumed:

        * a POSITIVE wheel joint velocity rolls that wheel BACKWARD (both wheel axes
          point -Y in the body frame), so moving forward (+X) needs a NEGATIVE common
          wheel speed;
        * therefore ``wheel_correction = -u_desired / wheel_radius``.
        """
        robot_pos = self._robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2]
        ex = self.cfg.target_position_x - robot_pos[:, 0]
        ey = self.cfg.target_position_y - robot_pos[:, 1]

        # work in the yaw frame so the correction means "forward" / "left" regardless
        quat = self._robot.data.root_quat_w
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        c, s = torch.cos(yaw), torch.sin(yaw)
        ex_f = c * ex + s * ey
        ey_l = -s * ex + c * ey

        # body-frame velocity, for damping
        v = self._robot.data.root_lin_vel_b
        u_f = self.cfg.position_kp * ex_f - self.cfg.position_kd * v[:, 0]
        # NOTE: the lateral error `ey_l` is deliberately NOT acted on -- a two-wheel
        # robot cannot translate sideways, so fighting it would only add wheel spin
        # and hurt the balance.  It is still observed (see mdp.position_error) so the
        # policy can lean if it ever needs to.
        u_f = torch.clamp(u_f, -self.cfg.position_max_speed, self.cfg.position_max_speed)
        return -u_f / self.cfg.wheel_radius

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        # the pose jumps on reset, so the stored leg state is no longer a valid
        # previous sample -- the first post-reset rate is reported as zero
        self._leg_prev_valid[env_ids] = False
        self.leg_rate[env_ids] = 0.0
        if isinstance(env_ids, slice):
            self._knee_seed[:] = self.model.nominal_knees.view(1, 2, 2)
            self._knee_targets[:] = self._nominal_knees
        else:
            self._knee_seed[env_ids] = self.model.nominal_knees.view(1, 2, 2)
            self._knee_targets[env_ids] = self._nominal_knees[env_ids]

    # ------------------------------------------------------------------ #
    # loop closure
    # ------------------------------------------------------------------ #
    def update_knee_targets(self) -> torch.Tensor:
        """Re-solve the four knee targets from the *measured* hip angles.

        Warm-started from the previous solution, which is what makes a 3-iteration
        batched Newton solve sufficient at a 50 Hz control rate.

        Returns:
            The knee position targets, ``(N, 4)`` in :data:`KNEE_ORDER`.
        """
        hip_pos = self._robot.data.joint_pos[:, self._hip_ids]
        knees, residual = self.model.solve_knees(
            hip_pos, self._knee_seed[:, 0, :], self._knee_seed[:, 1, :]
        )
        # keep the (possibly clamped) targets inside the URDF knee limits
        clamped = torch.clamp(knees, self._knee_lower, self._knee_upper)
        self.knee_target_clamped = (clamped - knees).abs() > 1e-9
        self._knee_clamped_total += self.knee_target_clamped.sum()
        self._knee_targets = clamped
        self.solver_residual = residual
        # the loop state changed, so the cached measured closure is stale
        self._closure_cache = None
        self._knee_seed = torch.stack(
            (knees[:, 0:2], knees[:, 2:4]), dim=1
        ).detach()  # (N, 2 sides, 2 knees), warm start for the next step
        return self._knee_targets

    def measured_closure_norm(self, reuse_cache: bool = False) -> torch.Tensor:
        """``|W - E|`` per side (m), computed from the *measured* joint angles.

        Args:
            reuse_cache: return the value already computed for this control step
                instead of recomputing it.  The quantity is needed twice per step --
                once by the ``closure_residual`` reward (inside ``env.step``) and once
                by the environment for logging -- and recomputing it doubled the
                number of small CUDA kernels for no benefit.  The cache is invalidated
                by :meth:`update_knee_targets`, so it is always fresh for the step.
        """
        if reuse_cache and self._closure_cache is not None:
            return self._closure_cache
        hip_pos = self._robot.data.joint_pos[:, self._hip_ids]
        knee_pos = self._robot.data.joint_pos[:, self._knee_ids]
        self.measured_residual = self.model.measured_closure(hip_pos, knee_pos)
        self._closure_cache = self.measured_residual
        return self.measured_residual

    @property
    def knee_clamped_total(self) -> int:
        """How many knee targets have been clipped to the limits since reset.

        Reading this syncs the GPU, so it must stay out of the control loop (only the
        smoke test uses it).
        """
        return int(self._knee_clamped_total.item())

    def solved_leg_state(self) -> torch.Tensor:
        """``(L0, phi)`` per side, from the commanded hips and solved knees: ``(N, 4)``."""
        return self.model.measured_leg_state(self._hip_targets, self._knee_targets)

    def measured_leg_state(self) -> torch.Tensor:
        """``(L0, phi)`` per side, from the measured joint angles: ``(N, 4)``."""
        hip_pos = self._robot.data.joint_pos[:, self._hip_ids]
        knee_pos = self._robot.data.joint_pos[:, self._knee_ids]
        return self.model.measured_leg_state(hip_pos, knee_pos)

    def update_leg_rates(self) -> torch.Tensor:
        """Estimate ``(dL0, dphi)`` per side from the measured leg state, ``(N, 4)``.

        Finite difference over one control step.  The order matches
        :meth:`measured_leg_state`, i.e. ``(dL0_l, dphi_l, dL0_r, dphi_r)``.

        This is the leg-velocity feedback the firmware LQR gets from its
        ``dleg_axis_body_rad_s`` state; without it the policy could only react to the
        leg *position* and would lag one derivative behind when the legs carry the
        attitude loop.

        Called once per control step from the observation term.  Environments that have
        just been reset report zero (their pose jumped, so a difference would be
        meaningless).
        """
        state = self.measured_leg_state()
        dt = float(self._env.step_dt)
        fresh = self._leg_prev_valid
        # where a previous sample exists, differentiate; elsewhere stay at zero
        rate = (state - self._leg_state_prev) / dt
        self.leg_rate = torch.where(fresh.unsqueeze(-1), rate, torch.zeros_like(rate))
        self._leg_state_prev = state
        self._leg_prev_valid[:] = True
        return self.leg_rate


@configclass
class FiveBarJointActionCfg(ActionTermCfg):
    """Configuration for :class:`FiveBarJointAction`."""

    class_type: type[ActionTerm] = FiveBarJointAction
    #: radians of hip offset per unit of action
    hip_scale: float = 0.25
    #: rad/s of wheel velocity target per unit of action
    wheel_scale: float = 5.0
    #: absolute bound on the hip offset from the nominal stance (rad)
    hip_range: float = 0.45
    #: keep commands this far inside the URDF limits (rad)
    limit_margin: float = 0.01
    #: fixed iterations of the batched Newton loop solver
    #: fixed iterations of the batched Newton loop solver.
    #:
    #: 2 is enough: measured against the exact URDF FK over 400 feasible hip pairs with
    #: a warm start, the worst-case loop residual is 3.6e-6 m at 2 iterations versus
    #: 7.7e-8 m at 3.  The closed loop error the *simulation* actually achieves is
    #: ~7.5e-4 m (PhysX compliance + the passive knees), so the extra iteration buys
    #: nothing while costing ~150 extra CUDA kernel launches per control step -- and
    #: this solve is the largest single source of launch overhead in the whole step.
    solver_iters: int = 2

    # -- optional outer heading loop (see FiveBarJointAction._effective_wheel_targets)
    #: enable the heading-hold correction on top of the policy's wheel commands
    heading_control: bool = False
    #: "override" (replace the policy's yaw-producing differential) or "additive"
    heading_mode: str = "override"
    #: heading (rad) to hold; all envs spawn at yaw = 0
    target_heading: float = 0.0
    #: PD gains in YAW-RATE units: omega_des = kp * heading_error - kd * yaw_rate
    heading_kp: float = 3.0
    heading_kd: float = 0.35
    #: clamp on the commanded yaw rate (rad/s)
    heading_max_yaw_rate: float = 3.0
    #: wheel radius (m) and track width (m), used to turn a yaw-rate command into a
    #: wheel-speed differential: delta = omega * track / (2 * radius)
    wheel_radius: float = 0.068
    track_width: float = 0.4379

    # -- optional outer position loop (see FiveBarJointAction._position_speed_correction)
    #: enable the hold-position correction on top of the policy's wheel commands
    position_control: bool = False
    #: planar target in the ENVIRONMENT frame (env_origins is the spawn point)
    target_position_x: float = 0.0
    target_position_y: float = 0.0
    #: PD gains: u_desired = kp * position_error - kd * body_velocity   [m/s]
    position_kp: float = 1.5
    position_kd: float = 0.6
    #: clamp on the commanded translation speed (m/s)
    position_max_speed: float = 0.6


__all__ = ["FiveBarJointAction", "FiveBarJointActionCfg"]
