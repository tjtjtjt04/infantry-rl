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

from robot_model import (
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
        #: ``lock_hips`` removes the four hip channels from the policy action and holds
        #: the hips at ``nominal + hip_bias``.  The squat-balance task uses it to train
        #: a two-wheel policy on a mechanically fixed, deepest-possible stance.
        self._lock_hips = bool(cfg.lock_hips)
        self._num_hip_actions = 0 if self._lock_hips else self._num_hips
        self._action_dim = self._num_hip_actions + self._num_wheels

        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._hip_targets = torch.zeros(self.num_envs, self._num_hips, device=self.device)
        self._wheel_targets = torch.zeros(self.num_envs, self._num_wheels, device=self.device)
        #: wheel velocity targets actually written to PhysX in the last step, after the
        #: optional heading-loop override.  Used by the env's wheel-speed diagnostics.
        self.applied_wheel_targets = self._wheel_targets.clone()

        # nominal stance (broadcast over environments)
        self._nominal_hips = self.model.nominal_hips.unsqueeze(0).repeat(self.num_envs, 1)
        self._nominal_knees = self.model.nominal_knees.unsqueeze(0).repeat(self.num_envs, 1)
        #: constant hip offset applied on top of the nominal stance.  Recovery uses
        #: it to make ZERO action the deepest squat, so the policy starts from the
        #: tuck pose and learns to extend from there.  Zero for every other task.
        self._hip_bias = torch.tensor(
            cfg.hip_bias, device=self.device, dtype=torch.float32
        ).view(1, -1)
        if self._hip_bias.shape[1] != self._num_hips:
            raise ValueError(
                f"hip_bias must have {self._num_hips} entries, got {cfg.hip_bias}"
            )
        if cfg.hip_symmetry not in ("off", "master_rf"):
            raise ValueError(
                f"hip_symmetry must be 'off' or 'master_rf', got {cfg.hip_symmetry!r}"
            )

        # --- scripted symmetric stance bias (leg extension / retraction) -------
        # Kept separate from `_hip_bias` on purpose: HandoffController rewrites
        # `_hip_bias` (1,4) on every phase change for the recovery tuck, while the
        # stance is a per-env (N,) scalar driven by training randomization or the
        # gamepad.  Both are added to the hip targets; neither clobbers the other.
        self._stance = torch.zeros(self.num_envs, device=self.device)
        self._stance_target = torch.zeros_like(self._stance)
        self._stance_sample = torch.zeros_like(self._stance)
        self._stance_timer = torch.zeros_like(self._stance)
        self._stance_guard = torch.full_like(self._stance, float(cfg.stance_reset_hold_s))
        self._stance_offsets = torch.zeros(self.num_envs, self._num_hips, device=self.device)
        # HIP_ORDER = (left_front, left_back, right_front, right_back):
        # extension is front +d / back -d on both legs.
        self._stance_sign = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device)
        #: 0 = independent hips, 1 = master_rf.  Slew-limited when the symmetry
        #: gate flips so the hip targets do not jump at the spin/drive boundary.
        self._sym_blend = torch.zeros(self.num_envs, device=self.device)
        self._reset_stance_timer(slice(None))

        self._scale = torch.tensor(
            [cfg.hip_scale] * self._num_hip_actions + [cfg.wheel_scale] * self._num_wheels,
            device=self.device,
            dtype=torch.float32,
        )

        # joint safety envelope: keep every command inside the measured feasible set
        limits = joint_limits()
        #: URDF limits (with the safety margin), before the configurable hip_range cap
        self._hip_limit_lower = torch.tensor(
            [limits[n][0] + cfg.limit_margin for n in HIP_ORDER], device=self.device
        )
        self._hip_limit_upper = torch.tensor(
            [limits[n][1] - cfg.limit_margin for n in HIP_ORDER], device=self.device
        )
        self._hip_lower = torch.maximum(
            self._hip_limit_lower, self._nominal_hips[0] - cfg.hip_range
        )
        self._hip_upper = torch.minimum(
            self._hip_limit_upper, self._nominal_hips[0] + cfg.hip_range
        )
        self._knee_lower = torch.tensor([limits[n][0] + cfg.limit_margin for n in KNEE_ORDER], device=self.device)
        self._knee_upper = torch.tensor([limits[n][1] - cfg.limit_margin for n in KNEE_ORDER], device=self.device)

        # --- optional fixed-hip (squat) pose -------------------------------
        # With ``lock_hips`` the four hip targets are ``nominal + hip_bias`` and the
        # policy only actuates the wheels.  Solve the matching knee pair once here, so
        # a reset and the first control step already start inside the closed chain.
        self._locked_knees = self._nominal_knees[0].clone()
        self._locked_knee_residual = 0.0
        if self._lock_hips:
            locked_hips = (self._nominal_hips[0] + self._hip_bias[0]).unsqueeze(0)
            # Start from the nominal knee pair; the repeated 2-iteration batched Newton
            # solve reaches the folded branch in 2-3 calls (verified over the full range).
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

        # loop-closure state
        seeds = []
        for side in ("left", "right"):
            col = nominal_pose()[side]
            seeds.append(torch.tensor([col["front_knee"], col["back_knee"]], device=self.device))
        self._knee_seed = torch.stack(seeds, dim=0).unsqueeze(0).repeat(self.num_envs, 1, 1)  # (N, side, 2)
        self._knee_targets = self._nominal_knees.clone()
        #: last differential wheel correction applied by the optional heading loop
        self.heading_delta = torch.zeros(self.num_envs, device=self.device)
        #: conditional heading hold (teleop): the heading is locked when the command
        #: is (nearly) straight, and the policy owns the differential while turning.
        self._heading_hold_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._heading_target_hold = torch.zeros(self.num_envs, device=self.device)
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

        # --- action history, for the jerk penalty ------------------------
        # action_rate penalises |a[t] - a[t-1]|, i.e. ALL motion, which is why raising it
        # makes the legs stiff.  The second difference a[t] - 2a[t-1] + a[t-2] is ~0 for
        # smooth motion (constant / ramp / parabola) and only large for a tremor that
        # reverses every step, so it can be penalised hard without stiffening the legs.
        self._action_prev1 = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._action_prev2 = torch.zeros(self.num_envs, self._action_dim, device=self.device)

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

    @property
    def locked_hips(self) -> torch.Tensor:
        """Absolute hip targets used by ``lock_hips``, ``(4,)`` in :data:`HIP_ORDER`."""
        return self._nominal_hips[0] + self._hip_bias[0]

    @property
    def reset_knees(self) -> torch.Tensor:
        """Knee pose that closes the loop at the reset hip pose, ``(4,)``.

        For every task except squat this is the nominal knee pair (the previous
        behaviour); with ``lock_hips`` it is the solved pair for the fixed hip pose.
        """
        return self._locked_knees if self._lock_hips else self._nominal_knees[0]

    # ------------------------------------------------------------------ #
    # scripted symmetric stance (leg extension / retraction)
    # ------------------------------------------------------------------ #
    def _make_stance_writable(self) -> None:
        """Turn inference tensors created inside ``torch.inference_mode()`` into normal ones.

        Deployment steps run inside ``torch.inference_mode()`` (isaac_teleop), so the
        out-of-place stance updates there produce inference tensors.  PyTorch forbids
        in-place writes to those OUTSIDE inference mode -- exactly what happens when
        B is pressed and HandoffController snaps the stance to zero.  ``clone()``
        outside inference mode returns a normal tensor (verified with torch 2.x).
        """
        tensors = (
            self._stance,
            self._stance_target,
            self._stance_sample,
            self._stance_timer,
            self._stance_guard,
            self._stance_offsets,
            self._sym_blend,
        )
        if any(t.is_inference() for t in tensors) and not torch.is_inference_mode_enabled():
            self._stance = self._stance.clone()
            self._stance_target = self._stance_target.clone()
            self._stance_sample = self._stance_sample.clone()
            self._stance_timer = self._stance_timer.clone()
            self._stance_guard = self._stance_guard.clone()
            self._stance_offsets = self._stance_offsets.clone()
            self._sym_blend = self._sym_blend.clone()

    def _reset_stance_timer(self, env_ids: slice | torch.Tensor) -> None:
        if isinstance(env_ids, slice):
            n = self.num_envs
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            n = int(env_ids.numel())
        lo = float(self.cfg.stance_resample_min_s)
        hi = max(float(self.cfg.stance_resample_max_s), lo)
        self._stance_timer[env_ids] = torch.empty(n, device=self.device).uniform_(lo, hi)

    def _stance_gate(self) -> torch.Tensor:
        """True where the command is a (near-)pure in-place rotation."""
        try:
            term = self._env.command_manager.get_term(self.cfg.stance_command_name)
            cmd = term.command
        except (AttributeError, KeyError, ValueError):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return (cmd[:, 0].abs() <= float(self.cfg.stance_gate_vx)) & (
            cmd[:, 2].abs() >= float(self.cfg.stance_gate_wz)
        )

    def _update_stance(self) -> None:
        """Advance the per-env stance by one control step.

        Training mode samples a DISCRETE stance (crouch / nominal / extend) every
        2-4 s while the command is an in-place spin, then slews towards it.  The
        discrete set matters: a uniform draw over the range would rarely visit the
        exact extension the gamepad holds.  In deployment mode the target is set
        externally (gamepad RB) and only the slew is applied here.
        """
        cfg = self.cfg
        dt = float(self._env.step_dt)
        if cfg.stance_randomize:
            self._stance_guard -= dt
            self._stance_timer -= dt
            resample = self._stance_timer <= 0.0
            u = torch.rand(self.num_envs, device=self.device)
            p0 = float(cfg.stance_prob_crouch)
            p1 = p0 + float(cfg.stance_prob_nominal)
            sample = torch.where(
                u < p0,
                torch.full_like(u, float(cfg.stance_min)),
                torch.where(
                    u < p1,
                    torch.zeros_like(u),
                    torch.full_like(u, float(cfg.stance_max)),
                ),
            )
            sample = sample + float(cfg.stance_jitter) * (2.0 * torch.rand_like(sample) - 1.0)
            sample = torch.clamp(sample, min=float(cfg.stance_min), max=float(cfg.stance_max))
            self._stance_sample = torch.where(resample, sample, self._stance_sample)
            new_timer = torch.empty_like(self._stance_timer).uniform_(
                float(cfg.stance_resample_min_s), float(cfg.stance_resample_max_s)
            )
            self._stance_timer = torch.where(resample, new_timer, self._stance_timer)
            self._stance_guard = self._stance_guard.clamp(min=0.0)
            allowed = self._stance_gate() & (self._stance_guard <= 0.0)
            self._stance_target = torch.where(
                allowed, self._stance_sample, torch.zeros_like(self._stance_sample)
            )

        step = max(0.0, float(cfg.stance_slew_rate)) * dt
        self._stance = self._stance + torch.clamp(
            self._stance_target - self._stance, min=-step, max=step
        )
        self._stance_offsets = self._stance.unsqueeze(-1) * self._stance_sign

    def set_hip_range(self, hip_range: float) -> None:
        """Change the hip travel cap at runtime (used by the handoff phase machine).

        The recovery and teleop checkpoints were trained with different ``hip_range``
        values (0.45 vs 0.60).  Deployment runs both policies through the SAME action
        term, so the cap has to follow the active phase: 0.45 in recovery/stand and
        0.60 in teleop.
        """
        value = float(hip_range)
        self.cfg.hip_range = value
        self._hip_lower = torch.maximum(
            self._hip_limit_lower, self._nominal_hips[0] - value
        )
        self._hip_upper = torch.minimum(
            self._hip_limit_upper, self._nominal_hips[0] + value
        )

    def set_stance_target(
        self,
        value: float | torch.Tensor,
        env_ids: slice | torch.Tensor | None = None,
        snap: bool = False,
    ) -> None:
        """Set the stance target (rad), used by the gamepad and by fixed-command play.

        ``snap=True`` writes the value to the applied stance immediately (used when
        recovery takes over and for static mechanical checks).
        """
        if env_ids is None:
            env_ids = slice(None)
        self._make_stance_writable()
        self._stance_target[env_ids] = value
        if snap:
            self._stance[env_ids] = self._stance_target[env_ids]
            self._stance_offsets = self._stance.unsqueeze(-1) * self._stance_sign

    @property
    def stance(self) -> torch.Tensor:
        """Current applied symmetric stance bias ``d`` in rad, ``(N,)``."""
        return self._stance

    @property
    def stance_offsets(self) -> torch.Tensor:
        """``[+d, -d, +d, -d]`` in :data:`HIP_ORDER`, ``(N, 4)``."""
        return self._stance_offsets

    @property
    def intended_hip_offsets(self) -> torch.Tensor:
        """Recovery hip bias + stance offsets, ``(N, 4)``; the pose the legs should hold."""
        return self._hip_bias + self._stance_offsets

    # ------------------------------------------------------------------ #
    # action pipeline
    # ------------------------------------------------------------------ #
    def process_actions(self, actions: torch.Tensor):
        # A non-finite policy output (possible once the observation normaliser has been
        # poisoned by a NaN, or right after a numerically bad solve) must never reach
        # PhysX: NaN joint targets corrupt the GPU solver and surface as
        # "CUDA error: an illegal memory access".  Sanitising here turns that into a
        # harmless zero action for the affected environment.
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        if self.cfg.sagittal_symmetry and not self._lock_hips:
            # Front/back falls are a sagittal-plane problem.  Averaging the left and
            # right channels removes the asymmetric actuator authority that rolls
            # the robot onto its side; the wheel differential is removed for the
            # same reason.  The 6-dim interface is unchanged (the redundant
            # channels receive the same target).
            actions = actions.clone()
            front = 0.5 * (actions[:, 0] + actions[:, 2])
            back = 0.5 * (actions[:, 1] + actions[:, 3])
            wheel = 0.5 * (actions[:, 4] + actions[:, 5])
            actions[:, 0] = front
            actions[:, 2] = front
            actions[:, 1] = back
            actions[:, 3] = back
            actions[:, 4] = wheel
            actions[:, 5] = wheel
        # shift the history BEFORE storing the new sample: after this call the order is
        # (_action_prev2, _action_prev1, _raw_actions) = (a[t-2], a[t-1], a[t])
        self._action_prev2 = self._action_prev1
        self._action_prev1 = self._raw_actions.clone()
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale
        # Recovery: the tuck bias fades out as the body becomes upright, which is
        # the "tuck, drive the wheels, then extend the legs" sequence.  The policy
        # therefore does not have to discover a 1.8-unit hip action from a small
        # exploration noise; it only has to balance while the stance extends.
        bias = self._hip_bias
        if self.cfg.dynamic_hip_bias:
            upright = (1.0 - self._robot.data.projected_gravity_b[:, 2]) * 0.5
            span = max(
                float(self.cfg.hip_bias_upright_end)
                - float(self.cfg.hip_bias_upright_start),
                1e-6,
            )
            scale = 1.0 - torch.clamp(
                (upright - float(self.cfg.hip_bias_upright_start)) / span, 0.0, 1.0
            )
            bias = self._hip_bias * scale.unsqueeze(-1)
        if self.cfg.hip_bias_time_ramp:
            # Stand-from-squat: hold the tuck, then extend the hips slowly.  A
            # one-step extension is too fast for the balance policy to follow.
            t = self._env.episode_length_buf.to(dtype=torch.float32) * float(self._env.step_dt)
            span = max(float(self.cfg.hip_bias_ramp_s), 1e-6)
            scale = 1.0 - torch.clamp(
                (t - float(self.cfg.hip_bias_hold_s)) / span, 0.0, 1.0
            )
            bias = self._hip_bias * scale.unsqueeze(-1)
        self._update_stance()
        if self._lock_hips:
            # The policy owns the wheels only.  Keep the four hips at the configured
            # fixed pose (``nominal + hip_bias``).  Stance/symmetry offsets are ignored
            # on purpose: moving the hips at all would defeat the locked-squat premise.
            raw_hips = self._nominal_hips + bias
        else:
            raw_hips = (
                self._nominal_hips
                + bias
                + self._stance_offsets
                + self._processed_actions[:, : self._num_hips]
            )
            if self.cfg.hip_symmetry == "master_rf":
                if self.cfg.hip_symmetry_gate:
                    # only in-place commands get the symmetric master_rf pose
                    gate = self._stance_gate().to(dtype=raw_hips.dtype)
                else:
                    gate = torch.ones_like(self._sym_blend)
                rate = max(0.0, float(self.cfg.hip_symmetry_blend_rate)) * float(
                    self._env.step_dt
                )
                self._sym_blend = self._sym_blend + torch.clamp(
                    gate - self._sym_blend, min=-rate, max=rate
                )
                # t = total right-front offset (stance + policy action).  Mirror it:
                # front = +t, back = -t, both sides identical.  This is exactly
                # "all four legs at the right-front leg's angle" and leaves zero
                # hip-induced pitch, so the wheels own pitch balance.
                t = raw_hips[:, 2] - self._nominal_hips[:, 2]
                mirrored = self._nominal_hips + torch.stack((t, -t, t, -t), dim=-1)
                w = self._sym_blend.unsqueeze(-1)
                raw_hips = raw_hips + w * (mirrored - raw_hips)
            else:
                self._sym_blend = torch.zeros_like(self._sym_blend)
        self._hip_targets = torch.clamp(raw_hips, self._hip_lower, self._hip_upper)
        self._wheel_targets = self._processed_actions[:, self._num_hip_actions :]

    def action_jerk(self) -> torch.Tensor:
        """Second difference of the action, ``a[t] - 2 a[t-1] + a[t-2]``, ``(N, 6)``.

        WHY THE SECOND DIFFERENCE AND NOT THE FIRST
        -------------------------------------------
        ``action_rate`` charges ``|a[t] - a[t-1]|``, which penalises ALL motion -- raising
        that weight is what makes the legs stiff and unable to catch a pitch disturbance.
        The second difference is ~0 for any *smooth* motion (constant, ramp, even a
        parabola), and large only when the action reverses direction every step, i.e. for
        exactly the high-frequency tremor we want to remove.  It lets the legs move fast
        while making a dithering leg expensive.
        """
        return self._raw_actions - 2.0 * self._action_prev1 + self._action_prev2

    def apply_actions(self):
        # hips: position targets relative to the nominal stance
        self._robot.set_joint_position_target(self._hip_targets, joint_ids=self._hip_ids)
        # wheels: velocity targets (+ optional heading-hold correction, see below)
        self.applied_wheel_targets = self._effective_wheel_targets()
        self._robot.set_joint_velocity_target(self.applied_wheel_targets, joint_ids=self._wheel_ids)

    # ------------------------------------------------------------------ #
    # heading hold (outer loop, optional)
    # ------------------------------------------------------------------ #
    def _conditional_heading_active(self) -> torch.Tensor:
        """Which envs should have their heading locked THIS step (teleop mode).

        The teleop command term sets ``wz`` directly.  While it is very small the
        policy is not asked to steer, and a two-wheel robot otherwise drifts
        ("松杆后走不直").  In that regime the PD owns the differential, holding the
        heading the robot had when the straight command arrived.  Once ``|wz_cmd|``
        exceeds the threshold the policy owns the differential again, so it can
        actually turn.  The command is the policy's observation, so the switch is
        visible to the network.
        """
        if not self.cfg.conditional_heading_control:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        try:
            term = self._env.command_manager.get_term(self.cfg.heading_hold_command_name)
            wz_cmd = term.command[:, 2]
        except (AttributeError, KeyError, ValueError):
            # no velocity command: fall back to the unconditional loop
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return wz_cmd.abs() < float(self.cfg.heading_hold_wz_threshold)

    def _heading_delta(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Differential wheel speed (rad/s) that drives the heading error to zero.

        Returns:
            ``(delta, active)``.  ``active`` is ``None`` in the classical
            (unconditional) mode; in conditional mode it is the per-env mask computed
            by :meth:`_conditional_heading_active`.
        """
        quat = self._robot.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        if self.cfg.conditional_heading_control:
            active = self._conditional_heading_active()
            # lock a NEW heading only on the rising edge (turn finished -> straight
            # command).  While the hold stays active the target must not follow the
            # robot, otherwise the loop would have nothing to correct.
            newly = active & ~self._heading_hold_active
            # masked assignment is a no-op when nothing is new; no CPU sync here
            self._heading_target_hold[newly] = yaw[newly]
            self._heading_hold_active[:] = active
            target = self._heading_target_hold
        else:
            active = None
            target = float(self.cfg.target_heading)  # scalar broadcast, no allocation

        err = target - yaw
        # wrap to (-pi, pi]: always correct the short way round
        err = torch.atan2(torch.sin(err), torch.cos(err))

        yaw_rate = self._robot.data.root_ang_vel_w[:, 2]
        omega_des = self.cfg.heading_kp * err - self.cfg.heading_kd * yaw_rate
        if (
            self.cfg.conditional_heading_control
            and self.cfg.heading_hold_max_lateral_accel > 0.0
        ):
            # The PD correction is a yaw demand just like a joystick command, so it
            # must obey the same feasibility cap |vx * wz| <= a_lat.  Without this,
            # a 2 rad/s correction at 3 m/s asks for 6 m/s^2 of lateral acceleration
            # and can tip the robot during a high-speed straight-line correction.
            # (Measured failure onset was ~2.1 m/s, exactly where 2.0 rad/s * vx
            # crosses 4 m/s^2.)
            vx_abs = self._robot.data.root_lin_vel_b[:, 0].abs().clamp(min=0.05)
            rate_cap = torch.clamp(
                self.cfg.heading_hold_max_lateral_accel / vx_abs,
                max=float(self.cfg.heading_max_yaw_rate),
            )
            omega_des = torch.maximum(torch.minimum(omega_des, rate_cap), -rate_cap)
        else:
            omega_des = torch.clamp(
                omega_des, -self.cfg.heading_max_yaw_rate, self.cfg.heading_max_yaw_rate
            )

        # yaw rate produced by a wheel-speed differential:
        #   omega_z = 2 * R * delta / track   ->   delta = omega_des * track / (2 R)
        # The sign was measured in simulation, not assumed: with BOTH wheel axes
        # pointing -Y in the body frame a POSITIVE joint velocity rolls that wheel
        # BACKWARD, so `left joint vel > right joint vel` means "left wheel back,
        # right wheel forward" = a +yaw (CCW) turn.  Hence left += delta, right -= delta.
        gain = self.cfg.track_width / (2.0 * self.cfg.wheel_radius)
        return omega_des * gain, active

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

        With ``conditional_heading_control`` the heading override is per-environment:
        straight/standing commands get the PD differential, turning commands keep the
        policy's differential.  That is what lets ONE policy drive, turn and stop.
        """
        needs_yaw = self.cfg.heading_control
        needs_pos = self.cfg.position_control
        needs_pitch = self.cfg.pitch_control
        conditional = needs_yaw and self.cfg.conditional_heading_control
        if not (needs_yaw or needs_pos or needs_pitch):
            return self._wheel_targets

        # --- common mode: position / velocity along the ground ---
        mean = 0.5 * (self._wheel_targets[:, 0] + self._wheel_targets[:, 1])
        if needs_pos:
            mean = mean + self._position_speed_correction()
        if needs_pitch:
            # The pitch-hold assist OWNS the common wheel speed: the teleop/recovery
            # policy's wheel common is out of distribution during the stand-up
            # transition, while the inverted-pendulum PD is well defined.
            mean = self._pitch_speed_correction()

        # --- differential: yaw rate ---
        if needs_yaw:
            delta, hold_active = self._heading_delta()
        else:
            delta = 0.5 * (self._wheel_targets[:, 0] - self._wheel_targets[:, 1])
            hold_active = None

        out = self._wheel_targets.clone()
        if conditional:
            # Per-env choice: PD differential while holding, the policy's own
            # differential while turning.  Rebuilding from the common mode keeps the
            # forward/brake command intact in both cases.
            policy_delta = 0.5 * (self._wheel_targets[:, 0] - self._wheel_targets[:, 1])
            applied = torch.where(hold_active, delta, policy_delta)
            out[:, 0] = mean + applied
            out[:, 1] = mean - applied
            self.heading_delta = torch.where(
                hold_active, delta, torch.zeros_like(delta)
            )
        elif self.cfg.heading_mode == "override" or needs_pos or needs_pitch:
            # rebuild from (common mode) +/- (differential)
            out[:, 0] = mean + delta
            out[:, 1] = mean - delta
            self.heading_delta = delta
        else:  # additive heading only
            out[:, 0] = self._wheel_targets[:, 0] + delta
            out[:, 1] = self._wheel_targets[:, 1] - delta
            self.heading_delta = delta
        return out

    def _pitch_speed_correction(self) -> torch.Tensor:
        """Common wheel speed (rad/s) from an inverted-pendulum pitch PD.

        A two-wheel body is stabilized by driving the wheels in the direction of the
        lean: ``u = kp*pitch + kd*pitch_rate`` (m/s), then the wheel joint target is
        ``-u / R`` (the same sign chain as the position loop).  This is used only in
        the handoff ALIGN phase, where the recovery squat has a pitch offset that the
        learned teleop policy has never seen.
        """
        quat = self._robot.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        pitch_rate = self._robot.data.root_ang_vel_b[:, 1]
        u = float(self.cfg.pitch_kp) * (pitch - float(self.cfg.pitch_target)) + float(
            self.cfg.pitch_kd
        ) * pitch_rate
        u = torch.clamp(u, -self.cfg.pitch_max_speed, self.cfg.pitch_max_speed)
        return -u / self.cfg.wheel_radius

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
        self._action_prev1[env_ids] = 0.0
        self._action_prev2[env_ids] = 0.0
        # the spawn pose has yaw 0; a hold target from the previous episode is stale
        self._heading_hold_active[env_ids] = False
        self._heading_target_hold[env_ids] = 0.0
        # stance: always respawn at the nominal stance and hold it briefly, so the
        # first post-reset command is the trained nominal pose
        self._make_stance_writable()
        self._stance[env_ids] = 0.0
        self._stance_target[env_ids] = 0.0
        self._stance_sample[env_ids] = 0.0
        self._stance_guard[env_ids] = float(self.cfg.stance_reset_hold_s)
        self._reset_stance_timer(env_ids)
        self._stance_offsets = self._stance.unsqueeze(-1) * self._stance_sign
        self._sym_blend[env_ids] = 0.0
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
        # A divergent Newton solve can return NaN.  PhysX joint targets must stay finite
        # (clamp does NOT repair NaN), and a bad warm start would otherwise keep
        # poisoning the next step, so fall back to the nominal knees for those envs.
        finite = torch.isfinite(knees).all(dim=-1, keepdim=True)
        knees = torch.where(finite, knees, self._nominal_knees)
        # keep the (possibly clamped) targets inside the URDF knee limits
        clamped = torch.clamp(knees, self._knee_lower, self._knee_upper)
        self.knee_target_clamped = (clamped - knees).abs() > 1e-9
        self._knee_clamped_total += self.knee_target_clamped.sum()
        self._knee_targets = clamped
        self.solver_residual = torch.nan_to_num(residual, nan=0.0, posinf=1.0, neginf=1.0)
        # the loop state changed, so the cached measured closure is stale
        self._closure_cache = None
        self._knee_seed = torch.stack(
            (clamped[:, 0:2], clamped[:, 2:4]), dim=1
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

        The result is sanitised: with the wide speed range the robot regularly ends up in
        poses where the five-bar has no valid solution, and ``leg_state`` then returns
        NaN/Inf.  A NaN that reaches the observation would poison the running
        normaliser and, through the policy, the joint targets handed to PhysX -- which is
        one of the ways a "CUDA illegal memory access" can appear.  Non-finite and
        absurd rates are therefore replaced by 0 / clamped.
        """
        state = self.measured_leg_state()
        dt = float(self._env.step_dt)
        fresh = self._leg_prev_valid
        # where a previous sample exists, differentiate; elsewhere stay at zero
        rate = (state - self._leg_state_prev) / dt
        rate = torch.where(fresh.unsqueeze(-1), rate, torch.zeros_like(rate))
        # a physically sane leg reaches ~1 m/s of length change and ~20 rad/s of swing;
        # anything beyond that is a bad solve, not motion
        rate = torch.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0)
        self.leg_rate = rate.clamp(-50.0, 50.0)
        # keep the stored previous sample finite as well, otherwise one bad frame would
        # keep producing NaN differences on every following step
        self._leg_state_prev = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
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
    #: Remove the four hip channels from the action and hold the hips at
    #: ``nominal + hip_bias``.  Used by the squat-balance task, where the policy must
    #: learn balance on the wheels alone.  ``action_dim`` becomes the number of wheels.
    lock_hips: bool = False
    #: constant hip offset (HIP_ORDER) added before the policy action.  Recovery
    #: sets this to the deepest squat so zero action means "hold the tuck".  With
    #: ``lock_hips`` it is the fixed squat pose itself.
    hip_bias: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    #: fade the hip_bias out as uprightness rises: tuck while fallen, nominal stance
    #: once upright.  This is the final "extend the legs" phase of recovery.
    dynamic_hip_bias: bool = False
    #: uprightness (0..1) where the fade starts / completes
    hip_bias_upright_start: float = 0.90
    hip_bias_upright_end: float = 0.99
    #: time-based fade (stand-from-squat): hold the tuck for hold_s, then extend
    #: the hip bias to zero over ramp_s.  Overrides the dynamic uprightness fade.
    hip_bias_time_ramp: bool = False
    hip_bias_hold_s: float = 0.5
    hip_bias_ramp_s: float = 1.5

    # -- scripted symmetric stance bias (A + RB leg extension) -------------
    #: randomize the stance during training.  The bias is applied as
    #: ``[+d, -d, +d, -d]`` on top of the nominal hips, i.e. a pure leg-length
    #: change that leaves the fore/aft balance authority untouched.  Off by
    #: default so every other task and every existing checkpoint is unchanged.
    stance_randomize: bool = False
    #: sampled stance range (rad of hip offset).  Negative = crouch, positive = extend.
    stance_min: float = -0.10
    stance_max: float = 0.30
    #: probability of the crouch / nominal / extension branches (sum <= 1; the
    #: remainder falls through to extension).  Discrete samples make sure the
    #: exact deployment targets are seen during training.
    stance_prob_crouch: float = 0.30
    stance_prob_nominal: float = 0.20
    #: uniform jitter added to every sample (rad)
    stance_jitter: float = 0.02
    #: maximum stance change rate (rad/s).  Also used when the gamepad drives the
    #: bias externally, so the body extends/retracts at the speed it was trained on.
    stance_slew_rate: float = 0.5
    #: only randomize while the command is a (near-)pure in-place rotation.  This
    #: keeps 2.5 m/s straight driving free of stance disturbances.
    stance_gate_vx: float = 0.10
    stance_gate_wz: float = 1.0
    #: command term inspected by the gate
    stance_command_name: str = "base_velocity"
    #: seconds between stance resamples
    stance_resample_min_s: float = 2.0
    stance_resample_max_s: float = 4.0
    #: after an episode reset the stance is held at zero for this long.  This keeps
    #: the reset pose nominal (and the ``train.py --check`` hip-sign probe valid).
    stance_reset_hold_s: float = 0.30
    #: recovery only: average left/right hip actions and zero the wheel differential,
    #: which removes the actuator authority that causes side rolls.
    sagittal_symmetry: bool = False
    #: "off" or "master_rf".  ``master_rf`` slaves all four hip targets to the
    #: RIGHT-FRONT total offset: ``[+t, -t, +t, -t]`` about nominal.  This makes the
    #: front/back and left/right leg angles identical (the right front leg is the
    #: reference) and removes the hip common-mode that pitches the body / lifts the
    #: nose.  The policy then balances pitch with the wheels.  Off by default so the
    #: recovery/stand tasks and every existing checkpoint are unchanged.
    hip_symmetry: str = "off"
    #: only apply master_rf for in-place commands (|vx_cmd| <= stance_gate_vx and
    #: |wz_cmd| >= stance_gate_wz).  Outside that, the four hips are independent, so
    #: normal driving/turning keeps the original asymmetric hip gait.  Must be
    #: trained with the same gate.
    hip_symmetry_gate: bool = False
    #: blend rate (1/s) between independent and master_rf targets when the gate
    #: flips. 0.2 s at 5/s avoids a hip-target jump when A is pressed/released.
    hip_symmetry_blend_rate: float = 5.0
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
    #: Teleop mode: apply the heading hold only while |wz_cmd| is below
    #: `heading_hold_wz_threshold`, locking the heading present when that command
    #: arrived.  Turning commands then keep the policy's differential untouched.
    conditional_heading_control: bool = False
    #: name of the command term inspected by the conditional hold
    heading_hold_command_name: str = "base_velocity"
    #: |wz_cmd| (rad/s) below which the conditional PD owns the yaw differential
    heading_hold_wz_threshold: float = 0.15
    #: cap on the PD correction's lateral acceleration |vx_actual * wz_pd| (m/s^2);
    #: 0 disables.  This is the SAME physical limit as the command sampler's cap.
    heading_hold_max_lateral_accel: float = 4.0
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

    # -- optional outer pitch-hold loop (handoff ALIGN phase) -----------------
    #: own the common wheel speed with an inverted-pendulum pitch PD
    pitch_control: bool = False
    #: desired pitch (rad) and PD gains in velocity units: u = kp*(pitch-target)+kd*rate
    pitch_target: float = 0.0
    pitch_kp: float = 3.0
    pitch_kd: float = 0.35
    #: clamp on the corrective base speed (m/s)
    pitch_max_speed: float = 0.6

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
