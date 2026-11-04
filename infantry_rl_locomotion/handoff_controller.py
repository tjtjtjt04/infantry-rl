"""Three-phase fall-recovery / stand / teleop handoff controller.

The three skills share the same 35-dim observation and 6-dim action, but they need
different ACTION-TERM semantics:

============  ==============  =============  ============  =========  =====
phase         policy          hip bias       sagittal      heading   clip
============  ==============  =============  ============  =========  =====
RECOVERY      recovery v5     tuck (-.45/+.45)  yes         off       4
STAND         teleop 11344    zero            yes           off       2
TELEOP        teleop 11344    zero            no            cond. PD  12
============  ==============  =============  ============  =========  =====

RECOVERY gets the robot from a front/back fall into a stable upright squat.  STAND
starts from that squat, removes the tuck bias and lets the position controller
extend the legs while the teleop policy balances.  TELEOP hands the same policy the
real driving command.  A fall during TELEOP (or STAND) switches back to RECOVERY.

The controller is deliberately policy-agnostic: it takes two callables and mutates
the Isaac Lab action/command term objects.  The same phase logic can be reused on
the MuJoCo side by providing an equivalent action/command interface.
"""

from __future__ import annotations

from enum import Enum

import torch


class HandoffPhase(str, Enum):
    RECOVERY = "recovery"
    ALIGN = "align"
    STAND = "stand"
    TELEOP = "teleop"


class HandoffController:
    """Phase state machine + per-phase action-term/command configuration."""

    def __init__(
        self,
        env,
        robot,
        action_term,
        command_term,
        recovery_policy,
        teleop_policy,
        drive_vx: float = 0.0,
        drive_wz: float = 0.0,
        device: str | torch.device = "cuda:0",
        *,
        upright_enter: float = 0.95,
        #: the current teleop policy naturally stands/crouches at base_z ~0.136
        #: with zero command; the old 0.15 gate never opened and STAND timed out.
        stand_height_enter: float = 0.12,
        squat_height_max: float = 0.16,
        fall_upright_min: float = 0.70,
        #: deep-crouch spins sit at base_z ~0.078-0.090 m; 0.09 would flag them as
        #: fallen and start recovery.  Orientation (upright) still catches real falls.
        fall_height_min: float = 0.06,
        still_lin: float = 0.20,
        still_ang: float = 0.50,
        hold_time: float = 0.50,
        fall_hold_time: float = 0.20,
        align_pitch: float = 0.09,
        align_pitch_rate: float = 0.35,
        tuck_front: float = -0.45,
        tuck_back: float = 0.45,
        heading_hold_max_lateral_accel: float = 4.0,
        teleop_hip_range: float = 0.60,
        recovery_hip_range: float = 0.45,
        teleop_hip_symmetry: str = "off",
        teleop_hip_symmetry_gate: bool = False,
    ):
        self.env = env
        self.robot = robot
        self.action_term = action_term
        self.command_term = command_term
        self.recovery_policy = recovery_policy
        self.teleop_policy = teleop_policy
        self.device = torch.device(device)
        self.drive_vx = float(drive_vx)
        self.drive_wz = float(drive_wz)

        self.upright_enter = upright_enter
        self.stand_height_enter = stand_height_enter
        self.squat_height_max = squat_height_max
        self.fall_upright_min = fall_upright_min
        self.fall_height_min = fall_height_min
        self.still_lin = still_lin
        self.still_ang = still_ang
        self.hold_steps = max(1, int(round(hold_time / float(env.step_dt))))
        self.fall_steps = max(1, int(round(fall_hold_time / float(env.step_dt))))
        self.align_pitch = align_pitch
        self.align_pitch_rate = align_pitch_rate
        self.tuck_front = tuck_front
        self.tuck_back = tuck_back
        #: lateral-acceleration cap of the teleop heading-hold PD.  The current
        #: teleop checkpoint was trained with 8.0; the hardcoded 4.0 used to make
        #: deployment gentler than training.
        self.heading_hold_max_lateral_accel = float(heading_hold_max_lateral_accel)
        # The recovery checkpoint was trained with hip_range 0.45; the current
        # teleop checkpoint with 0.60.  Both run through the same action term, so
        # the cap follows the phase (see _configure_phase).
        self.teleop_hip_range = float(teleop_hip_range)
        self.recovery_hip_range = float(recovery_hip_range)
        #: "off" or "master_rf"; only TELEOP uses it.  Recovery/stand always use
        #: independent hips, otherwise their tuck/align behaviour would be slaved
        #: to the right-front leg.
        self.teleop_hip_symmetry = str(teleop_hip_symmetry)
        #: only apply master_rf for in-place commands in TELEOP
        self.teleop_hip_symmetry_gate = bool(teleop_hip_symmetry_gate)

        self.phase = HandoffPhase.RECOVERY
        self._hold = 0
        self._fall = 0
        #: seconds spent in the current non-TELEOP phase (safety timeout)
        self.phase_time = 0.0
        #: set when a non-TELEOP phase exceeded its timeout; the runner resets
        self.timeout_requested = False
        self._configure_phase(print_info=True)

    # ------------------------------------------------------------------ #
    # state helpers
    # ------------------------------------------------------------------ #
    def _uprightness(self) -> torch.Tensor:
        return (1.0 - self.robot.data.projected_gravity_b[:, 2]) * 0.5

    def _height(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, 2] - self.env.scene.env_origins[:, 2]

    def _still(self) -> torch.Tensor:
        lin = torch.norm(self.robot.data.root_lin_vel_b, dim=-1)
        ang = torch.norm(self.robot.data.root_ang_vel_b, dim=-1)
        return (lin < self.still_lin) & (ang < self.still_ang)

    def _pitch(self) -> torch.Tensor:
        q = self.robot.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))

    def set_drive_command(self, vx: float, wz: float) -> None:
        self.drive_vx = float(vx)
        self.drive_wz = float(wz)

    def request_recovery(self) -> None:
        """Force the recovery phase (gamepad B button / external request)."""
        self._set_phase(HandoffPhase.RECOVERY)

    # ------------------------------------------------------------------ #
    # action-term configuration
    # ------------------------------------------------------------------ #
    def _set_hip_bias(self, front: float, back: float) -> None:
        bias = torch.tensor(
            [front, back, front, back], device=self.action_term.device, dtype=torch.float32
        ).view(1, -1)
        self.action_term._hip_bias = bias
        self.action_term.cfg.hip_bias = (front, back, front, back)

    def _set_wheel_scale(self, scale: float) -> None:
        # The action term caches its scale vector at construction; keep it in sync.
        scale_vec = self.action_term._scale
        scale_vec[self.action_term._num_hips :] = float(scale)

    def _set_stance(self, value: float, snap: bool = True) -> None:
        """Zero (or set) the scripted stance; separate from the recovery ``_hip_bias``."""
        setter = getattr(self.action_term, "set_stance_target", None)
        if setter is not None:
            setter(float(value), snap=snap)

    def _set_hip_range(self, value: float) -> None:
        """Match the hip clamp to the checkpoint of the phase's policy."""
        setter = getattr(self.action_term, "set_hip_range", None)
        if setter is not None:
            setter(float(value))

    def _set_hip_symmetry(self, value: str, gate: bool = False) -> None:
        """Only TELEOP may slave the four hip targets to the right-front leg."""
        self.action_term.cfg.hip_symmetry = str(value)
        self.action_term.cfg.hip_symmetry_gate = bool(gate)

    def _configure_phase(self, print_info: bool = False) -> None:
        cfg = self.action_term.cfg
        if self.phase == HandoffPhase.RECOVERY:
            self._set_hip_bias(self.tuck_front, self.tuck_back)
            self._set_hip_range(self.recovery_hip_range)
            self._set_hip_symmetry("off")
            cfg.dynamic_hip_bias = False
            cfg.hip_bias_time_ramp = False
            cfg.sagittal_symmetry = True
            cfg.heading_control = False
            cfg.conditional_heading_control = False
            cfg.position_control = False
            cfg.pitch_control = False
            self._set_wheel_scale(5.0)
            self._heading_clip = 4.0
        elif self.phase == HandoffPhase.ALIGN:
            # Hold the recovery squat (tuck) while the pitch PD drives the common
            # wheel speed to level the body.  The tuck gives the wheel-leg a low,
            # stiff stance; the earlier stand-hip variant made the pitch worse.
            self._set_hip_bias(self.tuck_front, self.tuck_back)
            self._set_hip_range(self.recovery_hip_range)
            self._set_hip_symmetry("off")
            cfg.dynamic_hip_bias = False
            cfg.hip_bias_time_ramp = False
            cfg.sagittal_symmetry = True
            cfg.heading_control = False
            cfg.conditional_heading_control = False
            cfg.position_control = False
            cfg.pitch_control = True
            self._set_wheel_scale(5.0)
            self._heading_clip = 4.0
        elif self.phase == HandoffPhase.STAND:
            self._set_hip_bias(0.0, 0.0)
            self._set_hip_range(self.recovery_hip_range)
            self._set_hip_symmetry("off")
            cfg.dynamic_hip_bias = False
            cfg.hip_bias_time_ramp = False
            cfg.sagittal_symmetry = True
            cfg.heading_control = False
            cfg.conditional_heading_control = False
            cfg.position_control = False
            cfg.pitch_control = False
            self._set_wheel_scale(5.0)
            self._heading_clip = 2.0
        else:  # TELEOP
            self._set_hip_bias(0.0, 0.0)
            self._set_hip_range(self.teleop_hip_range)
            self._set_hip_symmetry(
                self.teleop_hip_symmetry, self.teleop_hip_symmetry_gate
            )
            cfg.dynamic_hip_bias = False
            cfg.hip_bias_time_ramp = False
            cfg.sagittal_symmetry = False
            cfg.heading_control = True
            cfg.conditional_heading_control = True
            cfg.heading_mode = "override"
            cfg.target_heading = 0.0
            cfg.heading_hold_wz_threshold = 0.15
            cfg.heading_hold_max_lateral_accel = self.heading_hold_max_lateral_accel
            cfg.heading_max_yaw_rate = 2.0
            cfg.position_control = False
            cfg.pitch_control = False
            self._set_wheel_scale(5.0)
            self._heading_clip = 12.0
            # force a fresh lock of the current heading when teleop takes over
            self.action_term._heading_hold_active[:] = False

        # The scripted stance belongs to TELEOP only; recovery/stand must start
        # from the nominal stance, and TELEOP re-applies the gamepad target each step.
        self._set_stance(0.0, snap=True)

        if print_info:
            print(
                f"[handoff] phase={self.phase.value}  "
                f"hip_bias={self.action_term.cfg.hip_bias}  "
                f"hip_range={cfg.hip_range:.2f}  hip_sym={cfg.hip_symmetry}"
                f"{'/gated' if cfg.hip_symmetry_gate else ''}  "
                f"sagittal={cfg.sagittal_symmetry}  heading={cfg.heading_control}  "
                f"pitch_pd={cfg.pitch_control}  clip={self._heading_clip:.0f}"
            )

    # ------------------------------------------------------------------ #
    # phase machine
    # ------------------------------------------------------------------ #
    def _set_phase(self, phase: HandoffPhase) -> None:
        if phase == self.phase:
            return
        self.phase = phase
        self._hold = 0
        self._fall = 0
        self.phase_time = 0.0
        self.timeout_requested = False
        self._configure_phase(print_info=True)

    def _update_phase(self) -> None:
        self.phase_time += float(self.env.step_dt)
        if self.phase != HandoffPhase.TELEOP:
            limit = {
                HandoffPhase.RECOVERY: 8.0,
                HandoffPhase.ALIGN: 5.0,
                HandoffPhase.STAND: 5.0,
            }.get(self.phase, 10.0)
            if self.phase_time > limit:
                print(f"[handoff] {self.phase.value} timeout -> reset plant", flush=True)
                self.timeout_requested = True
                self.phase_time = 0.0
                return

        u = self._uprightness()[0]
        h = self._height()[0]
        still = self._still()[0]

        pitch = self._pitch()[0]
        pitch_rate = self.robot.data.root_ang_vel_b[0, 1]

        if self.phase == HandoffPhase.RECOVERY:
            # The recovery policy's endpoint is an upright squat (~0.10 m), not a
            # full stand.  Heap only the uprightness/stillness; ALIGN then levels
            # the body pitch before STAND rises.
            ok = (u > self.upright_enter) and (h < self.squat_height_max) and still
            self._hold = self._hold + 1 if ok else 0
            if self._hold >= self.hold_steps:
                self._set_phase(HandoffPhase.ALIGN)
        elif self.phase == HandoffPhase.ALIGN:
            fall = (u < self.fall_upright_min) or (h < self.fall_height_min)
            if fall:
                self._set_phase(HandoffPhase.RECOVERY)
                return
            # The pitch PD brings the squat pitch into a narrow window; hand over
            # to STAND at a favourable phase rather than waiting for it to settle
            # (the wheel/ground contact produces a small limit cycle).
            ok = (abs(float(pitch)) < 0.17) and (abs(float(pitch_rate)) < 1.0)
            self._hold = self._hold + 1 if ok else 0
            if self._hold >= max(1, self.hold_steps // 2):
                self._set_phase(HandoffPhase.STAND)
        elif self.phase == HandoffPhase.STAND:
            fall = (u < self.fall_upright_min) or (h < self.fall_height_min)
            self._fall = self._fall + 1 if fall else 0
            if self._fall >= self.fall_steps:
                self._set_phase(HandoffPhase.RECOVERY)
                return
            ok = (u > self.upright_enter) and (h > self.stand_height_enter) and still
            self._hold = self._hold + 1 if ok else 0
            if self._hold >= self.hold_steps:
                self._set_phase(HandoffPhase.TELEOP)
        else:  # TELEOP
            fall = (u < self.fall_upright_min) or (h < self.fall_height_min)
            self._fall = self._fall + 1 if fall else 0
            if self._fall >= self.fall_steps:
                self._set_phase(HandoffPhase.RECOVERY)

    # ------------------------------------------------------------------ #
    # command / action
    # ------------------------------------------------------------------ #
    def _set_command(self) -> None:
        cmd = self.command_term
        cmd.is_standing_env[:] = self.phase != HandoffPhase.TELEOP
        if self.phase == HandoffPhase.TELEOP:
            cmd.vel_command_b[:, 0] = self.drive_vx
            cmd.vel_command_b[:, 1] = 0.0
            cmd.vel_command_b[:, 2] = self.drive_wz
        else:
            cmd.vel_command_b[:, :] = 0.0
        cmd._update_command()

    def reset(self, phase: HandoffPhase = HandoffPhase.RECOVERY) -> None:
        """Call when the underlying environment auto-resets.

        ``phase`` is the phase that matches the reset pose: RECOVERY for the
        fallen recovery env, TELEOP for the standing teleop env.  Starting a
        recovery from a standing pose stalls (the recovery actor was trained
        only from falls), so the standing host must pass TELEOP.
        """
        self.phase = phase
        self._hold = 0
        self._fall = 0
        #: seconds spent in the current non-TELEOP phase (safety timeout)
        self.phase_time = 0.0
        #: set when a non-TELEOP phase exceeded its timeout; the runner resets
        self.timeout_requested = False
        self._configure_phase(print_info=True)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        self._update_phase()
        self._set_command()
        policy = (
            self.recovery_policy
            if self.phase in (HandoffPhase.RECOVERY, HandoffPhase.ALIGN)
            else self.teleop_policy
        )
        actions = policy(obs)
        return torch.clamp(actions, -self._heading_clip, self._heading_clip)

    # ------------------------------------------------------------------ #
    # logging
    # ------------------------------------------------------------------ #
    def log(self, step: int) -> None:
        u = float(self._uprightness()[0].item())
        h = float(self._height()[0].item())
        import math

        roll, pitch = 0.0, 0.0
        q = self.robot.data.root_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.asin(sinp)
        print(
            f"[handoff] step={step:4d} phase={self.phase.value:8s} "
            f"base_z={h:.3f} m  pitch={math.degrees(pitch):+6.1f} deg  "
            f"roll={math.degrees(roll):+5.1f} deg  upright={u:.3f}  "
            f"cmd=({self.drive_vx:+.2f}, {self.drive_wz:+.2f})  "
            f"stance={float(self.action_term.stance[0].item()):+.2f} rad",
            flush=True,
        )


__all__ = ["HandoffController", "HandoffPhase"]
