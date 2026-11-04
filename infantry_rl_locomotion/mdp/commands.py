"""Command terms for the turning, braking and unified teleop tasks.

The locomotion task derives ``wz`` from a heading-error loop, which is great for
"hold a heading" but cannot express "turn left at this rate": the heading target has
to be re-sampled and the controller decides the rate.  A joystick (and the turning
task) needs the angular rate to be a first-class command, so this module samples
``(vx, wz)`` directly and adds an explicit **in-place rotation** mode.

Why a dedicated mode instead of just letting ``vx`` reach zero?

* With ``vx`` sampled uniformly from ``[0.2, 3.0]`` the robot would essentially never
  see ``vx = 0`` with a non-zero ``wz``, so it would never learn to rotate on the spot.
* ``rel_standing_envs`` cannot provide it either: that flag zeroes the WHOLE command,
  including ``wz``, so those environments learn braking, not rotation.
* A uniform ``(vx, wz)`` product distribution is dominated by "drive forward", and the
  tightest turn (``vx = 0``) is the hardest one to learn; sampling it explicitly gives
  it a fixed share of every rollout.

The mode is drawn at every command resample, so an environment alternates between
driving, turning while driving, rotating in place and standing still within one
episode.

The second addition is a feasibility cap on the lateral acceleration ``|vx*wz|``.
Turning while driving is not free: a two-wheel robot has to bank, and at 3 m/s a
2 rad/s command asks for 6 m/s^2 (about 0.6 g) of lateral acceleration.  Sampling
that combination only produces episodes that end in a fall, so the command is
redrawn inside a configurable envelope (default 4 m/s^2 for the turning task).  The
cap leaves in-place rotation and all moderate-speed turns untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass


class TurnInPlaceVelocityCommand(UniformVelocityCommand):
    """``(vx, wz)`` sampled directly, with a fraction of pure in-place rotations.

    The base class already samples ``lin_vel_x``, ``lin_vel_y`` and ``ang_vel_z``
    uniformly when ``heading_command=False``.  This subclass only post-processes a
    random subset of the resampled environments:

    * ``vx = 0`` -- no translation, so the robot must rotate about its own centre;
    * ``|wz|`` in ``[turn_in_place_min_abs, max(|ang_vel_z|)]`` with a random sign --
      the lower bound avoids commands that are numerically "turn" but visually
      indistinguishable from standing still;
    * ``is_standing_env = False`` -- otherwise the base class's standing pass would
      zero the command again.

    Everything else (resampling timing, standing environments, the command tensor
    layout) is inherited unchanged, so the observation term ``velocity_commands``
    keeps returning ``(vx, wz)``.
    """

    cfg: TurnInPlaceVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        # 1) the standard uniform draw (vx, vy=0, wz) + standing flags
        super()._resample_command(env_ids)

        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return

        wz_low, wz_high = self.cfg.ranges.ang_vel_z
        wz_max = max(abs(float(wz_low)), abs(float(wz_high)))

        # 2) feasibility cap: the lateral acceleration of a differential drive is
        #    vx * wz.  A wheel-leg cannot bank arbitrarily far, so commands such as
        #    (vx=3 m/s, wz=2 rad/s) demand 6 m/s^2 of lateral acceleration and are
        #    effectively impossible to track -- the policy falls trying.  Redraw wz
        #    uniformly inside |wz| <= max_lateral_accel / |vx| instead.  In-place
        #    commands (vx=0) are unaffected because the cap then equals wz_max.
        if self.cfg.max_lateral_accel > 0.0 and wz_max > 0.0:
            vx_abs = self.vel_command_b[ids, 0].abs().clamp(min=0.05)
            cap = torch.clamp(self.cfg.max_lateral_accel / vx_abs, max=wz_max)
            u = torch.rand(ids.numel(), device=self.device)
            self.vel_command_b[ids, 2] = (2.0 * u - 1.0) * cap

        if self.cfg.turn_in_place_prob <= 0.0:
            return

        # 3) pick the in-place subset with a Bernoulli draw
        turn_ids = ids[torch.rand(ids.numel(), device=self.device) < self.cfg.turn_in_place_prob]
        if turn_ids.numel() == 0:
            return

        # 4) pure rotation: zero linear speed and a sizeable yaw rate
        min_abs = min(float(self.cfg.turn_in_place_min_abs), wz_max)
        n = turn_ids.numel()
        magnitude = torch.empty(n, device=self.device).uniform_(min_abs, wz_max)
        sign = torch.where(
            torch.rand(n, device=self.device) < 0.5,
            -torch.ones(n, device=self.device),
            torch.ones(n, device=self.device),
        )
        self.vel_command_b[turn_ids, 0] = 0.0
        self.vel_command_b[turn_ids, 1] = 0.0
        self.vel_command_b[turn_ids, 2] = magnitude * sign
        # the base class's `_update_command` zeroes standing envs AFTER resampling
        self.is_standing_env[turn_ids] = False


@configclass
class TurnInPlaceVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`TurnInPlaceVelocityCommand`."""

    class_type: type = TurnInPlaceVelocityCommand

    #: probability that an environment's newly sampled command is a pure rotation
    turn_in_place_prob: float = 0.25
    #: smallest |wz| (rad/s) used by the in-place mode; keeps the command visibly a turn
    turn_in_place_min_abs: float = 0.3
    #: cap on the commanded lateral acceleration ``|vx * wz|`` in m/s^2.  Values above
    #: this are physically unreachable for a two-wheel robot (it would need too much
    #: bank), so they only teach the policy to fall.  0 disables the cap.
    max_lateral_accel: float = 0.0


# --------------------------------------------------------------------------- #
# Braking
# --------------------------------------------------------------------------- #
class BrakingVelocityCommand(UniformVelocityCommand):
    """Drive at a target speed, then command an emergency stop.

    Each environment follows a two-phase schedule for one episode:

    1. **drive**: ``vx = target_speed`` for a randomly sampled ``drive_time``.  The
       robot starts from rest, so this phase also trains the acceleration needed to
       reach the target before the brake.
    2. **brake**: ``vx = 0`` for the rest of the episode.  The command step is the
       "emergency brake" signal; the policy must stop quickly and then balance.

    The term records, for every environment, the position and speed at the moment
    the brake command starts, the time it takes to get below
    ``stop_speed_threshold``, and the distance travelled by then.  The reward terms
    in :mod:`mdp.rewards` use these to reward a SHORT stopping distance rather than
    a specific deceleration profile, which is what allows a small amount of wheel
    slip: the robot may skid if that shortens the stop.

    No mid-episode resampling is performed (``resampling_time_range`` is set very
    large by the task config); the two phases are the whole episode.
    """

    cfg: BrakingVelocityCommandCfg

    def __init__(self, cfg: BrakingVelocityCommandCfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        #: target speed of the drive phase (m/s), ``(N,)``
        self.target_speed = torch.zeros(n, device=self.device)
        #: control steps since this episode's command was sampled
        self._step_count = torch.zeros(n, dtype=torch.long, device=self.device)
        #: control steps of the drive phase, sampled per environment
        self._drive_steps = torch.ones(n, dtype=torch.long, device=self.device)
        #: True once the brake command has been issued
        self.brake_started = torch.zeros(n, dtype=torch.bool, device=self.device)
        #: True once the robot has been below the stop thresholds after braking
        self.stopped = torch.zeros(n, dtype=torch.bool, device=self.device)
        #: world-frame planar position at the moment the brake started, ``(N, 2)``
        self.brake_position = torch.zeros(n, 2, device=self.device)
        #: |vx| at the moment the brake started (m/s), ``(N,)``
        self.brake_speed = torch.zeros(n, device=self.device)
        #: control-step indices of the brake onset and of the first stop
        self.brake_step = torch.zeros(n, dtype=torch.long, device=self.device)
        self.stop_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        #: planar distance travelled when `stopped` first became True (m)
        self.stop_distance = torch.zeros(n, device=self.device)

        # command-manager metrics (logged to TensorBoard at every reset)
        self.metrics["brake_speed"] = torch.zeros(n, device=self.device)
        self.metrics["brake_distance"] = torch.zeros(n, device=self.device)
        self.metrics["time_to_stop"] = torch.zeros(n, device=self.device)
        self.metrics["stopped"] = torch.zeros(n, device=self.device)

    # ------------------------------------------------------------------ #
    # schedule
    # ------------------------------------------------------------------ #
    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        r = torch.empty(ids.numel(), device=self.device)
        self.target_speed[ids] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        drive_s = torch.empty(ids.numel(), device=self.device).uniform_(*self.cfg.drive_time_range)
        self._drive_steps[ids] = (drive_s / self._env.step_dt).long().clamp(min=1)

        self._step_count[ids] = 0
        self.brake_started[ids] = False
        self.stopped[ids] = False
        self.brake_position[ids] = 0.0
        self.brake_speed[ids] = 0.0
        self.brake_step[ids] = 0
        self.stop_step[ids] = -1
        self.stop_distance[ids] = 0.0

        self.vel_command_b[ids, 0] = self.target_speed[ids]
        self.vel_command_b[ids, 1] = 0.0
        self.vel_command_b[ids, 2] = 0.0
        self.is_standing_env[ids] = False
        self.is_heading_env[ids] = False

    def _update_command(self):
        self._step_count += 1
        brake_now = self._step_count >= self._drive_steps

        # record the brake onset (position/speed) exactly once per episode
        newly = brake_now & ~self.brake_started
        if newly.any():
            ids = newly.nonzero(as_tuple=False).flatten()
            self.brake_started[ids] = True
            self.brake_step[ids] = self._step_count[ids]
            self.brake_position[ids] = self.robot.data.root_pos_w[ids, :2]
            self.brake_speed[ids] = self.robot.data.root_lin_vel_b[ids, 0].abs()

        drive_ids = (~brake_now).nonzero(as_tuple=False).flatten()
        brake_ids = brake_now.nonzero(as_tuple=False).flatten()
        if drive_ids.numel() > 0:
            self.vel_command_b[drive_ids, 0] = self.target_speed[drive_ids]
            self.vel_command_b[drive_ids, 1] = 0.0
            self.vel_command_b[drive_ids, 2] = 0.0
        if brake_ids.numel() > 0:
            self.vel_command_b[brake_ids, :] = 0.0

        # first instant the robot is slow AND not spinning -> "stopped"
        if self.brake_started.any():
            speed = self.robot.data.root_lin_vel_b[:, 0].abs()
            yaw_rate = self.robot.data.root_ang_vel_b[:, 2].abs()
            stop_now = (
                self.brake_started
                & ~self.stopped
                & (speed < self.cfg.stop_speed_threshold)
                & (yaw_rate < self.cfg.stop_yaw_threshold)
            )
            if stop_now.any():
                ids = stop_now.nonzero(as_tuple=False).flatten()
                self.stopped[ids] = True
                self.stop_step[ids] = self._step_count[ids]
                delta = self.robot.data.root_pos_w[ids, :2] - self.brake_position[ids]
                self.stop_distance[ids] = torch.norm(delta, dim=-1)

    def _update_metrics(self):
        """Braking metrics logged by the command manager at every reset."""
        dt = float(self._env.step_dt)
        pos = self.robot.data.root_pos_w[:, :2]
        distance = torch.norm(pos - self.brake_position, dim=-1)
        self.metrics["brake_speed"].copy_(
            torch.where(self.brake_started, self.brake_speed, torch.zeros_like(self.brake_speed))
        )
        self.metrics["brake_distance"].copy_(
            torch.where(self.brake_started, distance, torch.zeros_like(distance))
        )
        stopped_time = (self.stop_step - self.brake_step).float() * dt
        elapsed = (self._step_count - self.brake_step).float() * dt
        self.metrics["time_to_stop"].copy_(
            torch.where(
                self.stopped,
                stopped_time,
                torch.where(self.brake_started, elapsed, torch.zeros_like(elapsed)),
            )
        )
        self.metrics["stopped"].copy_(self.stopped.float())


@configclass
class BrakingVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`BrakingVelocityCommand`.

    ``ranges.lin_vel_x`` is the DRIVE-phase target speed; ``ranges.lin_vel_y`` and
    ``ranges.ang_vel_z`` are forced to zero by the implementation.
    """

    class_type: type = BrakingVelocityCommand

    #: duration of the drive phase before the brake command (s)
    drive_time_range: tuple[float, float] = (3.0, 5.0)
    #: |vx| below this counts as stopped (m/s)
    stop_speed_threshold: float = 0.1
    #: |wz| below this counts as stopped (rad/s)
    stop_yaw_threshold: float = 0.3


# --------------------------------------------------------------------------- #
# Unified teleop
# --------------------------------------------------------------------------- #
class TeleopVelocityCommand(UniformVelocityCommand):
    """Joystick-like ``(vx, wz)`` commands with random stops and explicit brakes.

    This is the command vocabulary a hand controller actually produces, so it is a
    single policy instead of switching between a locomotion, turning and braking
    checkpoint:

    * **moving**: ``vx`` is uniform in ``ranges.lin_vel_x`` (forward AND reverse);
      ``wz`` is uniform in ``ranges.ang_vel_z`` and redrawn under the feasibility
      cap ``|vx*wz| <= max_lateral_accel``;
    * **in-place**: a fraction of the moving samples gets ``vx = 0`` and
      ``|wz| in [turn_in_place_min_abs, turn_in_place_max_abs]``.  The in-place
      range is separate from the driving range, so the robot can learn a faster
      spin than it is allowed to combine with a forward speed;
    * **standing**: ``rel_standing_envs`` of every resample is ``(0, 0)``;
    * **low-speed creep**: ``low_speed_prob`` of the remaining moving samples is
      drawn in ``[low_speed_min, low_speed_max]`` (default 0.05-0.20 m/s) with the
      full yaw range, so small-stick forward + turning is trained together.  The
      band is clamped to the active vx range and the mode is disabled when they do
      not overlap (e.g. a 2.0-2.5 focus stage);
    * **brake**: ``brake_prob`` forces ``(0, 0)``.  If the PREVIOUS command was
      moving, this is recorded as a brake event (onset speed, position and time),
      which the braking rewards in :mod:`mdp.rewards` consume.

    The command is resampled every ``resampling_time_range`` seconds (2-4 s for the
    teleop task), so one episode contains several starts, turns and stops.  Unlike
    :class:`BrakingVelocityCommand` there is no fixed drive-then-brake schedule: a
    brake can arrive from any speed and any yaw rate, which is the behaviour a
    joystick needs.

    Brake bookkeeping mirrors :class:`BrakingVelocityCommand` on purpose
    (``brake_started`` / ``brake_position`` / ``brake_speed`` / ``stop_distance``),
    so :func:`mdp.rewards.brake_distance_exp`, ``_l1``, ``brake_overshoot`` and
    ``brake_standstill`` work unchanged.  A brake event ends at the next resample;
    the metrics average the completed events of the current episode.
    """

    cfg: TeleopVelocityCommandCfg

    def __init__(self, cfg: TeleopVelocityCommandCfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        #: True for the envs whose command was resampled by an episode reset.  A
        #: reset must NOT be mistaken for a moving->zero transition (that would
        #: record a phantom brake at the spawn pose).
        self._resetting = torch.zeros(n, dtype=torch.bool, device=self.device)
        #: control steps since this episode's reset, used for brake timing
        self._step_count = torch.zeros(n, dtype=torch.long, device=self.device)

        #: True while a brake event is being evaluated for this environment
        self.brake_started = torch.zeros(n, dtype=torch.bool, device=self.device)
        #: True once the robot has been below both stop thresholds during the event
        self.stopped = torch.zeros(n, dtype=torch.bool, device=self.device)
        #: world-frame planar position at the instant the brake command arrived
        self.brake_position = torch.zeros(n, 2, device=self.device)
        #: |vx| at the instant the brake command arrived (m/s)
        self.brake_speed = torch.zeros(n, device=self.device)
        self.brake_step = torch.zeros(n, dtype=torch.long, device=self.device)
        self.stop_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        #: planar distance from `brake_position` when `stopped` first became True
        self.stop_distance = torch.zeros(n, device=self.device)

        # per-episode aggregates of COMPLETED brake events (plus the active one when
        # logging), so the console's "braking" line is meaningful even though a
        # teleop episode has several stop events instead of one.
        self._events = torch.zeros(n, device=self.device)
        self._dist_sum = torch.zeros(n, device=self.device)
        self._time_sum = torch.zeros(n, device=self.device)
        self._stop_sum = torch.zeros(n, device=self.device)
        self._speed_sum = torch.zeros(n, device=self.device)

        # command-manager metrics (logged by the command manager at every reset)
        self.metrics["brake_speed"] = torch.zeros(n, device=self.device)
        self.metrics["brake_distance"] = torch.zeros(n, device=self.device)
        self.metrics["time_to_stop"] = torch.zeros(n, device=self.device)
        self.metrics["stopped"] = torch.zeros(n, device=self.device)
        #: brake events per environment (the denominator the other four metrics need)
        self.metrics["brake_events"] = torch.zeros(n, device=self.device)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _ids(self, env_ids: Sequence[int] | slice | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()

    # ------------------------------------------------------------------ #
    # episode reset: flag it so the transition detector can skip it
    # ------------------------------------------------------------------ #
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = self._ids(env_ids)
        self._resetting[ids] = True
        # Log event-weighted brake metrics instead of averaging per-env rates: an
        # environment with NO brake event in this episode has no stop to succeed or
        # fail at, but a plain per-env mean would count it as `stopped = 0` and make
        # the console look far worse than the real success rate.  The value is
        # written identically to every resetting env, so CommandTerm.reset's mean
        # over those envs returns the aggregate itself.
        self._write_episode_brake_metrics(ids)
        return super().reset(env_ids)

    def _write_episode_brake_metrics(self, ids: torch.Tensor) -> None:
        if ids.numel() == 0:
            return
        active = self.brake_started[ids]
        events = self._events[ids] + active.float()
        total_events = events.sum()
        denom = total_events.clamp(min=1.0)

        robot = self.robot.data
        distance_now = torch.norm(
            robot.root_pos_w[ids, :2] - self.brake_position[ids], dim=-1
        )
        dt = float(self._env.step_dt)
        time_now = torch.where(
            self.stopped[ids],
            (self.stop_step[ids] - self.brake_step[ids]).float() * dt,
            (self._step_count[ids] - self.brake_step[ids]).float() * dt,
        )
        successes = self._stop_sum[ids] + (active & self.stopped[ids]).float()
        distance_sum = self._dist_sum[ids] + torch.where(
            active, distance_now, torch.zeros_like(distance_now)
        )
        time_sum = self._time_sum[ids] + torch.where(
            active, time_now, torch.zeros_like(time_now)
        )
        speed_sum = self._speed_sum[ids] + torch.where(
            active, self.brake_speed[ids], torch.zeros_like(self.brake_speed[ids])
        )

        self.metrics["stopped"][ids] = successes.sum() / denom
        self.metrics["brake_distance"][ids] = distance_sum.sum() / denom
        self.metrics["time_to_stop"][ids] = time_sum.sum() / denom
        self.metrics["brake_speed"][ids] = speed_sum.sum() / denom
        self.metrics["brake_events"][ids] = total_events / float(ids.numel())

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    def _resample_command(self, env_ids: Sequence[int]):
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return

        on_reset = self._resetting[ids]
        # command BEFORE this draw: a non-zero row followed by (0, 0) is a brake
        prev_motion = (
            (self.vel_command_b[ids, 0].abs() > 1e-6)
            | (self.vel_command_b[ids, 2].abs() > 1e-6)
        )

        # 1) standard uniform (vx, vy=0, wz) draw + standing flags
        super()._resample_command(env_ids)

        # 2) standing envs: zero the whole command immediately, not only in
        #    `_update_command` (the brake detector below reads the command tensor)
        standing = self.is_standing_env[ids]
        if standing.any():
            self.vel_command_b[ids[standing], :] = 0.0

        # 3) forced stops: `brake_prob` applies to the non-standing remainder, so
        #    the total zero share is 1 - (1-rel_standing)*(1-brake_prob).
        candidates = ids[~standing]
        if candidates.numel() > 0 and self.cfg.brake_prob > 0.0:
            pick = torch.rand(candidates.numel(), device=self.device) < self.cfg.brake_prob
            stop_ids = candidates[pick]
            if stop_ids.numel() > 0:
                self.vel_command_b[stop_ids, :] = 0.0
                self.is_standing_env[stop_ids] = True

        # 4) feasibility cap on the lateral acceleration of the DRIVING commands.
        #    |vx*wz| grows with both, and a wheel-leg cannot bank beyond ~4 m/s^2.
        #    In-place turns are handled after this and are not capped by a tiny
        #    `vx` (they would otherwise lose their larger spin range).
        moving = ~self.is_standing_env[ids]
        moving_ids = ids[moving]
        wz_low, wz_high = self.cfg.ranges.ang_vel_z
        wz_max = max(abs(float(wz_low)), abs(float(wz_high)))
        if moving_ids.numel() > 0 and self.cfg.max_lateral_accel > 0.0 and wz_max > 0.0:
            vx_abs = self.vel_command_b[moving_ids, 0].abs().clamp(min=0.05)
            cap = torch.clamp(self.cfg.max_lateral_accel / vx_abs, max=wz_max)
            u = torch.rand(moving_ids.numel(), device=self.device)
            self.vel_command_b[moving_ids, 2] = (2.0 * u - 1.0) * cap

        # 5) in-place rotations: vx=0, |wz| up to the separate in-place maximum
        if moving_ids.numel() > 0 and self.cfg.turn_in_place_prob > 0.0:
            pick = torch.rand(moving_ids.numel(), device=self.device) < self.cfg.turn_in_place_prob
            turn_ids = moving_ids[pick]
            if turn_ids.numel() > 0:
                n = turn_ids.numel()
                lower = min(float(self.cfg.turn_in_place_min_abs), float(self.cfg.turn_in_place_max_abs))
                upper = max(float(self.cfg.turn_in_place_min_abs), float(self.cfg.turn_in_place_max_abs))
                magnitude = torch.empty(n, device=self.device).uniform_(lower, upper)
                sign = torch.where(
                    torch.rand(n, device=self.device) < 0.5,
                    -torch.ones(n, device=self.device),
                    torch.ones(n, device=self.device),
                )
                self.vel_command_b[turn_ids, 0] = 0.0
                self.vel_command_b[turn_ids, 1] = 0.0
                self.vel_command_b[turn_ids, 2] = magnitude * sign
                self.is_standing_env[turn_ids] = False

        # 6) low-speed creep / fine steering.  A uniform draw over a 0-2.5 m/s range
        #    gives only ~6-8% of the moving samples below 0.2 m/s, while a gamepad
        #    spends a lot of time there.  Oversample it explicitly on a subset of the
        #    remaining moving commands, with the full yaw range so low-speed
        #    forward + turning is trained together (in-place is excluded).
        moving_ids = ids[~self.is_standing_env[ids]]
        if (
            moving_ids.numel() > 0
            and self.cfg.low_speed_prob > 0.0
            and self.cfg.low_speed_max > self.cfg.low_speed_min
        ):
            in_place = (self.vel_command_b[moving_ids, 0].abs() < 1e-6) & (
                self.vel_command_b[moving_ids, 2].abs() > 1e-6
            )
            low_candidates = moving_ids[~in_place]
            if low_candidates.numel() > 0:
                pick = torch.rand(low_candidates.numel(), device=self.device) < self.cfg.low_speed_prob
                low_ids = low_candidates[pick]
                if low_ids.numel() > 0:
                    n = low_ids.numel()
                    vx = torch.empty(n, device=self.device).uniform_(
                        self.cfg.low_speed_min, self.cfg.low_speed_max
                    )
                    self.vel_command_b[low_ids, 0] = vx
                    self.vel_command_b[low_ids, 1] = 0.0
                    # At these speeds the lateral-accel cap is effectively inactive,
                    # so a tight low-speed turn is deliberately included.
                    if self.cfg.max_lateral_accel > 0.0 and wz_max > 0.0:
                        cap = torch.clamp(self.cfg.max_lateral_accel / vx.clamp(min=0.05), max=wz_max)
                        u = torch.rand(n, device=self.device)
                        self.vel_command_b[low_ids, 2] = (2.0 * u - 1.0) * cap
                    else:
                        self.vel_command_b[low_ids, 2] = torch.empty(n, device=self.device).uniform_(
                            *self.cfg.ranges.ang_vel_z
                        )
                    self.is_standing_env[low_ids] = False

        # 7) brake bookkeeping.  A periodic resample ends any active event; a reset
        #    throws the event away (its metrics were already logged by super.reset).
        reset_ids = ids[on_reset]
        if reset_ids.numel() > 0:
            self._reset_brake_state(reset_ids)
            self._step_count[reset_ids] = 0

        resample_ids = ids[~on_reset]
        if resample_ids.numel() > 0:
            self._finalize_brake_events(resample_ids)
            self._clear_brake_state(resample_ids)
            new_zero = (
                (self.vel_command_b[resample_ids, 0].abs() < 1e-6)
                & (self.vel_command_b[resample_ids, 2].abs() < 1e-6)
            )
            was_moving = prev_motion[~on_reset]
            start_ids = resample_ids[was_moving & new_zero]
            if start_ids.numel() > 0:
                self._begin_brake(start_ids)

        self._resetting[ids] = False

    # ------------------------------------------------------------------ #
    # brake event state
    # ------------------------------------------------------------------ #
    def _begin_brake(self, ids: torch.Tensor) -> None:
        ids = self._ids(ids)
        if ids.numel() == 0:
            return
        self.brake_started[ids] = True
        self.stopped[ids] = False
        self.brake_position[ids] = self.robot.data.root_pos_w[ids, :2]
        self.brake_speed[ids] = self.robot.data.root_lin_vel_b[ids, 0].abs()
        self.brake_step[ids] = self._step_count[ids]
        self.stop_step[ids] = -1
        self.stop_distance[ids] = 0.0

    def _clear_brake_state(self, ids: torch.Tensor) -> None:
        self.brake_started[ids] = False
        self.stopped[ids] = False
        self.stop_step[ids] = -1
        self.stop_distance[ids] = 0.0
        self.brake_speed[ids] = 0.0
        self.brake_position[ids] = 0.0

    def _reset_brake_state(self, ids: torch.Tensor) -> None:
        self._clear_brake_state(ids)
        self._events[ids] = 0.0
        self._dist_sum[ids] = 0.0
        self._time_sum[ids] = 0.0
        self._stop_sum[ids] = 0.0
        self._speed_sum[ids] = 0.0
        self.brake_step[ids] = 0

    def _finalize_brake_events(self, ids: torch.Tensor) -> None:
        """Fold still-active events into the per-episode averages before clearing."""
        active = self.brake_started[ids]
        a_ids = ids[active]
        if a_ids.numel() == 0:
            return
        robot = self.robot.data
        distance_now = torch.norm(robot.root_pos_w[a_ids, :2] - self.brake_position[a_ids], dim=-1)
        stopped = self.stopped[a_ids]

        dt = float(self._env.step_dt)
        time_to_stop = (self.stop_step[a_ids] - self.brake_step[a_ids]).float() * dt
        time_now = (self._step_count[a_ids] - self.brake_step[a_ids]).float() * dt
        final_distance = torch.where(stopped, self.stop_distance[a_ids], distance_now)
        final_time = torch.where(stopped, time_to_stop, time_now)

        self._events[a_ids] += 1.0
        self._dist_sum[a_ids] += final_distance
        self._time_sum[a_ids] += final_time
        self._stop_sum[a_ids] += stopped.float()
        self._speed_sum[a_ids] += self.brake_speed[a_ids]

    def force_brake(self, env_ids: Sequence[int] | None = None) -> None:
        """Zero the command and start a brake event now (play / a joystick button).

        Already-active events are not restarted, so calling this every frame keeps a
        single measurement instead of resetting its onset position.
        """
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        fresh = ids[~self.brake_started[ids]]
        if fresh.numel() > 0:
            self._begin_brake(fresh)
        self.vel_command_b[ids, :] = 0.0
        self.is_standing_env[ids] = True

    # ------------------------------------------------------------------ #
    # per-step command / stop detection
    # ------------------------------------------------------------------ #
    def _update_command(self):
        self._step_count += 1
        # heading_command is False; this only enforces standing envs at zero
        super()._update_command()

        if self.brake_started.any():
            robot = self.robot.data
            speed = robot.root_lin_vel_b[:, 0].abs()
            yaw_rate = robot.root_ang_vel_b[:, 2].abs()
            stop_now = (
                self.brake_started
                & ~self.stopped
                & (speed < self.cfg.brake_speed_threshold)
                & (yaw_rate < self.cfg.brake_yaw_threshold)
            )
            if stop_now.any():
                ids = stop_now.nonzero(as_tuple=False).flatten()
                self.stopped[ids] = True
                self.stop_step[ids] = self._step_count[ids]
                delta = robot.root_pos_w[ids, :2] - self.brake_position[ids]
                self.stop_distance[ids] = torch.norm(delta, dim=-1)

    def _update_metrics(self):
        """Per-episode brake averages: completed events + the currently active one."""
        super()._update_metrics()

        robot = self.robot.data
        active = self.brake_started
        denom = (self._events + active.float()).clamp(min=1.0)

        distance_now = torch.norm(robot.root_pos_w[:, :2] - self.brake_position, dim=-1)
        dt = float(self._env.step_dt)
        time_to_stop = (self.stop_step - self.brake_step).float() * dt
        time_now = (self._step_count - self.brake_step).float() * dt
        active_time = torch.where(self.stopped, time_to_stop, time_now)

        self.metrics["brake_speed"].copy_(
            (self._speed_sum + torch.where(active, self.brake_speed, torch.zeros_like(self.brake_speed)))
            / denom
        )
        self.metrics["brake_distance"].copy_(
            (self._dist_sum + torch.where(active, distance_now, torch.zeros_like(distance_now))) / denom
        )
        self.metrics["time_to_stop"].copy_(
            (self._time_sum + torch.where(active, active_time, torch.zeros_like(active_time))) / denom
        )
        stop_count = self._stop_sum + (active & self.stopped).float()
        self.metrics["stopped"].copy_(stop_count / denom)
        self.metrics["brake_events"].copy_(self._events + active.float())


@configclass
class TeleopVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`TeleopVelocityCommand`.

    ``ranges.lin_vel_x`` / ``ranges.ang_vel_z`` define NORMAL driving.  The in-place
    rotation range is separate (``turn_in_place_*``), so a joystick-style "spin"
    can be faster than the yaw rate allowed while driving fast.
    """

    class_type: type = TeleopVelocityCommand

    #: fraction of moving samples turned into pure in-place rotations
    turn_in_place_prob: float = 0.2
    #: smallest / largest |wz| (rad/s) used by the in-place mode
    turn_in_place_min_abs: float = 0.5
    turn_in_place_max_abs: float = 2.0
    #: feasibility cap |vx*wz| for driving commands (m/s^2); 0 disables.
    max_lateral_accel: float = 4.0
    #: fraction of the NON-standing samples forced to (0, 0).  If the previous
    #: command was moving, that transition starts a brake event.
    brake_prob: float = 0.2
    #: |vx| / |wz| below these count as stopped during a brake event
    brake_speed_threshold: float = 0.1
    brake_yaw_threshold: float = 0.3

    # -- low-speed creep / fine steering (P2) --------------------------------
    #: fraction of the remaining moving samples forced into the low-speed band
    low_speed_prob: float = 0.25
    #: low-speed band (m/s); the env config clamps it to the active vx range and
    #: disables the mode when the two do not overlap (e.g. a 2.0-2.5 focus stage)
    low_speed_min: float = 0.05
    low_speed_max: float = 0.2


__all__ = [
    "BrakingVelocityCommand",
    "BrakingVelocityCommandCfg",
    "TeleopVelocityCommand",
    "TeleopVelocityCommandCfg",
    "TurnInPlaceVelocityCommand",
    "TurnInPlaceVelocityCommandCfg",
]
