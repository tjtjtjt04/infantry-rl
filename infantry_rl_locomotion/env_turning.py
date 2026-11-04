"""Turning (direct yaw-rate command) environment for the five-bar wheel-leg.

This is the second task in the workspace.  The locomotion task answers "drive forward
at vx, keep the heading" and derives ``wz`` from a heading-error loop.  The turning
task answers "drive at vx and rotate at wz" directly, which is what a joystick sends
and what is needed for left/right turns and rotating on the spot.

What is shared with :mod:`env_locomotion` (and why):

* the scene, action term, terminations and the balance / speed reward terms are
  reused verbatim -- turning must not cost the robot its balance;
* the observation is the same 35-dim vector and the action is the same 6-dim vector,
  so a locomotion checkpoint can be loaded with ``--resume`` (see the README);
* the five-bar loop closure, hip/wheel scaling and the diagnostics are unchanged.

What is different:

* ``heading_command=False``: ``wz`` is sampled uniformly in ``[-wz_max, wz_max]``
  instead of being computed from a heading target;
* :class:`mdp.commands.TurnInPlaceVelocityCommand` forces a fraction of the commands
  to ``vx = 0`` with a sizeable ``wz``, so rotating in place is trained explicitly;
* the action-level PD heading loop is disabled (it would overwrite the policy's wheel
  differential, i.e. exactly the turning action we want the policy to learn).

Sign convention (measured in this simulation, not assumed): a POSITIVE ``wz`` command
is a counter-clockwise turn seen from above (+Z), i.e. "turn left" when driving in
+X.  ``play.py --cmd_wz`` uses the same sign.
"""

from __future__ import annotations

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

import mdp
from env_locomotion import (
    ActionsCfg,
    FiveBarEnv,
    LocomotionObservationsCfg,
    LocomotionRewardsCfg,
    RobotSceneCfg,
    TerminationsCfg,
)

# --------------------------------------------------------------------------- #
# Command envelope
# --------------------------------------------------------------------------- #
#: forward speed range (m/s).  Defaults match the 0.2-3.0 m/s locomotion run the
#: turning task is normally warm-started from.  Set INFANTRY_RL_VX_MIN=0.0 to also
#: train the 0-0.2 m/s band (useful for very gentle joystick inputs).
VX_MIN_MPS = float(os.environ.get("INFANTRY_RL_VX_MIN", "0.2"))
VX_MAX_MPS = float(os.environ.get("INFANTRY_RL_VX_MAX", "3.0"))

#: maximum |yaw rate| command (rad/s).  2.0 rad/s ~ 115 deg/s.  At 3 m/s this is a
#: 1.5 m turning radius; in place it needs wheel speeds of only ~6.5 rad/s.
WZ_MAX_RPS = float(os.environ.get("INFANTRY_RL_WZ_MAX", "2.0"))

#: fraction of resampled environments that get a pure in-place rotation command
TURN_IN_PLACE_PROB = float(os.environ.get("INFANTRY_RL_TURN_IN_PLACE_PROB", "0.25"))
#: smallest |wz| used by the in-place mode (rad/s): keeps it visibly a turn
TURN_IN_PLACE_MIN_WZ = float(os.environ.get("INFANTRY_RL_TURN_IN_PLACE_MIN_WZ", "0.3"))

#: cap on the commanded lateral acceleration |vx*wz| (m/s^2).  At 3 m/s this limits
#: the yaw command to 4.0/3 = 1.33 rad/s; at vx <= 2 m/s and in place the full
#: +/-WZ_MAX_RPS remains available.  0 disables the cap (the raw uniform product).
#: Rationale: a wheel-leg has to bank to corner; asking for 6 m/s^2 at 3 m/s + 2 rad/s
#: produces falls rather than useful training signal.
MAX_LATERAL_ACCEL = float(os.environ.get("INFANTRY_RL_MAX_LAT_ACCEL", "4.0"))

#: fraction of environments commanded to stand still (teaches braking); same as
#: locomotion.  An in-place rotation is NOT standing, so those envs override this flag.
REL_STANDING_ENVS = 0.1

#: command resampling interval (s).  Slightly shorter than locomotion (5-10 s) so each
#: 20 s episode contains 2-5 different (vx, wz) pairs, including in-place rotations.
RESAMPLING_TIME_RANGE = (4.0, 8.0)


# --------------------------------------------------------------------------- #
# Task config
# --------------------------------------------------------------------------- #
@configclass
class TurningCommandsCfg:
    """Direct ``(vx, wz)`` commands with an explicit in-place-rotation mode."""

    base_velocity = mdp.TurnInPlaceVelocityCommandCfg(
        asset_name="robot",
        # `wz` is the command itself, not a heading-loop output
        heading_command=False,
        rel_standing_envs=REL_STANDING_ENVS,
        resampling_time_range=RESAMPLING_TIME_RANGE,
        turn_in_place_prob=TURN_IN_PLACE_PROB,
        turn_in_place_min_abs=TURN_IN_PLACE_MIN_WZ,
        max_lateral_accel=MAX_LATERAL_ACCEL,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(VX_MIN_MPS, VX_MAX_MPS),
            lin_vel_y=(0.0, 0.0),  # not actuated on a differential drive
            ang_vel_z=(-WZ_MAX_RPS, WZ_MAX_RPS),
            heading=None,  # no heading target in this task
        ),
    )


@configclass
class TurningObservationsCfg(LocomotionObservationsCfg):
    """The locomotion observation, unchanged (35 dims).

    ``velocity_commands`` already returns ``(vx, wz)``; in this task both components
    are sampled commands instead of ``wz`` being derived from a heading target, so no
    observation term needs to change.
    """


@configclass
class TurningRewardsCfg(LocomotionRewardsCfg):
    """Balance and speed terms unchanged; yaw tracking now applies to direct commands.

    The inherited yaw terms already have enough authority to teach the new skill:

    * ``track_ang_vel_z`` (exp, +3.0) rewards matching the commanded rate;
    * ``track_ang_vel_z_l1`` (-1.0) and ``_l2`` (-2.0) give a constant / growing pull
      towards it, including when the robot is standing still with ``wz != 0``.

    Only one parameter is adjusted: ``roll_straight`` relaxes the roll penalty as the
    COMMANDED yaw rate grows, and that ramp has to end at this task's actual envelope
    (``WZ_MAX_RPS``) rather than at the locomotion heading controller's clamp (2.5).
    """

    def __post_init__(self):
        super().__post_init__()
        self.roll_straight.params["wz_scale"] = WZ_MAX_RPS


@configclass
class TurningEnvCfg(ManagerBasedRLEnvCfg):
    """Velocity-tracking environment with a directly commanded yaw rate."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: TurningObservationsCfg = TurningObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: TurningRewardsCfg = TurningRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: TurningCommandsCfg = TurningCommandsCfg()

    #: name of the action term that owns the loop-closing knee targets
    joint_action_name: str = "joints"

    def __post_init__(self):
        # general settings -- identical to locomotion
        self.decimation = 4  # 50 Hz control on a 200 Hz physics step
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15

        # -------------------------------------------------------------- #
        # GPU stability knobs (same environment variables as locomotion)
        # -------------------------------------------------------------- #
        _fabric_off = os.environ.get("INFANTRY_RL_USE_FABRIC", "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        )
        if _fabric_off:
            self.sim.use_fabric = False
        print(f"[cfg] PhysX fabric        : {'DISABLED (INFANTRY_RL_USE_FABRIC)' if _fabric_off else 'enabled (default)'}")

        _iters = os.environ.get("INFANTRY_RL_PHYSX_POS_ITERS")
        if _iters:
            try:
                self.sim.physx.max_position_iteration_count = int(_iters)
                print(f"[cfg] PhysX position iters: {int(_iters)} (INFANTRY_RL_PHYSX_POS_ITERS)")
            except ValueError:
                pass

        # -------------------------------------------------------------- #
        # The policy must own the wheel differential.
        #
        # The locomotion task can run with `INFANTRY_RL_HEADING_CONTROL=1`, where the
        # action term REPLACES the policy's differential wheel command with a PD loop
        # towards heading 0.  In this task that would cancel every turning command, so
        # the loop is forced off here no matter what the environment says.
        # -------------------------------------------------------------- #
        self.actions.joints.heading_control = False
        self.actions.joints.position_control = False
        if os.environ.get("INFANTRY_RL_HEADING_CONTROL", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            print(
                "[cfg] WARNING: INFANTRY_RL_HEADING_CONTROL=1 is set but is IGNORED by "
                "the turning task -- the action-level heading loop would overwrite the "
                "policy's turning commands.  Unset it for training/play of `turning`."
            )

        vx_min, vx_max = self.commands.base_velocity.ranges.lin_vel_x
        print(
            f"[cfg] command ranges      : vx [{vx_min:+.2f}, {vx_max:+.2f}] m/s, "
            f"wz +/-{self.commands.base_velocity.ranges.ang_vel_z[1]:.2f} rad/s (DIRECT), "
            f"standing envs {self.commands.base_velocity.rel_standing_envs:.0%}, "
            f"in-place turn {self.commands.base_velocity.turn_in_place_prob:.0%} "
            f"(|wz| >= {self.commands.base_velocity.turn_in_place_min_abs:.2f} rad/s)"
        )
        if self.commands.base_velocity.max_lateral_accel > 0.0:
            print(
                f"[cfg] lateral accel cap   : |vx*wz| <= "
                f"{self.commands.base_velocity.max_lateral_accel:.2f} m/s^2 "
                f"(INFANTRY_RL_MAX_LAT_ACCEL; 0 disables)"
            )
        else:
            print("[cfg] lateral accel cap   : DISABLED (INFANTRY_RL_MAX_LAT_ACCEL=0)")
        print(
            "[cfg] turning             : direct (vx, wz) commands; heading target not used "
            "(policy owns yaw, action-level heading loop disabled)"
        )

        # Viewer (same convention as locomotion)
        self.viewer.origin_type = "world"
        self.viewer.eye = (2.0, -2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.2)


@configclass
class TurningEnvCfg_PLAY(TurningEnvCfg):
    """Variant used for replay / scripted evaluation: no random command resampling."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1.0e6  # never time out during a replay
        self.observations.policy.enable_corruption = False
        # a fixed command (play.py --cmd_vx/--cmd_wz) owns the task
        self.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.turn_in_place_prob = 0.0


# --------------------------------------------------------------------------- #
# Environment class
# --------------------------------------------------------------------------- #
class FiveBarTurningEnv(FiveBarEnv):
    """Five-bar wheel-leg with direct ``(vx, wz)`` velocity commands.

    All loop-closure machinery and diagnostics live in :class:`FiveBarEnv`; this task
    only changes the command term and the task envelope.
    """

    cfg: TurningEnvCfg


__all__ = [
    "FiveBarTurningEnv",
    "MAX_LATERAL_ACCEL",
    "REL_STANDING_ENVS",
    "RESAMPLING_TIME_RANGE",
    "TURN_IN_PLACE_MIN_WZ",
    "TURN_IN_PLACE_PROB",
    "TurningCommandsCfg",
    "TurningEnvCfg",
    "TurningEnvCfg_PLAY",
    "TurningObservationsCfg",
    "TurningRewardsCfg",
    "VX_MAX_MPS",
    "VX_MIN_MPS",
    "WZ_MAX_RPS",
]
