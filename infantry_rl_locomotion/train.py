"""Train the infantry-rl five-bar wheel-leg with RSL-RL (flat, self-contained).

The training stack is unchanged from the original workspace: Isaac Lab's
manager-based environment, the five-bar loop-closure action term and rsl_rl's
``OnPolicyRunner``.  What changed is the plumbing -- no Hydra task registry and no
``gym.make``: the environment is constructed directly from its config.

Examples (run from this folder):
    python train.py --task locomotion --headless --num_envs 1024 --iterations 8000
    python train.py --task locomotion --check --num_envs 2
    python train.py --task locomotion --headless --max_iterations 3000 --resume --resume_latest
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RSL-RL agent for the infantry-rl robot.")
    # task / environment
    parser.add_argument("--task", type=str, default="locomotion",
                        help="locomotion | turning | braking | teleop | recovery | stand | "
                             "balance | squat | legacy Infantry-RL-* ID")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vx_min", type=float, default=None, help="locomotion forward command lower bound (m/s)")
    parser.add_argument("--vx_max", type=float, default=None, help="locomotion forward command upper bound (m/s)")
    parser.add_argument("--wz_max", type=float, default=None,
                        help="turning/teleop task: maximum |yaw rate| command while driving "
                             "(rad/s); turning default 2.0, teleop default 1.5")
    parser.add_argument("--turn_prob", type=float, default=None,
                        help="turning/teleop task: fraction of moving resamples given a pure "
                             "in-place rotation command (turning default 0.25, teleop 0.2)")
    parser.add_argument("--max_lat_accel", type=float, default=None,
                        help="turning/teleop task: cap on the commanded lateral acceleration "
                             "|vx*wz| in m/s^2 (default 4.0; 0 disables). Removes unreachable "
                             "high-speed sharp turns such as 3 m/s + 2 rad/s.")
    parser.add_argument("--teleop_vx_min", type=float, default=None,
                        help="teleop task: forward command lower bound (m/s); default -0.8. "
                             "Use 0.0 for the first forward-only stage.")
    parser.add_argument("--teleop_vx_max", type=float, default=None,
                        help="teleop task: forward command upper bound (m/s); default 3.0")
    parser.add_argument("--teleop_wz_inplace_max", type=float, default=None,
                        help="teleop task: maximum |yaw rate| for a pure in-place rotation "
                             "(rad/s); default 2.0. The driving cap is --wz_max (default 1.5).")
    parser.add_argument("--teleop_brake_prob", type=float, default=None,
                        help="teleop task: fraction of NON-standing resamples forced to "
                             "(vx=0, wz=0) and therefore trained as a brake (default 0.2)")
    parser.add_argument("--teleop_resample_min", type=float, default=None,
                        help="teleop task: minimum seconds between command resamples (default 2.0)")
    parser.add_argument("--teleop_resample_max", type=float, default=None,
                        help="teleop task: maximum seconds between command resamples (default 4.0)")
    parser.add_argument("--teleop_low_speed_prob", type=float, default=None,
                        help="teleop task: fraction of remaining moving samples overdrawn in "
                             "the low-speed creep band (default 0.25; P2, 0 disables)")
    parser.add_argument("--teleop_low_speed_min", type=float, default=None,
                        help="teleop task: low-speed creep band lower bound, m/s (default 0.05)")
    parser.add_argument("--teleop_low_speed_max", type=float, default=None,
                        help="teleop task: low-speed creep band upper bound, m/s (default 0.20)")
    parser.add_argument("--teleop_heading_hold_wz", type=float, default=None,
                        help="teleop task: |wz_cmd| below which the conditional heading PD "
                             "holds the current heading (rad/s; default 0.15)")
    parser.add_argument("--teleop_heading_lat_accel", type=float, default=None,
                        help="teleop task: cap on the conditional PD correction's lateral "
                             "acceleration |vx*wz_pd| (m/s^2; default 4.0, 0 disables). "
                             "Keep it equal to --max_lat_accel.")
    parser.add_argument("--teleop_turn_min_wz", type=float, default=None,
                        help="teleop task: minimum |wz| used by the in-place mode (rad/s; "
                             "default 0.5). Raise it to oversample fast spins.")
    parser.add_argument("--stance_randomize", action="store_true", default=False,
                        help="teleop task: randomize a symmetric leg stance (crouch / "
                             "nominal / extend) during in-place spins. Used to train the "
                             "A+RB extension; deployment drives the same bias from the gamepad.")
    parser.add_argument("--stance_min", type=float, default=None,
                        help="stance randomization: crouch hip offset in rad (default -0.35; "
                             "base_z ~0.122 m, the deep mechanical crouch)")
    parser.add_argument("--stance_max", type=float, default=None,
                        help="stance randomization: extension hip offset in rad (default 0.30; "
                             "+0.30 raises the body about 6 cm)")
    parser.add_argument("--stance_slew", type=float, default=None,
                        help="stance change rate in rad/s (default 0.5; also used at deployment)")
    parser.add_argument("--stance_gate_vx", type=float, default=None,
                        help="only randomize the stance below this |vx| command (default 0.10 m/s)")
    parser.add_argument("--stance_gate_wz", type=float, default=None,
                        help="only randomize the stance above this |wz| command (default 1.0 rad/s; "
                             "covers the A-spin spin-up while the |vx| gate protects straight driving)")
    parser.add_argument("--stance_prob_crouch", type=float, default=None,
                        help="stance sampling probability of the crouch branch (default 0.30; "
                             "use 0.5 for the crouch-default spin recipe)")
    parser.add_argument("--stance_prob_nominal", type=float, default=None,
                        help="stance sampling probability of the nominal branch (default 0.20; "
                             "the remainder is extension)")
    parser.add_argument("--stance_hold_weight", type=float, default=None,
                        help="weight of the gated stance-hold term (default -2.0; use -4.0 "
                             "if the policy cancels the extension)")
    parser.add_argument("--stance_hold_tol", type=float, default=None,
                        help="dead band of the stance-hold term in rad (default 0.10; use "
                             "0.06 to force the full extension/crouch)")
    parser.add_argument("--stance_jitter", type=float, default=None,
                        help="uniform jitter on every stance sample in rad (default 0.02; "
                             "set 0.0 for an exactly fixed crouch/extension)")
    parser.add_argument("--hip_symmetry", choices=("off", "master_rf"), default=None,
                        help="teleop task: 'master_rf' slaves all four hip targets to the "
                             "right-front leg ([+t,-t,+t,-t]); makes the leg angles identical "
                             "and removes hip-induced nose-up.  Default off; must be trained.")
    parser.add_argument("--hip_symmetry_gate", action="store_true", default=False,
                        help="teleop task: apply master_rf only for in-place spin commands; "
                             "normal driving/turning keeps independent hips.  Train and deploy "
                             "with the same gate.")
    parser.add_argument("--teleop_roll_weight", type=float, default=None,
                        help="teleop task: weight of the always-on roll penalty (default -1.5). "
                             "Set 0 for banked high-yaw training; roll_straight still covers "
                             "straight-line driving.")
    parser.add_argument("--hip_scale", type=float, default=None,
                        help="hip action authority (rad per action unit; default 0.25). "
                             "Must match between training, play and isaac_teleop.")
    parser.add_argument("--hip_range", type=float, default=None,
                        help="hard hip travel around nominal (rad; default 0.45). "
                             "Opening it to 0.55-0.60 enables banking into turns.")
    parser.add_argument("--hip_dev_tol", type=float, default=None,
                        help="hip_deviation dead band (rad; default 0.10). Raise it together "
                             "with --hip_range so the policy is allowed to use the extra travel.")
    parser.add_argument("--hip_dev_weight", type=float, default=None,
                        help="hip_deviation weight (default -2.0; recent runs used -3.0). "
                             "Use -1.0 to -2.0 when opening the hips.")
    parser.add_argument("--teleop_no_heading_hold", action="store_true", default=False,
                        help="teleop task: disable the conditional heading PD (the policy then "
                             "owns the wheel differential in every mode)")
    parser.add_argument("--recovery_pitch_min", type=float, default=None,
                        help="recovery task: minimum |pitch| of the fallen reset, degrees "
                             "(default 45; random sign, so both forward/back falls)")
    parser.add_argument("--recovery_pitch_max", type=float, default=None,
                        help="recovery task: maximum |pitch| of the fallen reset, degrees "
                             "(default 120)")
    parser.add_argument("--recovery_roll_max", type=float, default=None,
                        help="recovery task: maximum |roll| of the fallen reset, degrees "
                             "(default 8; keep small to train front/back falls only)")
    parser.add_argument("--recovery_z_min", type=float, default=None,
                        help="recovery task: minimum extra root height at reset, m (default 0.05)")
    parser.add_argument("--recovery_z_max", type=float, default=None,
                        help="recovery task: maximum extra root height at reset, m (default 0.15)")
    parser.add_argument("--recovery_episode", type=float, default=None,
                        help="recovery task: episode length in seconds (default 12)")
    parser.add_argument("--recovery_hold", type=float, default=None,
                        help="recovery task: upright-and-still hold time for success, seconds "
                             "(default 0.5)")
    parser.add_argument("--recovery_tuck_prob", type=float, default=None,
                        help="recovery task: fraction of resets that start in the deepest "
                             "symmetric squat (default 0.5; curriculum: 1.0 -> 0.6 -> 0.3)")
    parser.add_argument("--recovery_roll_limit", type=float, default=None,
                        help="recovery task: |roll| termination limit, radians (default 1.05)")
    parser.add_argument("--brake_speed_min", type=float, default=None,
                        help="braking task: minimum drive-phase target speed (m/s), default 2.0")
    parser.add_argument("--brake_speed_max", type=float, default=None,
                        help="braking task: maximum drive-phase target speed (m/s), default 3.0")
    parser.add_argument("--drive_time_min", type=float, default=None,
                        help="braking task: minimum drive-phase duration before the brake (s)")
    parser.add_argument("--drive_time_max", type=float, default=None,
                        help="braking task: maximum drive-phase duration before the brake (s)")
    parser.add_argument("--stop_dist_tol", type=float, default=None,
                        help="braking task: allowed stopping distance before the excess penalty "
                             "starts (m), default 1.0. Physical minimum from 3 m/s is ~0.53 m.")
    parser.add_argument("--enable_turning", action="store_true", default=False,
                        help="locomotion only: sample heading targets in [-pi, pi] "
                             "(same as INFANTRY_RL_ENABLE_TURNING=1). The `turning` task "
                             "commands wz directly and ignores this flag.")
    parser.add_argument("--heading_l1", type=float, default=None,
                        help="optional |heading error| L1 reward weight, e.g. -1.5 (0 = disabled, the default)")
    parser.add_argument("--yaw_scale", type=float, default=None,
                        help="multiply all yaw/heading reward weights (1.0 = default; try 1.5-2.0 "
                             "when the heading deviation is too large)")
    # training schedule
    parser.add_argument("--iterations", "--max_iterations", dest="iterations", type=int, default=None,
                        help="number of learning iterations to run (added to the resumed iteration)")
    parser.add_argument("--horizon", "--num_steps_per_env", dest="horizon", type=int, default=None,
                        help="rollout length per environment")
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--reset_obs_norm_count", type=int, default=None,
                        help="on resume, lower the observation normalizer's sample count so its "
                             "mean/std adapt to a new command range. Must be >= 5e6: smaller values "
                             "drift within one rollout and corrupt PPO's importance ratios. "
                             "Omit it (recommended) to leave the saved statistics alone.")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--log_root", type=str, default=None,
                        help="root for rsl_rl logs (default: <project>/logs/rsl_rl)")
    parser.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
    parser.add_argument("--log_project_name", type=str, default=None)
    # resume
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume_latest", action="store_true", default=False,
                        help="load the highest-numbered model_*.pt of the newest run (crash recovery)")
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    # diagnostics / stability
    parser.add_argument("--check", action="store_true", help="run environment diagnostics and exit")
    parser.add_argument("--no_code_state", action="store_true",
                        help="skip rsl_rl's git-diff logging (harmless; kept for CLI compatibility)")
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--video_interval", type=int, default=2000)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
if args_cli.check:
    args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True

# Command-range / turning overrides MUST be in the environment before env_locomotion
# is imported: the original module reads them at import time.  Setting them here
# (before the AppLauncher and before the delayed imports below) preserves that
# behaviour and keeps the printed envelope correct.
if args_cli.vx_min is not None:
    os.environ["INFANTRY_RL_VX_MIN"] = str(args_cli.vx_min)
if args_cli.vx_max is not None:
    os.environ["INFANTRY_RL_VX_MAX"] = str(args_cli.vx_max)
if args_cli.wz_max is not None:
    os.environ["INFANTRY_RL_WZ_MAX"] = str(args_cli.wz_max)
if args_cli.turn_prob is not None:
    os.environ["INFANTRY_RL_TURN_IN_PLACE_PROB"] = str(args_cli.turn_prob)
if args_cli.max_lat_accel is not None:
    os.environ["INFANTRY_RL_MAX_LAT_ACCEL"] = str(args_cli.max_lat_accel)
if args_cli.teleop_vx_min is not None:
    os.environ["INFANTRY_RL_TELEOP_VX_MIN"] = str(args_cli.teleop_vx_min)
if args_cli.teleop_vx_max is not None:
    os.environ["INFANTRY_RL_TELEOP_VX_MAX"] = str(args_cli.teleop_vx_max)
if args_cli.teleop_wz_inplace_max is not None:
    os.environ["INFANTRY_RL_TELEOP_WZ_INPLACE_MAX"] = str(args_cli.teleop_wz_inplace_max)
if args_cli.teleop_brake_prob is not None:
    os.environ["INFANTRY_RL_TELEOP_BRAKE_PROB"] = str(args_cli.teleop_brake_prob)
if args_cli.teleop_resample_min is not None:
    os.environ["INFANTRY_RL_TELEOP_RESAMPLE_MIN"] = str(args_cli.teleop_resample_min)
if args_cli.teleop_resample_max is not None:
    os.environ["INFANTRY_RL_TELEOP_RESAMPLE_MAX"] = str(args_cli.teleop_resample_max)
if args_cli.teleop_low_speed_prob is not None:
    os.environ["INFANTRY_RL_TELEOP_LOW_SPEED_PROB"] = str(args_cli.teleop_low_speed_prob)
if args_cli.teleop_low_speed_min is not None:
    os.environ["INFANTRY_RL_TELEOP_LOW_SPEED_MIN"] = str(args_cli.teleop_low_speed_min)
if args_cli.teleop_low_speed_max is not None:
    os.environ["INFANTRY_RL_TELEOP_LOW_SPEED_MAX"] = str(args_cli.teleop_low_speed_max)
if args_cli.teleop_heading_hold_wz is not None:
    os.environ["INFANTRY_RL_TELEOP_HEADING_HOLD_WZ"] = str(args_cli.teleop_heading_hold_wz)
if args_cli.teleop_heading_lat_accel is not None:
    os.environ["INFANTRY_RL_TELEOP_HEADING_LAT_ACCEL"] = str(args_cli.teleop_heading_lat_accel)
if args_cli.teleop_turn_min_wz is not None:
    os.environ["INFANTRY_RL_TELEOP_TURN_MIN_WZ"] = str(args_cli.teleop_turn_min_wz)
if args_cli.stance_randomize:
    os.environ["INFANTRY_RL_STANCE_RANDOMIZE"] = "1"
if args_cli.stance_min is not None:
    os.environ["INFANTRY_RL_STANCE_MIN"] = str(args_cli.stance_min)
if args_cli.stance_max is not None:
    os.environ["INFANTRY_RL_STANCE_MAX"] = str(args_cli.stance_max)
if args_cli.stance_slew is not None:
    os.environ["INFANTRY_RL_STANCE_SLEW"] = str(args_cli.stance_slew)
if args_cli.stance_gate_vx is not None:
    os.environ["INFANTRY_RL_STANCE_GATE_VX"] = str(args_cli.stance_gate_vx)
if args_cli.stance_gate_wz is not None:
    os.environ["INFANTRY_RL_STANCE_GATE_WZ"] = str(args_cli.stance_gate_wz)
if args_cli.stance_prob_crouch is not None:
    os.environ["INFANTRY_RL_STANCE_PROB_CROUCH"] = str(args_cli.stance_prob_crouch)
if args_cli.stance_prob_nominal is not None:
    os.environ["INFANTRY_RL_STANCE_PROB_NOMINAL"] = str(args_cli.stance_prob_nominal)
if args_cli.stance_hold_weight is not None:
    os.environ["INFANTRY_RL_STANCE_HOLD_WEIGHT"] = str(args_cli.stance_hold_weight)
if args_cli.stance_hold_tol is not None:
    os.environ["INFANTRY_RL_STANCE_HOLD_TOL"] = str(args_cli.stance_hold_tol)
if args_cli.stance_jitter is not None:
    os.environ["INFANTRY_RL_STANCE_JITTER"] = str(args_cli.stance_jitter)
if args_cli.hip_symmetry is not None:
    os.environ["INFANTRY_RL_TELEOP_HIP_SYMMETRY"] = str(args_cli.hip_symmetry)
if args_cli.hip_symmetry_gate:
    os.environ["INFANTRY_RL_TELEOP_HIP_SYMMETRY_GATE"] = "1"
if args_cli.teleop_roll_weight is not None:
    os.environ["INFANTRY_RL_TELEOP_ROLL_WEIGHT"] = str(args_cli.teleop_roll_weight)
if args_cli.hip_scale is not None:
    os.environ["INFANTRY_RL_HIP_SCALE"] = str(args_cli.hip_scale)
if args_cli.hip_range is not None:
    os.environ["INFANTRY_RL_HIP_RANGE"] = str(args_cli.hip_range)
if args_cli.hip_dev_tol is not None:
    os.environ["INFANTRY_RL_BRAKE_HIP_DEV_TOL"] = str(args_cli.hip_dev_tol)
if args_cli.hip_dev_weight is not None:
    os.environ["INFANTRY_RL_BRAKE_HIP_DEV_WEIGHT"] = str(args_cli.hip_dev_weight)
if args_cli.teleop_no_heading_hold:
    os.environ["INFANTRY_RL_TELEOP_HEADING_HOLD"] = "0"
if args_cli.recovery_pitch_min is not None:
    os.environ["INFANTRY_RL_RECOVERY_PITCH_MIN"] = str(args_cli.recovery_pitch_min)
if args_cli.recovery_pitch_max is not None:
    os.environ["INFANTRY_RL_RECOVERY_PITCH_MAX"] = str(args_cli.recovery_pitch_max)
if args_cli.recovery_roll_max is not None:
    os.environ["INFANTRY_RL_RECOVERY_ROLL_MAX"] = str(args_cli.recovery_roll_max)
if args_cli.recovery_z_min is not None:
    os.environ["INFANTRY_RL_RECOVERY_Z_MIN"] = str(args_cli.recovery_z_min)
if args_cli.recovery_z_max is not None:
    os.environ["INFANTRY_RL_RECOVERY_Z_MAX"] = str(args_cli.recovery_z_max)
if args_cli.recovery_episode is not None:
    os.environ["INFANTRY_RL_RECOVERY_EPISODE"] = str(args_cli.recovery_episode)
if args_cli.recovery_hold is not None:
    os.environ["INFANTRY_RL_RECOVERY_HOLD"] = str(args_cli.recovery_hold)
if args_cli.recovery_tuck_prob is not None:
    os.environ["INFANTRY_RL_RECOVERY_TUCK_PROB"] = str(args_cli.recovery_tuck_prob)
if args_cli.recovery_roll_limit is not None:
    os.environ["INFANTRY_RL_RECOVERY_ROLL_LIMIT"] = str(args_cli.recovery_roll_limit)
if args_cli.brake_speed_min is not None:
    os.environ["INFANTRY_RL_BRAKE_SPEED_MIN"] = str(args_cli.brake_speed_min)
if args_cli.brake_speed_max is not None:
    os.environ["INFANTRY_RL_BRAKE_SPEED_MAX"] = str(args_cli.brake_speed_max)
if args_cli.drive_time_min is not None:
    os.environ["INFANTRY_RL_DRIVE_TIME_MIN"] = str(args_cli.drive_time_min)
if args_cli.drive_time_max is not None:
    os.environ["INFANTRY_RL_DRIVE_TIME_MAX"] = str(args_cli.drive_time_max)
if args_cli.stop_dist_tol is not None:
    os.environ["INFANTRY_RL_STOP_DIST_TOL"] = str(args_cli.stop_dist_tol)
if args_cli.enable_turning:
    os.environ["INFANTRY_RL_ENABLE_TURNING"] = "1"
if args_cli.heading_l1 is not None:
    os.environ["INFANTRY_RL_HEADING_L1"] = str(args_cli.heading_l1)
if args_cli.yaw_scale is not None:
    os.environ["INFANTRY_RL_YAW_SCALE"] = str(args_cli.yaw_scale)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows (imports require Kit to be running)."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402

import agent as agent_lib  # noqa: E402
from config import DEFAULT_LOG_ROOT, normalize_task  # noqa: E402
from env_balance import BalanceEnvCfg, FiveBarBalanceEnv  # noqa: E402
from env_braking import BrakingEnvCfg, FiveBarBrakingEnv  # noqa: E402
from env_locomotion import LocomotionEnvCfg, FiveBarLocomotionEnv  # noqa: E402
from env_recovery import FiveBarRecoveryEnv, RecoveryEnvCfg  # noqa: E402
from env_squat import (  # noqa: E402
    SQUAT_BASE_HEIGHT,
    FiveBarSquatEnv,
    SquatEnvCfg,
)
from env_stand import FiveBarStandEnv, StandEnvCfg  # noqa: E402
from env_teleop import FiveBarTeleopEnv, TeleopEnvCfg  # noqa: E402
from env_turning import FiveBarTurningEnv, TurningEnvCfg  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402


# --------------------------------------------------------------------------- #
# Environment construction
# --------------------------------------------------------------------------- #
def build_env_cfg(task: str, num_envs: int | None, device: str, seed: int, log_dir: str):
    """Create the manager-based env config for the requested task."""
    if task == "locomotion":
        env_cfg = LocomotionEnvCfg()
    elif task == "turning":
        env_cfg = TurningEnvCfg()
    elif task == "braking":
        env_cfg = BrakingEnvCfg()
    elif task == "teleop":
        env_cfg = TeleopEnvCfg()
    elif task == "recovery":
        env_cfg = RecoveryEnvCfg()
    elif task == "stand":
        env_cfg = StandEnvCfg()
    elif task == "balance":
        env_cfg = BalanceEnvCfg()
    elif task == "squat":
        env_cfg = SquatEnvCfg()
    else:
        raise ValueError(f"unknown task {task!r}")
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.sim.device = device
    env_cfg.log_dir = log_dir
    return env_cfg


def make_env(task: str, env_cfg, render_mode: str | None = None):
    """Instantiate the environment directly (no gym registration)."""
    if task == "locomotion":
        return FiveBarLocomotionEnv(cfg=env_cfg, render_mode=render_mode)
    if task == "turning":
        return FiveBarTurningEnv(cfg=env_cfg, render_mode=render_mode)
    if task == "braking":
        return FiveBarBrakingEnv(cfg=env_cfg, render_mode=render_mode)
    if task == "teleop":
        return FiveBarTeleopEnv(cfg=env_cfg, render_mode=render_mode)
    if task == "recovery":
        return FiveBarRecoveryEnv(cfg=env_cfg, render_mode=render_mode)
    if task == "stand":
        return FiveBarStandEnv(cfg=env_cfg, render_mode=render_mode)
    if task == "balance":
        return FiveBarBalanceEnv(cfg=env_cfg, render_mode=render_mode)
    if task == "squat":
        return FiveBarSquatEnv(cfg=env_cfg, render_mode=render_mode)
    raise ValueError(f"unknown task {task!r}")


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def run_env_checks(env, task: str) -> bool:
    """Exercise the interfaces that must work before a long training run."""
    ok = True
    cfg = env.cfg
    robot = env.scene["robot"]
    joint_action = env.action_manager.get_term(cfg.joint_action_name)
    actions = torch.zeros((cfg.scene.num_envs, env.action_space.shape[-1]), device=env.device)

    # the manager-based env returns a dict keyed by observation group; the wrappers
    # concatenate it into the flat policy vector the actor sees
    def policy_obs(observation):
        return observation["policy"] if isinstance(observation, dict) else observation

    obs_dim = env.single_observation_space["policy"].shape[0]

    obs, _ = env.reset()
    obs = policy_obs(obs)
    print(f"[check] obs shape={tuple(obs.shape)} finite={bool(torch.isfinite(obs).all())}")
    ok &= obs.shape == (cfg.scene.num_envs, obs_dim)
    ok &= bool(torch.isfinite(obs).all())
    print(f"[check] num_envs={cfg.scene.num_envs}  ep_steps={env.max_episode_length}  "
          f"closure_max={env.max_closure_residual * 1000:.6f} mm")
    ok &= env.max_closure_residual < 5e-3

    # one zero step must not terminate anything: catches world-vs-local frame bugs
    obs, reward, terminated, truncated, extras = env.step(actions)
    done = terminated | truncated
    print(f"[check] initial zero step: done={int(done.sum())}/{cfg.scene.num_envs}")
    ok &= not bool(done.any())

    if task == "locomotion":
        # wheel-speed probe: 0.5 * 5 rad/s = 2.5 rad/s residual
        env.reset()
        probe = torch.zeros_like(actions)
        probe[:, 4:] = 0.5
        for _ in range(5):
            obs, reward, terminated, truncated, extras = env.step(probe)
        wheel_vel = robot.data.joint_vel[:, joint_action.wheel_ids].abs().max().item()
        print(f"[check] wheel probe: residual 2.5 rad/s, measured max={wheel_vel:.3f} rad/s")
        ok &= wheel_vel > 0.5

    # zero actions: the robot may start falling, but interfaces must stay finite
    env.reset()
    for step in range(10):
        obs, reward, terminated, truncated, extras = env.step(torch.zeros_like(actions))
        obs = policy_obs(obs)
        vx = robot.data.root_lin_vel_b[0, 0].item()
        height = robot.data.root_pos_w[0, 2].item()
        upright = -robot.data.projected_gravity_b[0, 2].item()
        print(
            f"[check] zero step={step} reward={reward.mean().item(): .4f} "
            f"vx={vx: .3f} h={height: .3f} upright={upright: .3f} "
            f"done={int((terminated | truncated).sum())}"
        )
        ok &= bool(torch.isfinite(obs).all())
        ok &= bool(torch.isfinite(reward).all())

    # force a timeout and verify the reset re-applies the loop-consistent home pose
    env.episode_length_buf[:] = env.max_episode_length - 1
    obs, reward, terminated, truncated, extras = env.step(torch.zeros_like(actions))
    done = terminated | truncated
    print(
        f"[check] forced timeout: done={int(done.sum())}/{cfg.scene.num_envs}, "
        f"episode_steps_max={int(env.episode_length_buf.max())}"
    )
    ok &= bool(done.all())
    home_error = (robot.data.joint_pos[:, joint_action.hip_ids] -
                  joint_action.nominal_hips).abs().max().item()
    if task in ("recovery", "stand", "squat"):
        # The recovery/stand/squat resets deliberately move the hips away from the
        # nominal stance, so the nominal-home check does not apply; only the loop
        # closure must survive the reset.
        hip_offsets = robot.data.joint_pos[:, joint_action.hip_ids] - joint_action.nominal_hips
        heights = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        # HIP_ORDER = (left_front, left_back, right_front, right_back)
        front_off = hip_offsets[:, [0, 2]].mean().item()
        back_off = hip_offsets[:, [1, 3]].mean().item()
        reset_symmetric = bool(
            (hip_offsets[:, 0] - hip_offsets[:, 2]).abs().max() < 1e-6
            and (hip_offsets[:, 1] - hip_offsets[:, 3]).abs().max() < 1e-6
        )
        if task == "squat":
            # chassis-rl's right joints are mirrored, so "left/right identical joint
            # values" is NOT the right symmetry test here; compare to locked_hips below.
            print(
                f"[check] after reset: hips intentionally moved "
                f"(max offset {home_error:.3f} rad), "
                f"closure={env.max_closure_residual * 1000:.6f} mm"
            )
        else:
            print(
                f"[check] after reset: hips intentionally moved "
                f"(max offset {home_error:.3f} rad), front {front_off:+.3f} / "
                f"back {back_off:+.3f} rad, left/right symmetric={reset_symmetric}, "
                f"closure={env.max_closure_residual * 1000:.6f} mm"
            )
        ok &= home_error < (0.75 if task == "squat" else 0.55)
        if task == "squat":
            # Deepest feasible pose: the hips must match the action term's locked
            # targets (whose right side is mirrored for chassis-rl), and the root
            # must sit at that asset's measured squat height (0.0764 m for
            # infantry-rl, 0.1735 m for the lower chassis-rl base frame).
            hip_err = (
                robot.data.joint_pos[:, joint_action.hip_ids] - joint_action.locked_hips
            ).abs().max().item()
            print(
                f"[check] squat reset: max |hip - locked|={hip_err:.4f} rad, "
                f"base_z {heights.min().item():.4f}-{heights.max().item():.4f} m "
                f"(target {SQUAT_BASE_HEIGHT:.4f})"
            )
            ok &= hip_err < 0.06
            ok &= abs(heights.max().item() - SQUAT_BASE_HEIGHT) < 0.03
            ok &= abs(heights.min().item() - SQUAT_BASE_HEIGHT) < 0.03
        else:
            ok &= reset_symmetric
            if float(env.cfg.events.reset_hips.params.get("tuck_prob", 0.0)) >= 0.999:
                ok &= abs(front_off + 0.45) < 0.12 and abs(back_off - 0.45) < 0.12
    else:
        print(f"[check] after reset: hip-home={home_error:.6f} rad, closure={env.max_closure_residual * 1000:.6f} mm")
        ok &= home_error < 1e-3
    ok &= env.max_closure_residual < 5e-3

    if task == "turning":
        # Command-sampler probe: the in-place share must match turn_in_place_prob and
        # every in-place command must be (vx = 0, |wz| >= turn_in_place_min_abs).
        # `_resample_command` + `_update_command` are exactly what the command manager
        # calls on a timeout; the environment is not used afterwards (the check exits).
        term = env.command_manager.get_term("base_velocity")
        env_ids = torch.arange(cfg.scene.num_envs, device=env.device)
        draws = 200
        n_in_place = n_turning = n_total = 0
        n_over_cap = 0
        max_lat = 0.0
        for _ in range(draws):
            term._resample_command(env_ids)
            term._update_command()
            cmd = term.command
            n_total += cmd.shape[0]
            n_in_place += int(
                ((cmd[:, 0].abs() < 1e-6) & (cmd[:, 2].abs() >= term.cfg.turn_in_place_min_abs - 1e-6)).sum()
            )
            n_turning += int((cmd[:, 2].abs() > 1e-6).sum())
            lateral = (cmd[:, 0].abs() * cmd[:, 2].abs())
            max_lat = max(max_lat, float(lateral.max()))
            if term.cfg.max_lateral_accel > 0.0:
                n_over_cap += int((lateral > term.cfg.max_lateral_accel + 1e-5).sum())
        frac_in_place = n_in_place / max(n_total, 1)
        frac_turning = n_turning / max(n_total, 1)
        print(
            f"[check] turning command sampler: in-place {frac_in_place:.1%} "
            f"(target {term.cfg.turn_in_place_prob:.0%}), any turn {frac_turning:.1%}, "
            f"max |vx*wz| {max_lat:.2f} m/s^2 "
            f"(cap {term.cfg.max_lateral_accel:.2f}, violations {n_over_cap})"
        )
        ok &= abs(frac_in_place - term.cfg.turn_in_place_prob) < 0.08
        ok &= n_over_cap == 0

    if task == "braking":
        # Schedule probe: the drive command must be the sampled target, and after the
        # sampled drive time every environment must have received the zero brake command.
        term = env.command_manager.get_term("base_velocity")
        env_ids = torch.arange(cfg.scene.num_envs, device=env.device)
        term._resample_command(env_ids)
        term._update_command()
        target = term.target_speed.clone()
        drive_steps = term._drive_steps.clone()
        lo, hi = term.cfg.ranges.lin_vel_x
        target_ok = bool(((target >= lo - 1e-6) & (target <= hi + 1e-6)).all())
        drive_cmd_ok = bool((term.command[:, 0] > 0).all())
        for _ in range(int(drive_steps.max().item())):
            term._update_command()
        braked = bool(term.brake_started.all())
        zero_cmd = bool((term.command.abs() < 1e-6).all())
        print(
            f"[check] braking schedule: target {target.min().item():.2f}-{target.max().item():.2f} m/s "
            f"(range [{lo:.2f}, {hi:.2f}]), drive "
            f"{drive_steps.min().item() * env.step_dt:.1f}-{drive_steps.max().item() * env.step_dt:.1f} s, "
            f"target_ok={target_ok}, braked={braked}, zero_cmd={zero_cmd}"
        )
        ok &= target_ok and drive_cmd_ok and braked and zero_cmd

    if task == "teleop":
        # Sampler probe: the command envelope must contain driving, in-place turns,
        # standing and enough moving->zero transitions to exercise the brake terms.
        term = env.command_manager.get_term("base_velocity")
        env_ids = torch.arange(cfg.scene.num_envs, device=env.device)
        draws = 200
        n_total = n_zero = n_in_place = n_moving = 0
        n_over_cap = 0
        n_negative = 0
        n_low_speed = 0
        inplace_mag_sum = 0.0
        inplace_mag_count = 0
        vx_obs_min = float("inf")
        vx_obs_max = -float("inf")
        max_lat = 0.0
        for _ in range(draws):
            term._resample(env_ids)  # realistic periodic resample (not an episode reset)
            term._update_command()
            cmd = term.command
            zero = (cmd[:, 0].abs() < 1e-6) & (cmd[:, 2].abs() < 1e-6)
            in_place = (cmd[:, 0].abs() < 1e-6) & (
                cmd[:, 2].abs() >= term.cfg.turn_in_place_min_abs - 1e-6
            )
            n_total += cmd.shape[0]
            n_zero += int(zero.sum())
            n_in_place += int(in_place.sum())
            inplace_mag_sum += float(cmd[in_place, 2].abs().sum())
            inplace_mag_count += int(in_place.sum())
            n_moving += int((~zero).sum())
            n_negative += int((cmd[:, 0] < -1e-6).sum())
            n_low_speed += int(
                ((cmd[:, 0] > 1e-6) & (cmd[:, 0] <= term.cfg.low_speed_max + 1e-6) & ~in_place).sum()
            )
            vx_obs_min = min(vx_obs_min, float(cmd[:, 0].min()))
            vx_obs_max = max(vx_obs_max, float(cmd[:, 0].max()))
            lateral = cmd[:, 0].abs() * cmd[:, 2].abs()
            max_lat = max(max_lat, float(lateral.max()))
            if term.cfg.max_lateral_accel > 0.0:
                n_over_cap += int((lateral > term.cfg.max_lateral_accel + 1e-5).sum())
        zero_frac = n_zero / max(n_total, 1)
        in_place_frac = n_in_place / max(n_total, 1)
        reverse_frac = n_negative / max(n_total, 1)
        expected_zero = 1.0 - (1.0 - term.cfg.rel_standing_envs) * (1.0 - term.cfg.brake_prob)
        expected_in_place = (1.0 - expected_zero) * term.cfg.turn_in_place_prob
        print(
            f"[check] teleop command sampler: zero {zero_frac:.1%} "
            f"(target {expected_zero:.0%}), in-place {in_place_frac:.1%} "
            f"(target {expected_in_place:.0%}), max |vx*wz| {max_lat:.2f} m/s^2 "
            f"(cap {term.cfg.max_lateral_accel:.2f}, violations {n_over_cap})"
        )
        inplace_mag_mean = inplace_mag_sum / max(inplace_mag_count, 1)
        print(
            f"[check] teleop in-place |wz|: mean {inplace_mag_mean:.2f} rad/s "
            f"(range [{term.cfg.turn_in_place_min_abs:.2f}, "
            f"{term.cfg.turn_in_place_max_abs:.2f}]), samples {inplace_mag_count}"
        )
        ok &= inplace_mag_mean >= term.cfg.turn_in_place_min_abs - 1e-3
        low_speed_frac = n_low_speed / max(n_total, 1)
        print(
            f"[check] teleop vx sampler: observed [{vx_obs_min:+.2f}, {vx_obs_max:+.2f}] m/s, "
            f"reverse {reverse_frac:.1%}"
        )
        print(
            f"[check] teleop low-speed: vx in (0, {term.cfg.low_speed_max:.2f}] m/s "
            f"{low_speed_frac:.1%} (mode {term.cfg.low_speed_prob:.0%} of moving)"
        )
        ok &= abs(zero_frac - expected_zero) < 0.08
        ok &= abs(in_place_frac - expected_in_place) < 0.06
        ok &= n_over_cap == 0
        lo, hi = term.cfg.ranges.lin_vel_x
        ok &= bool(((term.command[:, 0] >= lo - 1e-6) & (term.command[:, 0] <= hi + 1e-6)).all())
        if lo < 0.0:
            # the reverse curriculum is only meaningful if negative vx is really sampled
            ok &= reverse_frac > 0.0 and vx_obs_min < 0.0
        if term.cfg.low_speed_prob > 0.0:
            # the low-speed band must actually be present in the command stream
            ok &= low_speed_frac > 0.08

        # Forced-stop probe: after a moving command, a 100% stop draw must record the
        # onset speed/position so the shared brake rewards have a valid event.
        term.vel_command_b[env_ids, 0] = 1.0
        term.vel_command_b[env_ids, 2] = 0.0
        term.is_standing_env[env_ids] = False
        term._resetting[env_ids] = False
        saved = (term.cfg.brake_prob, term.cfg.turn_in_place_prob, term.cfg.rel_standing_envs)
        term.cfg.brake_prob = 1.0
        term.cfg.turn_in_place_prob = 0.0
        term.cfg.rel_standing_envs = 0.0
        term._resample_command(env_ids)
        term._update_command()
        braked = bool(term.brake_started.all())
        zero_cmd = bool(
            (term.command[:, 0].abs() < 1e-6).all() & (term.command[:, 2].abs() < 1e-6).all()
        )
        onset = float(term.brake_speed.mean())
        term.cfg.brake_prob, term.cfg.turn_in_place_prob, term.cfg.rel_standing_envs = saved
        print(
            f"[check] teleop forced stop: braked={braked}, zero_cmd={zero_cmd}, "
            f"mean onset speed {onset:.2f} m/s, stop fields present="
            f"{hasattr(term, 'stop_distance') and hasattr(term, 'stop_step')}"
        )
        ok &= braked and zero_cmd

        # Sign probes for the reverse curriculum.  The wheel joint convention is
        # measured, not assumed: a POSITIVE joint velocity rolls the wheel BACKWARD,
        # so reverse (+vx target negative) needs BOTH wheel targets positive.  Hips
        # are position offsets around the nominal stance and do NOT change sign with
        # the driving direction; a positive action must stay a positive hip offset.
        env.reset()
        wheel_probe = torch.zeros_like(actions)
        wheel_probe[:, 4:] = 0.25
        env.step(wheel_probe)
        wheel_targets = joint_action.applied_wheel_targets[0]
        print(
            f"[check] teleop wheel-sign mapping: action +0.25 -> wheel joint targets "
            f"{[round(float(v), 3) for v in wheel_targets]} rad/s (action sign preserved)"
        )
        ok &= bool((wheel_targets > 0).all())

        env.reset()
        hip_probe = torch.zeros_like(actions)
        hip_probe[:, :4] = 0.5
        if getattr(joint_action.cfg, "hip_symmetry", "off") == "master_rf":
            # probe the master mapping directly, not through the in-place gate
            joint_action.cfg.hip_symmetry_gate = False
            joint_action._sym_blend[:] = 1.0
        env.step(hip_probe)
        hip_delta = (joint_action.hip_targets - joint_action.nominal_hips)[0]
        if getattr(joint_action.cfg, "hip_symmetry", "off") == "master_rf":
            # only the right-front action matters; the other three are mirrored
            probe_delta = 0.5 * float(joint_action.cfg.hip_scale)
            expected = torch.tensor(
                [probe_delta, -probe_delta, probe_delta, -probe_delta],
                device=env.device,
            )
            sym_err = float((hip_delta - expected).abs().max().item())
            print(
                f"[check] teleop hip symmetry (master_rf): hip action +0.5 -> target "
                f"deltas {[round(float(v), 3) for v in hip_delta]} rad "
                f"(expected [+,-,+,-]{probe_delta:.3f}, err {sym_err:.2e})"
            )
            ok &= sym_err < 1e-5
        else:
            print(
                f"[check] teleop hip-sign mapping: hip action +0.5 -> target deltas "
                f"{[round(float(v), 3) for v in hip_delta]} rad (action sign preserved)"
            )
            ok &= bool((hip_delta > 1e-6).all())

        # Stance probe: a symmetric scripted bias must appear as [+d, -d, +d, -d]
        # on top of the nominal hips.  Disable randomization so the manually set
        # target is not overwritten by the sampler during this one step.
        env.reset()
        joint_action.cfg.stance_randomize = False
        if getattr(joint_action.cfg, "hip_symmetry", "off") == "master_rf":
            joint_action.cfg.hip_symmetry_gate = False
            joint_action._sym_blend[:] = 1.0
        probe_stance = 0.15
        joint_action.set_stance_target(probe_stance, snap=True)
        env.step(torch.zeros_like(actions))
        stance_delta = (joint_action.hip_targets - joint_action.nominal_hips)[0]
        expected = torch.tensor(
            [probe_stance, -probe_stance, probe_stance, -probe_stance],
            device=env.device,
        )
        stance_err = float((stance_delta - expected).abs().max().item())
        print(
            f"[check] teleop stance mapping: {probe_stance:+.2f} rad -> target deltas "
            f"{[round(float(v), 3) for v in stance_delta]} rad (max err {stance_err:.2e})"
        )
        ok &= stance_err < 1e-5
        joint_action.set_stance_target(0.0, snap=True)

    if task == "recovery":
        # The reset must really put the robot in a front/back fallen pose; a check
        # that only looks at observation dimensions would miss an ignored event.
        env.reset()
        roll, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
        heights = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        print(
            f"[check] recovery reset: |pitch| mean {pitch.abs().mean().item():.3f} rad, "
            f"max {pitch.abs().max().item():.3f}; |roll| max {roll.abs().max().item():.3f}; "
            f"base height {heights.min().item():.3f}-{heights.max().item():.3f} m"
        )
        ok &= float(pitch.abs().mean().item()) > 0.5
        ok &= float(roll.abs().max().item()) < 0.35
        ok &= hasattr(env, "recovery_upright_steps")

        # A few zero-action steps from the fallen pose must stay finite and must not
        # hit an ordinary fall termination (recovery deliberately allows lying down).
        done_count = 0
        for _ in range(10):
            obs, reward, terminated, truncated, extras = env.step(torch.zeros_like(actions))
            done_count += int((terminated | truncated).sum())
            ok &= bool(torch.isfinite(reward).all())
        # Random mixed-tuck resets can roll out within a few steps; roll_limit is an
        # intentional failure guard, so this probe only requires finite rewards (the
        # generic initial-zero-step check above still requires a clean first step).
        print(
            f"[check] recovery zero steps: done={done_count}/{cfg.scene.num_envs} "
            "(roll_limit allowed)"
        )

        # Zero action must HOLD the tuck: the action term's hip_bias makes the
        # nominal action the deepest squat, so a few zero steps should not extend
        # the legs back to nominal.  Only assert this when every reset is a tuck;
        # with a mixed curriculum the random resets are not expected at ±0.45.
        tuck_prob = float(env.cfg.events.reset_hips.params.get("tuck_prob", 0.0))
        hip_now = robot.data.joint_pos[:, joint_action.hip_ids] - joint_action.nominal_hips
        front_now = hip_now[:, [0, 2]].mean().item()
        back_now = hip_now[:, [1, 3]].mean().item()
        print(
            f"[check] recovery stance after 10 zero steps: front {front_now:+.3f} / "
            f"back {back_now:+.3f} rad (tuck at reset, dynamic bias may extend it; "
            f"tuck_prob={tuck_prob:.2f})"
        )

        # Sagittal symmetry: an asymmetric 6-dim action must be averaged into
        # symmetric left/right hip targets and equal wheel targets.
        env.reset()
        asym = torch.zeros_like(actions)
        asym[:, 0], asym[:, 2] = 1.0, -1.0   # front hips
        asym[:, 1], asym[:, 3] = 0.5, -0.5   # back hips
        asym[:, 4], asym[:, 5] = 1.0, -1.0   # wheels
        env.step(asym)
        hip_targets = joint_action.hip_targets[0] - joint_action.nominal_hips[0]
        wheel_targets = joint_action.applied_wheel_targets[0]
        symmetric = (
            abs(float(hip_targets[0] - hip_targets[2])) < 1e-6
            and abs(float(hip_targets[1] - hip_targets[3])) < 1e-6
            and abs(float(wheel_targets[0] - wheel_targets[1])) < 1e-6
        )
        print(
            f"[check] recovery symmetry: hip targets "
            f"{[round(float(v), 3) for v in hip_targets]}, wheel targets "
            f"{[round(float(v), 3) for v in wheel_targets]} -> {symmetric}"
        )
        ok &= symmetric

    if task == "stand":
        # The stand reset must be UPRIGHT but deliberately LOW (hips tucked).
        env.reset()
        hip_off = robot.data.joint_pos[:, joint_action.hip_ids] - joint_action.nominal_hips
        roll, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
        heights = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        symmetric = bool(
            (hip_off[:, 0] - hip_off[:, 2]).abs().max() < 1e-6
            and (hip_off[:, 1] - hip_off[:, 3]).abs().max() < 1e-6
        )
        print(
            f"[check] stand reset: hips front {hip_off[:, [0, 2]].mean().item():+.3f} / "
            f"back {hip_off[:, [1, 3]].mean().item():+.3f} rad (tuck), "
            f"base height {heights.min().item():.3f}-{heights.max().item():.3f} m, "
            f"|pitch| {pitch.abs().mean().item():.3f} rad, "
            f"|roll| {roll.abs().max().item():.3f} rad, symmetric={symmetric}"
        )
        ok &= abs(hip_off[:, [0, 2]].mean().item() + 0.45) < 0.12
        ok &= abs(hip_off[:, [1, 3]].mean().item() - 0.45) < 0.12
        ok &= heights.max().item() < 0.16
        ok &= pitch.abs().mean().item() < 0.30
        ok &= roll.abs().max().item() < 0.20
        ok &= symmetric

    if task == "squat":
        # The policy must own the wheels only, and the hips must stay locked at the
        # deepest feasible pose while it does.
        env.reset()
        action_dim = int(env.action_space.shape[-1])
        heights = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        locked = [round(float(v), 4) for v in joint_action.locked_hips]
        print(
            f"[check] squat: action_dim={action_dim} (expected 2), "
            f"base_z {heights.min().item():.4f}-{heights.max().item():.4f} m "
            f"(target {SQUAT_BASE_HEIGHT:.4f}), locked hips "
            f"(lf, lb, rf, rb)={locked} rad, "
            f"closure={env.max_closure_residual * 1000:.6f} mm"
        )
        ok &= action_dim == 2
        zero = torch.zeros((cfg.scene.num_envs, action_dim), device=env.device)
        for _ in range(5):
            env.step(zero)
        hip_drift = (
            robot.data.joint_pos[:, joint_action.hip_ids] - joint_action.locked_hips
        ).abs().max().item()
        print(
            f"[check] squat zero-action drift: max |hip - locked|={hip_drift:.4f} rad, "
            f"closure={env.max_closure_residual * 1000:.6f} mm"
        )
        ok &= hip_drift < 0.05
        ok &= env.max_closure_residual < 5e-3

    print("[check] RESULT:", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    import importlib.metadata as metadata

    task = normalize_task(args_cli.task)
    agent_cfg = agent_lib.apply_cli_overrides(agent_lib.make_agent_cfg(task, play=False), args_cli)
    # same deprecated-config shim as the original train.py (a no-op on rsl-rl 5.x,
    # but it keeps older runner configs working)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    if args_cli.iterations is not None:
        agent_cfg.max_iterations = args_cli.iterations
    if args_cli.horizon is not None:
        agent_cfg.num_steps_per_env = args_cli.horizon
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    log_root = Path(args_cli.log_root) if args_cli.log_root else DEFAULT_LOG_ROOT
    log_root_path = os.path.abspath(log_root / agent_cfg.experiment_name)

    # run directory: <log_root>/<experiment>/<timestamp>[_run_name]
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    env_cfg = build_env_cfg(
        task,
        args_cli.num_envs,
        args_cli.device or "cuda:0",
        agent_cfg.seed,
        log_dir,
    )
    env = make_env(task, env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.check:
        return 0 if run_env_checks(env, task) else 1

    # resolve the resume path before the runner is built, so the network shape can
    # be reconciled with the checkpoint it will load
    resume_path: str | None = None
    if agent_cfg.resume:
        # A checkpoint from ANOTHER experiment (e.g. a locomotion policy used to
        # warm-start `turning`) cannot be resolved under this task's log root, so an
        # existing file path in --checkpoint is used directly.
        #
        # IMPORTANT: only strings that actually contain a path separator are treated
        # as paths.  A bare filename / glob ("model_6447.pt", "model_.*.pt") is a regex
        # resolved under the task's log root by get_checkpoint_path.  Passing a Windows
        # path such as "D:\IsaacLab-main\..." to re.match used to crash with
        # "bad escape \I", which is why a missing path now raises a clear error.
        checkpoint_arg = agent_cfg.load_checkpoint
        looks_like_path = bool(checkpoint_arg) and (
            "/" in checkpoint_arg or "\\" in checkpoint_arg
        )
        if looks_like_path:
            direct = os.path.abspath(checkpoint_arg)
            if not os.path.isfile(direct):
                raise FileNotFoundError(
                    f"--checkpoint '{checkpoint_arg}' looks like a file path but does not exist.\n"
                    f"  resolved to: {direct}\n"
                    f"  fix: check the path, or use --load_run <run folder> with a bare "
                    f"filename, e.g. --load_run <run folder> --checkpoint model_6447.pt"
                )
            resume_path = direct
            print(f"[INFO] --checkpoint is an existing file, loading it directly: {resume_path}")
        elif args_cli.resume_latest:
            resume_path = agent_lib.newest_checkpoint(log_root_path, agent_cfg.load_run)
            print(f"[INFO] --resume_latest resolved to: {resume_path}")
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        agent_lib.match_network_to_checkpoint(resume_path, agent_cfg)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    try:
        runner.add_git_repo_to_log(__file__)
    except Exception as exc:  # noqa: BLE001 - logging must never break training
        print(f"[INFO] could not record the git state: {exc}")

    if resume_path is not None:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        if args_cli.reset_obs_norm_count is not None:
            agent_lib.reset_obs_normalizer_count(runner, args_cli.reset_obs_norm_count)

    # bounds on the policy's exploration noise.  Recovery starts from a fallen
    # pose and a large std (0.4) can roll the robot onto its side before it learns
    # anything; cap it lower for this task.
    if task in ("recovery", "stand"):
        agent_lib.clamp_policy_std(runner, low=0.06, high=0.25)
    else:
        agent_lib.clamp_policy_std(runner)
    # console: keep every term visible and append the converted tracking summary
    _step_dt = env_cfg.sim.dt * env_cfg.decimation
    agent_lib.filter_iteration_console(
        runner,
        step_dt=_step_dt,
        max_episode_steps=int(round(env_cfg.episode_length_s / _step_dt)),
    )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(
        f"[INFO] task={task}  num_envs={env_cfg.scene.num_envs}  "
        f"max_iterations={agent_cfg.max_iterations}  device={agent_cfg.device}"
    )
    print(f"[INFO] run directory: {log_dir}")

    if args_cli.no_code_state:
        try:
            n = len(runner.logger.git_status_repos)
            runner.logger.git_status_repos = []
            print(f"[INFO] --no_code_state: skipped rsl_rl git-diff logging ({n} repo path(s) dropped)")
        except AttributeError as exc:  # pragma: no cover
            print(f"[INFO] could not disable git-diff logging: {exc}")

    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    print(f"[INFO] final closure residual (m): {env.unwrapped.max_closure_residual:.3e}")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit shutdown can hang on Windows after a long headless run; every
        # checkpoint and log line is already flushed, so exit the process.
        os._exit(exit_code)
