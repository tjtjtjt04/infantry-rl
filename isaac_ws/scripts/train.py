"""Train an RL agent for the infantry-rl closed-chain robot with RSL-RL.

Mirrors ``IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`` (same order:
AppLauncher -> args -> hydra task config -> gym env -> RslRlVecEnvWrapper ->
OnPolicyRunner -> learn -> close), with one deviation: this project is an external
Isaac Lab extension, so its own package directory is added to ``sys.path``.

Run:
    D:\\miniconda\\envs\\RM\\python.exe isaac_ws\\scripts\\train.py --task Infantry-RL-Balance-v0 --headless
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))  # for the local cli_args module
_WS = _HERE.parent
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))  # for the isaac_ws_infantry package

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Infantry-RL-Balance-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--no_code_state",
    action="store_true",
    default=False,
    help=(
        "Skip rsl_rl's git-diff logging. Needed in this workspace because a DELETED "
        "non-UTF-8 blob in the git index (infantry-rl/export.log) makes GitPython "
        "produce lone surrogates and the log write raises UnicodeEncodeError, which "
        "aborts training before iteration 1."
    ),
)
parser.add_argument(
    "--resume_latest",
    action="store_true",
    default=False,
    help=(
        "With --resume: load the HIGHEST-numbered model_*.pt instead of relying on "
        "get_checkpoint_path's alphabetical sort (which puts model_0.pt AFTER "
        "model_900.pt, so a plain --resume would restart from an untrained policy). "
        "Scans the newest run directory, or the one matched by --load_run. "
        "Intended for crash recovery: rerun the same command after a crash."
    ),
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaac_ws_infantry  # noqa: F401  (registers the Infantry-RL gym tasks)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _hidden_dims_from_state_dict(state_dict: dict) -> list[int]:
    """Recover ``[h1, h2, ...]`` from the ``mlp.<i>.weight`` entries of a model state dict.

    rsl_rl stores the MLP as ``Linear -> ELU -> Linear -> ... -> Linear``, so the linear
    layers live at even indices and the hidden widths are the ``out_features`` of every
    layer except the output one.  Returns ``[]`` when the layout is not recognised.
    """
    widths: dict[int, int] = {}
    for key, value in state_dict.items():
        if not key.startswith("mlp.") or not key.endswith(".weight"):
            continue
        parts = key.split(".")
        if len(parts) != 3:
            continue
        try:
            index = int(parts[1])
        except ValueError:
            continue
        widths[index] = int(value.shape[0])
    ordered = [widths[i] for i in sorted(widths)]
    return ordered[:-1] if len(ordered) > 1 else []


def _match_network_to_checkpoint(checkpoint_path: str, agent_cfg) -> None:
    """Patch ``agent_cfg`` so a resumed model matches the checkpoint it loads.

    Changing ``hidden_dims`` in the config makes every existing checkpoint unloadable
    (``size mismatch for mlp.0.weight``), which would silently block all continuation
    runs.  When *resuming*, the checkpoint's architecture is the correct one -- you
    cannot change the network and continue the same weights anyway -- so read the widths
    back from the state dict and adopt them, with a notice.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - never let inspection break training
        print(f"[INFO] could not inspect the checkpoint architecture ({exc}); using config")
        return

    changed = False
    for name in ("actor", "critic"):
        state_dict = ckpt.get(f"{name}_state_dict")
        if not isinstance(state_dict, dict):
            continue
        dims = _hidden_dims_from_state_dict(state_dict)
        if not dims:
            continue
        current = list(getattr(agent_cfg, name).hidden_dims or [])
        if current != dims:
            print(f"[INFO] {name} hidden_dims: config {current} -> checkpoint {dims} (resuming with the checkpoint shape)")
            getattr(agent_cfg, name).hidden_dims = dims
            changed = True

    actor_sd = ckpt.get("actor_state_dict") or {}
    first_weight = actor_sd.get("mlp.0.weight")
    if first_weight is not None:
        print(f"[INFO] checkpoint observation dim = {int(first_weight.shape[1])} (env must match)")
    if not changed:
        print("[INFO] checkpoint architecture matches the configured network")


def _newest_checkpoint(log_root_path: str, load_run: str | None = None) -> str:
    """The most-trained ``model_<N>.pt`` of an experiment, across ALL run directories.

    Two traps this helper avoids:

    1. ``get_checkpoint_path`` sorts file names as *strings* padded to 15 characters,
       so ``model_0.pt`` sorts AFTER ``model_900.pt`` -- a plain ``--resume`` would
       restart from an untrained policy.
    2. Continuing a run creates a **new timestamped directory**, and that directory
       starts out containing only the checkpoint it just loaded.  Picking "the newest
       directory" is therefore wrong right after a resume.  Instead the iteration
       number is compared **globally**: rsl-rl keeps counting up across resumes
       (``total_it = start_it + num_learning_iterations``), so the largest ``N`` in the
       whole experiment is always the most-trained file.

    Args:
        log_root_path: the experiment directory, e.g. ``logs/rsl_rl/<experiment>``.
        load_run: optional run-directory name (substring) or glob to restrict the search.

    Returns:
        Absolute path to the checkpoint with the largest iteration number.

    Raises:
        FileNotFoundError: if no (matching) run directory contains any ``model_*.pt``.
    """
    import glob as _glob
    import re as _re

    if load_run in (None, "", ".*"):
        run_dirs = _glob.glob(os.path.join(log_root_path, "*"))
    else:
        pattern = load_run if any(c in load_run for c in "*?[]") else f"*{load_run}*"
        run_dirs = _glob.glob(os.path.join(log_root_path, pattern))

    candidates: list[tuple[int, str, str]] = []
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        for path in _glob.glob(os.path.join(run_dir, "model_*.pt")):
            m = _re.fullmatch(r"model_(\d+)\.pt", os.path.basename(path))
            if m is not None:
                candidates.append((int(m.group(1)), path, os.path.basename(run_dir)))

    if not candidates:
        raise FileNotFoundError(
            f"--resume_latest found no model_*.pt under '{log_root_path}'"
            + (f" matching '{load_run}'" if load_run else "")
        )

    # Pick the checkpoints of the NEWEST run directory that has any, then the highest
    # iteration inside it.  Taking the global maximum iteration instead would be wrong
    # once an experiment contains several independent runs: iteration numbers are only
    # comparable inside one lineage, so a long OLD run (trained with a superseded
    # reward) would win over the newer, correctly-trained one.
    by_run: dict[str, list[tuple[int, str]]] = {}
    for iteration, path, run_name in candidates:
        by_run.setdefault(run_name, []).append((iteration, path))

    def _mtime(run_name: str) -> float:
        try:
            return os.path.getmtime(os.path.join(log_root_path, run_name))
        except OSError:
            return 0.0

    newest_run = max(by_run, key=_mtime)
    iteration, path = max(by_run[newest_run], key=lambda c: c[0])

    global_iteration, _, global_run = max(candidates, key=lambda c: c[0])
    print(f"[INFO] --resume_latest: newest run '{newest_run}' -> iteration {iteration}")
    if global_run != newest_run:
        print(
            f"[INFO] NOTE: another run ('{global_run}') has a higher iteration count "
            f"({global_iteration}), but it is OLDER.  It is ignored on purpose; pass "
            f"--load_run {global_run} --checkpoint <file> if you really want it."
        )
    return os.path.abspath(path)


def _clamp_policy_std(runner, low: float = 0.06, high: float = 0.4) -> None:
    """Bound the Gaussian exploration noise of the policy in place.

    WHY THIS IS NEEDED
    ------------------
    rsl_rl's default ``GaussianDistribution`` keeps the standard deviation as a *raw*
    unconstrained ``nn.Parameter``:

        self.std_param = nn.Parameter(init_std * torch.ones(output_dim))

    With ``std_type="scalar"`` the entropy term alone gives it a gradient
    ``d(entropy_coef * H)/d(sigma) = entropy_coef * n_dims / sigma`` which is always
    POSITIVE.  If the policy-gradient side does not push back hard enough, sigma grows
    without bound.  Measured on the 2026-10-21 run with ``entropy_coef = 0.015``:

        iter 0    mean_std =   0.50
        iter 999  mean_std =  33.72
        iter 3999 mean_std = 168.90

    At that point every action is essentially uniform noise clipped to +/-4, so the
    robot thrashes (large yaw rate, stop-and-go forward speed) and falls.

    The opposite failure is just as real: with ``entropy_coef = 0.005`` on the earlier
    reward set the noise collapsed to 0.067 and the policy could not discover walking.
    So BOTH ends need a bound.  Since v1.0.5 the window is [0.03, 0.4]: at the previous
    [0.02, 1.0] the equilibrium sat ON the ceiling -- measured ``mean_std = 0.9999`` --
    so the policy was trained under ~14 deg of hip noise and came out jittery.  In
    physical units [0.03, 0.4] means:

        hip   noise: 0.03 * 0.25 rad = 0.43 deg  ..  0.4 * 0.25 rad = 5.7 deg
        wheel noise: 0.03 * 5 rad/s  = 0.15 rad/s ..  0.4 * 5 rad/s  = 2.0 rad/s

    The clamp is applied to the parameter BEFORE the distribution is built each step, so
    the sample itself already uses the bounded value (clamping afterwards would be too
    late).  It is a safety net, not a tuning knob: the entropy coefficient should be
    chosen so the equilibrium sits strictly inside the window.
    """
    # rsl_rl keeps the policy on ``PPO.actor`` (``get_policy()`` returns it); fall back to
    # the older ``alg.policy`` spelling in case the naming changes again.
    policy = None
    get_policy = getattr(runner.alg, "get_policy", None)
    if callable(get_policy):
        policy = get_policy()
    if policy is None:
        policy = getattr(runner.alg, "policy", None) or getattr(runner.alg, "actor", None)
    dist = getattr(policy, "distribution", None)
    std_param = getattr(dist, "std_param", None)
    if std_param is None:
        # heteroscedastic / log-parameterised policies have no single std tensor
        print("[INFO] policy std is not a single raw tensor -- no clamp applied")
        return

    original_update = dist.update

    def clamped_update(mlp_output):
        with torch.no_grad():
            dist.std_param.clamp_(low, high)
        original_update(mlp_output)

    dist.update = clamped_update
    with torch.no_grad():
        std_param.clamp_(low, high)
    print(
        f"[INFO] policy std clamped to [{low}, {high}] "
        f"(current {float(std_param.mean()):.4f})"
    )


#: Reward terms whose per-iteration console line is dropped (TensorBoard keeps them).
#: Listed by NAME, in the order of ``LocomotionRewardsCfg``; the numbers in the comment
#: are the positions in the "Active Reward Terms" table printed at startup.
#: Every ``track_*`` term is hidden as well (see ``_print_console_block``): the tracking
#: quality is reported instead by the live ``vel_x`` / ``pitch_deg`` / ``yaw_deg`` lines
#: and by ``error_vel_xy`` / ``error_vel_yaw``.
HIDDEN_CONSOLE_TERMS = (
    "alive",  # 1
    "fall_penalty",  # 2
    "base_height",  # 3
    "pitch",  # 4
    "roll",  # 5
    "roll_straight",  # 6
    "heading_error",  # 11
)


def _live_state(env) -> dict[str, float] | None:
    """Physical state straight from the simulator: base x velocity and roll/pitch/yaw.

    The reward terms only expose the *errors*; these are the quantities the errors are
    computed FROM, which is what makes the console readable at a glance.
    """
    try:
        from isaaclab.utils.math import euler_xyz_from_quat

        unwrapped = env.unwrapped
        asset = unwrapped.scene["robot"]
        vel = asset.data.root_lin_vel_b[:, 0]
        cmd = unwrapped.command_manager.get_command("base_velocity")[:, 0]
        roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
        rad2deg = 180.0 / 3.141592653589793
        return {
            "vel_x": float(vel.mean()),
            "vel_x_cmd": float(cmd.mean()),
            "roll_deg": float(roll.abs().mean()) * rad2deg,
            "pitch_deg": float(pitch.abs().mean()) * rad2deg,
            "yaw_deg": float(yaw.abs().mean()) * rad2deg,
        }
    except Exception:  # noqa: BLE001 - diagnostics must never break training
        return None


#: Console layout.  Everything a reader has to COMPARE sits on one line; only quantities
#: that are read on their own keep a line of their own.
#:   (group title, ((label, bare-name, format), ...))
_CONSOLE_GROUPS = (
    (
        "",
        (
            ("reward", "Mean reward", "{}"),
            ("value_loss", "Mean value", "{:.4f}"),
            ("surrogate", "Mean surrogate", "{:+.4f}"),
            ("entropy", "Mean entropy", "{:.4f}"),
            ("action_std", "Mean action std", "{:.2f}"),
        ),
    ),
    ("terminate", (("time_out", "time_out", "{:.4f}"), ("bad_orient", "bad_orientation", "{:.4f}"),
                   ("too_low", "base_too_low", "{:.4f}"))),
)


def _print_console_block(logger, kwargs, snapshots, hidden: set[str], live: dict[str, float] | None = None) -> None:
    """Print the iteration block, grouping the values that have to be compared.

    rsl_rl prints one line per term, which splits pairs that only mean something together
    (``vel_x`` vs its command, ``error_vel_xy`` vs ``error_vel_yaw``, pitch vs roll vs yaw).
    Here every comparable pair is emitted on ONE line.  TensorBoard still gets everything.
    """
    def mean_of(key: str) -> float | None:
        values = []
        for snap in snapshots:
            if key in snap:
                v = snap[key]
                v = v.detach().cpu().double() if hasattr(v, "detach") else float(v)
                values.append(float(v.mean()) if hasattr(v, "mean") else float(v))
        return sum(values) / len(values) if values else None

    # ---- collect every term that survived the filter ----------------------
    keys: list[str] = []
    for snap in snapshots:
        for key in snap:
            if key not in keys:
                keys.append(key)
    values: dict[str, float] = {}
    for key in keys:
        bare = key.rsplit("/", 1)[-1]
        if bare in hidden or bare.startswith("track_"):
            continue
        v = mean_of(key)
        if v is not None:
            values[bare] = v

    def row(*cells: tuple[str, str], width: int = 26) -> str:
        return "  " + "".join(f"{f'{label} {value}':<{width}}" for label, value in cells).rstrip() + "\n"

    def cell(label: str, fmt: str, width: int) -> tuple[str, str]:
        v = values.pop(label, None)
        return (label, "  n/a" if v is None else " " + fmt.format(v))

    out = "\n" + "#" * 78 + "\n"
    out += f"{'Learning iteration ' + str(kwargs.get('it', 0)) + '/' + str(kwargs.get('total_it', 0)):^78}\n\n"

    # ---- training summary ------------------------------------------------
    rewbuf = getattr(logger, "rewbuffer", None)
    std = kwargs.get("action_std")
    std = float(std.mean()) if std is not None else float("nan")
    losses = {k: float(v) for k, v in (kwargs.get("loss_dict") or {}).items()}
    out += row(
        ("reward", f"{sum(rewbuf) / len(rewbuf):.2f}" if rewbuf else "n/a"),
        ("value_loss", f"{losses.get('value', float('nan')):.4f}"),
        ("surrogate", f"{losses.get('surrogate', float('nan')):+.4f}"),
        ("entropy", f"{losses.get('entropy', float('nan')):.4f}"),
        ("action_std", f"{std:.2f}"),
    )

    # ---- the comparison block --------------------------------------------
    # vel_x / cmd / signed error, then the two angle sets, then the two error
    # norms on the next line so the two tracking errors can be compared directly.
    if live:
        out += row(
            ("vel_x m/s", f"{live['vel_x']:+.3f}"),
            ("cmd", f"{live['vel_x_cmd']:+.3f}"),
            ("diff", f"{live['vel_x'] - live['vel_x_cmd']:+.3f}"),
        )
        out += row(
            ("pitch deg", f"{live['pitch_deg']:.2f}"),
            ("roll deg", f"{live['roll_deg']:.2f}"),
            ("yaw deg", f"{live['yaw_deg']:.2f}"),
        )
    out += row(cell("error_vel_xy", "{:.4f}", 22), cell("error_vel_yaw", "{:.4f}", 22))
    out += "  (error_vel_* = 2x the true mean error)\n"

    # ---- termination reasons --------------------------------------------
    out += row(cell("time_out", "{:.4f}", 20), cell("bad_orientation", "{:.4f}", 22),
               cell("base_too_low", "{:.4f}", 20))

    # ---- penalties, paired so the two of each kind sit together ----------
    out += row(cell("action_rate", "{:+.4f}", 22), cell("action_jerk", "{:+.4f}", 22))
    out += row(cell("joint_torques", "{:+.4f}", 22), cell("joint_acc", "{:+.4f}", 22))
    out += row(cell("wheel_torques", "{:+.4f}", 22), cell("wheel_spin", "{:+.4f}", 22))
    out += row(cell("wheel_axle", "{:+.4f}", 22), cell("closure_residual", "{:+.4f}", 22))
    out += row(cell("roll_rate", "{:+.4f}", 22), cell("lin_vel_z", "{:+.4f}", 22),
               cell("lin_vel_y", "{:+.4f}", 22))

    # ---- anything the layout above does not know about -------------------
    for name, value in values.items():
        out += row((name, f"{value:+.4f}"))

    print(out, end="", flush=True)


def _filter_iteration_console(runner, hidden: tuple[str, ...] = HIDDEN_CONSOLE_TERMS) -> None:
    """Print the iteration block ourselves so that ``hidden`` terms are left out."""
    logger = getattr(runner, "logger", None)
    original = getattr(logger, "log", None)
    if not callable(original):
        return
    hidden_set = set(hidden)

    env = getattr(runner, "env", None)

    def filtered(**kwargs):
        snapshots = [dict(e) for e in getattr(logger, "ep_extras", [])]
        live = _live_state(env) if env is not None else None
        saved = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            original(**kwargs)  # writes every scalar to TensorBoard
        finally:
            sys.stdout.close()
            sys.stdout = saved
        _print_console_block(logger, kwargs, snapshots, hidden_set, live)

    logger.log = filtered
    print(f"[INFO] console hides {len(hidden)} reward terms: {', '.join(hidden)}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if args_cli.resume_latest:
            resume_path = _newest_checkpoint(log_root_path, agent_cfg.load_run)
            print(f"[INFO] --resume_latest resolved to: {resume_path}")
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        # the runner builds the network from agent_cfg, so the shape has to be
        # reconciled with the checkpoint BEFORE the runner is created
        _match_network_to_checkpoint(resume_path, agent_cfg)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            # 1 帧 = 1 个控制步 = 20 ms -> 50 fps 才是真实速度。
            # gymnasium 的 RecordVideo 默认 30 fps，会让回放看起来慢 1.67 倍。
            "fps": round(1.0 / float(env.unwrapped.step_dt)),
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    try:
        runner.add_git_repo_to_log(__file__)
    except Exception as exc:  # pragma: no cover - logging must never break training
        logger.warning(f"could not record the git state: {exc}")
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # Bounds on the policy's exploration noise (see the docstring).
    _clamp_policy_std(runner)

    # Console: drop the listed per-term lines, print the rest with bare names.
    # TensorBoard still receives every scalar.
    _filter_iteration_console(runner)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    print(
        f"[INFO] task={args_cli.task}  num_envs={env_cfg.scene.num_envs}  "
        f"max_iterations={agent_cfg.max_iterations}  device={agent_cfg.device}"
    )

    # ------------------------------------------------------------------ #
    # Optional: skip rsl_rl's "store the git diff of the repo" step.
    #
    # rsl_rl 5.0.1 runs ``repo.git.diff(head_tree)`` through GitPython and then
    # writes the result with ``open(..., encoding="utf-8")``.  GitPython decodes
    # git's output with surrogateescape, so any DELETED file whose blob is not
    # valid UTF-8 turns into lone surrogates and the write raises
    # ``UnicodeEncodeError: surrogates not allowed``.  In this workspace that is
    # ``infantry-rl/export.log`` (a 1 MB GBK-ish log) sitting in the index as a
    # pending deletion, which aborts training before the first iteration.
    #
    # Clearing ``git_status_repos`` only skips that bookkeeping file; TensorBoard
    # logging, checkpoints and the params YAML are unaffected.
    # ------------------------------------------------------------------ #
    if args_cli.no_code_state:
        try:
            n = len(runner.logger.git_status_repos)
            runner.logger.git_status_repos = []
            print(f"[INFO] --no_code_state: skipped rsl_rl git-diff logging ({n} repo path(s) dropped)")
        except AttributeError as exc:  # pragma: no cover
            logger.warning(f"could not disable git-diff logging: {exc}")

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    print(f"[INFO] final closure residual (m): {env.unwrapped.max_closure_residual:.3e}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
