"""Play (evaluate) a trained RSL-RL checkpoint on the infantry-rl balance task.

Run:
    D:\\miniconda\\envs\\RM\\python.exe isaac_ws\\scripts\\play.py --task Infantry-RL-Balance-Play-v0 ^
        --checkpoint <path\\to\\model_*.pt> --num_envs 16 --headless
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_WS = _HERE.parent
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a trained RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Infantry-RL-Balance-Play-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# NOTE: ``--checkpoint`` is registered by ``cli_args.add_rsl_rl_args`` further down
# (as ``--checkpoint`` with a glob default of "model_.*.pt").  Declaring it here as
# well made argparse abort with "conflicting option string: --checkpoint", so the
# resolution below reads ``agent_cfg.load_checkpoint`` instead of args_cli.
parser.add_argument("--use_last_checkpoint", action="store_true", default=False, help="Use the last checkpoint.")
parser.add_argument("--steps", type=int, default=500, help="Number of control steps to run before closing.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug_heading",
    action="store_true",
    default=False,
    help=(
        "Print the policy's wheel commands, the heading-loop delta and the ACTUAL wheel "
        "velocities every 50 steps. Use this to check whether the heading correction "
        "actually reaches the wheels."
    ),
)
parser.add_argument(
    "--cmd_vx",
    type=float,
    default=None,
    help=(
        "Velocity-tracking tasks only: hold this forward command (m/s) instead of the "
        "randomly sampled one. Use with --cmd_heading_deg for a repeatable evaluation, "
        "e.g. --cmd_vx 0.3 --cmd_heading_deg 0 drives straight ahead."
    ),
)
parser.add_argument(
    "--cmd_heading_deg",
    type=float,
    default=None,
    help="Velocity-tracking tasks only: hold this heading command (degrees). Defaults to 0 when --cmd_vx is given.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaac_ws_infantry  # noqa: F401  (registers the Infantry-RL gym tasks)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    import importlib.metadata as metadata

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    # resolve the checkpoint
    #
    # ``--checkpoint`` comes from ``cli_args.add_rsl_rl_args`` (there is no local
    # copy: declaring one here made argparse abort with a conflicting-option error).
    # ``agent_cfg.load_checkpoint`` therefore holds either its glob default
    # "model_.*.pt" or whatever the user passed.  Accept, in order:
    #   * an existing file path (absolute, or relative to the CWD),
    #   * a bare filename inside the log tree,
    #   * a glob/regex resolved inside the newest run directory.
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    requested = str(agent_cfg.load_checkpoint or "")
    if args_cli.use_last_checkpoint or not requested:
        pattern = "model_.*.pt"
    else:
        pattern = requested

    direct = os.path.abspath(pattern)
    if os.path.isfile(direct):
        resume_path = direct
    elif os.path.sep in pattern or "/" in pattern:
        # a path relative to the log root or to a run dir, e.g.
        # "2026-10-17_00-39-38_fixcheck/model_2.pt"
        candidates = [
            os.path.join(log_root_path, pattern),
            os.path.join(log_root_path, agent_cfg.load_run or "", pattern),
            os.path.join(os.getcwd(), pattern),
        ]
        hit = next((c for c in candidates if os.path.isfile(c)), None)
        if hit is None:
            raise FileNotFoundError(
                f"no checkpoint at '{pattern}'; tried:\n  " + "\n  ".join(os.path.abspath(c) for c in candidates)
            )
        resume_path = hit
    else:
        # a bare filename (or glob) resolved inside the newest matching run dir
        resume_path = get_checkpoint_path(
            log_root_path,
            agent_cfg.load_run,
            f"^{re.escape(pattern)}$" if ("*" not in pattern and "?" not in pattern) else pattern,
        )
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    steps = args_cli.steps
    residuals = []
    heights = []
    headings = []
    yaw_rates = []
    rolls = []
    pitches = []
    xs = []
    ys = []
    vxs = []

    robot = env.unwrapped.scene["robot"]
    origins = env.unwrapped.scene.env_origins

    # ------------------------------------------------------------------ #
    # optional fixed velocity command (locomotion tasks)
    #
    # Without this the command term samples a random command when the environment is
    # created, so two runs of the same checkpoint are not comparable.  Pinning the
    # command makes the numbers reproducible (and lets us test "drive straight").
    # ------------------------------------------------------------------ #
    fixed_cmd = args_cli.cmd_vx is not None or args_cli.cmd_heading_deg is not None

    def _apply_fixed_cmd(term) -> None:
        term.is_heading_env[:] = True  # heading controller owns wz
        term.is_standing_env[:] = False  # do not let it zero our command
        term.vel_command_b[:] = 0.0
        term.heading_target[:] = math.radians(args_cli.cmd_heading_deg or 0.0)
        if args_cli.cmd_vx is not None:
            term.vel_command_b[:, 0] = args_cli.cmd_vx

    if fixed_cmd:
        try:
            _cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
            _apply_fixed_cmd(_cmd_term)
            print(
                f"[play] fixed command: vx={args_cli.cmd_vx} m/s, "
                f"heading={args_cli.cmd_heading_deg or 0.0:.1f} deg "
                f"(wz is derived from the heading error every step)"
            )
        except (AttributeError, KeyError, ValueError):
            print("[play] --cmd_vx/--cmd_heading_deg ignored: this task has no 'base_velocity' command")
            fixed_cmd = False

    def _yaw():
        """Current heading (rad) of env 0, for the summary line."""
        q = robot.data.root_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _roll_pitch():
        """Roll and pitch (rad) of env 0, for the attitude summary line."""
        q = robot.data.root_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        # XYZ extrinsic convention, same as isaaclab.utils.math.euler_xyz_from_quat
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.asin(sinp)
        return roll, pitch

    def _xy():
        """Planar displacement of env 0 relative to its spawn point, metres."""
        p = robot.data.root_pos_w[0] - origins[0]
        return float(p[0]), float(p[1])

    for step in range(steps):
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        # a termination auto-resets the env, which re-samples a random command --
        # put the fixed command back so the evaluation stays comparable
        if fixed_cmd and bool(dones.any()):
            _apply_fixed_cmd(_cmd_term)
        residuals.append(env.unwrapped.max_closure_residual)
        heights.append(float(robot.data.root_pos_w[:, 2].mean().item()))
        headings.append(_yaw())
        yaw_rates.append(float(robot.data.root_ang_vel_w[:, 2].mean().item()))
        _roll, _pitch = _roll_pitch()
        rolls.append(_roll)
        pitches.append(_pitch)
        _x, _y = _xy()
        xs.append(_x)
        ys.append(_y)
        vxs.append(float(robot.data.root_lin_vel_w[0, 0].item()))
        if step % 50 == 0:
            print(
                f"[play] step {step:4d}  base_z={heights[-1]:.4f}  "
                f"x={xs[-1]:+6.3f}m y={ys[-1]:+6.3f}m vx={vxs[-1]:+6.3f}m/s  "
                f"heading={math.degrees(headings[-1]):+7.2f} deg  "
                f"yaw_rate={yaw_rates[-1]:+.3f} rad/s  "
                f"closure_max={residuals[-1]:.3e} m",
                flush=True,
            )
            if args_cli.debug_heading:
                term = env.unwrapped.action_manager.get_term("joints")
                wv = robot.data.joint_vel[0, term.wheel_ids].tolist()
                fd = (headings[-1] - headings[-2]) / dt if len(headings) > 1 else 0.0
                print(
                    f"       [dbg] policy wheel cmd={[round(float(v), 2) for v in term._wheel_targets[0]]}"
                    f"  heading_delta={float(term.heading_delta[0]):+.2f}"
                    f"  speed_delta={float(getattr(term, 'speed_delta', torch.zeros(1))[0]):+.2f}"
                    f"  actual wheel vel={[round(float(v), 2) for v in wv]}"
                    f"\n       [dbg] yaw_rate(world_z)={yaw_rates[-1]:+.4f}"
                    f"  yaw_rate(body_z)={float(robot.data.root_ang_vel_b[0, 2]):+.4f}"
                    f"  yaw_rate(finite diff)={fd:+.4f} rad/s"
                    f"  heading_control={term.cfg.heading_control}"
                    f"  position_control={getattr(term.cfg, 'position_control', False)}",
                    flush=True,
                )
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # wrap the accumulated heading to (-180, 180] before reporting the spread so a
    # robot that stays put does not look like it turned 2*pi
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    total_turn = math.degrees(headings[-1] - headings[0])
    max_dev = math.degrees(max(abs(_wrap(h - headings[0])) for h in headings))
    n = max(1, len(rolls))
    mean_roll = math.degrees(sum(abs(v) for v in rolls) / n)
    max_roll = math.degrees(max(abs(v) for v in rolls))
    mean_pitch = math.degrees(sum(abs(v) for v in pitches) / n)
    max_pitch = math.degrees(max(abs(v) for v in pitches))
    print(
        f"[play] steps={steps}  base_z mean={sum(heights) / len(heights):.4f} m  "
        f"max closure residual={max(residuals):.3e} m\n"
        f"[play] heading: net turn={total_turn:+.2f} deg  max |deviation|={max_dev:.2f} deg  "
        f"mean |yaw rate|={sum(abs(v) for v in yaw_rates) / max(1, len(yaw_rates)):.3f} rad/s\n"
        f"[play] attitude: mean |pitch|={mean_pitch:.2f} deg (max {max_pitch:.2f})  "
        f"mean |roll|={mean_roll:.2f} deg (max {max_roll:.2f})\n"
        f"[play] position: net displacement=({xs[-1] - xs[0]:+.3f}, {ys[-1] - ys[0]:+.3f}) m  "
        f"max |x|={max(abs(v - xs[0]) for v in xs):.3f} m  "
        f"mean |vx|={sum(abs(v) for v in vxs) / max(1, len(vxs)):.3f} m/s"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
