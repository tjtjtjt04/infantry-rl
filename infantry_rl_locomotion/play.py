"""Play (evaluate) a trained squat checkpoint on the chassis-rl robot.

Examples:
    python play.py --task squat --headless --steps 2000 --load_run <run> --checkpoint model_598.pt
    python play.py --task squat_torque --steps 0 --real-time --load_run <run> --checkpoint model_600.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained squat policy.")
    parser.add_argument("--task", type=str, default="squat", help="squat | squat_torque")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000,
                        help="control steps to run; 0 = forever")
    parser.add_argument("--real-time", action="store_true", default=False)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use_last_checkpoint", action="store_true", default=False)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--log_root", type=str, default=None)
    parser.add_argument("--random_policy", action="store_true", default=False,
                        help="skip the checkpoint and run a freshly initialised policy")
    parser.add_argument("--zero_action", action="store_true", default=False,
                        help="skip the policy and send zero actions")
    parser.add_argument("--check", action="store_true", default=False,
                        help="run the environment interface checks and exit")
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--follow", action="store_true", default=False,
                        help="camera follows the robot")
    parser.add_argument("--camera", choices=("free", "follow", "fixed"), default=None)
    parser.add_argument("--camera_eye", type=float, nargs=3, default=None)
    parser.add_argument("--camera_lookat", type=float, nargs=3, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
if args_cli.check:
    args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows (imports require Kit)."""

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.dict import print_dict  # noqa: E402

import agent as agent_lib  # noqa: E402
from config import DEFAULT_LOG_ROOT, normalize_task  # noqa: E402
from env_squat import (  # noqa: E402
    SQUAT_BASE_HEIGHT,
    FiveBarSquatEnv,
    FiveBarSquatTorqueEnv,
    SquatEnvCfg_PLAY,
    SquatTorqueEnvCfg_PLAY,
)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402


def build_play_env(task: str, num_envs: int | None, device: str, seed: int, render_mode: str | None = None):
    if task == "squat":
        env_cfg = SquatEnvCfg_PLAY()
        env_cls = FiveBarSquatEnv
    elif task == "squat_torque":
        env_cfg = SquatTorqueEnvCfg_PLAY()
        env_cls = FiveBarSquatTorqueEnv
    else:
        raise ValueError(f"unknown task {task!r}")
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.sim.device = device
    env_cfg.log_dir = os.path.join(os.getcwd(), "logs", "play")
    return env_cls(cfg=env_cfg, render_mode=render_mode)


def run_check(env, task: str) -> int:
    robot = env.scene["robot"]
    joints = env.action_manager.get_term(env.cfg.joint_action_name)
    print(
        f"[check] task={task} num_envs={env.cfg.scene.num_envs} "
        f"obs={tuple(env.single_observation_space['policy'].shape)} "
        f"actions={tuple(env.single_action_space.shape)}"
    )
    print(f"[check] joints={robot.joint_names}")
    print(f"[check] hip ids={joints.hip_ids}  knee ids={joints.knee_ids}  wheel ids={joints.wheel_ids}")
    print(f"[check] closure after reset: {env.max_closure_residual * 1000:.6f} mm")
    obs, extras = env.reset()
    action = torch.zeros((env.cfg.scene.num_envs, env.single_action_space.shape[0]), device=env.device)
    obs, reward, terminated, truncated, extras = env.step(action)
    obs = obs["policy"] if isinstance(obs, dict) else obs
    ok = bool(torch.isfinite(obs).all()) and bool(torch.isfinite(reward).all())
    ok &= env.max_closure_residual < 5e-3
    print(
        f"[check] one step: reward={reward.mean().item():.4f} "
        f"done={int((terminated | truncated).sum())} "
        f"closure={env.max_closure_residual * 1000:.6f} mm"
    )
    print("[check] RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    import importlib.metadata as metadata

    task = normalize_task(args_cli.task)
    agent_cfg = agent_lib.make_agent_cfg(task, play=True)
    agent_cfg = agent_lib.apply_cli_overrides(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    log_root = Path(args_cli.log_root) if args_cli.log_root else DEFAULT_LOG_ROOT
    log_root_path = os.path.abspath(log_root / agent_cfg.experiment_name)

    env = build_play_env(
        task,
        args_cli.num_envs,
        args_cli.device or "cuda:0",
        args_cli.seed,
        render_mode="rgb_array" if args_cli.video else None,
    )

    camera_mode = args_cli.camera or ("follow" if args_cli.follow else "free")
    cfg_viewer = env.cfg.viewer
    camera_eye = tuple(args_cli.camera_eye) if args_cli.camera_eye is not None else tuple(cfg_viewer.eye)
    camera_lookat = tuple(args_cli.camera_lookat) if args_cli.camera_lookat is not None else tuple(cfg_viewer.lookat)
    camera_enabled = (not args_cli.headless) or args_cli.video

    controller = getattr(env, "viewport_camera_controller", None)
    if camera_mode == "follow":
        if controller is None:
            print("[play] camera=follow ignored: no viewport (headless without --video)")
        else:
            controller.default_cam_eye = np.array(camera_eye, dtype=float)
            controller.default_cam_lookat = np.array(camera_lookat, dtype=float)
            controller.update_view_to_asset_root("robot")
            print(f"[play] camera=follow: eye offset={camera_eye}, lookat offset={camera_lookat}")
    elif camera_enabled:
        env.sim.set_camera_view(eye=camera_eye, target=camera_lookat, camera_prim_path=cfg_viewer.cam_prim_path)
        print(f"[play] camera={camera_mode}: eye={camera_eye}, lookat={camera_lookat}")

    apply_fixed_camera = camera_mode == "fixed" and camera_enabled

    if args_cli.check:
        return run_check(env, task)

    resume_path: str | None = None
    if args_cli.zero_action:
        print("[play] --zero_action: skipping the checkpoint; sending zero actions")
    elif not args_cli.random_policy:
        resume_path = agent_lib.resolve_play_checkpoint(
            log_root_path,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
            use_last_checkpoint=args_cli.use_last_checkpoint,
        )
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent_lib.match_network_to_checkpoint(resume_path, agent_cfg)
    else:
        print("[play] --random_policy: no checkpoint, running a freshly initialised policy")

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
    if resume_path is not None:
        runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)
    if args_cli.zero_action:
        action_dim = env.unwrapped.single_action_space.shape[0]
        policy = lambda _obs: torch.zeros(
            (env.unwrapped.num_envs, action_dim), device=env.unwrapped.device
        )

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    joints = env.unwrapped.action_manager.get_term(env.unwrapped.cfg.joint_action_name)

    heights: list[float] = []
    pitches: list[float] = []
    rolls: list[float] = []
    residuals: list[float] = []
    hip_lock_errs: list[float] = []
    hip_tau_max: list[float] = []
    wheel_tau_max: list[float] = []

    count = 0
    while simulation_app.is_running() and (args_cli.steps == 0 or count < args_cli.steps):
        if apply_fixed_camera:
            env.unwrapped.sim.set_camera_view(
                eye=camera_eye, target=camera_lookat, camera_prim_path=cfg_viewer.cam_prim_path
            )
        start = time.time()
        actions = policy(obs)
        obs, rewards, dones, infos = env.step(actions)
        count += 1

        if count % 50 == 0 or count == 1:
            heights.append(float(robot.data.root_pos_w[0, 2] - env.unwrapped.scene.env_origins[0, 2]))
            roll, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
            rolls.append(float(roll[0]))
            pitches.append(float(pitch[0]))
            residuals.append(env.unwrapped.max_closure_residual)
            hip_lock_errs.append(
                float((robot.data.joint_pos[0, joints.hip_ids] - joints.locked_hips).abs().max())
            )
            if getattr(joints, "torque_control", False):
                hip_tau_max.append(float(joints.hip_torques[0].abs().max()))
                wheel_tau_max.append(float(joints.wheel_torques[0].abs().max()))
            print(
                f"[play] step {count:5d}  base_z={heights[-1]:.4f}  "
                f"pitch={math.degrees(pitches[-1]):+6.2f} deg  "
                f"roll={math.degrees(rolls[-1]):+5.2f} deg  "
                f"closure={residuals[-1] * 1000:.3f} mm"
                + (f"  tau_hip={hip_tau_max[-1]:.2f} Nm" if hip_tau_max else ""),
                flush=True,
            )

        if args_cli.real_time:
            elapsed = time.time() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    if heights:
        print(
            f"[play] {task}: steps={count}  base_z mean={sum(heights) / len(heights):.4f} m "
            f"(target {SQUAT_BASE_HEIGHT:.4f}), |pitch| mean="
            f"{sum(abs(p) for p in pitches) / len(pitches):.3f} deg, "
            f"|roll| mean={sum(abs(r) for r in rolls) / len(rolls):.3f} deg, "
            f"max closure={max(residuals) * 1000:.3f} mm, "
            f"max |hip-locked|={math.degrees(max(hip_lock_errs)):.2f} deg"
        )
        if hip_tau_max:
            print(
                f"[play] torque: max |tau_hip|={max(hip_tau_max):.3f} Nm, "
                f"max |tau_wheel|={max(wheel_tau_max):.3f} Nm"
            )
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
