"""Train the pure-torque squat policy for the chassis-rl five-bar robot with RSL-RL.

The hardware is torque controlled: 4 hip torques + 2 wheel torques, 25-dim policy
observation.  The shared scene/action/loop environment lives in ``env_base.py``;
the task config lives in ``env_squat.py``.

Examples:
    python train.py --headless --num_envs 1024 --iterations 600
    python train.py --task squat_torque --headless --check --num_envs 4
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
    parser = argparse.ArgumentParser(description="Train an RSL-RL agent for the chassis-rl squat.")
    parser.add_argument("--task", type=str, default="squat_torque",
                        help="squat_torque (the only task)")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--iterations", "--max_iterations", dest="iterations", type=int, default=None,
                        help="number of learning iterations (added to a resumed iteration)")
    parser.add_argument("--horizon", "--num_steps_per_env", dest="horizon", type=int, default=None,
                        help="rollout length per environment")
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--reset_obs_norm_count", type=int, default=None,
                        help="on resume, lower the observation normalizer sample count")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--log_root", type=str, default=None,
                        help="root for rsl_rl logs (default: <project>/logs/rsl_rl)")
    parser.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
    parser.add_argument("--log_project_name", type=str, default=None,
                        help="wandb/neptune project name")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume_latest", action="store_true", default=False,
                        help="resume the newest checkpoint in --load_run")
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--check", action="store_true", default=False,
                        help="run the environment interface checks and exit")
    parser.add_argument("--no_code_state", action="store_true", default=False)
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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows (imports require Kit)."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402

import agent as agent_lib  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_LOG_ROOT,
    HIP_EFFORT_LIMIT,
    normalize_task,
)
from env_getup import FiveBarGetupEnv, GetupEnvCfg  # noqa: E402
from env_squat import (  # noqa: E402
    SQUAT_BASE_HEIGHT,
    FiveBarSquatEnv,
    SquatEnvCfg,
)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

#: The env exposes a 5-frame history of the 25-D per-frame observation.
#: The actor splits it into current 25 + encoder(125) -> 3 latent = 28 inputs.
#: Both tasks use exactly this layout (the ONNX contract with the firmware).
EXPECTED_OBS_DIM = 125

#: task -> (env cfg class, env class).  ``getup_torque`` shares the observation and
#: action layout with ``squat_torque`` on purpose, so checkpoints and the ONNX export
#: are interchangeable between the two.
ENV_CFGS: dict[str, type] = {"squat_torque": SquatEnvCfg, "getup_torque": GetupEnvCfg}
ENV_CLASSES: dict[str, type] = {"squat_torque": FiveBarSquatEnv, "getup_torque": FiveBarGetupEnv}


def build_env_cfg(task: str, num_envs: int | None, device: str, seed: int, log_dir: str):
    if task not in ENV_CFGS:
        raise ValueError(f"unknown task {task!r}; expected one of {tuple(ENV_CFGS)}")
    env_cfg = ENV_CFGS[task]()
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.sim.device = device
    env_cfg.log_dir = log_dir
    return env_cfg


def make_env(task: str, env_cfg, render_mode: str | None = None):
    if task not in ENV_CLASSES:
        raise ValueError(f"unknown task {task!r}; expected one of {tuple(ENV_CLASSES)}")
    return ENV_CLASSES[task](cfg=env_cfg, render_mode=render_mode)


def run_getup_checks(env, task: str) -> bool:
    """Exercise the interfaces the fall-recovery task must get right.

    Checks that the same 125/6 interface is in place, that the reset really starts the
    robot on the ground (not standing, not inside it), and that a zero-action rollout
    stays finite and does not terminate -- i.e. the measured contact height is right.
    """
    ok = True
    cfg = env.cfg
    n = cfg.scene.num_envs
    robot = env.scene["robot"]
    joint_action = env.action_manager.get_term(cfg.joint_action_name)
    joint_action.set_action_delay(0)
    joint_action.set_motor_torque_scale(1.0)
    actions = torch.zeros((n, env.action_space.shape[-1]), device=env.device)

    obs, _ = env.reset()
    obs = obs["policy"] if isinstance(obs, dict) else obs
    obs_dim = env.single_observation_space["policy"].shape[0]
    print(
        f"[check] obs shape={tuple(obs.shape)} (expected {EXPECTED_OBS_DIM}) "
        f"finite={bool(torch.isfinite(obs).all())}"
    )
    ok &= obs_dim == EXPECTED_OBS_DIM
    ok &= obs.shape == (n, obs_dim)
    ok &= bool(torch.isfinite(obs).all())
    print(f"[check] num_envs={n}  ep_steps={env.max_episode_length}  "
          f"closure_max={env.max_closure_residual * 1000:.6f} mm")
    ok &= env.max_closure_residual < 5e-3

    _, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    pitch_deg = torch.rad2deg(pitch)
    print(
        f"[check] reset pitch {pitch_deg.min().item():+.1f}..{pitch_deg.max().item():+.1f} deg "
        f"(|pitch| >= 10 expected), base_z {height.min().item():.4f}..{height.max().item():.4f} m, "
        f"reset mix: tripod 15-19 deg by default "
        f"(INFANTRY_RL_GETUP_REST_PROB < 1 adds the 30-40 deg fall)"
    )
    ok &= bool((pitch_deg.abs() > 10.0).all())
    ok &= bool((pitch_deg.abs() < 45.0).all())
    ok &= bool((height > 0.14).all())
    ok &= bool((height < 0.30).all())

    obs, reward, terminated, truncated, extras = env.step(actions)
    done = terminated | truncated
    print(f"[check] initial zero step: done={int(done.sum())}/{n}")
    ok &= not bool(done.any())

    for step in range(60):
        obs, reward, terminated, truncated, extras = env.step(actions)
        obs = obs["policy"] if isinstance(obs, dict) else obs
        ok &= bool(torch.isfinite(obs).all())
        ok &= bool(torch.isfinite(reward).all())
        done = terminated | truncated
        ok &= not bool(done.any())

    _, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    rate = robot.data.root_ang_vel_b.norm(dim=-1)
    print(
        f"[check] after 0.6 s of zero action: pitch "
        f"{torch.rad2deg(pitch).min().item():+.1f}..{torch.rad2deg(pitch).max().item():+.1f} deg, "
        f"base_z {height.min().item():.4f}..{height.max().item():.4f} m, |rate| max "
        f"{rate.max().item():.2f} rad/s"
    )
    ok &= bool((torch.rad2deg(pitch).abs() < 60.0).all())
    ok &= bool((height > 0.10).all())
    ok &= env.max_closure_residual < 5e-3

    print("[check] RESULT:", "PASS" if ok else "FAIL")
    return ok


def run_env_checks(env, task: str) -> bool:
    """Exercise the interfaces that must work before a long training run."""
    if task != "squat_torque":
        return run_getup_checks(env, task)
    ok = True
    cfg = env.cfg
    robot = env.scene["robot"]
    joint_action = env.action_manager.get_term(cfg.joint_action_name)
    # Keep the torque probes deterministic: no transport delay / motor strength
    # variation (both are domain randomization for training).
    joint_action.set_action_delay(0)
    joint_action.set_motor_torque_scale(1.0)
    actions = torch.zeros((cfg.scene.num_envs, env.action_space.shape[-1]), device=env.device)

    def policy_obs(observation):
        return observation["policy"] if isinstance(observation, dict) else observation

    obs_dim = env.single_observation_space["policy"].shape[0]
    obs, _ = env.reset()
    obs = policy_obs(obs)
    print(
        f"[check] obs shape={tuple(obs.shape)} (expected {EXPECTED_OBS_DIM}) "
        f"finite={bool(torch.isfinite(obs).all())}"
    )
    ok &= obs_dim == EXPECTED_OBS_DIM
    ok &= obs.shape == (cfg.scene.num_envs, obs_dim)
    ok &= bool(torch.isfinite(obs).all())
    print(f"[check] num_envs={cfg.scene.num_envs}  ep_steps={env.max_episode_length}  "
          f"closure_max={env.max_closure_residual * 1000:.6f} mm")
    ok &= env.max_closure_residual < 5e-3

    obs, reward, terminated, truncated, extras = env.step(actions)
    done = terminated | truncated
    print(f"[check] initial zero step: done={int(done.sum())}/{cfg.scene.num_envs}")
    ok &= not bool(done.any())

    env.reset()
    for step in range(10):
        obs, reward, terminated, truncated, extras = env.step(torch.zeros_like(actions))
        obs = policy_obs(obs)
        print(
            f"[check] zero step={step} reward={reward.mean().item(): .4f} "
            f"h={robot.data.root_pos_w[0, 2].item(): .3f} "
            f"done={int((terminated | truncated).sum())}"
        )
        ok &= bool(torch.isfinite(obs).all())
        ok &= bool(torch.isfinite(reward).all())

    env.episode_length_buf[:] = env.max_episode_length - 1
    obs, reward, terminated, truncated, extras = env.step(torch.zeros_like(actions))
    done = terminated | truncated
    print(f"[check] forced timeout: done={int(done.sum())}/{cfg.scene.num_envs}")
    ok &= bool(done.all())

    hip_offsets = robot.data.joint_pos[:, joint_action.hip_ids] - joint_action.nominal_hips
    heights = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    home_error = hip_offsets.abs().max().item()
    hip_err = (robot.data.joint_pos[:, joint_action.hip_ids] - joint_action.locked_hips).abs().max().item()
    print(
        f"[check] after reset: max |hip - locked|={hip_err:.4f} rad, "
        f"base_z {heights.min().item():.4f}-{heights.max().item():.4f} m "
        f"(target {SQUAT_BASE_HEIGHT:.4f}), closure={env.max_closure_residual * 1000:.6f} mm"
    )
    ok &= home_error < 0.75
    ok &= hip_err < 0.06
    ok &= abs(heights.max().item() - SQUAT_BASE_HEIGHT) < 0.03
    ok &= abs(heights.min().item() - SQUAT_BASE_HEIGHT) < 0.03
    ok &= env.max_closure_residual < 5e-3

    env.reset()
    action_dim = int(env.action_space.shape[-1])
    heights = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    locked = [round(float(v), 4) for v in joint_action.locked_hips]
    zero = torch.zeros((cfg.scene.num_envs, action_dim), device=env.device)

    print(
        f"[check] squat_torque: action_dim={action_dim} (expected 6), "
        f"base_z {heights.min().item():.4f}-{heights.max().item():.4f} m, "
        f"locked hips={locked} rad, closure={env.max_closure_residual * 1000:.6f} mm"
    )
    ok &= action_dim == 6
    env.step(zero)
    zero_hip_err = (
        joint_action.hip_targets - joint_action.locked_hips
    ).abs().max().item()
    zero_wheel = joint_action.wheel_targets.abs().max().item()
    print(
        f"[check] zero action: max |hip target - squat|={zero_hip_err:.3e} rad, "
        f"max |wheel velocity target|={zero_wheel:.3e} rad/s"
    )
    ok &= zero_hip_err < 1e-6
    ok &= zero_wheel < 1e-6

    probe = torch.zeros_like(zero)
    # 0.5 * 0.5 rad/unit = 0.25 rad hip position offset
    probe[:, :4] = 0.5
    env.step(probe)
    hip_err = (joint_action.hip_targets - joint_action.locked_hips)
    print(
        f"[check] 0.5 hip probe: target offset range "
        f"{hip_err.min().item():+.4f}..{hip_err.max().item():+.4f} rad (expected +0.25)"
    )
    # some hips may already sit at a URDF limit, so the target can clamp below 0.25
    ok &= bool((hip_err >= -1e-5).all() and (hip_err <= 0.2501).all())
    ok &= bool((hip_err - 0.25).abs().min() < 1e-4)

    # 0.5 * 10 rad/s = 5 rad/s wheel velocity target; 4.0 -> 40 rad/s
    probe[:, :4] = 4.0
    probe[:, 4:] = 0.5
    env.step(probe)
    wheel_target = joint_action.wheel_targets
    print(
        f"[check] wheel probe: target range "
        f"{wheel_target.min().item():+.3f}..{wheel_target.max().item():+.3f} rad/s "
        f"(expected +5.0)"
    )
    ok &= bool((wheel_target - 5.0).abs().max() < 1e-4)
    ok &= bool((joint_action.hip_torques.abs() <= HIP_EFFORT_LIMIT + 1e-5).all())
    ok &= bool((joint_action.wheel_torques.abs() <= 3.69 + 1e-5).all())
    applied_hip = robot.data.applied_torque[:, joint_action.hip_ids].abs().max().item()
    print(f"[check] squat_torque applied hip torque (sim): max |tau|={applied_hip:.4f} Nm")
    ok &= applied_hip > 1.0
    ok &= env.max_closure_residual < 5e-3

    print("[check] RESULT:", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    import importlib.metadata as metadata

    task = normalize_task(args_cli.task)
    agent_cfg = agent_lib.apply_cli_overrides(agent_lib.make_agent_cfg(task, play=False), args_cli)
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

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    env_cfg = build_env_cfg(task, args_cli.num_envs, args_cli.device or "cuda:0", agent_cfg.seed, log_dir)
    env = make_env(task, env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.check:
        return 0 if run_env_checks(env, task) else 1

    resume_path: str | None = None
    if agent_cfg.resume:
        checkpoint_arg = agent_cfg.load_checkpoint
        looks_like_path = bool(checkpoint_arg) and ("/" in checkpoint_arg or "\\" in checkpoint_arg)
        if looks_like_path:
            direct = os.path.abspath(checkpoint_arg)
            if not os.path.isfile(direct):
                raise FileNotFoundError(f"--checkpoint '{checkpoint_arg}' does not exist: {direct}")
            resume_path = direct
            print(f"[INFO] loading checkpoint file directly: {resume_path}")
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
    except Exception as exc:  # noqa: BLE001
        print(f"[INFO] could not record the git state: {exc}")

    if resume_path is not None:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        if args_cli.reset_obs_norm_count is not None:
            agent_lib.reset_obs_normalizer_count(runner, args_cli.reset_obs_norm_count)

    agent_lib.clamp_policy_std(runner)
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
        os._exit(exit_code)
