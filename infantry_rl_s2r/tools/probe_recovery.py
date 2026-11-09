#!/usr/bin/env python
"""How much pitch can a policy recover from? (needs Kit)

Places every environment at a chosen root pitch on the measured contact height,
runs a policy for a fixed time and reports whether it came back upright.  This
is the experiment that decides how the fall-recovery task has to be built:

* if the balance checkpoint already recovers from the tripod, relaxing its pitch
  termination may be enough;
* if it does not, the recovery task needs its own reward / resets.

Examples::

    C:/Users/32560/.conda/envs/isaaclab/python.exe tools/probe_recovery.py --headless
    C:/Users/32560/.conda/envs/isaaclab/python.exe tools/probe_recovery.py --headless \
        --checkpoint logs/.../model_2596.pt --pitches -5 0 5 10 15 17 20 25 30 35 --steps 300
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

os.environ.setdefault("INFANTRY_RL_DR", "0")

from isaaclab.app import AppLauncher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe closed-loop recovery from a pitch angle.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="policy checkpoint; default: the newest one under logs/rsl_rl")
    parser.add_argument("--random_policy", action="store_true", help="no checkpoint; random init policy")
    parser.add_argument("--zero_action", action="store_true", help="skip the policy; send zero actions")
    parser.add_argument("--constant_action", type=float, nargs=6, default=None,
                        help="diagnostic: apply a constant 6-D raw action every step")
    parser.add_argument("--rock_amp", type=float, default=0.0,
                        help="diagnostic: sinusoidal wheel-velocity action of this amplitude")
    parser.add_argument("--rock_freqs", type=float, nargs="+", default=[1.0],
                        help="rocking frequencies (Hz), one per env group")
    parser.add_argument("--task", type=str, default="squat_torque",
                        choices=("squat_torque", "getup_torque"),
                        help="which task's env to build (its domain randomization is used)")
    parser.add_argument("--pitches", type=float, nargs="+",
                        default=[-5.0, 0.0, 5.0, 10.0, 13.0, 15.0, 17.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    parser.add_argument("--steps", type=int, default=300, help="control steps per pitch (100 Hz)")
    parser.add_argument("--num_envs", type=int, default=None, help="defaults to len(--pitches)")
    parser.add_argument("--height_offset", type=float, default=0.001)
    parser.add_argument("--trace_every", type=int, default=0,
                        help="print a per-step state trace for env 0 every N steps (0 = off)")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows (imports require Kit)."""

import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

import agent as agent_lib  # noqa: E402
from config import DEFAULT_LOG_ROOT  # noqa: E402
from env_getup import FiveBarGetupEnv, GetupEnvCfg  # noqa: E402
from env_squat import SQUAT_HIP_OFFSETS, FiveBarSquatEnv, SquatEnvCfg  # noqa: E402
from robot_model import fallen_geometry  # noqa: E402


def build_env(num_envs: int):
    """Build the task's env so the domain randomization matches it.

    ``squat_torque`` also randomizes interval pushes (0.3 m/s every 5-8 s), while
    ``getup_torque`` has pushes off (they hit mid-manoeuvre).  Pick the task you are
    evaluating, otherwise the push bias shows up as a lower recovery rate.
    """
    if args_cli.task == "getup_torque":
        cfg = GetupEnvCfg()
        env_cls = FiveBarGetupEnv
    else:
        cfg = SquatEnvCfg()
        env_cls = FiveBarSquatEnv
    cfg.scene.num_envs = num_envs
    cfg.seed = 0
    cfg.episode_length_s = 1.0e6
    return env_cls(cfg=cfg)


def main() -> int:
    pitches = [float(p) for p in args_cli.pitches]
    rocking = args_cli.rock_amp > 0.0
    if rocking:
        env_pitches = [p for _f in args_cli.rock_freqs for p in pitches]
        env_freqs = [f for f in args_cli.rock_freqs for _p in pitches]
    else:
        env_pitches = pitches
        env_freqs = [0.0] * len(pitches)
    num_envs = int(args_cli.num_envs or len(env_pitches))
    env = build_env(num_envs)
    robot = env.scene["robot"]
    term = env.action_manager.get_term(env.cfg.joint_action_name)
    # keep every episode alive: we are measuring the controller, not the task limits
    env.termination_manager.compute = lambda: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.reset()

    agent_cfg = agent_lib.make_agent_cfg("squat_torque", play=True)
    import importlib.metadata as metadata

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---- policy ---------------------------------------------------------
    if rocking:
        freq_t = torch.tensor(env_freqs, dtype=torch.float32, device=env.device)
        state = {"k": 0}

        def policy(_obs):
            t = state["k"] * env.step_dt
            state["k"] += 1
            w = (args_cli.rock_amp * torch.sin(2.0 * math.pi * freq_t * t)).view(-1, 1)
            return torch.cat((torch.zeros(num_envs, 4, device=env.device), w, w), dim=-1)

        label = f"rocking +-{args_cli.rock_amp:g} at {env_freqs} Hz"
    elif args_cli.constant_action is not None:
        const = torch.tensor(args_cli.constant_action, dtype=torch.float32, device=env.device)
        policy = lambda _obs: const.unsqueeze(0).expand(num_envs, -1)
        label = "constant action " + str([round(float(v), 3) for v in const])
    elif args_cli.zero_action:
        policy = lambda _obs: torch.zeros((num_envs, env.action_space.shape[-1]), device=env.device)
        label = "zero action"
    elif args_cli.random_policy:
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        policy = runner.get_inference_policy(device=env.device)
        label = "random policy"
    else:
        ckpt = args_cli.checkpoint
        if ckpt is None:
            ckpt = agent_lib.newest_checkpoint(str(DEFAULT_LOG_ROOT / "infantry_rl_squat_torque"), None)
        agent_lib.match_network_to_checkpoint(ckpt, agent_cfg)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=env.device)
        label = f"policy {Path(ckpt).name}"

    geometry = fallen_geometry()
    print(f"\n[probe] controller: {label}", flush=True)
    print(f"[probe] {num_envs} envs, {args_cli.steps} control steps per episode\n", flush=True)

    pitch_rad = torch.tensor([math.radians(p) for p in env_pitches], device=env.device)
    z = geometry["contact_height"](pitch_rad) + args_cli.height_offset

    # nominal_hips is already batched as (num_envs, 4); reset_knees is the (4,) closed pair
    hips = term.nominal_hips + torch.tensor([SQUAT_HIP_OFFSETS], device=env.device)
    knees = term.reset_knees.unsqueeze(0).expand(num_envs, -1)
    joint_pos = torch.cat((hips, knees), dim=-1)
    joint_ids = term.hip_ids + term.knee_ids
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), joint_ids=joint_ids)

    quat = math_utils.quat_from_euler_xyz(
        torch.zeros(num_envs, device=env.device), pitch_rad, torch.zeros(num_envs, device=env.device)
    )
    xyz = torch.zeros(num_envs, 3, device=env.device) + env.scene.env_origins
    xyz[:, 2] = z
    robot.write_root_pose_to_sim(torch.cat((xyz, quat), dim=-1))
    robot.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=env.device))
    env.scene.update(dt=env.physics_dt)
    # drop the cached action history and the 5-frame observation history so the policy
    # sees the fallen frame, not the previous (standing) episode
    env.action_manager.reset(torch.arange(num_envs, device=env.device))
    obs = wrapped.get_observations()

    start_pitch = pitch_rad.clone()
    best = start_pitch.abs().clone()
    trace = []
    for step in range(int(args_cli.steps)):
        actions = policy(obs)
        obs, _reward, _dones, _extras = wrapped.step(actions)
        roll, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
        best = torch.minimum(best, pitch.abs())
        if args_cli.trace_every and step % args_cli.trace_every == 0:
            contact = env.scene["contact_forces"].data.net_forces_w[0].norm().item()
            wheel_v = robot.data.joint_vel[0, term.wheel_ids]
            hip_act = actions[0, :4]
            hip_dev = (robot.data.joint_pos[0, term.hip_ids] - term.locked_hips).abs().max()
            hip_tau = term.hip_torques[0].abs().max()
            hip_tgt = (term.hip_targets[0] - term.locked_hips).abs().max()
            print(
                f"  t={step * env.step_dt:4.2f} pitch={math.degrees(float(pitch[0])):+6.2f} "
                f"roll={math.degrees(float(roll[0])):+5.1f} z={float(robot.data.root_pos_w[0, 2]):.3f} "
                f"vx={float(robot.data.root_lin_vel_b[0, 0]):+5.2f} "
                f"wz={float(robot.data.root_ang_vel_b[0, 2]):+5.2f} "
                f"wheel={float(wheel_v[0]):+6.1f}/{float(wheel_v[1]):+6.1f} "
                f"tau={float(robot.data.applied_torque[0, term.wheel_ids[0]]):+5.2f} "
                f"baseF={contact:5.1f} "
                f"hip_act=[{float(hip_act[0]):+5.2f},{float(hip_act[1]):+5.2f},"
                f"{float(hip_act[2]):+5.2f},{float(hip_act[3]):+5.2f}] "
                f"hip_dev={float(hip_dev):.3f} hip_tgt_dev={float(hip_tgt):.3f} "
                f"hip_tau={float(hip_tau):5.2f}",
                flush=True,
            )
        if step % max(1, args_cli.steps // 6) == 0 or step == args_cli.steps - 1:
            trace.append((step, pitch.clone(), roll.clone()))

    roll, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)
    height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    rate = robot.data.root_ang_vel_b.norm(dim=-1)
    wheels = robot.data.joint_vel[:, term.wheel_ids].abs().mean(dim=-1)
    print(
        f"{'pitch0':>7} {'freq':>5} {'pitch_end':>9} {'min|p|':>7} {'base_z':>8} {'|w|':>6} {'wheel':>7}  result",
        flush=True,
    )
    results = {}
    for i in range(num_envs):
        p0 = float(env_pitches[i])
        pe = math.degrees(float(pitch[i]))
        ok = abs(pe) < 10.0 and abs(math.degrees(float(roll[i]))) < 20.0
        key = (round(p0, 1), round(float(env_freqs[i]), 2))
        results.setdefault(key, []).append(ok)
        print(
            f"{p0:>7.1f} {float(env_freqs[i]):>5.2f} {pe:>9.1f} {math.degrees(float(best[i])):>7.1f} "
            f"{float(height[i]):>8.4f} {float(rate[i]):>6.2f} {float(wheels[i]):>7.2f}  "
            f"{'UPRIGHT' if ok else 'down'}",
            flush=True,
        )

    print("\n[summary] recovery rate per (initial pitch, rock freq):", flush=True)
    for key in sorted(results):
        flags = results[key]
        print(
            f"  {key[0]:>6.1f} deg @ {key[1]:>4.2f} Hz : "
            f"{100.0 * sum(flags) / len(flags):5.1f}% upright",
            flush=True,
        )

    print("\n[history] pitch (deg) samples:", flush=True)
    for step, p, r in trace:
        print(
            f"  t={step * env.step_dt:5.2f}s  pitch min/mean/max = "
            f"{math.degrees(float(p.min())):+6.1f}/{math.degrees(float(p.mean())):+6.1f}/"
            f"{math.degrees(float(p.max())):+6.1f}",
            flush=True,
        )

    env.close()
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
