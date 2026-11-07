"""环境冒烟测试：obs/action 维度、奖励有限性、终止统计、蹲姿高度、五连杆闭链残差。

用法（本机 conda 环境 RM，headless）::

    D:\\miniconda\\envs\\RM\\python.exe tools\\check_env.py --num_envs 16 --steps 200
    D:\\miniconda\\envs\\RM\\python.exe tools\\check_env.py --num_envs 16 --steps 200 --random

零动作（默认）时应能保持 squat 姿态平衡若干秒；``--random`` 用来检查维度/数值稳定性。
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="chassis-rl stand-up env smoke test")
parser.add_argument("--task", type=str, default="Chassis-Stand-Up-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--random", action="store_true", help="use random actions instead of zero actions")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import chassis_rl_lab.tasks  # noqa: E402,F401  (注册 gym 任务；不依赖包根 __init__)
from chassis_rl_lab.kinematics import load_squat_pose  # noqa: E402
from chassis_rl_lab.tasks.common.mdp.sensor_noise import (  # noqa: E402
    BMI088_ACC_NOISE_X,
    BMI088_ACC_NOISE_Y,
    BMI088_ACC_NOISE_Z,
    BMI088_GYRO_NOISE_X,
    BMI088_GYRO_NOISE_Y,
    BMI088_GYRO_NOISE_Z,
    GRAVITY_MPS2,
)


def load_cfg_from_registry(task: str, entry_point_key: str):
    """从 gym registry 的 ``"module:ClassName"`` 入口加载 env cfg（支持多任务）。"""
    import importlib

    spec = gym.spec(task)
    entry_point = spec.kwargs.get(entry_point_key)
    if entry_point is None:
        raise ValueError(f"任务 {task!r} 的 registry kwargs 缺少 {entry_point_key!r}")
    if callable(entry_point):
        return entry_point()
    module_name, class_name = entry_point.split(":")
    return getattr(importlib.import_module(module_name), class_name)()


def _mean_tail(values: list[float], n: int = 50) -> float:
    return statistics.mean(values[-min(n, len(values)) :]) if values else float("nan")


def _policy_obs(obs_buf):
    """环境返回的 obs 是 {组名: tensor} 的 dict（concatenate_terms=True 时组内已拼接）。"""
    if isinstance(obs_buf, dict):
        return obs_buf.get("policy", next(iter(obs_buf.values())))
    return obs_buf


def main() -> int:
    squat = load_squat_pose()
    base_target = float(squat["base_height"])

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    device = args_cli.device if getattr(args_cli, "device", None) is not None else "cuda:0"
    env_cfg.sim.device = device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = env.unwrapped
    obs_buf, _info = env.reset()
    obs = _policy_obs(obs_buf)
    num_envs = env.num_envs
    action_dim = env.action_space.shape[-1]
    print(f"\n[check] task={args_cli.task} num_envs={num_envs} action_dim={action_dim}")
    print(f"[check] observation shape={tuple(obs.shape)}")
    print(f"[check] episode_length_s={env.max_episode_length_s} max_episode_length={env.max_episode_length}")

    act_term = env.action_manager.get_term("wheel_leg")
    print(
        f"[check] 动作项: {type(act_term).__name__}, use_hip_actions={act_term.cfg.use_hip_actions}, "
        f"hip_control={act_term.cfg.hip_control}, lock_hips={act_term.cfg.lock_hips}, "
        f"wheel_control={act_term.cfg.wheel_control}, "
        f"wheel_unpowered_steps={act_term.cfg.wheel_unpowered_steps}"
    )
    # 期望观测维度直接取 observation manager 自己算的 group dim（多任务通用）
    dim = env.observation_manager.group_obs_dim["policy"]
    expected_obs_dim = int(dim) if isinstance(dim, int) else int(sum(dim))
    print(f"[check] 期望 obs 维度: {expected_obs_dim}（last_action={action_dim} 维）")
    robot = env.scene["robot"]
    print(f"[check] robot num_joints={robot.num_joints} joint_names={robot.joint_names}")

    # ---- 观测噪声检测（BMI088 IMU）：同一状态连续 compute 两次 ----
    # 有噪声的项两次不同（d = n1 - n2，σ_d = √2·σ）；无噪声的项 d 恒为 0。
    # 25 维布局（Fudan 式）：[0:3] base_ang_vel（陀螺仪，scale 0.25）、[3:6] projected_gravity（加速度计）。
    obs_a = _policy_obs(env.observation_manager.compute()).clone()
    obs_b = _policy_obs(env.observation_manager.compute()).clone()
    d = (obs_a - obs_b).float()
    acc_exp = [BMI088_ACC_NOISE_X / GRAVITY_MPS2, BMI088_ACC_NOISE_Y / GRAVITY_MPS2, BMI088_ACC_NOISE_Z / GRAVITY_MPS2]
    gyro_exp = [BMI088_GYRO_NOISE_X * 0.25, BMI088_GYRO_NOISE_Y * 0.25, BMI088_GYRO_NOISE_Z * 0.25]  # base_ang_vel scale=0.25
    gyro_meas = [math.sqrt(max(float((d[:, i] ** 2).mean().item()), 0.0) / 2.0) for i in range(3)]
    acc_meas = [math.sqrt(max(float((d[:, 3 + i] ** 2).mean().item()), 0.0) / 2.0) for i in range(3)]
    other_max = float(d[:, 6:].abs().max().item()) if d.shape[1] > 6 else 0.0
    print("[check] ====== 观测噪声（BMI088） ======")
    print(f"  projected_gravity σ_obs = [{acc_meas[0]:.5f}, {acc_meas[1]:.5f}, {acc_meas[2]:.5f}] g"
          f"   期望 [{acc_exp[0]:.5f}, {acc_exp[1]:.5f}, {acc_exp[2]:.5f}]")
    print(f"  base_ang_vel      σ_obs = [{gyro_meas[0]:.5f}, {gyro_meas[1]:.5f}, {gyro_meas[2]:.5f}]"
          f"   期望 [{gyro_exp[0]:.5f}, {gyro_exp[1]:.5f}, {gyro_exp[2]:.5f}] (rad/s × 0.25)")
    print(f"  其它观测项最大噪声差     = {other_max:.3e} (应为 0，说明只有 IMU 两项加噪声)")

    rewards: list[float] = []
    base_heights: list[float] = []
    closures: list[float] = []
    term_counts: dict[str, int] = {}
    term_total = 0
    trunc_total = 0
    reward_log: dict[str, float] = {}

    torch.manual_seed(args_cli.seed)
    for step in range(args_cli.steps):
        if args_cli.random:
            actions = 2.0 * torch.rand(num_envs, action_dim, device=env.device) - 1.0
        else:
            actions = torch.zeros(num_envs, action_dim, device=env.device)
        obs_buf, rew, terminated, truncated, extras = env.step(actions)
        obs = _policy_obs(obs_buf)
        if not torch.isfinite(obs).all():
            raise RuntimeError(f"obs 出现 NaN/Inf @ step {step}")
        if not torch.isfinite(rew).all():
            raise RuntimeError(f"reward 出现 NaN/Inf @ step {step}")
        rewards.append(float(rew.mean().item()))
        base_heights.append(float(robot.data.root_pos_w[:, 2].mean().item()))
        closures.append(float(act_term.closure_residual.mean().item()))
        term_total += int(terminated.sum().item())
        trunc_total += int(truncated.sum().item())
        log = extras.get("log", {})
        if log:
            reward_log = {k: float(v) for k, v in log.items() if isinstance(v, (int, float, torch.Tensor))}
        if "episode" in extras:
            for key, value in extras["episode"].items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    term_counts[key] = term_counts.get(key, 0) + int(value.item())

    print("\n[check] ====== 数值结果（末 50 步均值） ======")
    print(f"  reward/step mean       = {_mean_tail(rewards):+.4f}")
    print(f"  base height            = {_mean_tail(base_heights):.5f} m   (target {base_target:.5f} m)")
    print(f"  五连杆闭链残差          = {_mean_tail(closures) * 1000.0:.3f} mm  (期望 < 5 mm)")
    print(f"  terminated 累计         = {term_total}")
    print(f"  truncated 累计          = {trunc_total}")
    if reward_log:
        print("  extras['log'] 键示例   :", list(reward_log.keys())[:8])
    if term_counts:
        print("  extras['episode']      :", {k: round(v, 3) for k, v in list(term_counts.items())[:8]})

    # ---- 断言 ----
    # 注意：零动作 = 两轮倒立摆无控制，机器人本来就该逐渐倾倒（这正是 RL 要学的），
    # 所以这里只检查结构正确性/数值有限性/闭链残差，不要求"站住"。
    problems: list[str] = []
    if tuple(obs.shape) != (num_envs, expected_obs_dim):
        problems.append(f"obs 维度 {tuple(obs.shape)} != ({num_envs}, {expected_obs_dim})")
    if action_dim not in (2, 6):
        problems.append(f"action_dim {action_dim} not in (2=仅轮子, 6=腿+轮)")
    # 噪声断言：实测 σ 应在期望值的 0.3~3 倍内（16 envs 的统计误差），其它观测应为 0
    for name, meas, exp in (("projected_gravity", acc_meas, acc_exp), ("base_ang_vel", gyro_meas, gyro_exp)):
        for axis in range(3):
            if not (0.3 * exp[axis] <= meas[axis] <= 3.0 * exp[axis]):
                problems.append(f"{name} 轴{axis} 噪声 σ={meas[axis]:.5f} 偏离期望 {exp[axis]:.5f}")
    if other_max > 1e-9:
        problems.append(f"非 IMU 观测项出现噪声（max diff={other_max:.3e}），应只有 projected_gravity/base_ang_vel 带噪声")
    mean_base = _mean_tail(base_heights)
    # 蹲姿平衡任务 base 高度 ≈ 0.17 m；起立任务从倒地开始，允许更低
    if not (0.0 <= mean_base <= 0.40):
        problems.append(f"base height 越界（0.0~0.40 m）: {mean_base:.4f} m  (squat 目标 {base_target:.4f})")
    if _mean_tail(closures) > 0.005:
        problems.append(f"闭链残差 > 5 mm: {_mean_tail(closures) * 1000:.2f} mm")
    # 零动作仿真只校验奖励数值有限（起立任务从倒地开始，奖励本来就可能是负的）
    if not math.isfinite(_mean_tail(rewards)):
        problems.append("平均奖励不是有限值")
    gravity_z = float(robot.data.projected_gravity_b[:, 2].mean().item())
    print(f"  projected_gravity_z 末值 = {gravity_z:+.4f}（-1=竖直；越接近 0 越倾斜）")

    print("\n[check] ====== 结论 ======")
    if problems:
        for p in problems:
            print(f"  [FAIL] {p}")
        return 1
    print("  [OK] 环境 smoke 测试通过")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        # 注意：Windows 上 Isaac Sim 的 close() 偶发挂起（会卡住退出），
        # 所以这里只 flush 后直接 os._exit；Kit 资源由进程退出回收。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
