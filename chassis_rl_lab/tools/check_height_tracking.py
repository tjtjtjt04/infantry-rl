"""检查 Height 任务：给不同目标高度，验证策略是否真的改变机体高度/腿长。

用法（无渲染，几秒跑完）::

    python tools/check_height_tracking.py --checkpoint <model_XXXX.pt>

给 3 个并行 env 分别下发 0.172 / 0.181 / 0.190 m 的固定高度指令（关闭 curriculum），
统计站起来稳定后的 base 高度与髋关节角度，打印对比表。
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Height 任务高度跟踪检查")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint 路径（默认最新 run 的最新）")
parser.add_argument("--num_envs", type=int, default=3, help="并行 env 数（每个 env 一个目标高度）")
parser.add_argument("--episode_steps", type=int, default=1000, help="每个 episode 步数")
parser.add_argument("--skip_steps", type=int, default=400, help="忽略起立阶段，只统计之后的稳态")
args = parser.parse_args()
args.headless = True
app = AppLauncher(args)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import chassis_rl_lab.tasks  # noqa: F401,E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

TASK = "Chassis-Stand-Up-Height-v0"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
env_cfg.scene.num_envs = int(args.num_envs)
env_cfg.sim.device = "cuda:0"
# 关闭 curriculum：确保指令就是下面给的固定目标（不是名义高度）
env_cfg.events.reset_height_command.params["curriculum_steps"] = 0

env = gym.make(TASK, cfg=env_cfg)
raw = env.unwrapped
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# 解析 checkpoint（缺省 = 该实验最新 run 最新 .pt）
if args.checkpoint is not None and Path(args.checkpoint).is_file():
    resume_path = str(Path(args.checkpoint).resolve())
else:
    from scripts.play import resolve_checkpoint

    resume_path = resolve_checkpoint("logs/rsl_rl/" + agent_cfg.experiment_name, ".*", "model_.*.pt")
resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=env.unwrapped.device)
print(f"[check] checkpoint: {resume_path}")

robot = raw.scene["robot"]
hip_ids, _ = robot.find_joints(
    ["left_front_hip", "left_back_hip", "right_front_hip", "right_back_hip"], preserve_order=True
)

from chassis_rl_lab.tasks.stand_up.stand_up_env_cfg import HEIGHT_COMMAND_RANGE  # noqa: E402

h_lo, h_hi = float(HEIGHT_COMMAND_RANGE[0]), float(HEIGHT_COMMAND_RANGE[1])
targets = torch.linspace(h_lo, h_hi, int(args.num_envs), device=raw.device)
raw.height_command[:] = targets.unsqueeze(1)
num = int(args.num_envs)

z_hist = [[] for _ in range(num)]
hip_hist = [[] for _ in range(num)]
obs = env.get_observations()
with torch.inference_mode():
    for step in range(int(args.episode_steps)):
        obs, _rew, dones, _extras = env.step(policy(obs))
        # reset 后重新写固定目标（否则 event 会重新随机）
        for i in range(num):
            if bool(dones[i]):
                raw.height_command[i, 0] = targets[i]
        if step >= int(args.skip_steps):
            for i in range(num):
                z_hist[i].append(float(robot.data.root_pos_w[i, 2]))
                hip_hist[i].append(float(robot.data.joint_pos[i, hip_ids[0]]))

print("=" * 92)
print(f"{'目标高度 h_cmd':>14} | {'实测 base_z(均值)':>18} | {'误差':>9} | {'髋角(前左)均值':>16} | {'base_z 波动':>14}")
print("-" * 92)
for i in range(num):
    if not z_hist[i]:
        print(f"  env{i}: 没有采样（步数太少）")
        continue
    cmd = float(targets[i])
    z_mean = sum(z_hist[i]) / len(z_hist[i])
    z_min, z_max = min(z_hist[i]), max(z_hist[i])
    hip_mean = sum(hip_hist[i]) / len(hip_hist[i])
    print(
        f"{cmd:>14.4f} | {z_mean:>18.4f} | {z_mean - cmd:>+9.4f} | {hip_mean:>16.4f} | {z_max - z_min:>14.4f}"
    )
z_means = [sum(h) / len(h) for h in z_hist if h]
if len(z_means) >= 2:
    print("-" * 92)
    print(f"不同目标高度下 base_z 均值差异（max-min）= {max(z_means) - min(z_means):.4f} m")
print("=" * 92)
print("done")
import os

os._exit(0)
