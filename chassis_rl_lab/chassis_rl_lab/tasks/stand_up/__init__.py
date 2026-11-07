"""chassis-rl 两阶段任务的 gym 注册。

- ``Chassis-Stand-Up-v0``（**阶段1**）：从倒地开始 → 起立 → 保持平衡；高度指令固定名义值，
  不跟踪高度（髋/轮 6 维纯力矩，25 维 Fudan 式观测）；
- ``Chassis-Stand-Up-Height-v0``（**阶段2**）：在阶段1模型上 ``--resume`` 继续训练，
  打开"按高度指令改变机体高度"（观测/动作维度与阶段1 完全一致）。
"""

import gymnasium as gym

from .stand_up_env_cfg import (
    ChassisStandUpEnvCfg,
    ChassisStandUpEnvCfg_PLAY,
    ChassisStandUpHeightEnvCfg,
    ChassisStandUpRecoveryEnvCfg,
)

__all__ = [
    "ChassisStandUpEnvCfg",
    "ChassisStandUpEnvCfg_PLAY",
    "ChassisStandUpHeightEnvCfg",
    "ChassisStandUpRecoveryEnvCfg",
]

# 阶段1：倒地起立 + 平衡
gym.register(
    id="Chassis-Stand-Up-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "chassis_rl_lab.tasks.stand_up.stand_up_env_cfg:ChassisStandUpEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "chassis_rl_lab.tasks.stand_up.agents.rsl_rl_ppo_cfg:ChassisStandUpPPORunnerCfg"
        ),
    },
)

# 阶段2：在阶段1基础上 + 按指令改变高度（同观测/动作维度，可 resume 阶段1）
gym.register(
    id="Chassis-Stand-Up-Height-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "chassis_rl_lab.tasks.stand_up.stand_up_env_cfg:ChassisStandUpHeightEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "chassis_rl_lab.tasks.stand_up.agents.rsl_rl_ppo_cfg:ChassisStandUpPPORunnerCfg"
        ),
    },
)

# 倒地自起（随机方向）+ 平衡；髋自由但不跟踪高度/腿长（从阶段1继续训练用）
gym.register(
    id="Chassis-Stand-Up-Recovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "chassis_rl_lab.tasks.stand_up.stand_up_env_cfg:ChassisStandUpRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "chassis_rl_lab.tasks.stand_up.agents.rsl_rl_ppo_cfg:ChassisStandUpPPORunnerCfg"
        ),
    },
)