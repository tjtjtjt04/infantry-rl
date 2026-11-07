"""起立任务的 PPO 配置（rsl-rl-lib 5.0.1 新式 actor/critic API）。"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

__all__ = ["ChassisStandUpPPORunnerCfg", "ChassisStandUpPPORunnerCfg_PLAY"]


@configclass
class ChassisStandUpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """起立任务的 PPO 训练配置。"""

    # 每 env 每次迭代采样 48 步（默认 24）：起立需要连续多步同向力矩，
    # 加长视野让 PPO 能把"推起来"的长期收益归因到动作上。
    num_steps_per_env = 48
    max_iterations = 1500
    save_interval = 50
    experiment_name = "chassis_rl_stand_up"
    #: 训练随机种子（默认 17；之前 seed=42 时 1024 envs 会在 iter 169 触发可复现的驱动崩溃）
    seed = 17

    empirical_normalization = False
    # 动作 6 维纯力矩：髋 ×5 Nm/unit（夹 ±40）、轮 ×1 Nm/unit（夹 ±4.92）。
    # 训练器按 ±8 裁剪原始动作，具体力矩上限由动作项再夹一次。
    clip_actions = 8.0
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 64],
        activation="elu",
        obs_normalization=True,
        # 初始探索 std=1.8 action unit：髋 ±9 Nm、轮 ±1.8 Nm
        # （起立需要 ~3 Nm 轮力矩 + ~4 Nm 髋力矩，探索要够大）。
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.8),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ChassisStandUpPPORunnerCfg_PLAY(ChassisStandUpPPORunnerCfg):
    """play/评估用配置：只看不训。"""

    max_iterations = 10
    save_interval = 1000
