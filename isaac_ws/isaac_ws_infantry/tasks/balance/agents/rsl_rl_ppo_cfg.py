"""RSL-RL agent configuration for the infantry-rl balance task.

rsl-rl 5.0.1 requires ``actor``/``critic`` (:class:`RslRlMLPModelCfg`) and takes the
exploration noise from ``distribution_cfg``; the deprecated ``policy`` /
``init_noise_std`` path is not used here.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class InfantryBalancePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "infantry_rl_balance"
    run_name = ""
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    clip_actions = 4.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
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
class InfantryBalancePPORunnerCfg_PLAY(InfantryBalancePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 0
        self.num_steps_per_env = 24


__all__ = ["InfantryBalancePPORunnerCfg", "InfantryBalancePPORunnerCfg_PLAY"]
