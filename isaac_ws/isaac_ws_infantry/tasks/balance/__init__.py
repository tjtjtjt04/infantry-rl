"""Gym registrations for the infantry-rl balance task."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Infantry-RL-Balance-v0",
    entry_point=f"{__name__}.balance_env:FiveBarBalanceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg:BalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InfantryBalancePPORunnerCfg",
    },
)

gym.register(
    id="Infantry-RL-Balance-Play-v0",
    entry_point=f"{__name__}.balance_env:FiveBarBalanceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg:BalanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InfantryBalancePPORunnerCfg_PLAY",
    },
)
