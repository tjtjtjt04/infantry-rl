"""Gym registrations for the infantry-rl locomotion task."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Infantry-RL-Locomotion-v0",
    entry_point=f"{__name__}.locomotion_env:FiveBarLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:LocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InfantryLocomotionPPORunnerCfg",
    },
)

gym.register(
    id="Infantry-RL-Locomotion-Play-v0",
    entry_point=f"{__name__}.locomotion_env:FiveBarLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_env_cfg:LocomotionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InfantryLocomotionPPORunnerCfg_PLAY",
    },
)
