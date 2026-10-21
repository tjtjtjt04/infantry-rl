"""Environment class for the locomotion task.

The only extra duty over :class:`ManagerBasedRLEnv` is the five-bar loop closure,
which is shared by every task, so the locomotion environment simply reuses
:class:`~isaac_ws_infantry.tasks.common.five_bar_env.FiveBarEnv`.  It is
re-exported (and subclassed for the type hint) here so the gym registration reads
naturally.
"""

from __future__ import annotations

from isaac_ws_infantry.tasks.common.five_bar_env import FiveBarEnv
from .locomotion_env_cfg import LocomotionEnvCfg


class FiveBarLocomotionEnv(FiveBarEnv):
    """Five-bar wheel-leg robot with velocity commands and loop-closure control."""

    cfg: LocomotionEnvCfg


__all__ = ["FiveBarLocomotionEnv"]
