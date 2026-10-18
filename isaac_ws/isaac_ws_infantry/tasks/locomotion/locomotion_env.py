"""Environment class for the locomotion task.

The only extra duty over :class:`ManagerBasedRLEnv` is the five-bar loop closure,
which is identical to the balance task, so the locomotion environment simply reuses
:class:`~isaac_ws_infantry.tasks.balance.balance_env.FiveBarBalanceEnv`.  It is
re-exported (and subclassed for the type hint) here so the gym registration reads
naturally.
"""

from __future__ import annotations

from ..balance.balance_env import FiveBarBalanceEnv
from .locomotion_env_cfg import LocomotionEnvCfg


class FiveBarLocomotionEnv(FiveBarBalanceEnv):
    """Five-bar wheel-leg robot with velocity commands and loop-closure control."""

    cfg: LocomotionEnvCfg


__all__ = ["FiveBarLocomotionEnv"]
