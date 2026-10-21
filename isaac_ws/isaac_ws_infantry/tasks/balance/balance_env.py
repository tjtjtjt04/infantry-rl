"""Balance task environment: the shared base env, unchanged."""

from __future__ import annotations

from isaac_ws_infantry.tasks.common.five_bar_env import FiveBarEnv


class FiveBarBalanceEnv(FiveBarEnv):
    """Balance task env (identical behaviour to :class:`FiveBarEnv`)."""


__all__ = ["FiveBarBalanceEnv"]
