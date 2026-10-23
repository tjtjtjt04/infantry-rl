"""Command terms: velocity commands that can also ask for an in-place spin.

WHY THIS FILE EXISTS
--------------------
``UniformVelocityCommand`` samples ``vx`` uniformly inside one range, so with a range like
``[-1.2, +1.2]`` the probability of commanding "vx = 0 and turn" -- i.e. a pure in-place
rotation -- is only a few percent.  A policy trained like that learns to drive and to turn
while driving, but never gets enough "spin on the spot" samples to learn that skill.

:class:`SpinAwareVelocityCommand` adds one extra rule on top of the stock term: for a
fraction ``rel_spin_envs`` of the environments the linear command is forced to ZERO while
the angular command keeps whatever the heading loop asks for.  Those environments therefore
receive exactly a "turn to the target heading without translating" command, which is the
in-place rotation skill.

Everything else is unchanged:

* ``rel_standing_envs`` still zeroes the WHOLE command (the braking / stand-still sample),
* ``heading_command=True`` still turns the heading error into a yaw-rate command,
* with ``rel_spin_envs = 0.0`` (the default) the behaviour is bit-for-bit the stock term,
  so the other training modes are unaffected.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass


@configclass
class SpinAwareVelocityCommandCfg(UniformVelocityCommandCfg):
    """``UniformVelocityCommandCfg`` plus an in-place-spin sample fraction."""

    rel_spin_envs: float = 0.0
    """Fraction of environments that get ``vx = 0`` and a heading target (in-place spin).

    They still receive the heading-loop yaw command, so the robot is asked to rotate on the
    spot towards a new heading.  Must be 0 when ``heading_command=False`` (there is no
    heading loop then, so "spin" would be meaningless).
    """


class SpinAwareVelocityCommand(UniformVelocityCommand):
    """Velocity command with a controllable share of pure in-place rotations."""

    cfg: SpinAwareVelocityCommandCfg

    def __init__(self, cfg: SpinAwareVelocityCommandCfg, env):
        super().__init__(cfg, env)
        # 1.0 = "normal env" (keep the sampled vx), 0.0 = "spin env" (force vx = 0)
        self._spin_mask = torch.ones(self.num_envs, device=self.device)
        self.spin_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        r = torch.empty(len(env_ids), device=self.device)
        spin = r.uniform_(0.0, 1.0) <= self.cfg.rel_spin_envs
        self.spin_env_mask[env_ids] = spin
        self._spin_mask[env_ids] = torch.where(spin, 0.0, 1.0).float()

    def _update_command(self):
        super()._update_command()
        # Applied AFTER the base class so that (a) the heading loop has already produced the
        # yaw command and (b) the standing (brake) envs stay fully zeroed.
        self.vel_command_b[:, 0] *= self._spin_mask
        self.vel_command_b[:, 1] *= self._spin_mask

    def __str__(self) -> str:
        msg = super().__str__()
        msg += f"\n\tSpin probability: {self.cfg.rel_spin_envs}"
        return msg


__all__ = ["SpinAwareVelocityCommand", "SpinAwareVelocityCommandCfg"]
