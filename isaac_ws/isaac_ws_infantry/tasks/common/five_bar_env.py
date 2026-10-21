"""Base environment shared by the infantry-rl tasks: keeps the five-bar loop closed.

PhysX drops the closed chain (the articulation has 10 joints), so the passive knee
targets are re-solved from the measured hip angles on every control step.  Every
task subclasses :class:`FiveBarEnv` and only adds its own commands/observations/
rewards on top.

The infantry-rl balance environment.

The only thing this class adds on top of :class:`ManagerBasedRLEnv` is the
closed-loop control duty that PhysX does not provide for this robot:

* ``loop_joints/*_calf_axle`` are **not** articulation DOFs (verified: the USD
  imports as a 10-joint articulation), so the five-bar loop has to be re-imposed
  every control step by writing the four knee position targets that the verified
  planar solver returns for the *measured* hip angles, warm-started from the
  previous step's solution;
* the resulting closure error ``|W - E|`` (wheel axle vs. loop anchor on the back
  calf), the solver residual and the solved leg length/angle are exposed for
  rewards, logging and the smoke test.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import ManagerBasedRLEnvCfg




class FiveBarEnv(ManagerBasedRLEnv):
    """Manager-based RL environment with control-level five-bar loop closure."""

    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._joint_action_name: str = getattr(cfg, "joint_action_name", "joints")
        self._joint_action = self.action_manager.get_term(self._joint_action_name)
        self._robot = self.scene["robot"]

        #: measured loop error ``|W - E|`` per side (metres), ``(N, 2)`` = (left, right)
        self.closure_residual = torch.zeros(self.num_envs, 2, device=self.device)
        #: solver residual of the same quantity (metres), ``(N, 2)``
        self.solver_residual = torch.zeros(self.num_envs, 2, device=self.device)
        #: number of knee targets that had to be clipped to the URDF limits
        self.knee_target_clamped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # make sure the very first physics step already sees closed knees
        self._write_knee_targets()

    # ------------------------------------------------------------------ #
    # properties
    # ------------------------------------------------------------------ #
    @property
    def joint_action(self):
        """The :class:`~isaac_ws_infantry.mdp.actions.FiveBarJointAction` term."""
        return self._joint_action

    @property
    def max_closure_residual(self) -> float:
        """Largest measured loop error over all environments and both legs (m)."""
        return float(self.closure_residual.max().item())

    # ------------------------------------------------------------------ #
    # loop closure
    # ------------------------------------------------------------------ #
    def _write_knee_targets(self) -> None:
        self._robot.set_joint_position_target(self._joint_action.knee_targets, joint_ids=self._joint_action.knee_ids)

    def step(self, action: torch.Tensor):
        """Recompute the knee targets, then run the standard environment step."""
        # 1) close the loop for the *measured* hip configuration of this control step
        self._joint_action.update_knee_targets()
        # 2) push the solved knee targets into the articulation buffers (they are
        #    re-written to PhysX on every physics sub-step inside super().step())
        self._write_knee_targets()

        obs, rewards, terminated, truncated, extras = super().step(action)

        # 3) record what the simulation actually did, for logging and verification.
        #
        # PERFORMANCE: everything here stays on the GPU.  In particular:
        #   * ``measured_closure_norm(reuse_cache=True)`` reuses the tensor the
        #     ``closure_residual`` reward already computed during ``super().step()``
        #     instead of recomputing it (that doubled the kernels for a value needed
        #     once);
        #   * ``extras["closure_residual_max"]`` is deliberately NOT filled in: it
        #     called ``float(t.max().item())``, i.e. a CUDA->CPU sync on EVERY control
        #     step.  Consumers that actually want it (play / train summary) call the
        #     ``max_closure_residual`` property themselves, once.
        self.closure_residual = self._joint_action.measured_closure_norm(reuse_cache=True)
        self.solver_residual = self._joint_action.solver_residual
        self.knee_target_clamped = self._joint_action.knee_target_clamped
        if isinstance(extras, dict):
            extras["closure_residual"] = self.closure_residual
            extras["solver_residual"] = self.solver_residual
        return obs, rewards, terminated, truncated, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset the loop solver warm start together with the environment."""
        super()._reset_idx(env_ids)
        # the base reset restored the nominal joint state; restart the solver there
        self._joint_action._knee_seed[env_ids] = self._joint_action.model.nominal_knees.view(1, 2, 2)
        self._joint_action._knee_targets[env_ids] = self._joint_action.nominal_knees[env_ids]
        self._robot.set_joint_position_target(
            self._joint_action.knee_targets[env_ids],
            joint_ids=self._joint_action.knee_ids,
            env_ids=env_ids,
        )
        self.closure_residual[env_ids] = 0.0
        self.solver_residual[env_ids] = 0.0


__all__ = ["FiveBarEnv"]
