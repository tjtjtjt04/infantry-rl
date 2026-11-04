"""Shared five-bar environment: scene, actions, loop closure, diagnostics.

The two remaining tasks (`squat` and `squat_torque`) build on this module.  The
robot is a two-wheel differential drive with a five-bar leg on each side; PhysX
sees a tree (10 DOFs) because the two loop joints are not articulation DOFs, so
the passive knees are driven to the positions that close the loop every control
step.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import mdp
from config import INFANTRY_RL_CFG, ensure_ground_usd
from robot_model import nominal_base_height

# --------------------------------------------------------------------------- #
# Shared stance / fall constants
# --------------------------------------------------------------------------- #
#: root height (m) of the measured nominal stance
NOMINAL_HEIGHT = nominal_base_height()
#: root height below which the robot counts as fallen (m)
FALL_HEIGHT = 0.11
#: tilt at which the robot counts as fallen (rad)
FALL_ANGLE = 1.2

#: heading (rad) the robot is supposed to hold (all envs spawn at identity yaw)
TARGET_HEADING = 0.0
#: planar position (m) the robot is supposed to hold, in the ENVIRONMENT frame
TARGET_X = 0.0
TARGET_Y = 0.0

#: hip action authority and hard travel
HIP_SCALE = float(os.environ.get("INFANTRY_RL_HIP_SCALE", "0.25"))
HIP_RANGE = float(os.environ.get("INFANTRY_RL_HIP_RANGE", "0.45"))


# --------------------------------------------------------------------------- #
# Scene / actions / terminations (shared)
# --------------------------------------------------------------------------- #
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat ground + the robot. Everything is local, nothing is fetched."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=ensure_ground_usd(),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot = INFANTRY_RL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(intensity=500.0, color=(0.6, 0.6, 0.6)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joints = mdp.FiveBarJointActionCfg(
        asset_name="robot",
        hip_scale=HIP_SCALE,
        wheel_scale=5.0,
        hip_range=HIP_RANGE,
        limit_margin=0.01,
        # 2 is enough: worst-case residual 3.6 um vs the ~750 um the sim achieves.
        solver_iters=2,
        heading_control=False,
        heading_mode="override",
        target_heading=TARGET_HEADING,
        heading_kp=3.0,
        heading_kd=0.35,
        heading_max_yaw_rate=3.0,
        wheel_radius=0.068,
        track_width=0.4379,
        position_control=False,
        target_position_x=TARGET_X,
        target_position_y=TARGET_Y,
        position_kp=1.5,
        position_kd=0.6,
        position_max_speed=0.6,
    )


@configclass
class TerminationsCfg:
    """Generic termination terms (squat uses its own tighter limits)."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": FALL_ANGLE})
    base_too_low = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": FALL_HEIGHT}
    )


class FiveBarEnv(ManagerBasedRLEnv):
    """Manager-based RL environment with control-level five-bar loop closure.

    PhysX drops the closed chain (the articulation has 10 joints), so the passive
    knee targets are re-solved from the measured hip angles on every control step.
    """

    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._joint_action_name: str = getattr(cfg, "joint_action_name", "joints")
        self._joint_action = self.action_manager.get_term(self._joint_action_name)
        self._robot = self.scene["robot"]

        #: measured loop error ``|W - E|`` per side (metres), ``(N, 2)`` = (left, right)
        self.closure_residual = torch.zeros(self.num_envs, 2, device=self.device)
        #: solver residual of the same quantity (metres)
        self.solver_residual = torch.zeros(self.num_envs, 2, device=self.device)
        #: number of knee targets that had to be clipped to the URDF limits
        self.knee_target_clamped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        #: weight-independent diagnostics, accumulated every control step and logged
        #: as episode means at reset.
        self._diag_sums: dict[str, torch.Tensor] = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "heading_error_abs_rad",
                "vx_cmd_abs",
                "vx_actual_abs",
                "wz_cmd_abs",
                "wz_actual_abs",
                "wz_inplace_cmd_abs",
                "wz_inplace_actual_abs",
                "inplace_count",
                "wheel_target_abs",
                "wheel_actual_abs",
                "wheel_error_abs",
                "roll_abs_rad",
                "pitch_abs_rad",
            )
        }
        self._diag_count = torch.zeros(self.num_envs, device=self.device)
        self._last_inplace = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._velocity_command_names = [
            name
            for name in self.command_manager.active_terms
            if getattr(getattr(self.command_manager.get_term(name).cfg, "ranges", None), "lin_vel_x", None)
            is not None
        ]
        self._heading_command_names = [
            name
            for name in self._velocity_command_names
            if getattr(self.command_manager.get_term(name).cfg, "heading_command", False)
        ]
        self._vx_hist_bins = 5
        self._vx_hist_counts = torch.zeros(self.num_envs, self._vx_hist_bins, device=self.device)
        self._vx_stand_count = torch.zeros(self.num_envs, device=self.device)
        self._cmd_low, self._cmd_high = 0.0, 0.0
        if self._velocity_command_names:
            rng = getattr(self.command_manager.get_term(self._velocity_command_names[0]).cfg, "ranges", None)
            lin = getattr(rng, "lin_vel_x", None)
            if lin is not None:
                self._cmd_low, self._cmd_high = float(lin[0]), float(lin[1])

        # make sure the very first physics step already sees closed knees
        self._write_knee_targets()

    @property
    def joint_action(self):
        """The :class:`~mdp.actions.FiveBarJointAction` term."""
        return self._joint_action

    @property
    def max_closure_residual(self) -> float:
        """Largest measured loop error over all environments and both legs (m)."""
        return float(self.closure_residual.max().item())

    def _write_knee_targets(self) -> None:
        self._robot.set_joint_position_target(self._joint_action.knee_targets, joint_ids=self._joint_action.knee_ids)

    def step(self, action: torch.Tensor):
        """Recompute the knee targets, then run the standard environment step."""
        self._joint_action.update_knee_targets()
        self._write_knee_targets()

        obs, rewards, terminated, truncated, extras = super().step(action)

        self.closure_residual = self._joint_action.measured_closure_norm(reuse_cache=True)
        self.solver_residual = self._joint_action.solver_residual
        self.knee_target_clamped = self._joint_action.knee_target_clamped

        if self._velocity_command_names:
            name = self._velocity_command_names[0]
            term = self.command_manager.get_term(name)
            cmd = self.command_manager.get_command(name)
            wheel_actual = self._robot.data.joint_vel[:, self._joint_action.wheel_ids]
            if getattr(self._joint_action, "torque_control", False):
                # No wheel velocity target exists under torque control.
                wheel_target = wheel_actual
            else:
                wheel_target = self._joint_action.applied_wheel_targets
            s = self._diag_sums
            if name in self._heading_command_names:
                s["heading_error_abs_rad"] += math_utils.wrap_to_pi(
                    term.heading_target - self._robot.data.heading_w
                ).abs()
            s["vx_cmd_abs"] += cmd[:, 0].abs()
            s["vx_actual_abs"] += self._robot.data.root_lin_vel_b[:, 0].abs()
            s["wz_cmd_abs"] += cmd[:, 2].abs()
            s["wz_actual_abs"] += self._robot.data.root_ang_vel_b[:, 2].abs()
            inplace = (cmd[:, 0].abs() < 1e-6) & (cmd[:, 2].abs() > 1e-6)
            s["wz_inplace_cmd_abs"] += cmd[:, 2].abs() * inplace
            s["wz_inplace_actual_abs"] += self._robot.data.root_ang_vel_b[:, 2].abs() * inplace
            s["inplace_count"] += inplace.float()
            self._last_inplace[:] = inplace
            s["wheel_target_abs"] += wheel_target.abs().mean(dim=1)
            s["wheel_actual_abs"] += wheel_actual.abs().mean(dim=1)
            s["wheel_error_abs"] += (wheel_actual - wheel_target).abs().mean(dim=1)
            roll, pitch, _ = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
            s["roll_abs_rad"] += roll.abs()
            s["pitch_abs_rad"] += pitch.abs()
            moving = cmd[:, 0].abs() > 1e-6
            if not self._heading_command_names:
                moving = moving | (cmd[:, 2].abs() > 1e-6)
            if self._cmd_high - self._cmd_low > 1e-9:
                frac = ((cmd[:, 0] - self._cmd_low) / (self._cmd_high - self._cmd_low)).clamp(0.0, 0.999999)
                idx = (frac * self._vx_hist_bins).long()
                for b in range(self._vx_hist_bins):
                    self._vx_hist_counts[:, b] += (moving & (idx == b)).float()
            self._vx_stand_count += (~moving).float()
            self._diag_count += 1.0

        if isinstance(extras, dict):
            extras["closure_residual"] = self.closure_residual
            extras["solver_residual"] = self.solver_residual
        return obs, rewards, terminated, truncated, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset the loop solver warm start together with the environment."""
        super()._reset_idx(env_ids)
        reset_knees = getattr(self._joint_action, "reset_knees", None)
        if reset_knees is None:
            reset_knees = self._joint_action.nominal_knees[0]
        self._joint_action._knee_seed[env_ids] = reset_knees.view(1, 2, 2)
        self._joint_action._knee_targets[env_ids] = reset_knees
        self._robot.set_joint_position_target(
            self._joint_action.knee_targets[env_ids],
            joint_ids=self._joint_action.knee_ids,
            env_ids=env_ids,
        )
        self.closure_residual[env_ids] = 0.0
        self.solver_residual[env_ids] = 0.0

        if self._velocity_command_names:
            count = self._diag_count[env_ids].clamp(min=1.0)
            log = self.extras.setdefault("log", {})
            if self._heading_command_names:
                log["Metrics/heading_error_mean_deg"] = float(
                    ((self._diag_sums["heading_error_abs_rad"][env_ids] / count) * 57.29577951308232)
                    .mean()
                    .item()
                )
            inplace_steps = self._diag_sums["inplace_count"][env_ids].sum()
            inplace_n = inplace_steps.clamp(min=1.0)
            log["Metrics/wz_inplace_cmd_abs_mean"] = float(
                (self._diag_sums["wz_inplace_cmd_abs"][env_ids].sum() / inplace_n).item()
            )
            log["Metrics/wz_inplace_actual_abs_mean"] = float(
                (self._diag_sums["wz_inplace_actual_abs"][env_ids].sum() / inplace_n).item()
            )
            log["Metrics/inplace_pct"] = float(
                (inplace_steps / self._diag_count[env_ids].sum().clamp(min=1.0) * 100.0).item()
            )
            tm = getattr(self, "termination_manager", None)
            flags_all = getattr(tm, "_last_episode_dones", None) if tm is not None else None
            if flags_all is not None:
                names = list(getattr(tm, "active_terms", []))
                flags = flags_all[env_ids]
                spin = self._last_inplace[env_ids]
                n_total = int(flags.shape[0])
                n_spin = int(spin.sum().item())
                n_drive = n_total - n_spin
                log["Metrics/spin_reset_pct"] = 100.0 * n_spin / max(n_total, 1)
                for term in ("pitch_limit", "base_too_low", "bad_orientation"):
                    if term not in names:
                        continue
                    idx = names.index(term)
                    log[f"Metrics/spin_term_{term}"] = (
                        float(flags[spin, idx].float().mean().item()) if n_spin else 0.0
                    )
                    log[f"Metrics/drive_term_{term}"] = (
                        float(flags[~spin, idx].float().mean().item()) if n_drive else 0.0
                    )
            for name, buf in self._diag_sums.items():
                if name in (
                    "heading_error_abs_rad",
                    "roll_abs_rad",
                    "pitch_abs_rad",
                    "wz_inplace_cmd_abs",
                    "wz_inplace_actual_abs",
                    "inplace_count",
                ):
                    continue
                log[f"Metrics/{name}_mean"] = float((buf[env_ids] / count).mean().item())
            rad2deg = 57.29577951308232
            for name, key in (
                ("roll_abs_rad", "roll_abs_mean_deg"),
                ("pitch_abs_rad", "pitch_abs_mean_deg"),
            ):
                log[f"Metrics/{key}"] = float(
                    ((self._diag_sums[name][env_ids] / count) * rad2deg).mean().item()
                )
            hist_pct = (self._vx_hist_counts[env_ids] / count.unsqueeze(1)) * 100.0
            for b in range(self._vx_hist_bins):
                log[f"Metrics/vx_cmd_hist{b}_pct"] = float(hist_pct[:, b].mean().item())
            log["Metrics/vx_cmd_standing_pct"] = float(
                ((self._vx_stand_count[env_ids] / count) * 100.0).mean().item()
            )
            for buf in self._diag_sums.values():
                buf[env_ids] = 0.0
            self._vx_hist_counts[env_ids] = 0.0
            self._vx_stand_count[env_ids] = 0.0
            self._diag_count[env_ids] = 0.0


__all__ = [
    "ActionsCfg",
    "FALL_ANGLE",
    "FALL_HEIGHT",
    "FiveBarEnv",
    "HIP_RANGE",
    "HIP_SCALE",
    "NOMINAL_HEIGHT",
    "RobotSceneCfg",
    "TARGET_HEADING",
    "TARGET_X",
    "TARGET_Y",
    "TerminationsCfg",
]
