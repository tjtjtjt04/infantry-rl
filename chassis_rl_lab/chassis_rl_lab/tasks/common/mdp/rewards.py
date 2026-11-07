"""蹲姿平衡任务的奖励项（env-writer 实现，对应设计规格 §6.5）。

所有奖励函数都是 **torch 向量化**（一次算全 batch）且 **NaN 安全**：
物理发散时 NaN/inf 会被 ``torch.nan_to_num`` 收敛成有限值，避免整批训练被一条坏轨迹污染。

奖励表（权重初值见 ``stand_up_env_cfg.RewardsCfg``）：

=====================================  ==========================================================
term                                   含义 / 公式
=====================================  ==========================================================
alive                                  +1：本步没终止
upright                                exp(-5 * ||g_xy||^2)，g = projected_gravity（机体系重力）
base_height                            exp(-|h - h_squat|^2 / 0.002)
hip_posture                            exp(-||q_hip - q_hip*||^2 / 0.01)
wheel_vel                              -Σ(v_wheel / 10)^2
base_lin_vel                           -(vx^2 + vy^2)
base_ang_vel_xy                        -(wx^2 + wy^2)
yaw_rate                               -wz^2
dof_vel                                -Σ dq^2（全部 10 关节）
dof_acc                                -Σ ddq^2（全部 10 关节）
torques                                -Σ τ^2（全部 10 关节）
action_rate                            -Σ (a_t - a_{t-1})^2
dof_pos_limits                         -关节超出软限位幅度之和（重点是 knee）
terminating                            -1：非 time-out 的终止（掉高/倾覆）
=====================================  ==========================================================
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from chassis_rl_lab.kinematics import HIP_ORDER, WHEEL_ORDER

from .observations import selected_joint_ids, squat_targets

__all__ = [
    # 任务奖励
    "is_alive",
    "upright",
    "tilt_l2",
    "pitch_progress",
    "base_height",
    "base_height_track",
    "height_rate_l2",
    "hip_posture",
    # 惩罚
    "wheel_vel_l2",
    "base_lin_vel_xy_l2",
    "base_ang_vel_xy_l2",
    "yaw_rate_l2",
    "dof_vel_l2",
    "dof_acc_l2",
    "torques_l2",
    "action_rate_l2",
    "dof_pos_limits",
    "is_terminated",
]


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _finite(t: torch.Tensor) -> torch.Tensor:
    """NaN 安全：NaN→0，+inf→1e6，-inf→-1e6（保证奖励是有限标量）。"""
    return torch.nan_to_num(t, nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def _exp_kernel(sq_distance: torch.Tensor, denom: float) -> torch.Tensor:
    """指数核 ``exp(-sq_distance / denom)``，NaN→0 并夹到 [0, 1]。"""
    return _finite(torch.exp(-sq_distance / denom)).clamp_(0.0, 1.0)


# --------------------------------------------------------------------------- #
# 任务奖励
# --------------------------------------------------------------------------- #
def is_alive(env) -> torch.Tensor:
    """存活奖励：本步没有被终止（time-out 不算终止）时为 1。"""
    return (~env.termination_manager.terminated).float()


def is_terminated(env) -> torch.Tensor:
    """终止惩罚项：非 time-out 的终止（掉太低/太高/倾覆）为 1，time-out 为 0。"""
    return (
        env.termination_manager.terminated & (~env.termination_manager.time_outs)
    ).float()


def upright(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), denom: float = 0.2
) -> torch.Tensor:
    """保持机身竖直：``exp(-||g_xy||^2 / denom)``，其中 g = 机体系投影重力。

    机身完全竖直时 g_xy = 0，奖励为 1；倾斜越大奖励越小。
    ``denom`` 越小核越窄（对倾角越严格）：
    - 蹲姿平衡任务：denom=0.2（= exp(-5x)）；
    - 起立任务：denom=0.02（= exp(-50x)），17° 倾角时奖励 ≈ 0，避免"歪着不动
      还能拿 65% 姿态奖励"的 reward hacking。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_xy_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return _exp_kernel(gravity_xy_sq, float(denom))


def tilt_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机身倾斜度平方 ``||g_xy||^2``（0 = 完全竖直，1 = 完全横躺）。

    用作**惩罚项**（负权重）：与 ``upright`` 的指数核不同，它在任何倾角下都有梯度，
    用于"从倒地起立"任务——窄核 exp(-50x) 在 17° 时已≈0，倒地姿态没有学习信号。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return _finite(
        torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    )


def pitch_progress(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """姿态改善过程量：``pitch * pitch_rate``（pitch 单位 rad，pitch_rate 单位 rad/s）。

    用作**负权重**奖励（如 weight=-1.5）：朝竖直方向转动时 product<0 → 奖励为正，
    背离竖直转动时 product>0 → 奖励为负。作用是给"起立"任务**每一步**都提供梯度，
    解决"必须连续多步同向力矩才有奖励、随机探索凑不出来"的稀疏奖励问题。

    pitch 用机体系投影重力：pitch = asin(g_b.x)；pitch_rate 用 body 角速度 y 分量。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    g_x = torch.clamp(asset.data.projected_gravity_b[:, 0], -1.0, 1.0)
    pitch = torch.asin(g_x)
    pitch_rate = asset.data.root_ang_vel_b[:, 1]
    return _finite(pitch * pitch_rate)


def base_height(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """保持 squat 离地高度：``exp(-|h - h_squat|^2 / 0.002)``。

    h 为 base_link 原点的世界 z；h_squat 来自 data/squat_pose.json。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    target = squat_targets(asset.device)["base_height"]
    return _exp_kernel(torch.square(height - target), 0.002)


def hip_posture(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """保持 squat 腿部姿态：``exp(-||q_hip - q_hip*||^2 / 0.01)``（HIP_ORDER）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = selected_joint_ids(asset, asset_cfg, HIP_ORDER)
    target = squat_targets(asset.device)["hips"].unsqueeze(0)
    error = asset.data.joint_pos[:, joint_ids] - target
    return _exp_kernel(torch.sum(torch.square(error), dim=1), 0.01)


# --------------------------------------------------------------------------- #
# 惩罚项
# --------------------------------------------------------------------------- #
def wheel_vel_l2(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), scale: float = 10.0
) -> torch.Tensor:
    """轮速惩罚：``Σ (v_wheel / scale)^2``，抑制无意义的轮子空转（原地平衡不应跑）。

    注意：函数名不能叫 ``wheel_vel``——``mdp/__init__.py`` 用 ``import *`` 汇总，
    会与 ``observations.wheel_vel``（2 维观测项）重名。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = selected_joint_ids(asset, asset_cfg, WHEEL_ORDER)
    wheel_vels = asset.data.joint_vel[:, joint_ids]
    return _finite(torch.sum(torch.square(wheel_vels / scale), dim=1))


def base_lin_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机身水平线速度惩罚：``vx^2 + vy^2``（机体系），抑制原地漂移。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return _finite(torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1))


def base_ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机身 roll/pitch 角速度惩罚：``wx^2 + wy^2``（机体系）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return _finite(torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1))


def yaw_rate_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """偏航角速度惩罚：``wz^2``（机体系），抑制原地自转。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return _finite(torch.square(asset.data.root_ang_vel_b[:, 2]))


def dof_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节速度惩罚：全部 10 个关节的 ``Σ dq^2``。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return _finite(torch.sum(torch.square(asset.data.joint_vel), dim=1))


def dof_acc_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节加速度惩罚：全部 10 个关节的 ``Σ ddq^2``（由 PhysX 上报）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return _finite(torch.sum(torch.square(asset.data.joint_acc), dim=1))


def torques_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节力矩惩罚：全部 10 个关节的 ``Σ τ^2``（隐式执行器为 PhysX 实际施加力矩）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return _finite(torch.sum(torch.square(asset.data.applied_torque), dim=1))


def action_rate_l2(env) -> torch.Tensor:
    """动作变化率惩罚：``Σ (a_t - a_{t-1})^2``（6 维动作，clip 后）。"""
    action_manager = env.action_manager
    return _finite(torch.sum(torch.square(action_manager.action - action_manager.prev_action), dim=1))


def dof_pos_limits(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节超软限位惩罚：超出软限位的幅度之和（单位 rad，重点防 knee 顶死）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    limits = asset.data.soft_joint_pos_limits
    out_of_limits = -(joint_pos - limits[..., 0]).clip(max=0.0)
    out_of_limits += (joint_pos - limits[..., 1]).clip(min=0.0)
    return _finite(torch.sum(out_of_limits, dim=1))


def base_height_track(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.01,
) -> torch.Tensor:
    """跟踪高度指令：``exp(-(h - h_cmd)^2 / sigma)``（1 = 完全到达目标高度）。

    ``h_cmd`` 来自 ``env.height_command``（由 reset 事件 ``reset_height_command`` 写入）；
    若尚未初始化则退回 squat 名义高度（仅兜底）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if hasattr(env, "height_command") and env.height_command is not None:
        target = env.height_command[:, 0]
    else:
        target = squat_targets(asset.device)["base_height"]
    height = asset.data.root_pos_w[:, 2]
    return _exp_kernel(torch.square(height - target), float(sigma))


def height_rate_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机身垂直速度平方 ``h_dot^2``（惩罚，防止靠上下猛甩来取巧）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return _finite(torch.square(asset.data.root_lin_vel_w[:, 2]))
