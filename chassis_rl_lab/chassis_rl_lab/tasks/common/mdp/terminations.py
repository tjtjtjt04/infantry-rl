"""蹲姿平衡任务的终止项（env-writer 实现，对应设计规格 §6.6）。

终止条件：

- ``time_out``      ：到达 episode 时限（由 ``isaaclab.envs.mdp.time_out`` 提供，time_out=True）；
- ``base_too_low``  ：base 高度低于 squat 高度 - 0.05 m（掉下去 / 腿塌了）；
- ``base_tilted``   ：机身相对竖直方向倾角超过 ``tilt_rad``（默认 0.5 rad）；
- ``base_too_high`` ：base 高度高于 squat 高度 + 0.15 m（防止"站起来"取巧）。

阈值来源
--------
优先读 ``chassis_rl_lab.kinematics.load_squat_pose()`` 返回的 ``termination`` 字段
（``base_too_low`` / ``base_too_high`` 为 base_link 世界 z 绝对高度 [m]，``tilt_rad`` 为倾角 [rad]）；
缺省时按规格 §6.6 由 squat base_height 推导。若文件里写的是相对 squat 高度的小偏移量，
本模块会按"相对偏移"兼容解释（见 :func:`squat_thresholds`）。

注意
----
Isaac Lab 的 ``projected_gravity_b`` 在机身竖直时约为 ``(0, 0, -1)``（重力方向）,
所以"倾角"要用 ``acos(-g_z)`` 计算，等价于“竖直方向 z 分量 = -g_z 小于 cos(limit_rad)”。
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .observations import squat_targets

__all__ = [
    "squat_thresholds",
    "base_too_low",
    "base_too_high",
    "base_tilted",
    "failed_to_recover",
    "state_invalid",
]

#: 规格 §6.6 的缺省阈值（相对于 squat base_height）
_DEFAULT_LOW_OFFSET = 0.05
_DEFAULT_HIGH_OFFSET = 0.15
_DEFAULT_TILT_RAD = 0.5


def _resolve_low(raw: object, base_height: float) -> float:
    """解析 ``base_too_low``：绝对高度，或（很小时）视为相对 squat 高度的下降量。"""
    if raw is None:
        low = base_height - _DEFAULT_LOW_OFFSET
    else:
        value = float(raw)
        if value <= 0.0:
            # 负值/0：相对偏移（如 -0.05 表示比 squat 低 5 cm）
            low = base_height + value
        elif value < 0.5 * base_height:
            # 很小的正值（不可能低于轮半径的绝对高度）：视为下降量
            low = base_height - value
        else:
            # 正常量级：视为绝对世界高度
            low = value
    # 物理上 base 不可能低到接近地面，夹一个合理下界
    return float(min(max(low, 0.02), base_height - 0.005))


def _resolve_high(raw: object, base_height: float) -> float:
    """解析 ``base_too_high``：绝对高度，或（不大于 squat 高度时）视为相对抬升量。"""
    if raw is None:
        high = base_height + _DEFAULT_HIGH_OFFSET
    else:
        value = float(raw)
        if value > base_height:
            # 正常量级：视为绝对世界高度
            high = value
        else:
            # 不大于 squat 高度：视为相对抬升量
            high = base_height + value
    return float(max(high, base_height + 0.01))


def _resolve_tilt(raw: object) -> float:
    """解析 ``tilt_rad``，夹到 (0, pi/2) 内。"""
    if raw is None:
        return _DEFAULT_TILT_RAD
    value = float(raw)
    return float(min(max(value, 0.1), 1.4))


def squat_thresholds() -> dict[str, float]:
    """返回解析后的终止阈值（绝对高度 [m] 与倾角 [rad]）。"""
    targets = squat_targets(None)
    base_height = float(targets["base_height"])
    termination = dict(targets["termination"])
    return {
        "base_too_low": _resolve_low(termination.get("base_too_low", None), base_height),
        "base_too_high": _resolve_high(termination.get("base_too_high", None), base_height),
        "tilt_rad": _resolve_tilt(termination.get("tilt_rad", None)),
    }


def base_too_low(
    env,
    minimum_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base_link 世界 z 低于阈值（或出现 NaN/inf）时终止。"""
    asset: Articulation = env.scene[asset_cfg.name]
    if minimum_height is None:
        minimum_height = squat_thresholds()["base_too_low"]
    height = asset.data.root_pos_w[:, 2]
    return (~torch.isfinite(height)) | (height < float(minimum_height))


def base_too_high(
    env,
    maximum_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base_link 世界 z 高于阈值（或出现 NaN/inf）时终止，防止"站起来"取巧。"""
    asset: Articulation = env.scene[asset_cfg.name]
    if maximum_height is None:
        maximum_height = squat_thresholds()["base_too_high"]
    height = asset.data.root_pos_w[:, 2]
    return (~torch.isfinite(height)) | (height > float(maximum_height))


def base_tilted(
    env,
    limit_rad: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身相对竖直方向的倾角超过 ``limit_rad``（或出现 NaN/inf）时终止。"""
    asset: Articulation = env.scene[asset_cfg.name]
    if limit_rad is None:
        limit_rad = squat_thresholds()["tilt_rad"]
    # projected_gravity_b 在竖直时约为 (0, 0, -1)，故 cos(倾角) = -g_z
    cos_tilt = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(cos_tilt)
    return (~torch.isfinite(tilt)) | (tilt > float(limit_rad))


def failed_to_recover(
    env,
    limit_rad: float = 0.5,
    grace_steps: int = 200,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """开局 ``grace_steps`` 个控制步后仍然倾斜超过 ``limit_rad`` → 终止。

    用于"起立/恢复平衡"任务：倒地后不能在原地一直躺着刷存活奖励，
    必须在这段时间内把机身立起来（超过 0.5 rad ≈ 28.6° 视为未恢复）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cos_tilt = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    tilt = torch.acos(cos_tilt)
    not_recovered = tilt > float(limit_rad)
    after_grace = env.episode_length_buf >= int(grace_steps)
    return (~torch.isfinite(tilt)) | (after_grace & not_recovered)


def state_invalid(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_lin_vel: float = 30.0,
    max_ang_vel: float = 60.0,
    max_joint_vel: float = 200.0,
) -> torch.Tensor:
    """状态异常终止：根状态/关节状态出现 NaN/Inf，或速度爆炸时终止该 env。

    用途：物理发散（接触爆炸、力矩异常等）产生的 NaN 会污染观测/奖励，进而把整个训练
    带成 NaN。这里把"已经坏掉"的 env 立刻终止并重置，防止异常扩散；同时把速度上限
    （线速度/角速度/关节速度）作为"数值爆炸"的兜底检测。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w
    root_lin = asset.data.root_lin_vel_w
    root_ang = asset.data.root_ang_vel_w
    joint_pos = asset.data.joint_pos
    joint_vel = asset.data.joint_vel

    bad = (
        (~torch.isfinite(root_pos)).any(dim=-1)
        | (~torch.isfinite(root_quat)).any(dim=-1)
        | (~torch.isfinite(root_lin)).any(dim=-1)
        | (~torch.isfinite(root_ang)).any(dim=-1)
        | (~torch.isfinite(joint_pos)).any(dim=-1)
        | (~torch.isfinite(joint_vel)).any(dim=-1)
    )
    # 速度爆炸（用 nan_to_num 防止比较本身被 NaN 干扰；NaN 已在上面判掉）
    lin = torch.nan_to_num(root_lin.norm(dim=-1), nan=0.0, posinf=1.0e9, neginf=1.0e9)
    ang = torch.nan_to_num(root_ang.norm(dim=-1), nan=0.0, posinf=1.0e9, neginf=1.0e9)
    jvel = torch.nan_to_num(joint_vel.abs().amax(dim=-1), nan=0.0, posinf=1.0e9, neginf=1.0e9)
    bad = bad | (lin > float(max_lin_vel)) | (ang > float(max_ang_vel)) | (jvel > float(max_joint_vel))
    return bad
