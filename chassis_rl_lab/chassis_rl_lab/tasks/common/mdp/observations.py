"""chassis-rl 任务共用观测项（自定义部分）。

各任务按需拼装观测组；本文件提供腿部/轮子状态、机身高度、高度指令等自定义项：

==============================  ====  ===============================================
term                            维度 说明
==============================  ====  ===============================================
hip_pos_rel                     4   q_hip - squat_hip（HIP_ORDER）
knee_pos_rel                    4   q_knee - squat_knee（KNEE_ORDER）
hip_vel                         4   髋关节角速度（HIP_ORDER）
wheel_vel                       2   轮关节角速度（WHEEL_ORDER，正 = 前进）
base_height_obs                 1   base_link 世界高度 [m]
height_command                  1   本 episode 的目标高度指令 [m]
==============================  ====  ===============================================

约定
----
- 所有关节都**按名字**选取（``SceneEntityCfg(joint_names=..., preserve_order=True)``），
  顺序严格等于 ``HIP_ORDER`` / ``KNEE_ORDER`` / ``WHEEL_ORDER``，不依赖 articulation 内部顺序；
- squat 目标姿态来自 ``chassis_rl_lab.kinematics.load_squat_pose()``（即 data/squat_pose.json），
  首次调用时缓存，之后按设备缓存成 torch 张量；
- 自定义观测都做了 NaN 安全处理，避免物理发散时把 NaN 送进策略网络。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from chassis_rl_lab.kinematics import HIP_ORDER, KNEE_ORDER, WHEEL_ORDER, load_squat_pose

__all__ = [
    "squat_targets",
    "selected_joint_ids",
    "hip_pos_rel",
    "knee_pos_rel",
    "hip_vel",
    "wheel_vel",
    "base_height_obs",
    "height_command",
    "commands_obs",
    "actuated_dof_vel",
]

# --------------------------------------------------------------------------- #
# squat 目标姿态的加载与缓存
# --------------------------------------------------------------------------- #
#: 原始 JSON 内容缓存（只加载一次文件）
_POSE_CACHE: dict[str, dict] = {}
#: 按设备缓存的目标姿态张量
_TARGET_CACHE: dict[str, dict] = {}


def _ordered(values: Mapping[str, float] | Sequence[float], order: Sequence[str]) -> tuple[float, ...]:
    """把 ``{"joint_name": value}`` 或已排好序的序列统一成按 ``order`` 排列的 tuple。"""
    if isinstance(values, Mapping):
        missing = [name for name in order if name not in values]
        if missing:
            raise KeyError(f"squat_pose 缺少关节 {missing}")
        return tuple(float(values[name]) for name in order)
    values = tuple(float(v) for v in values)
    if len(values) != len(order):
        raise ValueError(f"squat_pose 序列长度 {len(values)} != {len(order)}")
    return values


def squat_targets(device: torch.device | str | None = None) -> dict:
    """返回 squat 目标姿态（按关节名排好序，张量已放到 ``device``）。

    Returns:
        字典，键为：

        - ``hips``:   ``(4,)`` float32，HIP_ORDER；
        - ``knees``:  ``(4,)`` float32，KNEE_ORDER；
        - ``base_height``: float，squat 姿态下 base_link 原点离地高度 [m]；
        - ``termination``: dict，终止阈值建议值（base_too_low / base_too_high / tilt_rad）。
    """
    if "pose" not in _POSE_CACHE:
        # 只读一次 data/squat_pose.json；数值由运动学脚本测量，勿手写
        _POSE_CACHE["pose"] = load_squat_pose()
    pose = _POSE_CACHE["pose"]

    dev = torch.device("cpu") if device is None else torch.device(device)
    key = str(dev)
    if key not in _TARGET_CACHE:
        _TARGET_CACHE[key] = {
            "hips": torch.tensor(_ordered(pose["hips"], HIP_ORDER), dtype=torch.float32, device=dev),
            "knees": torch.tensor(_ordered(pose["knees"], KNEE_ORDER), dtype=torch.float32, device=dev),
            "base_height": float(pose["base_height"]),
            "termination": dict(pose.get("termination", {}) or {}),
        }
    return _TARGET_CACHE[key]


# --------------------------------------------------------------------------- #
# 关节索引工具
# --------------------------------------------------------------------------- #
def selected_joint_ids(asset: Articulation, asset_cfg: SceneEntityCfg | None, order: Sequence[str]) -> list[int]:
    """从 ``SceneEntityCfg`` 取出关节索引；没给关节名时按 ``order`` 现查。

    这样即使某个 term 忘了传 ``joint_names``，也会退化成"按常量顺序查名字"，
    而不是静默取到错误的关节。
    """
    joint_ids = slice(None) if asset_cfg is None else asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids, _ = asset.find_joints(list(order), preserve_order=True)
    return list(joint_ids)


def _finite(t: torch.Tensor) -> torch.Tensor:
    """把 NaN/inf 替换成 0，保证观测永远是有限值。"""
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# 自定义观测项
# --------------------------------------------------------------------------- #
def hip_pos_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """4 个 hip 关节相对 squat 目标的角度偏差（rad，HIP_ORDER）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = selected_joint_ids(asset, asset_cfg, HIP_ORDER)
    target = squat_targets(asset.device)["hips"].unsqueeze(0)
    return _finite(asset.data.joint_pos[:, joint_ids] - target)


def knee_pos_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """4 个 knee 关节相对 squat 目标的角度偏差（rad，KNEE_ORDER）。

    knee 是被动随动关节，实际角度由闭链解算目标 + PD 跟踪决定；观测"实测偏差"
    让策略能看到闭链跟踪误差。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = selected_joint_ids(asset, asset_cfg, KNEE_ORDER)
    target = squat_targets(asset.device)["knees"].unsqueeze(0)
    return _finite(asset.data.joint_pos[:, joint_ids] - target)


def hip_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """4 个 hip 关节角速度（rad/s，HIP_ORDER）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = selected_joint_ids(asset, asset_cfg, HIP_ORDER)
    return _finite(asset.data.joint_vel[:, joint_ids])


def wheel_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """2 个轮关节角速度（rad/s，WHEEL_ORDER）；正转 = 车体 +x 方向（前进）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = selected_joint_ids(asset, asset_cfg, WHEEL_ORDER)
    return _finite(asset.data.joint_vel[:, joint_ids])


# --------------------------------------------------------------------------- #
# 高度相关观测（"起立后按指令改变机体高度"任务用）
# --------------------------------------------------------------------------- #
def base_height_obs(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """base_link 世界高度 ``(N, 1)`` [m]，让策略知道当前高度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return _finite(asset.data.root_pos_w[:, 2].unsqueeze(-1))


def height_command(env) -> torch.Tensor:
    """本 episode 的目标高度指令 ``(N, 1)`` [m]。

    由事件 ``reset_height_command`` 在每个 episode reset 时写入 ``env.height_command``；
    若尚未初始化（例如观测先于事件被调用），返回零并补一个占位 buffer。
    """
    if not hasattr(env, "height_command") or env.height_command is None:
        env.height_command = torch.zeros(env.num_envs, 1, device=env.device)
    return _finite(env.height_command)


def commands_obs(env, height_scale: float = 5.0) -> torch.Tensor:
    """命令向量 ``(N, 3)``：``[lin_vel_x_cmd, ang_vel_yaw_cmd, height_cmd]``（Fudan 式布局）。

    - 本任务不做速度跟踪 → 前两维固定为 0（预留给后续速度指令）；
    - 第三维 = 目标机体高度 [m]（由事件 ``reset_height_command`` 写入 ``env.height_command``），
      为避免策略输入量级过小，**函数内部直接乘 ``height_scale``（默认 ×5，等同 Fudan 的
      obs_scales.height_measurements=5.0）**，ObsTerm 里不再缩放。
    """
    if not hasattr(env, "height_command") or env.height_command is None:
        env.height_command = torch.zeros(env.num_envs, 1, device=env.device)
    zeros = torch.zeros(env.num_envs, 2, device=env.device)
    height_scaled = _finite(env.height_command)[:, 0:1] * float(height_scale)
    return torch.cat([zeros, height_scaled], dim=-1)


def actuated_dof_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """6 个被驱动关节的角速度 ``(N, 6)``：4 hip（HIP_ORDER）+ 2 wheel（WHEEL_ORDER）。

    Fudan 观测里的 ``dof_vel`` 是"机器人全部 DOF"的速度（他们的模型 6 DOF = 4 hip + 2 wheel）；
    本机器人仿真里有 10 个关节（含 4 个被动 knee），这里只取被驱动的 6 个，
    维度与 Fudan 的 25 维观测保持一致。缩放（×0.05）在 ObsTerm.scale 里做。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    hip_ids = selected_joint_ids(asset, asset_cfg, HIP_ORDER)
    wheel_ids = selected_joint_ids(asset, asset_cfg, WHEEL_ORDER)
    return _finite(torch.cat([asset.data.joint_vel[:, hip_ids], asset.data.joint_vel[:, wheel_ids]], dim=-1))
