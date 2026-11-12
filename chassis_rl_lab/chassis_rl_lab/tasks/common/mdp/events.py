"""蹲姿平衡任务的事件项（env-writer 实现，对应设计规格 §6.7）。

本文件只包含需要自定义逻辑的事件 ``reset_closure``；其余事件直接使用
``isaaclab.envs.mdp`` 里的内置函数（在 ``stand_up_env_cfg.EventCfg`` 中配置）：

==============================  ==========  ==============================================================
事件                             mode        实现
==============================  ==========  ==============================================================
reset_root                      reset       ``mdp.reset_root_state_uniform``（x/y ±0.05 m，yaw ±0.1 rad，z=h_squat 固定）
reset_hips                      reset       ``mdp.reset_joints_by_offset``（4 hip 在 squat 基础上 ±0.02 rad）
reset_wheels                    reset       ``mdp.reset_joints_by_offset``（轮位置/速度都归 0）
reset_closure                   reset       **本文件**：读 hip 实测 → FiveBarSolver 解 knee → 写 knee 位置（速度 0）
randomize_friction              startup     ``mdp.randomize_rigid_body_material``（轮子材质 static/dynamic ∈ [0.6, 1.0]）
randomize_base_mass             startup     ``mdp.randomize_rigid_body_mass``（base_link ±0.5 kg）
randomize_actuator_gains        startup     ``mdp.randomize_actuator_gains``（stiffness/damping ×[0.8, 1.2]）
push_robot                      interval    ``mdp.push_by_setting_velocity``（每 2–4 s，x/y ±0.15 m/s，yaw ±0.2 rad/s）
==============================  ==========  ==============================================================

顺序很重要
----------
EventManager 按 ``EventCfg`` 里的属性顺序依次执行同一 mode 的事件，
所以 ``reset_closure`` 必须放在 ``reset_hips`` / ``reset_wheels`` **之后**：
``reset_joints_by_offset`` 会把写进 PhysX 的关节位置同步写入
``Articulation._data.joint_pos``（见 ``write_joint_position_to_sim``），
因此 ``reset_closure`` 读到的就是本步刚随机化过的 hip 实测角。
"""

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from chassis_rl_lab.kinematics import HIP_ORDER, KNEE_ORDER, FiveBarSolver

from .observations import selected_joint_ids, squat_targets

__all__ = [
    "reset_closure",
    "reset_hips_to_pose",
    "reset_height_command",
    "reset_root_fallen_directional",
    "reset_sensor_bias",
]

#: 每个 device 复用一个解算器（GPU 批量、内部无 python 循环）
_SOLVER_CACHE: dict[str, FiveBarSolver] = {}


def _get_solver(device: torch.device | str) -> FiveBarSolver:
    """按设备懒加载并缓存五连杆闭链解算器。"""
    key = str(device)
    if key not in _SOLVER_CACHE:
        _SOLVER_CACHE[key] = FiveBarSolver(device=key)
    return _SOLVER_CACHE[key]


def _as_env_ids(env, env_ids) -> torch.Tensor:
    """把 EventManager 传进来的 env_ids（tensor/list/slice/None）统一成 long tensor。"""
    device = env.device
    if env_ids is None:
        return torch.arange(env.num_envs, device=device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=device, dtype=torch.long)[env_ids]
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def reset_closure(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knee_cfg: SceneEntityCfg | None = None,
    knee_margin: float = 0.01,
) -> None:
    """reset 事件：用当前 hip 实测角解出 knee，并把 knee 关节状态写成闭链一致的值。

    五连杆闭链在 PhysX 里没有约束（sim URDF 删掉了 loop_joint），所以初始状态必须
    由解析解保证前后链的轮轴点/锚点重合；否则一开场 knee 就会朝错误方向猛追。

    Args:
        env: 环境实例。
        env_ids: 需要 reset 的环境索引。
        asset_cfg: 机器人实体、以及 hip 关节名（``joint_names=HIP_ORDER, preserve_order=True``）。
        knee_cfg: 可选，knee 关节名（``joint_names=KNEE_ORDER, preserve_order=True``）；
            None 时按常量 ``KNEE_ORDER`` 现查。
        knee_margin: knee 目标相对限位保留的余量 [rad]。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _as_env_ids(env, env_ids)
    if ids.numel() == 0:
        return

    solver = _get_solver(asset.device)
    hip_ids = selected_joint_ids(asset, asset_cfg, HIP_ORDER)
    knee_ids = selected_joint_ids(asset, knee_cfg, KNEE_ORDER)
    targets = squat_targets(asset.device)

    # 读上一步（reset_hips）写入的 hip 实测角；异常值退回 squat hip
    hips = asset.data.joint_pos[ids][:, hip_ids].to(dtype=torch.float32)
    hips = torch.where(torch.isfinite(hips), hips, targets["hips"].unsqueeze(0).expand_as(hips))

    # 闭链解算 + 夹到关节限位内并留 margin：直接复用 kinematics 的单一实现
    # （内部 warm start 首帧 = squat 解，与 ActionTerm 的处理一致）
    knees = solver.knee_targets_clamped(hips, margin=float(knee_margin))
    knees = torch.nan_to_num(knees, nan=0.0, posinf=0.0, neginf=0.0)

    # 只写 knee 关节的位置 + 速度（速度置 0），不覆盖 hip/root
    zeros = torch.zeros_like(knees)
    asset.write_joint_state_to_sim(knees, zeros, joint_ids=knee_ids, env_ids=ids)


def reset_height_command(
    env,
    env_ids,
    height_range: tuple[float, float] = (0.12, 0.19),
    nominal: float = 0.1716,
    curriculum_steps: int = 0,
    curriculum_start_steps: int = 0,
) -> None:
    """reset 事件：给每个 episode 随机一个目标高度指令，写入 ``env.height_command``。

    Args:
        env: 环境实例。
        env_ids: 需要 reset 的环境索引。
        height_range: 目标 base 高度范围 [m]（lo, hi）。
        nominal: 名义高度 [m]（curriculum 起点；默认 ≈ squat 平衡高度 0.1716）。
        curriculum_steps: >0 时按全局步数把指令从 nominal 逐步放开到完整范围
            （0 = 不启用 curriculum，直接随机）。
        curriculum_start_steps: curriculum 的起点全局步数（续训时由 train.py 设为
            对齐后的步数，使"更高高度范围"从续训时刻起再慢慢放开，而不是瞬间给满）。
    """
    ids = _as_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    if not hasattr(env, "height_command") or env.height_command is None:
        env.height_command = torch.zeros(env.num_envs, 1, device=env.device)
    lo, hi = float(height_range[0]), float(height_range[1])
    sampled = torch.rand(ids.numel(), 1, device=env.device) * (hi - lo) + lo
    if int(curriculum_steps) > 0:
        progress = max(0.0, float(env.common_step_counter) - float(curriculum_start_steps))
        frac = min(1.0, progress / float(curriculum_steps))
        sampled = float(nominal) + frac * (sampled - float(nominal))
    env.height_command[ids] = sampled


def reset_hips_to_pose(
    env,
    env_ids,
    positions: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    noise: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """reset 事件：把 4 个 hip 直接写成给定**绝对角**（可加均匀噪声），速度置 0。

    用于"倒地时先把腿长收到最短"：``positions`` 取机械收回端姿态（q=0，腿长最短）。
    必须排在 ``reset_closure`` 之前（闭链解会读取这里写入的 hip 实测角）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ids = _as_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    hip_ids = selected_joint_ids(asset, asset_cfg, HIP_ORDER)
    base = torch.tensor(list(positions), device=asset.device, dtype=torch.float32).unsqueeze(0)
    pos = base.expand(ids.numel(), -1).clone()
    if float(noise) > 0.0:
        pos = pos + (torch.rand_like(pos) * 2.0 - 1.0) * float(noise)
    zeros = torch.zeros_like(pos)
    asset.write_joint_state_to_sim(pos, zeros, joint_ids=hip_ids, env_ids=ids)


def reset_root_fallen_directional(
    env,
    env_ids,
    pitch_range: tuple[float, float] = (0.24, 0.34),
    phase_steps: int = 48 * 300,
    x_range: tuple[float, float] = (-0.05, 0.05),
    y_range: tuple[float, float] = (-0.05, 0.05),
    z_range: tuple[float, float] = (-0.004, 0.004),
    roll_range: tuple[float, float] = (-0.05, 0.05),
    yaw_range: tuple[float, float] = (-0.1, 0.1),
    velocity_range: tuple[float, float] = (-0.05, 0.05),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fixed_sign: int | None = None,
) -> None:
    """reset 事件：**按训练阶段切换倒地方向**的机身位姿重置。

    阶段划分按全局控制步 ``env.common_step_counter``：

    - 阶段0（前 ``phase_steps`` 步，默认 300 iter×48 步）：pitch = **正**方向倒地；
    - 阶段1（第 2 个 ``phase_steps`` 步，默认 301~600 iter）：pitch = **负**方向倒地；
    - 阶段2 之后：pitch 正/负**每个 episode 随机**。

    其余 x/y/z/roll/yaw 与速度仍做小随机。用于"两个倒地方向分别学，再随机混合"的课程训练。
    """
    from isaaclab.utils.math import quat_from_euler_xyz

    asset: Articulation = env.scene[asset_cfg.name]
    ids = _as_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    n = ids.numel()
    device = env.device

    root_states = asset.data.default_root_state[ids].clone()
    root_states[:, 0:3] += env.scene.env_origins[ids]

    # ---- 阶段 -> pitch 符号 ----
    if fixed_sign is not None:
        # 固定方向（课程第一段：始终从同一个方向倒地开始）
        sign = torch.full((n,), float(fixed_sign), device=device)
    else:
        phase = int(env.common_step_counter) // max(1, int(phase_steps))
        if phase == 0:
            sign = torch.ones(n, device=device)
        elif phase == 1:
            sign = -torch.ones(n, device=device)
        else:
            sign = torch.where(
                torch.rand(n, device=device) < 0.5,
                -torch.ones(n, device=device),
                torch.ones(n, device=device),
            )

    def _uniform(rng: tuple[float, float]) -> torch.Tensor:
        lo, hi = float(rng[0]), float(rng[1])
        return torch.rand(n, device=device) * (hi - lo) + lo

    pitch_lo, pitch_hi = float(pitch_range[0]), float(pitch_range[1])
    pitch = (torch.rand(n, device=device) * (pitch_hi - pitch_lo) + pitch_lo) * sign
    roll = _uniform(roll_range)
    yaw = _uniform(yaw_range)

    root_states[:, 0] += _uniform(x_range)
    root_states[:, 1] += _uniform(y_range)
    root_states[:, 2] += _uniform(z_range)
    root_states[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
    root_states[:, 7:10] = _uniform(velocity_range).unsqueeze(-1).expand(n, 3)
    root_states[:, 10:13] = _uniform(velocity_range).unsqueeze(-1).expand(n, 3)

    asset.write_root_pose_to_sim(root_states[:, :7], env_ids=ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:13], env_ids=ids)


def reset_sensor_bias(
    env,
    env_ids,
    gyro_bias_range: float = 0.01,
    tilt_bias_deg: float = 1.0,
) -> None:
    """reset 事件：为每个 episode 采样 **IMU 陀螺零偏** 与 **安装/零位倾角偏置**。

    模拟真机 IMU 的常数零偏（BMI088 零偏随机化）与安装误差：
    - ``gyro_bias_range``: 三轴陀螺零偏，均匀分布于 ±该值 [rad/s]；
    - ``tilt_bias_deg``: 投影重力的安装/标定倾角误差，均匀分布于 ±该值 [deg]。

    结果写入 ``env.gyro_bias`` (N,3) 与 ``env.tilt_bias`` (N,2: roll, pitch)，
    由观测项 ``imu_gyro`` / ``imu_gravity`` 读取并叠加。
    """
    ids = _as_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    if not hasattr(env, "gyro_bias") or env.gyro_bias is None:
        env.gyro_bias = torch.zeros(env.num_envs, 3, device=env.device)
        env.tilt_bias = torch.zeros(env.num_envs, 2, device=env.device)
    n = ids.numel()
    g = float(gyro_bias_range)
    t = math.radians(float(tilt_bias_deg))
    env.gyro_bias[ids] = (torch.rand(n, 3, device=env.device) * 2.0 - 1.0) * g
    env.tilt_bias[ids] = (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * t