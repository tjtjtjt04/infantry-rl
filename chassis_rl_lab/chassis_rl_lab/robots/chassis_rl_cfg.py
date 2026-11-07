"""chassis-rl 五连杆轮腿机器人的 Isaac Lab 资产配置。

机器人
------
- 10 个关节：4 hip + 4 knee + 2 wheel；真实执行器只有 4 hip + 2 wheel，
  4 个 knee 是五连杆闭链的被动随动件（由 ``mdp.actions.ChassisWheelLegAction``
  在每个控制步用解析闭链解算器给出位置目标）。
- 资产来自 ``D:\\Robomaster\\RL\\chassis-rl``（CAD URDF），经
  ``tools/build_sim_urdf.py`` 生成 ``chassis-rl-sim.urdf``，再由
  ``tools/build_asset.py`` 转成 USD。

关节符号约定（重要）
--------------------
**所有关节都以绕世界 +y 轴右手定则旋转为正**（从 +y 看向 -y 逆时针），左右同号：

- 左侧腿 5 个关节（4 个腿关节 + 左轮）在 sim URDF 里 axis = ``(0, 1, 0)``；
- 右侧腿 5 个关节的连杆坐标系绕 z 转了 180°，因此 axis = ``(0, -1, 0)``，
  但两者在世界系里都是绕 +y 正向转。
- 物理含义：前 hip 正转 / 后 hip 负转 = 腿向下伸；轮子正转 = 车体向 +x（前进）。

零位 ``q = 0`` 是 CAD 装配位（机械行程的"收回端"），squat 姿态在其附近，
具体数值由 ``chassis_rl_lab/data/squat_pose.json`` 给出（由运动学脚本测量，勿手写）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from chassis_rl_lab.kinematics import (
    HIP_ORDER,
    KNEE_ORDER,
    WHEEL_ORDER,
    load_squat_pose,
)

# --------------------------------------------------------------------------- #
# 资产路径
# --------------------------------------------------------------------------- #
ASSET_DIR: Path = Path(__file__).resolve().parents[1] / "assets" / "chassis-rl"
#: 统一 +y 关节约定、无 loop_joint 的仿真 URDF（由 tools/build_sim_urdf.py 生成）
URDF_PATH: Path = ASSET_DIR / "urdf" / "chassis-rl-sim.urdf"
#: 由 tools/build_asset.py 生成的 USD（USD_PATH.parent 是转换器输出目录）
USD_PATH: Path = ASSET_DIR / "urdf" / "chassis-rl-sim" / "chassis-rl-sim.usd"

# --------------------------------------------------------------------------- #
# 执行器参数（与真实机器人对齐：V8009P 电机 40 Nm / 16.755 rad/s，轮 4.92 Nm / 61.47 rad/s）
#
# 增益说明：squat 姿态在机械行程收回端附近（hip 距限位仅 ~0.035 rad），
# 若 hip PD 太软，重力会让腿先"折"到限位（静止偏差 ≈ 4.3 Nm / kp），
# 所以 hip kp 取 300（静态偏差 ~0.014 rad），knee kp 取 1000（knee 静态偏差
# ~0.008 rad，闭链残差 ~2 mm）。轮子只做速度跟踪，damping=2.0（与旧项目 squat 一致）。
# --------------------------------------------------------------------------- #
HIP_STIFFNESS: float = 300.0
HIP_DAMPING: float = 10.0
KNEE_STIFFNESS: float = 1000.0
KNEE_DAMPING: float = 25.0
WHEEL_STIFFNESS: float = 0.0  # 轮子只做速度跟踪
WHEEL_DAMPING: float = 2.0

#: 髋关节**持续**力矩上限 [Nm]（电机峰值 40 Nm 不能长时间保持，训练/部署按 20 Nm 算）
HIP_EFFORT_LIMIT: float = 20.0
#: 髋关节峰值力矩（仅文档参考；资产 USD 里 URDF 仍记录 40 Nm）
HIP_PEAK_EFFORT: float = 40.0
KNEE_EFFORT_LIMIT: float = 40.0
WHEEL_EFFORT_LIMIT: float = 4.92

HIP_VELOCITY_LIMIT: float = 16.755
KNEE_VELOCITY_LIMIT: float = 16.755
WHEEL_VELOCITY_LIMIT: float = 61.4705

#: hip 位置动作缩放（rad / 动作单位）；策略动作被 clip 到 [-6, 6] 时约 ±1.5 rad
HIP_ACTION_SCALE: float = 0.25
#: 轮速动作缩放（rad/s / 动作单位）
WHEEL_ACTION_SCALE: float = 10.0

#: 控制频率 = 1 / (sim.dt * decimation)
SIM_DT: float = 1.0 / 200.0
DECIMATION: int = 4
CONTROL_HZ: float = 1.0 / (SIM_DT * DECIMATION)


# --------------------------------------------------------------------------- #
# squat 姿态（来自运动学脚本测量，左右相同）
# --------------------------------------------------------------------------- #
def _ordered(values: Mapping[str, float] | Sequence[float], order: Sequence[str], what: str) -> tuple[float, ...]:
    """把 ``{"joint_name": value}`` 或按 ``order`` 排好的序列统一成 tuple。"""
    if isinstance(values, Mapping):
        missing = [name for name in order if name not in values]
        if missing:
            raise KeyError(f"squat_pose['{what}'] 缺少关节 {missing}")
        return tuple(float(values[name]) for name in order)
    values = tuple(float(v) for v in values)
    if len(values) != len(order):
        raise ValueError(f"squat_pose['{what}'] 长度 {len(values)} != {len(order)}")
    return values


_SQUAT = load_squat_pose()
SQUAT_HIPS: tuple[float, ...] = _ordered(_SQUAT["hips"], HIP_ORDER, "hips")
SQUAT_KNEES: tuple[float, ...] = _ordered(_SQUAT["knees"], KNEE_ORDER, "knees")
#: squat 姿态下 base_link 原点离地高度（车轮着地，m）
SQUAT_BASE_HEIGHT: float = float(_SQUAT["base_height"])
#: 轮子半径（m，从 STL 实测，运动学脚本给出）
WHEEL_RADIUS: float = float(_SQUAT.get("wheel_radius", 0.068))

#: 初始关节位置（按名字），供 ArticulationCfg.init_state 使用
INIT_JOINT_POS: dict[str, float] = {
    **dict(zip(HIP_ORDER, SQUAT_HIPS)),
    **dict(zip(KNEE_ORDER, SQUAT_KNEES)),
    **{name: 0.0 for name in WHEEL_ORDER},
}


# --------------------------------------------------------------------------- #
# 资产配置
# --------------------------------------------------------------------------- #
CHASSIS_RL_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 五连杆闭链靠 knee 目标维持，腿之间不应互撞（CAD 位折叠时容易穿插）
            enabled_self_collisions=False,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, SQUAT_BASE_HEIGHT),
        joint_pos=dict(INIT_JOINT_POS),
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hips": ImplicitActuatorCfg(
            joint_names_expr=list(HIP_ORDER),
            effort_limit_sim=HIP_EFFORT_LIMIT,
            velocity_limit_sim=HIP_VELOCITY_LIMIT,
            stiffness=HIP_STIFFNESS,
            damping=HIP_DAMPING,
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=list(KNEE_ORDER),
            effort_limit_sim=KNEE_EFFORT_LIMIT,
            velocity_limit_sim=KNEE_VELOCITY_LIMIT,
            stiffness=KNEE_STIFFNESS,
            damping=KNEE_DAMPING,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=list(WHEEL_ORDER),
            effort_limit_sim=WHEEL_EFFORT_LIMIT,
            velocity_limit_sim=WHEEL_VELOCITY_LIMIT,
            stiffness=WHEEL_STIFFNESS,
            damping=WHEEL_DAMPING,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


__all__ = [
    "ASSET_DIR",
    "URDF_PATH",
    "USD_PATH",
    "CHASSIS_RL_CFG",
    "HIP_ORDER",
    "KNEE_ORDER",
    "WHEEL_ORDER",
    "SQUAT_HIPS",
    "SQUAT_KNEES",
    "SQUAT_BASE_HEIGHT",
    "WHEEL_RADIUS",
    "HIP_ACTION_SCALE",
    "WHEEL_ACTION_SCALE",
    "SIM_DT",
    "DECIMATION",
    "CONTROL_HZ",
]
