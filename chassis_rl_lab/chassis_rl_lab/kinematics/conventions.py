"""关节角坐标转换：仿真关节角（URDF/CAD 零位） ↔ 电机返回角（底盘数据.txt 原始范围）。

背景
----
- **仿真/URDF 关节角**：零位 = CAD 装配位 = 机械收回端（q=0），
  前髋范围 [0, +85.29°]、后髋 [−85.35°, 0]、前膝 [−114.5°, 0]、后膝 [0, +114.5°]；
- **电机返回角（你最初填写、用于 LQR/五连杆解算）**：同一物理角，但带 CAD 连杆零位偏置，
  来源于平面模型 ``derived`` 里的 ``ham_front_zero / ham_back_zero / calf_front_zero / calf_back_zero``：
  - 前髋：−10.15° ~ 75.16°   （电机角 = 仿真角 − 10.148°）
  - 后髋：104.80° ~ 190.15°  （电机角 = 仿真角 + 190.15°，含 360° 回绕）
  - 前膝：114.53° ~ 143.75°  （电机角 = 仿真角 + 143.75°）
  - 后膝：36.25° ~ 65.47°    （电机角 = 仿真角 + 36.25°）

注意：髋的范围宽度完全一致（85.3°），只是零点不同；膝的"填写范围宽度（29.2°）"
与我们仿真的机械限位宽度（114.5°）不一致，需确认你的膝数据是否为电机轴角（减速前）。

用法::

    from chassis_rl_lab.kinematics.conventions import sim_to_motor_deg, motor_deg_to_sim
    motor_deg = sim_to_motor_deg("left_front_hip", 0.0)       # -> -10.148
    sim_rad = motor_deg_to_sim("left_back_hip", 190.15)       # -> 0.0
"""

from __future__ import annotations

import math

__all__ = [
    "SIM_TO_MOTOR_OFFSET_DEG",
    "SIM_TO_LINK_OFFSET_DEG",
    "sim_to_link_deg",
    "link_deg_to_sim",
    "JOINT_ALIASES",
    "sim_to_motor_deg",
    "motor_deg_to_sim",
    "sim_to_motor_rad",
    "motor_rad_to_sim",
    "motor_limits_deg",
    "print_conversion_table",
]

#: 仿真关节角 -> 电机返回角 的零位偏置（度）：motor_deg = sim_deg + offset
#: 数值来自 chassis_rl_lab/data/chassis_rl_planar.json 的 derived 零位（与底盘数据.txt 完全吻合）。
SIM_TO_MOTOR_OFFSET_DEG: dict[str, float] = {
    # 前髋 ham_front_zero = -0.1771128 rad = -10.148°（电机角 = 仿真角 - 10.148）
    "left_front_hip": -10.148,
    "right_front_hip": -10.148,
    # 后髋 ham_back_zero = -2.9644663 rad ≡ 190.148°（加一圈使其落在 104.8~190.15）
    "left_back_hip": 190.148,
    "right_back_hip": 190.148,
    # 前膝 calf_front_zero = 2.5088697 rad = 143.750°
    "left_front_knee": 143.750,
    "right_front_knee": 143.750,
    # 后膝 calf_back_zero = 0.6327289 rad = 36.250°
    "left_back_knee": 36.250,
    "right_back_knee": 36.250,
}

#: 连杆相对机体水平方向（base_link +x 轴）的绝对角 = 关节角 + 该零位偏置。
#: 与电机返回角的偏置相同：电机返回角本身就是"连杆绝对角"。
SIM_TO_LINK_OFFSET_DEG: dict[str, float] = dict(SIM_TO_MOTOR_OFFSET_DEG)

#: 关节名别名（前/后、左/右 简写）
JOINT_ALIASES: dict[str, str] = {
    "front_hip": "left_front_hip",
    "back_hip": "left_back_hip",
    "front_knee": "left_front_knee",
    "back_knee": "left_back_knee",
    "lf_hip": "left_front_hip",
    "lb_hip": "left_back_hip",
    "rf_hip": "right_front_hip",
    "rb_hip": "right_back_hip",
    "lf_knee": "left_front_knee",
    "lb_knee": "left_back_knee",
    "rf_knee": "right_front_knee",
    "rb_knee": "right_back_knee",
}


def _resolve(name: str) -> str:
    name = str(name)
    if name in SIM_TO_MOTOR_OFFSET_DEG:
        return name
    if name in JOINT_ALIASES:
        return JOINT_ALIASES[name]
    raise KeyError(f"未知关节名：{name!r}（可用：{[*SIM_TO_MOTOR_OFFSET_DEG, *JOINT_ALIASES]}）")


def sim_to_motor_deg(joint_name: str, sim_deg: float) -> float:
    """仿真关节角（度）-> 电机返回角（度）。"""
    return float(sim_deg) + SIM_TO_MOTOR_OFFSET_DEG[_resolve(joint_name)]


def motor_deg_to_sim(joint_name: str, motor_deg: float) -> float:
    """电机返回角（度）-> 仿真关节角（度）。"""
    return float(motor_deg) - SIM_TO_MOTOR_OFFSET_DEG[_resolve(joint_name)]


def sim_to_motor_rad(joint_name: str, sim_rad: float) -> float:
    """仿真关节角（rad）-> 电机返回角（rad）。"""
    return float(sim_rad) + math.radians(SIM_TO_MOTOR_OFFSET_DEG[_resolve(joint_name)])


def motor_rad_to_sim(joint_name: str, motor_rad: float) -> float:
    """电机返回角（rad）-> 仿真关节角（rad）。"""
    return float(motor_rad) - math.radians(SIM_TO_MOTOR_OFFSET_DEG[_resolve(joint_name)])


def motor_limits_deg(joint_name: str, sim_limits_rad: tuple[float, float]) -> tuple[float, float]:
    """把仿真的关节限位（rad）换算成电机返回角范围（度）。"""
    lo = sim_to_motor_deg(joint_name, math.degrees(sim_limits_rad[0]))
    hi = sim_to_motor_deg(joint_name, math.degrees(sim_limits_rad[1]))
    return (lo, hi)


def print_conversion_table() -> None:
    """打印仿真限位 ↔ 电机角范围对照表（供文档/部署核对）。"""
    from chassis_rl_lab.kinematics.planar_model import load_planar_model

    data = load_planar_model()
    limits = data["joint_limits"]
    print(f"{'关节':<22}{'仿真限位(rad)':<28}{'仿真限位(deg)':<26}{'电机角范围(deg)':<26}{'底盘数据.txt':<22}")
    print("-" * 120)
    expected = {
        "left_front_hip": "-10.15 ~ 75.16",
        "left_back_hip": "104.80 ~ 190.15",
        "left_front_knee": "114.53 ~ 143.75",
        "left_back_knee": "36.25 ~ 65.47",
    }
    for joint_name, sim_limits in limits.items():
        if joint_name not in SIM_TO_MOTOR_OFFSET_DEG:
            continue
        lo, hi = float(sim_limits[0]), float(sim_limits[1])
        m_lo, m_hi = motor_limits_deg(joint_name, (lo, hi))
        exp = expected.get(joint_name, "-")
        print(
            f"{joint_name:<22}[{lo:+.5f}, {hi:+.5f}]".ljust(50)
            + f"[{math.degrees(lo):+7.2f}, {math.degrees(hi):+7.2f}]".ljust(26)
            + f"[{m_lo:+8.2f}, {m_hi:+8.2f}]".ljust(26)
            + exp
        )


if __name__ == "__main__":
    print_conversion_table()


def sim_to_link_deg(joint_name: str, sim_deg: float) -> float:
    """仿真关节角（度）-> **连杆相对机体水平的绝对夹角 φ**（度）。

    φ = q + zero_offset（零位偏置见 ``SIM_TO_LINK_OFFSET_DEG``）：
    前髋 φ = q − 10.148°、后髋 φ = q + 190.148°（mod 360）、前膝 φ = q + 143.75°、后膝 φ = q + 36.25°。
    """
    return float(sim_deg) + SIM_TO_LINK_OFFSET_DEG[_resolve(joint_name)]


def link_deg_to_sim(joint_name: str, link_deg: float) -> float:
    """连杆绝对角 φ（度）-> 仿真关节角（度）。"""
    return float(link_deg) - SIM_TO_LINK_OFFSET_DEG[_resolve(joint_name)]