#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""五连杆轮腿运动学（纯 Python / numpy / torch，不依赖 Isaac Lab）。

对外导出
--------
关节顺序常量（全项目唯一事实来源）::

    HIP_ORDER  = ("left_front_hip", "left_back_hip", "right_front_hip", "right_back_hip")
    KNEE_ORDER = ("left_front_knee", "left_back_knee", "right_front_knee", "right_back_knee")
    WHEEL_ORDER = ("wheel_left", "wheel_right")

模型与数据::

    LegPlanarModel   单侧平面五连杆模型（numpy；fk_front / fk_back / ik_knees）
    FiveBarSolver    torch 批量闭链解算器（solve / closure_residual / axle_z /
                     base_height / knee_targets_clamped / knee_limits / solve_numpy）
    load_planar_model(path=None) -> dict   # data/chassis_rl_planar.json
    load_squat_pose(path=None)   -> dict   # data/squat_pose.json

角度约定：``phi = atan2(-z, x)``（base 系 x-z 投影，从 +x 起向下为正），
等价于“绕世界 +y 轴右手定则正转”，左右同号；q=0 = CAD 装配位 = 行程收回端。
"""

from .planar_model import (
    DATA_DIR,
    DEFAULT_PLANAR_PATH,
    DEFAULT_SQUAT_PATH,
    HIP_ORDER,
    KNEE_ORDER,
    LEG_SIDES,
    WHEEL_ORDER,
    LegPlanarModel,
    angle_of,
    load_planar_model,
    load_squat_pose,
    rotate_xz,
    wrap_angle,
)
from .solver import FiveBarSolver

__all__ = [
    # 关节顺序
    "HIP_ORDER",
    "KNEE_ORDER",
    "WHEEL_ORDER",
    "LEG_SIDES",
    # 模型 / 解算器
    "LegPlanarModel",
    "FiveBarSolver",
    # 数据加载
    "load_planar_model",
    "load_squat_pose",
    "DATA_DIR",
    "DEFAULT_PLANAR_PATH",
    "DEFAULT_SQUAT_PATH",
    # 小工具
    "wrap_angle",
    "angle_of",
    "rotate_xz",
]
