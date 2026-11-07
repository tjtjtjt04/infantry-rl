#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""平面五连杆模型（纯 numpy / Python，不依赖 Isaac Lab / torch）。

坐标系与角度约定
----------------
* 只使用 base_link 系的 **x-z 投影**。每条链的所有转动轴都平行于世界 y 轴，
  各链处于平行的 y 平面内，y 分量对闭链没有影响；因此闭链条件在 x-z 平面内是
  ``W_front == W_back``（米）。
* 角度约定 ``phi = atan2(-z, x)``：从 +x 起、向下（-z）为正，等价于
  “绕世界 +y 轴右手定则正转”（从 +y 看向 -y 逆时针）。
  向量 (x, z) 绕 +y 旋转 a 后：``x' = x cos a + z sin a``，``z' = z cos a - x sin a``，
  同时 ``phi' = phi + a``。因此对某关节施加 q 就是在对应连杆的绝对角上加 q。
* q = 0 = CAD 装配位 = 机械行程“收回端”（腿最短、base 最低）。

闭链（每侧）
------------
    base --(前 hip A_f)--> 前大腿 --(前膝 K_f)--> 前小腿 --> 轮轴 W
    base --(后 hip A_b)--> 后大腿 --(后膝 K_b)--> 后小腿 --> 轴孔锚点 W'
q = 0 时 W 与 W' 在同一点（轮轴中心），两条小腿长度固定，故给定两个 hip 角后
W 是两圆（圆心 K_f、K_b，半径 calf_front_len / calf_back_len）的交点，两解对应
两个装配支路；取离上一帧解最近的支路（warm start）。

本模块的模型数据全部由 ``tools/fit_planar_model.py`` 从 URDF 精确提取到
``chassis_rl_lab/data/chassis_rl_planar.json``，不要手写数字。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 关节顺序（全项目唯一事实来源；env / action / 观测 / 脚本均按此顺序）
# ---------------------------------------------------------------------------
HIP_ORDER = ("left_front_hip", "left_back_hip", "right_front_hip", "right_back_hip")
KNEE_ORDER = ("left_front_knee", "left_back_knee", "right_front_knee", "right_back_knee")
WHEEL_ORDER = ("wheel_left", "wheel_right")

#: hips -> 腿的顺序：每条腿 = (front side, back side)
#: HIP_ORDER 中 leg0 = left_front_hip + left_back_hip，leg1 = right_front_hip + right_back_hip
LEG_SIDES = ("left", "right")

# 数据文件默认路径：<pkg_root>/data/*.json
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_PLANAR_PATH = DATA_DIR / "chassis_rl_planar.json"
DEFAULT_SQUAT_PATH = DATA_DIR / "squat_pose.json"

# ---------------------------------------------------------------------------
# 角度/向量小工具
# ---------------------------------------------------------------------------


def wrap_angle(angle: float) -> float:
    """把角度归一化到 (-pi, pi]。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_of(x: float, z: float) -> float:
    """平面向量 (x, z) 的约定角 phi = atan2(-z, x)。"""
    return math.atan2(-z, x)


def rotate_xz(x: float, z: float, angle: float) -> tuple[float, float]:
    """把平面向量 (x, z) 绕世界 +y 轴旋转 angle（phi -> phi + angle）。"""
    c, s = math.cos(angle), math.sin(angle)
    return x * c + z * s, z * c - x * s


def _rotate_vec(vec: tuple[float, float], angle: float) -> tuple[float, float]:
    return rotate_xz(vec[0], vec[1], angle)


# ---------------------------------------------------------------------------
# 平面五连杆模型（单侧）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegPlanarModel:
    """单侧（left / right）平面五连杆几何，单位 m / rad，base_link 系 x-z 投影。

    字段含义
    --------
    hip_front / hip_back : 前/后 hip 关节在 base 系下的 (x, z)。
    ham_front / ham_back : q=0 时 base 系下的前/后大腿向量（hip -> knee）。
    calf_front           : q=0 时 base 系下的前小腿向量（knee -> 轮轴中心）。
    calf_back            : q=0 时 base 系下的后小腿向量（knee -> 轴孔锚点）。
    limits               : {"front_hip","back_hip","front_knee","back_knee"} -> (lower, upper)。
    wheel_radius         : 轮半径（由 meshes/*_wheel.STL 顶点实测）。
    """

    side: str
    hip_front: tuple[float, float]
    hip_back: tuple[float, float]
    ham_front: tuple[float, float]
    ham_back: tuple[float, float]
    calf_front: tuple[float, float]
    calf_back: tuple[float, float]
    limits: dict[str, tuple[float, float]]
    wheel_radius: float

    # ---------------- 派生长度 / 零位角 ----------------
    @property
    def ham_front_len(self) -> float:
        return math.hypot(*self.ham_front)

    @property
    def ham_back_len(self) -> float:
        return math.hypot(*self.ham_back)

    @property
    def calf_front_len(self) -> float:
        return math.hypot(*self.calf_front)

    @property
    def calf_back_len(self) -> float:
        return math.hypot(*self.calf_back)

    @property
    def ham_front_zero(self) -> float:
        """q=0 时前大腿的绝对角 phi。"""
        return angle_of(*self.ham_front)

    @property
    def ham_back_zero(self) -> float:
        return angle_of(*self.ham_back)

    @property
    def calf_front_zero(self) -> float:
        return angle_of(*self.calf_front)

    @property
    def calf_back_zero(self) -> float:
        return angle_of(*self.calf_back)

    # ---------------- 正运动学 ----------------
    def fk_front(self, q_hip: float, q_knee: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """前链正解：返回 (前膝点 K_f, 轮轴点 W_f)（base 系 x-z）。"""
        kx = self.hip_front[0] + rotate_xz(*self.ham_front, q_hip)[0]
        kz = self.hip_front[1] + rotate_xz(*self.ham_front, q_hip)[1]
        ax = kx + rotate_xz(*self.calf_front, q_hip + q_knee)[0]
        az = kz + rotate_xz(*self.calf_front, q_hip + q_knee)[1]
        return (kx, kz), (ax, az)

    def fk_back(self, q_hip: float, q_knee: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """后链正解：返回 (后膝点 K_b, 轴孔锚点 W_b)（base 系 x-z）。"""
        kx = self.hip_back[0] + rotate_xz(*self.ham_back, q_hip)[0]
        kz = self.hip_back[1] + rotate_xz(*self.ham_back, q_hip)[1]
        ax = kx + rotate_xz(*self.calf_back, q_hip + q_knee)[0]
        az = kz + rotate_xz(*self.calf_back, q_hip + q_knee)[1]
        return (kx, kz), (ax, az)

    # ---------------- 逆运动学（闭链解算） ----------------
    def knee_points(self, q_hip_front: float, q_hip_back: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """给定两个 hip 角，返回两个膝点 (K_f, K_b)。"""
        kx_f = self.hip_front[0] + rotate_xz(*self.ham_front, q_hip_front)[0]
        kz_f = self.hip_front[1] + rotate_xz(*self.ham_front, q_hip_front)[1]
        kx_b = self.hip_back[0] + rotate_xz(*self.ham_back, q_hip_back)[0]
        kz_b = self.hip_back[1] + rotate_xz(*self.ham_back, q_hip_back)[1]
        return (kx_f, kz_f), (kx_b, kz_b)

    def axle_candidates(
        self, q_hip_front: float, q_hip_back: float
    ) -> tuple[list[tuple[float, float]], bool]:
        """两圆交点（轮轴 W 的两个候选），返回 (候选列表, 是否几何不可达)。

        不可达时退化为两圆心连线上 h=0 的点（唯一候选），闭链残差会 > 0。
        """
        (kx_f, kz_f), (kx_b, kz_b) = self.knee_points(q_hip_front, q_hip_back)
        dx, dz = kx_b - kx_f, kz_b - kz_f
        d = math.hypot(dx, dz)
        if d < 1e-12:
            d, ex, ez = 1e-12, 1.0, 0.0
        else:
            ex, ez = dx / d, dz / d
        l1, l2 = self.calf_front_len, self.calf_back_len
        a = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
        h_sq = l1 * l1 - a * a
        unreachable = h_sq < 0.0
        h = math.sqrt(max(h_sq, 0.0))
        # 连线的左法向
        nx, nz = -ez, ex
        px, pz = kx_f + a * ex, kz_f + a * ez
        return [(px + s * h * nx, pz + s * h * nz) for s in (1.0, -1.0)], unreachable

    def ik_knees(
        self,
        q_hip_front: float,
        q_hip_back: float,
        prev: Optional[tuple[float, float]] = None,
    ) -> tuple[float, float, float]:
        """由两个 hip 角解出两个膝角，返回 (前膝角, 后膝角, 闭链残差[m])。

        两个装配支路取“离 prev 最近”的一个（warm start）；prev=None 时取与
        CAD 装配位（q=0，膝角 0）一致的支路。
        """
        (kx_f, kz_f), (kx_b, kz_b) = self.knee_points(q_hip_front, q_hip_back)
        candidates, _unreachable = self.axle_candidates(q_hip_front, q_hip_back)
        ref = (0.0, 0.0) if prev is None else (float(prev[0]), float(prev[1]))

        best = None
        best_dist = float("inf")
        for wx, wz in candidates:
            phi_f = angle_of(wx - kx_f, wz - kz_f)
            phi_b = angle_of(wx - kx_b, wz - kz_b)
            qk_f = wrap_angle(phi_f - self.calf_front_zero - q_hip_front)
            qk_b = wrap_angle(phi_b - self.calf_back_zero - q_hip_back)
            dist = wrap_angle(qk_f - ref[0]) ** 2 + wrap_angle(qk_b - ref[1]) ** 2
            if dist < best_dist:
                best_dist = dist
                best = (qk_f, qk_b)

        qk_f, qk_b = best
        _, w_f = self.fk_front(q_hip_front, qk_f)
        _, w_b = self.fk_back(q_hip_back, qk_b)
        residual = math.hypot(w_f[0] - w_b[0], w_f[1] - w_b[1])
        return qk_f, qk_b, residual

    def zero_pose(self) -> dict[str, tuple[float, float]]:
        """q=0（CAD 装配位）下的关键点，便于校验/工具打印。"""
        k_f, w_f = self.fk_front(0.0, 0.0)
        k_b, w_b = self.fk_back(0.0, 0.0)
        return {"knee_front": k_f, "axle_front": w_f, "knee_back": k_b, "axle_back": w_b}

    # ---------------- 构造 ----------------
    @classmethod
    def from_dict(cls, data: dict, side: str) -> "LegPlanarModel":
        """从 chassis_rl_planar.json 的 dict 构造某一侧模型。"""
        if side not in ("left", "right"):
            raise ValueError(f"side 必须是 'left'/'right'，实际 {side!r}")
        side_data = data["sides"][side]
        limits = {
            key: (float(value[0]), float(value[1]))
            for key, value in side_data["limits"].items()
        }
        return cls(
            side=side,
            hip_front=tuple(float(v) for v in side_data["hip_front"]),
            hip_back=tuple(float(v) for v in side_data["hip_back"]),
            ham_front=tuple(float(v) for v in side_data["ham_front"]),
            ham_back=tuple(float(v) for v in side_data["ham_back"]),
            calf_front=tuple(float(v) for v in side_data["calf_front"]),
            calf_back=tuple(float(v) for v in side_data["calf_back"]),
            limits=limits,
            wheel_radius=float(data.get("wheel_radius", side_data.get("wheel_radius", 0.0))),
        )

    @classmethod
    def from_json(cls, path: str | Path | None = None, side: str = "left") -> "LegPlanarModel":
        """从 JSON 文件构造某一侧模型（path=None 用默认数据文件）。"""
        return cls.from_dict(load_planar_model(path), side)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def _resolve_data_path(path: str | Path | None, default: Path) -> Path:
    return Path(path).resolve() if path is not None else default


def load_planar_model(path: str | Path | None = None) -> dict:
    """读取 ``data/chassis_rl_planar.json``（返回原始 dict）。"""
    data_path = _resolve_data_path(path, DEFAULT_PLANAR_PATH)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"平面模型数据不存在: {data_path}（先运行 python tools/fit_planar_model.py）"
        )
    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_squat_pose(path: str | Path | None = None) -> dict:
    """读取 ``data/squat_pose.json``（返回原始 dict）。"""
    data_path = _resolve_data_path(path, DEFAULT_SQUAT_PATH)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"squat 姿态数据不存在: {data_path}（先运行 python tools/fit_planar_model.py）"
        )
    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "HIP_ORDER",
    "KNEE_ORDER",
    "WHEEL_ORDER",
    "LEG_SIDES",
    "DATA_DIR",
    "DEFAULT_PLANAR_PATH",
    "DEFAULT_SQUAT_PATH",
    "LegPlanarModel",
    "load_planar_model",
    "load_squat_pose",
    "wrap_angle",
    "angle_of",
    "rotate_xz",
]
