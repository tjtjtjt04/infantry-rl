"""生成"电机角定义"示意图：docs/figures/motor_angle_definition.png。

电机角 m（按用户电机约定）：
- 参考：**中间连杆（base_link，两个髋轴之间的机架）**；
- 测量对象：大腿连杆（ham，连到膝点的那根）；
- 方向：逆时针为正（x 向右、z 向上看）；
- 现象：腿最短（收腿）时 m 为正，腿最长（伸展）时 m 为负。

图中用角度弧直接标出 m_LF / m_LB（最短姿态实线、最长姿态虚线），数值放说明框。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from chassis_rl_lab.kinematics import FiveBarSolver

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_PATH = PROJECT_ROOT / "docs" / "figures" / "motor_angle_definition.png"


def leg_pose(solver, q_front: float, q_back: float):
    model = solver.models["left"]
    q_hips = torch.tensor([[q_front, q_back, q_front, q_back]], dtype=torch.float64)
    knees = solver.solve(q_hips)[0][0].tolist()
    knee_f, axle_f = model.fk_front(q_front, knees[0])
    knee_b, axle_b = model.fk_back(q_back, knees[1])
    return {
        "hip_f": tuple(model.hip_front),
        "hip_b": tuple(model.hip_back),
        "knee_f": knee_f,
        "knee_b": knee_b,
        "axle": ((axle_f[0] + axle_b[0]) / 2.0, (axle_f[1] + axle_b[1]) / 2.0),
        "height": float(solver.base_height(q_hips)[0]),
    }


def angle_from_x(hip, knee) -> float:
    """大腿方向相对 +x 的标准角（逆时针为正，度）。"""
    return math.degrees(math.atan2(knee[1] - hip[1], knee[0] - hip[0]))


def motor_deg(hip, knee, is_front: bool) -> float:
    """电机角 m：中间连杆向外方向 → 大腿，逆时针为正（度）。"""
    dx, dz = knee[0] - hip[0], knee[1] - hip[1]
    return math.degrees(math.atan2(dz, dx if is_front else -dx))


def arc(center, radius, start_deg, end_deg, color, lw=2.0, ls="-", n=50):
    ang = np.deg2rad(np.linspace(start_deg, end_deg, n))
    xs = center[0] + radius * np.cos(ang)
    zs = center[1] + radius * np.sin(ang)
    ax.plot(xs, zs, color=color, lw=lw, ls=ls)


fig, ax = plt.subplots(figsize=(12, 7.0))
solver = FiveBarSolver(device="cpu")
pose_short = leg_pose(solver, 0.0, 0.0)  # 最短（收腿）
pose_long = leg_pose(solver, 0.45, -0.45)  # 最长（伸展）

# ---- 中间连杆（base_link） ----
ax.plot(
    [pose_short["hip_b"][0], pose_short["hip_f"][0]],
    [pose_short["hip_b"][1], pose_short["hip_f"][1]],
    color="#111111", lw=7, solid_capstyle="butt", zorder=4,
)
ax.text(0.0, pose_short["hip_f"][1] + 0.020, "中间连杆 / middle link (base_link)",
        ha="center", fontsize=10)

# ---- 两个姿态的腿 ----
for pose, color, lw, ls in (
    (pose_short, "#111111", 3.4, "-"),
    (pose_long, "#8a8a8a", 2.2, "--"),
):
    ax.plot([pose["hip_f"][0], pose["knee_f"][0]], [pose["hip_f"][1], pose["knee_f"][1]],
            color=color, lw=lw, ls=ls, zorder=3)  # 前大腿 ham
    ax.plot([pose["hip_b"][0], pose["knee_b"][0]], [pose["hip_b"][1], pose["knee_b"][1]],
            color=color, lw=lw, ls=ls, zorder=3)  # 后大腿 ham
    ax.plot([pose["knee_f"][0], pose["axle"][0]], [pose["knee_f"][1], pose["axle"][1]],
            color=color, lw=1.0, ls=ls, zorder=2)
    ax.plot([pose["knee_b"][0], pose["axle"][0]], [pose["knee_b"][1], pose["axle"][1]],
            color=color, lw=1.0, ls=ls, zorder=2)
    ax.add_patch(plt.Circle(pose["axle"], solver.wheel_radius, fill=False, color=color, lw=1.0, ls=ls))
    ax.plot([pose["hip_f"][0]], [pose["hip_f"][1]], marker="o", color="#1f77b4", ms=10, zorder=6)
    ax.plot([pose["hip_b"][0]], [pose["hip_b"][1]], marker="o", color="#d62728", ms=10, zorder=6)

# ---- 参考方向（中间连杆向外） ----
hip_f, hip_b = pose_short["hip_f"], pose_short["hip_b"]
ax.plot([hip_f[0], hip_f[0] + 0.075], [hip_f[1], hip_f[1]], color="#1f77b4", lw=1.6, ls=":", zorder=5)
ax.plot([hip_b[0], hip_b[0] - 0.075], [hip_b[1], hip_b[1]], color="#d62728", lw=1.6, ls=":", zorder=5)
ax.text(hip_f[0] + 0.040, hip_f[1] - 0.010, "参考 0°", fontsize=8, color="#1f77b4")
ax.text(hip_b[0] - 0.040, hip_b[1] - 0.010, "参考 0°", fontsize=8, color="#d62728", ha="right")

# ---- 电机角弧：最短姿态（实线）/ 最长姿态（虚线） ----
m_f_short = motor_deg(hip_f, pose_short["knee_f"], True)     # ≈ +10.15°
m_b_short = motor_deg(hip_b, pose_short["knee_b"], False)    # ≈ +10.15°
m_f_long = motor_deg(hip_f, pose_long["knee_f"], True)       # ≈ -15.64°
m_b_long = motor_deg(hip_b, pose_long["knee_b"], False)      # ≈ -15.63°

# 楔形弧：中间连杆方向 ↔ 大腿方向（不标旋转箭头，符号按电机约定）
ang_f_short = angle_from_x(hip_f, pose_short["knee_f"])  # ≈ +10.15°
ang_b_short = angle_from_x(hip_b, pose_short["knee_b"])  # ≈ 169.85°
ang_f_long = angle_from_x(hip_f, pose_long["knee_f"])    # ≈ -15.64°
ang_b_long = angle_from_x(hip_b, pose_long["knee_b"])    # ≈ -164.37°
arc(hip_f, 0.034, 0.0, ang_f_short, "#1f77b4", lw=2.2)
arc(hip_f, 0.048, ang_f_long, 0.0, "#1f77b4", lw=1.4, ls="--")
arc(hip_b, 0.034, ang_b_short, 180.0, "#d62728", lw=2.2)
arc(hip_b, 0.048, ang_b_long + 360.0, 180.0, "#d62728", lw=1.4, ls="--")

ax.text(hip_f[0] + 0.030, hip_f[1] + 0.045, "m_LF", fontsize=11, color="#1f77b4")
ax.text(hip_f[0] + 0.030, hip_f[1] + 0.028, f"最短 {m_f_short:+.1f}°", fontsize=8.5, color="#1f77b4")
ax.text(hip_b[0] - 0.050, hip_b[1] + 0.055, "m_LB", fontsize=11, color="#d62728", ha="right")
ax.text(hip_b[0] - 0.050, hip_b[1] + 0.038, f"最短 {m_b_short:+.1f}°", fontsize=8.5, color="#d62728", ha="right")
ax.annotate(f"最长 {m_f_long:+.1f}° (m<0)", xy=(hip_f[0] + 0.048 * math.cos(math.radians(ang_f_long)),
                                 hip_f[1] + 0.048 * math.sin(math.radians(ang_f_long))),
            xytext=(hip_f[0] + 0.075, hip_f[1] - 0.075), fontsize=8.5, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.9))
ax.annotate(f"最长 {m_b_long:+.1f}° (m<0)", xy=(hip_b[0] + 0.048 * math.cos(math.radians(ang_b_long)),
                                 hip_b[1] + 0.048 * math.sin(math.radians(ang_b_long))),
            xytext=(hip_b[0] - 0.075, hip_b[1] - 0.075), fontsize=8.5, color="#d62728", ha="right",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.9))

# ---- 图例 ----
ax.plot([], [], color="#111111", lw=3.4, label="最短姿态（收腿，m>0）")
ax.plot([], [], color="#8a8a8a", lw=2.2, ls="--", label="最长姿态（伸展 q=0.45 rad，m<0）")
ax.plot([], [], color="#dddddd", lw=7.0, label="中间连杆（角度参考）")
ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

# ---- 说明框（图外下方） ----
box = (
    "电机角 m = 中间连杆与大腿(ham)连杆的夹角，逆时针为正；腿最短时 m>0，腿最长时 m<0。\n"
    "与仿真关节角 q 的关系（q=0 = CAD 装配位 = 收腿端）：\n"
    "    前髋：m = +10.148° − q_front          后髋：m = +10.148° + q_back\n"
    f"    按图姿态计算：最短 m_LF={m_f_short:+.2f}°  m_LB={m_b_short:+.2f}°；"
    f"最长 m_LF={m_f_long:+.2f}°  m_LB={m_b_long:+.2f}°\n"
    "网络输入不用 m 本身，而是关节角相对 squat 的偏差：hip_pos_rel = q − q_squat\n"
    "    （换算：前髋 −Δm、后髋 +Δm，符号以你的电机驱动方向为准）"
)
fig.text(0.02, 0.012, box, fontsize=9.0, va="bottom",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.95, edgecolor="#bbbbbb"))

ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-0.28, 0.20)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x [m]  (forward = +x, base_link frame)")
ax.set_ylabel("z [m]")
ax.set_title(
    "电机角 m 是哪一个？ 中间连杆与大腿连杆的夹角，逆时针为正\n"
    "positive at shortest leg, negative at longest leg",
    fontsize=11,
)
ax.grid(True, alpha=0.25)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout(rect=(0, 0.20, 1, 1))
fig.savefig(OUT_PATH, dpi=160)
print(f"saved: {OUT_PATH}")
print(
    f"shortest: m_LF={m_f_short:+.2f} m_LB={m_b_short:+.2f} | "
    f"longest: m_LF={m_f_long:+.2f} m_LB={m_b_long:+.2f}"
)
