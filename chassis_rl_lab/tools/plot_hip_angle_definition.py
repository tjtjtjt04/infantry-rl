"""生成"髋角定义"示意图（base_link 系侧视 x-z 平面）：docs/figures/hip_angle_definition.png。

图中包含：
- 零位 q=0（CAD 装配位 = 机械收回端）与伸出 q=0.45 rad 两个姿态的连杆叠加；
- 前/后髋电机轴位置与髋角正方向（用两姿态的连杆方向画旋转弧）；
- 轮轴到髋电机轴的平面距离 L0、机体高度 h、轮半径 r 的标注；
- 所有关节约定：绕 +y 轴右手定则正转（左右同号）。
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

OUT_PATH = PROJECT_ROOT / "docs" / "figures" / "hip_angle_definition.png"


def leg_pose(solver, q_ext: float):
    """左腿姿态（base_link 系）：返回髋轴、膝点、轮轴与机体高度。"""
    model = solver.models["left"]
    q_hips = torch.tensor([[q_ext, -q_ext, q_ext, -q_ext]], dtype=torch.float64)
    knees = solver.solve(q_hips)[0][0].tolist()
    knee_f, axle_f = model.fk_front(q_ext, knees[0])
    knee_b, axle_b = model.fk_back(-q_ext, knees[1])
    return {
        "hip_f": tuple(model.hip_front),
        "hip_b": tuple(model.hip_back),
        "knee_f": knee_f,
        "knee_b": knee_b,
        "axle": ((axle_f[0] + axle_b[0]) / 2.0, (axle_f[1] + axle_b[1]) / 2.0),
        "height": float(solver.base_height(q_hips)[0]),
    }


def plane_angle(vec) -> float:
    """平面角约定 phi = atan2(-z, x)（x 向前、z 向上；向下为正）。"""
    return math.atan2(-vec[1], vec[0])


def arc_points(center, radius, phi0, phi1, n=60):
    """按 phi 约定（x=r cos phi, z=-r sin phi）生成圆弧点。"""
    phis = np.linspace(phi0, phi1, n)
    return center[0] + radius * np.cos(phis), center[1] - radius * np.sin(phis)


def main() -> None:
    solver = FiveBarSolver(device="cpu")
    r = solver.wheel_radius
    pose0 = leg_pose(solver, 0.0)
    poseq = leg_pose(solver, 0.45)

    fig, ax = plt.subplots(figsize=(12, 6.8))
    colors = {0.0: "#7f8c8d", 0.45: "#1f77b4"}

    # 地面（base 系：地面在 z = -h）
    ax.axhline(-pose0["height"], color="black", lw=1.2, ls="-")
    ax.text(-0.245, -pose0["height"] + 0.006, "ground at q=0", ha="left", fontsize=9)
    ax.axhline(-poseq["height"], color="#1f77b4", lw=1.0, ls=":")
    ax.text(0.245, -poseq["height"] - 0.016, "ground at q=0.45", ha="right", fontsize=9, color="#1f77b4")

    for q_ext, pose in ((0.0, pose0), (0.45, poseq)):
        color = colors[q_ext]
        lw = 2.2 if q_ext == 0.0 else 3.0
        ls = "--" if q_ext == 0.0 else "-"
        hip_f, hip_b = pose["hip_f"], pose["hip_b"]
        knee_f, knee_b = pose["knee_f"], pose["knee_b"]
        axle = pose["axle"]

        # base_link 机架（两髋轴之间）
        ax.plot([hip_f[0], hip_b[0]], [hip_f[1], hip_b[1]], color=color, lw=5.0, ls=ls, alpha=0.6)
        # 大腿（ham）与小腿（calf）
        for a, b in (
            (hip_f, knee_f),
            (hip_b, knee_b),
            (knee_f, axle),
            (knee_b, axle),
        ):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls)
        ax.add_patch(plt.Circle(axle, r, fill=False, color=color, lw=lw, ls=ls))
        ax.plot([axle[0]], [axle[1]], marker="o", color=color, ms=6)
        ax.plot([hip_f[0], hip_b[0]], [hip_f[1], hip_b[1]], marker="o", color=color, ms=8)

        if q_ext == 0.0:
            ax.annotate(
                "q = 0 : CAD assembly = fully retracted",
                xy=(0.5 * (knee_f[0] + pose["axle"][0]), 0.5 * (knee_f[1] + pose["axle"][1])),
                xytext=(-0.245, 0.075),
                fontsize=9,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            )
        else:
            ax.annotate(
                "q = 0.45 rad : extended (h increases)",
                xy=(axle[0] + 0.7 * r, axle[1] + 0.7 * r),
                xytext=(-0.245, -0.235),
                fontsize=9,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
            )
            # L0：前髋轴 -> 轮轴 的平面距离
            ax.plot([hip_f[0], axle[0]], [hip_f[1], axle[1]], ls="-.", color="#d62728", lw=1.6)
            mid = ((hip_f[0] + axle[0]) / 2, (hip_f[1] + axle[1]) / 2)
            ax.text(mid[0] + 0.012, mid[1] + 0.004, "L0 (hip axis → axle)", color="#d62728", fontsize=9)
            # h：base 原点 -> 地面
            ax.annotate(
                "", xy=(0.145, -poseq["height"]), xytext=(0.145, 0.0),
                arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=1.2),
            )
            ax.text(0.149, -poseq["height"] / 2, "h = 0.240 m", color="#2ca02c", fontsize=10, va="center")
            # r：轮半径
            ax.annotate(
                "", xy=(axle[0] + r + 0.012, axle[1]), xytext=(axle[0] + r + 0.012, axle[1] - r),
                arrowprops=dict(arrowstyle="<->", color="#8c564b", lw=1.2),
            )
            ax.text(axle[0] + r + 0.016, axle[1] - r / 2, "r", color="#8c564b", fontsize=11)

    # ---- 髋角正方向弧（用两姿态连杆方向计算，保证方向正确） ----
    for name, hip, knee0, kneeq, label in (
        ("front", pose0["hip_f"], pose0["knee_f"], poseq["knee_f"], "+q_front (extends)"),
        ("back", pose0["hip_b"], pose0["knee_b"], poseq["knee_b"], "-q_back (extends)"),
    ):
        v0 = (knee0[0] - hip[0], knee0[1] - hip[1])
        vq = (kneeq[0] - hip[0], kneeq[1] - hip[1])
        phi0, phiq = plane_angle(v0), plane_angle(vq)
        if phi0 - phiq > math.pi:  # 处理 ±pi 回绕
            phi0 -= 2 * math.pi
        if phiq - phi0 > math.pi:
            phiq -= 2 * math.pi
        radius = 0.030
        xs, zs = arc_points(hip, radius, phi0, phiq)
        ax.plot(xs, zs, color="#1f77b4", lw=1.8)
        # 箭头（沿弧方向）
        ax.annotate(
            "",
            xy=(xs[-1], zs[-1]),
            xytext=(xs[-2], zs[-2]),
            arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=1.8),
        )
        mid_phi = phi0 + 0.5 * (phiq - phi0) + 0.35
        tx = hip[0] + (radius + 0.045) * np.cos(mid_phi)
        tz = hip[1] - (radius + 0.045) * np.sin(mid_phi)
        if name == "front":
            tx, tz = hip[0] + 0.10, hip[1] - 0.085
        else:
            tx, tz = hip[0] - 0.155, hip[1] - 0.055
        ax.annotate(label, xy=(xs[len(xs) // 2], zs[len(zs) // 2]), xytext=(tx, tz),
                    fontsize=9, color="#1f77b4", arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.9))

    ax.text(pose0["hip_f"][0] + 0.02, pose0["hip_f"][1] + 0.028, "front hip axis", fontsize=9, ha="left")
    ax.text(pose0["hip_b"][0] - 0.02, pose0["hip_b"][1] + 0.028, "back hip axis", fontsize=9, ha="right")

    # base_link 水平参考（x 方向，过髋轴高度）
    ax.axhline(pose0["hip_f"][1], color="#555555", lw=0.9, ls=":", alpha=0.8)
    ax.text(0.245, pose0["hip_f"][1] + 0.004, "base_link horizontal (x)", fontsize=8.5, ha="right", color="#555555")

    # 连杆相对机体水平方向的绝对角 φ（LQR / 电机返回角习惯）
    phi_front_0 = math.degrees(
        plane_angle((pose0["knee_f"][0] - pose0["hip_f"][0], pose0["knee_f"][1] - pose0["hip_f"][1]))
    )
    phi_back_0 = math.degrees(
        plane_angle((pose0["knee_b"][0] - pose0["hip_b"][0], pose0["knee_b"][1] - pose0["hip_b"][1]))
    )
    phi_front_q = math.degrees(
        plane_angle((poseq["knee_f"][0] - poseq["hip_f"][0], poseq["knee_f"][1] - poseq["hip_f"][1]))
    )
    phi_back_q = math.degrees(
        plane_angle((poseq["knee_b"][0] - poseq["hip_b"][0], poseq["knee_b"][1] - poseq["hip_b"][1]))
    )
    ax.text(
        -0.245,
        0.155,
        "Link angle φ to base_link horizontal (ham link):\n"
        f"  q=0     : front {phi_front_0:+.2f}°   back {phi_back_0:+.2f}°\n"
        f"  q=0.45  : front {phi_front_q:+.2f}°   back {phi_back_q:+.2f}°",
        fontsize=8.6,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#bbbbbb"),
    )

    legend = (
        "Convention: all joints positive = right-hand rotation about +y (left/right same sign).\n"
        "q_front = +q, q_back = -q for symmetric leg extension; q=0 at CAD assembly (retracted).\n"
        "Base_link frame shown: +x = forward, +z = up; joint axes (+y) perpendicular to figure.\n"
        "q = joint angle (zero at CAD assembly); φ = link angle to base_link horizontal (= q + zero offset).\n"
        "φ convention: 0° = horizontal forward (+x); positive = downward (atan2(-z, x))."
    )
    ax.text(-0.245, -0.320, legend, fontsize=8.6, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#bbbbbb"))

    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.33, 0.19)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]  (forward = +x, base_link frame)")
    ax.set_ylabel("z [m]")
    ax.set_title(
        "Hip joint angle definition (left leg, side view, base_link x-z plane)\n"
        "grey dashed: q=0 (retracted)   |   blue: q=0.45 rad (extended)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=160)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
