"""生成"网络输入里的髋角是哪一个"示意图：docs/figures/obs_hip_angles.png。

要点：
- 策略观测第 [9:13] 维 = hip_pos_rel = 4 个 hip 关节角**相对 squat 姿态的偏差**（rad）；
- 顺序：LF / LB / RF / RB（left_front, left_back, right_front, right_back）；
- 图中在左腿侧视（base_link 坐标系）上标出 LF/LB 两个髋关节、对应大腿（ham）连杆、
  squat 参考姿态与伸出姿态，以及网络输入的 Δq 弧。
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch

from chassis_rl_lab.kinematics import FiveBarSolver, load_squat_pose

OUT_PATH = PROJECT_ROOT / "docs" / "figures" / "obs_hip_angles.png"


def leg_pose(solver, q_front: float, q_back: float):
    """返回左腿姿态（base 系）：髋轴、膝点、轮轴、base 高度。"""
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


def plane_angle(vec) -> float:
    """平面角约定：phi = atan2(-z, x)（x 向前、z 向上，向下为正）。"""
    return math.atan2(-vec[1], vec[0])


def arc_points(center, radius, phi_div: float, phi0: float, n=60):
    phis = np.linspace(phi_div, phi0, n)
    return center[0] + radius * np.cos(phis), center[1] - radius * np.sin(phis)


def main() -> None:
    solver = FiveBarSolver(device="cpu")
    squat = load_squat_pose()
    q_f_sq = float(squat["hips"]["left_front_hip"])
    q_b_sq = float(squat["hips"]["left_back_hip"])

    pose_sq = leg_pose(solver, q_f_sq, q_b_sq)  # squat 参考
    q_ext = 0.45
    pose_ext = leg_pose(solver, q_ext, -q_ext)  # 伸出姿态

    fig, ax = plt.subplots(figsize=(12, 6.8))

    # base_link 机架
    for pose, color, lw, ls in (
        (pose_sq, "#888888", 5.0, "--"),
        (pose_ext, "#1f77b4", 5.0, "-"),
    ):
        ax.plot(
            [pose["hip_b"][0], pose["hip_f"][0]],
            [pose["hip_b"][1], pose["hip_f"][1]],
            color=color,
            lw=lw,
            ls=ls,
            alpha=0.6,
        )

    # 连杆：squat（灰虚线）与伸出（蓝实线）；大腿用粗线突出（网络输入的角度载体）
    for pose, color, lw, ls, thigh_lw in (
        (pose_sq, "#888888", 2.0, "--", 4.5),
        (pose_ext, "#1f77b4", 2.6, "-", 5.5),
    ):
        ax.plot([pose["hip_f"][0], pose["knee_f"][0]], [pose["hip_f"][1], pose["knee_f"][1]],
                color=color, lw=thigh_lw, ls=ls)  # 前大腿 ham_f
        ax.plot([pose["hip_b"][0], pose["knee_b"][0]], [pose["hip_b"][1], pose["knee_b"][1]],
                color="#d62728" if pose is pose_ext else color, lw=thigh_lw, ls=ls)  # 后大腿 ham_b
        ax.plot([pose["knee_f"][0], pose["axle"][0]], [pose["knee_f"][1], pose["axle"][1]],
                color=color, lw=lw, ls=ls)
        ax.plot([pose["knee_b"][0], pose["axle"][0]], [pose["knee_b"][1], pose["axle"][1]],
                color=color, lw=lw, ls=ls)
        ax.add_patch(plt.Circle(pose["axle"], solver.wheel_radius, fill=False, color=color, lw=lw, ls=ls))
        ax.plot([pose["axle"][0]], [pose["axle"][1]], marker="o", color=color, ms=5)

    for pose, color, label_y in ((pose_sq, "#888888", 0.068), (pose_ext, "#1f77b4", 0.088)):
        ax.plot([pose["hip_f"][0]], [pose["hip_f"][1]], marker="o", color=color, ms=9)
        ax.plot([pose["hip_b"][0]], [pose["hip_b"][1]], marker="o", color="#d62728" if pose is pose_ext else color, ms=9)

    # ---- 网络输入的 Δq 弧：从 squat 大腿方向转到伸出大腿方向 ----
    for name, hip, knee_sq, knee_ext, color, label, off in (
        ("LF", pose_sq["hip_f"], pose_sq["knee_f"], pose_ext["knee_f"], "#1f77b4",
         "Δq_LF → hip_pos_rel[0] (obs[9])", (0.035, -0.155)),
        ("LB", pose_sq["hip_b"], pose_sq["knee_b"], pose_ext["knee_b"], "#d62728",
         "Δq_LB → hip_pos_rel[1] (obs[10])", (-0.235, -0.105)),
    ):
        v_sq = (knee_sq[0] - hip[0], knee_sq[1] - hip[1])
        v_ex = (knee_ext[0] - hip[0], knee_ext[1] - hip[1])
        phi_sq, phi_ex = plane_angle(v_sq), plane_angle(v_ex)
        if phi_sq - phi_ex > math.pi:
            phi_sq -= 2 * math.pi
        if phi_ex - phi_sq > math.pi:
            phi_ex -= 2 * math.pi
        r = 0.026
        xs, zs = arc_points(hip, r, phi_sq, phi_ex)
        ax.plot(xs, zs, color=color, lw=2.0)
        ax.annotate("", xy=(xs[-1], zs[-1]), xytext=(xs[-2], zs[-2]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0))
        tx, tz = hip[0] + off[0], hip[1] + off[1]
        ax.annotate(label, xy=(xs[len(xs) // 2], zs[len(zs) // 2]), xytext=(tx, tz),
                    fontsize=9, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.9))

    ax.text(pose_sq["hip_f"][0] + 0.012, pose_sq["hip_f"][1] + 0.026,
            "LF hip joint\n(left_front_hip)", fontsize=9, color="#1f77b4")
    ax.text(pose_sq["hip_b"][0] - 0.012, pose_sq["hip_b"][1] + 0.026,
            "LB hip joint\n(left_back_hip)", fontsize=9, color="#d62728", ha="right")

    # squat / extended 图例
    ax.plot([], [], color="#888888", lw=2.0, ls="--", label="squat pose (reference)")
    ax.plot([], [], color="#1f77b4", lw=2.6, label="extended pose q=0.45 rad")
    ax.plot([], [], color="#d62728", lw=4.0, label="back thigh (ham_back)")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # 观测向量说明框
    box = (
        "Policy obs [9:13] = hip_pos_rel  (4 dims, rad, order LF/LB/RF/RB)\n"
        "  = [q_LF − q_LF,squat ,  q_LB − q_LB,squat ,  q_RF − q_RF,squat ,  q_RB − q_RB,squat]\n"
        "  ≈ 电机角(now) − 电机角(squat)  （零点/零位在相减中抵消；单位 rad）\n"
        "Squat 参考（图中灰虚线）：q_LF,squat = +0.035 rad, q_LB,squat = −0.035 rad\n"
        "右腿对称：对称站姿下 q_RF = q_LF, q_RB = q_LB（网络仍用 4 个独立值）\n"
        "索引：LF→obs[9], LB→obs[10], RF→obs[11], RB→obs[12]（rad，未缩放）"
    )
    ax.text(-0.245, 0.175, box, fontsize=8.6, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="#bbbbbb"))

    # 地面参考
    ax.axhline(-pose_ext["height"], color="#555555", lw=0.8, ls=":")
    ax.text(0.245, -pose_ext["height"] - 0.012, "ground (extended pose)", fontsize=8, ha="right", color="#555555")

    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.27, 0.20)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]  (forward = +x, base_link frame)")
    ax.set_ylabel("z [m]")
    ax.set_title(
        "Which hip angles go into the network?  →  hip_pos_rel (relative to squat pose)\n"
        "side view, left leg; base_link x-z plane; joint axis +y is perpendicular to the figure",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=160)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
