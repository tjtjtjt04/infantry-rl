#!/usr/bin/env python
"""Measure ``robot_data/chassis-rl`` from the new ``chassis-rl`` URDF.

The training stack does not read the URDF at runtime.  It uses three measured
JSON files:

* ``planar_leg.json``  -- the two planar five-bar chains per side (fitted to the
  exact URDF forward kinematics by :class:`kinematics.planar_leg.LegPlanar`);
* ``nominal_pose.json`` -- a mid-stroke stance (same leg length as the original
  robot, 0.24977 m) plus every joint limit;
* ``squat_pose.json``  -- the DEEPEST feasible locked-hip pose: the hip angles
  that minimise the leg length while every solved knee stays inside its URDF
  limit with the action term's 0.01 rad safety margin.

Why the deepest pose is near q = 0
----------------------------------
``chassis-rl`` re-references the joint zeros to the retracted end of the
mechanical stroke (``q = 0`` lies 1e-4 rad inside every range).  The base cannot
be lowered by driving the hips to +/-1.49 rad: that folds the linkage past the
knee limits and the closed chain cannot close.  Dense scanning shows the minimum
leg length is 0.13587 m at
``front_hip = -0.0221``, ``back_hip = +0.0099`` (left leg), which is the
retracted end of the stroke -- exactly the "hips at the limit" pose the robot can
actually hold.  Pushing to the raw hip stops only buys ~1.8 mm and clips the knees.

Run with the Isaac Lab Python (numpy only; no Kit):

    C:/Users/32560/.conda/envs/isaaclab/python.exe tools/measure_chassis_asset.py

Then convert the USD and train with ``INFANTRY_RL_ASSET=chassis``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402

from kinematics.planar_leg import (  # noqa: E402
    LegPlanar,
    leg_limits,
    neutral_pose,
    solve_loop,
)
from kinematics.urdf_tree import UrdfTree  # noqa: E402

URDF = PROJECT_DIR / "chassis-rl" / "urdf" / "chassis-rl.urdf"
OUT_DIR = PROJECT_DIR / "robot_data" / "chassis-rl"

#: measured from ``meshes/*_wheel.STL`` (same 136 mm wheel as the original robot)
WHEEL_RADIUS = 0.068
#: the action term keeps every command this far inside the URDF limits
LIMIT_MARGIN = 0.01
#: nominal stance: same leg length as the original robot, vertical leg
NOMINAL_LEG_LENGTH = 0.24977467318622631
NOMINAL_PHI = 0.0
#: deepest-pose termination / attitude limits
BASE_SAG = 0.04
PITCH_LIMIT = 0.24
ROLL_LIMIT = 0.30

#: chassis-rl: the wheel JOINTS are wheel_left/wheel_right, the LINKS are left_wheel/right_wheel
JOINT_NAMES = {
    "left": {
        "front_hip": "left_front_hip",
        "back_hip": "left_back_hip",
        "front_knee": "left_front_knee",
        "back_knee": "left_back_knee",
        "wheel": "wheel_left",
    },
    "right": {
        "front_hip": "right_front_hip",
        "back_hip": "right_back_hip",
        "front_knee": "right_front_knee",
        "back_knee": "right_back_knee",
        "wheel": "wheel_right",
    },
}


def fit_legs(tree: UrdfTree) -> dict[str, LegPlanar]:
    legs = {}
    for side in ("left", "right"):
        leg, _front, _back = LegPlanar.from_urdf(
            tree, side, verbose=True, joint_names=JOINT_NAMES[side]
        )
        legs[side] = leg
    return legs


def _scan_window(leg: LegPlanar, limits: dict, q1s, q4s, best):
    """Scan a rectangular window and return the best feasible (min L0) pose."""
    m = LIMIT_MARGIN
    q5lo, q5hi = limits["front_knee"]
    q6lo, q6hi = limits["back_knee"]
    for q1 in q1s:
        seed = (0.0, 0.0)
        for q4 in q4s:
            q5, q6, res, _ = solve_loop(leg, float(q1), float(q4), seed=seed, iters=50)
            if res > 1e-9:
                continue
            if not (q5lo + m <= q5 <= q5hi - m and q6lo + m <= q6 <= q6hi - m):
                continue
            seed = (q5, q6)
            L0, phi = leg.leg_state(float(q1), q5)
            if best is None or L0 < best[0]:
                best = (L0, float(q1), float(q4), q5, q6, phi)
    return best


def scan_deepest(leg: LegPlanar, limits: dict, n: int = 241):
    """Return the feasible pose with the smallest leg length.

    Both hips and both knees are kept ``LIMIT_MARGIN`` inside their URDF limits.
    A full-range coarse scan is followed by a local refinement, because the
    optimum sits on the knee-limit boundary and a 6 mrad grid is ~0.1 mm high.
    The seed is carried along each ``q1`` column so the solver stays on the
    assembled branch.
    """
    m = LIMIT_MARGIN
    q1lo, q1hi = limits["front_hip"]
    q4lo, q4hi = limits["back_hip"]
    q5lo, q5hi = limits["front_knee"]
    q6lo, q6hi = limits["back_knee"]

    best = _scan_window(
        leg,
        limits,
        np.linspace(q1lo + m, q1hi - m, n),
        np.linspace(q4lo + m, q4hi - m, n),
        None,
    )
    if best is None:
        raise RuntimeError("no feasible closed configuration found inside the joint limits")

    # local refinement around the coarse optimum (+/-4 cm in each hip)
    for radius, samples in ((0.04, 161), (0.008, 161)):
        best = _scan_window(
            leg,
            limits,
            np.linspace(best[1] - radius, best[1] + radius, samples),
            np.linspace(best[2] - radius, best[2] + radius, samples),
            best,
        )
    return best


def mirror_pose(leg: LegPlanar, pose: tuple[float, float, float, float]) -> tuple[float, float]:
    """Mirror a left-side pose onto the right leg; returns ``(sign, mirrored_pose)``.

    ``chassis-rl`` mirrors its right-leg joint axes, so the same physical pose is
    ``q_right = -q_left`` there; the original robot uses ``q_right = +q_left``.
    Pick whichever of the two candidates closes the loop.
    """
    candidates = []
    for sign in (1.0, -1.0):
        q = tuple(sign * v for v in pose)
        res = math.hypot(*leg.closure(*q))
        candidates.append((res, sign, q))
    res, sign, q = min(candidates, key=lambda item: item[0])
    if res > 1e-6:
        raise RuntimeError(f"mirrored pose does not close (residual {res:.3e} m)")
    return sign, q  # type: ignore[return-value]


def pose_entry(leg: LegPlanar, pose: tuple[float, float, float, float], limits: dict) -> dict:
    L0, phi = leg.leg_state(pose[0], pose[2])
    residual = math.hypot(*leg.closure(*pose))
    return {
        "L0": L0,
        "phi_deg": math.degrees(phi),
        "residual": residual,
        "front_hip": pose[0],
        "back_hip": pose[1],
        "front_knee": pose[2],
        "back_knee": pose[3],
        "limits": {key: [float(lo), float(hi)] for key, (lo, hi) in limits.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=URDF)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--scan", type=int, default=241, help="samples per hip axis")
    parser.add_argument("--dry-run", action="store_true", help="print without writing")
    args = parser.parse_args()

    if not args.urdf.is_file():
        raise FileNotFoundError(f"URDF not found: {args.urdf}")
    tree = UrdfTree(args.urdf)
    legs = fit_legs(tree)
    limits = {side: leg_limits(args.urdf, side, JOINT_NAMES[side]) for side in ("left", "right")}

    # ---- nominal stance (same leg length as the original robot) ---------- #
    q1, q4, q5, q6, res, L0, phi = neutral_pose(
        legs["left"],
        leg_length=NOMINAL_LEG_LENGTH,
        phi=NOMINAL_PHI,
        n_scan=args.scan,
        limits=limits["left"],
    )
    nominal_left = (q1, q4, q5, q6)
    right_sign, nominal_right = mirror_pose(legs["right"], nominal_left)
    print(
        f"[nominal] left  q=({q1:+.6f}, {q4:+.6f}, {q5:+.6f}, {q6:+.6f}) "
        f"L0={L0:.6f} phi={phi:+.6f} res={res:.2e}"
    )
    print(f"[nominal] right sign={right_sign:+.0f} q={tuple(round(v, 6) for v in nominal_right)}")

    # ---- deepest feasible closed-chain pose ------------------------------- #
    best = scan_deepest(legs["left"], limits["left"], n=args.scan)
    L0_min, sq1, sq4, sq5, sq6, sphi = best
    squat_left = (sq1, sq4, sq5, sq6)
    right_sign_s, squat_right = mirror_pose(legs["right"], squat_left)
    if right_sign_s != right_sign:
        raise RuntimeError("left/right mirror sign disagrees between nominal and squat poses")
    hip_axis_z = legs["left"].hip_midpoint()[1]
    base_height = WHEEL_RADIUS + L0_min - hip_axis_z
    base_floor = base_height - BASE_SAG
    print(
        f"[deepest] left  q=({sq1:+.6f}, {sq4:+.6f}, {sq5:+.6f}, {sq6:+.6f}) "
        f"L0={L0_min:.6f} phi={sphi:+.6f}"
    )
    print(f"[deepest] right q={tuple(round(v, 6) for v in squat_right)}")
    print(
        f"[deepest] base_z={base_height:.5f} m (nominal {WHEEL_RADIUS + NOMINAL_LEG_LENGTH - hip_axis_z:.5f}), "
        f"base floor={base_floor:.5f} m, hip-axis z={hip_axis_z:.5f} m"
    )

    data = {
        "planar_leg": {side: legs[side].to_dict() for side in ("left", "right")},
        "nominal_pose": {
            "left": pose_entry(legs["left"], nominal_left, limits["left"]),
            "right": pose_entry(legs["right"], nominal_right, limits["right"]),
        },
        "squat_pose": {
            "hips": {
                "left": {"front_hip": sq1, "back_hip": sq4},
                "right": {"front_hip": squat_right[0], "back_hip": squat_right[1]},
            },
            "right_hip_sign": right_sign_s,
            "leg_length": L0_min,
            "base_height": base_height,
            "base_floor": base_floor,
            "pitch_limit": PITCH_LIMIT,
            "roll_limit": ROLL_LIMIT,
            "knee_margin": LIMIT_MARGIN,
            "knees": {
                "left": {"front_knee": sq5, "back_knee": sq6},
                "right": {"front_knee": squat_right[2], "back_knee": squat_right[3]},
            },
        },
    }

    if args.dry_run:
        print(json.dumps(data, indent=2))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for key, name in (
        ("planar_leg", "planar_leg.json"),
        ("nominal_pose", "nominal_pose.json"),
        ("squat_pose", "squat_pose.json"),
    ):
        path = args.out / name
        path.write_text(json.dumps(data[key], indent=2) + "\n", encoding="utf-8")
        print(f"[write] {path}")
    print(
        "\n[note] use INFANTRY_RL_ASSET=chassis so robot_model reads "
        f"{args.out} and {PROJECT_DIR / 'chassis-rl' / 'urdf' / 'chassis-rl' / 'chassis-rl.usd'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
