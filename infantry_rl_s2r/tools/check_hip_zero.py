"""What does hip q = 0 mean? (numpy only, no Kit)

Answers the question with the measured model instead of the URDF comments:

* leg length (hip midpoint -> wheel axle) and base height at ``q_hip = 0``;
* the same at the squat pose the policy holds;
* the closed-chain knee angles at q = 0 and whether they sit inside the URDF
  limits (which is why the squat pose is *not* exactly q = 0);
* the direction of the stroke: which way extends the leg.

Run::

    C:/Users/32560/.conda/envs/isaaclab/python.exe tools/check_hip_zero.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from kinematics.planar_leg import LegPlanar, leg_limits, solve_loop  # noqa: E402
from robot_model import (  # noqa: E402
    URDF_PATH,
    WHEEL_RADIUS,
    hip_axis_height,
    nominal_pose,
    squat_pose,
)


def main() -> int:
    data = squat_pose()
    nominal = nominal_pose()
    limits = leg_limits(URDF_PATH, "left")
    leg = LegPlanar.from_dict(__import__("json").loads((PROJECT_DIR / "robot_data/chassis-rl/planar_leg.json").read_text())["left"])

    print("URDF hip limits (left leg, the right leg is mirrored):")
    for key in ("front_hip", "back_hip", "front_knee", "back_knee"):
        lo, hi = limits[key]
        print(f"  {key:<12} [{lo:+.6f}, {hi:+.6f}] rad")

    print("\nLeg geometry:")
    for label, q1, q4 in (
        ("q = (0, 0)  URDF/CAD zero", 0.0, 0.0),
        ("squat pose  (left)", data["hips"]["left"]["front_hip"], data["hips"]["left"]["back_hip"]),
        ("nominal pose(left)", nominal["left"]["front_hip"], nominal["left"]["back_hip"]),
    ):
        q5, q6, res, _ = solve_loop(leg, q1, q4, seed=(0.0, 0.0), iters=200)
        L0, phi = leg.leg_state(q1, q5)
        base_z = WHEEL_RADIUS + L0 - hip_axis_height()
        k5, k6 = limits["front_knee"], limits["back_knee"]
        inside = (k5[0] + 1e-9 <= q5 <= k5[1] - 1e-9) and (k6[0] + 1e-9 <= q6 <= k6[1] - 1e-9)
        print(
            f"  {label:<26} q=({q1:+.4f}, {q4:+.4f})  knees=({q5:+.4f}, {q6:+.4f})  "
            f"closure={res * 1e6:.2f} um  L0={L0:.5f} m  base_z={base_z:.5f} m  "
            f"knees inside limits: {inside}"
        )

    # which direction extends the leg?  sample the *left* leg along its feasible hip range
    print("\nStroke direction (left leg, knee limits respected, 0.01 rad margin):")
    m = 0.01
    k5, k6 = limits["front_knee"], limits["back_knee"]
    samples = []
    for i in range(21):
        q1 = -0.0117 - 0.06 * i  # move away from zero: does the leg get longer?
        q4 = 0.0118 + 0.06 * i
        if not (limits["front_hip"][0] < q1 < limits["front_hip"][1]):
            break
        if not (limits["back_hip"][0] < q4 < limits["back_hip"][1]):
            break
        q5, q6, res, _ = solve_loop(leg, q1, q4, seed=(0.0, 0.0), iters=200)
        if res > 1e-9 or not (k5[0] + m <= q5 <= k5[1] - m and k6[0] + m <= q6 <= k6[1] - m):
            continue
        L0, phi = leg.leg_state(q1, q5)
        samples.append((q1, q4, L0, WHEEL_RADIUS + L0 - hip_axis_height()))
    for q1, q4, L0, bz in samples[:6]:
        print(f"  hips=({q1:+.4f}, {q4:+.4f}) -> L0={L0:.5f} m, base_z={bz:.5f} m")
    if len(samples) > 6:
        print("  ...")
        for q1, q4, L0, bz in samples[-2:]:
            print(f"  hips=({q1:+.4f}, {q4:+.4f}) -> L0={L0:.5f} m, base_z={bz:.5f} m")

    print(
        "\nSquat pose (the policy's zero action): "
        f"L0={data['leg_length']:.5f} m, base_z={data['base_height']:.5f} m, "
        f"hips L(front {data['hips']['left']['front_hip']:+.6f} / back {data['hips']['left']['back_hip']:+.6f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
