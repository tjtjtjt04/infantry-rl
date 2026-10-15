"""Measured leg geometry + mobility evidence for the infantry-rl URDFs.

Part 1 (measurement) reports the five-bar link lengths and how far each calf's
distal end sits from the wheel axle - this is the evidence behind the loop-joint
parameters hard-coded in build_closed_urdf.py.

Part 2 (mobility) shows that the leg has a single independent motion no matter
where on the carrier the rear calf's pin sits: the closure is 3 equations in the
2 knee angles, so its Jacobian has rank 2 and both knee angles are pinned.

Compare with FiveBar/five_bar.c, which treats the two hip motors as independent
inputs (a 2-DOF iso-five-bar).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinematics import Robot, rpy_to_R, axis_rot  # noqa

URDF_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "infantry-rl", "urdf"))
H = 1e-6


def measure(rob, side):
    """Link lengths + how each calf's distal end sits relative to the wheel axle.

    radial = distance in the leg plane from the axle centre to the closest point
    of the calf mesh;  lateral = position of that point along the axle (positive
    = inboard, towards the robot centre line).
    """
    q0 = {n: 0.0 for n in rob.joints}
    A = rob.joints[f"{side}_front_hip"]["xyz"]
    E = rob.joints[f"{side}_back_hip"]["xyz"]
    l1 = np.linalg.norm(rob.joints[f"{side}_front_knee"]["xyz"])
    l4 = np.linalg.norm(rob.joints[f"{side}_back_knee"]["xyz"])
    jw = rob.joints[f"{side}_wheel"]
    Rp, pp = rob.pose(jw["parent"], q0)
    axle = pp + Rp @ jw["xyz"]
    axw = Rp @ (jw["axis"] / np.linalg.norm(jw["axis"]))
    Rf, pf = rob.pose(f"{side}_front_calf", q0)
    Rb, pb = rob.pose(f"{side}_back_calf", q0)
    l2 = np.linalg.norm(axle - pf)
    l3 = np.linalg.norm(axle - pb)
    print(f"  {side}: l1={l1:.5f}  l2={l2:.5f}  l3={l3:.5f}  l4={l4:.5f}  "
          f"base={np.linalg.norm(E - A):.5f} m")
    for name in (f"{side}_front_calf", f"{side}_back_calf"):
        R, p = rob.pose(name, q0)
        v = rob.mesh(name).reshape(-1, 3) @ R.T + p
        rel = v - axle
        along = rel @ axw
        perp = np.linalg.norm(rel - np.outer(along, axw), axis=1)
        k = int(np.argmin(perp))
        print(f"      {name:18s} closest mesh point -> axle: radial {perp[k]*1000:5.2f} mm, "
              f"lateral {along[k]*1000:+6.2f} mm")


def residual(rob, side, qf, qb, qkf, qkb, pin_local):
    """front-calf axle point  minus  pin point seen through the rear calf."""
    q = {n: 0.0 for n in rob.joints}
    q[f"{side}_front_hip"] = qf
    q[f"{side}_back_hip"] = qb
    q[f"{side}_front_knee"] = qkf
    q[f"{side}_back_knee"] = qkb
    j = rob.joints[f"{side}_back_calf_axle"]
    jw = rob.joints[f"{side}_wheel"]
    Rp, pp = rob.pose(jw["parent"], q)
    front = pp + Rp @ jw["xyz"]
    Rc, pc = rob.pose(j["child"], q)
    return front - (pc + Rc @ pin_local)


def main():
    rob = Robot(os.path.join(URDF_DIR, "infantry-rl-closed-fixed.urdf"))

    print("=" * 74)
    print("1. measured leg geometry (this is the evidence for the loop joints)")
    for side in ("left", "right"):
        measure(rob, side)

    side = "left"
    q0 = {n: 0.0 for n in rob.joints}
    j = rob.joints[f"{side}_back_calf_axle"]
    Rp, pp = rob.pose(j["parent"], q0)
    axle = pp + Rp @ j["xyz"]
    Rc, pc = rob.pose(j["child"], q0)
    pin_home = Rc.T @ (axle - pc)          # (0,0,0) by construction
    axw = Rp @ (j["axis"] / np.linalg.norm(j["axis"]))
    ex = np.array([1.0, 0, 0.0])
    ex = ex - (ex @ axw) * axw
    ex /= np.linalg.norm(ex)

    print()
    print("=" * 74)
    print("2. mobility vs. where the rear calf's pin sits on the carrier")
    print("   offset(mm)   rank of d(residual)/d(knee_f, knee_b)")
    for dperp in (0.0, 0.005, 0.010, 0.020, 0.030, 0.045):
        pin_local = pin_home + Rc.T @ (dperp * ex)
        rank = 0
        for trial in range(6):
            rng = np.random.default_rng(trial)
            if dperp == 0.0:
                qf = qb = qkf = qkb = 0.0
            else:
                qf = rng.uniform(0.0, 1.0)
                qb = rng.uniform(1.4, 2.6)
                qkf = rng.uniform(-2.8, 2.8)
                qkb = rng.uniform(-2.8, 2.8)
            J = np.zeros((3, 2))
            for k, which in enumerate(("kf", "kb")):
                if which == "kf":
                    rp = residual(rob, side, qf, qb, qkf + H, qkb, pin_local)
                    rm = residual(rob, side, qf, qb, qkf - H, qkb, pin_local)
                else:
                    rp = residual(rob, side, qf, qb, qkf, qkb + H, pin_local)
                    rm = residual(rob, side, qf, qb, qkf, qkb - H, pin_local)
                J[:, k] = (rp - rm) / (2 * H)
            sv = np.linalg.svd(J, compute_uv=False)
            rank = max(rank, int((sv > 1e-6 * max(sv[0], 1e-12)).sum()))
        print(f"     {dperp*1000:6.1f}        {rank}")
    print()
    print("  rank 2 everywhere -> the two knee angles are pinned by the closure, so")
    print("  the leg has ONE independent motion. The firmware's FiveBar_Solve treats")
    print("  the two hip motors as independent inputs (2-DOF iso-five-bar), which")
    print("  does not match this CAD topology. See URDF闭环说明.md section 5.")


if __name__ == "__main__":
    main()
