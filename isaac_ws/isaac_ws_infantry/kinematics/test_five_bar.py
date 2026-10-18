"""Tests for the five-bar model, the loop-closure solver and the neutral pose.

Every check compares against the EXACT 3-D URDF forward kinematics, so a pass means
the planar reduction is faithful.

Run:  python isaac_ws/isaac_ws_infantry/kinematics/test_five_bar.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent  # isaac_ws
URDF = WS / "assets" / "infantry-rl" / "urdf" / "infantry-rl-sim.urdf"
sys.path.insert(0, str(HERE))

from five_bar import (  # noqa: E402
    PlanarLeg,
    UrdfTree,
    leg_state_np,
    neutral_pose_np,
    solve_loop_np,
)

FAILURES: list[str] = []
N_RANDOM = 3000


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    tree = UrdfTree(URDF)
    j = tree.joints
    print(f"URDF   : {URDF}")
    print(f"joints : {len(tree.joints)}   loops: {[lp['name'] for lp in tree.loops]}\n")

    legs = {s: PlanarLeg.from_urdf(tree, s) for s in ("left", "right")}
    for side, leg in legs.items():
        print(f"--- {side} leg planar constants (X-Z plane of the leg, +X fwd, +Z up) ---")
        print(f"    A = ({leg.A[0]:+.6f}, {leg.A[1]:+.6f})    B = ({leg.B[0]:+.6f}, {leg.B[1]:+.6f})")
        for nm, v, ang in (
            ("p  (A -> knee)", leg.p, leg.a_fh),
            ("r  (knee -> axle)", leg.r, leg.a_fc),
            ("s  (B -> knee)", leg.s, leg.b_bh),
            ("t  (knee -> anchor)", leg.t, leg.b_bc),
        ):
            print(
                f"    {nm:<20} = ({v[0]:+.6f}, {v[1]:+.6f})  len={math.hypot(*v):.6f}"
                f"  angle0={math.degrees(ang):+9.3f} deg"
            )
        print(
            f"    joint signs: front_hip={leg.s_fh:+.0f} front_knee={leg.s_fk:+.0f}"
            f" back_hip={leg.s_bh:+.0f} back_knee={leg.s_bk:+.0f}"
        )
        print(f"    body-Y plane: front chain {leg.y_front:+.6f}, back chain {leg.y_back:+.6f}")

    # ------------------------------------------------------------------ 1
    print("\n[1] planar front chain vs exact 3-D URDF FK  (X and Z)")
    rng = np.random.default_rng(7)
    for side, leg in legs.items():
        worstx = worstz = 0.0
        for _ in range(N_RANDOM):
            q1 = float(rng.uniform(j[f"{side}_front_hip"]["lower"], j[f"{side}_front_hip"]["upper"]))
            q5 = float(rng.uniform(j[f"{side}_front_knee"]["lower"], j[f"{side}_front_knee"]["upper"]))
            fr = tree.solve_frames({f"{side}_front_hip": q1, f"{side}_front_knee": q5})
            W3 = fr[j[f"{side}_wheel"]["child"]][:3, 3]
            _, planar = leg.front(q1, q5)
            worstx = max(worstx, abs(float(W3[0]) - planar[0]))
            worstz = max(worstz, abs(float(W3[2]) - planar[1]))
        check(
            f"{side} front chain",
            max(worstx, worstz) < 1e-9,
            f"max |dx|={worstx:.2e}  max |dz|={worstz:.2e}  over {N_RANDOM} poses",
        )

    # ------------------------------------------------------------------ 2
    print("\n[2] planar back chain vs exact 3-D URDF FK  (X and Z)")
    for side, leg in legs.items():
        loop = next(lp for lp in tree.loops if lp["link1"] == f"{side}_wheel")
        worstx = worstz = 0.0
        for _ in range(N_RANDOM):
            q4 = float(rng.uniform(j[f"{side}_back_hip"]["lower"], j[f"{side}_back_hip"]["upper"]))
            q6 = float(rng.uniform(j[f"{side}_back_knee"]["lower"], j[f"{side}_back_knee"]["upper"]))
            fr = tree.solve_frames({f"{side}_back_hip": q4, f"{side}_back_knee": q6})
            R = fr[loop["link2"]][:3, :3]
            E3 = fr[loop["link2"]][:3, 3] + R @ loop["xyz2"]
            _, planar = leg.back(q4, q6)
            worstx = max(worstx, abs(float(E3[0]) - planar[0]))
            worstz = max(worstz, abs(float(E3[2]) - planar[1]))
        check(
            f"{side} back chain",
            max(worstx, worstz) < 1e-9,
            f"max |dx|={worstx:.2e}  max |dz|={worstz:.2e}  over {N_RANDOM} poses",
        )

    # ------------------------------------------------------------------ 3
    print("\n[3] the model reproduces the URDF assembly (all joints = 0)")
    for side, leg in legs.items():
        dx, dz = leg.closure(0.0, 0.0, 0.0, 0.0)
        check(f"{side} closure at q=0", math.hypot(dx, dz) < 1e-9, f"|W-E| = {math.hypot(dx, dz):.3e} m")

    # ------------------------------------------------------------------ 4
    print("\n[4] solve_loop_np closes the loop over the hip workspace")
    q1_lo, q1_hi = j["left_front_hip"]["lower"], j["left_front_hip"]["upper"]
    q4_lo, q4_hi = j["left_back_hip"]["lower"], j["left_back_hip"]["upper"]
    for side, leg in legs.items():
        worst, n_ok, n_tot, best_seed = 0.0, 0, 0, (0.0, 0.0)
        for q1 in np.linspace(q1_lo + 0.05, q1_hi - 0.05, 14):
            for q4 in np.linspace(q4_lo + 0.05, q4_hi - 0.05, 14):
                q5, q6, res, conv = solve_loop_np(leg, float(q1), float(q4), seed=best_seed)
                # verify with the exact residual, not just the solver's own value
                w = leg.front(float(q1), q5)[1]
                e = leg.back(float(q4), q6)[1]
                exact = math.dist(w, e)
                n_tot += 1
                worst = max(worst, exact)
                if exact < 1e-9:
                    n_ok += 1
                    best_seed = (q5, q6)
        check(f"{side} solver", n_ok == n_tot, f"{n_ok}/{n_tot} closed, worst |W-E| = {worst:.3e} m")

    # ------------------------------------------------------------------ 5
    print("\n[5] solved knee angles stay inside the URDF limits")
    for side, leg in legs.items():
        lo5, hi5 = j[f"{side}_front_knee"]["lower"], j[f"{side}_front_knee"]["upper"]
        lo6, hi6 = j[f"{side}_back_knee"]["lower"], j[f"{side}_back_knee"]["upper"]
        bad, n_tot, seed = [], 0, (0.0, 0.0)
        for q1 in np.linspace(q1_lo + 0.05, q1_hi - 0.05, 14):
            for q4 in np.linspace(q4_lo + 0.05, q4_hi - 0.05, 14):
                q5, q6, res, conv = solve_loop_np(leg, float(q1), float(q4), seed=seed)
                if math.dist(leg.front(float(q1), q5)[1], leg.back(float(q4), q6)[1]) > 1e-9:
                    continue
                seed = (q5, q6)
                n_tot += 1
                if not (lo5 - 1e-9 <= q5 <= hi5 + 1e-9 and lo6 - 1e-9 <= q6 <= hi6 + 1e-9):
                    bad.append((round(float(q1), 3), round(float(q4), 3), round(q5, 3), round(q6, 3)))
        check(
            f"{side} knees in limits",
            not bad,
            f"{len(bad)}/{n_tot} violations" + (f" e.g. {bad[0]}" if bad else ""),
        )

    # ------------------------------------------------------------------ 6
    print("\n[6] neutral standing pose, found by solving the closure")
    for side, leg in legs.items():
        for L in (0.16, 0.175, 0.19, 0.205):
            q1, q4, q5, q6, res = neutral_pose_np(tree, leg, leg_length=L)
            L0, phi = leg_state_np(leg, q1, q5)
            print(
                f"    {side} L0={L:0.3f} -> front_hip={q1:+.4f} back_hip={q4:+.4f}"
                f" front_knee={q5:+.4f} back_knee={q6:+.4f}  achieved L0={L0:.5f}"
                f" phi={math.degrees(phi):+.2f} deg  res={res:.2e}"
            )
        ok = True
        for L in (0.16, 0.175, 0.19, 0.205):
            q1, q4, q5, q6, res = neutral_pose_np(tree, leg, leg_length=L)
            lo5, hi5 = j[f"{side}_front_knee"]["lower"], j[f"{side}_front_knee"]["upper"]
            lo6, hi6 = j[f"{side}_back_knee"]["lower"], j[f"{side}_back_knee"]["upper"]
            lo1, hi1 = j[f"{side}_front_hip"]["lower"], j[f"{side}_front_hip"]["upper"]
            lo4, hi4 = j[f"{side}_back_hip"]["lower"], j[f"{side}_back_hip"]["upper"]
            ok &= res < 1e-9
            ok &= lo1 <= q1 <= hi1 and lo4 <= q4 <= hi4 and lo5 <= q5 <= hi5 and lo6 <= q6 <= hi6
        check(f"{side} neutral poses closed and inside limits", ok)

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
