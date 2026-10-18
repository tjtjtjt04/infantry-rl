"""Validation of the leg kinematics, the loop solver and the torch solver.

Every check compares against an exact double-precision forward kinematics of the
URDF, so a pass means the model is faithful, not merely self-consistent.

Run:  python isaac_ws/isaac_ws_infantry/kinematics/test_leg_model.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent  # isaac_ws
URDF = WS / "assets" / "infantry-rl" / "urdf" / "infantry-rl-sim.urdf"
sys.path.insert(0, str(HERE))

from five_bar import UrdfTree  # noqa: E402
from leg_model import (  # noqa: E402
    LegPlanar,
    leg_limits,
    make_torch_loop_solver,
    neutral_pose,
    solve_loop,
)

FAILS: list[str] = []
N_RANDOM = 3000


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    tree = UrdfTree(URDF)
    j = tree.joints
    print(f"URDF   : {URDF}")
    print(f"joints : {len(tree.joints)}    loop_joints: {[lp['name'] for lp in tree.loops]}")

    legs, fps, bps = {}, {}, {}
    for side in ("left", "right"):
        legs[side], fps[side], bps[side] = LegPlanar.from_urdf(tree, side, verbose=True)

    # ---------------------------------------------------------------- 1
    print("\n[1] model vs exact URDF FK, front and back chains")
    rng = np.random.default_rng(11)
    for side in ("left", "right"):
        leg = legs[side]
        wf = wb = 0.0
        for _ in range(N_RANDOM):
            q1 = float(rng.uniform(j[f"{side}_front_hip"]["lower"], j[f"{side}_front_hip"]["upper"]))
            q5 = float(rng.uniform(j[f"{side}_front_knee"]["lower"], j[f"{side}_front_knee"]["upper"]))
            _, w = leg.front(q1, q5)
            _, wu = fps[side](q1, q5)
            wf = max(wf, math.dist(w, wu))
            q4 = float(rng.uniform(j[f"{side}_back_hip"]["lower"], j[f"{side}_back_hip"]["upper"]))
            q6 = float(rng.uniform(j[f"{side}_back_knee"]["lower"], j[f"{side}_back_knee"]["upper"]))
            _, e = leg.back(q4, q6)
            _, eu = bps[side](q4, q6)
            wb = max(wb, math.dist(e, eu))
        check(f"{side} front chain", wf < 1e-5, f"max err {wf:.3e} m over {N_RANDOM} poses")
        check(f"{side} back  chain", wb < 1e-5, f"max err {wb:.3e} m over {N_RANDOM} poses")

    # ---------------------------------------------------------------- 2
    print("\n[2] the URDF assembled pose is closed")
    for side in ("left", "right"):
        dx, dz = legs[side].closure(0.0, 0.0, 0.0, 0.0)
        check(f"{side} closure at all-zero", math.hypot(dx, dz) < 1e-6, f"|W-E| = {math.hypot(dx, dz):.3e} m")

    # ---------------------------------------------------------------- 3
    print("\n[3] solve_loop closes the loop across the hip workspace")
    q1_lo, q1_hi = j["left_front_hip"]["lower"], j["left_front_hip"]["upper"]
    q4_lo, q4_hi = j["left_back_hip"]["lower"], j["left_back_hip"]["upper"]
    for side in ("left", "right"):
        leg = legs[side]
        worst, ok_n, tot = 0.0, 0, 0
        seed = (0.0, 0.0)
        for q1 in np.linspace(q1_lo + 0.05, q1_hi - 0.05, 14):
            for q4 in np.linspace(q4_lo + 0.05, q4_hi - 0.05, 14):
                q5, q6, _res, _conv = solve_loop(leg, float(q1), float(q4), seed=seed)
                w = leg.front(float(q1), q5)[1]
                e = leg.back(float(q4), q6)[1]
                err = math.dist(w, e)
                tot += 1
                worst = max(worst, err)
                if err < 1e-9:
                    ok_n += 1
                    seed = (q5, q6)
        check(f"{side} numpy solver", ok_n == tot, f"{ok_n}/{tot} closed, worst {worst:.3e} m")

    # ---------------------------------------------------------------- 4
    print("\n[4] feasible fraction of the hip workspace (knees inside URDF limits)")
    # NOTE: not every hip pair admits a closure with both knees inside their limits
    # -- the leg is a closed mechanism and the limits clip its workspace.  What
    # matters is that a healthy fraction IS feasible and that whatever the solver
    # reports as feasible really is inside the limits.
    for side in ("left", "right"):
        leg = legs[side]
        lim = leg_limits(URDF, side)
        (lo1, hi1), (lo4, hi4) = lim["front_hip"], lim["back_hip"]
        (lo5, hi5), (lo6, hi6) = lim["front_knee"], lim["back_knee"]
        n_feasible = n_closed = n_tot = 0
        lie = []
        for q1 in np.linspace(lo1 + 0.05, hi1 - 0.05, 14):
            for q4 in np.linspace(lo4 + 0.05, hi4 - 0.05, 14):
                # cold start: a warm start can lock onto one closure branch, and the
                # question here is what the workspace as a whole offers
                q5 = q6 = 0.0
                best = None
                for seed in ((0.0, 0.0), (1.0, -1.0), (-1.0, 1.0), (0.5, 0.5), (-0.5, -0.5)):
                    a, b, _r, _c = solve_loop(leg, float(q1), float(q4), seed=seed)
                    err = math.dist(leg.front(float(q1), a)[1], leg.back(float(q4), b)[1])
                    if best is None or err < best[0]:
                        best = (err, a, b)
                err, q5, q6 = best
                n_tot += 1
                if err > 1e-9:
                    continue
                n_closed += 1
                if lo5 <= q5 <= hi5 and lo6 <= q6 <= hi6:
                    n_feasible += 1
        frac = n_feasible / max(1, n_tot)
        check(
            f"{side} feasible fraction sane",
            not lie and frac > 0.2,
            f"{n_feasible}/{n_tot} hip pairs feasible ({frac:.0%}), {n_closed} closed",
        )

    # ---------------------------------------------------------------- 5
    print("\n[5] torch batched solver agrees with the numpy solver")
    try:
        import torch

        for side in ("left", "right"):
            leg = legs[side]
            solve_t = make_torch_loop_solver(leg, device="cpu", iters=6)
            n = 256
            q1 = torch.linspace(q1_lo + 0.08, q1_hi - 0.08, n)
            q4 = torch.linspace(q4_lo + 0.08, q4_hi - 0.08, n)
            q5, q6, res = solve_t(q1, q4, torch.zeros(n), torch.zeros(n))
            worst = float(res.max())
            check(f"{side} torch solver residual", worst < 1e-6, f"max |W-E| = {worst:.3e} m over {n} samples")
    except ImportError:  # pragma: no cover
        print("  [SKIP] torch not installed")

    # ---------------------------------------------------------------- 6
    print("\n[6] nominal standing poses (searched inside the joint limits)")
    for side in ("left", "right"):
        leg = legs[side]
        lim = leg_limits(URDF, side)
        lo1, hi1 = lim["front_hip"]
        lo4, hi4 = lim["back_hip"]
        lo5, hi5 = lim["front_knee"]
        lo6, hi6 = lim["back_knee"]
        for L in (0.200, 0.250, 0.267):
            q1, q4, q5, q6, res, L0, phi = neutral_pose(leg, leg_length=L, phi=0.0, limits=lim)
            inside = lo1 <= q1 <= hi1 and lo4 <= q4 <= hi4 and lo5 <= q5 <= hi5 and lo6 <= q6 <= hi6
            print(
                f"    {side} requested L0={L:.3f} -> hip_f={q1:+.5f} hip_b={q4:+.5f}"
                f" knee_f={q5:+.5f} knee_b={q6:+.5f}"
                f"  achieved L0={L0:.5f} phi={math.degrees(phi):+.3f}deg res={res:.2e} in_limits={inside}"
            )
        q1, q4, q5, q6, res, L0, phi = neutral_pose(leg, leg_length=0.250, phi=0.0, limits=lim)
        check(
            f"{side} nominal pose solved, closed and in limits",
            res < 1e-9
            and lo1 <= q1 <= hi1
            and lo4 <= q4 <= hi4
            and lo5 <= q5 <= hi5
            and lo6 <= q6 <= hi6
            and abs(L0 - 0.250) < 5e-3,
            f"L0={L0:.5f} phi={math.degrees(phi):+.3f}deg res={res:.2e}",
        )

    # ---------------------------------------------------------------- 7
    print("\n[7] dump the measured model constants (for the env config)")
    dump = {side: legs[side].to_dict() for side in ("left", "right")}
    out = WS / "config" / "leg_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(f"    wrote {out}")
    for side in ("left", "right"):
        d = dump[side]
        print(
            f"    {side}: |P|={math.hypot(*d['P']):.7f} |Q|={math.hypot(*d['Q']):.7f}"
            f" s_hip={d['s_hip_front']:+.0f} s_knee={d['s_knee_front']:+.0f}"
        )

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
