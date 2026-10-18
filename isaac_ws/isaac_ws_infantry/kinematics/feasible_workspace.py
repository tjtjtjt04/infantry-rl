"""Find the physically feasible standing poses of the five-bar leg.

``solve_loop`` can close the loop for many hip pairs, but only some of those
closures keep the knee joints inside their URDF limits.  This script scans the
actuated (hip) space, keeps only closures that are inside the limits, and reports
the reachable leg-length / leg-angle envelope.  That envelope is what the RL task
must command within.

Run:  python isaac_ws/isaac_ws_infantry/kinematics/feasible_workspace.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
URDF = WS / "assets" / "infantry-rl" / "urdf" / "infantry-rl-sim.urdf"
sys.path.insert(0, str(HERE))

from five_bar import UrdfTree  # noqa: E402
from leg_model import LegPlanar, solve_loop  # noqa: E402


def main() -> int:
    tree = UrdfTree(URDF)
    j = tree.joints
    out = {}

    for side in ("left", "right"):
        leg, _, _ = LegPlanar.from_urdf(tree, side)
        lo1, hi1 = j[f"{side}_front_hip"]["lower"], j[f"{side}_front_hip"]["upper"]
        lo4, hi4 = j[f"{side}_back_hip"]["lower"], j[f"{side}_back_hip"]["upper"]
        lo5, hi5 = j[f"{side}_front_knee"]["lower"], j[f"{side}_front_knee"]["upper"]
        lo6, hi6 = j[f"{side}_back_knee"]["lower"], j[f"{side}_back_knee"]["upper"]

        print(f"=== {side} leg ===")
        print(f"  limits hip_f=[{lo1:+.4f},{hi1:+.4f}] hip_b=[{lo4:+.4f},{hi4:+.4f}]")
        print(f"         knee_f=[{lo5:+.4f},{hi5:+.4f}] knee_b=[{lo6:+.4f},{hi6:+.4f}]")

        rows = []
        n = 121
        for q1 in np.linspace(lo1 + 1e-3, hi1 - 1e-3, n):
            for q4 in np.linspace(lo4 + 1e-3, hi4 - 1e-3, n):
                q5, q6, res, _ = solve_loop(leg, float(q1), float(q4), seed=(0.0, 0.0))
                if res > 1e-9:
                    continue
                if not (lo5 <= q5 <= hi5 and lo6 <= q6 <= hi6):
                    continue
                L0, phi = leg.leg_state(float(q1), q5)
                rows.append((L0, math.degrees(phi), float(q1), float(q4), q5, q6))

        print(f"  feasible closures inside limits: {len(rows)} / {n*n}")
        if not rows:
            out[side] = None
            continue
        Ls = [r[0] for r in rows]
        ph = [r[1] for r in rows]
        print(f"  leg length L0 : {min(Ls):.4f} .. {max(Ls):.4f} m")
        print(f"  leg angle phi : {min(ph):+.2f} .. {max(ph):+.2f} deg")

        # the most upright pose: phi closest to 0, tie-broken by largest L0
        upright = sorted(rows, key=lambda r: (abs(r[1]), -r[0]))
        print("  8 most upright feasible poses (L0, phi, hip_f, hip_b, knee_f, knee_b):")
        for r in upright[:8]:
            print(
                f"    L0={r[0]:.4f} phi={r[1]:+7.2f}deg  hip_f={r[2]:+.4f} hip_b={r[3]:+.4f}"
                f"  knee_f={r[4]:+.4f} knee_b={r[5]:+.4f}"
            )

        # a good standing pose: phi ~ 0, L0 as large as possible
        best = upright[0]
        out[side] = {
            "leg_length": best[0],
            "phi_deg": best[1],
            "front_hip": best[2],
            "back_hip": best[3],
            "front_knee": best[4],
            "back_knee": best[5],
            "L0_min": min(Ls),
            "L0_max": max(Ls),
            "phi_min_deg": min(ph),
            "phi_max_deg": max(ph),
            "n_feasible": len(rows),
        }

        # envelope of feasible (L0, phi)
        by_L = {}
        for r in rows:
            by_L.setdefault(round(r[0], 3), []).append(r[1])
        print("  L0 -> phi range (feasible):")
        for L in sorted(by_L)[:: max(1, len(by_L) // 12)]:
            phs = by_L[L]
            print(f"    L0={L:5.3f}  phi [{min(phs):+7.2f}, {max(phs):+7.2f}] deg")
        print()

    p = WS / "config" / "feasible_poses.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
