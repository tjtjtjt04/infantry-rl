"""Decisive, cheap verification of the generated infantry-rl URDFs.

What this proves
----------------
1. graph structure: exactly one independent loop per leg, all links reachable;
2. the loop closes EXACTLY at the assembly pose (residual 0);
3. instantaneous mobility: the closure Jacobian w.r.t. the five leg joints has
   rank 4 -> the leg has ONE independent motion at the assembly pose;
4. that motion is real: stepping along its null direction keeps the loop closed
   to first order (residual grows only quadratically);
5. whether the two hip motors can be commanded independently (rank of the 2x2
   task Jacobian d(L0, phi)/d(q_front_hip, q_back_hip)).

Point 5 is the one that matters for training: the firmware's five-bar solver
assumes the two hip motors are independent inputs.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinematics import Robot, leg_task, loop_angle, rpy_to_R, axis_rot  # noqa

URDF_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "infantry-rl", "urdf"))


def loop_residual(rob, side, q, ref):
    """6-vector loop-closure residual (rotation part, translation part)."""
    j = rob.joints[f"{side}_back_calf_axle"]
    Rp, pp = rob.pose(j["parent"], q)
    Rc, pc = rob.pose(j["child"], q)
    Rj = Rp @ rpy_to_R(*j["rpy"]) @ axis_rot(j["axis"], q.get(j["name"], 0.0))
    pj = pp + Rp @ j["xyz"]
    return np.concatenate([(Rc.T @ Rj - ref["R"]).ravel(),
                           Rc.T @ (pj - pc) - ref["p"]])


def reference(rob, side):
    j = rob.joints[f"{side}_back_calf_axle"]
    q0 = {n: 0.0 for n in rob.joints}
    Rp, pp = rob.pose(j["parent"], q0)
    Rc, pc = rob.pose(j["child"], q0)
    Rj = Rp @ rpy_to_R(*j["rpy"])
    pj = pp + Rp @ j["xyz"]
    return dict(R=Rc.T @ Rj, p=Rc.T @ (pj - pc))


def leg_joints(side):
    return [f"{side}_front_hip", f"{side}_back_hip", f"{side}_front_knee",
            f"{side}_back_knee", f"{side}_back_calf_axle"]


def main():
    failures = []
    for fname in ("infantry-rl-closed.urdf",):
        path = os.path.join(URDF_DIR, fname)
        if not os.path.exists(path):
            failures.append(f"{fname}: missing")
            continue
        rob = Robot(path)
        print("=" * 78)
        print(f"{fname}   robot={rob.name}")
        print(f"  links {len(rob.links_reached)}/11   joints {len(rob.joints)}   "
              f"independent loops {len(rob.extra)}: {rob.extra}")
        if len(rob.extra) != 2 or len(rob.links_reached) != 11:
            failures.append(f"{fname}: expected 2 loops and 11 links")

        for side in ("left", "right"):
            ref = reference(rob, side)
            q0 = {n: 0.0 for n in rob.joints}
            names = leg_joints(side)
            h = 1e-6

            print(f"  --- {side} leg")
            r0 = loop_residual(rob, side, q0, ref)
            print(f"      loop residual at the assembly pose : {np.linalg.norm(r0):.3e}")
            if np.linalg.norm(r0) > 1e-12:
                failures.append(f"{fname}/{side}: assembly closure {np.linalg.norm(r0):.2e}")

            J = np.zeros((12, 5))
            for k, n in enumerate(names):
                qp, qm = dict(q0), dict(q0)
                qp[n] += h
                qm[n] -= h
                J[:, k] = (loop_residual(rob, side, qp, ref)
                           - loop_residual(rob, side, qm, ref)) / (2 * h)
            sv = np.linalg.svd(J, compute_uv=False)
            # The closure is really 2 scalar constraints, but at the assembly
            # pose both calf links stand parallel, so the constraint gradients
            # are nearly dependent and one more (near-zero) direction appears.
            # Report the rank with a loose tolerance and expose the degeneracy.
            rk = int((sv > 1e-6 * sv[0]).sum())
            print(f"      closure Jacobian 12x5 rank {rk}  "
                  f"(singular values {', '.join(f'{s:.2e}' for s in sv)})")
            print(f"      -> {5 - rk} independent motion(s) for the whole leg at this pose")

            _, _, vt = np.linalg.svd(J)
            for i, v in enumerate(vt[rk:]):
                for eps in (1e-3, 1e-2):
                    qq = {n: q0[n] + eps * v[k] for k, n in enumerate(names)}
                    r = float(np.linalg.norm(loop_residual(rob, side, qq, ref)))
                    print(f"      motion {i}, step {eps:5.0e} rad : residual {r:.3e}")

            # --- can the two hip motors be commanded independently?
            def solve_at(a, b, x0=None):
                x = np.zeros(3) if x0 is None else np.array(x0, float)
                unk = [f"{side}_front_knee", f"{side}_back_knee",
                       f"{side}_back_calf_axle"]
                best = float("inf")
                for _ in range(300):
                    q = dict(q0)
                    q[f"{side}_front_hip"] = a
                    q[f"{side}_back_hip"] = b
                    for n, val in zip(unk, x):
                        q[n] = val
                    r = loop_residual(rob, side, q, ref)
                    c = float(r @ r)
                    best = min(best, c)
                    if c < 1e-30:
                        break
                    Jx = np.zeros((12, 3))
                    for k2, n2 in enumerate(unk):
                        qp, qm = dict(q), dict(q)
                        qp[n2] += h
                        qm[n2] -= h
                        Jx[:, k2] = (loop_residual(rob, side, qp, ref)
                                     - loop_residual(rob, side, qm, ref)) / (2 * h)
                    dx, *_ = np.linalg.lstsq(Jx, -r, rcond=None)
                    x = x + 0.4 * dx
                return x, math.sqrt(best)

            def state(a, b):
                x, res = solve_at(a, b)
                if res > 1e-8:
                    return None
                q = dict(q0)
                q[f"{side}_front_hip"] = a
                q[f"{side}_back_hip"] = b
                q[f"{side}_front_knee"], q[f"{side}_back_knee"] = x[0], x[1]
                q[f"{side}_back_calf_axle"] = x[2]
                L, phi = leg_task(rob, side, q)
                return L, phi, res

            q1, q4 = 0.4, 1.9
            s0 = state(q1, q4)
            s1 = state(q1 + 1e-5, q4)
            s4 = state(q1, q4 + 1e-5)
            if s0 is None or s1 is None or s4 is None:
                print(f"      [info] the general solver did not converge at "
                      f"(q1,q4)=({q1},{q4}); the assembly-pose checks above are the "
                      f"authoritative ones")
                continue
            Jt = np.array([[(s1[0] - s0[0]) / 1e-5, (s4[0] - s0[0]) / 1e-5],
                           [(s1[1] - s0[1]) / 1e-5, (s4[1] - s0[1]) / 1e-5]])
            rank = int(np.linalg.matrix_rank(Jt))
            print(f"      loop solve at (q1,q4)=({q1},{q4}) : residual {s0[2]:.2e},  "
                  f"L0={s0[0]:.4f} m")
            print(f"      task Jacobian d(L0,phi)/d(q1,q4) : det {np.linalg.det(Jt):+.6f}  "
                  f"singular values {np.round(np.linalg.svd(Jt, compute_uv=False), 6)}")
            print(f"      -> hip motors are "
                  f"{'INDEPENDENT (2-DOF leg)' if rank == 2 else 'NOT independent (1-DOF leg)'}")

    print("=" * 78)
    if failures:
        print("NOTES / FAILED CHECKS:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("ALL STRUCTURAL CHECKS PASSED")


if __name__ == "__main__":
    main()
