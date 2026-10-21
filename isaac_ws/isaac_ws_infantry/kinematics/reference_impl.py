"""numpy reference implementation of the planar five-bar leg (verification only).

This is the first, straightforward version of the planar leg model and its loop solver.
The RUNTIME uses :mod:`isaac_ws_infantry.kinematics.planar_leg` instead, which is the
same geometry rewritten as a batched torch solver with an analytic Jacobian; this file
is kept because the tests compare the two implementations against the exact URDF
forward kinematics in :mod:`isaac_ws_infantry.kinematics.urdf_tree`.

Do not import this from the tasks -- it is numpy/scalar and far too slow for training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from urdf_tree import UrdfTree, ang_xz, rot_xz, _rpy

# --------------------------------------------------------------------------- #
# Planar model of one leg
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlanarLeg:
    """Planar (X-Z) model of one leg, derived from the URDF.

    Angles are ABSOLUTE: ``a_fh + q1`` is the orientation of the front ham measured
    from the body +X axis toward the body -Z axis (that is the sense of a positive
    rotation about body +Y).
    """

    name: str
    A: tuple[float, float]  # front hip axis, in the leg's own X-Z frame
    B: tuple[float, float]  # back hip axis
    p: tuple[float, float]  # A  -> front knee, expressed in the front ham frame
    r: tuple[float, float]  # front knee -> wheel axle, in the front calf frame
    s: tuple[float, float]  # B  -> back knee, in the back ham frame
    t: tuple[float, float]  # back knee  -> loop anchor, in the back calf frame
    a_fh: float  # angle of the front ham offset p at q = 0 (in the rot_xz sense)
    a_fc: float  # angle of the front calf offset r at q = 0
    b_bh: float  # angle of the back ham offset s at q = 0
    b_bc: float  # angle of the back calf offset t at q = 0
    s_fh: float  # sign of the front hip  joint value in the planar rotation
    s_fk: float  # sign of the front knee joint value
    s_bh: float  # sign of the back  hip  joint value
    s_bk: float  # sign of the back  knee joint value
    y_front: float  # body-Y plane of the front chain (measurement, for checks)
    y_back: float  # body-Y plane of the back chain
    # ---- planar kinematics ------------------------------------------------
    def front(self, q1: float, q5: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return (front knee K1, wheel axle W).

        ``p`` and ``r`` are the link offsets AS MEASURED at zero joint value, so
        they already sit at their zero-configuration angles.  A joint therefore
        only applies its *delta* to them::

            K1 = A + R(p, ang(p) - q1)
            W  = K1 + R(r, ang(r) + q5)

        The deltas (hip ``-q1``, knee ``+q5``) are measured on the exact URDF FK
        and are identical for the left and right leg and for both axis signs.
        """
        k1 = _add(self.A, rot_xz(self.p, self.a_fh - q1))
        w = _add(k1, rot_xz(self.r, self.a_fc + q5))
        return k1, w

    def back(self, q4: float, q6: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return (back knee K4, loop anchor E); same convention as :meth:`front`."""
        k4 = _add(self.B, rot_xz(self.s, self.b_bh - q4))
        e = _add(k4, rot_xz(self.t, self.b_bc + q6))
        return k4, e

    def closure(self, q1: float, q4: float, q5: float, q6: float) -> tuple[float, float]:
        """Loop residual W - E; both components must vanish."""
        _, w = self.front(q1, q5)
        _, e = self.back(q4, q6)
        return (w[0] - e[0], w[1] - e[1])

    def hip_midpoint(self) -> tuple[float, float]:
        return (0.5 * (self.A[0] + self.B[0]), 0.5 * (self.A[1] + self.B[1]))

    # ---- construction -----------------------------------------------------
    @staticmethod
    def from_urdf(tree: UrdfTree, side: str) -> "PlanarLeg":
        j = tree.joints
        fh, fk, wh = f"{side}_front_hip", f"{side}_front_knee", f"{side}_wheel"
        bh, bk = f"{side}_back_hip", f"{side}_back_knee"
        loop = next(lp for lp in tree.loops if lp["link1"] == wh)

        f0 = tree.solve_frames({})

        def xz(p3) -> tuple[float, float]:
            return (float(p3[0]), float(p3[2]))

        def pivot(joint: str) -> tuple[float, float]:
            """Position of a joint's rotation axis at the neutral pose.

            A revolute joint's rotation centre is the ORIGIN of the joint frame --
            which for a child link is that link's origin, but the chain pivot we
            need is the one of the joint itself, so read it off the parent link's
            frame composed with the joint origin (independent of the joint value).
            """
            fr = tree.solve_frames({})
            T = np.eye(4)
            T[:3, :3] = _rpy(*j[joint]["rpy"])[:3, :3]
            T[:3, 3] = j[joint]["xyz"]
            return xz((fr[j[joint]["parent"]] @ T)[:3, 3])

        A = pivot(fh)  # front hip axis
        B = pivot(bh)  # back hip axis
        K1_0 = pivot(fk)  # front knee axis at the neutral pose
        K4_0 = pivot(bk)  # back knee axis at the neutral pose
        W3 = f0[j[wh]["child"]][:3, 3]
        bcR = f0[loop["link2"]][:3, :3]
        E3 = f0[loop["link2"]][:3, 3] + bcR @ loop["xyz2"]  # loop anchor on the back calf

        def pos_front(q1: float, q5: float) -> tuple[tuple[float, float], tuple[float, float]]:
            fr = tree.solve_frames({fh: q1, fk: q5})
            return (
                xz(fr[j[fk]["child"]][:3, 3]),
                xz(fr[j[wh]["child"]][:3, 3]),
            )

        def pos_back(q4: float, q6: float) -> tuple[tuple[float, float], tuple[float, float]]:
            fr = tree.solve_frames({bh: q4, bk: q6})
            R = fr[loop["link2"]][:3, :3]
            e3 = fr[loop["link2"]][:3, 3] + R @ loop["xyz2"]
            return (xz(fr[j[bk]["child"]][:3, 3]), xz(e3))

        def fit_chain(base, pos_fn, label):
            """Fit ``K = base + R(s_h qh) P`` and ``W = K + R(s_h qh) R(s_k qk) Q``.

            ``base`` is the hip axis -- the one point that does not move when the hip
            turns.  The two joint signs are the only discrete unknowns, so all four
            combinations are tried; for a fixed pair the offsets are the average of
            the back-rotated measurements, and the winner is the pair with the
            smallest model residual.
            """
            eps = 0.2
            samples = [(a, b) for a in (-eps, 0.0, eps) for b in (-eps, 0.0, eps)]
            data = [pos_fn(a, b) for a, b in samples]

            best = None
            for s_h in (1.0, -1.0):
                for s_k in (1.0, -1.0):
                    acc_p = [0.0, 0.0]
                    for (a, _b), (k, _w) in zip(samples, data):
                        rr = rot_xz((k[0] - base[0], k[1] - base[1]), -s_h * a)
                        acc_p[0] += rr[0]
                        acc_p[1] += rr[1]
                    P = (acc_p[0] / len(samples), acc_p[1] / len(samples))

                    acc_q = [0.0, 0.0]
                    for (a, b), (k, w) in zip(samples, data):
                        rr = rot_xz((w[0] - k[0], w[1] - k[1]), -(s_h * a + s_k * b))
                        acc_q[0] += rr[0]
                        acc_q[1] += rr[1]
                    Q = (acc_q[0] / len(samples), acc_q[1] / len(samples))

                    res = 0.0
                    for (a, b), (k, w) in zip(samples, data):
                        kp = rot_xz(P, s_h * a)
                        k1 = (base[0] + kp[0], base[1] + kp[1])
                        qp = rot_xz(Q, s_h * a + s_k * b)
                        w1 = (k1[0] + qp[0], k1[1] + qp[1])
                        res = max(res, math.hypot(w1[0] - w[0], w1[1] - w[1]))
                    if best is None or res < best[0]:
                        best = (res, P, Q, s_h, s_k)
            res, P, Q, s_h, s_k = best
            # 1e-6 m is far below any physical scale (links are 0.15-0.27 m) and far
            # above the float64 FK noise (~1e-7 m); a wrong sign pair gives ~1e-1 m.
            if res > 1e-6:
                raise RuntimeError(f"{side}/{label}: chain model does not fit FK, residual {res:.3e} m")
            return P, Q, s_h, s_k

        p, r, s_fh, s_fk = fit_chain(A, pos_front, "front")
        s_off, t, s_bh, s_bk = fit_chain(B, pos_back, "back")

        return PlanarLeg(
            name=side,
            A=A,
            B=B,
            p=p,
            r=r,
            s=s_off,
            t=t,
            a_fh=ang_xz(p),  # absolute polar angle of p at q=0
            a_fc=ang_xz(r),
            b_bh=ang_xz(s_off),
            b_bc=ang_xz(t),
            s_fh=s_fh,
            s_fk=s_fk,
            s_bh=s_bh,
            s_bk=s_bk,
            y_front=float(f0[j[fk]["child"]][1, 3]),
            y_back=float(f0[loop["link2"]][1, 3]),
        )


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


# --------------------------------------------------------------------------- #
# Loop-closure solver (numpy)
# --------------------------------------------------------------------------- #
def solve_loop_np(
    leg: PlanarLeg,
    q1: float,
    q4: float,
    seed: tuple[float, float] = (0.0, 0.0),
    iters: int = 200,
    tol: float = 1e-12,
    max_step: float = 0.5,
) -> tuple[float, float, float, bool]:
    """Solve the knee pair (q5, q6) that closes the loop for given hips (q1, q4).

    Damped Newton on the 2-D closure residual with a finite-difference Jacobian.
    Returns ``(q5, q6, residual_norm, converged)``.
    """
    x = [float(seed[0]), float(seed[1])]
    h = 1e-7

    def res(v):
        return leg.closure(q1, q4, v[0], v[1])

    lam = 1e-12
    for _ in range(iters):
        f0 = res(x)
        err = math.hypot(*f0)
        if err < tol:
            return x[0], x[1], err, True
        f5 = res((x[0] + h, x[1]))
        f6 = res((x[0], x[1] + h))
        J = np.array(
            [
                [(f5[0] - f0[0]) / h, (f6[0] - f0[0]) / h],
                [(f5[1] - f0[1]) / h, (f6[1] - f0[1]) / h],
            ]
        )
        M = J @ J.T + lam * np.eye(2)
        try:
            y = np.linalg.solve(M, np.array(f0))
        except np.linalg.LinAlgError:
            break
        dx = -(J.T @ y)
        nrm = math.hypot(*dx)
        if nrm > max_step:  # globalisation
            dx = (dx[0] * max_step / nrm, dx[1] * max_step / nrm)
        step = 1.0
        for _ in range(60):
            cand = (x[0] + step * dx[0], x[1] + step * dx[1])
            if math.hypot(*res(cand)) < err:
                break
            step *= 0.5
        x = [x[0] + step * dx[0], x[1] + step * dx[1]]
    return x[0], x[1], math.hypot(*res(x)), False


def leg_state_np(leg: PlanarLeg, q1: float, q5: float) -> tuple[float, float]:
    """Firmware-style leg state ``(L0, phi)``.

    ``L0``  distance from the hip-axis midpoint to the wheel axle, metres.
    ``phi`` angle of the leg axis from straight down (body -Z), positive toward
    body +X (forward), radians.
    """
    mid = leg.hip_midpoint()
    _, w = leg.front(q1, q5)
    dx = w[0] - mid[0]
    dz = w[1] - mid[1]
    return math.hypot(dx, dz), math.atan2(dx, -dz)


def neutral_pose_np(tree: UrdfTree, leg: PlanarLeg, leg_length: float | None = None, phi: float = 0.0):
    """Find one leg's four joint angles for a chosen stance.

    Puts the wheel axle at distance ``leg_length`` from the hip-axis midpoint, at
    leg angle ``phi`` (0 = straight down), and closes the loop.  Returns
    ``(q_front_hip, q_back_hip, q_front_knee, q_back_knee, residual)``.
    """
    mid = leg.hip_midpoint()
    default_L = math.hypot(*(v - m for v, m in zip(leg.front(0.0, 0.0)[1], mid)))
    L = default_L if leg_length is None else leg_length
    target = (mid[0] + L * math.sin(phi), mid[1] - L * math.cos(phi))

    def front_err(q1, q5):
        _, w = leg.front(q1, q5)
        return (w[0] - target[0], w[1] - target[1])

    # 2-D Newton on (q1, q5) so the front chain reaches the target
    x = [0.0, 0.0]
    h = 1e-7
    for _ in range(200):
        f0 = front_err(*x)
        if math.hypot(*f0) < 1e-13:
            break
        f1 = front_err(x[0] + h, x[1])
        f5 = front_err(x[0], x[1] + h)
        J = np.array(
            [
                [(f1[0] - f0[0]) / h, (f5[0] - f0[0]) / h],
                [(f1[1] - f0[1]) / h, (f5[1] - f0[1]) / h],
            ]
        )
        try:
            dx = np.linalg.solve(J, -np.array(f0))
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(dx) > 0.3:
            dx = dx * (0.3 / np.linalg.norm(dx))
        step = 1.0
        for _ in range(40):
            if math.hypot(*front_err(x[0] + step * dx[0], x[1] + step * dx[1])) < math.hypot(*f0):
                break
            step *= 0.5
        x = [x[0] + step * dx[0], x[1] + step * dx[1]]

    q1, q5 = x
    # then the back chain: 2-D Newton on (q4, q6) to land the loop anchor on the wheel
    y = [0.0, 0.0]

    def back_err(q4, q6):
        _, e = leg.back(q4, q6)
        _, w = leg.front(q1, q5)
        return (e[0] - w[0], e[1] - w[1])

    for _ in range(200):
        f0 = back_err(*y)
        if math.hypot(*f0) < 1e-13:
            break
        f1 = back_err(y[0] + h, y[1])
        f6 = back_err(y[0], y[1] + h)
        J = np.array(
            [
                [(f1[0] - f0[0]) / h, (f6[0] - f0[0]) / h],
                [(f1[1] - f0[1]) / h, (f6[1] - f0[1]) / h],
            ]
        )
        try:
            dy = np.linalg.solve(J, -np.array(f0))
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(dy) > 0.3:
            dy = dy * (0.3 / np.linalg.norm(dy))
        step = 1.0
        for _ in range(40):
            if math.hypot(*back_err(y[0] + step * dy[0], y[1] + step * dy[1])) < math.hypot(*f0):
                break
            step *= 0.5
        y = [y[0] + step * dy[0], y[1] + step * dy[1]]

    q4, q6 = y
    _, w = leg.front(q1, q5)
    _, e = leg.back(q4, q6)
    return q1, q4, q5, q6, math.dist(w, e)


# --------------------------------------------------------------------------- #
# Batched torch solver for the RL step
# --------------------------------------------------------------------------- #
def make_torch_solver(leg: PlanarLeg, device: str = "cpu", iters: int = 4, h: float = 1e-5):
    """Return ``solve(q1, q4, seed5, seed6) -> (q5, q6, residual)`` over torch batches.

    All tensors have shape ``(N,)``.  ``iters`` damped-Newton steps with a
    finite-difference Jacobian, fully vectorised on ``device``.
    """
    import torch

    def T(v):
        return torch.tensor(v, dtype=torch.float32, device=device)

    A_, B_ = T(leg.A), T(leg.B)
    p_, r_, s_, t_ = T(leg.p), T(leg.r), T(leg.s), T(leg.t)
    a_fh, a_fc, b_bh, b_bc = leg.a_fh, leg.a_fc, leg.b_bh, leg.b_bc
    s_fh, s_fk, s_bh, s_bk = leg.s_fh, leg.s_fk, leg.s_bh, leg.s_bk

    def rot2(v, a):
        c, s = torch.cos(a), torch.sin(a)
        return torch.stack((v[0] * c - v[1] * s, v[0] * s + v[1] * c), dim=-1)
    def closure(q1, q4, q5, q6):
        k1 = A_ + rot2(p_, a_fh + s_fh * q1)
        w = k1 + rot2(r_, a_fc + s_fh * q1 + s_fk * q5)
        k4 = B_ + rot2(s_, b_bh + s_bh * q4)
        e = k4 + rot2(t_, b_bc + s_bh * q4 + s_bk * q6)
        return w - e

    def solve(q1, q4, seed5, seed6, iters_: int = iters, max_step: float = 0.5):
        q5 = seed5.clone()
        q6 = seed6.clone()
        for _ in range(iters_):
            f0 = closure(q1, q4, q5, q6)
            f5 = closure(q1, q4, q5 + h, q6)
            f6 = closure(q1, q4, q5, q6 + h)
            j11 = (f5[:, 0] - f0[:, 0]) / h
            j21 = (f5[:, 1] - f0[:, 1]) / h
            j12 = (f6[:, 0] - f0[:, 0]) / h
            j22 = (f6[:, 1] - f0[:, 1]) / h
            det = j11 * j22 - j12 * j21
            det = torch.where(det.abs() < 1e-6, torch.full_like(det, 1e-6), det)
            dq5 = (-j22 * f0[:, 0] + j12 * f0[:, 1]) / det
            dq6 = (j21 * f0[:, 0] - j11 * f0[:, 1]) / det
            nrm = torch.sqrt(dq5 * dq5 + dq6 * dq6).clamp_min(1e-9)
            scale = (max_step / nrm).clamp_max(1.0)
            q5 = q5 + dq5 * scale
            q6 = q6 + dq6 * scale
        return q5, q6, closure(q1, q4, q5, q6).norm(dim=-1)

    return solve


