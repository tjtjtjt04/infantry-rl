"""Exact planar kinematics of the infantry-rl five-bar legs, plus the loop solver.

WHAT THIS FILE IS FOR
---------------------
``infantry-rl-sim.urdf`` describes each leg as a TREE plus a non-standard
``<loop_joint>`` that closes it into a parallel five-bar linkage.  Neither URDF nor
a PhysX articulation can express a closed chain -- measured on this machine
(Isaac Sim 5.1.0.0), importing the URDF gives an articulation with exactly **10**
joints, the two ``loop_joints/*_calf_axle`` are NOT among them, and ``*_back_calf``
just dangles.  The loop therefore has to be re-imposed by CONTROL: the knee joints
are passive followers whose targets must satisfy the closure every step.

This module provides exactly that closure, and nothing about it is hand-derived.

HOW THE MODEL IS BUILT
----------------------
Every joint axis is ``(0, +/-1, 0)`` in its own parent frame, so the whole leg is
planar in the body X-Z plane (verified: every joint moves the wheel axle with
``dY/dq == 0``).  Per leg there are two 2-link chains::

    front:   A --ham--> K1 --calf--> W     A  = front hip axis
    back:    B --ham--> K4 --calf--> E     B  = back hip axis

``W`` is the wheel axle, ``E`` is the loop anchor on the back calf, and closure
means ``W == E``.  Each chain is rigid in shape, so with ``P`` and ``Q`` the two
link vectors measured at the zero pose::

    K = base + R(s_hip * q_hip) @ P
    X = K    + R(s_hip * q_hip + s_knee * q_knee) @ Q

The calf inherits the hip's rotation because the knee axis is carried by the ham.
``P``, ``Q`` and the two signs are MEASURED from an exact double-precision FK of
the URDF by :meth:`LegPlanar.from_urdf`, which tries every sign combination and
keeps the one that reproduces the FK.  For this robot the answer is
``s_hip = -1``, ``s_knee = +1`` for both chains of both legs, and the fit residual
is at the float32 noise floor of the URDF (1.7e-16 m left, 4.5e-7 m right).

CONTROL INTERFACE
-----------------
The two hip joints are the actuated ones (matching the real robot's two hip motors
per leg); the two knees are solved for.  Given the commanded hips, :func:`solve_loop`
returns the knee pair that closes the loop, which the environment writes into the
knee position targets every control step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .urdf_tree import UrdfTree, _rpy  # exact URDF FK


# --------------------------------------------------------------------------- #
# Planar rotation of an (x, z) pair
# --------------------------------------------------------------------------- #
def rot(v, a: float):
    """Rotate ``v`` by ``a`` in the (x, z) plane, CCW: ``[[c,-s],[s,c]]``."""
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def polar(v) -> float:
    """The angle ``a`` for which ``rot(v, a)`` is aligned with +x."""
    return math.atan2(-v[1], v[0])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LegPlanar:
    """Two planar 2-link chains forming one five-bar leg."""

    name: str
    A: tuple[float, float]  # front hip axis
    B: tuple[float, float]  # back hip axis
    P: tuple[float, float]  # front hip  -> front knee, at the zero pose
    Q: tuple[float, float]  # front knee -> wheel axle, at the zero pose
    S: tuple[float, float]  # back hip   -> back knee, at the zero pose
    T: tuple[float, float]  # back knee  -> loop anchor, at the zero pose
    ham0_front: float
    calf0_front: float
    ham0_back: float
    calf0_back: float
    s_hip_front: float
    s_knee_front: float
    s_hip_back: float
    s_knee_back: float

    # ---- one chain -------------------------------------------------------
    @staticmethod
    def _chain(base, P, Q, s_hip, s_knee, q_hip, q_knee):
        th_h = s_hip * q_hip
        th_c = s_hip * q_hip + s_knee * q_knee
        k = _add(base, rot(P, th_h))
        x = _add(k, rot(Q, th_c))
        return k, x

    def front(self, q1: float, q5: float):
        """Return ``(K1, W)``: front knee and wheel axle."""
        return self._chain(self.A, self.P, self.Q, self.s_hip_front, self.s_knee_front, q1, q5)

    def back(self, q4: float, q6: float):
        """Return ``(K4, E)``: back knee and loop anchor."""
        return self._chain(self.B, self.S, self.T, self.s_hip_back, self.s_knee_back, q4, q6)

    def closure(self, q1: float, q4: float, q5: float, q6: float):
        """Return the 2-D loop residual ``W - E``; both components must vanish."""
        _, w = self.front(q1, q5)
        _, e = self.back(q4, q6)
        return (w[0] - e[0], w[1] - e[1])

    def hip_midpoint(self):
        return (0.5 * (self.A[0] + self.B[0]), 0.5 * (self.A[1] + self.B[1]))

    def leg_state(self, q1: float, q5: float):
        """Firmware-style ``(L0, phi)``: leg length and leg angle from straight down.

        ``L0`` is the distance from the hip-axis midpoint to the wheel axle, in
        metres; ``phi`` is positive toward body +X (forward).
        """
        mid = self.hip_midpoint()
        _, w = self.front(q1, q5)
        dx, dz = w[0] - mid[0], w[1] - mid[1]
        return math.hypot(dx, dz), math.atan2(dx, -dz)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "A": list(self.A),
            "B": list(self.B),
            "P": list(self.P),
            "Q": list(self.Q),
            "S": list(self.S),
            "T": list(self.T),
            "s_hip_front": self.s_hip_front,
            "s_knee_front": self.s_knee_front,
            "s_hip_back": self.s_hip_back,
            "s_knee_back": self.s_knee_back,
            "ham0_front": self.ham0_front,
            "calf0_front": self.calf0_front,
            "ham0_back": self.ham0_back,
            "calf0_back": self.calf0_back,
        }

    @staticmethod
    def from_dict(d: dict) -> "LegPlanar":
        t = lambda k: tuple(d[k])  # noqa: E731
        return LegPlanar(
            name=d["name"],
            A=t("A"),
            B=t("B"),
            P=t("P"),
            Q=t("Q"),
            S=t("S"),
            T=t("T"),
            ham0_front=d["ham0_front"],
            calf0_front=d["calf0_front"],
            ham0_back=d["ham0_back"],
            calf0_back=d["calf0_back"],
            s_hip_front=d["s_hip_front"],
            s_knee_front=d["s_knee_front"],
            s_hip_back=d["s_hip_back"],
            s_knee_back=d["s_knee_back"],
        )

    # ---- construction ----------------------------------------------------
    @staticmethod
    def from_urdf(
        tree: UrdfTree,
        side: str,
        tol: float = 1e-5,
        verbose: bool = False,
        joint_names: dict[str, str] | None = None,
    ):
        """Measure the model from the exact URDF FK.

        ``joint_names`` optionally overrides the joint names for the five roles
        (``front_hip``, ``back_hip``, ``front_knee``, ``back_knee``, ``wheel``);
        it defaults to the ``{side}_*`` convention.  The wheel override is needed by
        ``chassis-rl``, whose wheel joints are ``wheel_left`` / ``wheel_right`` while
        the link is still named ``left_wheel`` / ``right_wheel``.

        Returns ``(leg, front_pos, back_pos)`` where the two callables evaluate the
        exact FK, so callers can verify the model instead of trusting it.
        """
        names = dict(joint_names or {})
        j = tree.joints
        fh = names.get("front_hip", f"{side}_front_hip")
        fk = names.get("front_knee", f"{side}_front_knee")
        wh = names.get("wheel", f"{side}_wheel")
        bh = names.get("back_hip", f"{side}_back_hip")
        bk = names.get("back_knee", f"{side}_back_knee")
        wheel_link = j[wh]["child"]
        loop = next(lp for lp in tree.loops if lp["link1"] == wheel_link)

        def xz(p3):
            return (float(p3[0]), float(p3[2]))

        def pivot(joint: str):
            """Rotation centre of a joint at the zero pose (a joint's own origin)."""
            o = j[joint]
            fr = tree.solve_frames({})
            T = np.eye(4)
            T[:3, :3] = _rpy(*o["rpy"])[:3, :3]
            T[:3, 3] = o["xyz"]
            return xz((fr[o["parent"]] @ T)[:3, 3])

        def front_pos(q1, q5):
            fr = tree.solve_frames({fh: q1, fk: q5})
            return xz(fr[j[fk]["child"]][:3, 3]), xz(fr[j[wh]["child"]][:3, 3])

        def back_pos(q4, q6):
            fr = tree.solve_frames({bh: q4, bk: q6})
            R = fr[loop["link2"]][:3, :3]
            return xz(fr[j[bk]["child"]][:3, 3]), xz(fr[loop["link2"]][:3, 3] + R @ loop["xyz2"])

        A, B = pivot(fh), pivot(bh)

        def fit(base, pos_fn, label):
            eps = 0.25
            samples = [(a, b) for a in (-eps, 0.0, eps) for b in (-eps, 0.0, eps)]
            data = [pos_fn(a, b) for a, b in samples]
            k0, x0 = data[samples.index((0.0, 0.0))]
            P = (k0[0] - base[0], k0[1] - base[1])
            Q = (x0[0] - k0[0], x0[1] - k0[1])

            best = None
            for s_hip in (1.0, -1.0):
                for s_knee in (1.0, -1.0):
                    worst = 0.0
                    for (a, b), (k, x) in zip(samples, data):
                        th_h = s_hip * a
                        th_c = s_hip * a + s_knee * b
                        kk = _add(base, rot(P, th_h))
                        xx = _add(kk, rot(Q, th_c))
                        worst = max(worst, math.hypot(xx[0] - x[0], xx[1] - x[1]))
                    if best is None or worst < best[0]:
                        best = (worst, s_hip, s_knee)
            worst, s_hip, s_knee = best
            if worst > tol:
                raise RuntimeError(f"{side}/{label}: model does not fit the FK (residual {worst:.3e} m)")
            if verbose:
                print(
                    f"    {label:<5} |P|={math.hypot(*P):.7f} |Q|={math.hypot(*Q):.7f}"
                    f" s_hip={s_hip:+.0f} s_knee={s_knee:+.0f} residual={worst:.2e} m"
                )
            return P, Q, polar(P), polar(Q), s_hip, s_knee

        Pf, Qf, hf, cf, shf, skf = fit(A, front_pos, "front")
        Pb, Qb, hb, cb, shb, skb = fit(B, back_pos, "back")
        leg = LegPlanar(
            name=side,
            A=A,
            B=B,
            P=Pf,
            Q=Qf,
            S=Pb,
            T=Qb,
            ham0_front=hf,
            calf0_front=cf,
            ham0_back=hb,
            calf0_back=cb,
            s_hip_front=shf,
            s_knee_front=skf,
            s_hip_back=shb,
            s_knee_back=skb,
        )
        return leg, front_pos, back_pos


# --------------------------------------------------------------------------- #
# Loop-closure solver
# --------------------------------------------------------------------------- #
def solve_loop(
    leg: LegPlanar,
    q1: float,
    q4: float,
    seed: tuple[float, float] = (0.0, 0.0),
    iters: int = 60,
    tol: float = 1e-12,
) -> tuple[float, float, float, bool]:
    """Solve the knee pair ``(q5, q6)`` that closes the loop for given hips.

    Damped Newton on the 2-D closure residual.  Returns
    ``(q5, q6, residual_norm, converged)``.

    The solver is warm-startable: pass the previous step's solution as ``seed`` and
    it converges in one or two iterations.
    """
    x = [float(seed[0]), float(seed[1])]
    h = 1e-7

    def res(v):
        return leg.closure(q1, q4, v[0], v[1])

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
        try:
            dx = np.linalg.solve(J + 1e-12 * np.eye(2), -np.array(f0))
        except np.linalg.LinAlgError:
            break
        nrm = float(np.linalg.norm(dx))
        if nrm > 0.5:
            dx = dx * (0.5 / nrm)
        step = 1.0
        for _ in range(40):
            if math.hypot(*res((x[0] + step * dx[0], x[1] + step * dx[1]))) < err:
                break
            step *= 0.5
        x = [x[0] + step * float(dx[0]), x[1] + step * float(dx[1])]
    return x[0], x[1], math.hypot(*res(x)), False


def neutral_pose(
    leg: LegPlanar,
    leg_length: float | None = None,
    phi: float = 0.0,
    n_scan: int = 241,
    limits: dict | None = None,
):
    """Find HIP angles whose closed configuration sits at ``(leg_length, phi)``.

    There is no open-loop IK for this linkage that respects the knee limits: the
    two hips are not independent (the planar loop is a single scalar constraint),
    so an arbitrary ``(leg_length, phi)`` may simply not be reachable by a
    configuration that keeps the knees inside their limits.  This routine therefore
    SCANS the actuated hip space, keeps the closures that are both exact and
    inside ``limits``, and returns the feasible one closest to the request.

    ``limits`` is a mapping with keys ``front_hip``, ``back_hip``, ``front_knee``,
    ``back_knee`` -> ``(lower, upper)``.  When it is None the URDF limits must be
    supplied by the caller via :func:`leg_limits`.

    Returns ``(q_front_hip, q_back_hip, q_front_knee, q_back_knee, residual, L0, phi)``.
    Raises ``RuntimeError`` if no feasible closure is found at all.
    """
    if leg_length is None:
        mid = leg.hip_midpoint()
        _, w = leg.front(0.0, 0.0)
        leg_length = math.hypot(w[0] - mid[0], w[1] - mid[1])
    if limits is None:
        raise ValueError("neutral_pose needs `limits`; use leg_limits(urdf_path, side)")

    lo1, hi1 = limits["front_hip"]
    lo4, hi4 = limits["back_hip"]
    lo5, hi5 = limits["front_knee"]
    lo6, hi6 = limits["back_knee"]

    best = None
    # Seed each column of the scan from the previous solution so the solver stays
    # on the assembled branch instead of jumping to the mirrored closure.
    for q1 in np.linspace(lo1 + 1e-3, hi1 - 1e-3, n_scan):
        seed = (0.0, 0.0)
        for q4 in np.linspace(lo4 + 1e-3, hi4 - 1e-3, n_scan):
            q5, q6, res, _ = solve_loop(leg, float(q1), float(q4), seed=seed)
            if res > 1e-9:
                continue
            if not (lo5 <= q5 <= hi5 and lo6 <= q6 <= hi6):
                continue
            seed = (q5, q6)
            L0, ph = leg.leg_state(float(q1), q5)
            score = abs(L0 - leg_length) + 0.25 * abs(ph - phi)
            if best is None or score < best[0]:
                best = (score, float(q1), float(q4), q5, q6, L0, ph)

    if best is None:
        raise RuntimeError("no feasible closed configuration found inside the joint limits")
    _, q1, q4, q5, q6, L0, ph = best
    _, w = leg.front(q1, q5)
    _, e = leg.back(q4, q6)
    return q1, q4, q5, q6, math.dist(w, e), L0, ph


def leg_limits(urdf_path, side: str, joint_names: dict[str, str] | None = None) -> dict:
    """Read the four revolute joint limits of one leg out of the URDF."""
    tree = UrdfTree(urdf_path)
    j = tree.joints
    names = dict(joint_names or {})
    fh = names.get("front_hip", f"{side}_front_hip")
    bh = names.get("back_hip", f"{side}_back_hip")
    fk = names.get("front_knee", f"{side}_front_knee")
    bk = names.get("back_knee", f"{side}_back_knee")
    return {
        "front_hip": (j[fh]["lower"], j[fh]["upper"]),
        "back_hip": (j[bh]["lower"], j[bh]["upper"]),
        "front_knee": (j[fk]["lower"], j[fk]["upper"]),
        "back_knee": (j[bk]["lower"], j[bk]["upper"]),
    }


# --------------------------------------------------------------------------- #
# Batched torch version, for use inside the RL step
# --------------------------------------------------------------------------- #
def make_torch_loop_solver(leg: LegPlanar, device: str = "cpu", iters: int = 3, h: float = 1e-5):
    """Return ``solve(q1, q4, seed5, seed6) -> (q5, q6, residual)`` over torch batches.

    All tensors are shape ``(N,)``.  A fixed-iteration damped Newton solve, fully
    vectorised on ``device``, with an **analytic** Jacobian.

    Why analytic instead of finite differences
    ------------------------------------------
    This solve runs once per control step inside the action term, and it used to be the
    single largest source of CUDA kernel launches in the whole step: the finite
    difference Jacobian evaluated the closure residual three times per Newton iteration
    (``f0``, ``f0+dq5``, ``f0+dq6``), and each evaluation built four rotated vector
    pairs -- roughly 150 kernel launches per iteration, ~450 per step.  The analytic
    Jacobian needs exactly one closure evaluation per iteration (two rotations), so the
    launch count drops by well over half while producing the same Newton step.

    The closure is

    .. code-block:: text

        W(q1,q5) = A + R(P, shf*q1) + R(Q, shf*q1 + skf*q5)
        E(q4,q6) = B + R(S, shb*q4) + R(T, shb*q4 + skb*q6)
        f = W - E = 0

    so ``k1 = A + R(P, shf*q1)`` and ``k4 = B + R(S, shb*q4)`` do not depend on
    ``q5``/``q6`` and are hoisted out of the iteration, and

    .. code-block:: text

        dW/dq5 = skf * R'(Q, shf*q1 + skf*q5)
        dE/dq6 = skb * R'(T, shb*q4 + skb*q6)

    with ``R'(v, a) = d/da R(v, a) = (-vx sin a - vy cos a, vx cos a - vy sin a)``.

    Warm starting matters: ``iters=2`` is plenty when the seed is the previous step's
    solution (worst case residual 3.6e-6 m over 400 feasible hip pairs, versus the
    ~7.5e-4 m the simulation itself achieves).
    """
    import torch

    def T(v):
        return torch.tensor(v, dtype=torch.float32, device=device)

    A_, B_ = T(leg.A), T(leg.B)
    P_, Q_, S_, T_ = T(leg.P), T(leg.Q), T(leg.S), T(leg.T)
    shf, skf, shb, skb = leg.s_hip_front, leg.s_knee_front, leg.s_hip_back, leg.s_knee_back

    def rot2(v, a):
        c, s = torch.cos(a), torch.sin(a)
        return torch.stack((v[0] * c - v[1] * s, v[0] * s + v[1] * c), dim=-1)

    def drot2(v, a):
        """``d/da rot2(v, a)``, same shape, same cost as :func:`rot2`."""
        c, s = torch.cos(a), torch.sin(a)
        return torch.stack((-v[0] * s - v[1] * c, v[0] * c - v[1] * s), dim=-1)

    def solve(q1, q4, seed5, seed6, iters_: int = iters, max_step: float = 0.5):
        # parts that do not depend on q5/q6: hoisted out of the Newton loop
        ang5_base = shf * q1
        ang6_base = shb * q4
        k1 = A_ + rot2(P_, ang5_base)
        k4 = B_ + rot2(S_, ang6_base)

        q5 = seed5.clone()
        q6 = seed6.clone()
        for _ in range(iters_):
            a5 = ang5_base + skf * q5
            a6 = ang6_base + skb * q6
            f = (k1 + rot2(Q_, a5)) - (k4 + rot2(T_, a6))

            d_w = drot2(Q_, a5) * skf  # (N, 2) == dW/dq5
            d_e = drot2(T_, a6) * skb  # (N, 2) == dE/dq6
            j11, j21 = d_w[:, 0], d_w[:, 1]
            j12, j22 = -d_e[:, 0], -d_e[:, 1]  # f = W - E

            det = j11 * j22 - j12 * j21
            det = torch.where(det.abs() < 1e-6, torch.full_like(det, 1e-6), det)
            dq5 = (-j22 * f[:, 0] + j12 * f[:, 1]) / det
            dq6 = (j21 * f[:, 0] - j11 * f[:, 1]) / det
            nrm = torch.sqrt(dq5 * dq5 + dq6 * dq6).clamp_min(1e-9)
            scale = (max_step / nrm).clamp_max(1.0)
            q5 = q5 + dq5 * scale
            q6 = q6 + dq6 * scale

        residual = ((k1 + rot2(Q_, ang5_base + skf * q5)) - (k4 + rot2(T_, ang6_base + skb * q6))).norm(dim=-1)
        return q5, q6, residual

    return solve
