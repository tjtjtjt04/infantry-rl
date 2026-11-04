"""URDF parsing + exact forward kinematics (the ground truth for the leg model).

``chassis-rl.urdf`` describes each leg as a TREE plus a non-standard
``<loop_joint>`` that closes it into a parallel five-bar linkage.  URDF cannot express
a closed chain and neither can a PhysX articulation, so the only thing the URDF gives us
directly is a tree -- this module parses that tree and evaluates the exact
double-precision forward kinematics for it.

That exact FK is the reference every derived model is checked against:

* :mod:`isaac_ws_infantry.kinematics.planar_leg` (the runtime model) is fitted to it and
  verified to agree to < 1e-9 m;
* :func:`isaac_ws_infantry.robot_model.joint_limits` reads the URDF limits through
  :class:`UrdfTree`.

Only the tree + FK lives here.  The older numpy planar implementation that used to share
this file is in :mod:`isaac_ws_infantry.kinematics.reference_impl`.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Minimal exact URDF tree + forward kinematics
# --------------------------------------------------------------------------- #
def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
            [-sp, cp * sr, cp * cr, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _axis_angle(axis, q: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-15:
        return np.eye(4)
    x, y, z = a / n
    c, s, C = math.cos(q), math.sin(q), 1.0 - math.cos(q)
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s, 0.0],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s, 0.0],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rot_xz(v, a: float) -> tuple[float, float]:
    """Rotate an (x, z) offset by ``a``, standard CCW: ``[[c,-s],[s,c]]``.

    Defined so that ``rot_xz(v, a)`` moves the polar angle of ``v`` by exactly
    ``+a``.  That makes the model's joint bookkeeping literal: if a measurement
    says a link's polar angle changes at rate ``-1`` per unit joint value, the
    model simply writes ``a0 - q``.
    """
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def ang_xz(v) -> float:
    """The polar angle ``atan2(z, x)``; ``rot_xz(v, a - ang_xz(v))`` is axis aligned."""
    return math.atan2(v[1], v[0])


def _check_rot_ang() -> None:
    """Assert that rot_xz advances the polar angle by exactly its argument."""
    for v in ((0.3, -0.4), (0.145, -0.001), (-0.22, -0.15653)):
        for a in (0.0, 0.37, -1.1):
            r = rot_xz(v, a)
            d = (ang_xz(r) - (ang_xz(v) + a) + math.pi) % (2 * math.pi) - math.pi
            assert abs(d) < 1e-12, f"rot_xz angle drift for {v}, a={a}"
            assert abs(math.hypot(*r) - math.hypot(*v)) < 1e-12, f"rot_xz not norm preserving for {v}"


_check_rot_ang()


class UrdfTree:
    """Exact URDF forward kinematics; enough to derive and verify the model."""

    def __init__(self, path: str | Path):
        root = ET.parse(str(path)).getroot()
        self.joints: dict[str, dict] = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else "0 0 0".split())])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else "0 0 0".split())]
            a = j.find("axis")
            axis = np.array([float(v) for v in (a.get("xyz", "1 0 0").split() if a is not None else "1 0 0".split())])
            lim = j.find("limit")
            self.joints[j.get("name")] = {
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
                "type": j.get("type"),
                "lower": float(lim.get("lower", "-inf")) if lim is not None else -math.inf,
                "upper": float(lim.get("upper", "inf")) if lim is not None else math.inf,
                "effort": float(lim.get("effort", "0")) if lim is not None else 0.0,
                "velocity": float(lim.get("velocity", "0")) if lim is not None else 0.0,
            }
        self.loops: list[dict] = []
        for lj in root.findall("loop_joint"):
            l1, l2 = lj.find("link1"), lj.find("link2")
            self.loops.append(
                {
                    "name": lj.get("name"),
                    "type": lj.get("type"),
                    "link1": l1.get("link"),
                    "xyz1": np.array([float(v) for v in l1.get("xyz", "0 0 0").split()]),
                    "link2": l2.get("link"),
                    "xyz2": np.array([float(v) for v in l2.get("xyz", "0 0 0").split()]),
                }
            )

    def solve_frames(self, q: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        q = q or {}
        frames = {"base_link": np.eye(4)}
        remaining = dict(self.joints)
        while remaining:
            progressed = False
            for name, j in list(remaining.items()):
                if j["parent"] not in frames:
                    continue
                T = np.eye(4)
                T[:3, :3] = _rpy(*j["rpy"])[:3, :3]
                T[:3, 3] = j["xyz"]
                t = frames[j["parent"]] @ T
                if j["type"] in ("revolute", "continuous"):
                    t = t @ _axis_angle(j["axis"], q.get(name, 0.0))
                frames[j["child"]] = t
                del remaining[name]
                progressed = True
            if not progressed:
                raise RuntimeError(f"URDF is not a tree; unresolved joints: {list(remaining)}")
        return frames


