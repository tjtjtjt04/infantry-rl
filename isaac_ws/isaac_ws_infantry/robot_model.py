"""Shared, measured data bridge for the infantry-rl five-bar robot.

Nothing in this module is hand-derived:

* the planar leg model, its rotation signs, the loop solver and the batched torch
  solver come from :mod:`isaac_ws_infantry.kinematics.planar_leg`, which is
  verified against an exact double-precision URDF forward kinematics;
* the numeric model constants (``A, B, P, Q, S, T`` and the signs) come from the
  measured dump ``isaac_ws/config/planar_leg.json``;
* the nominal standing pose comes from ``isaac_ws/config/nominal_pose.json``;
* the loop-anchor offsets used to *measure* the closure error from simulated body
  poses come from the URDF itself (the same source ``planar_leg`` was fitted to).

The kinematics package is imported read-only: this file never writes to it.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

WS_DIR = Path(__file__).resolve().parents[1]  # .../isaac_ws
CONFIG_DIR = WS_DIR / "config"
ASSETS_DIR = WS_DIR / "assets" / "infantry-rl"
URDF_PATH = ASSETS_DIR / "urdf" / "infantry-rl-sim.urdf"

#: Full-fidelity CAD asset (about 96 MB; ``base_link.STL`` alone is 57 MB).
USD_PATH_FULL = ASSETS_DIR / "infantry_rl.usd"
#: Primitive proxy asset: same 11 links, same 10 joints, same axes and limits.
USD_PATH_PROXY = ASSETS_DIR / "infantry_rl_proxy.usd"

# --------------------------------------------------------------------------- #
# Which asset to train with
# --------------------------------------------------------------------------- #
# MEASURED on this machine (Isaac Sim 5.1.0.0), two separate findings:
#
# 1. The FULL CAD asset (~96 MB; ``base_link.STL`` alone is 57 MB) gives a
#    physically correct robot: with ``num_envs=1`` the loop-closure error stays at
#    0.79 mm max / 0.47 mm mean and the base holds its height with zero falls.
#    But it cannot be instantiated for more than ONE environment -- ``gym.make``
#    blocks forever in env construction with ``num_envs >= 2``.  Isolated with
#    ``isaac_ws/scripts/diag_clone.py``: NOT physics replication and NOT collision
#    filtering, both were toggled off independently and the hang persisted.
# 2. The PROXY asset (``build_proxy_asset.py``; primitives sized from each mesh's
#    bounding box, same 11 links / 10 joints) builds 256 environments in ~3 s, but
#    its crude collision boxes do not match the real link shapes, so the robot
#    falls (measured: 489 mm closure error, 448 terminations in 200 steps).  It is
#    a starting point for tuning collisions, NOT a drop-in replacement.
#
# So the default is the FULL asset, which is the validated one.  Use
# ``--num_envs 1`` with it.  To try the proxy, export INFANTRY_RL_ASSET=proxy.
# --------------------------------------------------------------------------- #
_ASSET_CHOICE = os.environ.get("INFANTRY_RL_ASSET", "full").strip().lower()
if _ASSET_CHOICE == "proxy" and USD_PATH_PROXY.is_file():
    USD_PATH = USD_PATH_PROXY
else:
    USD_PATH = USD_PATH_FULL

_KINEMATICS_DIR = Path(__file__).resolve().parent / "kinematics"
if str(_KINEMATICS_DIR) not in sys.path:
    sys.path.insert(0, str(_KINEMATICS_DIR))

from planar_leg import LegPlanar, make_torch_loop_solver  # noqa: E402

SIDES: tuple[str, ...] = ("left", "right")

# --------------------------------------------------------------------------- #
# Joint naming / ordering
# --------------------------------------------------------------------------- #
# The action vector and every per-leg computation use these hand-written orders.
# They are resolved against the articulation *by name* at runtime (never by a
# hard-coded index), see ``mdp.actions.FiveBarJointAction``.
HIP_ORDER: tuple[str, ...] = ("left_front_hip", "left_back_hip", "right_front_hip", "right_back_hip")
KNEE_ORDER: tuple[str, ...] = ("left_front_knee", "left_back_knee", "right_front_knee", "right_back_knee")
WHEEL_ORDER: tuple[str, ...] = ("left_wheel", "right_wheel")

#: per-leg joint names, in the ``(q1, q4, q5, q6)`` order the model expects
LEG_JOINTS: dict[str, dict[str, str]] = {
    "left": {
        "front_hip": "left_front_hip",
        "back_hip": "left_back_hip",
        "front_knee": "left_front_knee",
        "back_knee": "left_back_knee",
        "wheel": "left_wheel",
        "back_calf": "left_back_calf",
    },
    "right": {
        "front_hip": "right_front_hip",
        "back_hip": "right_back_hip",
        "front_knee": "right_front_knee",
        "back_knee": "right_back_knee",
        "wheel": "right_wheel",
        "back_calf": "right_back_calf",
    },
}

#: index of each side inside the flat 4-wide hip / knee / 2-wide wheel vectors
SIDE_SLICE: dict[str, slice] = {"left": slice(0, 2), "right": slice(2, 4)}
WHEEL_SLICE: dict[str, slice] = {"left": slice(0, 1), "right": slice(1, 2)}


# --------------------------------------------------------------------------- #
# Measured configuration files
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"measured configuration file '{path}' is missing; it is produced by the "
            f"kinematics verification scripts and is required by this task."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def nominal_pose() -> dict:
    """Measured nominal stance: ``{side: {front_hip, back_hip, front_knee, back_knee, L0, phi_deg}}``."""
    return _load_json(CONFIG_DIR / "nominal_pose.json")


#: Rolling radius of the wheel, measured from ``meshes/*_wheel.STL``
#: (max ``hypot(x, z)`` over the mesh vertices, the wheel spins about ``+/-y``).
WHEEL_RADIUS: float = 0.068


@lru_cache(maxsize=1)
def _planar_leg_json() -> dict:
    return _load_json(CONFIG_DIR / "planar_leg.json")


def hip_axis_height() -> float:
    """Height of the hip-axis midpoint above the ``base_link`` origin (m)."""
    data = _planar_leg_json()
    return float(sum(0.5 * (data[s]["A"][1] + data[s]["B"][1]) for s in SIDES) / len(SIDES))


def nominal_leg_length() -> float:
    """Measured nominal leg length ``L0`` (hip midpoint -> wheel axle), in metres."""
    nom = nominal_pose()
    return float(sum(nom[s]["L0"] for s in SIDES) / len(SIDES))


def nominal_base_height() -> float:
    """Root height at which the nominal stance puts the wheels exactly on ``z = 0``."""
    return WHEEL_RADIUS + nominal_leg_length() - hip_axis_height()


@lru_cache(maxsize=1)
def joint_limits() -> dict[str, tuple[float, float]]:
    """URDF joint limits in radians, keyed by joint name (wheels are continuous)."""
    data = nominal_pose()
    limits: dict[str, tuple[float, float]] = {}
    for side in SIDES:
        for role, key in (("front_hip", "front_hip"), ("back_hip", "back_hip"),
                          ("front_knee", "front_knee"), ("back_knee", "back_knee")):
            lo, hi = data[side]["limits"][role]
            limits[LEG_JOINTS[side][key]] = (float(lo), float(hi))
    limits[LEG_JOINTS["left"]["wheel"]] = (-float("inf"), float("inf"))
    limits[LEG_JOINTS["right"]["wheel"]] = (-float("inf"), float("inf"))
    return limits


@lru_cache(maxsize=1)
def loop_anchor_offsets() -> dict[str, tuple[float, float, float]]:
    """Offset of the loop anchor in the ``*_back_calf`` link frame, per side.

    Read from the URDF that ``planar_leg`` was fitted to (``<loop_joint>`` /
    ``link2/xyz``).  Its magnitude is ~0.2715 m on the left leg, which is exactly
    the ``|p(wheel) - p(back_calf)|`` distance measured in the PhysX probe.
    """
    from urdf_tree import UrdfTree  # noqa: PLC0415  (kept lazy: needs numpy only)

    tree = UrdfTree(URDF_PATH)
    out: dict[str, tuple[float, float, float]] = {}
    for loop in tree.loops:
        wheel = loop["link1"]
        for side in SIDES:
            if LEG_JOINTS[side]["wheel"] == wheel:
                out[side] = tuple(float(v) for v in loop["xyz2"])
    missing = [s for s in SIDES if s not in out]
    if missing:
        raise RuntimeError(f"could not find the loop anchor for {missing} in {URDF_PATH}")
    return out


# --------------------------------------------------------------------------- #
# Batched torch model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LegRuntime:
    """One leg: the verified planar model plus its batched torch loop solver."""

    side: str
    leg: LegPlanar
    solve: object  # callable(q1, q4, seed5, seed6) -> (q5, q6, residual)


class FiveBarModel:
    """Batched interface to the verified leg model, in the action-vector layout."""

    def __init__(self, device: str | torch.device = "cpu", iters: int = 3):
        data = _load_json(CONFIG_DIR / "planar_leg.json")
        self.device = torch.device(device)
        self.runtime: dict[str, LegRuntime] = {}
        for side in SIDES:
            leg = LegPlanar.from_dict(data[side])
            self.runtime[side] = LegRuntime(
                side=side,
                leg=leg,
                solve=make_torch_loop_solver(leg, device=str(self.device), iters=iters),
            )
        # planar geometry of each leg, as (2,) tensors in the base frame
        self._geom = {
            side: {
                key: torch.tensor(getattr(self.runtime[side].leg, key), dtype=torch.float32, device=self.device)
                for key in ("A", "B", "P", "Q", "S", "T")
            }
            for side in SIDES
        }
        nom = nominal_pose()
        self.nominal_hips = torch.tensor(
            [nom[s][r] for s in SIDES for r in ("front_hip", "back_hip")],
            dtype=torch.float32,
            device=self.device,
        )
        self.nominal_knees = torch.tensor(
            [nom[s][r] for s in SIDES for r in ("front_knee", "back_knee")],
            dtype=torch.float32,
            device=self.device,
        )
        self.nominal_leg_length = float(sum(nom[s]["L0"] for s in SIDES) / len(SIDES))

    # -- geometry ---------------------------------------------------------- #
    def _rot2(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(a), torch.sin(a)
        return torch.stack((v[0] * c - v[1] * s, v[0] * s + v[1] * c), dim=-1)

    def _chain(self, side: str, base_key: str, p_key: str, q_key: str, s_hip: float, s_knee: float,
               q_hip: torch.Tensor, q_knee: torch.Tensor):
        g = self._geom[side]
        th_h = s_hip * q_hip
        th_c = th_h + s_knee * q_knee
        k = g[base_key] + self._rot2(g[p_key], th_h)
        x = k + self._rot2(g[q_key], th_c)
        return k, x

    def front(self, side: str, q1: torch.Tensor, q5: torch.Tensor):
        """``(K1, W)``: front knee and wheel axle, in the base-frame (x, z) plane."""
        leg = self.runtime[side].leg
        return self._chain(side, "A", "P", "Q", leg.s_hip_front, leg.s_knee_front, q1, q5)

    def back(self, side: str, q4: torch.Tensor, q6: torch.Tensor):
        """``(K4, E)``: back knee and loop anchor, in the base-frame (x, z) plane."""
        leg = self.runtime[side].leg
        return self._chain(side, "B", "S", "T", leg.s_hip_back, leg.s_knee_back, q4, q6)

    def closure(self, side: str, q1: torch.Tensor, q4: torch.Tensor,
                q5: torch.Tensor, q6: torch.Tensor) -> torch.Tensor:
        """Loop residual ``W - E`` (metres, ``(N, 2)``) for the given joint angles."""
        _, w = self.front(side, q1, q5)
        _, e = self.back(side, q4, q6)
        return w - e

    def leg_state(self, side: str, q1: torch.Tensor, q5: torch.Tensor) -> torch.Tensor:
        """Firmware-style ``(L0, phi)`` of the wheel axle w.r.t. the hip-axis midpoint."""
        _, w = self.front(side, q1, q5)
        mid = 0.5 * (self._geom[side]["A"] + self._geom[side]["B"])
        d = w - mid
        length = torch.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        phi = torch.atan2(d[:, 0], -d[:, 1])
        return torch.stack((length, phi), dim=-1)

    # -- solving ----------------------------------------------------------- #
    def solve_knees(self, hips: torch.Tensor, seed5: torch.Tensor, seed6: torch.Tensor):
        """Solve all four knees from the four measured hip angles.

        Args:
            hips: ``(N, 4)`` measured hips in :data:`HIP_ORDER`.
            seed5, seed6: ``(N, 2)`` warm-start knee values, columns ``(left, right)``.

        Returns:
            ``(knees, residuals)`` with ``knees`` ``(N, 4)`` in :data:`KNEE_ORDER` and
            ``residuals`` ``(N, 2)`` = ``|W - E|`` per side.
        """
        q5_out, q6_out, res_out = [], [], []
        for side in SIDES:
            sl = SIDE_SLICE[side]
            q5, q6, res = self.runtime[side].solve(
                hips[:, sl.start], hips[:, sl.start + 1], seed5[:, SIDES.index(side)], seed6[:, SIDES.index(side)]
            )
            q5_out.append(q5)
            q6_out.append(q6)
            res_out.append(res)
        knees = torch.stack((q5_out[0], q6_out[0], q5_out[1], q6_out[1]), dim=-1)
        residuals = torch.stack(res_out, dim=-1)
        return knees, residuals

    def measured_closure(self, hips: torch.Tensor, knees: torch.Tensor) -> torch.Tensor:
        """Per-side ``|W - E|`` from *measured* joint angles (metres), shape ``(N, 2)``."""
        out = []
        for side in SIDES:
            sl = SIDE_SLICE[side]
            out.append(
                self.closure(side, hips[:, sl.start], hips[:, sl.start + 1],
                             knees[:, sl.start], knees[:, sl.start + 1]).norm(dim=-1)
            )
        return torch.stack(out, dim=-1)

    def measured_leg_state(self, hips: torch.Tensor, knees: torch.Tensor) -> torch.Tensor:
        """Per-side ``(L0, phi)`` from *measured* joint angles, shape ``(N, 4)``."""
        cols = []
        for side in SIDES:
            sl = SIDE_SLICE[side]
            cols.append(self.leg_state(side, hips[:, sl.start], knees[:, sl.start]))
        return torch.cat(cols, dim=-1)
