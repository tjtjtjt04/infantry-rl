"""Shared, measured data bridge for the chassis-rl five-bar robot.

Nothing in this module is hand-derived:

* the planar leg model, its rotation signs, the loop solver and the batched torch
  solver come from :mod:`kinematics.planar_leg`, fitted to the exact URDF FK;
* the numeric model constants and the nominal pose come from
  ``robot_data/chassis-rl/{planar_leg,nominal_pose}.json``, produced by
  ``tools/measure_chassis_asset.py``;
* the loop-anchor offsets come from the URDF itself.

The kinematics package is imported read-only: this file never writes to it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Robot asset (chassis-rl only)
# --------------------------------------------------------------------------- #
ASSETS_DIR = PROJECT_DIR / "chassis-rl"
CONFIG_DIR = PROJECT_DIR / "robot_data" / "chassis-rl"
URDF_PATH = ASSETS_DIR / "urdf" / "chassis-rl.urdf"
#: Full-fidelity CAD asset converted from ``chassis-rl.urdf``.
USD_PATH = ASSETS_DIR / "urdf" / "chassis-rl" / "chassis-rl.usd"
#: ``chassis-rl`` names the wheel JOINTS ``wheel_left`` / ``wheel_right``; the
#: links are still ``left_wheel`` / ``right_wheel``.
_WHEEL_JOINTS: tuple[str, str] = ("wheel_left", "wheel_right")

from kinematics.planar_leg import LegPlanar, make_torch_loop_solver

SIDES: tuple[str, ...] = ("left", "right")

# --------------------------------------------------------------------------- #
# Joint naming / ordering
# --------------------------------------------------------------------------- #
# The action vector and every per-leg computation use these hand-written orders.
# They are resolved against the articulation *by name* at runtime (never by a
# hard-coded index), see ``mdp.actions.FiveBarJointAction``.
HIP_ORDER: tuple[str, ...] = ("left_front_hip", "left_back_hip", "right_front_hip", "right_back_hip")
KNEE_ORDER: tuple[str, ...] = ("left_front_knee", "left_back_knee", "right_front_knee", "right_back_knee")
#: ``chassis-rl`` names the wheel *joints* ``wheel_left`` / ``wheel_right`` (the
#: links are still ``left_wheel`` / ``right_wheel``); the original asset uses the
#: link name for the joint too.
WHEEL_ORDER: tuple[str, ...] = _WHEEL_JOINTS

#: per-leg joint names, in the ``(q1, q4, q5, q6)`` order the model expects
LEG_JOINTS: dict[str, dict[str, str]] = {
    "left": {
        "front_hip": "left_front_hip",
        "back_hip": "left_back_hip",
        "front_knee": "left_front_knee",
        "back_knee": "left_back_knee",
        "wheel": _WHEEL_JOINTS[0],
        "back_calf": "left_back_calf",
    },
    "right": {
        "front_hip": "right_front_hip",
        "back_hip": "right_back_hip",
        "front_knee": "right_front_knee",
        "back_knee": "right_back_knee",
        "wheel": _WHEEL_JOINTS[1],
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
def squat_pose() -> dict:
    """Deepest feasible locked-hip pose for chassis-rl.

    Produced by ``tools/measure_chassis_asset.py`` and required at runtime.  Keys
    used by ``env_squat``:

    ``hips`` -- ``{"left": {front_hip, back_hip}, "right": {...}}`` absolute angles (rad)
    ``right_hip_sign`` -- the right-side joint values are ``sign * left`` for the same
                          physical pose (-1 for chassis-rl's mirrored right leg)
    ``leg_length``     -- hip-midpoint -> wheel-axle distance at that pose (m)
    ``base_height``    -- root height with the wheels on the ground (m)
    ``base_floor``     -- ``base_too_low`` termination height (m)
    ``pitch_limit`` / ``roll_limit`` -- attitude terminations (rad)
    """
    path = CONFIG_DIR / "squat_pose.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run tools/measure_chassis_asset.py first"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("right_hip_sign", -1.0)
    data.setdefault("leg_length", nominal_leg_length())
    data.setdefault("base_height", WHEEL_RADIUS + float(data["leg_length"]) - hip_axis_height())
    data.setdefault("base_floor", 0.04)
    data.setdefault("pitch_limit", 0.24)
    data.setdefault("roll_limit", 0.30)
    return data


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
    from kinematics.urdf_tree import UrdfTree  # noqa: PLC0415  (kept lazy: needs numpy only)

    tree = UrdfTree(URDF_PATH)
    # ``chassis-rl`` calls the wheel joint ``wheel_left`` but the loop is anchored
    # to the LINK ``left_wheel``, so compare against the joint's child link.
    wheel_link = {side: tree.joints[LEG_JOINTS[side]["wheel"]]["child"] for side in SIDES}
    out: dict[str, tuple[float, float, float]] = {}
    for loop in tree.loops:
        wheel = loop["link1"]
        for side in SIDES:
            if wheel_link[side] == wheel:
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


# --------------------------------------------------------------------------- #
# Fallen-pose ground contact geometry (measured, not guessed)
# --------------------------------------------------------------------------- #
#: Root height (m) at first ground contact vs ``|root pitch|`` (deg) for the locked
#: squat stance, measured by ``tools/probe_fallen_geometry.py`` with a 2 mm
#: lowering test against the real ``chassis-rl`` collision meshes.  A bounding box
#: is not accurate here: the leg links' local bounds are conservative and the
#: wheels are discs.
#:
#: Shape of the curve (this is what the fall-recovery task is built around):
#:
#: * 0-17 deg  -> the **wheels** touch first, and the height *decreases* as the
#:   robot tilts, because the root sits 1.8 cm above the wheel axle;
#: * 17-20 deg -> the chassis corner joins the two wheels: the "tripod" that the
#:   balance task's 0.24 rad pitch limit exists to stop just before;
#: * >=20 deg  -> **only** the chassis corner touches and the wheels are in the air
#:   (4.8 cm at 35 deg).  That pose is unstable: the CoM projects behind the corner,
#:   so gravity rotates the robot back down -- 30-40 deg is a *transient*, not a
#:   resting state.
FALLEN_CONTACT_TABLE: tuple[tuple[float, float], ...] = (
    (0.0, 0.1720),
    (5.0, 0.1720),
    (10.0, 0.1700),
    (15.0, 0.1680),
    (17.0, 0.1660),
    (20.0, 0.1720),
    (25.0, 0.1900),
    (30.0, 0.2060),
    (35.0, 0.2200),
    (40.0, 0.2360),
    (45.0, 0.2480),
    (60.0, 0.2700),
    (90.0, 0.3000),
)

#: measured file written by ``tools/probe_fallen_geometry.py``; overrides the table
FALLEN_GEOMETRY_PATH: Path = CONFIG_DIR / "fallen_geometry.json"


def _fallen_table() -> tuple[tuple[float, float], ...]:
    """``FALLEN_CONTACT_TABLE``, replaced by the measured JSON when it is present."""
    path = FALLEN_GEOMETRY_PATH
    if not path.is_file():
        return FALLEN_CONTACT_TABLE
    try:
        rows = json.loads(path.read_text(encoding="utf-8")).get("rows", [])
        table = sorted(
            (float(r["pitch_deg"]), float(r["contact_height"]))
            for r in rows
            if r.get("contact_height") is not None and math.isfinite(float(r["contact_height"]))
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[robot_model] WARNING: could not read {path} ({exc}); using the built-in table")
        return FALLEN_CONTACT_TABLE
    if len(table) < 3:
        return FALLEN_CONTACT_TABLE
    if table[0][0] > 0.0:
        table.insert(0, (0.0, table[0][1]))
    return tuple(table)


@lru_cache(maxsize=1)
def fallen_geometry() -> dict:
    """Ground-contact geometry of the retracted-leg stance.

    Returns a dict with

    * ``table`` -- ``[(pitch_deg, contact_height_m), ...]`` (absolute ``|pitch|``);
    * ``contact_height`` -- callable ``pitch_rad (float or tensor) -> height_m``,
      linear between the measured samples and clamped outside them;
    * ``source`` -- the JSON that was used, or ``None`` for the built-in table.

    The height is the root ``z`` at which the *lowest* collision geometry first
    touches ``z = 0``.  Callers add their own margin so the robot never starts
    inside the ground.
    """
    table = _fallen_table()
    angles = torch.tensor([p for p, _ in table], dtype=torch.float64)
    heights = torch.tensor([z for _, z in table], dtype=torch.float64)

    def contact_height(pitch):
        if isinstance(pitch, torch.Tensor):
            deg = pitch.abs() * (180.0 / math.pi)
            deg = deg.clamp(float(angles[0]), float(angles[-1]))
            a = angles.to(deg.device)
            h = heights.to(deg.device)
            idx = torch.searchsorted(a, deg, right=True).clamp(1, a.numel() - 1)
            a0, a1 = a[idx - 1], a[idx]
            h0, h1 = h[idx - 1], h[idx]
            return (h0 + (h1 - h0) * (deg - a0) / (a1 - a0)).to(torch.float32)
        deg = min(max(math.degrees(abs(float(pitch))), float(angles[0])), float(angles[-1]))
        for i in range(1, angles.numel()):
            a0, a1 = float(angles[i - 1]), float(angles[i])
            if deg <= a1:
                h0, h1 = float(heights[i - 1]), float(heights[i])
                return h0 + (h1 - h0) * (deg - a0) / (a1 - a0)
        return float(heights[-1])

    source = str(FALLEN_GEOMETRY_PATH) if FALLEN_GEOMETRY_PATH.is_file() else None
    return {"table": table, "contact_height": contact_height, "source": source}
