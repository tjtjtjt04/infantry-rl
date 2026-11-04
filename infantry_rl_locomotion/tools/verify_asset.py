"""Verify the generated robot USD at the stage level.

Checks (no PhysX needed beyond the Kit runtime that hosts ``pxr``):

* the default prim and the articulation root exist;
* the 10 articulation joints are ``PhysicsRevoluteJoint`` prims with the expected
  names (hips, knees, wheels) and the URDF limits from ``robot_data``;
* the two ``loop_joints/*_calf_axle`` prims exist, are **active**
  ``PhysicsSphericalJoint`` constraints and stay outside the articulation
  (``physics:excludeFromArticulation = True``);
* ``config.yaml`` records the conversion parameters the original workspace used.

Run (the converter must have produced the asset first):
    python tools/verify_asset.py
    python build_asset.py --verify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
INFANTRY_USD = PROJECT_DIR / "assets" / "infantry-rl" / "infantry_rl.usd"
CHASSIS_USD = PROJECT_DIR / "chassis-rl" / "urdf" / "chassis-rl" / "chassis-rl.usd"
INFANTRY_DATA = PROJECT_DIR / "robot_data"
CHASSIS_DATA = PROJECT_DIR / "robot_data" / "chassis-rl"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify a converted robot USD.")
parser.add_argument("--asset", choices=("infantry", "chassis"), default="infantry",
                    help="which measured config / wheel-joint naming to verify against")
parser.add_argument("--usd", type=Path, default=None)
parser.add_argument("--report", type=Path, default=PROJECT_DIR / "tools" / "verify_asset_report.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics  # noqa: E402

SIDES = ("left", "right")
ROLES = ("front_hip", "back_hip", "front_knee", "back_knee")
DATA_DIR = CHASSIS_DATA if args_cli.asset == "chassis" else INFANTRY_DATA
DEFAULT_USD = CHASSIS_USD if args_cli.asset == "chassis" else INFANTRY_USD
if args_cli.usd is None:
    args_cli.usd = DEFAULT_USD
if args_cli.asset == "chassis":
    # chassis-rl: joint names differ from link names
    WHEEL_JOINTS = {"left": "wheel_left", "right": "wheel_right"}
else:
    WHEEL_JOINTS = {side: f"{side}_wheel" for side in SIDES}
EXPECTED_REVOLUTE = {
    f"{side}_{role}": (side, role) for side in SIDES for role in ROLES
} | {WHEEL_JOINTS[side]: (side, "wheel") for side in SIDES}
EXPECTED_LOOPS = {f"{side}_calf_axle": side for side in SIDES}

#: conversion parameters that must match the original workspace
EXPECTED_CONFIG = {
    "force_usd_conversion": True,
    "make_instanceable": True,
    "fix_base": False,
    "root_link_name": None,
    "link_density": 0.0,
    "merge_fixed_joints": True,
    "convert_mimic_joints_to_normal_joints": False,
    "collider_type": "convex_hull",
    "self_collision": False,
    "replace_cylinders_with_capsules": False,
    "collision_from_visuals": False,
}


def attr(prim, name, default=None):
    a = prim.GetAttribute(name)
    return a.Get() if a and a.IsValid() else default


def main() -> int:
    report: dict = {}
    failures: list[str] = []

    if not args_cli.usd.is_file():
        print(f"[verify] USD not found: {args_cli.usd}", flush=True)
        return 1
    stage = Usd.Stage.Open(str(args_cli.usd))
    if stage is None:
        print(f"[verify] could not open {args_cli.usd}", flush=True)
        return 1

    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        print("[verify] the USD has no default prim", flush=True)
        return 1
    root = default_prim.GetPath().pathString
    report["stage"] = str(args_cli.usd)
    report["default_prim"] = root

    joints_scope = stage.GetPrimAtPath(f"{root}/joints")
    loop_scope = stage.GetPrimAtPath(f"{root}/loop_joints")
    if not joints_scope or not joints_scope.IsValid():
        failures.append(f"missing joint scope {root}/joints")
        joints_scope = None
    if not loop_scope or not loop_scope.IsValid():
        failures.append(f"missing loop scope {root}/loop_joints")
        loop_scope = None

    # ------------------------------------------------------------------ #
    # 1. articulation joints
    # ------------------------------------------------------------------ #
    found_revolute: dict[str, dict] = {}
    if joints_scope is not None:
        for prim in joints_scope.GetChildren():
            name = prim.GetName()
            if prim.GetTypeName() != "PhysicsRevoluteJoint":
                failures.append(f"{name}: expected PhysicsRevoluteJoint, got {prim.GetTypeName()}")
                continue
            found_revolute[name] = {
                "lower_deg": attr(prim, "physics:lowerLimit"),
                "upper_deg": attr(prim, "physics:upperLimit"),
                "exclude_from_articulation": attr(prim, "physics:excludeFromArticulation", False),
            }
    report["revolute_joints"] = found_revolute

    if len(found_revolute) != 10:
        failures.append(f"expected 10 articulation joints, found {len(found_revolute)}")
    missing = set(EXPECTED_REVOLUTE) - set(found_revolute)
    extra = set(found_revolute) - set(EXPECTED_REVOLUTE)
    if missing:
        failures.append(f"missing joints: {sorted(missing)}")
    if extra:
        failures.append(f"unexpected joints: {sorted(extra)}")
    if any("calf_axle" in name for name in found_revolute):
        failures.append("a loop joint leaked into the articulation")

    # URDF limits (radians) against the USD values (degrees)
    nominal = json.loads((DATA_DIR / "nominal_pose.json").read_text(encoding="utf-8"))
    limit_errors = []
    for name, (side, role) in EXPECTED_REVOLUTE.items():
        if name not in found_revolute:
            continue
        if role == "wheel":
            lo, hi = found_revolute[name]["lower_deg"], found_revolute[name]["upper_deg"]
            if lo != float("-inf") or hi != float("inf"):
                limit_errors.append(f"{name}: expected continuous limits, got ({lo}, {hi})")
            continue
        lo_rad, hi_rad = (float(v) for v in nominal[side]["limits"][role])
        lo_deg, hi_deg = found_revolute[name]["lower_deg"], found_revolute[name]["upper_deg"]
        if abs(float(lo_deg) - lo_rad * 180.0 / 3.141592653589793) > 0.01:
            limit_errors.append(f"{name}: lower {lo_deg} deg != {lo_rad} rad")
        if abs(float(hi_deg) - hi_rad * 180.0 / 3.141592653589793) > 0.01:
            limit_errors.append(f"{name}: upper {hi_deg} deg != {hi_rad} rad")
    report["limit_errors"] = limit_errors
    failures.extend(limit_errors)

    # ------------------------------------------------------------------ #
    # 2. loop constraints -- must be ACTIVE and outside the articulation
    # ------------------------------------------------------------------ #
    found_loops: dict[str, dict] = {}
    if loop_scope is not None:
        for prim in loop_scope.GetChildren():
            name = prim.GetName()
            found_loops[name] = {
                "type": prim.GetTypeName(),
                "active": prim.IsActive(),
                "exclude_from_articulation": attr(prim, "physics:excludeFromArticulation", False),
                "joint_enabled": attr(prim, "physics:jointEnabled", False),
            }
    report["loop_constraints"] = found_loops

    if set(found_loops) != set(EXPECTED_LOOPS):
        failures.append(
            f"loop constraints: expected {sorted(EXPECTED_LOOPS)}, found {sorted(found_loops)}"
        )
    for name, info in found_loops.items():
        if info["type"] != "PhysicsSphericalJoint":
            failures.append(f"{name}: expected PhysicsSphericalJoint, got {info['type']}")
        if not info["active"]:
            failures.append(f"{name}: constraint is INACTIVE (the validated asset keeps it active)")
        if info["exclude_from_articulation"] is not True:
            failures.append(f"{name}: physics:excludeFromArticulation must be True")
        if not info["joint_enabled"]:
            failures.append(f"{name}: physics:jointEnabled is False")

    # ------------------------------------------------------------------ #
    # 3. articulation root
    # ------------------------------------------------------------------ #
    roots = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    report["articulation_roots"] = roots
    if len(roots) != 1:
        failures.append(f"expected exactly one articulation root, found {roots}")

    # ------------------------------------------------------------------ #
    # 4. conversion parameters
    # ------------------------------------------------------------------ #
    config = {}
    config_path = args_cli.usd.parent / "config.yaml"
    if config_path.is_file():
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        for key, expected in EXPECTED_CONFIG.items():
            actual = config.get(key)
            if actual != expected:
                failures.append(f"config.yaml {key}: expected {expected!r}, got {actual!r}")
        drive = config.get("joint_drive") or {}
        if drive.get("drive_type") != "force":
            failures.append(f"config.yaml joint_drive.drive_type: expected 'force', got {drive.get('drive_type')!r}")
        if drive.get("target_type") != "position":
            failures.append(f"config.yaml joint_drive.target_type: expected 'position', got {drive.get('target_type')!r}")
        gains = drive.get("gains") or {}
        for key in ("stiffness", "damping"):
            if float(gains.get(key, 1.0)) != 0.0:
                failures.append(f"config.yaml joint_drive.gains.{key}: expected 0.0, got {gains.get(key)!r}")
    else:
        failures.append(f"config.yaml not found next to the USD: {config_path}")
    report["config"] = config

    # ------------------------------------------------------------------ #
    # verdict
    # ------------------------------------------------------------------ #
    report["failures"] = failures
    report["passed"] = not failures

    print()
    print("[verify] ================ SUMMARY ================")
    print(f"[verify] stage           : {args_cli.usd}")
    print(f"[verify] default prim    : {root}")
    print(f"[verify] articulation    : {len(found_revolute)} revolute joints (expected 10)")
    print(f"[verify] loop constraints: {len(found_loops)} spherical, active="
          f"{all(i['active'] for i in found_loops.values())}, "
          f"excluded from articulation="
          f"{all(i['exclude_from_articulation'] for i in found_loops.values())}")
    print(f"[verify] articulation root: {roots}")
    args_cli.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[verify] report written   : {args_cli.report}")

    if failures:
        for failure in failures:
            print(f"[verify] FAIL: {failure}")
        print("[verify] RESULT: FAIL")
        return 1
    print("[verify] RESULT: PASS")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
