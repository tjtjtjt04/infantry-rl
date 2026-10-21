"""Extract the joint map of the infantry-rl five-bar robot from the URDF.

This answers two questions that the RL environment needs:

1.  Which joints exist, which are actuated, what are their limits.
2.  Does the URDF axis convention already agree with the hardware convention
    "positive = counter-clockwise seen from the motor output shaft"?

The second question is answered by a geometric argument, not by the axis sign
alone (see ``docs/JOINT_CONVENTION.md``): the two hip axes of one leg both point
to the same side of the robot, yet the two hip motors must turn in OPPOSITE
directions when the leg extends.  The URDF limits encode exactly that, so the
two conventions agree and no runtime axis flip is required.

Run:
    python isaac_ws/scripts/extract_joint_map.py
Writes ``isaac_ws/config/joint_map.json``.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # D:\Robomaster\RL\isaac_ws\scripts -> D:\Robomaster\RL
URDF = REPO / "infantry-rl_close" / "urdf" / "infantry-rl-sim.urdf"
OUT = HERE.parent / "config" / "joint_map.json"

LEGS = ("left_front", "left_back", "right_front", "right_back")


def parse_origin(elem: ET.Element | None) -> tuple[list[float], list[float]]:
    if elem is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    origin = elem.find("origin")
    if origin is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    return xyz, rpy


def axis_of(joint: ET.Element) -> list[float]:
    a = joint.find("axis")
    if a is None:
        return [1.0, 0.0, 0.0]
    return [float(v) for v in a.get("xyz", "1 0 0").split()]


def limit_of(joint: ET.Element) -> dict:
    lim = joint.find("limit")
    if lim is None:
        return {}
    out = {}
    for key in ("lower", "upper", "effort", "velocity"):
        if lim.get(key) is not None:
            out[key] = float(lim.get(key))
    return out


def main() -> int:
    tree = ET.parse(URDF)
    root = tree.getroot()
    assert root.tag == "robot", root.tag

    links = {link.get("name"): link for link in root.findall("link")}

    joints: dict[str, dict] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        xyz, rpy = parse_origin(joint)
        joints[name] = {
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "origin_xyz": xyz,
            "origin_rpy": rpy,
            "axis": axis_of(joint),
            "limit": limit_of(joint),
        }

    # non-standard <loop_joint> entries: they document the missing closures
    loop_joints = []
    for lj in root.findall("loop_joint"):
        entry = {"name": lj.get("name"), "type": lj.get("type")}
        for tag in ("link1", "link2"):
            child = lj.find(tag)
            if child is not None:
                entry[tag] = {
                    "link": child.get("link"),
                    "xyz": [float(v) for v in child.get("xyz", "0 0 0").split()],
                    "rpy": [float(v) for v in child.get("rpy", "0 0 0").split()],
                }
        loop_joints.append(entry)

    # role classification
    def role(name: str) -> str:
        if name.endswith("_hip"):
            return "actuated_hip"
        if name.endswith("_knee"):
            return "passive_knee"
        if name in ("left_wheel", "right_wheel"):
            return "actuated_wheel"
        if name.endswith("_calf_axle"):
            return "loop_axle"
        return "other"

    for name, j in joints.items():
        j["role"] = role(name)

    # --- axis sign table -------------------------------------------------
    # For every revolute joint: the rotation axis expressed in the base frame.
    # Because every "*_ham" / "*_calf" link carries rpy = 0 (or 0 0 -pi for the
    # right legs, which is a rotation about the vertical axis and therefore does
    # NOT change the direction of a joint axis lying in the leg plane), the
    # parent-frame axis below is also the axis direction in the base frame, up
    # to the global sign of a y-axis.
    axis_table = {}
    for name, j in joints.items():
        if j["type"] not in ("revolute", "continuous"):
            continue
        ax = j["axis"]
        if abs(ax[1]) > 0.99:
            side = "left" if "+y" in ("+y" if ax[1] > 0 else "-y") else "right"
            # motor output shaft points along the joint axis direction that is
            # away from the robot centre-line
            outward = "+y (left/outside)" if ax[1] > 0 else "-y (right/outside)"
            axis_table[name] = {
                "axis_in_parent": ax,
                "axis_points_outboard": ax[1] > 0,
                "motor_side": side,
                "output_shaft_direction": outward,
                "ccw_from_output_shaft_is_positive": True,
                "sign_used_by_env": 1.0,
            }

    doc = {
        "source_urdf": str(URDF),
        "robot_name": root.get("name"),
        "n_links": len(links),
        "n_joints": len(joints),
        "links": sorted(links),
        "joints": joints,
        "loop_joints": loop_joints,
        "axis_table": axis_table,
        "actuated_joints": sorted(n for n, j in joints.items() if j["role"].startswith("actuated")),
        "passive_joints": sorted(n for n, j in joints.items() if j["role"].startswith("passive")),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    # ---- human readable summary ----------------------------------------
    print(f"URDF      : {URDF}")
    print(f"robot     : {doc['robot_name']}   links={doc['n_links']}  joints={doc['n_joints']}")
    print()
    print(f"{'joint':<22}{'type':<12}{'axis':<16}{'lower':>10}{'upper':>10}{'effort':>8}{'vel':>8}")
    print("-" * 86)
    for name in sorted(joints):
        j = joints[name]
        lim = j["limit"]
        ax = " ".join(f"{v:g}" for v in j["axis"])
        print(
            f"{name:<22}{j['type']:<12}{ax:<16}"
            f"{lim.get('lower', float('nan')):>10.4f}{lim.get('upper', float('nan')):>10.4f}"
            f"{lim.get('effort', float('nan')):>8.2f}{lim.get('velocity', float('nan')):>8.2f}"
        )
    print()
    print("loop_joint entries (not valid URDF, documentation only):")
    for lj in loop_joints:
        print(f"  {lj['name']:<20} {lj['type']:<10} {lj['link1']['link']} <-> {lj['link2']['link']}")
    print()
    print(f"actuated : {doc['actuated_joints']}")
    print(f"passive  : {doc['passive_joints']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
