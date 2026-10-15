"""Build the final closed-loop URDF.

Two problems with the raw SW2URDF export:
  1. the loop is open (the rear calf floats),
  2. the RIGHT leg is neither a mirror of the left one nor internally
     consistent (e.g. `right_front_hip` lands at world x = -0.075 instead of
     +0.075, `right_front_knee` origin is (-0.14273, 0.0275, +0.025549) with
     rpy 0, and `right_wheel` is off by 14 mm), so the two legs are not
     kinematically identical.

This script writes a single file:
  infantry-rl-closed.urdf  - original tree + loop-closure joints, with the right
                             leg symmetrised against the left (the reference).
The untouched export stays at infantry-rl.urdf.
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinematics import Robot  # noqa


class RobotFromString(Robot):
    """Robot built from an in-memory XML string (same code path as a file)."""

    def __init__(self, xml_text):
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".urdf")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(xml_text)
        try:
            super().__init__(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

ROOT = r"D:\Robomaster\RL\infantry-rl"
SRC = os.path.join(ROOT, "urdf", "infantry-rl.urdf")
DST = os.path.join(ROOT, "urdf", "infantry-rl-closed.urdf")

IND = "  "


def pretty(elem, level=0):
    pad = "\n" + IND * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + IND
        for i, ch in enumerate(elem):
            pretty(ch, level + 1)
            ch.tail = pad + IND if i < len(elem) - 1 else pad
    elif level and not (elem.tail or "").strip():
        elem.tail = pad


def fmt(v):
    return " ".join("%.6g" % x for x in v)


def mirror_rpy(rpy):
    """Mirror an rpy about the xz plane (y -> -y)."""
    r, p, y = rpy
    return [-r, p, -y]


def loop_frames(rob, side):
    """(origin, rpy) of the loop joint, expressed in the wheel link's own frame.

    parent = wheel link, child = rear calf.  At the assembly pose the child frame
    must land exactly on the rear calf's own frame, so

        origin = R_wheel^T (p_rear_calf - p_wheel)
        R_joint = R_wheel^T R_rear_calf        -> rpy

    The rpy is NOT always zero: on the raw export the right wheel's link frame is
    rotated 180 deg about z, so the joint has to carry that 180 deg to place the
    rear calf's frame correctly.
    """
    q0 = {n: 0.0 for n in rob.joints}
    Rw, pw = rob.pose(f"{side}_wheel", q0)
    Rc, pc = rob.pose(f"{side}_back_calf", q0)
    return Rw.T @ (pc - pw), R_to_rpy(Rw.T @ Rc)


def R_to_rpy(R):
    """Fixed-axis roll-pitch-yaw such that R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(R[2, 0]) < 1 - 1e-12:
        return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])
    return math.atan2(-R[1, 2], R[1, 1]), pitch, 0.0


# --- loop-closure joints -----------------------------------------------------
# (name, parent, child, effort, velocity).  Origin/axis are derived below.
LOOPS = [
    ("left_back_calf_axle", "left_wheel", "left_back_calf", 40.0, 16.755),
    ("right_back_calf_axle", "right_wheel", "right_back_calf", 40.0, 16.755),
]

IND = "  "
# --- right leg, made an exact mirror of the left -----------------------------
#
# The right leg's meshes are the left meshes rotated 180 deg about z, so the
# right hips keep the exported frame convention (front hip rpy = (0,0,-pi),
# back hip rpy = 0).  Inside that convention the mirrored local values are
#     front limb:  xyz -> ( x, -y,  z)   of the LEFT value
#     back  limb:  xyz -> ( x, -y,  z)   of the LEFT value
# (derived mechanically in tools/_derive_mirror.py; it reproduces
#  |M p_left - p_right| = 0 for every link).
MIRROR = {
    # front limb: flipped frame, so the LEFT xyz is used with y negated and the
    # parent's Rz(pi) supplies the rest of the mirror
    "right_front_hip": dict(xyz=[0.075033, -0.168, 0.128], rpy=[0, 0, -3.14159265358979],
                            axis=[-9.9049e-05, -1, 0], lower=-0.17715, upper=1.3121),
    "right_front_knee": dict(xyz=[-0.145, 0.0275, -0.00099998], rpy=[0, 0, 0],
                             axis=[0, -1, 0], lower=0.68679, upper=2.6859),
    "right_wheel": dict(xyz=[0.22, 0.023445, -0.15653], rpy=[0, 0, 0],
                        axis=[0, -1, 0]),
    # back limb: unflipped frame, only y is negated
    "right_back_hip": dict(xyz=[-0.074967, -0.1655, 0.128], rpy=[0, 0, 0],
                           axis=[9.9049e-05, 1, 0], lower=1.3121, upper=3.3187),
    "right_back_knee": dict(xyz=[-0.145, -0.025, -0.00099998], rpy=[0, 0, 0],
                            axis=[0, -1, 0], lower=0.45571, upper=2.4548),
    "right_back_calf_axle": dict(xyz=[0.0, 0.0, 0.0], rpy=[0, 0, 0],
                                 axis=[0, -1, 0]),
}


def add_joint(root, name, jtype, parent, child, xyz, rpy, axis,
              lower=None, upper=None, effort=None, velocity=None):
    j = ET.SubElement(root, "joint")
    j.set("name", name)
    j.set("type", jtype)
    o = ET.SubElement(j, "origin")
    o.set("xyz", fmt(xyz))
    o.set("rpy", fmt(rpy))
    ET.SubElement(j, "parent").set("link", parent)
    ET.SubElement(j, "child").set("link", child)
    ET.SubElement(j, "axis").set("xyz", fmt(axis))
    if lower is not None:
        lim = ET.SubElement(j, "limit")
        lim.set("lower", "%.6g" % lower)
        lim.set("upper", "%.6g" % upper)
        if effort is not None:
            lim.set("effort", "%.6g" % effort)
        if velocity is not None:
            lim.set("velocity", "%.6g" % velocity)
    return j


def build(dst, sym_right, name_suffix):
    tree = ET.parse(SRC)
    root = tree.getroot()
    base_name = root.get("name")

    # 1) symmetrise the right leg in place, if requested
    if sym_right:
        for j in root.findall("joint"):
            n = j.get("name")
            if n not in MIRROR:
                continue
            m = MIRROR[n]
            o = j.find("origin")
            o.set("xyz", fmt(m["xyz"]))
            o.set("rpy", fmt(m["rpy"]))
            j.find("axis").set("xyz", fmt(m["axis"]))
            lim = j.find("limit")
            if lim is not None and "lower" in m:
                lim.set("lower", "%.6g" % m["lower"])
                lim.set("upper", "%.6g" % m["upper"])

    # 2) add the loop-closure joints, with origins derived from the (possibly
    #    symmetrised) kinematics so the loop is closed exactly at q = 0
    pretty(root)
    xml_txt = ET.tostring(root, encoding="unicode")
    rob = RobotFromString(xml_txt)
    for name, parent, child, eff, vel in LOOPS:
        side = name.split("_")[0]
        origin, rpy = loop_frames(rob, side)
        add_joint(root, name, "revolute", parent, child, origin, rpy,
                  [0.0, -1.0, 0.0],
                  -3.14159265358979, 3.14159265358979, eff, vel)

    pretty(root)
    root.set("name", base_name + name_suffix)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    print(f"wrote {os.path.basename(dst)}")
    rob2 = Robot(dst)
    for name, parent, child, _e, _v in LOOPS:
        side = name.split("_")[0]
        o, r = loop_frames(rob2, side)
        print(f"   + {name}: {parent} -> {child}")
        print(f"       origin ({o[0]:+.6f}, {o[1]:+.6f}, {o[2]:+.6f})  "
              f"rpy ({r[0]:+.6f}, {r[1]:+.6f}, {r[2]:+.6f})")


if __name__ == "__main__":
    build(DST, sym_right=True, name_suffix="-closed")
