"""Generate the Isaac Sim URDF for the infantry-rl five-bar wheel-leg.

Source (read-only):  infantry-rl/urdf/infantry-rl.urdf   -- the raw SW2URDF export, an
                     already-legal 11-link / 10-joint TREE.  This is the file whose meshes
                     land in the right places; it is the ground truth.

Why not the closed one:  `infantry-rl-closed.urdf` "corrected" the right leg into an exact
mirror of the left.  That is wrong.  Forcing the joint origins to mirror moves every link
frame, and the meshes hang off those frames -- so the right leg's geometry moves with them.
Measured against the left leg's meshes, the closed file's right thigh lands 258 mm away
(its `right_back_hip` rpy was changed from `0 0 -pi` to `0 0 0` without rotating the mesh),
and its right front calf 26.8 mm.  The original lands within 5-7 mm, which is the real CAD's
own small asymmetry.  See tools/check_mesh_alignment.py.

Three transformations, nothing else is touched:

  1. Axis signs, so both legs use one consistent convention ("writing B": left and right
     carry the same sign; the hardware sign map is a deploy-time table, not part of the
     model).  `<axis>` is written in the joint's own frame, and both right hips carry
     `rpy="0 0 -pi"`, so anything below them expressed as `0 -1 0` means `+Y` in the base
     frame -- the same as the left.  `right_*_knee` and `right_wheel` were exported without
     that conjugation and must be flipped to `0 1 0`.
     The `9.9049E-05` on the hip axes is SW noise (0.006 deg); cleaned to exactly `0 1 0`.

  2. Joint limits swapped on the right leg.  The export gives `right_front_hip` the *back*
     hip's range and vice versa (same for the knees).  Ranges are re-assigned by mechanical
     role, then re-referenced (see 3).

  3. Limits re-referenced onto the joint zero.  The URDF zero is the CAD assembly pose, but
     the exported `<limit>` values are written relative to the downstream link's *absolute*
     orientation in the leg plane, not relative to that zero.  Taken literally, `q = 0`
     falls outside its own limits on 3 of the 4 joint types and PhysX snaps the robot on the
     first step.  The offset is measured at runtime below (no magic numbers) and subtracted.
     The physical workspace is unchanged -- only the labels move.

  4. Right front knee / wheel hinge origins corrected, with a matching mesh <origin> so the
     parts stay where SolidWorks put them.  The raw export has the mate-derived origin
     26.65 mm / 15.88 mm away from the true hinge while the mesh compensates, so the raw file
     renders right but is kinematically wrong.  See CORRECTED_ORIGINS below.

The closed loop is then added as a `<loop_joint>` -- the Isaac Sim URDF extension, which the
importer turns into a PhysicsSphericalJoint outside the articulation that PhysX enforces as a
real constraint.  Pass --no-loop for a plain tree instead (that variant needs a Python-side
loop solver, and is only kept for comparison).

Run:  python build_sim_urdf.py [--no-loop]

Reference for the extension:
  isaacsim.asset.importer.urdf/docs/README.md  ("Limitations" section)
  isaacsim.asset.importer.urdf/docs/CHANGELOG.md  (2.1.0, 2024-10-14)
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = r"D:\IsaacLab-main\parallel_leg\infantry-rl-main\infantry-rl-main\infantry-rl\urdf\infantry-rl.urdf"
DST = os.path.normpath(os.path.join(HERE, "..", "urdf", "infantry-rl-sim.urdf"))

# axis each joint must carry in its own frame, for writing B
FIXED_AXES = {
    "left_front_hip": (0.0, 1.0, 0.0),
    "right_front_hip": (0.0, -1.0, 0.0),    # frame rotated pi -> +Y in base
    "left_back_hip": (0.0, 1.0, 0.0),
    "right_back_hip": (0.0, -1.0, 0.0),     # frame rotated pi -> +Y in base
    "left_front_knee": (0.0, -1.0, 0.0),
    "right_front_knee": (0.0, 1.0, 0.0),    # was 0 -1 0: not conjugated -> wrong sign
    "left_back_knee": (0.0, -1.0, 0.0),
    "right_back_knee": (0.0, 1.0, 0.0),     # was 0 -1 0: same bug
    "left_wheel": (0.0, -1.0, 0.0),
    "right_wheel": (0.0, 1.0, 0.0),         # was 0 -1 0: same bug
}

# mechanical role of each joint -- decides which range it gets
ROLE = {
    "left_front_hip": "front_hip", "right_front_hip": "front_hip",
    "left_back_hip": "back_hip", "right_back_hip": "back_hip",
    "left_front_knee": "front_knee", "right_front_knee": "front_knee",
    "left_back_knee": "back_knee", "right_back_knee": "back_knee",
}

# real mechanical travel, quoted in the absolute-link-angle reference (deg).
# taken from the left leg, which is the reference for both legs.
ROLE_RANGES_DEG = {
    "front_hip": (-10.150, 75.178),
    "back_hip": (75.178, 190.148),
    "front_knee": (39.350, 153.891),
    "back_knee": (26.110, 140.650),
}

# Right front knee / wheel: the SW mate-derived origins are 26.65 mm and 15.88 mm away from
# the true hinge, while the mesh files were authored in a frame that compensates exactly.
# That is why the raw export renders correctly but is kinematically wrong -- and why forcing
# the mirrored origin without touching the mesh (which infantry-rl-closed.urdf did) renders
# wrong.  Evidence, against the left leg as reference, is the distance from a link's own
# hinge to the nearest vertex of its mesh:
#     left front calf 22.07 mm   right front calf: 4.41 (raw)  22.07 (mirrored + offset)
#     left wheel      51.02 mm   right wheel:     35.98 (raw)  51.13 (mirrored + offset)
# Both changes are needed together: the origin puts the hinge on the part, the visual/collision
# <origin> (computed below, not hardcoded) keeps the mesh where SolidWorks had it.
# The back leg is fine as exported and is left alone.
CORRECTED_ORIGINS = {
    "right_front_knee": (-0.145, 0.0275, -0.00099998),   # mirror of left_front_knee
    "right_wheel": (0.22, 0.023445, -0.15653),           # mirror of left_wheel
}

# [this joint's origin -> distal joint's origin] whose XZ direction is the reference the
# limits are written in
REFERENCE_DIR = {
    "left_front_hip": ("left_front_hip", "left_front_knee"),
    "left_back_hip": ("left_back_hip", "left_back_knee"),
    "left_front_knee": ("left_front_knee", "left_wheel"),
    "left_back_knee": ("left_back_knee", "left_wheel"),
    "right_front_hip": ("right_front_hip", "right_front_knee"),
    "right_back_hip": ("right_back_hip", "right_back_knee"),
    "right_front_knee": ("right_front_knee", "right_wheel"),
    "right_back_knee": ("right_back_knee", "right_wheel"),
}


parser = argparse.ArgumentParser()
parser.add_argument("--no-loop", action="store_true",
                    help="emit a plain tree (no <loop_joint>); the articulation then needs a"
                         " Python-side loop solver")
NO_LOOP = parser.parse_args().no_loop


def rpy_to_mat(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def tf(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


root = ET.parse(SRC).getroot()
links = {l.get("name") for l in root.findall("link")}
joints, by_parent = {}, {}
for j in root.findall("joint"):
    o = j.find("origin")
    rec = dict(el=j, name=j.get("name"), type=j.get("type"),
               parent=j.find("parent").get("link"), child=j.find("child").get("link"),
               xyz=np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]),
               rpy=[float(v) for v in (o.get("rpy") or "0 0 0").split()])
    joints[rec["name"]] = rec
    by_parent.setdefault(rec["parent"], []).append(rec)

children = {j["child"] for j in joints.values()}
roots = sorted(links - children)
assert roots == ["base_link"], roots
print(f"source: {os.path.basename(SRC)}   links={len(links)} joints={len(joints)} "
      f"(already a tree, no loop joints to drop)")

def build_fk(overrides):
    """Link frames at q = 0, with selected joint origins replaced."""
    out = {}

    def rec(link, T):
        out[link] = T
        for j in by_parent.get(link, []):
            xyz = np.array(overrides.get(j["name"], j["xyz"]), float)
            rec(j["child"], T @ tf(rpy_to_mat(*j["rpy"]), xyz))

    rec("base_link", np.eye(4))
    return out


W_RAW = build_fk({})                    # joint origins exactly as exported
W = build_fk(CORRECTED_ORIGINS)         # with the hinges put back on the parts


def joint_origin(name):
    """Hinge position in the CORRECTED model -- W[child] already folds in the origin."""
    return W[joints[name]["child"]][:3, 3]


print("\n=== loop closure at the assembled pose (q = 0) ===")
for leg in ("left", "right"):
    knee = joint_origin(f"{leg}_back_knee")
    wheel = joint_origin(f"{leg}_wheel")
    d = np.linalg.norm(wheel - knee)
    print(f"  {leg:5s} |wheel - back_knee| = {d:.6f} m   "
          f"(back calf length; both legs must share it -- same part)")

print("\n=== re-referencing the limits onto the joint zero ===")
print(f"  {'joint':18s} {'axis(base)':>11s} {'a0 [deg]':>9s} {'old limits':>21s} {'new limits':>21s}")
NEW_LIMITS = {}
for jname, ref in REFERENCE_DIR.items():
    j = joints[jname]
    lo_a, hi_a = [math.radians(v) for v in ROLE_RANGES_DEG[ROLE[jname]]]
    a0 = math.atan2(-(joint_origin(ref[1]) - joint_origin(ref[0]))[2],
                    (joint_origin(ref[1]) - joint_origin(ref[0]))[0])
    axis_base = W[j["parent"]][:3, :3] @ rpy_to_mat(*j["rpy"]) @ np.array(FIXED_AXES[jname])
    lo, hi = (lo_a - a0, hi_a - a0) if axis_base[1] > 0 else (a0 - hi_a, a0 - lo_a)

    old = j["el"].find("limit")
    old_s = f"[{float(old.get('lower')):+.3f},{float(old.get('upper')):+.3f}]"
    margin = math.degrees(min(0.0 - lo, hi - 0.0))
    warn = "  <-- q=0 only %.3f deg from a stop!" % margin if margin < 0.5 else ""
    NEW_LIMITS[jname] = (lo, hi)
    print(f"  {jname:18s} {axis_base[1]:+11.4f} {math.degrees(a0):9.3f} {old_s:>21s} "
          f"[{lo:+.4f},{hi:+.4f}]{warn}")

# ---- emit -------------------------------------------------------------------------
def fmt(v):
    return " ".join(f"{x:g}" for x in v)


def indent(el, level):
    pad = "\n" + "  " * level
    if len(el):
        if not (el.text or "").strip():
            el.text = pad + "  "
        for i, c in enumerate(el):
            indent(c, level + 1)
            c.tail = pad + "  " if i < len(el) - 1 else pad
    return el


for j in root.findall("joint"):
    name = j.get("name")
    if name not in FIXED_AXES:
        continue
    if name in CORRECTED_ORIGINS:
        j.find("origin").set("xyz", fmt(CORRECTED_ORIGINS[name]))
    j.find("axis").set("xyz", fmt(FIXED_AXES[name]))
    limit = j.find("limit")
    if j.get("type") == "continuous":
        limit.attrib.pop("lower", None)      # SW writes lower=0 upper=0 for continuous
        limit.attrib.pop("upper", None)
    else:
        lo, hi = NEW_LIMITS[name]
        limit.set("lower", f"{lo:.6f}")
        limit.set("upper", f"{hi:.6f}")

# keep every mesh where SolidWorks placed it: <visual>/<collision> origin = T_new^-1 . T_old
print("\n=== mesh compensation (visual/collision <origin>) ===")
for link_el in root.findall("link"):
    link = link_el.get("name")
    O = np.linalg.inv(W[link]) @ W_RAW[link]
    if np.linalg.norm(O[:3, 3]) < 1e-9 and np.allclose(O[:3, :3], np.eye(3)):
        continue
    xyz = O[:3, 3]
    print(f"  {link:18s} origin xyz = ({xyz[0]:+.6f}, {xyz[1]:+.6f}, {xyz[2]:+.6f})  "
          f"|{np.linalg.norm(xyz)*1000:.2f} mm|")
    for tag in ("visual", "collision"):
        el = link_el.find(tag)
        if el is not None:
            el.find("origin").set("xyz", f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")

# ---- closed chain: the Isaac Sim <loop_joint> URDF extension ------------------------
# URDF only describes trees.  isaacsim.asset.importer.urdf (>= 2.1.0) adds a custom
# <loop_joint> tag that the importer turns into a PhysicsSphericalJoint OUTSIDE the
# articulation, which PhysX then enforces as a real constraint.  Verified on 5.1.0:
# the articulation keeps its 10 tree DOFs and the axle hole stays on the wheel centre
# to ~1.5 um while the mechanism moves (tools/test_loop_physx.py).
#
# Each link contributes a point in its OWN link frame; the two must coincide.  For a
# five-bar leg that point is the wheel axle centre.
if not NO_LOOP:
    for leg in ("left", "right"):
        wheel, calf = f"{leg}_wheel", f"{leg}_back_calf"
        axle_w = W[wheel][:3, 3]
        on_wheel = (np.linalg.inv(W[wheel]) @ np.append(axle_w, 1.0))[:3]
        on_calf = (np.linalg.inv(W[calf]) @ np.append(axle_w, 1.0))[:3]
        lj = ET.SubElement(root, "loop_joint")
        lj.set("name", f"{leg}_calf_axle")
        lj.set("type", "spherical")
        for tag, link, xyz in (("link1", wheel, on_wheel), ("link2", calf, on_calf)):
            e = ET.SubElement(lj, tag)
            e.set("link", link)
            e.set("rpy", "0 0 0")
            e.set("xyz", " ".join(f"{v:.8f}" for v in xyz))
        print(f"  loop_joint {leg:5s}  {wheel:12s} {np.round(on_wheel,6)}"
              f"   {calf:15s} {np.round(on_calf,6)}")

root.set("name", "infantry-rl-sim")

tree = ET.ElementTree(indent(root, 0))
os.makedirs(os.path.dirname(DST), exist_ok=True)
tree.write(DST, encoding="utf-8", xml_declaration=True)

with open(DST, "r", encoding="utf-8") as fh:
    text = fh.read()
text = text.replace(
    "<?xml version='1.0' encoding='utf-8'?>",
    "<?xml version='1.0' encoding='utf-8'?>\n"
    "<!--\n"
    "  infantry-rl-sim : Isaac Sim 5.1.0 asset for the five-bar wheel-leg.\n"
    "  GENERATED by assets/infantry-rl/tools/build_sim_urdf.py . Do not hand-edit.\n"
    "  Derived from infantry-rl/urdf/infantry-rl.urdf (the raw SW export, a legal tree).\n"
    "  Changes, and nothing else:\n"
    "    * right_*_knee / right_wheel axis signs flipped so both legs share one sign\n"
    "      convention (their parent frames carry rpy 0 0 -pi; see the generator docstring)\n"
    "    * hip axis noise 9.9049E-05 cleaned to 0\n"
    "    * right-leg joint limit ranges re-assigned by mechanical role (the export had\n"
    "      front and back swapped on that leg)\n"
    "    * limits re-referenced onto the joint zero of the assembled pose\n"
    "  Geometry, inertias and meshes are untouched.\n"
    "  The closed loop is NOT represented here: a PhysX articulation cannot close a loop.\n"
    "  It is enforced by the five-bar solver in the environment (tools/five_bar.py).\n"
    "  See assets/infantry-rl/README.md .\n"
    "-->")
with open(DST, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(text)

print(f"\nwrote {DST}")
print(f"  links={len(root.findall('link'))}  joints={len(root.findall('joint'))}")
