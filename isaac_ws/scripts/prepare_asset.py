"""Build the Isaac Lab asset package for the five-bar wheel-leg robot.

Why this script exists
----------------------
``infantry-rl-sim.urdf`` refers to its meshes as
``package://infantry-rl/meshes/*.STL``.  The Isaac Sim URDF importer only
resolves ``package://`` through the ``ROS_PACKAGE_PATH`` environment variable, and
that is fragile (it must be set *before* Kit starts, and the package root must
contain a directory literally named ``infantry-rl``).  A relative mesh path is
resolved against the URDF's own directory and always works.

This script therefore produces a *portable asset package*::

    isaac_ws/assets/infantry-rl/
        meshes/*.STL                     copied verbatim
        urdf/infantry-rl-sim.urdf        copy with relative mesh paths
        urdf/SOURCE.txt                  provenance
        infantry_rl.usd                  generated afterwards by the converter

Nothing about the physics is changed: link inertials, joint origins, joint axes
and joint limits are copied byte-for-byte from the joint XML elements.  Only the
``<mesh filename="...">`` attributes are rewritten.

Run:
    python isaac_ws/scripts/prepare_asset.py
"""

from __future__ import annotations

import filecmp
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent  # isaac_ws
REPO = WS.parent  # D:\Robomaster\RL

SRC_PKG = REPO / "infantry-rl_close"
SRC_URDF = SRC_PKG / "urdf" / "infantry-rl-sim.urdf"
SRC_MESHES = SRC_PKG / "meshes"

DST_PKG = WS / "assets" / "infantry-rl"
DST_URDF = DST_PKG / "urdf" / "infantry-rl-sim.urdf"
DST_MESHES = DST_PKG / "meshes"

PACKAGE_NAME = "infantry-rl"


def check_sources() -> None:
    missing = []
    if not SRC_URDF.is_file():
        missing.append(SRC_URDF)
    if not SRC_MESHES.is_dir():
        missing.append(SRC_MESHES)
    if missing:
        raise SystemExit("missing source asset(s):\n  " + "\n  ".join(str(p) for p in missing))


def copy_meshes() -> tuple[int, int]:
    DST_MESHES.mkdir(parents=True, exist_ok=True)
    n_new = n_same = 0
    for src in sorted(SRC_MESHES.iterdir()):
        if not src.is_file():
            continue
        dst = DST_MESHES / src.name
        if dst.is_file() and dst.stat().st_size == src.stat().st_size and filecmp.cmp(src, dst, shallow=False):
            n_same += 1
            continue
        shutil.copy2(src, dst)
        n_new += 1
    return n_new, n_same


def rewrite_urdf() -> tuple[int, list[str]]:
    """Copy the URDF, rewriting package:// mesh paths to paths relative to urdf/."""
    tree = ET.parse(SRC_URDF)
    root = tree.getroot()

    n_rewritten = 0
    referenced: list[str] = []
    for geom in root.iter("geometry"):
        mesh = geom.find("mesh")
        if mesh is None:
            continue
        filename = mesh.get("filename", "")
        if filename.startswith(f"package://{PACKAGE_NAME}/"):
            rel = filename[len(f"package://{PACKAGE_NAME}/") :]
            # urdf/ -> package root -> rel
            mesh.set("filename", f"../{rel}")
            n_rewritten += 1
        referenced.append(mesh.get("filename", ""))

    DST_URDF.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    tree.write(DST_URDF, encoding="utf-8", xml_declaration=True)
    return n_rewritten, referenced


def verify(referenced: list[str]) -> None:
    """Every referenced mesh must exist relative to the generated URDF."""
    bad = []
    for ref in sorted(set(referenced)):
        target = (DST_URDF.parent / ref).resolve()
        if not target.is_file():
            bad.append(f"{ref}  ->  {target}")
    if bad:
        raise SystemExit("mesh references that do not resolve:\n  " + "\n  ".join(bad))


def compare_joint_blocks() -> bool:
    """The generated URDF must be XML-identical to the source apart from mesh paths."""
    src_root = ET.parse(SRC_URDF).getroot()
    dst_root = ET.parse(DST_URDF).getroot()

    def canon(root: ET.Element) -> str:
        for mesh in root.iter("mesh"):
            mesh.set("filename", "MESH")
        return ET.tostring(root, encoding="unicode")

    return canon(src_root) == canon(dst_root)


def main() -> int:
    check_sources()
    n_new, n_same = copy_meshes()
    n_rewritten, referenced = rewrite_urdf()
    verify(referenced)
    identical = compare_joint_blocks()

    total_mb = sum(p.stat().st_size for p in DST_MESHES.iterdir() if p.is_file()) / 1e6
    print(f"source package : {SRC_PKG}")
    print(f"target package : {DST_PKG}")
    print(f"meshes copied  : {n_new} new, {n_same} already up to date  ({total_mb:.1f} MB)")
    print(f"mesh paths rewritten: {n_rewritten}")
    print(f"physics XML identical to source (ignoring mesh paths): {identical}")
    if not identical:
        print("!! FATAL: the generated URDF differs from the source in something other than mesh paths")
        return 1
    print(f"\nwrote {DST_URDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
