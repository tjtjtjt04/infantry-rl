"""Validate every URDF in the package: XML well-formedness, link/joint counts,
tree reachability, and the number of independent kinematic loops."""
import glob
import os
import xml.etree.ElementTree as ET

URDF_DIR = r"D:\Robomaster\RL\infantry-rl\urdf"

for p in sorted(glob.glob(os.path.join(URDF_DIR, "*.urdf"))):
    try:
        root = ET.parse(p).getroot()
    except Exception as e:
        print(f"FAIL {os.path.basename(p)}: {e}")
        continue
    links = {l.get("name") for l in root.findall("link")}
    joints = root.findall("joint")
    kids = {}
    for j in joints:
        kids.setdefault(j.find("parent").get("link"), []).append(
            (j.get("name"), j.find("child").get("link")))
    seen, queue, tree_joints = {"base_link"}, ["base_link"], 0
    while queue:
        link = queue.pop(0)
        for jn, ch in kids.get(link, []):
            if ch in seen:
                continue
            seen.add(ch)
            tree_joints += 1
            queue.append(ch)
    loops = len(joints) - tree_joints
    orphan = links - seen
    status = "OK  " if (not orphan and loops >= 0) else "FAIL"
    print(f"{status} {os.path.basename(p):32s} robot={root.get('name'):28s} "
          f"links={len(links):2d} joints={len(joints):2d} loops={loops}"
          + (f"  ORPHAN={sorted(orphan)}" if orphan else ""))
