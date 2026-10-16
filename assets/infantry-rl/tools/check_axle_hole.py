"""Does the back calf's axle hole actually sit on the wheel centre at the assembled pose?

This is the one test that does not assume the two legs are mirror images: it only uses the
calf mesh and the wheel centre, both of which are fixed by the URDF's own link frames.

Expectation for a correctly exported assembly: the wheel centre lands in the middle of a
HOLED region, so very few mesh vertices are within a few mm of it, and the nearest ones sit
out at the hole radius.  If it lands in solid material, the export has the calf misplaced.
"""

import math
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

BASE = r"D:\IsaacLab-main\parallel_leg\infantry-rl-main\infantry-rl-main\infantry-rl"
MESH_DIR = os.path.join(BASE, "meshes")

FILES = {
    "original  infantry-rl.urdf": os.path.join(BASE, "urdf", "infantry-rl.urdf"),
    "closed    infantry-rl-closed.urdf": os.path.join(BASE, "urdf", "infantry-rl-closed.urdf"),
}


def rpy_to_mat(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def tf(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def read_stl(path):
    with open(path, "rb") as fh:
        head = fh.read(84)
        n = struct.unpack("<I", head[80:84])[0]
        data = np.frombuffer(fh.read(n * 50), dtype=np.uint8).reshape(n, 50)
        v = data[:, 12:48].copy().view("<f4").reshape(n, 3, 3).reshape(-1, 3)
    return v.astype(np.float64)


def load(path):
    root = ET.parse(path).getroot()
    by_parent, links = {}, {}
    for el in root.findall("joint"):
        o = el.find("origin")
        rec = dict(parent=el.find("parent").get("link"), child=el.find("child").get("link"),
                   xyz=np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]),
                   rpy=[float(v) for v in (o.get("rpy") or "0 0 0").split()])
        by_parent.setdefault(rec["parent"], []).append(rec)
    for l in root.findall("link"):
        m = l.find("visual/geometry/mesh")
        links[l.get("name")] = m.get("filename").split("/")[-1] if m is not None else None
    W = {}

    def rec(link, T):
        W[link] = T
        for j in by_parent.get(link, []):
            rec(j["child"], T @ tf(rpy_to_mat(*j["rpy"]), j["xyz"]))

    rec("base_link", np.eye(4))
    return links, W


for label, path in FILES.items():
    links, W = load(path)
    print(f"\n=== {label} ===")
    for leg in ("left", "right"):
        calf = links[f"{leg}_back_calf"]
        # the wheel centre expressed in the back calf's own link frame
        local = np.linalg.inv(W[f"{leg}_back_calf"]) @ W[f"{leg}_wheel"]
        centre = local[:3, 3]
        v = read_stl(os.path.join(MESH_DIR, calf))
        d = np.linalg.norm(v - centre, axis=1)
        bins = [1, 2, 3, 5, 8, 12, 20, 30]
        counts = [int((d < b).sum()) for b in bins]
        print(f"  {leg:5s} {calf:24s} wheel centre = {np.round(centre,4)}")
        print(f"        vertices within {bins} mm: {counts}   (min dist {d.min()*1000:.2f} mm)")
        verdict = ("centre is in a HOLE" if counts[3] <= 5 and counts[5] > 20
                   else "centre is in SOLID material" if counts[0] > 20
                   else "ambiguous")
        print(f"        -> {verdict}")
