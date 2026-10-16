"""Decide which URDF the meshes actually agree with.

For a correctly built mirrored pair, transforming the right link's mesh by its link frame
must land on the mirror image of the left link's mesh (also transformed by its own link
frame).  This tests the meshes, not the joint origins -- the joint tree can be a perfect
mirror while the meshes it carries are rotated or displaced.
"""

import math
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

BASE = r"D:\IsaacLab-main\parallel_leg\infantry-rl-main\infantry-rl-main\infantry-rl"
MESH_DIR = os.path.join(BASE, "meshes")
MIRROR = np.diag([1.0, -1.0, 1.0])

FILES = {
    "original infantry-rl.urdf": os.path.join(BASE, "urdf", "infantry-rl.urdf"),
    "closed infantry-rl-closed.urdf": os.path.join(BASE, "urdf", "infantry-rl-closed.urdf"),
    "mine infantry-rl-sim.urdf": os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "urdf", "infantry-rl-sim.urdf")),
}

PAIRS = [("left_front_ham", "right_front_ham"), ("left_front_calf", "right_front_calf"),
         ("left_wheel", "right_wheel"), ("left_back_ham", "right_back_ham"),
         ("left_back_calf", "right_back_calf")]


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


def read_stl(path, stride=97):
    """Vertex sample of a binary/ascii STL (strided: these files get big)."""
    with open(path, "rb") as fh:
        head = fh.read(84)
        if head[:5] == b"solid" and b"facet" in head:
            fh.seek(0)
            pts = [[float(v) for v in ln.split()[1:4]]
                   for ln in fh.read().decode("ascii", "ignore").splitlines()
                   if ln.strip().startswith("vertex")]
            v = np.array(pts)
        else:
            n = struct.unpack("<I", head[80:84])[0]
            data = np.frombuffer(fh.read(n * 50), dtype=np.uint8).reshape(n, 50)
            v = data[:, 12:48].copy().view("<f4").reshape(n, 3, 3).reshape(-1, 3)
    return v[::stride].astype(np.float64)


def load(path):
    root = ET.parse(path).getroot()
    by_parent, links, VO = {}, {}, {}
    global _vo
    _vo = VO
    for el in root.findall("joint"):
        o = el.find("origin")
        rec = dict(parent=el.find("parent").get("link"), child=el.find("child").get("link"),
                   xyz=np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]),
                   rpy=[float(v) for v in (o.get("rpy") or "0 0 0").split()])
        by_parent.setdefault(rec["parent"], []).append(rec)
    for l in root.findall("link"):
        m = l.find("visual/geometry/mesh")
        links[l.get("name")] = m.get("filename").split("/")[-1] if m is not None else None
        v = l.find("visual/origin")
        VO[l.get("name")] = np.array([float(x) for x in (v.get("xyz") or "0 0 0").split()]) \
            if v is not None else np.zeros(3)
    W = {}

    def rec(link, T):
        W[link] = T
        for j in by_parent.get(link, []):
            rec(j["child"], T @ tf(rpy_to_mat(*j["rpy"]), j["xyz"]))

    rec("base_link", np.eye(4))
    return links, W, VO


_cache = {}


def mesh(name):
    if name not in _cache:
        _cache[name] = read_stl(os.path.join(MESH_DIR, name))
    return _cache[name]


for label, path in FILES.items():
    links, W, VO = load(path)
    print(f"\n=== {label} ===")
    print(f"  {'link pair':34s} {'centroid dev':>13s} {'rms dev':>10s}   verdict")
    worst = 0.0
    for ln, rn in PAIRS:
        vl = mesh(links[ln])
        vr = mesh(links[rn])
        wl = (W[ln][:3, :3] @ vl.T).T + W[ln][:3, 3] + W[ln][:3, :3] @ VO[ln]
        wr = (W[rn][:3, :3] @ vr.T).T + W[rn][:3, 3] + W[rn][:3, :3] @ VO[rn]
        # compare the mirrored left cloud against the right cloud, point-to-point-ish
        # (same vertex ordering is not guaranteed, so compare centroids and extents)
        cl, cr = wl.mean(axis=0), wr.mean(axis=0)
        dev = np.linalg.norm(MIRROR @ cl - cr)
        # rms of the per-axis extent difference gives a scale/shape sanity number
        el = wl.max(axis=0) - wl.min(axis=0)
        er = wr.max(axis=0) - wr.min(axis=0)
        rms = float(np.sqrt(np.mean((el - er) ** 2)))
        worst = max(worst, dev)
        flag = "ok" if dev < 5e-3 and rms < 5e-3 else "<-- meshes disagree"
        print(f"  {ln[5:] + ' / ' + rn[6:]:34s} {dev*1000:10.2f} mm {rms*1000:8.2f} mm   {flag}")
    print(f"  worst centroid deviation: {worst*1000:.2f} mm")
