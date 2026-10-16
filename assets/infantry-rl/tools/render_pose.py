"""Plot the assembled pose (q = 0) of a URDF so the mesh placement can be eyeballed.

    python render_pose.py <urdf> <out.png> "<title>"
"""

import math
import os
import struct
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = r"D:\IsaacLab-main\parallel_leg\infantry-rl-main\infantry-rl-main\infantry-rl"
MESH_DIR = os.path.join(BASE, "meshes")

palette = {"ham": "#1f77b4", "calf": "#ff7f0e", "wheel": "#2ca02c", "base": "#cccccc"}


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


def read_stl(path, stride):
    with open(path, "rb") as fh:
        head = fh.read(84)
        n = struct.unpack("<I", head[80:84])[0]
        data = np.frombuffer(fh.read(n * 50), dtype=np.uint8).reshape(n, 50)
        v = data[:, 12:48].copy().view("<f4").reshape(n, 3, 3).reshape(-1, 3)
    return v[::stride].astype(np.float64)


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
    W, J = {}, {}

    def rec(link, T):
        W[link] = T
        for j in by_parent.get(link, []):
            J[j["child"]] = T @ tf(rpy_to_mat(*j["rpy"]), j["xyz"])
            rec(j["child"], J[j["child"]])

    rec("base_link", np.eye(4))
    return links, W, J


urdf, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
links, W, J = load(urdf)

fig, axes = plt.subplots(2, 1, figsize=(13, 11))
for ax, (ix, iy, xlabel) in zip(axes, [(0, 2, "x [m]"), (1, 2, "y [m]")]):
    for name, meshfile in links.items():
        if meshfile is None:
            continue
        stride = 400 if "base_link" in meshfile else 12
        v = read_stl(os.path.join(MESH_DIR, meshfile), stride)
        w = (W[name][:3, :3] @ v.T).T + W[name][:3, 3]
        key = ("base" if "base_link" in name else
               "wheel" if "wheel" in name else "calf" if "calf" in name else "ham")
        ax.plot(w[:, ix], w[:, iy], ".", ms=0.25, color=palette[key], alpha=0.25)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("z [m]")
    ax.grid(alpha=0.25)

# joint origins
for name, T in J.items():
    if name == "base_link":
        continue
    ax_ = axes[0]
    ax_.plot(T[0, 3], T[2, 3], "rx", ms=5)
    ax_.annotate(name, (T[0, 3], T[2, 3]), fontsize=6, color="r")
axes[1].set_title("front view (looking along -x)", fontsize=9)
axes[0].set_title(title, fontsize=11)

plt.tight_layout()
plt.savefig(out, dpi=110)
print("wrote", out)
