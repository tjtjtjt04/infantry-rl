"""Kinematic core shared by the infantry-rl URDF tools.

Parses a URDF into a spanning tree (URDF files produced here contain extra
loop-closing joints), computes forward kinematics, and exposes the helpers used
by build/verify scripts.
"""
import math
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "infantry-rl", "meshes")
MESH_DIR = os.path.abspath(MESH_DIR)


def rpy_to_R(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def axis_rot(ax, ang):
    a = np.asarray(ax, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


class Robot:
    """A URDF robot with its kinematic spanning tree."""

    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.path = path
        self.name = root.get("name")
        self.joints = {}
        self.meshes = {}
        for l in root.findall("link"):
            m = l.find("visual/geometry/mesh")
            self.meshes[l.get("name")] = (
                os.path.basename(m.get("filename")) if m is not None else None)
        for j in root.findall("joint"):
            o, ax, lim = j.find("origin"), j.find("axis"), j.find("limit")
            self.joints[j.get("name")] = dict(
                name=j.get("name"), type=j.get("type"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                xyz=np.array([float(x) for x in (o.get("xyz", "0 0 0")).split()]),
                rpy=[float(x) for x in (o.get("rpy", "0 0 0")).split()],
                axis=(np.array([float(x) for x in ax.get("xyz", "1 0 0").split()])
                      if ax is not None else None),
                lower=float(lim.get("lower", 0)) if lim is not None else None,
                upper=float(lim.get("upper", 0)) if lim is not None else None)

        # spanning tree from base_link: the first joint to reach a link claims it
        kids = {}
        for n, j in self.joints.items():
            kids.setdefault(j["parent"], []).append(n)
        queue, visited = ["base_link"], {"base_link"}
        self.child_of, self.order = {}, []
        while queue:
            link = queue.pop(0)
            for n in kids.get(link, []):
                ch = self.joints[n]["child"]
                if ch in visited:
                    continue
                visited.add(ch)
                self.child_of[ch] = n
                self.order.append(n)
                queue.append(ch)
        self.links_reached = visited
        self.extra = [n for n in self.joints if n not in self.order]

    def pose(self, link, q):
        """World (R, p) of a link, following the spanning tree."""
        chain, cur = [], link
        while cur != "base_link":
            jn = self.child_of[cur]
            chain.append(jn)
            cur = self.joints[jn]["parent"]
        R, p = np.eye(3), np.zeros(3)
        for jn in reversed(chain):
            j = self.joints[jn]
            p = p + R @ j["xyz"]
            R = R @ rpy_to_R(*j["rpy"])
            if j["type"] in ("revolute", "continuous"):
                R = R @ axis_rot(j["axis"], q.get(jn, 0.0))
        return R, p

    def mesh(self, link):
        """Triangle vertices of a link's visual mesh, in the link frame."""
        path = os.path.join(MESH_DIR, self.meshes[link])
        with open(path, "rb") as f:
            head = f.read(84)
            n = struct.unpack("<I", head[80:84])[0]
            body = f.read(n * 50)
        arr = np.frombuffer(body, dtype=np.uint8).reshape(n, 50)
        return arr[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)


def joint_index(rob, names):
    return {n: i for i, n in enumerate(names)}


def leg_task(rob, side, q):
    """(leg length L0, leg axis angle phi) measured from O = midpoint of AE."""
    jw = rob.joints[f"{side}_wheel"]
    Rp, pp = rob.pose(jw["parent"], q)
    axle = pp + Rp @ jw["xyz"]
    A = rob.joints[f"{side}_front_hip"]["xyz"]
    E = rob.joints[f"{side}_back_hip"]["xyz"]
    O = 0.5 * (A + E)
    d = axle - O
    return math.hypot(d[0], d[2]), math.atan2(d[0], d[2])


def loop_angle(rob, side, q):
    """Rotation of the loop joint, recovered from the two link frames."""
    Rw, _ = rob.pose(f"{side}_wheel", q)
    Rb, _ = rob.pose(f"{side}_back_calf", q)
    R = Rw.T @ Rb
    return math.atan2(R[0, 2] - R[2, 0], R[0, 0] + R[2, 2])
