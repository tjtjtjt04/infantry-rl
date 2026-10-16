"""Compare the right leg in the original SW export against the 'corrected' closed URDF.

The original renders correctly in the GUI. The closed URDF forced an exact mirror, which
may have moved the right-leg joint origins (and therefore every mesh below them).
"""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = r"D:\IsaacLab-main\parallel_leg\infantry-rl-main\infantry-rl-main\infantry-rl\urdf\infantry-rl.urdf"
CLOSED = r"D:\IsaacLab-main\parallel_leg\infantry-rl-main\infantry-rl-main\infantry-rl\urdf\infantry-rl-closed.urdf"
MIRROR = np.diag([1.0, -1.0, 1.0])


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


def load(path):
    root = ET.parse(path).getroot()
    J, by_parent = {}, {}
    for el in root.findall("joint"):
        o = el.find("origin")
        rec = dict(
            parent=el.find("parent").get("link"), child=el.find("child").get("link"),
            xyz=np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]),
            rpy=[float(v) for v in (o.get("rpy") or "0 0 0").split()])
        J[el.get("name")] = rec
        by_parent.setdefault(rec["parent"], []).append(rec)
    W = {}

    def rec(link, T):
        W[link] = T
        for j in by_parent.get(link, []):
            rec(j["child"], T @ tf(rpy_to_mat(*j["rpy"]), j["xyz"]))

    rec("base_link", np.eye(4))
    return J, W


def joint_origin(J, W, name):
    j = J[name]
    return (W[j["parent"]] @ tf(rpy_to_mat(*j["rpy"]), j["xyz"]))[:3, 3]


Jo, Wo = load(ORIG)
Jc, Wc = load(CLOSED)

print("=== right-leg joint origin, in the .urdf text ===")
for n in ["right_front_hip", "right_front_knee", "right_wheel",
          "right_back_hip", "right_back_knee"]:
    a, b = Jo[n], Jc[n]
    same = np.allclose(a["xyz"], b["xyz"]) and np.allclose(a["rpy"], b["rpy"])
    print(f"  {n:20s} orig xyz={np.round(a['xyz'],6)} rpy={np.round(a['rpy'],5)}")
    print(f"  {'':20s} clos xyz={np.round(b['xyz'],6)} rpy={np.round(b['rpy'],5)}"
          f"   {'same' if same else '<-- CHANGED'}")

print("\n=== right-leg joint origin, in world at q = 0 ===")
for n in ["right_front_hip", "right_front_knee", "right_wheel",
          "right_back_hip", "right_back_knee"]:
    po = joint_origin(Jo, Wo, n)
    pc = joint_origin(Jc, Wc, n)
    print(f"  {n:20s} orig ({po[0]:+.5f},{po[1]:+.5f},{po[2]:+.5f})  "
          f"closed ({pc[0]:+.5f},{pc[1]:+.5f},{pc[2]:+.5f})  moved {np.linalg.norm(po-pc)*1000:6.2f} mm")

print("\n=== is the right leg the mirror of the left, in each file? ===")
for label, (J, W) in [("original", (Jo, Wo)), ("closed", (Jc, Wc))]:
    print(f"  {label}:")
    for ln, rn in [("left_front_hip", "right_front_hip"), ("left_front_knee", "right_front_knee"),
                   ("left_wheel", "right_wheel"), ("left_back_hip", "right_back_hip"),
                   ("left_back_knee", "right_back_knee")]:
        pl = joint_origin(J, W, ln)
        pr = joint_origin(J, W, rn)
        print(f"    {ln[5:]:16s} |M*p_L - p_R| = {np.linalg.norm(MIRROR @ pl - pr)*1000:8.3f} mm")
