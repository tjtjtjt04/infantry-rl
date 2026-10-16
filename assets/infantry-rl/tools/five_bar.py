"""Closed-loop kinematics for one five-bar leg of infantry-rl.

The URDF we import into PhysX is a *tree*: base -> hip -> ham -> knee -> calf -> wheel,
plus a second branch base -> back_hip -> back_ham -> back_knee -> back_calf.  The real
mechanism is closed: the back calf's axle hole sits on the wheel axle.  PhysX cannot
represent that, so the environment enforces it every control step by driving the two knee
joints to the angles that close the loop.

Joints per leg, in the tree:
    front hip    (+Y)  <- actuator q1   (policy)
    front knee   (-Y)  <- slaved  qkf
    back  hip    (+Y)  <- actuator q4   (policy)
    back  knee   (-Y)  <- slaved  qkb
    wheel        (-Y)  <- actuator (velocity)

Constraint:  tip(back_calf) == wheel_centre          (2 equations, y is automatic)
Unknowns:    qkf, qkb                                 (given q1 and q4)

Usage:
    python five_bar.py                       # summary + home-pose candidates
"""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.normpath(os.path.join(HERE, "..", "urdf", "infantry-rl-sim.urdf"))

LEGS = {
    "left": dict(hip="left_front_hip", knee="left_front_knee", wheel="left_wheel",
                 bhip="left_back_hip", bknee="left_back_knee"),
    "right": dict(hip="right_front_hip", knee="right_front_knee", wheel="right_wheel",
                  bhip="right_back_hip", bknee="right_back_knee"),
}


def rpy_to_mat(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle(axis, q):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)


def tf(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


class Model:
    def __init__(self, path=URDF):
        root = ET.parse(path).getroot()
        self.joints = {}
        self.by_parent = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
            ax = j.find("axis")
            lim = j.find("limit")
            rec = dict(
                name=j.get("name"), type=j.get("type"),
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                origin=tf(rpy_to_mat(*rpy), xyz),
                axis=np.array([float(v) for v in (ax.get("xyz") or "1 0 0").split()])
                if ax is not None else np.zeros(3),
                lo=float(lim.get("lower")) if lim is not None and lim.get("lower") is not None else None,
                hi=float(lim.get("upper")) if lim is not None and lim.get("upper") is not None else None,
            )
            self.joints[rec["name"]] = rec
            self.by_parent.setdefault(rec["parent"], []).append(rec)

        self._fk_cache = {}
        # the back calf's axle hole, in the calf's own link frame (measured at q = 0,
        # where the tree is exactly the assembled CAD pose)
        for leg, n in LEGS.items():
            W = self.fk({})
            R = W[n["bknee"].replace("knee", "calf")][:3, :3]
            self._fk_cache[leg] = R.T @ (W[n["wheel"]][:3, 3]
                                         - W[n["bknee"].replace("knee", "calf")][:3, 3])

    def tip_offset(self, leg):
        """Axle-hole position in the back calf link frame."""
        return self._fk_cache[leg]

    def fk(self, q):
        """World transform of every link frame; q maps joint name -> angle."""
        out = {}

        def rec(link, T):
            out[link] = T
            for j in self.by_parent.get(link, []):
                Rj = axis_angle(j["axis"], q.get(j["name"], 0.0)) if j["type"] != "fixed" else np.eye(3)
                rec(j["child"], T @ j["origin"] @ tf(Rj, np.zeros(3)))

        rec("base_link", np.eye(4))
        return out

    # -- leg level ---------------------------------------------------------------
    def solve_knees(self, leg, q1, q4, seed=(0.0, 0.0), iters=60, tol=1e-12):
        """Given the two hip angles, return (qkf, qkb) that closes the loop."""
        n = LEGS[leg]
        tip_local = self.tip_offset(leg)

        def residual(x):
            q = {n["hip"]: q1, n["bhip"]: q4, n["knee"]: x[0], n["bknee"]: x[1]}
            W = self.fk(q)
            p_wheel = W[n["wheel"]][:3, 3]
            p_tip = (W[n["bknee"].replace("knee", "calf")] @ np.append(tip_local, 1.0))[:3]
            return (p_tip - p_wheel)[[0, 2]]          # y is satisfied identically

        x = np.array(seed, float)
        for _ in range(iters):
            r = residual(x)
            if np.max(np.abs(r)) < tol:
                break
            J = np.zeros((2, 2))
            for k in range(2):
                dx = np.zeros(2)
                dx[k] = 1e-7
                J[:, k] = (residual(x + dx) - r) / 1e-7
            x = x - np.linalg.solve(J, r)
        return float(x[0]), float(x[1]), float(np.max(np.abs(residual(x))))

    def leg_state(self, leg, q):
        """Leg length and leg-axis angle, measured from the hip midpoint to the wheel."""
        n = LEGS[leg]
        W = self.fk(q)
        hip_mid = 0.5 * (W[n["hip"].replace("hip", "ham")][:3, 3]
                         + W[n["bhip"].replace("hip", "ham")][:3, 3])
        v = W[n["wheel"]][:3, 3] - hip_mid
        return float(math.hypot(v[0], v[2])), float(math.atan2(-v[2], v[0]))

    def pose_for(self, leg, length, angle=math.pi / 2):
        """Hip angles that put the wheel at `length` from the hip midpoint, `angle` down."""
        n = LEGS[leg]
        W0 = self.fk({})
        hip_mid = 0.5 * (W0[n["hip"].replace("hip", "ham")][:3, 3]
                         + W0[n["bhip"].replace("hip", "ham")][:3, 3])
        target = hip_mid + np.array([length * math.cos(angle), 0.0, -length * math.sin(angle)])

        def residual(x):
            q = {n["hip"]: x[0], n["bhip"]: x[1]}
            kf, kb, _ = self.solve_knees(leg, x[0], x[1], seed=(1.0, -1.0))
            q[n["knee"]] = kf
            q[n["bknee"]] = kb
            p = self.fk(q)[n["wheel"]][:3, 3] - target
            return p[[0, 2]]

        x = np.array([0.4, -0.4])
        for _ in range(80):
            r = residual(x)
            if np.max(np.abs(r)) < 1e-10:
                break
            J = np.zeros((2, 2))
            for k in range(2):
                dx = np.zeros(2)
                dx[k] = 1e-6
                J[:, k] = (residual(x + dx) - r) / 1e-6
            x = x - np.linalg.solve(J, r)
        kf, kb, _ = self.solve_knees(leg, x[0], x[1], seed=(1.0, -1.0))
        return dict(hip=float(x[0]), bknee_hip=float(x[1]), knee=kf, bknee=kb)


if __name__ == "__main__":
    m = Model()
    print("=== assembled (CAD) pose, q = 0 ===")
    for leg in LEGS:
        L, a = m.leg_state(leg, {})
        kf, kb, res = m.solve_knees(leg, 0.0, 0.0)
        print(f"  {leg:5s} leg_len={L:.5f} m  leg_ang={math.degrees(a):7.2f} deg   "
              f"knee-solve -> qkf={kf:+.6f} qkb={kb:+.6f} (residual {res:.1e})")

    print("\n=== candidate home poses (leg straight down) ===")
    print(f"  {'leg_len':>9s}  {'q_front_hip':>12s} {'q_back_hip':>11s} "
          f"{'q_front_knee':>13s} {'q_back_knee':>12s}")
    for length in (0.20, 0.22, 0.25, 0.28, 0.30):
        p = m.pose_for("left", length)
        print(f"  {length:9.3f}  {p['hip']:+12.5f} {p['bknee_hip']:+11.5f} "
              f"{p['knee']:+13.5f} {p['bknee']:+12.5f}")

    print("\n=== reachable leg length (both hips inside their limits) ===")
    jf, jb = m.joints["left_front_hip"], m.joints["left_back_hip"]
    q1s = np.linspace(jf["lo"], jf["hi"], 121)
    q4s = np.linspace(jb["lo"], jb["hi"], 121)
    lens, feasible = [], 0
    for q1 in q1s:
        for q4 in q4s:
            kf, kb, res = m.solve_knees("left", q1, q4, seed=(-0.5, -0.5))
            jk, jbk = m.joints["left_front_knee"], m.joints["left_back_knee"]
            if res > 1e-9 or not (jk["lo"] <= kf <= jk["hi"]) or not (jbk["lo"] <= kb <= jbk["hi"]):
                continue
            feasible += 1
            lens.append(m.leg_state("left", {"left_front_hip": q1, "left_back_hip": q4,
                                             "left_front_knee": kf, "left_back_knee": kb})[0])
    print(f"  {feasible} / {q1s.size * q4s.size} hip combinations close with both knees "
          f"inside their limits")
    if lens:
        print(f"  leg length: {min(lens):.4f} .. {max(lens):.4f} m")
