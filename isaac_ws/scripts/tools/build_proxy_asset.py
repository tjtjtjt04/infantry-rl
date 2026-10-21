"""Build a lightweight proxy USD for the infantry-rl robot.

WHY
---
The full-fidelity asset carries the original CAD meshes; ``base_link.STL`` alone is
57 MB and the converted USD is ~96 MB.  With that asset the scene cannot be built
for more than ONE environment: ``gym.make`` hangs indefinitely inside env
construction with ``num_envs >= 2``.  This was isolated with
``isaac_ws/scripts/diag_clone.py`` and is NOT caused by physics replication or
collision filtering -- both were toggled off independently and the hang persisted.

WHAT THIS SCRIPT DOES
---------------------
It regenerates the SAME articulation -- same 11 links, same 10 joints, same joint
names, axes, origins and limits, so the verified kinematics in
``isaac_ws_infantry/kinematics`` applies unchanged -- but replaces every visual and
collision mesh with a primitive sized from that mesh's bounding box:

* every link except the wheels becomes a box (the chassis, hams and calves are
  box-like, and a box collision is far more stable for PhysX than a 57 MB
  triangle soup);
* the wheels become cylinders about the wheel axis, with the radius measured from
  the mesh, which is what actually matters for rolling contact.

The geometry is re-derived from the URDF, and the original USD is left untouched.
Output: ``isaac_ws/assets/infantry-rl/infantry_rl_proxy.usd``.

Run:  python isaac_ws/scripts/build_proxy_asset.py
"""

from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent
REPO = WS.parent
URDF = WS / "assets" / "infantry-rl" / "urdf" / "infantry-rl-sim.urdf"
MESH_DIR = WS / "assets" / "infantry-rl" / "meshes"
OUT = WS / "assets" / "infantry-rl" / "infantry_rl_proxy.usd"
REPORT = WS / "config" / "proxy_geometry.json"


# --------------------------------------------------------------------------- #
# STL reader (binary and ASCII), only needs vertices
# --------------------------------------------------------------------------- #
def read_stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        return np.zeros((0, 3))
    # binary STL: 80-byte header, uint32 count, then 50 bytes per triangle
    n = int.from_bytes(data[80:84], "little")
    if 84 + n * 50 == len(data):
        rec = np.frombuffer(data, dtype=np.uint8, count=n * 50, offset=84).reshape(n, 50)
        tri = rec[:, 12:48].copy().view("<f4").reshape(n, 3, 3)
        return tri.reshape(-1, 3).astype(np.float64)
    # ASCII fallback
    verts = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("vertex"):
            parts = s.split()
            if len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3))


def bbox_of(name: str):
    p = MESH_DIR / f"{name}.STL"
    if not p.is_file():
        return None
    v = read_stl_vertices(p)
    if v.size == 0:
        return None
    return v.min(axis=0), v.max(axis=0), len(v)


def main() -> int:
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=False)
    app = launcher.app

    import isaaclab.sim as sim_utils
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    root = ET.parse(URDF).getroot()

    # ---- collect the URDF structure -------------------------------------
    links = {}
    for link in root.findall("link"):
        name = link.get("name")
        mass = 0.0
        inertial = link.find("inertial")
        if inertial is not None:
            mass = float(inertial.find("mass").get("value"))
        links[name] = {"mass": mass}

    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else "0 0 0".split())]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else "0 0 0".split())]
        a = j.find("axis")
        axis = [float(v) for v in (a.get("xyz", "1 0 0").split() if a is not None else "1 0 0".split())]
        lim = j.find("limit")
        joints.append(
            {
                "name": j.get("name"),
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
                "lower": float(lim.get("lower", "-inf")) if lim is not None else None,
                "upper": float(lim.get("upper", "inf")) if lim is not None else None,
                "effort": float(lim.get("effort", "0")) if lim is not None else 0.0,
                "velocity": float(lim.get("velocity", "0")) if lim is not None else 0.0,
            }
        )

    # ---- measure each link's box from its mesh --------------------------
    geom = {}
    for name in links:
        r = bbox_of(name)
        if r is None:
            geom[name] = None
            continue
        lo, hi, npts = r
        centre = (lo + hi) / 2.0
        size = np.maximum(hi - lo, 1e-3)
        geom[name] = {
            "centre": centre.tolist(),
            "size": size.tolist(),
            "points": int(npts),
        }

    # The wheel radius: the wheel's mesh centre is the axle, so half the box height
    # in the plane perpendicular to the axle is the radius.  The axle is +/-Y, so
    # the radius comes from the X/Z extent.
    wheel_radius = {}
    for side in ("left", "right"):
        g = geom.get(f"{side}_wheel")
        wheel_radius[side] = float(min(g["size"][0], g["size"][2]) / 2.0) if g else 0.07

    # ---- build the stage ------------------------------------------------
    if OUT.exists():
        OUT.unlink()
    stage = Usd.Stage.CreateNew(str(OUT))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    robot = UsdGeom.Xform.Define(stage, "/infantry_rl_proxy")
    stage.SetDefaultPrim(robot.GetPrim())

    # materials (one grey, so the render is not black)
    mat = UsdShade.Material.Define(stage, "/infantry_rl_proxy/Looks/grey")
    pbr = UsdShade.Shader.Define(stage, "/infantry_rl_proxy/Looks/grey/Shader")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.7, 0.72, 0.75))
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")

    def add_box(path: str, centre, size, density: float):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
        xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateDensityAttr(density)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
        return cube.GetPrim()

    def add_cylinder(path: str, centre, radius: float, height: float, density: float):
        cyl = UsdGeom.Cylinder.Define(stage, path)
        cyl.CreateRadiusAttr(float(radius))
        cyl.CreateHeightAttr(float(height))
        cyl.CreateAxisAttr(UsdGeom.Tokens.y)  # the wheel axle is body Y
        xf = UsdGeom.Xformable(cyl.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
        UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateDensityAttr(density)
        UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(mat)
        return cyl.GetPrim()

    for name, info in links.items():
        link_prim = UsdGeom.Xform.Define(stage, f"/infantry_rl_proxy/{name}")
        UsdPhysics.RigidBodyAPI.Apply(link_prim.GetPrim())
        UsdPhysics.MassAPI.Apply(link_prim.GetPrim()).CreateMassAttr(float(info["mass"]))
        g = geom.get(name)
        if g is None:
            # no mesh: fall back to a small box so the link still has collision
            add_box(f"/infantry_rl_proxy/{name}/collisions/box", (0, 0, 0), (0.02, 0.02, 0.02), 1000.0)
            continue
        # Boxes are centred on the mesh centroid, expressed in the link frame.
        if name.endswith("_wheel"):
            side = name.split("_")[0]
            r = wheel_radius[side]
            width = g["size"][1]
            add_cylinder(f"/infantry_rl_proxy/{name}/collisions/cyl", g["centre"], r, width, 1000.0)
        else:
            add_box(f"/infantry_rl_proxy/{name}/collisions/box", g["centre"], g["size"], 1000.0)

    # ---- joints ---------------------------------------------------------
    joints_scope = UsdGeom.Scope.Define(stage, "/infantry_rl_proxy/joints")
    joint_report = []
    for jd in joints:
        p = f"/infantry_rl_proxy/joints/{jd['name']}"
        if jd["type"] in ("revolute", "continuous"):
            cls = UsdPhysics.RevoluteJoint
        else:
            cls = UsdPhysics.FixedJoint
        jp = cls.Define(stage, p)
        jp.CreateBody0Rel().SetTargets([f"/infantry_rl_proxy/{jd['parent']}"])
        jp.CreateBody1Rel().SetTargets([f"/infantry_rl_proxy/{jd['child']}"])
        jp.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in jd["xyz"]]))
        jp.CreateLocalRot0Attr().Set(Gf.Quatf(*_rpy_to_quat(*jd["rpy"])))
        jp.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        jp.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        if jd["type"] in ("revolute", "continuous"):
            jp.CreateAxisAttr(_axis_token(jd["axis"]))
            if jd["lower"] is not None and math.isfinite(jd["lower"]):
                jp.CreateLowerLimitAttr(math.degrees(jd["lower"]))
            if jd["upper"] is not None and math.isfinite(jd["upper"]):
                jp.CreateUpperLimitAttr(math.degrees(jd["upper"]))
        joint_report.append({"name": jd["name"], "type": jd["type"], "axis": jd["axis"]})

    # articulation root on the base link
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/infantry_rl_proxy/base_link"))

    stage.GetRootLayer().Save()

    report = {
        "out": str(OUT),
        "links": len(links),
        "joints": len(joints),
        "wheel_radius": wheel_radius,
        "geometry": geom,
        "total_source_points": sum(g["points"] for g in geom.values() if g),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)")
    print(f"links={len(links)} joints={len(joints)}  source points={report['total_source_points']:,}")
    for side, r in wheel_radius.items():
        print(f"  {side} wheel radius = {r:.4f} m")
    print(f"wrote {REPORT}")

    app.close()
    return 0


def _rpy_to_quat(r: float, p: float, y: float):
    """URDF fixed-axis rpy (X then Y then Z) -> (w, x, y, z)."""
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yy = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, yy, z)


def _axis_token(axis):
    from pxr import UsdGeom

    a = np.asarray(axis, dtype=float)
    i = int(np.argmax(np.abs(a)))
    return (UsdGeom.Tokens.x, UsdGeom.Tokens.y, UsdGeom.Tokens.z)[i]


if __name__ == "__main__":
    sys.exit(main())
