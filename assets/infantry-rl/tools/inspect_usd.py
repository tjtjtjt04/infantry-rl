"""Dump the joints of a converted USD so we can check what PhysX actually got.

Run:  python inspect_usd.py [path-to.usd]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("usd", nargs="?", default=(
    r"D:\IsaacLab-main\parallel_leg\assets\infantry-rl\infantry_rl.usd"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open(args.usd)
print(f"\nstage: {stage.GetDefaultPrim().GetPath()}\n")

n = 0
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.Joint):
        continue
    n += 1
    rj = UsdPhysics.RevoluteJoint(prim)
    axis, lim, drive = "", "", ""
    if rj:
        a = rj.GetAxisAttr().Get()
        axis = str(a) if a else ""
        lim = f"[{rj.GetLowerLimitAttr().Get()}, {rj.GetUpperLimitAttr().Get()}]"
        d = UsdPhysics.DriveAPI.Get(prim, "angular")
        if d:
            drive = (f"stiff={d.GetStiffnessAttr().Get()} damp={d.GetDampingAttr().Get()} "
                     f"maxF={d.GetMaxForceAttr().Get()}")
    print(f"  {str(prim.GetPath()):58s} {prim.GetTypeName():14s} axis={axis:10s} {lim:24s} {drive}")

print(f"\ntotal physics joints: {n}")

arts = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
print(f"articulation roots: {arts}")

# fixed joints / link count
links = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
print(f"rigid bodies: {len(links)}")

sys.stdout.flush()
os._exit(0)  # app.close() can drop buffered stdout
