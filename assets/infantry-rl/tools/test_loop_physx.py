"""Does PhysX actually honour a closed loop imported from a URDF?

Loads the asset as an IsaacLab Articulation, then runs the mechanism with no gravity and no
drives but an initial joint velocity.  If the loop is a real constraint, the back calf's axle
hole stays on the wheel centre for the whole run.  If PhysX dropped or ignored the loop joint,
the calf swings free and the hole walks away.

    python test_loop_physx.py <usd> [--mid-range]
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("usd")
parser.add_argument("--mid-range", action="store_true",
                    help="seed joints at limit midpoints instead of the assembled pose")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_rotate_inverse  # noqa: E402

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0",
                                      gravity=(0.0, 0.0, 0.0)))
sim_utils.create_prim("/World", "Xform")

stage = Usd.Stage.Open(args.usd)
init = {}
for prim in stage.Traverse():
    j = UsdPhysics.RevoluteJoint(prim)
    if not j:
        continue
    name = prim.GetName()
    lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
    init[name] = (0.0 if lo is None else
                  0.5 * (float(lo) + float(hi)) * torch.pi / 180.0) if args.mid_range else 0.0

cfg = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=args.usd),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0), joint_pos=init),
    actuators={},
)
robot = Articulation(cfg)
sim.reset()

print(f"\nnum_joints = {robot.num_joints}   num_bodies = {robot.num_bodies}")
print("joints:", robot.joint_names)

LEGS = {}
for leg in ("left", "right"):
    ci = robot.body_names.index(f"{leg}_back_calf")
    wi = robot.body_names.index(f"{leg}_wheel")
    # axle hole expressed in the calf's own link frame (loop is closed at t=0)
    tip = quat_rotate_inverse(robot.data.body_quat_w[0, ci].unsqueeze(0),
                              (robot.data.body_pos_w[0, wi] - robot.data.body_pos_w[0, ci]).unsqueeze(0)).squeeze(0)
    LEGS[leg] = (ci, wi, tip)


def gap(leg):
    ci, wi, tip = LEGS[leg]
    pc, pw = robot.data.body_pos_w[0, ci], robot.data.body_pos_w[0, wi]
    q = robot.data.body_quat_w[0, ci]
    return torch.norm(pc + quat_apply(q.unsqueeze(0), tip.unsqueeze(0)).squeeze(0) - pw).item()


for leg in LEGS:
    print(f"{leg:6s} |axle hole - wheel centre| at t=0 : {gap(leg)*1000:8.4f} mm")

# no gravity, no drives: only a real constraint can hold the loop shut while it moves
vel = torch.zeros_like(robot.data.joint_vel)
vel[:, [robot.joint_names.index(f"{l}_front_hip") for l in LEGS]] = 1.0
robot.write_joint_state_to_sim(robot.data.joint_pos, vel)

print(f"\n{'t [s]':>7s} {'gap L [mm]':>12s} {'gap R [mm]':>12s} "
      f"{'q_front_hip L':>15s} {'q_back_knee L':>15s}")
for step in range(241):
    sim.step()
    robot.update(sim.get_physics_dt())
    if step % 40 == 0:
        q = robot.data.joint_pos[0]
        print(f"{step/120:7.2f} {gap('left')*1000:12.4f} {gap('right')*1000:12.4f} "
              f"{q[robot.joint_names.index('left_front_hip')].item():15.4f} "
              f"{q[robot.joint_names.index('left_back_knee')].item():15.4f}")

sys.stdout.flush()
os._exit(0)
