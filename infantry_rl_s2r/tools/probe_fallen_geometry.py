#!/usr/bin/env python
"""Measure the fallen-state ground contact geometry of chassis-rl (needs Kit).

Why this exists
---------------
The fall-recovery task must place the robot at a pitch angle where it really
*rests* on the ground, and the episode must start from a state the robot can
actually be in.  Two facts cannot be guessed:

1. **the contact height** at a given root pitch -- the root height at which the
   lowest collision geometry first touches ``z = 0``;
2. **which body touches first** -- if it is a wheel, the wheels still have
   traction and a wheel-torque swing-up is possible; if it is the chassis
   corner, the wheels are in the air and the policy has to use wheel reaction
   torque / wait for the wheels to land.

Method
------
``--mode lower`` lowers the robot 2 mm at a time with zero velocity until a
contact force appears, using a contact sensor on **every** link, and reports the
height and the touching link.  This is exact for the real collision meshes, which
a bounding box is not (the leg links' local bounds are conservative).

``--mode settle`` drops the robot at that height and runs the physics with zero
action, reporting where it actually comes to rest -- i.e. what "fallen" really
looks like for this robot.

Examples::

    C:/Users/32560/.conda/envs/isaaclab/python.exe tools/probe_fallen_geometry.py --headless
    C:/Users/32560/.conda/envs/isaaclab/python.exe tools/probe_fallen_geometry.py --headless --mode settle
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# the probe must not depend on the training-time randomization
os.environ.setdefault("INFANTRY_RL_DR", "0")

from isaaclab.app import AppLauncher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe the fallen-state contact geometry.")
    parser.add_argument("--pitches", type=float, nargs="+",
                        default=[0.0, 5.0, 10.0, 15.0, 17.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 60.0],
                        help="root pitch angles (deg) to probe")
    parser.add_argument("--mode", choices=("lower", "settle", "both"), default="both")
    parser.add_argument("--settle-steps", type=int, default=120,
                        help="control steps to let each pose settle (100 Hz)")
    parser.add_argument("--json", type=str, default=None, help="write the table to this JSON file")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows (imports require Kit)."""

import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from env_squat import (  # noqa: E402
    SQUAT_BASE_HEIGHT,
    SQUAT_HIP_OFFSETS,
    FiveBarSquatEnv,
    SquatEnvCfg,
)
from robot_model import WHEEL_RADIUS  # noqa: E402


def build_env() -> FiveBarSquatEnv:
    cfg = SquatEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 0
    cfg.episode_length_s = 1.0e6
    # one sensor on every link, so the touch-down test does not have to guess
    cfg.scene.contact_all = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=0,
        track_air_time=False,
        debug_vis=False,
    )
    return FiveBarSquatEnv(cfg=cfg)


def link_bounds(stage, link_names: set[str]) -> dict[str, tuple[float, float, float, float, float, float]]:
    """Local-frame AABB per link (``xmin, xmax, ymin, ymax, zmin, zmax``).

    The USD world bound of a link is mapped back through the link's stage
    transform, so only the *relative* geometry is used: the stage xforms are the
    spawn pose (fabric is on, so PhysX does not refresh them).
    """
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    xf = UsdGeom.XformCache(Usd.TimeCode.Default())
    out: dict[str, tuple[float, float, float, float, float, float]] = {}
    for prim in stage.Traverse():
        name = prim.GetName()
        if name not in link_names or not prim.IsA(UsdGeom.Xformable):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        if box.IsEmpty():
            continue
        inv = xf.GetLocalToWorldTransform(prim).GetInverse()
        lo = [float("inf")] * 3
        hi = [float("-inf")] * 3
        lo3, hi3 = box.GetMin(), box.GetMax()
        for i in range(8):
            p = Gf.Vec3d(
                hi3[0] if i & 1 else lo3[0],
                hi3[1] if i & 2 else lo3[1],
                hi3[2] if i & 4 else lo3[2],
            )
            q = inv.Transform(p)
            for k in range(3):
                lo[k] = min(lo[k], q[k])
                hi[k] = max(hi[k], q[k])
        out[name] = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    return out


class Probe:
    def __init__(self, env: FiveBarSquatEnv):
        self.env = env
        self.robot = env.scene["robot"]
        self.term = env.action_manager.get_term(env.cfg.joint_action_name)
        self.sensor = env.scene["contact_all"]
        self.zeros_act = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
        self.joint_ids = self.term.hip_ids + self.term.knee_ids
        self.hip_targets = self.term.nominal_hips + torch.tensor(
            [SQUAT_HIP_OFFSETS], device=env.device
        )
        self.joint_pos = torch.cat((self.hip_targets, self.term.reset_knees.unsqueeze(0)), dim=-1)

    def place(self, pitch_deg: float, z: float, roll_deg: float = 0.0) -> None:
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)
        quat = math_utils.quat_from_euler_xyz(
            torch.tensor([roll], device=self.env.device),
            torch.tensor([pitch], device=self.env.device),
            torch.tensor([0.0], device=self.env.device),
        )
        self.robot.write_root_pose_to_sim(
            torch.cat((torch.tensor([[0.0, 0.0, z]], device=self.env.device), quat), dim=-1)
        )
        self.robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.env.device))
        self.robot.write_joint_state_to_sim(
            self.joint_pos, torch.zeros_like(self.joint_pos), joint_ids=self.joint_ids
        )

    def contact(self) -> tuple[float, list[str]]:
        """``(max per-body contact force, names with force > 1 N)``."""
        forces = self.sensor.data.net_forces_w[0].norm(dim=-1)  # (B,)
        touching = [self.sensor.body_names[i] for i in torch.nonzero(forces > 1.0).flatten().tolist()]
        return float(forces.max()), touching

    def lower_until_touch(self, pitch_deg: float, z_hi: float = 0.42, step: float = 0.002):
        """Return ``(contact_height, touching_bodies)`` by lowering with zero velocity."""
        z = z_hi
        while z > 0.02:
            self.place(pitch_deg, z)
            self.env.sim.step(render=False)
            self.env.scene.update(dt=self.env.physics_dt)
            force, touching = self.contact()
            if force > 1.0:
                return z, touching
            z -= step
        return float("nan"), []

    def settle(self, pitch_deg: float, z: float, steps: int):
        self.place(pitch_deg, z)
        self.env.scene.update(dt=self.env.physics_dt)
        for _ in range(int(steps)):
            self.env.step(self.zeros_act)
        roll, pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        pos = self.robot.data.body_pos_w[0]
        names = list(self.robot.body_names)
        wheels = [names.index(n) for n in ("left_wheel", "right_wheel") if n in names]
        force, touching = self.contact()
        return {
            "pitch_deg": math.degrees(float(pitch[0])),
            "roll_deg": math.degrees(float(roll[0])),
            "base_z": float(pos[names.index("base_link"), 2]),
            "wheel_z": [float(pos[i, 2]) for i in wheels],
            "contact_force": force,
            "touching": touching,
        }


def main() -> int:
    print("[probe] building env", flush=True)
    env = build_env()
    probe = Probe(env)
    stage = getattr(env.sim, "stage", None)
    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()

    # never reset during the probe
    env.termination_manager.compute = lambda: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.reset()

    bounds = link_bounds(stage, set(probe.robot.body_names))
    print("\n[bbox] local AABB of every link (m; leg links are conservative):", flush=True)
    for name in probe.robot.body_names:
        b = bounds.get(name)
        if b is None:
            print(f"  {name:<18} <none>")
            continue
        print(
            f"  {name:<18} x[{b[0]:+.4f},{b[1]:+.4f}] y[{b[2]:+.4f},{b[3]:+.4f}] "
            f"z[{b[4]:+.4f},{b[5]:+.4f}]"
        )
    print(
        f"\n[state] squat base height {SQUAT_BASE_HEIGHT:.5f} m, wheel radius {WHEEL_RADIUS:.4f} m, "
        f"reset hip offsets {tuple(round(v, 4) for v in SQUAT_HIP_OFFSETS)}",
        flush=True,
    )

    rows = []
    if args_cli.mode in ("lower", "both"):
        print("\n[lower] contact height by lowering 2 mm/step with zero velocity:", flush=True)
        print(f"{'pitch':>6} {'contact h':>10}  touching", flush=True)
        for pitch_deg in args_cli.pitches:
            z, touching = probe.lower_until_touch(float(pitch_deg))
            rows.append({"pitch_deg": float(pitch_deg), "contact_height": z, "touching": touching})
            print(f"{pitch_deg:>6.1f} {z:>10.4f}  {','.join(touching)}", flush=True)

    if args_cli.mode in ("settle", "both"):
        print("\n[settle] zero action for %.2f s from the contact height:" % (args_cli.settle_steps / 100.0), flush=True)
        print(
            f"{'pitch0':>6} {'contact h':>10} | {'pitch_end':>9} {'base_z':>8} "
            f"{'wheel z':>15}  touching",
            flush=True,
        )
        for row in rows or [{"pitch_deg": float(p), "contact_height": None} for p in args_cli.pitches]:
            pitch_deg = row["pitch_deg"]
            z = row["contact_height"]
            if z is None or not math.isfinite(z):
                z, _ = probe.lower_until_touch(float(pitch_deg))
            result = probe.settle(float(pitch_deg), z + 0.001, args_cli.settle_steps)
            row["settled"] = result
            wheel_z = "/".join(f"{v:.3f}" for v in result["wheel_z"])
            print(
                f"{pitch_deg:>6.1f} {z:>10.4f} | {result['pitch_deg']:>9.2f} {result['base_z']:>8.4f} "
                f"{wheel_z:>15}  {','.join(result['touching'])}",
                flush=True,
            )

    if args_cli.json:
        Path(args_cli.json).write_text(json.dumps({"bounds": bounds, "rows": rows}, indent=2), encoding="utf-8")
        print(f"\n[wrote] {args_cli.json}", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
