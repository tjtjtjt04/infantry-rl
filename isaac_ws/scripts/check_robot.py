"""Smoke test: does the robot spawn, stand, and keep the five-bar loop closed?

What it does
------------
Builds ``Infantry-RL-Balance-v0``, resets it, and steps 200 times with small random
actions.  Every step it measures, *independently of the solver*:

1. the real loop error, from the simulated body poses --
   ``|p(left_wheel)  - (p(left_back_calf)  + R(left_back_calf)  @ anchor_left)|``
   and the same for the right leg.  ``anchor`` is the ``<loop_joint>`` attachment
   point of the URDF, so this number is ~0 exactly when the closed chain is
   re-imposed by the knee targets;
2. the same residual from the measured joint angles through the verified planar
   model (cross-check between the two independent computations);
3. the base height trace;
4. ``|p(wheel) - p(back_calf_origin)|``, which is a *rigid* offset (0.2715 m on the
   left leg) and therefore stays constant whether or not the loop is closed -- it is
   reported only to document that it is not a closure metric.

PASS requires: 10 DOFs with the expected names, max loop error < 5 mm, and the base
staying within 5 cm of the measured nominal stance height for all 200 steps.

Run:
    D:\\miniconda\\envs\\RM\\python.exe isaac_ws\\scripts\\check_robot.py --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
for _p in (_HERE, _WS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test of the infantry-rl balance environment.")
parser.add_argument("--task", type=str, default="Infantry-RL-Balance-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=200, help="Number of random control steps to run.")
parser.add_argument("--hip_action_range", type=float, default=0.35, help="Uniform range of the raw hip action.")
parser.add_argument("--wheel_action_range", type=float, default=0.2, help="Uniform range of the raw wheel action.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for the actions.")
parser.add_argument(
    "--report", type=str, default=str(_HERE / "check_robot_report.json"), help="Where to write the JSON report."
)
parser.add_argument("--residual_tol", type=float, default=5.0e-3, help="Loop-error tolerance (m).")
parser.add_argument("--height_tol", type=float, default=5.0e-2, help="Base-height tolerance (m).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_ws_infantry  # noqa: E402,F401  (registers the gym tasks)
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaac_ws_infantry.robot_model import (  # noqa: E402
    HIP_ORDER,
    KNEE_ORDER,
    SIDES,
    WHEEL_ORDER,
    loop_anchor_offsets,
    nominal_base_height,
)

EXPECTED_JOINTS = set(HIP_ORDER) | set(KNEE_ORDER) | set(WHEEL_ORDER)


def main() -> int:
    report: dict = {}
    failures: list[str] = []

    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    num_envs = env.num_envs
    device = env.device

    action_dim = env.action_space.shape[-1]
    report["task"] = args_cli.task
    report["num_envs"] = num_envs
    report["action_dim"] = int(action_dim)
    print(f"[smoke] task={args_cli.task} num_envs={num_envs} action_dim={action_dim} device={device}", flush=True)

    env.reset()
    robot = env.scene["robot"]
    joint_action = env.joint_action
    model = joint_action.model

    # ------------------------------------------------------------------ #
    # 1. articulation sanity
    # ------------------------------------------------------------------ #
    report["num_joints"] = int(robot.num_joints)
    report["num_bodies"] = int(robot.num_bodies)
    report["joint_names"] = list(robot.joint_names)
    report["body_names"] = list(robot.body_names)
    print(f"[smoke] num_joints={robot.num_joints} num_bodies={robot.num_bodies}")
    print(f"[smoke] joint_names={list(robot.joint_names)}")
    if robot.num_joints != 10:
        failures.append(f"expected 10 articulation DOFs, got {robot.num_joints}")
    missing = EXPECTED_JOINTS - set(robot.joint_names)
    if missing:
        failures.append(f"missing joints in the articulation: {sorted(missing)}")
    if any("calf_axle" in n for n in robot.joint_names):
        failures.append("loop joints unexpectedly present as DOFs")

    # body ids for the independent geometric measurement
    wheel_bids = [robot.body_names.index(f"{s}_wheel") for s in SIDES]
    calf_bids = [robot.body_names.index(f"{s}_back_calf") for s in SIDES]
    anchors = loop_anchor_offsets()
    anchor_vec = torch.tensor([anchors[s] for s in SIDES], dtype=torch.float32, device=device)  # (2, 3)
    report["loop_anchor_offsets"] = {s: list(anchors[s]) for s in SIDES}
    report["loop_anchor_norm"] = {s: float(sum(v * v for v in anchors[s]) ** 0.5) for s in SIDES}

    target_height = nominal_base_height()
    report["target_base_height"] = target_height
    print(f"[smoke] nominal base height = {target_height:.5f} m   loop anchor |offset| = "
          f"{ {s: round(report['loop_anchor_norm'][s], 5) for s in SIDES} }")

    # ------------------------------------------------------------------ #
    # 2. random steps
    # ------------------------------------------------------------------ #
    height_trace, residual_trace, solver_trace, link_trace = [], [], [], []
    terminations = 0
    for step in range(args_cli.steps):
        action = torch.zeros(num_envs, action_dim, device=device)
        action[:, :4] = (torch.rand(num_envs, 4, device=device) * 2.0 - 1.0) * args_cli.hip_action_range
        action[:, 4:] = (torch.rand(num_envs, 2, device=device) * 2.0 - 1.0) * args_cli.wheel_action_range
        _, _, terminated, truncated, _ = env.step(action)
        terminations += int(terminated.sum().item()) + int(truncated.sum().item())

        # (1) geometric loop error from the simulated link poses
        p_wheel = robot.data.body_pos_w[:, wheel_bids]            # (N, 2, 3)
        p_calf = robot.data.body_pos_w[:, calf_bids]              # (N, 2, 3)
        q_calf = robot.data.body_quat_w[:, calf_bids]             # (N, 2, 4) wxyz
        anchor_w = p_calf + quat_apply(
            q_calf.reshape(-1, 4), anchor_vec.unsqueeze(0).expand(num_envs, -1, -1).reshape(-1, 3)
        ).reshape(num_envs, 2, 3)
        res_geom = (p_wheel - anchor_w).norm(dim=-1)              # (N, 2)
        link_dist = (p_wheel - p_calf).norm(dim=-1)               # (N, 2) rigid offset

        # (2) the same residual through the verified planar model, from measured joints
        res_model = env.closure_residual

        base_z = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

        residual_trace.append(float(res_geom.max().item()))
        solver_trace.append(float(res_model.max().item()))
        link_trace.append(float(link_dist.max().item()))
        height_trace.append(float(base_z.mean().item()))

        if step % 20 == 0 or step == args_cli.steps - 1:
            print(
                f"[smoke] step {step:4d}  base_z={height_trace[-1]:.4f}  "
                f"loop_err(geom)={residual_trace[-1]:.3e}  loop_err(joints)={solver_trace[-1]:.3e}  "
                f"|wheel-calf|={link_trace[-1]:.4f}  clamped={int(env.knee_target_clamped.sum().item())}",
                flush=True,
            )

    max_res = max(residual_trace)
    max_res_model = max(solver_trace)
    min_z, max_z = min(height_trace), max(height_trace)
    height_err = max(abs(min_z - target_height), abs(max_z - target_height))

    report["steps"] = args_cli.steps
    report["hip_action_range"] = args_cli.hip_action_range
    report["wheel_action_range"] = args_cli.wheel_action_range
    report["max_loop_error_geom_m"] = max_res
    report["max_loop_error_joints_m"] = max_res_model
    report["final_loop_error_geom_m"] = residual_trace[-1]
    report["mean_loop_error_geom_m"] = sum(residual_trace) / len(residual_trace)
    report["base_height_min"] = min_z
    report["base_height_max"] = max_z
    report["base_height_mean"] = sum(height_trace) / len(height_trace)
    report["base_height_final"] = height_trace[-1]
    report["max_height_error_m"] = height_err
    report["wheel_calf_distance_min"] = min(link_trace)
    report["wheel_calf_distance_max"] = max(link_trace)
    report["terminations"] = terminations
    report["height_trace"] = height_trace
    report["loop_error_trace_geom"] = residual_trace
    report["loop_error_trace_joints"] = solver_trace
    report["knee_targets_clamped_total"] = joint_action.knee_clamped_total
    report["nominal_joint_pos"] = {
        "hips": [float(v) for v in joint_action.nominal_hips[0]],
        "knees": [float(v) for v in joint_action.nominal_knees[0]],
    }
    report["knee_targets_final"] = [float(v) for v in joint_action.knee_targets[0]]

    # ------------------------------------------------------------------ #
    # 3. verdict
    # ------------------------------------------------------------------ #
    if max_res >= args_cli.residual_tol:
        failures.append(
            f"max loop error {max_res * 1e3:.3f} mm exceeds the {args_cli.residual_tol * 1e3:.1f} mm tolerance"
        )
    if height_err >= args_cli.height_tol:
        failures.append(
            f"base height left the target by {height_err * 1e3:.1f} mm (tolerance "
            f"{args_cli.height_tol * 1e3:.1f} mm): z in [{min_z:.4f}, {max_z:.4f}] vs target {target_height:.4f}"
        )

    report["failures"] = failures
    report["passed"] = not failures

    print()
    print("[smoke] ================ SUMMARY ================")
    print(f"[smoke] articulation      : {robot.num_joints} DOFs, action dim {action_dim}")
    print(f"[smoke] max loop error    : {max_res * 1e3:.4f} mm   (from simulated link poses, tol "
          f"{args_cli.residual_tol * 1e3:.1f} mm)")
    print(f"[smoke] max loop error    : {max_res_model * 1e3:.4f} mm   (same residual from measured joint angles)")
    print(f"[smoke] base height       : min {min_z:.4f}  max {max_z:.4f}  mean "
          f"{report['base_height_mean']:.4f}  target {target_height:.4f} m")
    print(f"[smoke] max height error  : {height_err * 1e3:.2f} mm   (tol {args_cli.height_tol * 1e3:.1f} mm)")
    print(f"[smoke] |wheel - calf|    : {min(link_trace):.4f} .. {max(link_trace):.4f} m  (rigid offset, not a "
          f"closure metric)")
    print(f"[smoke] knee targets clamped: {joint_action.knee_clamped_total} times")
    print(f"[smoke] terminations/resets : {terminations}")
    print(f"[smoke] report written to   : {args_cli.report}")

    Path(args_cli.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if failures:
        print()
        for f in failures:
            print(f"[smoke] FAIL: {f}")
        print("[smoke] RESULT: FAIL")
        return 1
    print("[smoke] RESULT: PASS")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
