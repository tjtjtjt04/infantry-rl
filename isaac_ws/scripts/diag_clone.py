"""Diagnose the multi-environment clone hang.

With ``num_envs >= 2`` the scene hangs inside ``sim.reset()`` right after Kit prints
"Starting the simulation".  That is where the scene clones env_0 into the remaining
environments, so this script rebuilds the same environment with the clone-related
flags selectable from the command line, and prints a heartbeat so it is obvious
whether physics ever starts.

Run one variant at a time, e.g.::

    python isaac_ws/scripts/diag_clone.py --num_envs 4 --replicate-physics false

Options:
    --replicate-physics {true,false}   SceneCfg.replicate_physics
    --clone-in-fabric   {true,false}   SceneCfg.clone_in_fabric
    --num_envs N                       number of environments
    --steps N                          physics steps to run after reset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent  # .../isaac_ws
if str(WS) not in sys.path:
    sys.path.insert(0, str(WS))


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose multi-env cloning.")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--replicate-physics", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--clone-in-fabric", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--filter-collisions", type=str, default="true", choices=["true", "false"])
    parser.add_argument(
        "--asset",
        type=str,
        default="full",
        choices=["full", "proxy"],
        help="full = the converted CAD USD (96 MB); proxy = the primitive proxy USD",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    args, _ = parser.parse_known_args()

    import os

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=args.headless, enable_cameras=False)
    app = launcher.app

    t0 = time.time()

    def beat(msg: str) -> None:
        print(f"[diag +{time.time() - t0:7.1f}s] {msg}", flush=True)

    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import isaac_ws_infantry  # noqa: F401  (registers the tasks)

    beat("Kit up; parsing env cfg")
    env_cfg = parse_env_cfg("Infantry-RL-Balance-v0", device="cuda:0", num_envs=args.num_envs)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.replicate_physics = args.replicate_physics == "true"
    env_cfg.scene.clone_in_fabric = args.clone_in_fabric == "true"
    env_cfg.scene.filter_collisions = args.filter_collisions == "true"
    if args.asset == "proxy":
        proxy = WS / "assets" / "infantry-rl" / "infantry_rl_proxy.usd"
        env_cfg.scene.robot.spawn.usd_path = str(proxy)
    beat(
        f"scene: num_envs={env_cfg.scene.num_envs}"
        f" replicate_physics={env_cfg.scene.replicate_physics}"
        f" clone_in_fabric={env_cfg.scene.clone_in_fabric}"
        f" filter_collisions={env_cfg.scene.filter_collisions}"
        f" asset={args.asset}"
    )

    beat("building env (gym.make) -- this is where the hang would show up")
    env = gym.make("Infantry-RL-Balance-v0", cfg=env_cfg).unwrapped
    beat("gym.make returned")

    env.reset()
    beat("env.reset() returned")

    robot = env.scene["robot"]
    beat(f"scene up: num_envs={env.num_envs} bodies={robot.num_bodies} joints={robot.num_joints}")

    action_dim = env.action_space.shape[-1]
    for i in range(args.steps):
        action = torch.zeros(env.num_envs, action_dim, device=env.device)
        env.step(action)
        if i % max(1, args.steps // 5) == 0:
            beat(f"step {i} base_z={robot.data.root_pos_w[0, 2].item():.4f}")

    beat(f"DONE num_envs={args.num_envs} steps={args.steps}")
    env.close()
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
