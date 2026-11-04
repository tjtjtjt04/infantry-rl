"""Play (evaluate) a trained RSL-RL checkpoint on the infantry-rl robot.

Examples (run from this folder):
    python play.py --checkpoint runs/.../model_5000.pt --vx 1.0 --headless --steps 2000
    python play.py --checkpoint model_5000.pt --cmd_vx 1.0 --cmd_heading_deg 0 --real-time
    python play.py --random_policy --headless --steps 100 --cmd_vx 0.3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained RSL-RL agent for the infantry-rl robot.")
    parser.add_argument("--task", type=str, default="locomotion",
                        help="locomotion | turning | braking | teleop | recovery | stand | "
                             "balance | squat | legacy Infantry-RL-* ID")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--load_run", type=str, default=None)
    parser.add_argument("--log_root", type=str, default=None,
                        help="root of the rsl_rl logs (default: <project>/logs/rsl_rl)")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--use_last_checkpoint", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=500, help="control steps to run (0 = until the GUI closes)")
    parser.add_argument("--real-time", action="store_true", default=False)
    parser.add_argument("--cmd_vx", type=float, default=None,
                        help="hold this forward command (m/s) instead of a random one")
    parser.add_argument("--cmd_heading_deg", type=float, default=None,
                        help="hold this heading command (degrees); defaults to 0 with --cmd_vx "
                             "(heading-command tasks: locomotion/balance)")
    parser.add_argument("--cmd_wz", type=float, default=None,
                        help="hold this yaw-rate command (rad/s, + = left/CCW) instead of a "
                             "random one. Direct-command tasks only (`turning` / `teleop`); "
                             "with only --cmd_wz the forward command is 0, i.e. rotate in place")
    parser.add_argument("--brake_after", type=float, default=None,
                        help="braking task: seconds at the target speed before the stop command. "
                             "With --cmd_vx this fixes the whole scenario, e.g. "
                             "--cmd_vx 3.0 --brake_after 3.0")
    parser.add_argument("--stop_after", type=float, default=None,
                        help="teleop task: seconds at the fixed --cmd_vx/--cmd_wz before issuing "
                             "(vx=0, wz=0). Uses the same stop-distance measurement as the "
                             "braking task; recommended with --cmd_vx, e.g. --cmd_vx 3.0 "
                             "--stop_after 6.0")
    parser.add_argument("--cmd_ramp_s", type=float, default=0.0,
                        help="teleop direct command: ramp the fixed command from zero to the "
                             "target over this many seconds (smoothstep). Useful for high-speed "
                             "starts; 0 disables.")
    parser.add_argument("--switch_after", type=float, default=None,
                        help="teleop task: after N seconds switch the fixed command to "
                             "--cmd_vx2/--cmd_wz2, e.g. reverse -> forward or curve -> straight")
    parser.add_argument("--cmd_vx2", type=float, default=None,
                        help="teleop task: second-phase forward command for --switch_after (m/s)")
    parser.add_argument("--cmd_wz2", type=float, default=None,
                        help="teleop task: second-phase yaw-rate command for --switch_after (rad/s)")
    parser.add_argument("--teleop_no_heading_hold", action="store_true", default=False,
                        help="teleop task: disable the conditional heading PD for an A/B test")
    parser.add_argument("--teleop_heading_hold_wz", type=float, default=None,
                        help="teleop task: override the conditional heading PD's |wz_cmd| "
                             "threshold (rad/s)")
    parser.add_argument("--teleop_heading_lat_accel", type=float, default=None,
                        help="teleop task: override the conditional heading PD's lateral "
                             "acceleration cap |vx*wz_pd| (m/s^2; 0 disables)")
    parser.add_argument("--hip_scale", type=float, default=None,
                        help="hip action authority (rad per action unit; default 0.25). "
                             "Must match the training run.")
    parser.add_argument("--hip_range", type=float, default=None,
                        help="hard hip travel around nominal (rad; default 0.45). "
                             "Must match the training run (e.g. 0.60).")
    parser.add_argument("--stance_bias", type=float, default=None,
                        help="fixed scripted stance in rad for replay: 0 = nominal, "
                             "+0.30 = legs extended (~6 cm higher), -0.55 = deep crouch (~10 cm lower)")
    parser.add_argument("--stance_snap", action="store_true", default=False,
                        help="apply --stance_bias immediately instead of slewing at the "
                             "trained stance rate")
    parser.add_argument("--hip_symmetry", choices=("off", "master_rf"), default=None,
                        help="teleop: master_rf slaves all four hip targets to the right-front "
                             "leg ([+t,-t,+t,-t]); use it with a checkpoint trained that way")
    parser.add_argument("--hip_symmetry_gate", action="store_true", default=False,
                        help="teleop: apply master_rf only for in-place commands; must match "
                             "the checkpoint's training gate")
    parser.add_argument("--debug_heading", action="store_true", default=False,
                        help="print wheel commands / heading loop / actual wheel speeds every 50 steps")
    parser.add_argument("--random_policy", action="store_true", default=False,
                        help="skip the checkpoint and run a freshly initialised policy (smoke test)")
    parser.add_argument("--zero_action", action="store_true", default=False,
                        help="skip the policy and send zero actions; used to check the "
                             "position-controller / scripted stand-up behaviour")
    parser.add_argument("--check", action="store_true", default=False,
                        help="run a short environment diagnostic instead of a policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--snapshot", action="store_true", default=False,
                        help="save one PNG of the rendered view at the end (needs cameras; "
                             "reliable where --video's mp4 needs a clean shutdown)")
    parser.add_argument("--camera", choices=("free", "follow", "fixed"), default=None,
                        help="free: set once, then user-controlled (default); "
                             "follow: track the robot at a fixed relative offset; "
                             "fixed: keep a fixed world view every frame")
    parser.add_argument("--camera_eye", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                        help="camera position: world coordinates (free/fixed) or an offset "
                             "from the robot root (follow)")
    parser.add_argument("--camera_lookat", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                        help="camera target: world coordinates (free/fixed) or an offset "
                             "from the robot root (follow)")
    parser.add_argument("--follow", action="store_true", default=False,
                        help="alias for --camera follow (kept for compatibility)")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
if args_cli.check:
    args_cli.headless = True
if args_cli.video or args_cli.snapshot:
    args_cli.enable_cameras = True

# setting a fixed command does not need the range override, but keeping the env vars
# consistent avoids a surprising printed envelope in play
if args_cli.cmd_vx is not None:
    os.environ.setdefault("INFANTRY_RL_VX_MIN", str(args_cli.cmd_vx))
    os.environ.setdefault("INFANTRY_RL_VX_MAX", str(args_cli.cmd_vx))
# teleop heading-PD overrides must be in the environment before env_teleop is
# imported (it reads them at module import time)
if args_cli.teleop_no_heading_hold:
    os.environ["INFANTRY_RL_TELEOP_HEADING_HOLD"] = "0"
if args_cli.teleop_heading_hold_wz is not None:
    os.environ["INFANTRY_RL_TELEOP_HEADING_HOLD_WZ"] = str(args_cli.teleop_heading_hold_wz)
if args_cli.teleop_heading_lat_accel is not None:
    os.environ["INFANTRY_RL_TELEOP_HEADING_LAT_ACCEL"] = str(args_cli.teleop_heading_lat_accel)
if args_cli.hip_scale is not None:
    os.environ["INFANTRY_RL_HIP_SCALE"] = str(args_cli.hip_scale)
if args_cli.hip_range is not None:
    os.environ["INFANTRY_RL_HIP_RANGE"] = str(args_cli.hip_range)
if args_cli.hip_symmetry is not None:
    os.environ["INFANTRY_RL_TELEOP_HIP_SYMMETRY"] = str(args_cli.hip_symmetry)
if args_cli.hip_symmetry_gate:
    os.environ["INFANTRY_RL_TELEOP_HIP_SYMMETRY_GATE"] = "1"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows (imports require Kit to be running)."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.dict import print_dict  # noqa: E402

import agent as agent_lib  # noqa: E402
from config import DEFAULT_LOG_ROOT, normalize_task  # noqa: E402
from env_balance import BalanceEnvCfg_PLAY, FiveBarBalanceEnv  # noqa: E402
from env_braking import BrakingEnvCfg_PLAY, FiveBarBrakingEnv  # noqa: E402
from env_locomotion import LocomotionEnvCfg_PLAY, FiveBarLocomotionEnv  # noqa: E402
from env_recovery import FiveBarRecoveryEnv, RecoveryEnvCfg_PLAY  # noqa: E402
from env_squat import SQUAT_BASE_HEIGHT, FiveBarSquatEnv, SquatEnvCfg_PLAY  # noqa: E402
from env_stand import FiveBarStandEnv, StandEnvCfg_PLAY  # noqa: E402
from env_teleop import FiveBarTeleopEnv, TeleopEnvCfg_PLAY  # noqa: E402
from env_turning import FiveBarTurningEnv, TurningEnvCfg_PLAY  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402


def build_play_env(task: str, num_envs: int | None, device: str, seed: int, render_mode: str | None = None):
    if task == "locomotion":
        env_cfg = LocomotionEnvCfg_PLAY()
        env_cls = FiveBarLocomotionEnv
    elif task == "turning":
        env_cfg = TurningEnvCfg_PLAY()
        env_cls = FiveBarTurningEnv
    elif task == "braking":
        env_cfg = BrakingEnvCfg_PLAY()
        env_cls = FiveBarBrakingEnv
    elif task == "teleop":
        env_cfg = TeleopEnvCfg_PLAY()
        env_cls = FiveBarTeleopEnv
    elif task == "recovery":
        env_cfg = RecoveryEnvCfg_PLAY()
        env_cls = FiveBarRecoveryEnv
    elif task == "stand":
        env_cfg = StandEnvCfg_PLAY()
        env_cls = FiveBarStandEnv
    elif task == "balance":
        env_cfg = BalanceEnvCfg_PLAY()
        env_cls = FiveBarBalanceEnv
    elif task == "squat":
        env_cfg = SquatEnvCfg_PLAY()
        env_cls = FiveBarSquatEnv
    else:
        raise ValueError(f"unknown task {task!r}")
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.sim.device = device
    env_cfg.log_dir = os.path.join(os.getcwd(), "logs", "play")
    return env_cls(cfg=env_cfg, render_mode=render_mode)


def run_check(env, task: str) -> int:
    robot = env.scene["robot"]
    joints = env.action_manager.get_term(env.cfg.joint_action_name)
    print(f"[check] task={task} num_envs={env.cfg.scene.num_envs} "
          f"obs={tuple(env.single_observation_space['policy'].shape)} "
          f"actions={tuple(env.single_action_space.shape)}")
    print(f"[check] joints={robot.joint_names}")
    print(f"[check] hip ids={joints.hip_ids}  knee ids={joints.knee_ids}  wheel ids={joints.wheel_ids}")
    print(f"[check] closure after reset: {env.max_closure_residual * 1000:.6f} mm")
    obs, extras = env.reset()
    action = torch.zeros((env.cfg.scene.num_envs, env.single_action_space.shape[0]), device=env.device)
    obs, reward, terminated, truncated, extras = env.step(action)
    obs = obs["policy"] if isinstance(obs, dict) else obs
    ok = bool(torch.isfinite(obs).all()) and bool(torch.isfinite(reward).all())
    ok &= env.max_closure_residual < 5e-3
    print(f"[check] one step: reward={reward.mean().item():.4f} "
          f"done={int((terminated | truncated).sum())} closure={env.max_closure_residual * 1000:.6f} mm")
    print("[check] RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    import importlib.metadata as metadata

    task = normalize_task(args_cli.task)
    agent_cfg = agent_lib.make_agent_cfg(task, play=True)
    agent_cfg = agent_lib.apply_cli_overrides(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    log_root = Path(args_cli.log_root) if args_cli.log_root else DEFAULT_LOG_ROOT
    log_root_path = os.path.abspath(log_root / agent_cfg.experiment_name)

    env = build_play_env(
        task,
        args_cli.num_envs,
        args_cli.device or "cuda:0",
        args_cli.seed,
        render_mode="rgb_array" if (args_cli.video or args_cli.snapshot) else None,
    )
    # ------------------------------------------------------------------ #
    # Camera modes
    #   free   : set once, then the viewport is under the user's control (default)
    #   follow : the viewport tracks the robot at a fixed relative offset
    #   fixed  : a fixed world view, re-applied every control step
    # ------------------------------------------------------------------ #
    if args_cli.camera is None:
        camera_mode = "follow" if args_cli.follow else "free"
    else:
        camera_mode = args_cli.camera
        if args_cli.follow and camera_mode != "follow":
            print("[play] --follow is ignored because --camera was given")

    cfg_viewer = env.cfg.viewer
    camera_eye = tuple(args_cli.camera_eye) if args_cli.camera_eye is not None else tuple(cfg_viewer.eye)
    camera_lookat = (
        tuple(args_cli.camera_lookat) if args_cli.camera_lookat is not None else tuple(cfg_viewer.lookat)
    )
    # Rendering is available in the GUI, or in headless mode when cameras were enabled
    # (--snapshot / --video).
    camera_enabled = (not args_cli.headless) or args_cli.video or args_cli.snapshot

    controller = getattr(env, "viewport_camera_controller", None)
    if camera_mode == "follow":
        if controller is None:
            print("[play] camera=follow ignored: no viewport (headless without --snapshot/--video)")
        else:
            # eye/lookat are offsets FROM THE ROBOT ROOT; the controller adds the root pos.
            controller.default_cam_eye = np.array(camera_eye, dtype=float)
            controller.default_cam_lookat = np.array(camera_lookat, dtype=float)
            controller.update_view_to_asset_root("robot")
            print(
                f"[play] camera=follow: eye offset={camera_eye}, lookat offset={camera_lookat} "
                "(relative to the robot root)"
            )
    elif camera_enabled:
        env.sim.set_camera_view(
            eye=camera_eye, target=camera_lookat, camera_prim_path=cfg_viewer.cam_prim_path
        )
        note = " (re-applied every step)" if camera_mode == "fixed" else " (user-controlled from now on)"
        print(f"[play] camera={camera_mode}: eye={camera_eye}, lookat={camera_lookat}{note}")
    else:
        print(f"[play] camera={camera_mode}: no rendering in this headless run; camera settings ignored")

    apply_fixed_camera = camera_mode == "fixed" and camera_enabled

    if args_cli.check:
        return run_check(env, task)

    resume_path: str | None = None
    if args_cli.zero_action:
        print("[play] --zero_action: skipping the checkpoint; sending zero actions")
    elif not args_cli.random_policy:
        resume_path = agent_lib.resolve_play_checkpoint(
            log_root_path,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
            use_last_checkpoint=args_cli.use_last_checkpoint,
        )
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        # match the network shape to the checkpoint, not to the current config
        agent_lib.match_network_to_checkpoint(resume_path, agent_cfg)
    else:
        print("[play] --random_policy: no checkpoint, running a freshly initialised policy")

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if resume_path is not None:
        runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.device)
    if args_cli.zero_action:
        action_dim = env.unwrapped.single_action_space.shape[0]
        policy = lambda _obs: torch.zeros(
            (env.unwrapped.num_envs, action_dim), device=env.unwrapped.device
        )

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    steps = args_cli.steps
    residuals: list[float] = []
    heights: list[float] = []
    headings: list[float] = []
    yaw_rates: list[float] = []
    rolls: list[float] = []
    pitches: list[float] = []
    hip_devs: list[float] = []
    phis: list[float] = []
    l0_spans: list[float] = []
    dphis: list[float] = []
    dl0s: list[float] = []
    axle_skew: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    vxs: list[float] = []
    vxs_b: list[float] = []
    cmd_vxs: list[float] = []
    cmd_wzs: list[float] = []
    wz_errs: list[float] = []
    # braking-task measurements (first brake event of the run)
    brake_onset_speed: float | None = None
    brake_stop_distance: float | None = None
    brake_stop_time: float | None = None
    brake_last_distance: float = 0.0

    robot = env.unwrapped.scene["robot"]
    origins = env.unwrapped.scene.env_origins

    # tracking statistics for whichever task is active (locomotion / turning)
    _cmd_mgr = env.unwrapped.command_manager
    _has_vel_cmd = "base_velocity" in getattr(_cmd_mgr, "active_terms", [])

    # ------------------------------------------------------------------ #
    # optional fixed velocity command (locomotion task)
    #
    # Without this the command term samples a random command when the environment
    # is created, so two runs of the same checkpoint are not comparable.  Pinning
    # the command makes the numbers reproducible.
    # ------------------------------------------------------------------ #
    is_braking = task == "braking"
    is_recovery = task == "recovery"
    is_stand = task == "stand"
    is_squat = task == "squat"
    if (is_recovery or is_stand or is_squat) and (
        args_cli.cmd_vx is not None
        or args_cli.cmd_wz is not None
        or args_cli.cmd_heading_deg is not None
    ):
        print(
            f"[play] NOTE: {task} uses a constant zero command; the --cmd_* flags are ignored"
        )
    fixed_cmd = (
        args_cli.cmd_vx is not None
        or args_cli.cmd_heading_deg is not None
        or args_cli.cmd_wz is not None
    ) and not is_braking and not is_recovery and not is_stand and not is_squat
    _cmd_term = None
    _cmd_is_heading = True
    _brake_term = None

    if fixed_cmd:
        try:
            _cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        except (AttributeError, KeyError, ValueError):
            print("[play] fixed velocity commands ignored: this task has no 'base_velocity' command")
            fixed_cmd = False
            _cmd_term = None

    if fixed_cmd:
        # The locomotion task derives wz from a heading target; the turning task takes
        # (vx, wz) directly.  Pin the command in the form the active term understands.
        _cmd_is_heading = bool(getattr(_cmd_term.cfg, "heading_command", False))

        if _cmd_is_heading and args_cli.cmd_wz is not None:
            print(
                "[play] NOTE: --cmd_wz is ignored by a heading-command task; use "
                "--cmd_heading_deg instead (or --task turning for a direct yaw rate)"
            )
        if not _cmd_is_heading and args_cli.cmd_heading_deg is not None:
            print(
                "[play] NOTE: --cmd_heading_deg is ignored by a direct (vx, wz) task; "
                "use --cmd_wz <rad/s> instead"
            )

        def _apply_fixed_cmd(term) -> None:
            term.is_standing_env[:] = False  # do not let the standing pass zero our command
            term.vel_command_b[:] = 0.0
            if _cmd_is_heading:
                term.is_heading_env[:] = True  # heading controller owns wz
                term.heading_target[:] = math.radians(args_cli.cmd_heading_deg or 0.0)
                if args_cli.cmd_vx is not None:
                    term.vel_command_b[:, 0] = args_cli.cmd_vx
            else:
                term.is_heading_env[:] = False
                term.vel_command_b[:, 0] = args_cli.cmd_vx if args_cli.cmd_vx is not None else 0.0
                term.vel_command_b[:, 2] = args_cli.cmd_wz if args_cli.cmd_wz is not None else 0.0
            # recompute the post-processed command (heading loop / standing pass) so the
            # next observation is consistent with what the policy is about to be told
            term._update_command()

        _apply_fixed_cmd(_cmd_term)
        fixed_target_vx = args_cli.cmd_vx if args_cli.cmd_vx is not None else 0.0
        fixed_target_wz = args_cli.cmd_wz if args_cli.cmd_wz is not None else 0.0
        ramp_start_step = 0
        if _cmd_is_heading:
            if args_cli.cmd_heading_deg is not None and args_cli.cmd_vx is None:
                print(
                    "[play] NOTE: --cmd_heading_deg without --cmd_vx pins vx=0 "
                    "(turn in place while holding the requested heading)"
                )
            print(
                f"[play] fixed command: vx={args_cli.cmd_vx} m/s, "
                f"heading={args_cli.cmd_heading_deg or 0.0:.1f} deg "
                f"(wz is derived from the heading error every step)"
            )
        else:
            v_cmd = args_cli.cmd_vx if args_cli.cmd_vx is not None else 0.0
            w_cmd = args_cli.cmd_wz if args_cli.cmd_wz is not None else 0.0
            print(
                f"[play] fixed command: vx={v_cmd:.3f} m/s, wz={w_cmd:+.3f} rad/s "
                f"(direct; + = left/CCW)"
            )
            # A differential drive couples forward speed and yaw rate: the body follows
            # an arc of radius |vx/wz|.  Printing it makes "why is it a big circle?"
            # a number instead of a guess.
            if abs(w_cmd) > 1e-9:
                side = "in-place rotation" if abs(v_cmd) < 1e-9 else ("left" if w_cmd > 0 else "right")
                print(
                    f"[play] expected path: circle of radius |vx/wz| = "
                    f"{abs(v_cmd / w_cmd):.2f} m, {side}"
                )
            else:
                print("[play] expected path: straight line (wz=0)")
        # refresh the observation so the first policy action already sees the pinned
        # command instead of the random one sampled at startup
        obs = env.get_observations()

    if is_braking:
        # The braking task owns its own two-phase schedule, so the generic fixed-command
        # path above is skipped.  Pin the target speed and drive time in the term's cfg
        # and restart the schedule; every later reset reuses these values.
        _brake_term = env.unwrapped.command_manager.get_term("base_velocity")
        if args_cli.cmd_wz is not None or args_cli.cmd_heading_deg is not None:
            print("[play] NOTE: --cmd_wz/--cmd_heading_deg are ignored by the braking task")
        if args_cli.cmd_vx is not None:
            _brake_term.cfg.ranges.lin_vel_x = (float(args_cli.cmd_vx), float(args_cli.cmd_vx))
        if args_cli.brake_after is not None:
            _brake_term.cfg.drive_time_range = (
                float(args_cli.brake_after),
                float(args_cli.brake_after),
            )
        env_ids = torch.arange(env.unwrapped.num_envs, device=env.device)
        _brake_term._resample_command(env_ids)
        _brake_term._update_command()
        target = float(_brake_term.target_speed[0])
        drive_s = float(_brake_term._drive_steps[0]) * env.unwrapped.step_dt
        print(
            f"[play] braking scenario: drive at {target:.2f} m/s for {drive_s:.2f} s, "
            f"then vx=0 (stop criterion |vx|<{_brake_term.cfg.stop_speed_threshold:.2f} m/s)"
        )
        obs = env.get_observations()

    # ------------------------------------------------------------------ #
    # teleop task: a fixed command, optionally followed by a stop
    # ------------------------------------------------------------------ #
    teleop_stop_after = args_cli.stop_after if task == "teleop" else None
    teleop_switch_after = args_cli.switch_after if task == "teleop" else None
    stop_issued = False
    switch_issued = False
    if args_cli.stop_after is not None and task != "teleop":
        print("[play] NOTE: --stop_after is only used by the teleop task")
    if args_cli.switch_after is not None and task != "teleop":
        print("[play] NOTE: --switch_after is only used by the teleop task")
    if teleop_stop_after is not None:
        if _cmd_term is None:
            _cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        if not fixed_cmd:
            print(
                "[play] NOTE: --stop_after without --cmd_vx/--cmd_wz stops whatever random "
                "command the env sampled at startup; pass a fixed command for a repeatable test"
            )
        # Reuse the braking measurement block (onset speed, distance, time, stopped).
        _brake_term = _cmd_term
        print(
            f"[play] teleop stop scheduled after {teleop_stop_after:.2f} s "
            "(command becomes vx=0, wz=0; heading hold stays active)"
        )

    def _apply_second_cmd() -> None:
        """Second phase of --switch_after (teleop only)."""
        _cmd_term.is_standing_env[:] = False
        _cmd_term.vel_command_b[:] = 0.0
        _cmd_term.vel_command_b[:, 0] = args_cli.cmd_vx2 if args_cli.cmd_vx2 is not None else 0.0
        _cmd_term.vel_command_b[:, 2] = args_cli.cmd_wz2 if args_cli.cmd_wz2 is not None else 0.0
        if abs(float(_cmd_term.vel_command_b[0, 0])) < 1e-6 and abs(float(_cmd_term.vel_command_b[0, 2])) < 1e-6:
            _cmd_term.force_brake()
        else:
            _cmd_term._update_command()

    if teleop_switch_after is not None:
        if _cmd_term is None:
            _cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        if not fixed_cmd:
            print(
                "[play] NOTE: --switch_after without --cmd_vx/--cmd_wz switches from the "
                "random startup command; pass a fixed first command for a repeatable test"
            )
        # A second command of (0, 0) is a stop: reuse the braking measurement block.
        if (args_cli.cmd_vx2 in (None, 0.0)) and (args_cli.cmd_wz2 in (None, 0.0)):
            _brake_term = _cmd_term
        print(
            f"[play] teleop command switch scheduled after {teleop_switch_after:.2f} s -> "
            f"vx={args_cli.cmd_vx2 if args_cli.cmd_vx2 is not None else 0.0:+.2f} m/s, "
            f"wz={args_cli.cmd_wz2 if args_cli.cmd_wz2 is not None else 0.0:+.2f} rad/s"
        )

    def _yaw():
        q = robot.data.root_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _roll_pitch():
        q = robot.data.root_quat_w[0]
        w, x, y, z = (float(v) for v in q)
        # XYZ extrinsic convention, same as isaaclab.utils.math.euler_xyz_from_quat
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch = math.asin(sinp)
        return roll, pitch

    def _xy():
        p = robot.data.root_pos_w[0] - origins[0]
        return float(p[0]), float(p[1])

    # ------------------------------------------------------------------ #
    # leg activity -- "are the legs doing anything, or is the robot balancing
    # purely with the wheels?"  A wheel-only balance keeps the hips pinned at the
    # nominal stance, so all of these stay near zero.
    # ------------------------------------------------------------------ #
    _joints = env.unwrapped.action_manager.get_term("joints")

    if is_squat and args_cli.stance_bias is not None:
        print(
            "[play] NOTE: --stance_bias is ignored by the squat task "
            "(the hips are locked at the deepest pose)"
        )
        args_cli.stance_bias = None
    if args_cli.stance_bias is not None:
        setter = getattr(_joints, "set_stance_target", None)
        if setter is None:
            print("[play] WARNING: this task's action term has no scripted stance; "
                  "--stance_bias ignored")
        else:
            setter(float(args_cli.stance_bias), snap=True)
            print(
                f"[play] scripted stance: {args_cli.stance_bias:+.2f} rad "
                f"(expected base_z shift ~{0.1901 * args_cli.stance_bias + 0.0173 * args_cli.stance_bias ** 2:+.3f} m)"
            )

    def _leg_metrics():
        hip_dev = (_joints._hip_targets[0] - _joints.nominal_hips[0]).abs().mean().item()
        state = _joints.solved_leg_state()[0]  # (L0_l, phi_l, L0_r, phi_r)
        phi = 0.5 * (abs(float(state[1])) + abs(float(state[3])))
        l0_span = abs(float(state[0]) - float(state[2]))
        rate = _joints.leg_rate[0]  # (dL0_l, dphi_l, dL0_r, dphi_r)
        dphi = 0.5 * (abs(float(rate[1])) + abs(float(rate[3])))
        dl0 = 0.5 * (abs(float(rate[0])) + abs(float(rate[2])))
        return float(hip_dev), phi, l0_span, dphi, dl0

    count = 0
    n_dones = 0
    while simulation_app.is_running() and (steps == 0 or count < steps):
        if apply_fixed_camera:
            # keep the chosen world view even if the user moves the viewport by accident
            env.unwrapped.sim.set_camera_view(
                eye=camera_eye, target=camera_lookat, camera_prim_path=cfg_viewer.cam_prim_path
            )
        if (
            fixed_cmd
            and not _cmd_is_heading
            and args_cli.cmd_ramp_s > 0.0
            and not stop_issued
            and not switch_issued
        ):
            ramp_t = min(
                max((count - ramp_start_step) * dt / args_cli.cmd_ramp_s, 0.0), 1.0
            )
            scale = ramp_t * ramp_t * (3.0 - 2.0 * ramp_t)
            _cmd_term.is_standing_env[:] = False
            _cmd_term.vel_command_b[:, 0] = fixed_target_vx * scale
            _cmd_term.vel_command_b[:, 2] = fixed_target_wz * scale
            _cmd_term._update_command()
        if teleop_switch_after is not None and not switch_issued and count * dt >= teleop_switch_after:
            _apply_second_cmd()
            switch_issued = True
            print(
                f"[play] step {count}: command switched to "
                f"vx={args_cli.cmd_vx2 if args_cli.cmd_vx2 is not None else 0.0:+.2f} m/s, "
                f"wz={args_cli.cmd_wz2 if args_cli.cmd_wz2 is not None else 0.0:+.2f} rad/s "
                f"(measured vx={float(robot.data.root_lin_vel_b[0, 0]):+.2f}, "
                f"wz={float(robot.data.root_ang_vel_b[0, 2]):+.2f})",
                flush=True,
            )
        if teleop_stop_after is not None and not stop_issued and count * dt >= teleop_stop_after:
            # Issue the stop BEFORE the step so the policy sees it immediately and the
            # teleop command term records the onset speed/position for the metrics.
            _cmd_term.force_brake()
            stop_issued = True
            print(
                f"[play] step {count}: STOP issued at "
                f"|vx|={abs(float(robot.data.root_lin_vel_b[0, 0])):.2f} m/s, "
                f"|wz|={abs(float(robot.data.root_ang_vel_b[0, 2])):.3f} rad/s",
                flush=True,
            )
        start_time = time.time()
        if args_cli.stance_bias is not None and hasattr(_joints, "set_stance_target"):
            _joints.set_stance_target(
                float(args_cli.stance_bias), snap=bool(args_cli.stance_snap)
            )
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        # a termination auto-resets the env, which re-samples a random command --
        # put the fixed command back so the evaluation stays comparable
        n_dones += int(dones.sum())
        if fixed_cmd and bool(dones.any()):
            _apply_fixed_cmd(_cmd_term)
            if args_cli.stance_bias is not None and hasattr(_joints, "set_stance_target"):
                # the action term reset cleared the stance; restore the replay value
                _joints.set_stance_target(float(args_cli.stance_bias), snap=True)
            # restart a smooth start after an auto-reset
            ramp_start_step = count
            if switch_issued:
                # keep the second phase after an auto-reset
                _apply_second_cmd()
            if stop_issued:
                # stay stopped after a reset: the fixed command was just restored
                _cmd_term.force_brake()
        if _brake_term is not None:
            if bool(_brake_term.brake_started[0]):
                if brake_onset_speed is None:
                    brake_onset_speed = float(_brake_term.brake_speed[0])
                brake_last_distance = float(
                    torch.norm(
                        robot.data.root_pos_w[0, :2] - _brake_term.brake_position[0]
                    ).item()
                )
                if bool(_brake_term.stopped[0]) and brake_stop_distance is None:
                    brake_stop_distance = float(_brake_term.stop_distance[0])
                    brake_stop_time = (
                        float(_brake_term.stop_step[0] - _brake_term.brake_step[0]) * dt
                    )
        residuals.append(env.unwrapped.max_closure_residual)
        heights.append(float(robot.data.root_pos_w[:, 2].mean().item()))
        headings.append(_yaw())
        yaw_rates.append(float(robot.data.root_ang_vel_w[:, 2].mean().item()))
        _roll, _pitch = _roll_pitch()
        rolls.append(_roll)
        pitches.append(_pitch)
        _hdev, _phi, _l0, _dphi, _dl0 = _leg_metrics()
        hip_devs.append(_hdev)
        phis.append(_phi)
        l0_spans.append(_l0)
        dphis.append(_dphi)
        dl0s.append(_dl0)
        # |L0_l*sin(phi_l) - L0_r*sin(phi_r)|: how far the two wheel contact points
        # are out of line fore-aft.  Should be ~0 when the axle is straight.
        _st = _joints.solved_leg_state()[0]
        axle_skew.append(abs(float(_st[0]) * math.sin(float(_st[1]))
                             - float(_st[2]) * math.sin(float(_st[3]))))
        _x, _y = _xy()
        xs.append(_x)
        ys.append(_y)
        vxs.append(float(robot.data.root_lin_vel_w[0, 0].item()))
        vxs_b.append(float(robot.data.root_lin_vel_b[0, 0].item()))
        _cmd_now = None
        if _has_vel_cmd:
            _cmd_now = _cmd_mgr.get_command("base_velocity")[0]
            cmd_vxs.append(float(_cmd_now[0]))
            cmd_wzs.append(float(_cmd_now[2]))
            wz_errs.append(abs(float(_cmd_now[2]) - float(robot.data.root_ang_vel_b[0, 2])))
        if count % 50 == 0:
            _cmd_txt = (
                f"cmd(vx={float(_cmd_now[0]):+5.2f} wz={float(_cmd_now[2]):+5.2f})  "
                if _cmd_now is not None
                else ""
            )
            print(
                f"[play] step {count:4d}  base_z={heights[-1]:.4f}  "
                f"x={xs[-1]:+6.3f}m y={ys[-1]:+6.3f}m vx={vxs[-1]:+6.3f}m/s  "
                f"{_cmd_txt}"
                f"heading={math.degrees(headings[-1]):+7.2f} deg  "
                f"pitch={math.degrees(pitches[-1]):+6.2f} deg  "
                f"yaw_rate={yaw_rates[-1]:+.3f} rad/s  "
                f"closure_max={residuals[-1]:.3e} m",
                flush=True,
            )
            if args_cli.debug_heading:
                term = env.unwrapped.action_manager.get_term("joints")
                wv = robot.data.joint_vel[0, term.wheel_ids].tolist()
                fd = (headings[-1] - headings[-2]) / dt if len(headings) > 1 else 0.0
                _cmd = env.unwrapped.command_manager.get_command("base_velocity")[0]
                print(
                    f"       [dbg] cmd_vx={float(_cmd[0]):+.3f} m/s  cmd_wz={float(_cmd[2]):+.4f} rad/s"
                    f"  heading_err={math.degrees(math.atan2(math.sin(-headings[-1]), math.cos(-headings[-1]))):+.2f} deg"
                    f"\n       [dbg] policy wheel cmd={[round(float(v), 2) for v in term._wheel_targets[0]]}"
                    f"  heading_delta={float(term.heading_delta[0]):+.2f}"
                    f"  actual wheel vel={[round(float(v), 2) for v in wv]}"
                    f"\n       [dbg] yaw_rate(world_z)={yaw_rates[-1]:+.4f}"
                    f"  yaw_rate(body_z)={float(robot.data.root_ang_vel_b[0, 2]):+.4f}"
                    f"  yaw_rate(finite diff)={fd:+.4f} rad/s"
                    f"  heading_control={term.cfg.heading_control}"
                    f"  position_control={getattr(term.cfg, 'position_control', False)}",
                    flush=True,
                )
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        count += 1

    if not headings:
        print("[play] no steps were run")
        return 0

    if task == "recovery":
        print(
            f"[play] recovery: base_z {heights[0]:.3f} -> {heights[-1]:.3f} m, "
            f"|pitch| {math.degrees(abs(pitches[0])):.1f} -> "
            f"{math.degrees(abs(pitches[-1])):.1f} deg, "
            f"|roll| {math.degrees(abs(rolls[0])):.1f} -> "
            f"{math.degrees(abs(rolls[-1])):.1f} deg, termin=0 is expected"
        )
    if task == "stand":
        print(
            f"[play] stand: base_z {heights[0]:.3f} -> {heights[-1]:.3f} m, "
            f"|pitch| {math.degrees(abs(pitches[0])):.1f} -> "
            f"{math.degrees(abs(pitches[-1])):.1f} deg, "
            f"|roll| {math.degrees(abs(rolls[0])):.1f} -> "
            f"{math.degrees(abs(rolls[-1])):.1f} deg"
        )
    if task == "squat":
        hip_lock_err = (
            robot.data.joint_pos[0, _joints.hip_ids] - _joints.locked_hips
        ).abs().max().item()
        print(
            f"[play] squat: base_z {heights[0]:.3f} -> {heights[-1]:.3f} m "
            f"(target {SQUAT_BASE_HEIGHT:.3f}), "
            f"|pitch| {math.degrees(abs(pitches[0])):.1f} -> "
            f"{math.degrees(abs(pitches[-1])):.1f} deg, "
            f"|roll| {math.degrees(abs(rolls[0])):.1f} -> "
            f"{math.degrees(abs(rolls[-1])):.1f} deg, "
            f"max |hip - locked|={math.degrees(hip_lock_err):.2f} deg"
        )

    # wrap the accumulated heading to (-180, 180] before reporting the spread so a
    # robot that stays put does not look like it turned 2*pi
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    # unwrap the heading trace: yaw is reported in (-pi, pi], so a robot that rotates
    # continuously wraps around and a raw `headings[-1] - headings[0]` would under-report
    # a multi-turn in-place rotation.  The straight-line "deviation" below deliberately
    # stays on the wrapped trace (it is meant for small heading deviations).
    unwrapped = []
    _acc = headings[0]
    for i, h in enumerate(headings):
        if i:
            _acc += _wrap(h - headings[i - 1])
        unwrapped.append(_acc)
    total_turn_rad = unwrapped[-1] - unwrapped[0]
    total_turn = math.degrees(total_turn_rad)
    mean_yaw_rate_signed = total_turn_rad / (count * dt) if count > 0 else 0.0
    max_dev = math.degrees(max(abs(_wrap(h - headings[0])) for h in headings))
    n = max(1, len(rolls))
    mean_roll = math.degrees(sum(abs(v) for v in rolls) / n)
    max_roll = math.degrees(max(abs(v) for v in rolls))
    mean_pitch = math.degrees(sum(abs(v) for v in pitches) / n)
    max_pitch = math.degrees(max(abs(v) for v in pitches))

    # leg-activity summary: tells a wheel-only balance apart from a leg-based one
    nl = max(1, len(hip_devs))
    mean_hip_dev = math.degrees(sum(hip_devs) / nl)
    max_hip_dev = math.degrees(max(hip_devs))
    mean_phi = math.degrees(sum(phis) / nl)
    max_phi = math.degrees(max(phis))
    mean_l0_span = 1000.0 * sum(l0_spans) / nl  # mm
    mean_dphi = sum(dphis) / max(1, len(dphis))
    max_dphi = max(dphis) if dphis else 0.0
    mean_dl0 = sum(dl0s) / max(1, len(dl0s))
    mean_axle = 1000.0 * sum(axle_skew) / max(1, len(axle_skew))
    max_axle = 1000.0 * (max(axle_skew) if axle_skew else 0.0)

    nc = max(1, len(cmd_wzs))
    command_line = ""
    if _has_vel_cmd:
        mean_vx_body = sum(abs(v) for v in vxs_b) / max(1, len(vxs_b))
        mean_yaw_abs = sum(abs(v) for v in yaw_rates) / max(1, len(yaw_rates))
        if mean_yaw_abs > 1e-4:
            radius_txt = f"  radius~{mean_vx_body / mean_yaw_abs:.2f} m"
        else:
            radius_txt = "  radius=inf"
        command_line = (
            f"[play] command: mean cmd vx={sum(cmd_vxs) / nc:+.3f} m/s  "
            f"wz={sum(cmd_wzs) / nc:+.3f} rad/s  "
            f"mean |vx body|={mean_vx_body:.3f} m/s  "
            f"wz |err|={sum(wz_errs) / nc:.3f} rad/s{radius_txt}\n"
        )

    print(
        f"[play] steps={count}  base_z mean={sum(heights) / len(heights):.4f} m  "
        f"max closure residual={max(residuals):.3e} m  terminations={n_dones}\n"
        f"{command_line}"
        f"[play] heading: net turn={total_turn:+.2f} deg  max |deviation|={max_dev:.2f} deg  "
        f"mean yaw rate={mean_yaw_rate_signed:+.3f} rad/s "
        f"(|yaw rate| {sum(abs(v) for v in yaw_rates) / max(1, len(yaw_rates)):.3f} rad/s)\n"
        f"[play] attitude: mean |pitch|={mean_pitch:.2f} deg (max {max_pitch:.2f})  "
        f"mean |roll|={mean_roll:.2f} deg (max {max_roll:.2f})\n"
        f"[play] legs: mean |hip dev|={mean_hip_dev:.2f} deg (max {max_hip_dev:.2f})  "
        f"mean |phi|={mean_phi:.2f} deg (max {max_phi:.2f})  "
        f"mean |dphi|={mean_dphi:.2f} rad/s (max {max_dphi:.2f})\n"
        f"[play] stance: mean |dL0|={mean_dl0:.3f} m/s  "
        f"mean |L0_left - L0_right|={mean_l0_span:.2f} mm\n"
        f"[play] axle: mean |wheel fore-aft skew|={mean_axle:.1f} mm (max {max_axle:.1f})\n"
        f"[play] position: net displacement=({xs[-1] - xs[0]:+.3f}, {ys[-1] - ys[0]:+.3f}) m  "
        f"max |x|={max(abs(v - xs[0]) for v in xs):.3f} m  "
        f"mean |vx world|={sum(abs(v) for v in vxs) / max(1, len(vxs)):.3f} m/s"
    )

    if _brake_term is not None:
        stop_label = "teleop stop" if task == "teleop" else "braking"
        final_vx = abs(float(robot.data.root_lin_vel_b[0, 0]))
        final_wz = abs(float(robot.data.root_ang_vel_b[0, 2]))
        if brake_onset_speed is None:
            print(f"[play] {stop_label}: stop phase never started (increase --steps)")
        elif brake_stop_distance is not None:
            print(
                f"[play] {stop_label}: stop at {brake_onset_speed:.2f} m/s -> stopped in "
                f"{brake_stop_time:.2f} s / {brake_stop_distance:.2f} m; "
                f"final |vx|={final_vx:.3f} m/s, |wz|={final_wz:.3f} rad/s"
            )
        else:
            print(
                f"[play] {stop_label}: stop at {brake_onset_speed:.2f} m/s -> NOT STOPPED; "
                f"distance so far {brake_last_distance:.2f} m; "
                f"final |vx|={final_vx:.3f} m/s, |wz|={final_wz:.3f} rad/s"
            )

    if args_cli.snapshot:
        # A single PNG instead of a video: RecordVideo finalises its mp4 on close(),
        # and this script exits with os._exit to avoid Kit's Windows shutdown hang.
        # The renderer needs a few frames to warm up after the render product is
        # created -- the first render() calls return an all-zero array.
        from PIL import Image

        out_dir = os.path.join(log_root_path, "videos", "play")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"snapshot_step_{count}.png")
        frame = None
        for attempt in range(60):
            frame = env.unwrapped.render()
            if frame is not None and frame.any():
                Image.fromarray(frame).save(out_path)
                print(f"[play] snapshot saved: {out_path} (after {attempt + 1} render calls)")
                break
            time.sleep(0.05)
        else:
            print("[play] --snapshot: renderer produced no image (rendering disabled?)")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
