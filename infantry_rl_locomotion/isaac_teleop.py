"""Gamepad teleop in Isaac Sim with recovery -> stand -> teleop handoff.

This is the Isaac counterpart of ``mujoco_teleop.py`` and uses the same
``gamepad_controller.GamepadCommandLayer``.  Unlike the MuJoCo plant, the policy
was trained in Isaac, so turning, in-place rotation and the recovery chain all
work here.

The robot starts in a front/back fallen pose.  The handoff controller recovers,
levels and stands it, then hands control to the gamepad.  Press **B** while
driving to fall back into recovery; a fall during driving also triggers it.

Usage (from the project root, GUI):
    python isaac_teleop.py --real_time --config gamepad_config.json
    python isaac_teleop.py --teleop_only --real_time --config gamepad_config.json
    python isaac_teleop.py --headless --cmd_vx 1.0 --cmd_wz 0.0 --steps 600

``--teleop_only`` respawns standing and drives immediately, but it still loads
the recovery policy: a fall, or B from any pose (upright included), runs
recovery -> align -> stand and then hands the stick back.

Controls:
    left stick  forward 2.5 / reverse 1.0 m/s, normal turn up to 2.0 rad/s
    A (hold)    crouched in-place rotation at ``spin_max`` from gamepad_config.json,
                clamped to --max_wz (default 8.5).  The default checkpoint is
                teleop_gated_final: symmetric ``master_rf`` legs only during
                in-place spin (gate), deep crouch down to d=-0.65.
    A + RB      extend the legs (raise the body); release RB to crouch
    B           recovery / self-right request
    e-stop      only if ``button_estop`` is set in gamepad_config.json
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import math
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# The recovery env reads its reset envelope at import time.
os.environ["INFANTRY_RL_RECOVERY_PITCH_MIN"] = "45"
os.environ["INFANTRY_RL_RECOVERY_PITCH_MAX"] = "70"
os.environ["INFANTRY_RL_RECOVERY_ROLL_MAX"] = "2"
os.environ["INFANTRY_RL_RECOVERY_TUCK_PROB"] = "1.0"

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gamepad teleop in Isaac Sim.")
    parser.add_argument("--steps", type=int, default=0, help="control steps (0 = until the window closes)")
    parser.add_argument("--real_time", "--real-time", dest="real_time", action="store_true",
                        default=True, help="pace the loop to 50 Hz (on by default)")
    parser.add_argument("--fast", action="store_true", help="do not sleep (benchmarking)")
    parser.add_argument("--no_gamepad", action="store_true",
                        help="ignore the gamepad and use --cmd_vx/--cmd_wz")
    parser.add_argument("--cmd_vx", type=float, default=0.0, help="fixed forward command (no gamepad)")
    parser.add_argument("--cmd_wz", type=float, default=0.0, help="fixed yaw command (no gamepad)")
    parser.add_argument("--config", type=str, default="gamepad_config.json")
    parser.add_argument("--index", type=int, default=0, help="XInput/pygame joystick index")
    parser.add_argument("--xinput", action="store_true",
                        help="use native XInput instead of pygame (not all Bluetooth pads expose it)")
    parser.add_argument("--pygame", action="store_true",
                        help="deprecated: pygame is the default reader")
    parser.add_argument("--sim_gamepad", action="store_true",
                        help="debug: synthetic stick (forward after 5 s, turn after 8.5 s)")
    parser.add_argument("--input_test", action="store_true",
                        help="startup diagnostic: print raw gamepad + command-layer output only")
    parser.add_argument("--teleop_only", action="store_true",
                        help="drive immediately from the standing teleop env; recovery is "
                             "still loaded, so B (from any pose) and falls self-right")
    parser.add_argument("--max_vx", type=float, default=2.5, help="safety clamp forward (m/s)")
    parser.add_argument("--max_wz", type=float, default=8.5,
                        help="safety clamp yaw (rad/s).  The default teleop checkpoint "
                             "(teleop_yaw4_b2) was trained to 5.5 rad/s in place; the 8 rad/s "
                             "recipe raises that to 8.5.  A-spin uses gamepad_config spin_max.")
    parser.add_argument("--stance_bias", type=float, default=0.0,
                        help="fixed scripted stance in rad when --no_gamepad is used "
                             "(0 = nominal, +0.30 = extended, -0.55 = deep crouch)")
    parser.add_argument("--teleop_heading_lat_accel", type=float, default=8.0,
                        help="lateral-accel cap of the teleop heading-hold PD (m/s^2); "
                             "the current checkpoint was trained with 8.0")
    parser.add_argument("--hip_symmetry", choices=("off", "master_rf"), default="master_rf",
                        help="master_rf slaves all four hip targets to the right-front leg "
                             "([+t,-t,+t,-t]), making all leg angles identical.  The default "
                             "checkpoint was trained with it; pass 'off' for older checkpoints.")
    parser.add_argument("--hip_symmetry_gate", action="store_true", default=True,
                        help="apply master_rf only during in-place spin; normal driving/turning "
                             "uses independent hips.  The default checkpoint was trained with "
                             "the gate; pass --no_hip_symmetry_gate for older checkpoints.")
    parser.add_argument("--no_hip_symmetry_gate", dest="hip_symmetry_gate",
                        action="store_false",
                        help="disable the in-place symmetry gate (independent hips everywhere)")
    parser.add_argument("--hip_scale", type=float, default=None,
                        help="hip action authority (rad per action unit; default 0.25). "
                             "Must match the trained checkpoint.")
    parser.add_argument("--hip_range", type=float, default=0.7,
                        help="hard hip travel around nominal (rad; default 0.7 for the "
                             "deep-crouch recipe; recovery/stand still use 0.45)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fall_pitch_sign", type=float, default=0.0,
                        help="0 = random fall direction, +1 = forward, -1 = backward")
    parser.add_argument("--recover_at", type=int, default=-1,
                        help="debug: force a recovery request at this control step")
    parser.add_argument("--camera", choices=("follow", "manual", "fixed", "off"), default="follow",
                        help="follow = Isaac render-rate controller (default); manual = the "
                             "previous set_camera_view loop; fixed = one-shot static view")
    parser.add_argument("--camera_rate", type=int, default=5,
                        help="manual camera update every N control steps")
    parser.add_argument("--camera_eye", nargs=3, type=float, default=(0.0, -2.6, 1.2),
                        metavar=("X", "Y", "Z"), help="follow-camera offset from the robot root")
    parser.add_argument("--camera_lookat", nargs=3, type=float, default=(0.0, 0.0, 0.25),
                        metavar=("X", "Y", "Z"), help="follow-camera target offset")
    parser.add_argument(
        "--recovery_checkpoint",
        type=str,
        default=(
            "logs\\rsl_rl\\infantry_rl_recovery\\2026-10-26_00-42-37_recovery_tuck_v5\\"
            "model_399.pt"
        ),
    )
    parser.add_argument(
        "--teleop_checkpoint",
        type=str,
        default=(
            "logs\\rsl_rl\\infantry_rl_teleop\\2026-10-29_15-02-39_teleop_gated_final\\"
            "model_13444.pt"
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
if args_cli.fall_pitch_sign != 0.0:
    os.environ["INFANTRY_RL_RECOVERY_PITCH_SIGN"] = str(args_cli.fall_pitch_sign)
# Hip overrides must reach env_locomotion.ActionsCfg at import time (the env modules
# are imported after Kit starts).
if args_cli.hip_scale is not None:
    os.environ["INFANTRY_RL_HIP_SCALE"] = str(args_cli.hip_scale)
if args_cli.hip_range is not None:
    os.environ["INFANTRY_RL_HIP_RANGE"] = str(args_cli.hip_range)
# The teleop env reads this at import time; HandoffController gets the value
# directly too, because the full script runs both policies in a recovery env.
os.environ["INFANTRY_RL_TELEOP_HEADING_LAT_ACCEL"] = str(args_cli.teleop_heading_lat_accel)
os.environ["INFANTRY_RL_TELEOP_HIP_SYMMETRY"] = str(args_cli.hip_symmetry)
if args_cli.hip_symmetry_gate:
    os.environ["INFANTRY_RL_TELEOP_HIP_SYMMETRY_GATE"] = "1"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Imports that require Kit."""

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402

import mdp  # noqa: E402
import agent as agent_lib  # noqa: E402
from env_recovery import (  # noqa: E402
    FiveBarRecoveryEnv,
    RecoveryEnvCfg_PLAY,
    RecoveryTerminationsCfg,
)
from env_teleop import FiveBarTeleopEnv, TeleopEnvCfg_PLAY  # noqa: E402
from gamepad_controller import (  # noqa: E402
    GamepadCommandLayer,
    GamepadReader,
    GamepadState,
    XInputReader,
    _load_config,
)
from handoff_controller import HandoffController, HandoffPhase  # noqa: E402


def _resolve(path: str) -> str:
    direct = os.path.abspath(path)
    if not os.path.isfile(direct):
        raise FileNotFoundError(f"checkpoint not found: {direct}")
    return direct


def _make_policy(task: str, checkpoint: str, wrapper, device: str):
    agent_cfg = agent_lib.make_agent_cfg(task, play=True)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.device = device
    agent_lib.match_network_to_checkpoint(checkpoint, agent_cfg)
    runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint)
    return runner.get_inference_policy(device=device)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _fallen_state(robot, env, controller: HandoffController) -> tuple[bool, float, float]:
    """Return ``(fallen, upright, height)`` for the B-press log / diagnostics.

    ``upright`` is 1 standing, 0 upside down; ``height`` is the root height above
    the env origin.  B is NOT gated on this: recovery is requested from any pose
    (from standing the policy tucks and then stands back up).
    """
    upright = float((1.0 - robot.data.projected_gravity_b[0, 2].item()) * 0.5)
    height = float(
        robot.data.root_pos_w[0, 2].item() - env.scene.env_origins[0, 2].item()
    )
    fallen = (upright < controller.fall_upright_min) or (height < controller.fall_height_min)
    return fallen, upright, height


class FollowCamera:
    """follow = render-rate ViewportCameraController; manual = the old per-step loop.

    The env already creates a controller when a GUI is present.  Its ``asset_root``
    mode subscribes to the app's post-update event and follows at the render rate.
    The previous implementation called ``sim.set_camera_view`` from the control
    loop; it is kept as ``--camera manual`` so the user can compare the two.
    """

    def __init__(self, env, robot, eye_offset, lookat_offset, mode: str, rate: int):
        self.env = env
        self.robot = robot
        self.eye = np.asarray(eye_offset, dtype=float)
        self.lookat = np.asarray(lookat_offset, dtype=float)
        self.mode = mode
        self.rate = max(1, int(rate))
        self.enabled = False
        self.prim = env.cfg.viewer.cam_prim_path

        if mode == "follow":
            self.controller = getattr(env, "viewport_camera_controller", None)
            if self.controller is None:
                print("[teleop] camera=follow unavailable (no viewport); leaving the camera alone")
                return
            self.controller.default_cam_eye = self.eye
            self.controller.default_cam_lookat = self.lookat
            self.controller.update_view_to_asset_root("robot")
            self.enabled = True
            print("[teleop] camera=follow via ViewportCameraController (render rate)")
        elif mode == "manual":
            self.enabled = True
            print(f"[teleop] camera=manual (every {self.rate} control steps)")

    def update(self) -> None:
        if not self.enabled or self.mode != "manual":
            return  # follow is driven by the controller callback
        root = self.robot.data.root_pos_w[0].detach().cpu().numpy()
        try:
            self.env.sim.set_camera_view(
                eye=tuple(root + self.eye),
                target=tuple(root + self.lookat),
                camera_prim_path=self.prim,
            )
        except Exception:
            self.enabled = False


def run_input_test() -> int:
    """Reader + command layer only (no robot, no handoff, no camera)."""
    mapping, cfg = _load_config(args_cli.config)
    layer = GamepadCommandLayer(cfg, mapping)
    reader = GamepadReader(args_cli.index, mapping)
    layer._prev_connected = True
    layer._need_neutral = False
    layer.set_locked(False)
    name = reader.js.get_name() if reader.js is not None else "not found"
    print(f"[input-test] gamepad: pygame {name}")
    if reader.js is not None:
        print(
            f"[input-test] axes={reader.js.get_numaxes()} "
            f"buttons={reader.js.get_numbuttons()} "
            f"mapping lx=axis{layer.mapping.left_x} ly=axis{layer.mapping.left_y} "
            f"A={layer.mapping.button_a} B={layer.mapping.button_b} "
            f"RB={layer.mapping.button_rb}"
        )
    print("[input-test] NOTE: the viewport is intentionally EMPTY (no robot).")
    print("[input-test] Watch THIS CONSOLE.  Lines are printed whenever the stick/buttons change.")
    print("[input-test] B locks the layer; this mode simulates the handoff release after 1.5 s.")
    print("[input-test] Ctrl+C to stop.")
    release_at = None
    last = None
    last_heartbeat = 0.0
    try:
        while simulation_app.is_running():
            now = time.monotonic()
            state = reader.poll()
            out = layer.update(state, now=now, dt=0.02)
            if out.recovery_requested:
                release_at = now + 1.5
                print("[input-test] B -> simulated recovery (locked 1.5 s)", flush=True)
            if release_at is not None and now >= release_at:
                layer.release_recovery()
                release_at = None
                print("[input-test] simulated handoff complete; centre the stick 0.25 s",
                      flush=True)

            axes = []
            if reader.js is not None:
                try:
                    axes = [float(reader.js.get_axis(i)) for i in range(reader.js.get_numaxes())]
                except Exception:
                    axes = []
            snapshot = (state.connected, state.left_x, state.left_y, state.a, state.b,
                        out.vx, out.wz, out.mode, state.rb, out.stance)
            changed = last is None or (
                abs(state.left_x - last[1]) > 0.03
                or abs(state.left_y - last[2]) > 0.03
                or state.a != last[3]
                or state.b != last[4]
                or state.connected != last[0]
                or out.mode != last[7]
                or state.rb != last[8]
                or abs(out.stance - last[9]) > 0.01
            )
            if changed or now - last_heartbeat > 1.0:
                last = snapshot
                last_heartbeat = now
                axes_text = " ".join(f"{v:+.2f}" for v in axes)
                print(
                    f"[input-test] conn={int(state.connected)} "
                    f"raw lx={state.left_x:+.2f} ly={state.left_y:+.2f} "
                    f"a={int(state.a)} b={int(state.b)} rb={int(state.rb)} "
                    f"| vx={out.vx:+.2f} wz={out.wz:+.2f} stance={out.stance:+.2f} "
                    f"mode={out.mode.value:12s}"
                    f" | axes[{len(axes)}]={axes_text}",
                    flush=True,
                )
            simulation_app.update()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[input-test] stopped")
    finally:
        reader.close()
    return 0


def run_teleop_only() -> int:
    """Standing teleop env + gamepad; B / falls still run recovery -> teleop."""
    device = args_cli.device or "cuda:0"
    cfg = TeleopEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    cfg.seed = args_cli.seed
    cfg.sim.device = device
    cfg.log_dir = os.path.join(os.getcwd(), "logs", "isaac_teleop")
    # The teleop suite ends the episode as soon as the robot falls
    # (bad_orientation / base_too_low / pitch_limit) and respawns it standing, so
    # a fall never leaves a pose for B to recover from.  Use the recovery suite
    # (long timeout only) instead: the robot stays down after a fall, which lets B
    # and the auto-fall detector hand it to recovery.
    cfg.terminations = RecoveryTerminationsCfg()
    # The Euler-angle roll guard false-triggers as soon as a front/back fall passes
    # pitch ~90 deg (Euler roll flips to ~180 deg), which respawned the robot before
    # recovery could act.  Replace it with a gravity-based guard: side falls and
    # poses beyond the 120 deg recovery envelope reset quickly, while a front/back
    # fall is left to the recovery policy.
    cfg.terminations.roll_limit = None
    cfg.terminations.unrecoverable_fall = DoneTerm(
        func=mdp.unrecoverable_fall,
        params={"side_limit": 1.05, "upside_down_gz": 0.5},
    )
    env = FiveBarTeleopEnv(cfg=cfg)
    teleop_path = _resolve(args_cli.teleop_checkpoint)
    recovery_path = _resolve(args_cli.recovery_checkpoint)
    print(f"[teleop-only] teleop checkpoint  : {teleop_path}")
    print(f"[teleop-only] recovery checkpoint: {recovery_path}")
    wrapper = RslRlVecEnvWrapper(env, clip_actions=12.0)
    teleop_policy = _make_policy("teleop", teleop_path, wrapper, device)
    recovery_policy = _make_policy("recovery", recovery_path, wrapper, device)
    robot = env.scene["robot"]
    controller = HandoffController(
        env=env,
        robot=robot,
        action_term=env.action_manager.get_term("joints"),
        command_term=env.command_manager.get_term("base_velocity"),
        recovery_policy=recovery_policy,
        teleop_policy=teleop_policy,
        device=device,
        # A B-triggered recovery from some standing poses settles in a ~30 deg
        # squat (upright ~0.93).  The default 0.95 gate then never opens and the
        # robot waits for the timeout; 0.93 hands that squat to ALIGN's pitch PD.
        upright_enter=0.93,
        heading_hold_max_lateral_accel=float(args_cli.teleop_heading_lat_accel),
        teleop_hip_range=float(args_cli.hip_range),
        teleop_hip_symmetry=str(args_cli.hip_symmetry),
        teleop_hip_symmetry_gate=bool(args_cli.hip_symmetry_gate),
    )
    # The teleop env respawns standing, so the matching skill is TELEOP; the full
    # script starts in RECOVERY because its env respawns fallen.
    controller.reset(HandoffPhase.TELEOP)

    use_fixed = bool(args_cli.no_gamepad)
    layer = reader = None
    if not use_fixed:
        mapping, ccfg = _load_config(args_cli.config)
        layer = GamepadCommandLayer(ccfg, mapping)
        if args_cli.sim_gamepad:
            print("[teleop-only] simulated gamepad (debug)")
        else:
            reader = GamepadReader(args_cli.index, mapping)
            name = reader.js.get_name() if reader.js is not None else "not found"
            print(f"[teleop-only] gamepad: pygame {name}")
        # start armed; centering is not required for this mode
        layer._prev_connected = True
        layer._need_neutral = False
        layer.set_locked(False)

    camera_mode = "off" if args_cli.headless else args_cli.camera
    camera = FollowCamera(
        env, robot, args_cli.camera_eye, args_cli.camera_lookat,
        camera_mode, args_cli.camera_rate,
    )
    camera.update()
    if args_cli.camera == "fixed" and not args_cli.headless:
        env.sim.set_camera_view(
            eye=tuple(args_cli.camera_eye),
            target=tuple(args_cli.camera_lookat),
            camera_prim_path=env.cfg.viewer.cam_prim_path,
        )

    obs = wrapper.get_observations()
    dt = float(env.step_dt)
    step = 0
    state = None
    out = None
    # B from a moving upright pose brakes first, then starts recovery from rest:
    # the recovery actor stalls in a ~30 deg squat if it starts with forward
    # momentum, while it reliably rises from a standstill.
    pending_recovery = False
    armed = False
    pending_since = 0.0
    try:
        while simulation_app.is_running() and (args_cli.steps == 0 or step < args_cli.steps):
            start = time.time()
            now = time.monotonic()
            recovery_requested = False
            if use_fixed:
                vx, wz = args_cli.cmd_vx, args_cli.cmd_wz
            else:
                if args_cli.sim_gamepad:
                    t = step * dt
                    left_y = -0.6 if (5.0 <= t < 6.0 or 12.0 <= t < 15.0) else 0.0
                    left_x = 0.5 if 15.0 <= t < 17.0 else 0.0
                    b_press = 6.0 <= t < 6.02
                    state = GamepadState(
                        connected=True, left_x=left_x, left_y=left_y,
                        b=b_press, timestamp=now,
                    )
                else:
                    state = reader.poll()
                out = layer.update(state, now=now, dt=dt)
                recovery_requested = bool(out.recovery_requested)
                vx = _clamp(out.vx, -args_cli.max_vx, args_cli.max_vx)
                wz = _clamp(out.wz, -args_cli.max_wz, args_cli.max_wz)
                if out.estop:
                    vx = wz = 0.0

            # Handle B / debug recovery BEFORE the lock bookkeeping: the B edge
            # itself sets layer.locked, and if the phase is still TELEOP at that
            # moment the release branch below would mistake it for a finished
            # handoff and print/release on every B press.
            force_recovery = args_cli.recover_at >= 0 and step == args_cli.recover_at
            if force_recovery:
                recovery_requested = True
            # Snapshot before request_recovery() mutates the phase, so the branch
            # below sees the phase the input arrived in.
            phase_at_input = controller.phase
            if (
                recovery_requested
                and phase_at_input == HandoffPhase.TELEOP
                and not pending_recovery
            ):
                # B always requests recovery, upright or fallen.  From a standing
                # pose the recovery policy tucks/squats first, then rises through
                # align -> stand; this was verified with the tuck_v5 checkpoint.
                # Do not silently ignore B just because the robot is upright.
                fallen, upright, height = _fallen_state(robot, env, controller)
                speed = float(torch.norm(robot.data.root_lin_vel_b[0]).item())
                if force_recovery or fallen or speed < 0.20:
                    print(
                        f"[teleop-only] B pressed -> recovery "
                        f"(upright={upright:.2f}, z={height:.2f}, fallen={int(fallen)})",
                        flush=True,
                    )
                    controller.request_recovery()
                else:
                    print(
                        f"[teleop-only] B pressed at {speed:.2f} m/s -> braking "
                        "before recovery",
                        flush=True,
                    )
                    pending_recovery = True
                    pending_since = now

            elif recovery_requested and phase_at_input != HandoffPhase.TELEOP:
                # B while recovery/align/stand owns the robot.  Do NOT abort a
                # handoff that just started (auto-fall recovery and B ask for the
                # same thing); only give up once the current phase has stalled.
                if controller.phase_time > 2.0:
                    print(
                        "[teleop-only] B during a stalled handoff -> abort, reset plant "
                        "to standing; centre the stick for 0.25 s to re-arm",
                        flush=True,
                    )
                    with torch.inference_mode():
                        obs, _ = wrapper.reset()
                    controller.reset(HandoffPhase.TELEOP)
                    if layer is not None:
                        layer.release_recovery()
                else:
                    print(
                        "[teleop-only] B ignored: recovery is already running "
                        f"(phase={controller.phase.value})",
                        flush=True,
                    )

            if pending_recovery:
                # Keep the layer locked (the B edge did that) and feed a zero
                # command so the teleop policy brakes.  Start recovery once the
                # robot is (nearly) at rest; 2 s is the hard cap.
                vx = wz = 0.0
                speed_now = float(torch.norm(robot.data.root_lin_vel_b[0]).item())
                elapsed = now - pending_since
                if (elapsed >= 0.8 and speed_now < 0.25) or elapsed >= 2.0:
                    print(
                        f"[teleop-only] stopped ({speed_now:.2f} m/s); starting recovery",
                        flush=True,
                    )
                    controller.request_recovery()
                    pending_recovery = False

            # While recovery/align/stand owns the robot the stick is locked.  A
            # still-locked layer in TELEOP means the handoff just finished: release
            # once; the layer then asks for a centred stick for 0.25 s.
            if controller.phase == HandoffPhase.TELEOP:
                if layer is not None and layer.locked and not pending_recovery:
                    print(
                        "[teleop-only] handoff complete; centre the stick for 0.25 s to re-arm",
                        flush=True,
                    )
                    layer.release_recovery()
            elif layer is not None:
                layer.set_locked(True)

            if layer is not None:
                ready = (
                    controller.phase == HandoffPhase.TELEOP
                    and not layer.locked
                    and not layer._need_neutral
                )
                if ready and not armed:
                    print("[teleop-only] ARMED -- stick is active", flush=True)
                    armed = True
                elif not ready:
                    armed = False

            controller.set_drive_command(vx, wz)
            with torch.inference_mode():
                actions = controller.act(obs)
                # Scripted leg stance: only while TELEOP owns the robot.  A+RB
                # extends, releasing RB crouches (the gamepad layer computes it);
                # recovery/align/stand and the braking-to-recovery handoff force 0.
                if use_fixed:
                    stance = args_cli.stance_bias
                elif out is not None and not pending_recovery and not recovery_requested:
                    stance = out.stance
                else:
                    stance = 0.0
                if controller.phase != HandoffPhase.TELEOP:
                    stance = 0.0
                env.action_manager.get_term("joints").set_stance_target(stance)
                obs, _, dones, extras = wrapper.step(actions)
            if bool(torch.as_tensor(dones).any()):
                log = extras.get("log", {}) if isinstance(extras, dict) else {}
                terms = {k: v for k, v in (log or {}).items() if "Termination" in k}
                print(f"[teleop-only] episode reset by env {terms or ''}", flush=True)
                with torch.inference_mode():
                    obs, _ = wrapper.reset()
                # The teleop env respawns standing, so go straight back to TELEOP.
                controller.reset(HandoffPhase.TELEOP)
                if layer is not None:
                    layer.release_recovery()
            if controller.timeout_requested:
                # A non-TELEOP phase stalled (typically recovery from a side/odd
                # fall).  Respawning standing is the only safe fallback here; the
                # fallen retry used by the full script belongs to the recovery env.
                controller.timeout_requested = False
                print(
                    "[teleop-only] handoff timeout -> reset plant to standing; "
                    "centre the stick for 0.25 s to re-arm",
                    flush=True,
                )
                with torch.inference_mode():
                    obs, _ = wrapper.reset()
                controller.reset(HandoffPhase.TELEOP)
                if layer is not None:
                    layer.release_recovery()

            if step % 25 == 0:
                controller.log(step)
                vel = robot.data.root_lin_vel_b[0].detach().cpu().numpy()
                ang = robot.data.root_ang_vel_b[0].detach().cpu().numpy()
                pos = robot.data.root_pos_w[0].detach().cpu().numpy()
                raw = ""
                if not use_fixed and state is not None:
                    raw = (
                        f" raw lx={state.left_x:+.2f} ly={state.left_y:+.2f} "
                        f"a={int(state.a)} b={int(state.b)} rb={int(state.rb)}"
                    )
                    if layer is not None and out is not None:
                        raw += (
                            f" | layer {out.mode.value} locked={int(layer.locked)}"
                            f" neutral={int(layer._need_neutral)}"
                        )
                print(
                    f"[teleop-only] cmd=({vx:+.2f},{wz:+.2f}) "
                    f"stance={float(env.action_manager.get_term('joints').stance[0]):+.2f} "
                    f"z={pos[2]:+.2f} "
                    f"phase={controller.phase.value} "
                    f"vx={vel[0]:+.2f} wz={ang[2]:+.2f} "
                    f"x={pos[0]:+.2f} y={pos[1]:+.2f}{raw}",
                    flush=True,
                )
            if step % camera.rate == 0:
                camera.update()
            if args_cli.real_time and not args_cli.fast:
                sleep = dt - (time.time() - start)
                if sleep > 0.0:
                    time.sleep(sleep)
            step += 1
    except KeyboardInterrupt:
        print("\n[teleop-only] stopped")
    finally:
        if reader is not None:
            reader.close()
    return 0


def main() -> int:
    if args_cli.input_test:
        return run_input_test()
    if args_cli.teleop_only:
        return run_teleop_only()
    device = args_cli.device or "cuda:0"
    env_cfg = RecoveryEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = device
    env_cfg.log_dir = os.path.join(os.getcwd(), "logs", "isaac_teleop")
    env = FiveBarRecoveryEnv(cfg=env_cfg)

    recovery_path = _resolve(args_cli.recovery_checkpoint)
    teleop_path = _resolve(args_cli.teleop_checkpoint)
    print(f"[teleop] recovery checkpoint: {recovery_path}")
    print(f"[teleop] teleop checkpoint  : {teleop_path}")

    wrapper = RslRlVecEnvWrapper(env, clip_actions=12.0)
    recovery_policy = _make_policy("recovery", recovery_path, wrapper, device)
    teleop_policy = _make_policy("teleop", teleop_path, wrapper, device)

    robot = env.scene["robot"]
    controller = HandoffController(
        env=env,
        robot=robot,
        action_term=env.action_manager.get_term("joints"),
        command_term=env.command_manager.get_term("base_velocity"),
        recovery_policy=recovery_policy,
        teleop_policy=teleop_policy,
        drive_vx=0.0,
        drive_wz=0.0,
        device=device,
        heading_hold_max_lateral_accel=float(args_cli.teleop_heading_lat_accel),
        teleop_hip_range=float(args_cli.hip_range),
        teleop_hip_symmetry=str(args_cli.hip_symmetry),
        teleop_hip_symmetry_gate=bool(args_cli.hip_symmetry_gate),
    )

    # ---- gamepad ----------------------------------------------------------
    layer = None
    reader = None
    use_fixed = bool(args_cli.no_gamepad)
    if not use_fixed:
        try:
            mapping, cfg = _load_config(args_cli.config)
            layer = GamepadCommandLayer(cfg, mapping)
            if args_cli.sim_gamepad:
                print("[teleop] simulated gamepad (debug)")
            else:
                reader = None
                if args_cli.xinput:
                    try:
                        xi = XInputReader(args_cli.index)
                        if xi.poll().connected:
                            reader = xi
                            print("[teleop] gamepad: XInput (native, no SDL)")
                        else:
                            print("[teleop] XInput not connected; falling back to pygame")
                    except Exception as exc:
                        print(f"[teleop] XInput unavailable ({exc}); falling back to pygame")
                if reader is None:
                    # pygame is the default: it is the reader the mapping was probed
                    # with, and some Bluetooth pads expose a different/empty XInput
                    # interface than the one they use for actual stick input.
                    reader = GamepadReader(args_cli.index, mapping)
                    name = reader.js.get_name() if reader.js is not None else "not found"
                    print(f"[teleop] gamepad: pygame {name}")
                    try:
                        count = reader.pygame.joystick.get_count()
                        print(f"[teleop] joysticks: {count}")
                        for i in range(count):
                            js = reader.pygame.joystick.Joystick(i)
                            js.init()
                            print(f"[teleop]   [{i}] {js.get_name()}")
                    except Exception:
                        pass
                    if reader.js is None:
                        print("[teleop] no gamepad found; use --no_gamepad --cmd_vx ... or plug it in")
        except Exception as exc:  # pygame missing / SDL failure
            print(f"[teleop] gamepad unavailable ({exc}); fixed commands only")
            use_fixed = True
            layer = None
            reader = None

    camera_mode = "off" if args_cli.headless else args_cli.camera
    camera = FollowCamera(
        env, robot, args_cli.camera_eye, args_cli.camera_lookat,
        camera_mode, args_cli.camera_rate,
    )
    camera.update()
    if args_cli.camera == "fixed" and not args_cli.headless:
        # One-shot world-space view (use a wide eye/lookat for driving).
        env.sim.set_camera_view(
            eye=tuple(args_cli.camera_eye),
            target=tuple(args_cli.camera_lookat),
            camera_prim_path=env.cfg.viewer.cam_prim_path,
        )
        print(f"[teleop] camera fixed at eye={tuple(args_cli.camera_eye)} "
              f"lookat={tuple(args_cli.camera_lookat)}", flush=True)

    obs = wrapper.get_observations()
    controller.reset()
    # The layer owns motion only in TELEOP; while recovering every command is zero.
    if layer is not None:
        layer.set_locked(True)

    dt = float(env.step_dt)
    step = 0
    state = None
    out = None
    armed = False
    try:
        while simulation_app.is_running() and (args_cli.steps == 0 or step < args_cli.steps):
            start = time.time()
            now = time.monotonic()

            recovery_requested = False
            if use_fixed:
                vx, wz = args_cli.cmd_vx, args_cli.cmd_wz
            else:
                if args_cli.sim_gamepad:
                    t = step * dt
                    left_y = -0.6 if (5.0 <= t < 6.0 or 12.0 <= t < 15.0) else 0.0
                    left_x = 0.5 if 15.0 <= t < 17.0 else 0.0
                    b_press = 6.0 <= t < 6.02
                    state = GamepadState(
                        connected=True, left_x=left_x, left_y=left_y,
                        b=b_press, timestamp=now,
                    )
                else:
                    state = reader.poll()
                out = layer.update(state, now=now, dt=dt)
                recovery_requested = bool(out.recovery_requested)
                vx = _clamp(out.vx, -args_cli.max_vx, args_cli.max_vx)
                wz = _clamp(out.wz, -args_cli.max_wz, args_cli.max_wz)
                if out.estop:
                    vx = wz = 0.0

            # Handle B / debug recovery BEFORE the lock bookkeeping: the B edge
            # itself sets layer.locked, and if the phase is still TELEOP at that
            # moment the release branch below would mistake it for a finished
            # handoff and print/release on every B press.
            force_recovery = args_cli.recover_at >= 0 and step == args_cli.recover_at
            if force_recovery:
                recovery_requested = True
            if recovery_requested and controller.phase == HandoffPhase.TELEOP:
                # B always requests recovery, upright or fallen.  From a standing
                # pose the recovery policy tucks/squats first, then rises through
                # align -> stand.
                fallen, upright, height = _fallen_state(robot, env, controller)
                print(
                    f"[teleop] B pressed -> recovery "
                    f"(upright={upright:.2f}, z={height:.2f}, fallen={int(fallen)})",
                    flush=True,
                )
                controller.request_recovery()

            if controller.phase == HandoffPhase.TELEOP:
                # NOTE: the handoff transition happens INSIDE controller.act() below,
                # so a prev_phase flag updated at the end of the loop never sees it.
                # `layer.locked` is the reliable edge: release once, then it is False.
                if layer is not None and layer.locked:
                    print("[teleop] handoff complete; centre the stick for 0.25 s to re-arm",
                          flush=True)
                    layer.release_recovery()
            else:
                if layer is not None:
                    layer.set_locked(True)

            if layer is not None:
                ready = (
                    controller.phase == HandoffPhase.TELEOP
                    and not layer.locked
                    and not layer._need_neutral
                )
                if ready and not armed:
                    print("[teleop] ARMED -- stick is active", flush=True)
                    armed = True
                elif not ready:
                    armed = False

            controller.set_drive_command(vx, wz)
            with torch.inference_mode():
                actions = controller.act(obs)
                if use_fixed:
                    stance = args_cli.stance_bias
                elif out is not None and not recovery_requested:
                    stance = out.stance
                else:
                    stance = 0.0
                if (
                    controller.phase != HandoffPhase.TELEOP
                    or (layer is not None and layer.locked)
                ):
                    stance = 0.0
                env.action_manager.get_term("joints").set_stance_target(stance)
                obs, _, dones, _ = wrapper.step(actions)
            if bool(torch.as_tensor(dones).any()):
                # the manager env has already re-applied the fallen reset
                controller.reset()
                if layer is not None:
                    layer.set_locked(True)
            if controller.timeout_requested:
                # Safety net: a non-TELEOP phase stalled (e.g. a recovery from an
                # upright pose).  Reset to the fallen training pose and retry.
                controller.timeout_requested = False
                print("[teleop] handoff timeout -> reset plant", flush=True)
                with torch.inference_mode():
                    obs, _ = wrapper.reset()
                controller.reset()
                if layer is not None:
                    layer.set_locked(True)

            if step % 25 == 0:
                controller.log(step)
                vel = robot.data.root_lin_vel_b[0].detach().cpu().numpy()
                ang = robot.data.root_ang_vel_b[0].detach().cpu().numpy()
                pos = robot.data.root_pos_w[0].detach().cpu().numpy()
                print(
                    f"[teleop] gamepad cmd=({vx:+.2f}, {wz:+.2f}) "
                    f"stance={float(env.action_manager.get_term('joints').stance[0]):+.2f} "
                    f"z={pos[2]:+.2f} "
                    f"phase={controller.phase.value} "
                    f"vx={vel[0]:+.2f} wz={ang[2]:+.2f} "
                    f"x={pos[0]:+.2f} y={pos[1]:+.2f}",
                    flush=True,
                )
                if layer is not None and state is not None and out is not None:
                    print(
                        f"[teleop]   raw lx={state.left_x:+.2f} ly={state.left_y:+.2f} "
                        f"a={int(state.a)} b={int(state.b)} rb={int(state.rb)} "
                        f"conn={int(state.connected)} "
                        f"| layer {out.mode.value} locked={int(layer.locked)} "
                        f"neutral={int(layer._need_neutral)}",
                        flush=True,
                    )
            if step % camera.rate == 0:
                camera.update()

            if args_cli.real_time and not args_cli.fast:
                sleep = dt - (time.time() - start)
                if sleep > 0.0:
                    time.sleep(sleep)
            step += 1
    except KeyboardInterrupt:
        print("\n[teleop] stopped")
    finally:
        if reader is not None:
            reader.close()

    controller.log(step)
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
