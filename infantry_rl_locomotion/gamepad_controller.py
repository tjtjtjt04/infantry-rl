"""Gamepad -> teleop command layer (no MuJoCo / no Isaac dependency).

This module is deliberately hardware- and simulator-agnostic.  It turns a raw
gamepad snapshot into a safe ``(vx, wz)`` command, and exposes the phase/edge
flags a MuJoCo loop needs:

* left stick: forward 2.5 m/s, reverse 1.0 m/s, normal turning 2.0 rad/s;
* speed-dependent yaw cap ``|wz| <= min(2.0, 4.0/max(|vx|, 0.05))``;
* A held: in-place spin at ``spin_max``, direction from the left stick x;
* A + RB held: extend the legs (raise the body); release RB -> crouch;
* B: one-shot recovery request + command lock;
* deadzone, expo, first-order low-pass, acceleration/deceleration slew rate;
* BRAKE on rapid stick release / direction reversal (stop first, then reverse);
* watchdog + reconnect neutral requirement, optional e-stop button.

It does not know about the recovery/stand policies.  The MuJoCo (or Isaac) loop
owns that handoff: when ``out.recovery_requested`` appears it starts
recovery -> align -> stand; while recovering it ignores the layer; after standing
it calls :meth:`GamepadCommandLayer.release_recovery`.

Standalone diagnostics (no gamepad needed):
    python gamepad_controller.py --self-test

Live gamepad demo:
    python gamepad_controller.py
    python gamepad_controller.py --config gamepad_config.json --rate 50

Example JSON:
    {
      "mapping": {"left_x": 0, "left_y": 1, "button_a": 0, "button_b": 1,
                  "button_rb": 5, "button_estop": 7, "invert_left_y": true},
      "command": {"forward_max": 2.5, "reverse_max": 1.0,
                  "yaw_max": 2.0, "spin_max": 8.0,
                  "spin_stance_crouch": -0.55, "spin_stance_extend": 0.30}
    }
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, fields
from enum import Enum


class Mode(str, Enum):
    NORMAL = "normal"
    BRAKE = "brake"
    REVERSE = "reverse"
    SPIN = "spin"
    RECOVERY = "recovery"
    ESTOP = "estop"
    DISCONNECTED = "disconnected"


@dataclass
class GamepadMapping:
    left_x: int = 0
    left_y: int = 1
    button_a: int = 0
    button_b: int = 1
    #: right shoulder (RB).  XInput/pygame layout: 4 = LB, 5 = RB.
    button_rb: int = 5
    button_estop: int | None = None
    #: XInput: up = -1, down = +1; forward is the negated axis
    invert_left_y: bool = True
    #: XInput: right = +1, left = -1.  The robot uses wz > 0 = LEFT (CCW),
    #: so the x axis must be negated for "stick right -> turn right".
    invert_left_x: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "GamepadMapping":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class CommandConfig:
    forward_max: float = 2.5
    reverse_max: float = 1.0
    yaw_max: float = 2.0
    spin_max: float = 4.0
    #: feasibility cap on |vx*wz| (m/s^2); the yaw is reduced at high speed
    lateral_accel_limit: float = 4.0

    deadzone: float = 0.07
    #: looser dead band used only by the post-recovery "centre the stick" re-arm.
    #: Real sticks often rest at 0.08-0.12, which with the 0.07 control deadzone
    #: would keep neutral=1 forever and the stick would appear dead after B.
    rearm_deadzone: float = 0.12
    expo: float = 0.25
    vx_tau: float = 0.10
    wz_tau: float = 0.12

    vx_accel: float = 2.0
    vx_decel: float = 3.0
    vx_brake_decel: float = 6.0
    reverse_accel: float = 2.0
    wz_accel: float = 10.0
    spin_accel: float = 10.0

    #: a direction reversal above this speed goes through BRAKE first
    brake_threshold: float = 0.15
    #: a released stick while faster than this enters BRAKE
    release_speed: float = 0.25
    release_threshold: float = 0.05

    #: scripted symmetric stance bias (hip offset, rad) used while A spins.
    #: A alone -> crouched spin; A+RB -> extended spin; release RB -> crouch;
    #: release A -> nominal stance.  Measured kinematics:
    #: base_z ~= 0.189 + 0.190 d + 0.017 d^2.  d=+0.30 -> ~0.247 m;
    #: d=-0.55 -> ~0.090 m (very deep mechanical crouch); d=-0.60 -> ~0.084 m.
    #: The teleop fall threshold is 0.075 m, so -0.60 has only ~9 mm margin;
    #: -0.55 is the safer deployment default.  Training covers [-0.60, +0.30].
    spin_stance_crouch: float = -0.55
    spin_stance_extend: float = 0.35

    watchdog_timeout: float = 0.15
    reconnect_neutral_time: float = 0.25
    max_dt: float = 0.05
    spin_dir_default: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "CommandConfig":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class GamepadState:
    connected: bool
    left_x: float = 0.0
    left_y: float = 0.0
    a: bool = False
    b: bool = False
    rb: bool = False
    estop: bool = False
    timestamp: float = 0.0


@dataclass
class CommandOutput:
    vx: float
    wz: float
    mode: Mode
    connected: bool
    recovery_requested: bool = False
    spin: bool = False
    brake: bool = False
    estop: bool = False
    target_vx: float = 0.0
    target_wz: float = 0.0
    #: scripted symmetric stance bias in rad (0 = nominal).  Non-zero only in
    #: SPIN: crouch by default, extension while RB is held.
    stance: float = 0.0


def _shape(v: float, deadzone: float, expo: float) -> float:
    """Deadzone + cubic blend response curve."""
    if abs(v) <= deadzone:
        return 0.0
    u = (abs(v) - deadzone) / (1.0 - deadzone)
    magnitude = u * (1.0 - expo) + u**3 * expo
    return math.copysign(magnitude, v)


def _lpf(current: float, target: float, dt: float, tau: float) -> float:
    if tau <= 0.0:
        return target
    alpha = 1.0 - math.exp(-dt / tau)
    return current + alpha * (target - current)


def _move_toward(current: float, target: float, max_delta: float) -> float:
    return current + max(-max_delta, min(max_delta, target - current))


class GamepadReader:
    """pygame/SDL2 reader with disconnect handling."""

    def __init__(self, index: int = 0, mapping: GamepadMapping | None = None):
        try:
            import pygame  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - hardware path
            raise RuntimeError("pygame is not installed; run: pip install pygame") from exc
        self.pygame = pygame
        self.index = int(index)
        self.mapping = mapping or GamepadMapping()
        self.js = None
        pygame.init()
        pygame.joystick.init()
        self._open()

    def _open(self) -> None:
        try:
            if self.pygame.joystick.get_count() > self.index:
                js = self.pygame.joystick.Joystick(self.index)
                js.init()
                self.js = js
            else:
                self.js = None
        except Exception:
            self.js = None

    def _axis(self, idx: int) -> float:
        if self.js is None or idx < 0 or idx >= self.js.get_numaxes():
            return 0.0
        try:
            return float(self.js.get_axis(idx))
        except Exception:
            return 0.0

    def _button(self, idx: int | None) -> bool:
        if self.js is None or idx is None or idx < 0 or idx >= self.js.get_numbuttons():
            return False
        try:
            return bool(self.js.get_button(idx))
        except Exception:
            return False

    def poll(self) -> GamepadState:
        now = time.monotonic()
        try:
            self.pygame.event.pump()
            if self.js is None or not self.js.get_init():
                self.pygame.joystick.quit()
                self.pygame.joystick.init()
                self._open()
            if self.js is None:
                return GamepadState(connected=False, timestamp=now)
            m = self.mapping
            return GamepadState(
                connected=True,
                left_x=self._axis(m.left_x),
                left_y=self._axis(m.left_y),
                a=self._button(m.button_a),
                b=self._button(m.button_b),
                rb=self._button(m.button_rb),
                estop=self._button(m.button_estop),
                timestamp=now,
            )
        except Exception:
            # unplugged device or SDL error: report disconnected, retry next poll
            self.js = None
            return GamepadState(connected=False, timestamp=now)

    def close(self) -> None:
        try:
            self.pygame.joystick.quit()
            self.pygame.quit()
        except Exception:
            pass


class XInputReader:
    """Windows XInput reader that does not touch SDL/pygame.

    Isaac Sim already owns the window/event loop; initialising pygame's SDL video
    and calling ``pygame.event.pump()`` from the same process can make the
    viewport flicker.  XInput talks to the controller driver directly.

    The axes are normalised to the SAME convention as the pygame reader
    (right = +1, up = -1) so ``GamepadMapping.invert_left_*`` keeps its meaning.
    """

    A_BUTTON = 0x1000
    B_BUTTON = 0x2000
    RB_BUTTON = 0x0200

    def __init__(self, index: int = 0):
        import ctypes  # noqa: PLC0415

        try:
            dll = ctypes.windll.xinput1_4
        except OSError:
            dll = ctypes.windll.xinput9_1_0
        self.ctypes = ctypes
        self.dll = dll
        self.index = int(index)

        class _Gamepad(ctypes.Structure):
            _fields_ = [
                ("wButtons", ctypes.c_ushort),
                ("bLeftTrigger", ctypes.c_ubyte),
                ("bRightTrigger", ctypes.c_ubyte),
                ("sThumbLX", ctypes.c_short),
                ("sThumbLY", ctypes.c_short),
                ("sThumbRX", ctypes.c_short),
                ("sThumbRY", ctypes.c_short),
            ]

        class _State(ctypes.Structure):
            _fields_ = [("dwPacketNumber", ctypes.c_ulong), ("Gamepad", _Gamepad)]

        self._state = _State()
        self.js = None  # compatibility with GamepadReader callers

    @property
    def name(self) -> str:
        return f"XInput #{self.index}"

    def poll(self) -> GamepadState:
        now = time.monotonic()
        rc = self.dll.XInputGetState(self.index, self.ctypes.byref(self._state))
        if rc != 0:
            return GamepadState(connected=False, timestamp=now)
        g = self._state.Gamepad
        return GamepadState(
            connected=True,
            left_x=max(-1.0, min(1.0, g.sThumbLX / 32768.0)),
            # XInput: up = +; the pygame/XInput shared convention is up = -.
            left_y=max(-1.0, min(1.0, -g.sThumbLY / 32768.0)),
            a=bool(g.wButtons & self.A_BUTTON),
            b=bool(g.wButtons & self.B_BUTTON),
            rb=bool(g.wButtons & self.RB_BUTTON),
            estop=False,
            timestamp=now,
        )

    def close(self) -> None:
        pass


class GamepadCommandLayer:
    """Stateful mapping from GamepadState to safe (vx, wz)."""

    def __init__(self, config: CommandConfig | None = None, mapping: GamepadMapping | None = None):
        self.cfg = config or CommandConfig()
        self.mapping = mapping or GamepadMapping()

        self.vx = 0.0
        self.wz = 0.0
        self.mode = Mode.NORMAL
        self.connected = False
        self.estop = False
        self.locked = False
        self.recovery_requested = False

        self._vx_lpf = 0.0
        self._wz_lpf = 0.0
        self._reversing = False
        self._last_spin_sign = 1 if self.cfg.spin_dir_default >= 0 else -1
        self._prev_b = False
        self._prev_estop = False
        self._prev_connected = False
        self._need_neutral = False
        self._neutral_time = 0.0

    # ------------------------------------------------------------------ #
    # external handoff hooks
    # ------------------------------------------------------------------ #
    def set_locked(self, locked: bool) -> None:
        """Lock commands while a recovery/stand controller owns the robot."""
        self.locked = bool(locked)
        if self.locked:
            self.vx = 0.0
            self.wz = 0.0
            self._vx_lpf = 0.0
            self._wz_lpf = 0.0
            self._reversing = False

    def release_recovery(self) -> None:
        """Hand control back; require a neutral stick for a moment first."""
        self.locked = False
        self._need_neutral = True
        self._neutral_time = 0.0
        self.vx = 0.0
        self.wz = 0.0
        self._vx_lpf = 0.0
        self._wz_lpf = 0.0

    def clear_estop(self) -> None:
        self.estop = False
        self._need_neutral = True
        self._neutral_time = 0.0
        self.vx = 0.0
        self.wz = 0.0

    def reset_command(self) -> None:
        """Clear dynamic command state (e.g. after an env reset), keep e-stop."""
        self.vx = 0.0
        self.wz = 0.0
        self._vx_lpf = 0.0
        self._wz_lpf = 0.0
        self._reversing = False

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _zero(self, mode: Mode, *, brake: bool = False, recovery: bool = False) -> CommandOutput:
        self.vx = 0.0
        self.wz = 0.0
        self._vx_lpf = 0.0
        self._wz_lpf = 0.0
        self.mode = mode
        return CommandOutput(
            vx=0.0,
            wz=0.0,
            mode=mode,
            connected=self.connected,
            recovery_requested=recovery,
            spin=False,
            brake=brake,
            estop=self.estop,
            target_vx=0.0,
            target_wz=0.0,
        )

    def _vx_rate(self, target: float, braking: bool) -> float:
        cfg = self.cfg
        if braking:
            return cfg.vx_brake_decel
        if self.vx == 0.0 or target * self.vx < 0.0:
            return cfg.reverse_accel if target < 0.0 else cfg.vx_accel
        if abs(target) > abs(self.vx):
            return cfg.vx_accel
        return cfg.vx_decel

    def _yaw_cap(self) -> float:
        cfg = self.cfg
        return min(cfg.yaw_max, cfg.lateral_accel_limit / max(abs(self.vx), 0.05))

    # ------------------------------------------------------------------ #
    # main update
    # ------------------------------------------------------------------ #
    def update(self, state: GamepadState, now: float | None = None, dt: float = 0.02) -> CommandOutput:
        cfg = self.cfg
        dt = max(0.0, min(float(dt), cfg.max_dt))
        if now is None:
            now = state.timestamp
        self.connected = bool(state.connected)
        self.recovery_requested = False

        # button edges are computed once, before any early return
        estop_edge = bool(state.estop) and not self._prev_estop
        b_edge = bool(state.b) and not self._prev_b
        self._prev_estop = bool(state.estop)
        self._prev_b = bool(state.b)

        # e-stop is latched until clear_estop()
        if self.estop or estop_edge:
            self.estop = True
            return self._zero(Mode.ESTOP, brake=True)

        # watchdog / unplugged
        if (not state.connected) or (now - state.timestamp > cfg.watchdog_timeout):
            self._prev_connected = False
            return self._zero(Mode.DISCONNECTED, brake=True)

        # after a reconnect, require a neutral gamepad before accepting motion
        if not self._prev_connected:
            self._prev_connected = True
            self._need_neutral = True
            self._neutral_time = 0.0
        if self._need_neutral:
            neutral = (
                abs(state.left_x) < cfg.rearm_deadzone
                and abs(state.left_y) < cfg.rearm_deadzone
                and not state.a
                and not state.b
                and not state.rb
            )
            self._neutral_time = self._neutral_time + dt if neutral else 0.0
            if self._neutral_time < cfg.reconnect_neutral_time:
                return self._zero(Mode.NORMAL)
            self._need_neutral = False

        # B: one-shot recovery request; lock the command layer immediately.
        if b_edge:
            self.locked = True
            self.vx = 0.0
            self.wz = 0.0
            self.mode = Mode.RECOVERY
            return CommandOutput(
                vx=0.0,
                wz=0.0,
                mode=Mode.RECOVERY,
                connected=self.connected,
                recovery_requested=True,
                brake=False,
                estop=self.estop,
            )
        if self.locked:
            return self._zero(Mode.RECOVERY)

        # A held: in-place spin, direction from the left stick x.
        if state.a and not self.estop:
            sx = _shape(state.left_x, cfg.deadzone, cfg.expo)
            if self.mapping.invert_left_x:
                sx = -sx
            if abs(sx) > 1e-6:
                self._last_spin_sign = 1 if sx > 0.0 else -1
            target_wz = cfg.spin_max * self._last_spin_sign
            self.vx = _move_toward(self.vx, 0.0, cfg.vx_brake_decel * dt)
            self.wz = _move_toward(self.wz, target_wz, cfg.spin_accel * dt)
            self._vx_lpf = 0.0
            self._wz_lpf = 0.0
            self.mode = Mode.SPIN
            # A+RB: extend the legs (raise the body); release RB -> crouch.
            # The value is a symmetric hip offset in rad, applied by the action term.
            stance = cfg.spin_stance_extend if state.rb else cfg.spin_stance_crouch
            return CommandOutput(
                vx=self.vx,
                wz=self.wz,
                mode=Mode.SPIN,
                connected=self.connected,
                spin=True,
                brake=False,
                estop=self.estop,
                target_vx=0.0,
                target_wz=target_wz,
                stance=stance,
            )

        # normal stick mapping
        y = _shape(state.left_y, cfg.deadzone, cfg.expo)
        if self.mapping.invert_left_y:
            y = -y
        x = _shape(state.left_x, cfg.deadzone, cfg.expo)
        if self.mapping.invert_left_x:
            x = -x

        raw_vx = cfg.forward_max * y if y >= 0.0 else cfg.reverse_max * y
        raw_wz = cfg.yaw_max * x

        if self._reversing and abs(self.vx) < cfg.brake_threshold:
            self._reversing = False

        if (raw_vx * self.vx < 0.0) and abs(self.vx) > cfg.brake_threshold:
            self._reversing = True

        if self._reversing:
            target_vx = 0.0
            target_wz = 0.0
            mode = Mode.BRAKE
            braking = True
        elif abs(raw_vx) < cfg.release_threshold and abs(self.vx) > cfg.release_speed:
            # sudden release / stick back to neutral: stop with the brake ramp
            target_vx = 0.0
            target_wz = 0.0
            mode = Mode.BRAKE
            braking = True
        else:
            braking = False
            self._vx_lpf = _lpf(self._vx_lpf, raw_vx, dt, cfg.vx_tau)
            self._wz_lpf = _lpf(self._wz_lpf, raw_wz, dt, cfg.wz_tau)
            target_vx = self._vx_lpf
            target_wz = self._wz_lpf
            mode = Mode.REVERSE if (target_vx < 0.0 and self.vx < 0.0) else Mode.NORMAL

        if braking:
            self._vx_lpf = 0.0
            self._wz_lpf = 0.0

        # longitudinal slew (stop first, then reverse at a gentler rate)
        rate = self._vx_rate(target_vx, braking)
        self.vx = _move_toward(self.vx, target_vx, rate * dt)

        # yaw is capped by the lateral-acceleration envelope at the new speed
        if not braking:
            cap = self._yaw_cap()
            target_wz = max(-cap, min(cap, target_wz))
        self.wz = _move_toward(self.wz, target_wz, cfg.wz_accel * dt)

        self.mode = mode
        return CommandOutput(
            vx=self.vx,
            wz=self.wz,
            mode=mode,
            connected=self.connected,
            spin=False,
            brake=braking,
            estop=self.estop,
            target_vx=target_vx,
            target_wz=target_wz,
        )


# --------------------------------------------------------------------------- #
# self test / demo
# --------------------------------------------------------------------------- #
def _self_test() -> bool:
    layer = GamepadCommandLayer()
    dt = 0.02
    now = 100.0

    def state(**kwargs) -> GamepadState:
        base = dict(connected=True, left_x=0.0, left_y=0.0, a=False, b=False,
                    rb=False, estop=False, timestamp=now)
        base.update(kwargs)
        return GamepadState(**base)

    def run(steps: int, **kwargs):
        nonlocal now
        out = None
        for _ in range(steps):
            now += dt
            out = layer.update(state(timestamp=now, **kwargs), now=now, dt=dt)
        return out

    ok = True

    # reconnect neutral
    out = run(20)
    ok &= out.vx == 0.0 and out.wz == 0.0

    # forward 0.8 stick: expo maps it below full scale; still a healthy speed
    out = run(120, left_y=-0.8)
    ok &= 1.5 < out.vx <= 2.5 + 1e-6
    ok &= out.mode == Mode.NORMAL

    # full forward: 2.5, yaw must respect the lateral-accel cap (1.6 rad/s)
    out = run(150, left_y=-1.0, left_x=1.0)
    ok &= abs(out.vx - 2.5) < 0.06
    ok &= abs(out.wz) <= 1.60 + 0.05

    # rapid reversal from ~2.5: first BRAKE, never a negative step in one tick
    out = run(1, left_y=0.8)
    ok &= out.mode == Mode.BRAKE and out.vx >= 0.0
    saw_negative = False
    for _ in range(300):
        out = run(1, left_y=0.8)
        saw_negative |= out.vx < 0.0
    ok &= saw_negative and out.vx <= -0.2

    # A spin: 4 rad/s, no forward motion (right stick -> clockwise = negative wz)
    out = run(120, a=True, left_x=1.0)
    ok &= out.mode == Mode.SPIN and abs(out.wz) > 3.5 and abs(out.vx) < 0.05
    # A alone -> crouched spin stance; A+RB -> extension; release RB -> crouch
    ok &= abs(out.stance - layer.cfg.spin_stance_crouch) < 1e-9
    out = run(5, a=True, left_x=1.0, rb=True)
    ok &= abs(out.stance - layer.cfg.spin_stance_extend) < 1e-9
    out = run(5, a=True, left_x=1.0)
    ok &= abs(out.stance - layer.cfg.spin_stance_crouch) < 1e-9

    # B: one-shot request and lock
    out = run(1, b=True)
    ok &= out.recovery_requested and out.mode == Mode.RECOVERY and layer.locked
    out = run(5)
    ok &= not out.recovery_requested and out.vx == 0.0
    layer.release_recovery()
    out = run(30)  # neutral re-arm
    ok &= out.mode == Mode.NORMAL

    # watchdog disconnect
    out = layer.update(state(connected=False, timestamp=now), now=now, dt=dt)
    ok &= out.mode == Mode.DISCONNECTED and out.vx == 0.0

    # reconnect neutral
    out = run(20)
    ok &= out.vx == 0.0

    # e-stop latch
    out = run(1, estop=True)
    ok &= out.mode == Mode.ESTOP
    out = run(10)
    ok &= out.mode == Mode.ESTOP and out.vx == 0.0
    layer.clear_estop()
    out = run(30)
    ok &= out.mode == Mode.NORMAL

    print("self-test:", "PASS" if ok else "FAIL")
    return bool(ok)


def _load_config(path: str | None) -> tuple[GamepadMapping, CommandConfig]:
    if not path:
        return GamepadMapping(), CommandConfig()
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return (
        GamepadMapping.from_dict(data.get("mapping", {})),
        CommandConfig.from_dict(data.get("command", {})),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Gamepad command layer")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--self-test", action="store_true", help="run without hardware")
    parser.add_argument("--duration", type=float, default=0.0, help="0 = until Ctrl+C")
    parser.add_argument("--rate", type=float, default=50.0, help="poll/update rate (Hz)")
    parser.add_argument(
        "--b_lock_s",
        type=float,
        default=2.0,
        help="demo only: auto-release the B recovery lock after N s (0 = never; "
             "the real handoff calls release_recovery() itself)",
    )
    args = parser.parse_args()

    if args.self_test:
        return 0 if _self_test() else 1

    mapping, cfg = _load_config(args.config)
    reader = GamepadReader(args.index, mapping)
    layer = GamepadCommandLayer(cfg, mapping)
    dt = 1.0 / max(1.0, args.rate)
    start = time.monotonic()
    last_print = 0.0
    locked_since: float | None = None
    print("gamepad:", reader.js.get_name() if reader.js is not None else "not found")
    print("Ctrl+C to stop.  B requests recovery; the handoff must call release_recovery().")
    try:
        while True:
            now = time.monotonic()
            state = reader.poll()
            out = layer.update(state, now=now, dt=dt)
            if layer.locked and locked_since is None:
                locked_since = now
            elif not layer.locked:
                locked_since = None
            if (
                layer.locked
                and args.b_lock_s > 0.0
                and locked_since is not None
                and now - locked_since >= args.b_lock_s
            ):
                # demo stand-in for the recovery/stand handoff finishing
                layer.release_recovery()
                locked_since = None
                print(
                    "\n[handoff] demo released the B lock; "
                    "return the stick to neutral to re-arm"
                )
            if now - last_print >= 0.1:
                last_print = now
                print(
                    f"\r[{out.mode.value:12s}] vx={out.vx:+.2f} wz={out.wz:+.2f} "
                    f"target=({out.target_vx:+.2f},{out.target_wz:+.2f}) "
                    f"a={state.a} b={state.b} rb={state.rb} stance={out.stance:+.2f} "
                    f"estop={out.estop} "
                    f"rec={out.recovery_requested} locked={layer.locked} "
                    f"conn={out.connected}",
                    end="",
                    flush=True,
                )
            if args.duration > 0.0 and now - start >= args.duration:
                break
            time.sleep(max(0.0, dt - (time.monotonic() - now)))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
