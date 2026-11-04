"""Live gamepad axis/button probe.

Use this first to find the real axis/button indices of the Flydigi Vader 4 Pro
(Bluetooth / XInput).  The exact indices differ between XInput, DirectInput and
Flydigi's own modes, so never hard-code them before looking here.

Run:
    python gamepad_probe.py
    python gamepad_probe.py --index 0
    python gamepad_probe.py --list

Expected on a standard XInput device:
    left stick x/y = axes 0/1 (y is usually negative when pushed forward)
    A/B            = buttons 0/1
but verify on your controller.
"""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gamepad axis/button probe")
    parser.add_argument("--index", type=int, default=0, help="joystick index")
    parser.add_argument("--list", action="store_true", help="list devices and exit")
    parser.add_argument("--period", type=float, default=0.05, help="print period (s)")
    return parser.parse_args()


def main() -> int:
    try:
        import pygame
    except ImportError:
        print("pygame is not installed. Run: pip install pygame")
        return 1

    args = parse_args()
    pygame.init()
    pygame.joystick.init()

    n = pygame.joystick.get_count()
    print(f"joysticks: {n}")
    for i in range(n):
        js = pygame.joystick.Joystick(i)
        js.init()
        print(
            f"  [{i}] {js.get_name()}  guid={js.get_guid()}  "
            f"axes={js.get_numaxes()} buttons={js.get_numbuttons()} hats={js.get_numhats()}"
        )
    if args.list or n == 0:
        pygame.quit()
        return 0 if n > 0 else 1

    index = min(args.index, n - 1)
    js = pygame.joystick.Joystick(index)
    js.init()
    print(f"\nprobing [{index}] {js.get_name()}")
    print("Move ONE axis / press ONE button at a time. Ctrl+C to stop.\n")

    try:
        while True:
            pygame.event.pump()
            axes = [round(js.get_axis(i), 3) for i in range(js.get_numaxes())]
            buttons = [i for i in range(js.get_numbuttons()) if js.get_button(i)]
            hats = [js.get_hat(i) for i in range(js.get_numhats())]
            changed_axes = [f"{i}:{v:+.3f}" for i, v in enumerate(axes) if abs(v) > 0.1]
            text = f"axes[{len(axes)}] " + " ".join(f"{i}:{v:+.3f}" for i, v in enumerate(axes))
            if changed_axes:
                text += "   | active " + " ".join(changed_axes)
            text += f"   buttons {buttons}   hats {hats}"
            print("\r" + text + "        ", end="", flush=True)
            time.sleep(args.period)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
