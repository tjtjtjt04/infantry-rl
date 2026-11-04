"""Standalone XInput probe (no pygame / no SDL).

Use this to check whether the physical controller exposes a working XInput
interface.  Compare with ``gamepad_probe.py`` (pygame).  If pygame shows the
sticks moving but this tool does not, use the pygame reader in Isaac
(``isaac_teleop.py`` defaults to pygame; do not pass ``--xinput``).

Run:
    python xinput_probe.py
    python xinput_probe.py --index 1
"""

from __future__ import annotations

import argparse
import time

from gamepad_controller import XInputReader


def main() -> int:
    parser = argparse.ArgumentParser(description="XInput controller probe")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    reader = XInputReader(args.index)
    if not reader.poll().connected:
        found = []
        for i in range(4):
            if XInputReader(i).poll().connected:
                found.append(i)
        print(f"XInput index {args.index} not connected; connected indices: {found}")
        print("If the gamepad works in gamepad_probe.py, it is not exposing XInput;")
        print("use the pygame reader (isaac_teleop.py default) instead.")
        return 1

    print(
        f"XInput #{args.index} connected.  Push the left stick, press A/B and RB "
        "(right shoulder).  Ctrl+C to stop."
    )
    print("Expected (pygame convention): up -> ly=-1, down -> ly=+1, right -> lx=+1.")
    try:
        while True:
            s = reader.poll()
            print(
                f"\rconn={int(s.connected)} lx={s.left_x:+.3f} ly={s.left_y:+.3f} "
                f"a={int(s.a)} b={int(s.b)} rb={int(s.rb)}",
                end="",
                flush=True,
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
