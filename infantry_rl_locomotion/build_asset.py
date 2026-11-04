"""Convert the chassis-rl URDF to USD.

The URDF and meshes ship in ``chassis-rl/``; this script only runs Isaac Lab's
URDF importer with the original conversion settings (``merge_fixed_joints=True``,
zero joint drive, force conversion).  The importer authors the two
``<loop_joint>`` tags as active external ``PhysicsSphericalJoint`` constraints
outside the 10-DOF articulation, which is the layout the task expects.

Run with the Isaac Lab Python:
    python build_asset.py            # convert if needed
    python build_asset.py --force    # rebuild
    python build_asset.py --verify   # structural check (launches Kit)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
URDF_PATH = PROJECT_DIR / "chassis-rl" / "urdf" / "chassis-rl.urdf"
USD_PATH = PROJECT_DIR / "chassis-rl" / "urdf" / "chassis-rl" / "chassis-rl.usd"
DEFAULT_CONVERTER = Path(r"D:\IsaacLab-main\IsaacLab-main\scripts\tools\convert_urdf.py")


def convert(converter: Path) -> None:
    if not converter.is_file():
        raise FileNotFoundError(f"converter not found: {converter}")
    if not URDF_PATH.is_file():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")
    cmd = [
        sys.executable,
        str(converter),
        str(URDF_PATH),
        str(USD_PATH),
        "--merge-joints",
        "--joint-stiffness",
        "0.0",
        "--joint-damping",
        "0.0",
        "--joint-target-type",
        "position",
        "--headless",
    ]
    print("[asset] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument("--verify", action="store_true", help="verify the generated USD (launches Kit)")
    parser.add_argument("--force", action="store_true", help="convert even if the USD already exists")
    args = parser.parse_args()

    if args.verify:
        return subprocess.run([sys.executable, str(PROJECT_DIR / "tools" / "verify_asset.py")]).returncode

    if USD_PATH.is_file() and not args.force:
        print(f"[asset] USD already exists: {USD_PATH} (use --force to rebuild)")
        return 0

    convert(args.converter)
    print(f"[asset] generated {USD_PATH}")
    print("[asset] next: python tools/verify_asset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
