"""Build the portable robot asset package used by this project.

The original infantry-rl workspace shipped a 96 MB USD that is not part of this
copy (``*.usd`` / ``*.STL`` are gitignored).  The same URDF ships in this repo, and
the 11 meshes live in ``parallel_leg_rl/assets/infantry-rl/meshes``.

This script performs the two steps that produced the original asset:

1. copy the URDF, its meshes and the measured JSON files into this project;
2. re-run Isaac Lab's URDF importer with the *original* conversion settings
   (``merge_fixed_joints=True``, zero joint drive, force conversion).

The Isaac Sim 5.1 importer authors the two ``<loop_joint>`` tags as **active**
external ``PhysicsSphericalJoint`` constraints
(``physics:excludeFromArticulation=True``).  That is the layout the original
workspace was trained with: PhysX enforces the closed chain while the two
``loop_joints`` stay outside the 10-DOF articulation, and the knee position
servo (``mdp.actions.FiveBarJointAction``) writes targets that already satisfy
the same loop.  Keep the constraints active -- deactivating them makes the legs
noticeably softer than the dynamics the 1 m/s policy was trained on.

Measured fact: the root layer and ``configuration/infantry_rl_physics.usd`` are
md5-identical to ``parallel_leg_rl/assets/infantry-rl``, and the robot/sensor
layers are identical too.  Only the 96 MB mesh layer differs (serialization
offsets, 1 byte of file size), so the validated dynamics are reproduced.

Run with the Isaac Lab Python:
    python build_asset.py            # copy + convert
    python build_asset.py --verify   # check the generated USD (launches Kit)

The converter itself launches Kit, so it is executed as a subprocess.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_WS = Path(r"D:\IsaacLab-main\infantry-rl-main\infantry-rl-main\isaac_ws")
DEFAULT_MESH_SOURCE = Path(r"D:\IsaacLab-main\parallel_leg_rl\assets\infantry-rl\meshes")
DEFAULT_CONVERTER = Path(r"D:\IsaacLab-main\IsaacLab-main\scripts\tools\convert_urdf.py")

URDF_DIR = PROJECT_DIR / "assets" / "infantry-rl" / "urdf"
MESH_DIR = PROJECT_DIR / "assets" / "infantry-rl" / "meshes"
USD_PATH = PROJECT_DIR / "assets" / "infantry-rl" / "infantry_rl.usd"
DATA_DIR = PROJECT_DIR / "robot_data"

#: The new chassis-rl URDF ships inside this project; only the USD conversion runs.
CHASSIS_URDF = PROJECT_DIR / "chassis-rl" / "urdf" / "chassis-rl.urdf"
CHASSIS_USD = PROJECT_DIR / "chassis-rl" / "urdf" / "chassis-rl" / "chassis-rl.usd"
SOURCE_URDF = DEFAULT_SOURCE_WS / "assets" / "infantry-rl" / "urdf" / "infantry-rl-sim.urdf"
SOURCE_CONFIG = DEFAULT_SOURCE_WS / "config"


def copy_inputs(source_ws: Path, mesh_source: Path) -> None:
    """Copy the URDF, meshes and measured JSON files into this project."""
    URDF_DIR.mkdir(parents=True, exist_ok=True)
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    src_urdf = source_ws / "assets" / "infantry-rl" / "urdf" / "infantry-rl-sim.urdf"
    if not src_urdf.is_file():
        raise FileNotFoundError(f"URDF not found: {src_urdf}")
    shutil.copy2(src_urdf, URDF_DIR / src_urdf.name)

    if not mesh_source.is_dir():
        raise FileNotFoundError(f"mesh directory not found: {mesh_source}")
    meshes = sorted(mesh_source.glob("*.STL"))
    if len(meshes) != 11:
        raise RuntimeError(f"expected 11 STL meshes, found {len(meshes)} in {mesh_source}")
    for mesh in meshes:
        shutil.copy2(mesh, MESH_DIR / mesh.name)

    for name in ("nominal_pose.json", "planar_leg.json"):
        shutil.copy2(source_ws / "config" / name, DATA_DIR / name)

    print(f"[asset] URDF   -> {URDF_DIR / src_urdf.name}")
    print(f"[asset] meshes -> {MESH_DIR} ({len(meshes)} files)")
    print(f"[asset] data   -> {DATA_DIR}")


def convert(
    converter: Path,
    extra_args: list[str],
    urdf: Path | None = None,
    usd: Path | None = None,
) -> None:
    """Run Isaac Lab's URDF converter with the original infantry-rl settings."""
    if not converter.is_file():
        raise FileNotFoundError(f"converter not found: {converter}")
    urdf = urdf or (URDF_DIR / "infantry-rl-sim.urdf")
    usd = usd or USD_PATH
    if not urdf.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf}")
    cmd = [
        sys.executable,
        str(converter),
        str(urdf),
        str(usd),
        "--merge-joints",
        "--joint-stiffness",
        "0.0",
        "--joint-damping",
        "0.0",
        "--joint-target-type",
        "position",
        "--headless",
        *extra_args,
    ]
    print("[asset] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("infantry", "chassis"), default="infantry",
                        help="infantry = original infantry-rl asset; chassis = the new "
                             "chassis-rl URDF (its URDF and meshes already live in this project)")
    parser.add_argument("--source-ws", type=Path, default=DEFAULT_SOURCE_WS)
    parser.add_argument("--mesh-source", type=Path, default=DEFAULT_MESH_SOURCE)
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument("--skip-copy", action="store_true", help="only run the converter")
    parser.add_argument("--verify", action="store_true", help="verify the generated USD (launches Kit)")
    parser.add_argument("--force", action="store_true", help="convert even if the USD already exists")
    args, extra = parser.parse_known_args()

    if args.verify:
        return subprocess.run([sys.executable, str(PROJECT_DIR / "tools" / "verify_asset.py")]).returncode

    if args.robot == "chassis":
        if CHASSIS_USD.is_file() and not args.force:
            print(f"[asset] chassis USD already exists: {CHASSIS_USD} (use --force to rebuild)")
            return 0
        convert(args.converter, extra, urdf=CHASSIS_URDF, usd=CHASSIS_USD)
        print(f"[asset] generated {CHASSIS_USD}")
        print("[asset] next: tools/measure_chassis_asset.py, then train.py --check")
        return 0

    if not args.skip_copy:
        copy_inputs(args.source_ws, args.mesh_source)

    if USD_PATH.is_file() and not args.force:
        print(f"[asset] USD already exists: {USD_PATH} (use --force to rebuild)")
        return 0

    convert(args.converter, extra)
    print(f"[asset] generated {USD_PATH}")
    print("[asset] next: python build_asset.py --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
