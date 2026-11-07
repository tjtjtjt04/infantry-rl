"""把 ``chassis-rl-sim.urdf`` 转成 USD（Isaac Lab URDF Converter）。

用法（Windows，conda 环境 RM）::

    D:\\miniconda\\envs\\RM\\python.exe tools\\build_asset.py          # 已有 USD 时跳过
    D:\\miniconda\\envs\\RM\\python.exe tools\\build_asset.py --force  # 强制重建

输出：``chassis_rl_lab/assets/chassis-rl/urdf/chassis-rl-sim/chassis-rl-sim.usd``

说明：仿真里用"开链 URDF"（无 loop_joint），五连杆闭链由 knee 位置目标维持，
详见 ``_design/00_design_spec.md`` §3。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_DIR / "chassis_rl_lab" / "assets" / "chassis-rl"
URDF_PATH = ASSET_DIR / "urdf" / "chassis-rl-sim.urdf"
USD_PATH = ASSET_DIR / "urdf" / "chassis-rl-sim" / "chassis-rl-sim.usd"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert chassis-rl sim URDF to USD.")
    parser.add_argument("--force", action="store_true", help="force re-conversion even if the USD exists")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    # 转换不需要 GUI
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils  # noqa: PLC0415  (import after Kit start)
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: PLC0415

    if not URDF_PATH.is_file():
        raise FileNotFoundError(f"sim URDF 不存在: {URDF_PATH}\n先运行 python tools/build_sim_urdf.py")

    force = bool(args_cli.force) or not USD_PATH.is_file()
    converter_cfg = UrdfConverterCfg(
        asset_path=str(URDF_PATH),
        usd_dir=str(USD_PATH.parent),
        usd_file_name=USD_PATH.name,
        force_usd_conversion=force,
        # 多个 env 共享一份几何：instanceable 更省显存
        make_instanceable=True,
        fix_base=False,
        merge_fixed_joints=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            target_type="position",
        ),
    )
    converter = UrdfConverter(converter_cfg)
    print(f"[build_asset] URDF : {URDF_PATH}")
    print(f"[build_asset] USD  : {converter.usd_path}", flush=True)

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
