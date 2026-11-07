"""生成"展示用 USD"：地面 + 灯光 + 机器人（自动抬高到轮子刚好着地）。

背景
----
``chassis-rl-sim.usd`` 的根节点在自身原点，直接打开时轮子会陷进 z=0 的地面；
本脚本生成一个组合场景 ``assets/chassis-rl/display/chassis-rl_display.usda``：
- 引用机器人 USD（相对路径）并把根节点抬高到轮子刚好触地的高度；
- 加一块 20 m × 20 m 的地面（视觉 mesh + 静态碰撞体）和 DomeLight；
- 加 physicsScene，按 Play 也能跑（机器人会被重力拉到地上、关节无驱动，仅用于查看）。

高度说明
--------
USD 里关节默认值 = q = 0 = CAD 装配位（不是训练用的 squat 姿态），
CAD 位下 base_link 原点到轮轴 z = -0.103707 m、轮半径 0.068 m，
所以抬高量 = **0.171707 m** 时轮子刚好接触 z=0 地面。

用法
----
生成::

    D:\\miniconda\\envs\\RM\\python.exe tools\\make_display_usd.py

生成并 headless 校验（需要 GPU/Isaac Sim）::

    D:\\miniconda\\envs\\RM\\python.exe tools\\make_display_usd.py --verify

然后在 Isaac Sim GUI 里 ``File -> Open`` 选择
``chassis_rl_lab/assets/chassis-rl/display/chassis-rl_display.usda`` 即可。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ASSET_DIR = PROJECT_ROOT / "chassis_rl_lab" / "assets" / "chassis-rl"
ROBOT_USD = ASSET_DIR / "urdf" / "chassis-rl-sim" / "chassis-rl-sim.usd"
DISPLAY_DIR = ASSET_DIR / "display"
DISPLAY_USD = DISPLAY_DIR / "chassis-rl_display.usda"

#: 机器人在 CAD 装配位（q=0）下的 base_link 原点离地高度（m）：-axi_z + R = 0.103707 + 0.068
CAD_BASE_HEIGHT: float = 0.1717070043


def _relative_robot_ref() -> str:
    """display 文件到机器人 USD 的相对路径（用 / 分隔，USD 语法）。"""
    rel = Path("../urdf/chassis-rl-sim/" + ROBOT_USD.name)
    return str(rel).replace("\\", "/")


def build_usda() -> str:
    """拼装展示用 .usda 文本（纯文本，不需要启动 Isaac Sim）。"""
    ref = _relative_robot_ref()
    return f"""#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    metersPerUnit = 1
    doc = "chassis-rl 展示场景：机器人抬高 {CAD_BASE_HEIGHT:.6f} m，轮子刚好接触 z=0 地面（CAD 装配位）。"
)

def Xform "World"
{{
    def Xform "Robot" (
        prepend references = @{ref}@
    )
    {{
        double3 xformOp:translate = (0, 0, {CAD_BASE_HEIGHT:.10f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}

    def Mesh "Ground" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, 10, 0)]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )
        color3f[] primvars:displayColor = [(0.55, 0.55, 0.55)] (
            interpolation = "constant"
        )
        uniform token subdivisionScheme = "none"
    }}

    def DomeLight "DomeLight"
    {{
        float inputs:intensity = 1000
        color3f inputs:color = (0.9, 0.9, 0.9)
    }}

    def DistantLight "KeyLight"
    {{
        float inputs:intensity = 3000
        float inputs:angle = 2
        float3 xformOp:rotateXYZ = (0, 0, 0)
        double3 xformOp:translate = (0, 0, 5)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }}

    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}
}}
"""


def write_display_usd() -> Path:
    if not ROBOT_USD.is_file():
        raise FileNotFoundError(
            f"机器人 USD 不存在：{ROBOT_USD}\n先运行 tools/build_asset.py --force 生成资产。"
        )
    DISPLAY_DIR.mkdir(parents=True, exist_ok=True)
    DISPLAY_USD.write_text(build_usda(), encoding="utf-8")
    print(f"[display] 已生成: {DISPLAY_USD}")
    print(f"[display] 相对引用: @{_relative_robot_ref()}@")
    print(f"[display] 机器人抬高: {CAD_BASE_HEIGHT:.6f} m（CAD 装配位轮子刚好触地）")
    print("[display] 用 Isaac Sim GUI: File -> Open 选择该 .usda 即可查看")
    return DISPLAY_USD


def verify() -> int:
    """headless 打开展示 USD，检查机器人在场且包围盒最低点 ≈ 0。"""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="verify display USD")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(["--headless"])
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import omni.usd  # noqa: PLC0415
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    usd_path = str(DISPLAY_USD).replace("\\", "/")
    omni.usd.get_context().open_stage(usd_path)
    stage = omni.usd.get_context().get_stage()
    print(f"[verify] stage={stage.GetRootLayer().identifier}")
    print(f"[verify] defaultPrim={stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None}")
    print(f"[verify] prims={sum(1 for _ in stage.Traverse())}")

    robot = stage.GetPrimAtPath("/World/Robot")
    if not robot or not robot.IsValid():
        print("[verify][FAIL] /World/Robot 不存在")
        return 1
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    world_bound = bbox_cache.ComputeWorldBound(robot).ComputeAlignedRange()
    zmin, zmax = float(world_bound.GetMin()[2]), float(world_bound.GetMax()[2])
    print(f"[verify] robot 包围盒 z: [{zmin:+.5f}, {zmax:+.5f}] m（期望 min ≈ 0 = 轮子触地）")
    ok = abs(zmin) < 0.01
    print("[verify][OK] 轮子刚好贴地（|zmin| < 1 cm）" if ok else f"[verify][FAIL] zmin={zmin:+.5f} 偏离地面")
    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="make display USD")
    ap.add_argument("--verify", action="store_true", help="生成后启动 Isaac Sim headless 校验")
    args, _ = ap.parse_known_args()
    path = write_display_usd()
    if not args.verify:
        print(f"[display] 跳过校验（加 --verify 可 headless 检查轮子是否贴地）: {path}")
        raise SystemExit(0)
    exit_code = verify()
    sys.stdout.flush()
    sys.stderr.flush()
    import os

    os._exit(exit_code)
