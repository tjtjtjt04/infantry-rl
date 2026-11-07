#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CAD URDF -> 仿真用开链 URDF 生成器（chassis-rl）。

用途
----
把 SolidWorks 导出的 CAD URDF（含 2 个 <loop_joint> 闭链约束、`package://` mesh 路径、
每侧 axis=(0,-1,0) 的原始关节约定）转换成 Isaac Lab / PhysX 可直接加载的 sim URDF：

1. 删除 2 个 <loop_joint>。PhysX 的 articulation 不约束闭环（旧项目已验证），
   仿真里 4 个 knee 由 ``chassis_rl_lab.kinematics.FiveBarSolver`` 解析闭链解算后
   以位置随动的方式跟踪，因此 URDF 里必须只保留开链。
2. 左侧 5 个关节 axis 改为 (0,1,0)，revolute 限位取负并交换
   （new_lower = -old_upper, new_upper = -old_lower）；右侧 5 个关节保持 CAD 原值
   （右髋 origin rpy 已含 ``0 0 -3.1416``，axis (0,-1,0) 在旋转 180° 后恰好等价于
   世界 +y 正转，无需改动）。
3. mesh 路径 ``package://chassis-rl/meshes/X`` -> ``../meshes/X``（相对
   ``urdf/chassis-rl-sim.urdf`` 的 ../meshes，即 assets/chassis-rl/meshes）。
4. 输出文件头写中文注释，说明关节约定与生成命令。

同时把 CAD URDF 原样拷贝到 ``assets/chassis-rl/urdf/chassis-rl.urdf``（只读参考，
保留 loop_joint 与原始 axis/限位），保证资产目录自包含。

约定（与 ``_design/00_design_spec.md`` §2 一致）
------------------------------------------------
所有关节以绕世界 +y 轴右手定则旋转为正（从 +y 看向 -y 逆时针），左右同号；
前 hip 正转 = 前大腿向下摆（腿伸长）；后 hip 负转 = 后大腿向下摆（腿伸长）；
前膝负转 = 伸长；后膝正转 = 伸长；轮子正转 = 车体向 +x（前进）。

用法
----
    python tools/build_sim_urdf.py
    python tools/build_sim_urdf.py --cad <CAD URDF> --out <输出 URDF>
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径（不依赖当前工作目录）：<project_root>/tools/build_sim_urdf.py
#   project_root = D:\Robomaster\RL\chassis_rl_lab
#   pkg_root     = <project_root>/chassis_rl_lab
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = PROJECT_ROOT / "chassis_rl_lab"
ASSET_URDF_DIR = PKG_ROOT / "assets" / "chassis-rl" / "urdf"
# CAD 源 URDF 在项目外：D:\Robomaster\RL\chassis-rl\urdf\chassis-rl.urdf
DEFAULT_CAD_URDF = PROJECT_ROOT.parent / "chassis-rl" / "urdf" / "chassis-rl.urdf"
DEFAULT_CAD_COPY = ASSET_URDF_DIR / "chassis-rl.urdf"
DEFAULT_SIM_URDF = ASSET_URDF_DIR / "chassis-rl-sim.urdf"

#: 左侧需要翻轴/翻限位的 5 个关节（右侧保持 CAD 原值）
LEFT_JOINTS = (
    "left_front_hip",
    "left_front_knee",
    "left_back_hip",
    "left_back_knee",
    "wheel_left",
)
#: 右侧 5 个关节（只做一致性自检，不改动）
RIGHT_JOINTS = (
    "right_front_hip",
    "right_front_knee",
    "right_back_hip",
    "right_back_knee",
    "wheel_right",
)

MESH_PREFIX_OLD = "package://chassis-rl/meshes/"
MESH_PREFIX_NEW = "../meshes/"

HEADER_TEMPLATE = """<!--
  chassis-rl-sim.urdf —— 仿真用开链 URDF（自动生成，请勿手改）

  关节约定（与 _design/00_design_spec.md §2 一致）：
    所有关节以“绕世界 +y 轴右手定则”旋转为正（从 +y 看向 -y 逆时针），左右同号。
    正转 = 前 hip 向下摆 / 后 hip 向下摆 = 腿伸长方向；
    前膝负转 = 伸长，后膝正转 = 伸长；轮子正转 = 车体向 +x（前进）。

  生成来源：
    CAD URDF: chassis_rl_lab/assets/chassis-rl/urdf/chassis-rl.urdf（原样拷贝，只读参考）
    生成命令: python tools/build_sim_urdf.py

  相对 CAD URDF 的改动：
    1) 删除 2 个 <loop_joint>：PhysX 不约束闭环，4 个 knee 由 FiveBarSolver 解析解算做位置随动；
    2) 左侧 5 个关节 axis=(0,1,0)，revolute 限位取负交换
       （new_lower = -old_upper, new_upper = -old_lower）；
       右侧 5 个关节保持 CAD 原值（右髋 origin rpy 已含 0 0 -3.1416，
       axis (0,-1,0) 旋转 180° 后恰好等价于世界 +y 正转）；
    3) mesh 路径 package://chassis-rl/meshes/X -> ../meshes/X。
-->
"""


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _enable_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，强制 UTF-8 以便中文日志可读。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - 重定向/不支持时忽略
            pass


def _find_joint(root: ET.Element, name: str) -> ET.Element:
    """按名字查找 <joint>，找不到/重名则抛错。"""
    hits = [j for j in root.findall("joint") if j.get("name") == name]
    if len(hits) != 1:
        raise RuntimeError(f"URDF 中关节 {name!r} 出现 {len(hits)} 次（期望 1 次）")
    return hits[0]


def _flip_revolute_limits(joint: ET.Element) -> tuple[tuple[float, float], tuple[float, float]]:
    """revolute 关节限位取负交换：new_lower=-old_upper, new_upper=-old_lower。"""
    limit = joint.find("limit")
    if limit is None:
        raise RuntimeError(f"关节 {joint.get('name')!r} 缺少 <limit>")
    old_lower = float(limit.get("lower"))
    old_upper = float(limit.get("upper"))
    new_lower = -old_upper
    new_upper = -old_lower
    # CAD 限位保留 6 位小数，这里同样格式化为 6 位，保证可重复执行且文本稳定。
    limit.set("lower", f"{new_lower:.6f}")
    limit.set("upper", f"{new_upper:.6f}")
    return (old_lower, old_upper), (new_lower, new_upper)


def _replace_mesh_paths(root: ET.Element) -> int:
    """package://chassis-rl/meshes/X -> ../meshes/X，返回替换个数。"""
    n = 0
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith(MESH_PREFIX_OLD):
            mesh.set("filename", MESH_PREFIX_NEW + filename[len(MESH_PREFIX_OLD):])
            n += 1
    return n


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build(cad_urdf: Path, out_urdf: Path, cad_copy: Path | None = None) -> dict:
    """读 CAD URDF，生成 sim URDF；可选把 CAD URDF 原样拷贝到 assets。

    返回摘要 dict（供工具/校验打印）。
    """
    cad_urdf = Path(cad_urdf).resolve()
    out_urdf = Path(out_urdf).resolve()
    if not cad_urdf.is_file():
        raise FileNotFoundError(f"CAD URDF 不存在: {cad_urdf}")

    tree = ET.parse(cad_urdf)
    root = tree.getroot()
    if root.tag != "robot":
        raise RuntimeError(f"URDF 根节点应为 <robot>，实际为 <{root.tag}>")

    # 1) 删除 loop_joint
    loop_joints = root.findall("loop_joint")
    loop_names = [e.get("name") for e in loop_joints]
    for elem in loop_joints:
        root.remove(elem)

    # 2) mesh 路径相对化
    n_mesh = _replace_mesh_paths(root)

    # 3) 左侧 5 关节：axis -> (0,1,0)，revolute 限位取负交换；右侧保持
    flipped: dict[str, dict] = {}
    for name in LEFT_JOINTS:
        joint = _find_joint(root, name)
        axis = joint.find("axis")
        if axis is None:
            raise RuntimeError(f"关节 {name!r} 缺少 <axis>")
        axis.set("xyz", "0 1 0")
        if joint.get("type") == "revolute":
            old, new = _flip_revolute_limits(joint)
            flipped[name] = {"old": old, "new": new}
        else:
            flipped[name] = {"old": None, "new": None}

    # 右侧必须与 CAD 完全一致（仅自检，不做修改）
    for name in RIGHT_JOINTS:
        joint = _find_joint(root, name)
        axis = joint.find("axis")
        if axis is None or axis.get("xyz", "").split() != ["0", "-1", "0"]:
            raise RuntimeError(f"右侧关节 {name!r} 的 axis 不是 CAD 原值 (0,-1,0): "
                               f"{None if axis is None else axis.get('xyz')}")

    # 4) 序列化：XML 声明 + 中文文件头注释 + 手工缩进的 body
    ET.indent(tree, space="  ")
    body = ET.tostring(root, encoding="unicode")
    output = '<?xml version="1.0" encoding="utf-8"?>\n' + HEADER_TEMPLATE + body + "\n"
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_urdf.write_text(output, encoding="utf-8")

    # 5) CAD URDF 原样拷贝（字节级一致）
    copied = False
    if cad_copy is not None:
        cad_copy = Path(cad_copy).resolve()
        if cad_copy != cad_urdf:
            cad_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cad_urdf, cad_copy)
            copied = True

    # 6) 生成后自检：重新解析输出文件
    check_tree = ET.parse(out_urdf)
    check_root = check_tree.getroot()
    if check_root.findall("loop_joint"):
        raise RuntimeError("生成结果仍含 loop_joint")
    joints = check_root.findall("joint")
    if len(joints) != 10:
        raise RuntimeError(f"生成结果 joint 数 {len(joints)}（期望 10）")
    for joint in joints:
        axis = joint.find("axis")
        if axis is None:
            raise RuntimeError(f"生成结果关节 {joint.get('name')!r} 缺少 axis")
    for mesh in check_root.iter("mesh"):
        if not mesh.get("filename", "").startswith(MESH_PREFIX_NEW):
            raise RuntimeError(f"mesh 路径未相对化: {mesh.get('filename')}")

    return {
        "source": str(cad_urdf),
        "output": str(out_urdf),
        "cad_copy": str(cad_copy) if copied else None,
        "n_loop_removed": len(loop_names),
        "loop_names": loop_names,
        "n_mesh_replaced": n_mesh,
        "flipped": flipped,
        "n_joints": len(joints),
    }


def main(argv: list[str] | None = None) -> int:
    _enable_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="由 CAD URDF 生成仿真用开链 URDF（统一 +y 关节约定、删 loop_joint、mesh 路径相对化）"
    )
    parser.add_argument("--cad", type=Path, default=DEFAULT_CAD_URDF,
                        help=f"CAD 源 URDF（默认 {DEFAULT_CAD_URDF}）")
    parser.add_argument("--out", type=Path, default=DEFAULT_SIM_URDF,
                        help=f"输出 sim URDF（默认 {DEFAULT_SIM_URDF}）")
    parser.add_argument("--cad-copy", type=str, default=str(DEFAULT_CAD_COPY),
                        help=f"CAD URDF 原样拷贝目标（默认 {DEFAULT_CAD_COPY}；传空串跳过）")
    args = parser.parse_args(argv)

    cad_copy = Path(args.cad_copy) if args.cad_copy else None
    info = build(args.cad, args.out, cad_copy)

    print("[build_sim_urdf] 完成")
    print(f"  源 CAD URDF : {info['source']}")
    print(f"  输出 sim    : {info['output']}")
    if info["cad_copy"]:
        print(f"  CAD 原样拷贝: {info['cad_copy']}")
    print(f"  删除 loop_joint: {info['n_loop_removed']} 个 {info['loop_names']}")
    print(f"  mesh 路径相对化: {info['n_mesh_replaced']} 处")
    print(f"  关节总数: {info['n_joints']}")
    print("  左侧翻轴/翻限位:")
    for name, item in info["flipped"].items():
        if item["old"] is None:
            print(f"    {name:18s} axis -> (0,1,0)   [continuous，无限位]")
        else:
            print(f"    {name:18s} axis -> (0,1,0)   "
                  f"limit {item['old']} -> {item['new']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
