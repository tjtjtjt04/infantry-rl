#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""chassis-rl 运动学 + 资产校验（规格 ``_design/00_design_spec.md`` §4.2 的全部断言）。

纯 CPU / numpy / torch 运行，不启动 Isaac Sim、不占用 GPU::

    python tools/verify_kinematics.py          # 退出码 0 = 全部通过
    python tools/verify_kinematics.py -v       # 额外打印表格细节

校验内容
--------
A. sim URDF 结构：10 关节、无 loop_joint、逐关节 world_axis≈(0,1,0)（<1e-3）、
   限位表与规格 §2 一致（左侧对 CAD 取负交换、右侧保持原值）、mesh 路径相对化、
   CAD URDF 原样拷贝（字节一致）、拓扑/origin 未被改动、文件头注释。
B. 平面模型 vs URDF：A/B/大腿/小腿/零位角误差 <1e-9（独立实现重新提取）；
   右侧镜像归一化合理性（CAD 原始提取偏差 <1e-6 并留证）。
C. CAD 位姿态：前后链轮轴点与锚点重合 <1e-9；base_height≈0.1717（±2 mm）；
   与 ``底盘数据.txt`` 的角度/限位吻合（±1e-3 rad）。
D. squat 姿态：hip/knee 在限位内且 margin≥0.03；base_height∈[0.16,0.20]；
   闭链残差 <1e-9；且确实是“最深”（δ-1e-3 会破坏 margin、base height 更低）。
E. 数值一致性：网格扫描 hip 邻域（9×9）+ 远端点，numpy 与 torch 解一致 <1e-6、
   闭链残差 <1e-9；warm start 分支连续；float32/float64 dtype 行为。
F. 轮半径：解析二进制 STL 顶点 max hypot(x,z)，与 JSON 一致并打印实测值。
G. 包接口：HIP/KNEE/WHEEL_ORDER、load_planar_model/load_squat_pose、
   FiveBarSolver(solve/closure_residual/axle_z/base_height/knee_targets_clamped/
   knee_limits/solve_numpy) 的签名与行为。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = PROJECT_ROOT / "chassis_rl_lab"
ASSET_DIR = PKG_ROOT / "assets" / "chassis-rl"
ASSET_URDF_DIR = ASSET_DIR / "urdf"
DATA_DIR = PKG_ROOT / "data"
SIM_URDF = ASSET_URDF_DIR / "chassis-rl-sim.urdf"
CAD_COPY_URDF = ASSET_URDF_DIR / "chassis-rl.urdf"
CAD_SOURCE_URDF = PROJECT_ROOT.parent / "chassis-rl" / "urdf" / "chassis-rl.urdf"
PLANAR_JSON = DATA_DIR / "chassis_rl_planar.json"
SQUAT_JSON = DATA_DIR / "squat_pose.json"


_KINEMATICS_BYPASSED_PACKAGE = False


def _load_kinematics():
    """纯 CPU 方式加载 ``chassis_rl_lab.kinematics``（不触发 Isaac Lab / pxr）。

    ``chassis_rl_lab/__init__.py`` 会 import tasks（进而依赖 Isaac Lab），这是训练入口
    需要的副作用；本校验脚本必须在纯 CPU 下可跑，故失败时绕过父包 __init__ 直接按
    文件路径加载 kinematics 子包。
    """
    global _KINEMATICS_BYPASSED_PACKAGE
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import chassis_rl_lab.kinematics as kinematics  # noqa: F401

        return kinematics
    except Exception:  # noqa: BLE001 - 纯 CPU 缺 pxr/isaaclab 属预期
        for name in [n for n in sys.modules if n == "chassis_rl_lab" or n.startswith("chassis_rl_lab.")]:
            del sys.modules[name]

    import importlib.util
    import types

    _KINEMATICS_BYPASSED_PACKAGE = True
    pkg_dir = PROJECT_ROOT / "chassis_rl_lab"
    kin_dir = pkg_dir / "kinematics"
    parent = types.ModuleType("chassis_rl_lab")
    parent.__path__ = [str(pkg_dir)]
    parent.__package__ = "chassis_rl_lab"
    sys.modules["chassis_rl_lab"] = parent
    spec = importlib.util.spec_from_file_location(
        "chassis_rl_lab.kinematics",
        kin_dir / "__init__.py",
        submodule_search_locations=[str(kin_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["chassis_rl_lab.kinematics"] = module
    spec.loader.exec_module(module)
    return module


kin = _load_kinematics()
HIP_ORDER = kin.HIP_ORDER
KNEE_ORDER = kin.KNEE_ORDER
WHEEL_ORDER = kin.WHEEL_ORDER
LegPlanarModel = kin.LegPlanarModel
FiveBarSolver = kin.FiveBarSolver
load_planar_model = kin.load_planar_model
load_squat_pose = kin.load_squat_pose
angle_of = kin.angle_of
wrap_angle = kin.wrap_angle

# ---------------------------------------------------------------------------
# 规格常量（_design/00_design_spec.md §2 / §1 / 底盘数据.txt）
# ---------------------------------------------------------------------------
#: sim URDF 期望限位（§2；left_front_hip 按 §2 规则 new_lower=-old_upper 计算 = -0.000100）
EXPECTED_LIMITS = {
    "left_front_hip": (-0.000100, 1.488940),
    "left_front_knee": (-1.999047, 0.000100),
    "left_back_hip": (-1.489639, 0.000100),
    "left_back_knee": (-0.000100, 1.999500),
    "right_front_hip": (-0.000100, 1.488940),
    "right_front_knee": (-1.999047, 0.000100),
    "right_back_hip": (-1.489639, 0.000100),
    "right_back_knee": (-0.000100, 1.999500),
}
LEFT_JOINTS = ("left_front_hip", "left_front_knee", "left_back_hip", "left_back_knee", "wheel_left")
RIGHT_JOINTS = ("right_front_hip", "right_front_knee", "right_back_hip", "right_back_knee", "wheel_right")

#: §1 CAD 零位（base 系）参考值
SPEC_HIP_FRONT = (0.071525, 0.030406)
SPEC_HIP_BACK = (-0.078475, 0.030406)
SPEC_HAM_FRONT = (0.14273, 0.025547)
SPEC_HAM_BACK = (-0.14273, 0.025549)
SPEC_CALF_FRONT = (-0.21773, -0.159660)
SPEC_CALF_BACK = (0.21773, -0.159662)
SPEC_AXLE_XZ = (-0.003475, -0.103707)
SPEC_CAD_BASE_HEIGHT = 0.1717  # ±2 mm

#: 底盘数据.txt 中的绝对连杆角（度 -> rad）与执行器数据
CHASSIS_DATA = {
    "front_ham_range_deg": (-10.15, 75.16),
    "back_ham_range_deg": (104.80, 190.15),
    "front_calf_range_deg": (114.53, 143.75),
    "back_calf_range_deg": (36.25, 65.47),
    "hip_knee_velocity_max": 16.755,
    "wheel_velocity_max": 61.4705,
    "wheel_effort_max": 4.92,
    "base_mass": 11.339,
}

#: 规格 §4.2 的网格扫描范围（front x back 各 9 点）
GRID_FRONT = (-0.05, 0.25, 9)
GRID_BACK = (-0.25, 0.05, 9)
#: 远端点：机械行程内（hip 在限位内且 knee 在限位内）
FAR_POINTS_IN_STROKE = (
    (0.30, 0.00),
    (0.00, -0.30),
    (0.50, -0.50),
    (0.60, -0.50),
    (0.40, -0.60),
    (0.25, 0.00),
    (0.00, -0.25),
    (0.10, -0.50),
    (0.50, -0.10),
    (0.05, -0.05),
    (0.50, -0.25),
)
#: 远端点：超出关节行程（只校验数值一致性与闭链残差，并打印说明）
FAR_POINTS_OUT_OF_STROKE = (
    (1.40, -1.40),  # hip 在限位内，但闭链解要求前膝 > 0（越过膝收回端限位）
    (-0.20, 0.20),  # hip 输入本身超限
)


# ---------------------------------------------------------------------------
# 校验框架
# ---------------------------------------------------------------------------
def _enable_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，强制 UTF-8 以便中文日志可读。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - 重定向/不支持时忽略
            pass


class Verifier:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.passed = 0
        self.failed: list[tuple[str, str]] = []

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        ok = bool(condition)
        line = f"  [{'PASS' if ok else 'FAIL'}] {name}"
        if detail:
            line += f"   ({detail})"
        print(line)
        if ok:
            self.passed += 1
        else:
            self.failed.append((name, detail))
        return ok

    def info(self, text: str) -> None:
        print(f"         · {text}")

    def summary(self) -> int:
        total = self.passed + len(self.failed)
        print("\n" + "=" * 78)
        if self.failed:
            print(f"结果：FAIL —— {len(self.failed)}/{total} 项失败")
            for name, detail in self.failed:
                print(f"  ✗ {name}  {detail}")
            return 1
        print(f"结果：ALL PASS ✔  （{self.passed}/{total} 项检查全部通过）")
        return 0


# ---------------------------------------------------------------------------
# 通用小工具：URDF / 旋转 / STL
# ---------------------------------------------------------------------------
def parse_floats(text: str | None, count: int = 3, default: tuple[float, ...] | None = None) -> np.ndarray:
    if not text:
        if default is None:
            return np.zeros(count, dtype=np.float64)
        return np.array(default, dtype=np.float64)
    values = np.array([float(v) for v in text.split()], dtype=np.float64)
    if values.size != count:
        raise ValueError(f"期望 {count} 个浮点数，实际 {text!r}")
    return values


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF 固定轴 rpy：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


class UrdfData:
    """独立的 URDF 解析/零位前向运动学实现（不复用 tools/fit_planar_model.py 的代码）。"""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.root = ET.parse(self.path).getroot()
        self.links: dict[str, ET.Element] = {link.get("name"): link for link in self.root.findall("link")}
        self.joints: dict[str, dict] = {}
        self.child_to_joint: dict[str, str] = {}
        self.loop_joints: list[ET.Element] = list(self.root.findall("loop_joint"))
        for joint in self.root.findall("joint"):
            name = joint.get("name")
            origin = joint.find("origin")
            axis = joint.find("axis")
            limit = joint.find("limit")
            info = {
                "name": name,
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "xyz": parse_floats(origin.get("xyz") if origin is not None else None, 3),
                "rpy": parse_floats(origin.get("rpy") if origin is not None else None, 3),
                "axis": parse_floats(axis.get("xyz") if axis is not None else None, 3),
                "lower": (float(limit.get("lower")) if limit is not None and limit.get("lower") is not None else None),
                "upper": (float(limit.get("upper")) if limit is not None and limit.get("upper") is not None else None),
                "effort": (float(limit.get("effort")) if limit is not None and limit.get("effort") is not None else None),
                "velocity": (float(limit.get("velocity")) if limit is not None and limit.get("velocity") is not None else None),
            }
            self.joints[name] = info
            self.child_to_joint[info["child"]] = name
        self._pose_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def link_pose(self, link: str) -> tuple[np.ndarray, np.ndarray]:
        """link 坐标系在 base_link 系（q=0）下的 (R, p)。"""
        if link in self._pose_cache:
            return self._pose_cache[link]
        chain: list[dict] = []
        cursor = link
        while cursor != "base_link":
            joint_name = self.child_to_joint[cursor]
            joint = self.joints[joint_name]
            chain.append(joint)
            cursor = joint["parent"]
        rot = np.eye(3)
        pos = np.zeros(3)
        for joint in reversed(chain):  # base -> ... -> link
            pos = pos + rot @ joint["xyz"]
            rot = rot @ rpy_matrix(joint["rpy"])
        self._pose_cache[link] = (rot, pos)
        return rot, pos

    def joint_origin_in_base(self, name: str) -> np.ndarray:
        joint = self.joints[name]
        rot, pos = self.link_pose(joint["parent"])
        return pos + rot @ joint["xyz"]

    def joint_world_axis(self, name: str) -> np.ndarray:
        joint = self.joints[name]
        rot_parent, _ = self.link_pose(joint["parent"])
        return rot_parent @ rpy_matrix(joint["rpy"]) @ joint["axis"]

    def loop_anchor_in_base(self, calf_link: str) -> np.ndarray:
        for loop in self.loop_joints:
            for tag in ("link1", "link2"):
                elem = loop.find(tag)
                if elem is not None and elem.get("link") == calf_link:
                    rot, pos = self.link_pose(calf_link)
                    return pos + rot @ parse_floats(elem.get("xyz"), 3)
        raise RuntimeError(f"CAD URDF 中找不到 {calf_link!r} 的 loop 锚点")


def read_stl_wheel(path: str | Path) -> dict:
    """解析二进制 STL，返回全部顶点与 max hypot(x, z)（轮子绕 y 轴）。"""
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path.name} 太小，不是 STL")
    n_tri = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * n_tri
    if len(data) != expected:
        raise ValueError(f"{path.name} 不是二进制 STL（{len(data)} != {expected} 字节）")
    raw = np.frombuffer(data, dtype=np.uint8, count=50 * n_tri, offset=84).reshape(n_tri, 50)
    floats = np.frombuffer(raw[:, :48].tobytes(), dtype="<f4").reshape(n_tri, 12).astype(np.float64)
    verts = floats[:, 3:12].reshape(-1, 3)
    radii = np.hypot(verts[:, 0], verts[:, 2])
    return {
        "path": str(path.resolve()),
        "n_triangles": int(n_tri),
        "radius": float(radii.max()),
        "radius_p99_9": float(np.percentile(radii, 99.9)),
        "y_min": float(verts[:, 1].min()),
        "y_max": float(verts[:, 1].max()),
        "binary": True,
    }


def locate_mesh_dir() -> Path:
    for candidate in (ASSET_DIR / "meshes", PROJECT_ROOT.parent / "chassis-rl" / "meshes"):
        if (candidate / "left_wheel.STL").is_file():
            return candidate
    raise FileNotFoundError(f"找不到 wheel STL：( {ASSET_DIR / 'meshes'} 或 CAD 源目录 )")


# ---------------------------------------------------------------------------
# A. sim URDF 结构
# ---------------------------------------------------------------------------
def check_sim_urdf(v: Verifier, verbose: bool) -> tuple[UrdfData, UrdfData]:
    v.section("A. sim URDF 结构（10 关节 / +y 约定 / 限位表 / mesh 路径）")
    v.check("sim URDF 存在", SIM_URDF.is_file(), str(SIM_URDF))
    sim = UrdfData(SIM_URDF)
    cad = UrdfData(CAD_COPY_URDF if CAD_COPY_URDF.is_file() else CAD_SOURCE_URDF)

    expected_joints = set(HIP_ORDER) | set(KNEE_ORDER) | set(WHEEL_ORDER)
    v.check("关节名集合 = 10 个（4 hip + 4 knee + 2 wheel）",
            set(sim.joints) == expected_joints, f"{len(sim.joints)} 个关节")
    revolute = [n for n, j in sim.joints.items() if j["type"] == "revolute"]
    continuous = [n for n, j in sim.joints.items() if j["type"] == "continuous"]
    v.check("关节类型：8 revolute + 2 continuous(wheel)",
            len(revolute) == 8 and set(continuous) == {"wheel_left", "wheel_right"},
            f"revolute={len(revolute)}, continuous={sorted(continuous)}")
    v.check("无 loop_joint（开链 URDF）", len(sim.loop_joints) == 0, f"{len(sim.loop_joints)} 个")
    v.check("CAD URDF 保留 2 个 loop_joint（参考文件）", len(cad.loop_joints) == 2,
            f"{len(cad.loop_joints)} 个")

    # --- world axis ---
    axis_errors = {name: float(np.linalg.norm(sim.joint_world_axis(name) - np.array([0.0, 1.0, 0.0])))
                   for name in sim.joints}
    worst_name = max(axis_errors, key=axis_errors.get)
    v.check("每个关节 world_axis = R_origin·axis_urdf ≈ (0,1,0)（误差 < 1e-3）",
            max(axis_errors.values()) < 1e-3,
            f"max={axis_errors[worst_name]:.3e} @ {worst_name}")
    if verbose:
        for name in sim.joints:
            ax = sim.joint_world_axis(name)
            print(f"         · {name:18s} world_axis=({ax[0]:+.9f}, {ax[1]:+.9f}, {ax[2]:+.9f})  err={axis_errors[name]:.2e}")

    # --- 限位表 ---
    limit_ok = True
    for name, (lo, hi) in EXPECTED_LIMITS.items():
        joint = sim.joints[name]
        if joint["lower"] is None or joint["upper"] is None:
            limit_ok = False
            continue
        if abs(joint["lower"] - lo) > 1e-9 or abs(joint["upper"] - hi) > 1e-9:
            limit_ok = False
    v.check("限位表与规格 §2 一致（8 个 revolute 关节，容差 1e-9）", limit_ok)
    v.info("§2 表 left_front_hip 新 lower 写作 0.000100；按同节规则 new_lower=-old_upper（CAD old_upper=0.000100）"
           "且与 right_front_hip 对称，实际值取 -0.000100（本校验按规则断言）")
    if verbose:
        print("         · 期望限位 vs sim URDF:")
        for name, (lo, hi) in EXPECTED_LIMITS.items():
            joint = sim.joints[name]
            print(f"           {name:18s} expected=[{lo:+.6f}, {hi:+.6f}]  actual=[{joint['lower']:+.6f}, {joint['upper']:+.6f}]")
        for name in ("wheel_left", "wheel_right"):
            print(f"           {name:18s} continuous（无限位）")

    # --- 左侧翻轴/翻限位规则、右侧保持 CAD ---
    flip_ok = True
    for name in LEFT_JOINTS:
        sim_joint, cad_joint = sim.joints[name], cad.joints[name]
        if sim_joint["lower"] is not None:
            if abs(sim_joint["lower"] - (-cad_joint["upper"])) > 1e-9:
                flip_ok = False
            if abs(sim_joint["upper"] - (-cad_joint["lower"])) > 1e-9:
                flip_ok = False
    v.check("左侧 5 关节限位 = CAD 取负交换（new_lower=-old_upper, new_upper=-old_lower）", flip_ok)
    v.check("左侧 5 关节 axis = (0,1,0)",
            all(np.allclose(sim.joints[n]["axis"], np.array([0.0, 1.0, 0.0]), atol=0.0, rtol=0.0)
                for n in LEFT_JOINTS))

    right_ok = True
    for name in RIGHT_JOINTS:
        sim_joint, cad_joint = sim.joints[name], cad.joints[name]
        if not np.array_equal(sim_joint["axis"], cad_joint["axis"]):
            right_ok = False
        if sim_joint["lower"] is not None:
            if abs(sim_joint["lower"] - cad_joint["lower"]) > 1e-12:
                right_ok = False
            if abs(sim_joint["upper"] - cad_joint["upper"]) > 1e-12:
                right_ok = False
        if abs(sim_joint["effort"] - cad_joint["effort"]) > 1e-12 or \
                abs(sim_joint["velocity"] - cad_joint["velocity"]) > 1e-12:
            right_ok = False
    v.check("右侧 5 关节 axis/限位保持 CAD 原值（right hip origin rpy 已含 0 0 -3.1416）", right_ok)

    # --- 拓扑 / origin 未改动 ---
    topo_ok = set(sim.links) == set(cad.links) and set(sim.joints) == set(cad.joints)
    if topo_ok:
        for name, sim_joint in sim.joints.items():
            cad_joint = cad.joints[name]
            if (sim_joint["type"] != cad_joint["type"]
                    or sim_joint["parent"] != cad_joint["parent"]
                    or sim_joint["child"] != cad_joint["child"]
                    or not np.array_equal(sim_joint["xyz"], cad_joint["xyz"])
                    or not np.array_equal(sim_joint["rpy"], cad_joint["rpy"])):
                topo_ok = False
                break
    v.check("sim 与 CAD 的 link/joint 拓扑 + origin 完全一致（只允许改 axis/限位/mesh/删 loop）", topo_ok)

    # --- mesh 路径 ---
    sim_meshes = [mesh.get("filename") for mesh in sim.root.iter("mesh")]
    cad_meshes = [mesh.get("filename") for mesh in cad.root.iter("mesh")]
    rel_ok = all(name.startswith("../meshes/") for name in sim_meshes)
    names_ok = {Path(n).name for n in sim_meshes} == {Path(n).name for n in cad_meshes}
    v.check("mesh 路径全部改为 ../meshes/X（数量与 CAD 一致）",
            rel_ok and names_ok and len(sim_meshes) == len(cad_meshes),
            f"{len(sim_meshes)} 处（CAD {len(cad_meshes)} 处）")
    v.check("sim URDF 无残留 package:// 路径", not any("package://" in (n or "") for n in sim_meshes))
    # mesh 文件是否已就位（Lead 负责把 meshes/ 拷到 assets；未拷时只提示不失败）
    mesh_dir = ASSET_DIR / "meshes"
    unique_meshes = sorted({Path(name).name for name in sim_meshes})
    if mesh_dir.is_dir():
        missing = [name for name in unique_meshes if not (mesh_dir / name).is_file()]
        v.check("sim URDF 引用的 mesh 文件在 assets/chassis-rl/meshes 下就位",
                not missing, f"{len(unique_meshes)} 个 STL" + (f", 缺 {missing}" if missing else ""))
    else:
        v.info(f"assets/chassis-rl/meshes 尚不存在（Lead 负责拷贝）；sim URDF 引用 {len(unique_meshes)} 个 STL")

    # --- CAD 原样拷贝 ---
    if CAD_COPY_URDF.is_file() and CAD_SOURCE_URDF.is_file():
        digest_copy = hashlib.sha256(CAD_COPY_URDF.read_bytes()).hexdigest()
        digest_source = hashlib.sha256(CAD_SOURCE_URDF.read_bytes()).hexdigest()
        v.check("assets 下 CAD URDF 与源文件字节一致（sha256）", digest_copy == digest_source,
                digest_copy[:16] + "…")
    else:
        v.check("assets 下 CAD URDF 存在", CAD_COPY_URDF.is_file(), str(CAD_COPY_URDF))

    # --- 文件头注释 ---
    text = SIM_URDF.read_text(encoding="utf-8")
    v.check("sim URDF 文件头含中文约定与生成命令注释",
            "python tools/build_sim_urdf.py" in text and "绕世界 +y 轴" in text and "loop_joint" in text)
    return sim, cad


# ---------------------------------------------------------------------------
# B. 平面模型 vs URDF（独立提取）
# ---------------------------------------------------------------------------
def extract_side_expected(cad: UrdfData, side: str) -> dict:
    """用独立实现从 CAD URDF 提取单侧平面几何（base 系 x-z 投影）。"""
    calf_f_link = f"{side}_front_calf"
    calf_b_link = f"{side}_back_calf"
    wheel_link = f"{side}_wheel"
    A_f = cad.joint_origin_in_base(f"{side}_front_hip")
    A_b = cad.joint_origin_in_base(f"{side}_back_hip")
    K_f = cad.link_pose(calf_f_link)[1]
    K_b = cad.link_pose(calf_b_link)[1]
    W = cad.link_pose(wheel_link)[1]
    anchor = cad.loop_anchor_in_base(calf_b_link)

    def proj(vec):
        return np.array([vec[0], vec[2]], dtype=np.float64)

    result = {
        "hip_front": proj(A_f),
        "hip_back": proj(A_b),
        "ham_front": proj(K_f - A_f),
        "ham_back": proj(K_b - A_b),
        "calf_front": proj(W - K_f),
        "calf_back": proj(anchor - K_b),
        "zero_angles": {
            "ham_front": angle_of(float(proj(K_f - A_f)[0]), float(proj(K_f - A_f)[1])),
            "ham_back": angle_of(float(proj(K_b - A_b)[0]), float(proj(K_b - A_b)[1])),
            "calf_front": angle_of(float(proj(W - K_f)[0]), float(proj(W - K_f)[1])),
            "calf_back": angle_of(float(proj(anchor - K_b)[0]), float(proj(anchor - K_b)[1])),
        },
        "raw_axle_3d": W,
        "raw_anchor_3d": anchor,
    }
    result["closure"] = float(np.linalg.norm(W - anchor))
    return result


def check_planar_model(v: Verifier, planar: dict, cad: UrdfData, verbose: bool) -> dict[str, dict]:
    v.section("B. 平面模型 vs URDF（逐项 < 1e-9）")
    left = planar["sides"]["left"]
    right = planar["sides"]["right"]
    left_exp = extract_side_expected(cad, "left")
    right_exp = extract_side_expected(cad, "right")

    def vec_diff(a, b):
        return float(np.linalg.norm(np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)))

    left_errors = {key: vec_diff(left[key], left_exp[key])
                   for key in ("hip_front", "hip_back", "ham_front", "ham_back", "calf_front", "calf_back")}
    left_angle_errors = {key: abs(left["derived"][f"{key}_zero"] - left_exp["zero_angles"][key])
                         for key in ("ham_front", "ham_back", "calf_front", "calf_back")}
    max_left = max(list(left_errors.values()) + list(left_angle_errors.values()))
    v.check("左侧平面参数（A_f/A_b/大腿/小腿/零位角）vs URDF 误差 < 1e-9",
            max_left < 1e-9, f"max={max_left:.3e}")
    if verbose:
        for key, err in left_errors.items():
            print(f"         · {key:10s} model={left[key]}  urdf={left_exp[key].tolist()}  err={err:.3e}")
        for key, err in left_angle_errors.items():
            print(f"         · zero[{key:10s}] err={err:.3e} rad")

    # 左侧派生字段与显式字段自洽
    derived_ok = True
    for key in ("ham_front", "ham_back", "calf_front", "calf_back"):
        expect_len = float(np.linalg.norm(np.array(left[key])))
        expect_zero = angle_of(left[key][0], left[key][1])
        if abs(left["derived"][f"{key}_len"] - expect_len) > 1e-12:
            derived_ok = False
        if abs(left["derived"][f"{key}_zero"] - expect_zero) > 1e-12:
            derived_ok = False
    v.check("JSON derived（长度/零位角）与向量字段自洽 <1e-12", derived_ok)

    # 右侧 = 左侧精确镜像
    mirror_errors = {key: vec_diff(right[key], left[key])
                     for key in ("hip_front", "hip_back", "ham_front", "ham_back", "calf_front", "calf_back")}
    mirror_errors.update({f"zero_{key}": abs(right["derived"][f"{key}_zero"] - left["derived"][f"{key}_zero"])
                          for key in ("ham_front", "ham_back", "calf_front", "calf_back")})
    max_mirror = max(mirror_errors.values())
    v.check("右侧平面参数 = 左侧精确镜像（左右严格同值）< 1e-12",
            max_mirror < 1e-12, f"max={max_mirror:.3e}")

    # 右侧 CAD 原始提取偏差（镜像归一化的合理性证据）
    raw_errors = {key: vec_diff(right[key], right_exp[key])
                  for key in ("hip_front", "hip_back", "ham_front", "ham_back", "calf_front", "calf_back")}
    raw_errors.update({f"zero_{key}": abs(right["derived"][f"{key}_zero"] - right_exp["zero_angles"][key])
                       for key in ("ham_front", "ham_back", "calf_front", "calf_back")})
    max_raw = max(raw_errors.values())
    v.check("右侧 CAD 原始提取 vs 镜像模型 偏差 < 1e-6（CAD rpy=-3.1416 / loop 锚点导出舍入）",
            max_raw < 1e-6, f"max={max_raw:.3e}")
    v.info(f"右侧 CAD 原始闭链残差 = {right_exp['closure']:.3e} m（归一化后模型 = 3e-17）; "
           f"左右原始最大偏差 = {planar['symmetry']['raw_left_right_max_deviation']:.3e}")

    # 轮半径 / 限位 / 关节顺序元数据
    v.check("planar JSON 含 source_urdf 且文件存在",
            Path(planar["source_urdf"]).is_file(), planar["source_urdf"])
    v.check("planar JSON 关节顺序元数据 = HIP/KNEE/WHEEL_ORDER",
            planar["hip_order"] == list(HIP_ORDER) and planar["knee_order"] == list(KNEE_ORDER)
            and planar["wheel_order"] == list(WHEEL_ORDER))
    limits_ok = True
    for name, (lo, hi) in EXPECTED_LIMITS.items():
        got = planar["joint_limits"][name]
        if got[0] is None or abs(got[0] - lo) > 1e-12 or abs(got[1] - hi) > 1e-12:
            limits_ok = False
    for name in ("wheel_left", "wheel_right"):
        if planar["joint_limits"][name] != [None, None]:
            limits_ok = False
    v.check("planar JSON joint_limits（10 关节；wheel 为连续关节 null）与 §2 一致", limits_ok)
    return {"left": left_exp, "right": right_exp}


# ---------------------------------------------------------------------------
# C. CAD 位姿态 + 底盘数据
# ---------------------------------------------------------------------------
def check_cad_pose(v: Verifier, planar: dict, models: dict[str, LegPlanarModel],
                   expected: dict[str, dict], sim: UrdfData) -> None:
    v.section("C. CAD 装配位（q=0）姿态 / base_height / 底盘数据吻合")
    wheel_radius = planar["wheel_radius"]
    axle_points = {}
    for side, model in models.items():
        kf, wf = model.fk_front(0.0, 0.0)
        kb, wb = model.fk_back(0.0, 0.0)
        residual = math.hypot(wf[0] - wb[0], wf[1] - wb[1])
        v.check(f"{side}: CAD 位前后链轮轴点/锚点重合 < 1e-9", residual < 1e-9, f"residual={residual:.3e} m")
        v.check(f"{side}: CAD 位膝点/轮轴点与 URDF 独立提取一致 < 1e-9",
                abs(kf[0] - planar["sides"][side]["zero_pose"]["knee_front"][0]) < 1e-9
                and abs(kf[1] - planar["sides"][side]["zero_pose"]["knee_front"][1]) < 1e-9
                and abs(wf[0] - planar["sides"][side]["zero_pose"]["axle_front"][0]) < 1e-9
                and abs(wf[1] - planar["sides"][side]["zero_pose"]["axle_front"][1]) < 1e-9)
        axle_points[side] = wf
    axle_x = 0.5 * (axle_points["left"][0] + axle_points["right"][0])
    axle_z = 0.5 * (axle_points["left"][1] + axle_points["right"][1])
    v.check("CAD 位轮轴中心 (x,z) ≈ (-0.003475, -0.103707)（±1e-6，规格 §1）",
            abs(axle_x - SPEC_AXLE_XZ[0]) < 1e-6 and abs(axle_z - SPEC_AXLE_XZ[1]) < 1e-6,
            f"({axle_x:+.9f}, {axle_z:+.9f})")
    base_height = -axle_z + wheel_radius
    v.check("CAD 位 base_height = -axle_z + R ≈ 0.1717（±2 mm）",
            abs(base_height - SPEC_CAD_BASE_HEIGHT) < 0.002, f"{base_height:.9f} m")
    v.check("planar JSON cad_pose.base_height 与重算一致 < 1e-12",
            abs(planar["cad_pose"]["base_height"] - base_height) < 1e-12,
            f"{planar['cad_pose']['base_height']:.9f}")
    # URDF 原始两条链在 CAD 位的落点（用于说明右侧 CAD 导出舍入）
    v.info(f"CAD 位原始 URDF 前后链末端距: left={expected['left']['closure']:.3e} m, "
           f"right={expected['right']['closure']:.3e} m（右侧为 rpy=-3.1416 的导出舍入；模型归一化后两侧均 <1e-16）")

    # ---- 与 底盘数据.txt 的绝对连杆角吻合（±1e-3 rad）----
    left = planar["sides"]["left"]
    ham0_f = left["derived"]["ham_front_zero"]
    ham0_b = left["derived"]["ham_back_zero"]
    calf0_f = left["derived"]["calf_front_zero"]
    calf0_b = left["derived"]["calf_back_zero"]
    front_lo, front_hi = (math.radians(v) for v in CHASSIS_DATA["front_ham_range_deg"])
    back_lo, back_hi = (math.radians(v) for v in CHASSIS_DATA["back_ham_range_deg"])
    fc_lo, fc_hi = (math.radians(v) for v in CHASSIS_DATA["front_calf_range_deg"])
    bc_lo, bc_hi = (math.radians(v) for v in CHASSIS_DATA["back_calf_range_deg"])

    angle_checks = {
        "前大腿零位角 = -10.15°": abs(ham0_f - front_lo),
        "后大腿零位角 = 190.15°(=-169.85°)": abs(wrap_angle(ham0_b) - wrap_angle(back_hi)),
        "前小腿零位角 = 143.75°": abs(calf0_f - fc_hi),
        "后小腿零位角 = 36.25°": abs(calf0_b - bc_lo),
    }
    for label, err in angle_checks.items():
        v.check(f"{label}（±1e-3 rad）", err < 1e-3, f"err={err:.3e} rad")

    # 行程端点（由 §2 限位 + 零位角推出）与底盘数据角度范围吻合
    hip_front_hi = planar["joint_limits"]["left_front_hip"][1]
    hip_back_lo = planar["joint_limits"]["left_back_hip"][0]
    knee_front_lo = planar["joint_limits"]["left_front_knee"][0]
    knee_back_hi = planar["joint_limits"]["left_back_knee"][1]
    stroke_checks = {
        "前大腿行程端点 = +75.16°": abs((ham0_f + hip_front_hi) - front_hi),
        "后大腿行程端点 = 104.80°": abs(wrap_angle(ham0_b + hip_back_lo) - wrap_angle(back_lo)),
        "前小腿行程端点 = 114.53°": abs((calf0_f + hip_front_hi + knee_front_lo) - fc_lo),
        "后小腿行程端点 = 65.47°": abs((calf0_b + hip_back_lo + knee_back_hi) - bc_hi),
    }
    for label, err in stroke_checks.items():
        v.check(f"{label}（±1e-3 rad）", err < 1e-3, f"err={err:.3e} rad")

    # ---- 执行器/质量数据（底盘数据.txt）----
    wheel_eff_ok = all(abs(sim.joints[n]["effort"] - CHASSIS_DATA["wheel_effort_max"]) < 1e-12
                       for n in ("wheel_left", "wheel_right"))
    wheel_vel_ok = all(abs(sim.joints[n]["velocity"] - CHASSIS_DATA["wheel_velocity_max"]) < 1e-12
                       for n in ("wheel_left", "wheel_right"))
    hip_knee_eff_ok = all(abs(sim.joints[n]["effort"] - 40.0) < 1e-12
                          for n in HIP_ORDER + KNEE_ORDER)
    hip_knee_vel_ok = all(abs(sim.joints[n]["velocity"] - CHASSIS_DATA["hip_knee_velocity_max"]) < 1e-12
                          for n in HIP_ORDER + KNEE_ORDER)
    v.check("轮 effort/velocity 与底盘数据一致（4.92 / 61.4705）", wheel_eff_ok and wheel_vel_ok)
    v.check("hip/knee effort/velocity 与底盘数据一致（40 / 16.755）", hip_knee_eff_ok and hip_knee_vel_ok)

    base_mass = sim.links["base_link"].find("inertial/mass").get("value")
    v.check("base_link 质量 = 11.339 kg（底盘数据）",
            base_mass is not None and abs(float(base_mass) - CHASSIS_DATA["base_mass"]) < 1e-9,
            f"{base_mass} kg")


# ---------------------------------------------------------------------------
# D. squat 姿态
# ---------------------------------------------------------------------------
def _squat_margin(values: dict[str, float], planar: dict) -> tuple[float, str]:
    worst, worst_name = float("inf"), ""
    for name, value in values.items():
        lower, upper = planar["joint_limits"][name]
        margin = min(value - lower, upper - value)
        if margin < worst:
            worst, worst_name = margin, name
    return worst, worst_name


def check_squat(v: Verifier, planar: dict, models: dict[str, LegPlanarModel],
                solver: FiveBarSolver) -> dict:
    v.section("D. squat 姿态（限位 / margin≥0.03 / base_height / 最深性）")
    squat = load_squat_pose()
    required = ("hips", "knees", "base_height", "wheel_radius", "limits_margin", "termination")
    v.check("squat_pose.json 含 hips/knees/base_height/wheel_radius/limits_margin/termination",
            all(key in squat for key in required))
    v.check("hips/knees 为“关节名 -> 值”dict 且各 4 个关节",
            isinstance(squat["hips"], dict) and isinstance(squat["knees"], dict)
            and set(squat["hips"]) == set(HIP_ORDER) and set(squat["knees"]) == set(KNEE_ORDER))

    all_values = dict(squat["hips"]) | dict(squat["knees"])
    in_limits = all(planar["joint_limits"][n][0] <= value <= planar["joint_limits"][n][1]
                    for n, value in all_values.items())
    v.check("squat 的 8 个 hip/knee 均在关节限位内", in_limits)
    margin, margin_name = _squat_margin(all_values, planar)
    v.check("squat 所有 hip/knee 相对限位 margin ≥ 0.03 rad", margin >= 0.03,
            f"min_margin={margin:.9f} @ {margin_name}")
    v.check("squat base_height ∈ [0.16, 0.20] m",
            0.16 <= squat["base_height"] <= 0.20, f"{squat['base_height']:.9f} m")
    v.check("squat limits_margin = 0.03", abs(squat["limits_margin"] - 0.03) < 1e-12)

    # 对称性：左右同值
    v.check("squat 左右对称（左右同名关节值相同 <1e-12）",
            abs(squat["hips"]["left_front_hip"] - squat["hips"]["right_front_hip"]) < 1e-12
            and abs(squat["hips"]["left_back_hip"] - squat["hips"]["right_back_hip"]) < 1e-12
            and abs(squat["knees"]["left_front_knee"] - squat["knees"]["right_front_knee"]) < 1e-12
            and abs(squat["knees"]["left_back_knee"] - squat["knees"]["right_back_knee"]) < 1e-12)

    # solver 复算
    q_hips = torch.tensor([[squat["hips"][n] for n in HIP_ORDER]], dtype=torch.float64)
    knees, residual = solver.solve(q_hips)
    knees_expected = torch.tensor([[squat["knees"][n] for n in KNEE_ORDER]], dtype=torch.float64)
    knee_err = float((knees - knees_expected).abs().max())
    v.check("solver.solve(squat hips) 复现 squat knees < 1e-6", knee_err < 1e-6, f"max_err={knee_err:.3e} rad")
    v.check("squat 闭链残差 < 1e-9 m", float(residual.max()) < 1e-9, f"{float(residual.max()):.3e} m")
    solver_base = float(solver.base_height(q_hips).item())
    v.check("squat base_height = -axle_z + R（solver 复算一致 <1e-9）",
            abs(solver_base - squat["base_height"]) < 1e-9, f"{solver_base:.9f} vs {squat['base_height']:.9f}")
    v.check("squat wheel_radius = planar wheel_radius",
            abs(squat["wheel_radius"] - planar["wheel_radius"]) < 1e-12)

    # termination 建议值
    term = squat["termination"]
    v.check("termination: base_too_low = h_squat-0.05 / base_too_high = h_squat+0.15 / tilt_rad=0.5",
            abs(term["base_too_low"] - (squat["base_height"] - 0.05)) < 1e-12
            and abs(term["base_too_high"] - (squat["base_height"] + 0.15)) < 1e-12
            and abs(term["tilt_rad"] - 0.5) < 1e-12)

    # 最深性：δ 再小 1e-3 会破坏 margin，且 base height 更低
    delta = squat["hips_delta"]
    prev = {
        "left": (squat["knees"]["left_front_knee"], squat["knees"]["left_back_knee"]),
        "right": (squat["knees"]["right_front_knee"], squat["knees"]["right_back_knee"]),
    }
    def eval_delta(d: float) -> tuple[float, float]:
        heights, values = [], {}
        for side, model in models.items():
            qkf, qkb, _ = model.ik_knees(d, -d, prev=prev[side])
            _, axle = model.fk_front(d, qkf)
            heights.append(-axle[1] + model.wheel_radius)
            values[f"{side}_front_hip"] = d
            values[f"{side}_back_hip"] = -d
            values[f"{side}_front_knee"] = qkf
            values[f"{side}_back_knee"] = qkb
        m, _ = _squat_margin(values, planar)
        return float(np.mean(heights)), m

    deeper_height, deeper_margin = eval_delta(delta - 1e-3)
    v.check("squat 已最深：δ-1e-3 时 min margin < 0.03（再深会违反限位余量）",
            deeper_margin < 0.03, f"margin(δ-1e-3)={deeper_margin:.9f}")
    v.check("squat 已最深：δ-1e-3 的 base_height 更低（则当前为可行最小高度）",
            deeper_height < squat["base_height"] - 1e-9,
            f"{deeper_height:.9f} < {squat['base_height']:.9f}")
    v.info(f"squat: δ={delta:.9f} rad, hips=±{delta:.9f}, knees=({squat['knees']['left_front_knee']:+.9f}, "
           f"{squat['knees']['left_back_knee']:+.9f}), base_height={squat['base_height']:.9f} m, "
           f"min_margin={margin:.9f} rad")
    return squat


# ---------------------------------------------------------------------------
# E. numpy vs torch（网格 + 远端点 + warm start）
# ---------------------------------------------------------------------------
def _numpy_knees(models: dict[str, LegPlanarModel], fronts: np.ndarray, backs: np.ndarray,
                 prev: dict) -> tuple[np.ndarray, np.ndarray]:
    knees = np.zeros((len(fronts), 4), dtype=np.float64)
    residuals = np.zeros(len(fronts), dtype=np.float64)
    for index, (front, back) in enumerate(zip(fronts, backs)):
        qkf_l, qkb_l, res_l = models["left"].ik_knees(float(front), float(back), prev=prev["left"])
        qkf_r, qkb_r, res_r = models["right"].ik_knees(float(front), float(back), prev=prev["right"])
        knees[index] = (qkf_l, qkb_l, qkf_r, qkb_r)
        residuals[index] = max(res_l, res_r)
    return knees, residuals


def check_numpy_vs_torch(v: Verifier, planar: dict, models: dict[str, LegPlanarModel],
                         solver: FiveBarSolver, squat: dict, verbose: bool) -> None:
    v.section("E. numpy ↔ torch 一致性 / 闭链残差 / knee 限位 / warm start")
    prev = {
        "left": (squat["knees"]["left_front_knee"], squat["knees"]["left_back_knee"]),
        "right": (squat["knees"]["right_front_knee"], squat["knees"]["right_back_knee"]),
    }
    knee_lower = np.array([planar["joint_limits"][name][0] for name in KNEE_ORDER])
    knee_upper = np.array([planar["joint_limits"][name][1] for name in KNEE_ORDER])
    front_limits = planar["joint_limits"]["left_front_hip"]
    back_limits = planar["joint_limits"]["left_back_hip"]

    def sample_points(fronts, backs, label, assert_knee_limits):
        q = np.stack([fronts, backs, fronts, backs], axis=-1).astype(np.float64)
        torch_knees, torch_res = solver.solve(torch.tensor(q), prev_knees=None)
        torch_knees = torch_knees.numpy()
        torch_res = torch_res.numpy()
        numpy_knees, numpy_res = _numpy_knees(models, fronts, backs, prev)
        max_diff = float(np.abs(torch_knees - numpy_knees).max())
        v.check(f"{label}：numpy vs torch 解一致 < 1e-6", max_diff < 1e-6, f"max_diff={max_diff:.3e} rad")
        v.check(f"{label}：闭链残差 < 1e-9 m", float(torch_res.max()) < 1e-9 and float(numpy_res.max()) < 1e-9,
                f"torch={float(torch_res.max()):.3e}, numpy={float(numpy_res.max()):.3e}")
        # closure_residual 独立复算
        closure = solver.closure_residual(torch.tensor(q), torch.tensor(torch_knees)).numpy()
        v.check(f"{label}：closure_residual(q_hips, q_knees) < 1e-9 m",
                float(closure.max()) < 1e-9, f"{float(closure.max()):.3e}")
        # 左右对称（模型严格镜像）
        v.check(f"{label}：左右同名 knee 解相同 <1e-12 rad",
                float(np.abs(torch_knees[:, 0] - torch_knees[:, 2]).max()) < 1e-12
                and float(np.abs(torch_knees[:, 1] - torch_knees[:, 3]).max()) < 1e-12)
        # knee 限位分类
        hips_ok = (fronts >= front_limits[0]) & (fronts <= front_limits[1]) \
            & (backs >= back_limits[0]) & (backs <= back_limits[1])
        knees_ok = np.all((torch_knees >= knee_lower - 1e-12) & (torch_knees <= knee_upper + 1e-12), axis=-1)
        in_stroke = hips_ok & knees_ok
        v.info(f"{label}：hip 输入在限位内 {int(hips_ok.sum())}/{len(fronts)} 点；"
               f"机械行程内 {int(in_stroke.sum())} 点；"
               f"hip 超限 {int((~hips_ok).sum())} 点；行程内 knee 全部在限位内 "
               f"{bool(np.all(knees_ok[in_stroke]))}")
        if assert_knee_limits:
            bad = hips_ok & ~knees_ok
            v.check(f"{label}：hip 在限位内的点 knee 均在 knee 限位内",
                    bool(bad.sum() == 0),
                    "" if bad.sum() == 0 else f"超出行程 {int(bad.sum())} 点: "
                    + str([(round(float(f), 3), round(float(b), 3)) for f, b in zip(fronts[bad], backs[bad])]))
        return torch_knees

    # ---- 网格扫描（规格 §4.2）----
    fronts = np.linspace(*GRID_FRONT)
    backs = np.linspace(*GRID_BACK)
    grid_front, grid_back = np.meshgrid(fronts, backs)
    grid_front = grid_front.ravel()
    grid_back = grid_back.ravel()
    grid_knees = sample_points(grid_front, grid_back, "网格扫描 9×9", assert_knee_limits=True)
    if verbose:
        print("         · 网格 knee 值范围：",
              f"front[{grid_knees[:, 0].min():+.4f}, {grid_knees[:, 0].max():+.4f}]",
              f"back[{grid_knees[:, 1].min():+.4f}, {grid_knees[:, 1].max():+.4f}]")

    # ---- 远端点 ----
    far_front = np.array([p[0] for p in FAR_POINTS_IN_STROKE], dtype=np.float64)
    far_back = np.array([p[1] for p in FAR_POINTS_IN_STROKE], dtype=np.float64)
    sample_points(far_front, far_back, "远端点（机械行程内）", assert_knee_limits=True)
    out_front = np.array([p[0] for p in FAR_POINTS_OUT_OF_STROKE], dtype=np.float64)
    out_back = np.array([p[1] for p in FAR_POINTS_OUT_OF_STROKE], dtype=np.float64)
    sample_points(out_front, out_back, "远端点（超出关节行程，仅查一致性与残差）", assert_knee_limits=False)

    # ---- float32 / float64 dtype 行为 ----
    q32 = torch.zeros(4, 4, dtype=torch.float32)
    knees32, res32 = solver.solve(q32, prev_knees=None)
    v.check("solve 保持 float32 dtype / (N,4)+(N,) 形状",
            knees32.dtype == torch.float32 and res32.dtype == torch.float32
            and knees32.shape == (4, 4) and res32.shape == (4,))
    v.check("solve float32 输入时闭链残差 < 1e-6（float32 输出精度内）",
            float(res32.max()) < 1e-6, f"{float(res32.max()):.3e}")
    q64 = torch.zeros(4, 4, dtype=torch.float64)
    knees64, res64 = solver.solve(q64, prev_knees=None)
    v.check("solve float64 输入时闭链残差 < 1e-9", float(res64.max()) < 1e-9, f"{float(res64.max()):.3e}")
    v.check("CAD 位（hips=0）knee 解 ≈ 0（±1e-6 rad）",
            float(knees64.abs().max()) < 1e-6, f"{float(knees64.abs().max()):.3e} rad")
    # 1-D 输入
    knee1, res1 = solver.solve(torch.zeros(4, dtype=torch.float64))
    v.check("solve 支持 (4,) 单样本输入（输出 (4,) 与标量残差）",
            knee1.shape == (4,) and res1.shape == ())

    # ---- knee_targets_clamped ----
    clamped = solver.knee_targets_clamped(torch.tensor(np.stack([grid_front, grid_back, grid_front, grid_back], -1)),
                                          margin=0.02)
    lower = torch.tensor([planar["joint_limits"][n][0] for n in KNEE_ORDER], dtype=torch.float64) + 0.02
    upper = torch.tensor([planar["joint_limits"][n][1] for n in KNEE_ORDER], dtype=torch.float64) - 0.02
    v.check("knee_targets_clamped(margin=0.02) 全部落在 [lower+0.02, upper-0.02]",
            bool(torch.all(clamped >= lower - 1e-12) and torch.all(clamped <= upper + 1e-12)))
    v.check("knee_targets_clamped 形状 (N,4)", tuple(clamped.shape) == (len(grid_front), 4))

    # ---- warm start 分支连续性 ----
    deltas = np.linspace(0.0, 0.6, 121)
    q_seq = torch.tensor(np.stack([deltas, -deltas, deltas, -deltas], axis=-1), dtype=torch.float64)
    prev_knees = None
    seq_knees = []
    seq_res = []
    for index in range(len(deltas)):
        knees, residual = solver.solve(q_seq[index:index + 1], prev_knees=prev_knees)
        prev_knees = knees.detach()
        seq_knees.append(knees.numpy()[0])
        seq_res.append(float(residual))
    seq_knees = np.array(seq_knees)
    seq_res = np.array(seq_res)
    steps = np.abs(np.diff(seq_knees, axis=0)).max()
    v.check("warm start 序列连续（δ:0→0.6 rad，单步 knee 变化 < 0.05 rad，无分支跳变）",
            float(steps) < 0.05, f"max_step={float(steps):.3e} rad")
    v.check("warm start 序列闭链残差 < 1e-9 m", float(seq_res.max()) < 1e-9, f"{float(seq_res.max()):.3e}")
    # 首帧语义：prev_knees=None 时以 squat 的 knee 解作为装配支路参考 ⇒
    # 输入 squat hips 应精确复现 squat knees。
    squat_hips = torch.tensor([[squat["hips"][n] for n in HIP_ORDER]], dtype=torch.float64)
    first_knees, _ = solver.solve(squat_hips, prev_knees=None)
    v.check("warm start 首帧（prev=None → squat 参考分支）用 squat hips 精确复现 squat knees <1e-9",
            float((first_knees - torch.tensor([[squat["knees"][n] for n in KNEE_ORDER]],
                                              dtype=torch.float64)).abs().max()) < 1e-9)
    v.info("注：warm start 首帧的参考分支是 squat 解；若首次输入就远离 squat（如对称伸长 >0.68 rad），"
           "另一装配支路可能更接近 squat 参考。env 每步都传上一帧解（reset 传 squat 解），因此不受影响。")


# ---------------------------------------------------------------------------
# F. 轮半径（STL 实测）
# ---------------------------------------------------------------------------
def check_wheel_radius(v: Verifier, planar: dict, squat: dict) -> None:
    v.section("F. 轮半径（二进制 STL 顶点 max hypot(x,z) 实测）")
    mesh_dir = locate_mesh_dir()
    left = read_stl_wheel(mesh_dir / "left_wheel.STL")
    right = read_stl_wheel(mesh_dir / "right_wheel.STL")
    v.check("左右轮 STL 均为二进制格式且可解析",
            left["binary"] and right["binary"], f"{left['n_triangles']}/{right['n_triangles']} 三角形")
    v.check("左右轮半径一致 < 1e-9 m", abs(left["radius"] - right["radius"]) < 1e-9,
            f"{left['radius']:.9f} vs {right['radius']:.9f}")
    v.check("planar JSON wheel_radius = STL 实测值 < 1e-12",
            abs(planar["wheel_radius"] - left["radius"]) < 1e-12)
    v.check("squat JSON wheel_radius = STL 实测值 < 1e-12",
            abs(squat["wheel_radius"] - left["radius"]) < 1e-12)
    v.info(f"实测轮半径 = {left['radius']:.9f} m（p99.9={left['radius_p99_9']:.9f}，"
           f"轮宽 y∈[{left['y_min']:+.5f}, {left['y_max']:+.5f}]，旧值 0.068 m）")
    v.info(f"meshes 目录 = {mesh_dir}")


# ---------------------------------------------------------------------------
# G. 包接口
# ---------------------------------------------------------------------------
def check_interfaces(v: Verifier, planar: dict, solver: FiveBarSolver, squat: dict) -> None:
    v.section("G. 包接口 / 数据加载")
    v.check("HIP_ORDER 正确",
            tuple(HIP_ORDER) == ("left_front_hip", "left_back_hip", "right_front_hip", "right_back_hip"))
    v.check("KNEE_ORDER 正确",
            tuple(KNEE_ORDER) == ("left_front_knee", "left_back_knee", "right_front_knee", "right_back_knee"))
    v.check("WHEEL_ORDER 正确", tuple(WHEEL_ORDER) == ("wheel_left", "wheel_right"))
    v.check("load_planar_model() 返回 dict 且含 sides/joint_limits/convention",
            isinstance(load_planar_model(), dict)
            and {"sides", "joint_limits", "convention", "wheel_radius"} <= set(load_planar_model()))
    v.check("load_squat_pose() 返回 dict 且 hips/knees 为关节名 dict",
            isinstance(load_squat_pose(), dict) and set(load_squat_pose()["hips"]) == set(HIP_ORDER)
            and set(load_squat_pose()["knees"]) == set(KNEE_ORDER))

    v.check("FiveBarSolver(device='cpu') 可构造", isinstance(solver, FiveBarSolver), str(solver))
    v.check("solver.knee_limits 为 4 个 knee 的“名 -> (lo,hi)”dict 且与 JSON 一致",
            set(solver.knee_limits) == set(KNEE_ORDER)
            and all(abs(solver.knee_limits[n][0] - planar["joint_limits"][n][0]) < 1e-12
                    and abs(solver.knee_limits[n][1] - planar["joint_limits"][n][1]) < 1e-12
                    for n in KNEE_ORDER))
    q = torch.tensor([[0.1, -0.1, 0.1, -0.1]], dtype=torch.float64)
    result = solver.solve(q)
    v.check("solve() 返回 (knees, residual) 二元组，形状 (N,4)/(N,)",
            isinstance(result, tuple) and len(result) == 2
            and tuple(result[0].shape) == (1, 4) and tuple(result[1].shape) == (1,))
    axle_z = solver.axle_z(q)
    base_height = solver.base_height(q)
    v.check("axle_z()/base_height() 返回 (N,)", tuple(axle_z.shape) == (1,) and tuple(base_height.shape) == (1,))
    v.check("base_height = -axle_z + 轮半径（恒等式）",
            float((base_height + axle_z - solver.wheel_radius).abs().max()) < 1e-12)
    # solve_numpy 单侧接口
    qkf, qkb, residual = solver.solve_numpy(0.1, -0.1, prev=(
        squat["knees"]["left_front_knee"], squat["knees"]["left_back_knee"]), side="left")
    qkf_model, qkb_model, residual_model = solver.models["left"].ik_knees(
        0.1, -0.1, prev=(squat["knees"]["left_front_knee"], squat["knees"]["left_back_knee"]))
    v.check("solve_numpy() 单侧接口与 LegPlanarModel.ik_knees 一致",
            abs(qkf - qkf_model) < 1e-12 and abs(qkb - qkb_model) < 1e-12
            and abs(residual - residual_model) < 1e-12)
    # 与 torch 单侧结果一致
    knees, _ = solver.solve(q)
    v.check("solve_numpy() 与 torch solve() 左侧结果一致 < 1e-9",
            abs(qkf - float(knees[0, 0])) < 1e-9 and abs(qkb - float(knees[0, 1])) < 1e-9)
    v.check("LegPlanarModel.from_dict 左右模型可用",
            all(isinstance(solver.models[side], LegPlanarModel) for side in ("left", "right")))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    _enable_utf8_stdout()
    parser = argparse.ArgumentParser(description="chassis-rl 运动学 + 资产校验（规格 §4.2）")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印细节表格")
    args = parser.parse_args(argv)

    torch.manual_seed(0)  # 无随机性，仅保证环境可复现
    v = Verifier(verbose=args.verbose)
    print("=" * 78)
    print("chassis-rl 运动学校验（_design/00_design_spec.md §4.2）")
    print(f"  python  : {sys.version.split()[0]}")
    print(f"  torch   : {torch.__version__}  device=cpu（不占用 GPU）")
    print(f"  project : {PROJECT_ROOT}")
    if _KINEMATICS_BYPASSED_PACKAGE:
        print("  kinematics: 纯 CPU 方式加载（绕过 chassis_rl_lab/__init__.py 的 Isaac Lab 依赖）")

    # 数据文件先加载（后面各段都要用）
    planar = load_planar_model()
    squat = load_squat_pose()
    models = {side: LegPlanarModel.from_dict(planar, side) for side in ("left", "right")}
    solver = FiveBarSolver(device="cpu")

    sim, cad = check_sim_urdf(v, args.verbose)
    expected = check_planar_model(v, planar, cad, args.verbose)
    check_cad_pose(v, planar, models, expected, sim)
    squat = check_squat(v, planar, models, solver)
    check_numpy_vs_torch(v, planar, models, solver, squat, args.verbose)
    check_wheel_radius(v, planar, squat)
    check_interfaces(v, planar, solver, squat)

    return v.summary()


if __name__ == "__main__":
    sys.exit(main())
