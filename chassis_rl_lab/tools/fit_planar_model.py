#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 URDF 精确提取平面五连杆模型，并搜索 squat（蹲姿）姿态。

产出（不要手写这些数字，全部由本脚本生成）
------------------------------------------
1. ``chassis_rl_lab/data/chassis_rl_planar.json``
   - 每侧（left/right）的平面五连杆参数：hip 点 A_f/A_b、大腿向量 ham_*、
     小腿向量 calf_*（q=0 时 base 系 x-z 投影）、关节限位、派生长度/零位角；
   - 10 个关节的绝对限位（sim URDF，continuous 轮子记为 null）；
   - 轮半径（由 ``meshes/*_wheel.STL`` 顶点实测）；
   - 关节约定 / 角度约定 / 数据来源说明。
2. ``chassis_rl_lab/data/squat_pose.json``
   - 对称 squat 姿态的 hips / knees（按关节名的 dict）、base_height、wheel_radius、
     limits_margin、termination 阈值；
   - 目标：**最小 base height**（最深）且 4 个 hip + 4 个 knee 相对各自限位的
     margin 都 >= 0.03 rad；front_hip = +delta、back_hip = -delta（左右相同），
     knees 由闭链解算得到。

关键实现细节
------------
* 所有几何量由 ElementTree 解析 URDF，在 q=0（CAD 装配位）下做 3D 正运动学后取
  x-z 投影；``<loop_joint>``（仅 CAD URDF 有）提供后小腿的轴孔锚点。
* 角度约定 ``phi = atan2(-z, x)``：从 +x 起、向下为正 = 绕世界 +y 轴右手定则正转。
* **左右对称归一化**：物理上左右腿是精确镜像，但 CAD URDF 的右髋 ``origin rpy`` 写成
  ``0 0 -3.1416``（-pi 的导出舍入，误差 7.35e-6 rad）且右侧 loop 锚点写作
  ``-0.21772998``（左侧 +0.21773），直接提取会让左右参数差 ~4e-7 m、右侧 CAD 位
  闭链残差 ~1.6e-9 m。因此平面模型统一采用左侧（精确闭合，残差 2.8e-17）参数并精确
  镜像到右侧；右侧原始提取值与模型的偏差（<= 1e-6）在 JSON 与 verify 脚本里显式记录。

用法
----
    python tools/build_sim_urdf.py       # 先生成 sim URDF（本脚本需要它取限位/约定）
    python tools/fit_planar_model.py     # 再生成 data/*.json
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 路径与包导入（不依赖当前工作目录）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = PROJECT_ROOT / "chassis_rl_lab"
ASSET_URDF_DIR = PKG_ROOT / "assets" / "chassis-rl" / "urdf"
DATA_DIR = PKG_ROOT / "data"
DEFAULT_SIM_URDF = ASSET_URDF_DIR / "chassis-rl-sim.urdf"
DEFAULT_CAD_COPY = ASSET_URDF_DIR / "chassis-rl.urdf"
CAD_SOURCE_URDF = PROJECT_ROOT.parent / "chassis-rl" / "urdf" / "chassis-rl.urdf"
ASSET_MESH_DIR = PKG_ROOT / "assets" / "chassis-rl" / "meshes"
CAD_MESH_DIR = PROJECT_ROOT.parent / "chassis-rl" / "meshes"
DEFAULT_PLANAR_JSON = DATA_DIR / "chassis_rl_planar.json"
DEFAULT_SQUAT_JSON = DATA_DIR / "squat_pose.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _enable_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，强制 UTF-8 以便中文日志可读。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - 重定向/不支持时忽略
            pass


def _load_kinematics():
    """纯 CPU 方式加载 ``chassis_rl_lab.kinematics``（不触发 Isaac Lab / pxr）。

    ``chassis_rl_lab/__init__.py`` 会 import ``tasks``（进而依赖 Isaac Lab），这是
    训练/仿真入口需要的副作用，但本工具必须在纯 CPU 环境可运行：优先常规导入
    （Kit 环境），失败则绕过父包 ``__init__`` 直接按文件路径加载 kinematics 子包。
    """
    try:
        import chassis_rl_lab.kinematics as kinematics  # noqa: F401

        return kinematics
    except Exception:  # noqa: BLE001 - 纯 CPU 环境缺 pxr/isaaclab 属预期
        for name in [n for n in sys.modules if n == "chassis_rl_lab" or n.startswith("chassis_rl_lab.")]:
            del sys.modules[name]

    import importlib.util
    import types

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


_kinematics = _load_kinematics()
HIP_ORDER = _kinematics.HIP_ORDER
KNEE_ORDER = _kinematics.KNEE_ORDER
WHEEL_ORDER = _kinematics.WHEEL_ORDER
LegPlanarModel = _kinematics.LegPlanarModel
angle_of = _kinematics.angle_of

#: squat 搜索的关节 margin 目标（rad）—— 规格 §4.1/§4.2
SQUAT_MARGIN = 0.03
#: 数值安全余量：搜索时要求 margin >= SQUAT_MARGIN + 该值，避免浮点比较踩线
SQUAT_MARGIN_EPS = 1e-6
#: squat base height 允许区间（规格 §4.2）
SQUAT_HEIGHT_RANGE = (0.16, 0.20)

CONVENTION_TEXT = (
    "所有关节以绕世界 +y 轴右手定则旋转为正（从 +y 看向 -y 逆时针），左右同号。"
    "平面角度约定 phi = atan2(-z, x)：base_link 系 x-z 投影，从 +x 起、向下（-z）为正；"
    "对某关节施加 q 即在对应连杆的绝对角上加 q。q = 0 = CAD 装配位 = 机械行程收回端。"
    "正转 = 前 hip/后 hip 向下摆（腿伸长），前膝负转 = 伸长，后膝正转 = 伸长，"
    "轮子正转 = 车体向 +x（前进）。y 分量对闭链无影响：各链在平行的 y 平面内绕 +y/-y "
    "轴转动，闭链条件为 x-z 平面内前链轮轴点与后链轴孔锚点重合。"
)

SYMMETRY_NOTE = (
    "物理上左右腿是精确镜像；但 CAD 导出的右髋 origin rpy=0 0 -3.1416 是 -pi 的舍入"
    "（误差 7.35e-6 rad），右侧 loop 锚点写作 -0.21772998（左侧 +0.21773），直接提取会使"
    "左右平面参数相差约 4e-7 m、右侧 CAD 位闭链残差约 1.6e-9 m。为满足“对称姿态左右"
    "同值”与“CAD 位闭链 < 1e-9”，平面模型采用左侧精确提取值并精确镜像到右侧；"
    "右侧 CAD 原始提取值与本模型的偏差记录在 cad_raw 中，verify_kinematics.py 会断言其 <= 1e-6。"
)


# ---------------------------------------------------------------------------
# 3D 变换小工具（URDF rpy = 固定轴 roll-pitch-yaw：R = Rz(yaw) Ry(pitch) Rx(roll)）
# ---------------------------------------------------------------------------
def parse_xyz(text: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.array(default, dtype=np.float64)
    values = [float(v) for v in text.split()]
    if len(values) != 3:
        raise ValueError(f"向量应为 3 个分量，实际 {text!r}")
    return np.array(values, dtype=np.float64)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


class UrdfZeroPoseTree:
    """q = 0（CAD 装配位）下的 URDF 刚体树，提供 link 坐标系到 base 系的变换。"""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"URDF 不存在: {self.path}")
        self.root = ET.parse(self.path).getroot()
        self.joints_by_name: dict[str, ET.Element] = {}
        self.joint_by_child_link: dict[str, ET.Element] = {}
        self.loop_joints: list[ET.Element] = list(self.root.findall("loop_joint"))
        for joint in self.root.findall("joint"):
            name = joint.get("name")
            child = joint.find("child").get("link")
            self.joints_by_name[name] = joint
            self.joint_by_child_link[child] = joint
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # -- 基础查询 ------------------------------------------------------
    def joint(self, name: str) -> ET.Element:
        if name not in self.joints_by_name:
            raise KeyError(f"URDF 中没有关节 {name!r}")
        return self.joints_by_name[name]

    def joint_origin(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """返回关节零位 origin 的 (xyz, R)（在父 link 坐标系下）。"""
        joint = self.joint(name)
        origin = joint.find("origin")
        xyz = parse_xyz(origin.get("xyz") if origin is not None else None)
        rpy = parse_xyz(origin.get("rpy") if origin is not None else None)
        return xyz, rpy_to_matrix(rpy)

    def link_transform(self, link: str) -> tuple[np.ndarray, np.ndarray]:
        """返回 link 坐标系在 base 系（q=0）下的 (R, p)。"""
        if link in self._cache:
            return self._cache[link]
        if link == "base_link":
            result = (np.eye(3), np.zeros(3))
        else:
            if link not in self.joint_by_child_link:
                raise KeyError(f"URDF 中 link {link!r} 不是任何 joint 的 child")
            joint = self.joint_by_child_link[link]
            parent = joint.find("parent").get("link")
            parent_r, parent_p = self.link_transform(parent)
            xyz, local_r = self.joint_origin(joint.get("name"))
            result = (parent_r @ local_r, parent_p + parent_r @ xyz)
        self._cache[link] = result
        return result

    def link_position(self, link: str) -> np.ndarray:
        return self.link_transform(link)[1]

    def point_on_link(self, link: str, xyz: np.ndarray) -> np.ndarray:
        rot, pos = self.link_transform(link)
        return pos + rot @ xyz

    def loop_anchor(self, calf_link: str, wheel_link: str) -> tuple[np.ndarray, str, str]:
        """CAD URDF 的 loop_joint：返回后小腿轴孔锚点在 base 系（q=0）的位置及来源信息。"""
        for loop in self.loop_joints:
            refs = [("link1", loop.find("link1")), ("link2", loop.find("link2"))]
            for tag, elem in refs:
                if elem is None:
                    continue
                if elem.get("link") == calf_link:
                    xyz = parse_xyz(elem.get("xyz"))
                    return self.point_on_link(calf_link, xyz), loop.get("name"), tag
        for loop in self.loop_joints:
            for tag in ("link1", "link2"):
                elem = loop.find(tag)
                if elem is not None and elem.get("link") == wheel_link:
                    xyz = parse_xyz(elem.get("xyz"))
                    return self.point_on_link(wheel_link, xyz), loop.get("name"), tag
        raise RuntimeError(f"CAD URDF 中找不到 {calf_link!r}/{wheel_link!r} 对应的 loop_joint 锚点")


# ---------------------------------------------------------------------------
# 单侧平面几何提取
# ---------------------------------------------------------------------------
def extract_side_raw(tree: UrdfZeroPoseTree, side: str) -> dict:
    """从 q=0 的 URDF 精确提取单侧平面五连杆几何（base 系 x-z 投影）。"""
    hip_f_name = f"{side}_front_hip"
    hip_b_name = f"{side}_back_hip"
    calf_f_link = f"{side}_front_calf"
    calf_b_link = f"{side}_back_calf"
    wheel_link = f"{side}_wheel"

    # hip 点：关节 origin 在父 link（base_link）系下的位置
    A_f_xyz = tree.point_on_link(tree.joint(hip_f_name).find("parent").get("link"),
                                 tree.joint_origin(hip_f_name)[0])
    A_b_xyz = tree.point_on_link(tree.joint(hip_b_name).find("parent").get("link"),
                                 tree.joint_origin(hip_b_name)[0])
    # 膝点 = knee 关节 child link（小腿）坐标系原点
    K_f_xyz = tree.link_position(calf_f_link)
    K_b_xyz = tree.link_position(calf_b_link)
    # 轮轴点 = wheel 关节 child link（轮子）坐标系原点
    W_xyz = tree.link_position(wheel_link)
    # 后小腿轴孔锚点 = loop_joint 在 back_calf link 上的点
    anchor_xyz, loop_name, loop_tag = tree.loop_anchor(calf_b_link, wheel_link)

    def project(vec: np.ndarray) -> tuple[float, float]:
        return (float(vec[0]), float(vec[2]))

    params = {
        "side": side,
        "hip_front": project(A_f_xyz),
        "hip_back": project(A_b_xyz),
        "ham_front": project(K_f_xyz - A_f_xyz),
        "ham_back": project(K_b_xyz - A_b_xyz),
        "calf_front": project(W_xyz - K_f_xyz),
        "calf_back": project(anchor_xyz - K_b_xyz),
        # 证据/调试信息（不参与模型计算）
        "raw_3d": {
            "hip_front": [float(v) for v in A_f_xyz],
            "hip_back": [float(v) for v in A_b_xyz],
            "knee_front": [float(v) for v in K_f_xyz],
            "knee_back": [float(v) for v in K_b_xyz],
            "wheel_axle": [float(v) for v in W_xyz],
            "loop_anchor": [float(v) for v in anchor_xyz],
            "loop_joint": loop_name,
            "loop_link_tag": loop_tag,
        },
    }

    # 该侧原始 URDF 在 CAD 位的闭链误差（x-z 平面）：两条链末端点距离
    kf = (params["hip_front"][0] + params["ham_front"][0],
          params["hip_front"][1] + params["ham_front"][1])
    wf = (kf[0] + params["calf_front"][0], kf[1] + params["calf_front"][1])
    kb = (params["hip_back"][0] + params["ham_back"][0],
          params["hip_back"][1] + params["ham_back"][1])
    wb = (kb[0] + params["calf_back"][0], kb[1] + params["calf_back"][1])
    params["cad_pose_closure"] = float(math.hypot(wf[0] - wb[0], wf[1] - wb[1]))
    params["zero_pose"] = {
        "knee_front": list(kf), "axle_front": list(wf),
        "knee_back": list(kb), "axle_back": list(wb),
    }
    params["zero_angles"] = {
        "ham_front": angle_of(*params["ham_front"]),
        "ham_back": angle_of(*params["ham_back"]),
        "calf_front": angle_of(*params["calf_front"]),
        "calf_back": angle_of(*params["calf_back"]),
    }
    return params


def derived_fields(params: dict) -> dict:
    """派生长度 / 零位角（与 LegPlanarModel 的 property 一致）。"""
    def length(vec):
        return float(math.hypot(*vec))

    return {
        "ham_front_len": length(params["ham_front"]),
        "ham_front_zero": angle_of(*params["ham_front"]),
        "ham_back_len": length(params["ham_back"]),
        "ham_back_zero": angle_of(*params["ham_back"]),
        "calf_front_len": length(params["calf_front"]),
        "calf_front_zero": angle_of(*params["calf_front"]),
        "calf_back_len": length(params["calf_back"]),
        "calf_back_zero": angle_of(*params["calf_back"]),
    }


# ---------------------------------------------------------------------------
# 轮半径：解析二进制 STL 顶点，max hypot(x, z)（轮子绕 y 轴转）
# ---------------------------------------------------------------------------
def measure_stl_wheel(stl_path: str | Path) -> dict:
    stl_path = Path(stl_path)
    data = stl_path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{stl_path.name} 太小，不是 STL 文件")
    n_tri = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * n_tri
    if len(data) != expected:
        raise ValueError(
            f"{stl_path.name} 不是二进制 STL（文件 {len(data)} 字节 != 84+50*{n_tri}={expected}）"
        )
    raw = np.frombuffer(data, dtype=np.uint8, count=50 * n_tri, offset=84).reshape(n_tri, 50)
    floats = np.frombuffer(raw[:, :48].tobytes(), dtype="<f4").reshape(n_tri, 12).astype(np.float64)
    verts = floats[:, 3:12].reshape(-1, 3)
    radii = np.hypot(verts[:, 0], verts[:, 2])
    return {
        "path": str(stl_path.resolve()),
        "n_triangles": int(n_tri),
        "n_vertices": int(verts.shape[0]),
        "radius": float(radii.max()),
        "radius_p99_9": float(np.percentile(radii, 99.9)),
        "y_min": float(verts[:, 1].min()),
        "y_max": float(verts[:, 1].max()),
    }


def locate_mesh_dir() -> Path:
    """优先用 assets 里已拷贝的 meshes（Lead 负责拷贝），否则回退到 CAD 源目录。"""
    for candidate in (ASSET_MESH_DIR, CAD_MESH_DIR):
        if (candidate / "left_wheel.STL").is_file():
            return candidate
    raise FileNotFoundError(f"找不到 wheel STL：{ASSET_MESH_DIR} 与 {CAD_MESH_DIR} 都不存在")


def measure_wheel_radius(mesh_dir: Path) -> dict:
    left = measure_stl_wheel(mesh_dir / "left_wheel.STL")
    right = measure_stl_wheel(mesh_dir / "right_wheel.STL")
    if abs(left["radius"] - right["radius"]) > 1e-9:
        raise RuntimeError(
            f"左右轮半径不一致: {left['radius']} vs {right['radius']}"
        )
    return {
        "radius": 0.5 * (left["radius"] + right["radius"]),
        "left": left,
        "right": right,
        "mesh_dir": str(mesh_dir.resolve()),
    }


# ---------------------------------------------------------------------------
# sim URDF：限位表 + 与 CAD URDF 的拓扑一致性
# ---------------------------------------------------------------------------
def read_joint_limits(tree: UrdfZeroPoseTree) -> dict[str, list[float | None]]:
    limits: dict[str, list[float | None]] = {}
    for joint in tree.root.findall("joint"):
        name = joint.get("name")
        joint_type = joint.get("type")
        limit = joint.find("limit")
        if joint_type == "continuous":
            limits[name] = [None, None]  # 无限位（轮子）
        else:
            if limit is None or limit.get("lower") is None or limit.get("upper") is None:
                raise RuntimeError(f"关节 {name!r} 缺少 lower/upper 限位")
            limits[name] = [float(limit.get("lower")), float(limit.get("upper"))]
    expected = set(HIP_ORDER) | set(KNEE_ORDER) | set(WHEEL_ORDER)
    if set(limits) != expected:
        raise RuntimeError(f"sim URDF 关节集合与约定不一致: {sorted(limits)} vs {sorted(expected)}")
    return limits


def check_sim_matches_cad(cad_tree: UrdfZeroPoseTree, sim_tree: UrdfZeroPoseTree) -> None:
    """sim URDF 只允许改 axis/限位/删 loop/mesh 路径：拓扑、origin 必须与 CAD 完全一致。"""
    cad_joints = {j.get("name"): j for j in cad_tree.root.findall("joint")}
    sim_joints = {j.get("name"): j for j in sim_tree.root.findall("joint")}
    if set(cad_joints) != set(sim_joints):
        raise RuntimeError("CAD 与 sim URDF 的关节集合不一致")
    if sim_tree.loop_joints:
        raise RuntimeError("sim URDF 不应包含 loop_joint（先运行 build_sim_urdf.py）")
    if len(cad_tree.loop_joints) != 2:
        raise RuntimeError(f"CAD URDF 应含 2 个 loop_joint，实际 {len(cad_tree.loop_joints)}")
    for name, cad_joint in cad_joints.items():
        sim_joint = sim_joints[name]
        for field in ("type",):
            if cad_joint.get(field) != sim_joint.get(field):
                raise RuntimeError(f"关节 {name!r} 的 {field} 被改动")
        for link_tag in ("parent", "child"):
            if cad_joint.find(link_tag).get("link") != sim_joint.find(link_tag).get("link"):
                raise RuntimeError(f"关节 {name!r} 的 {link_tag} 被改动")
        cad_origin, sim_origin = cad_joint.find("origin"), sim_joint.find("origin")
        for attr in ("xyz", "rpy"):
            cad_val = parse_xyz(cad_origin.get(attr) if cad_origin is not None else None)
            sim_val = parse_xyz(sim_origin.get(attr) if sim_origin is not None else None)
            if not np.allclose(cad_val, sim_val, atol=0.0, rtol=0.0):
                raise RuntimeError(f"关节 {name!r} 的 origin {attr} 被改动（sim 只允许翻 axis/限位）")
    cad_links = {link.get("name") for link in cad_tree.root.findall("link")}
    sim_links = {link.get("name") for link in sim_tree.root.findall("link")}
    if cad_links != sim_links:
        raise RuntimeError("CAD 与 sim URDF 的 link 集合不一致")


# ---------------------------------------------------------------------------
# squat 姿态搜索
# ---------------------------------------------------------------------------
def evaluate_squat(models: dict[str, LegPlanarModel], delta: float, prev: dict | None = None) -> dict:
    """评估对称 hip 姿态 front_hip=+delta / back_hip=-delta 的闭链解与各项指标。"""
    hip_front, hip_back = float(delta), -float(delta)
    knees: dict[str, tuple[float, float]] = {}
    residuals: dict[str, float] = {}
    heights: dict[str, float] = {}
    margins: dict[str, dict[str, float]] = {}
    for side, model in models.items():
        prev_side = None if prev is None else prev[side]
        qk_f, qk_b, residual = model.ik_knees(hip_front, hip_back, prev=prev_side)
        _, axle = model.fk_front(hip_front, qk_f)
        knees[side] = (qk_f, qk_b)
        residuals[side] = residual
        heights[side] = -axle[1] + model.wheel_radius
        lim = model.limits
        margins[side] = {
            "front_hip": min(hip_front - lim["front_hip"][0], lim["front_hip"][1] - hip_front),
            "back_hip": min(hip_back - lim["back_hip"][0], lim["back_hip"][1] - hip_back),
            "front_knee": min(qk_f - lim["front_knee"][0], lim["front_knee"][1] - qk_f),
            "back_knee": min(qk_b - lim["back_knee"][0], lim["back_knee"][1] - qk_b),
        }
    side_knees = next(iter(knees.values()))
    return {
        "delta": float(delta),
        "hip_front": hip_front,
        "hip_back": hip_back,
        "knees": knees,
        "min_knee": side_knees,
        "residuals": residuals,
        "base_height": float(np.mean(list(heights.values()))),
        "margins": margins,
        "min_margin": float(min(m for side_m in margins.values() for m in side_m.values())),
    }


def search_squat_pose(
    models: dict[str, LegPlanarModel],
    margin_target: float = SQUAT_MARGIN,
    margin_eps: float = SQUAT_MARGIN_EPS,
    delta_max: float = 0.6,
    coarse_step: float = 0.002,
    tol: float = 1e-7,
) -> dict:
    """搜索“最深”（最小 base height）且所有 hip/knee margin >= margin_target 的对称 squat。

    由于 base height 随 delta 单调上升（腿越伸长 base 越高），最小 base height 等价于
    满足 margin 约束的最小 delta；用粗扫描定界 + 二分细化，并用 warm start 保证始终
    落在 CAD 装配位对应的物理装配支路。
    """
    target = margin_target + margin_eps

    def with_prev(evaluation: dict) -> dict:
        return {side: evaluation["knees"][side] for side in models}

    # 1) 粗扫描，找第一个可行 delta（warm start 逐点传递）
    prev = None
    last_infeasible: dict | None = None
    feasible: dict | None = None
    delta = 0.0
    while delta <= delta_max + 1e-12:
        evaluation = evaluate_squat(models, delta, prev=prev)
        prev = with_prev(evaluation)
        if evaluation["min_margin"] >= target:
            feasible = evaluation
            break
        last_infeasible = evaluation
        delta += coarse_step
    if feasible is None:
        raise RuntimeError(f"在 delta<=[0,{delta_max}] 内找不到满足 margin>={target} 的 squat 姿态")
    if last_infeasible is None:
        raise RuntimeError("squat 搜索失败：delta=0 已满足 margin（不应发生）")

    # 2) 二分细化（warm start 参考取不可行下界的解，保证同一物理支路）
    lo, hi = last_infeasible, feasible
    prev_fixed = with_prev(lo)
    while hi["delta"] - lo["delta"] > tol:
        mid = 0.5 * (lo["delta"] + hi["delta"])
        evaluation = evaluate_squat(models, mid, prev=prev_fixed)
        if evaluation["min_margin"] >= target:
            hi = evaluation
        else:
            lo = evaluation

    result = hi
    # 3) 结果自检
    if result["base_height"] < SQUAT_HEIGHT_RANGE[0] or result["base_height"] > SQUAT_HEIGHT_RANGE[1]:
        raise RuntimeError(
            f"squat base_height={result['base_height']:.6f} 超出规格区间 {SQUAT_HEIGHT_RANGE}"
        )
    for side, margin_map in result["margins"].items():
        for joint, value in margin_map.items():
            if value < margin_target:
                raise RuntimeError(f"{side} {joint} margin={value:.6f} < {margin_target}")
    # 更深（delta 更小）的姿态必须违反 margin，即当前解已是最深
    deeper = evaluate_squat(models, max(0.0, result["delta"] - 1e-3), prev=with_prev(lo))
    if deeper["min_margin"] >= margin_target:
        raise RuntimeError(
            f"squat 搜索未达到最深：delta={result['delta']-1e-3:.6f} 仍满足 margin "
            f"({deeper['min_margin']:.6f})"
        )

    result["deeper_check"] = {
        "delta": deeper["delta"],
        "min_margin": deeper["min_margin"],
        "base_height": deeper["base_height"],
    }
    return result


# ---------------------------------------------------------------------------
# 组装 JSON
# ---------------------------------------------------------------------------
def build_planar_data(cad_tree: UrdfZeroPoseTree, sim_tree: UrdfZeroPoseTree, mesh_dir: Path) -> tuple[dict, dict]:
    check_sim_matches_cad(cad_tree, sim_tree)
    wheel = measure_wheel_radius(mesh_dir)
    radius = wheel["radius"]

    raw = {side: extract_side_raw(cad_tree, side) for side in ("left", "right")}
    # 左右原始提取偏差（CAD 导出舍入的量级，应 <= 1e-6）
    deviation = 0.0
    for key in ("hip_front", "hip_back", "ham_front", "ham_back", "calf_front", "calf_back"):
        lv, rv = raw["left"][key], raw["right"][key]
        deviation = max(deviation, math.hypot(lv[0] - rv[0], lv[1] - rv[1]))
    for key in ("ham_front", "ham_back", "calf_front", "calf_back"):
        deviation = max(deviation, abs(raw["left"]["zero_angles"][key] - raw["right"]["zero_angles"][key]))
    if deviation > 1e-6:
        raise RuntimeError(f"左右原始几何偏差 {deviation:.3e} 超出 CAD 导出舍入量级（1e-6），请检查 URDF")

    joint_limits = read_joint_limits(sim_tree)
    model_params = raw["left"]  # 精确闭合的一侧（残差 2.8e-17），作为统一镜像模型

    def side_limits(side: str) -> dict:
        return {
            "front_hip": list(joint_limits[f"{side}_front_hip"]),
            "back_hip": list(joint_limits[f"{side}_back_hip"]),
            "front_knee": list(joint_limits[f"{side}_front_knee"]),
            "back_knee": list(joint_limits[f"{side}_back_knee"]),
        }

    sides: dict[str, dict] = {}
    for side in ("left", "right"):
        sides[side] = {
            "hip_front": [model_params["hip_front"][0], model_params["hip_front"][1]],
            "hip_back": [model_params["hip_back"][0], model_params["hip_back"][1]],
            "ham_front": list(model_params["ham_front"]),
            "ham_back": list(model_params["ham_back"]),
            "calf_front": list(model_params["calf_front"]),
            "calf_back": list(model_params["calf_back"]),
            "limits": side_limits(side),
            "wheel_radius": radius,
            "derived": derived_fields(model_params),
            "zero_pose": model_params["zero_pose"],
            "note": "平面几何取左侧精确值并精确镜像（见顶层 symmetry 字段）",
        }

    data = {
        "source_urdf": str(sim_tree.path),
        "source_cad_urdf": str(cad_tree.path),
        "generated_by": "tools/fit_planar_model.py",
        "units": "m, rad",
        "convention": CONVENTION_TEXT,
        "symmetry": {
            "normalized": True,
            "note": SYMMETRY_NOTE,
            "raw_left_right_max_deviation": deviation,
            "raw_cad_pose_closure": {
                side: raw[side]["cad_pose_closure"] for side in ("left", "right")
            },
            "cad_raw": {
                side: {
                    key: list(raw[side][key])
                    for key in ("hip_front", "hip_back", "ham_front", "ham_back",
                                "calf_front", "calf_back")
                }
                | {"zero_angles": raw[side]["zero_angles"]}
                for side in ("left", "right")
            },
        },
        "cad_pose": {
            "wheel_axle_3d": raw["left"]["raw_3d"]["wheel_axle"],
            "axle_xz": list(raw["left"]["zero_pose"]["axle_front"]),
            "base_height": -raw["left"]["zero_pose"]["axle_front"][1] + radius,
            "closure_left": raw["left"]["cad_pose_closure"],
            "closure_right": raw["right"]["cad_pose_closure"],
        },
        "wheel_radius": radius,
        "wheel_radius_source": wheel["left"]["path"],
        "wheel_radius_measure": {
            "method": "二进制 STL 全部顶点 r = max hypot(x, z)（轮子绕 y 轴）",
            "left": wheel["left"],
            "right": wheel["right"],
        },
        "sides": sides,
        "joint_limits": joint_limits,
        "hip_order": list(HIP_ORDER),
        "knee_order": list(KNEE_ORDER),
        "wheel_order": list(WHEEL_ORDER),
    }
    return data, raw


def build_squat_data(models: dict[str, LegPlanarModel], planar_data: dict) -> dict:
    result = search_squat_pose(models)
    delta = result["delta"]
    hips = {
        "left_front_hip": result["hip_front"],
        "left_back_hip": result["hip_back"],
        "right_front_hip": result["hip_front"],
        "right_back_hip": result["hip_back"],
    }
    knee_values = {
        "left_front_knee": result["knees"]["left"][0],
        "left_back_knee": result["knees"]["left"][1],
        "right_front_knee": result["knees"]["right"][0],
        "right_back_knee": result["knees"]["right"][1],
    }
    base_height = result["base_height"]
    # 各关节 margin（按关节名，取左右两侧较小值）
    margins: dict[str, float] = {}
    for joint_name in HIP_ORDER + KNEE_ORDER:
        side = "left" if joint_name.startswith("left") else "right"
        key = joint_name.split("_", 1)[1]  # front_hip / back_knee / ...
        margins[joint_name] = result["margins"][side][key]

    squat = {
        "hips": hips,
        "knees": knee_values,
        "base_height": base_height,
        "wheel_radius": planar_data["wheel_radius"],
        "limits_margin": SQUAT_MARGIN,
        "termination": {
            "base_too_low": base_height - 0.05,
            "base_too_high": base_height + 0.15,
            "tilt_rad": 0.5,
        },
        "hips_delta": delta,
        "margins": margins,
        "closure_residual": {side: result["residuals"][side] for side in ("left", "right")},
        "base_height_side": None,  # 占位，下面填
        "search": {
            "objective": "最小 base_height（最深）s.t. 4 个 hip + 4 个 knee 相对限位 margin >= 0.03 rad",
            "symmetry": "front_hip = +delta, back_hip = -delta, 左右相同；knees 由闭链解算",
            "method": "粗扫描(步长 0.002 rad)定界 + 二分细化(tol 1e-7 rad)，warm start 保持物理装配支路",
            "note": "q=0 是机械行程收回端，base height 随 delta 单调上升；因此最深的可行姿态即满足 margin 的最小 delta",
            "deeper_check": result["deeper_check"],
        },
        "convention": CONVENTION_TEXT,
        "source_planar_json": "chassis_rl_lab/data/chassis_rl_planar.json",
        "generated_by": "tools/fit_planar_model.py",
    }
    # 左右 base height（模型对称时应完全相等）
    heights = {}
    for side, model in models.items():
        qk_f = knee_values[f"{side}_front_knee"]
        _, axle = model.fk_front(result["hip_front"], qk_f)
        heights[side] = -axle[1] + model.wheel_radius
    squat["base_height_side"] = heights
    return squat


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _fmt_vec(vec) -> str:
    return f"({vec[0]:+.8f}, {vec[1]:+.8f})"


def main(argv: list[str] | None = None) -> int:
    _enable_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="从 URDF 提取平面五连杆模型并搜索 squat 姿态，生成 data/*.json"
    )
    parser.add_argument("--sim-urdf", type=Path, default=DEFAULT_SIM_URDF)
    parser.add_argument("--cad-urdf", type=Path, default=None,
                        help="CAD URDF（默认优先用 assets 下的原样拷贝，其次 D:\\Robomaster\\RL\\chassis-rl\\urdf\\chassis-rl.urdf）")
    parser.add_argument("--mesh-dir", type=Path, default=None)
    parser.add_argument("--planar-out", type=Path, default=DEFAULT_PLANAR_JSON)
    parser.add_argument("--squat-out", type=Path, default=DEFAULT_SQUAT_JSON)
    args = parser.parse_args(argv)

    sim_urdf = Path(args.sim_urdf).resolve()
    if not sim_urdf.is_file():
        print(f"[fit_planar_model] 错误：找不到 sim URDF {sim_urdf}\n"
              f"  请先运行: python tools/build_sim_urdf.py", file=sys.stderr)
        return 1
    if args.cad_urdf is not None:
        cad_urdf = Path(args.cad_urdf).resolve()
    elif DEFAULT_CAD_COPY.is_file():
        cad_urdf = DEFAULT_CAD_COPY
    else:
        cad_urdf = CAD_SOURCE_URDF
    mesh_dir = Path(args.mesh_dir).resolve() if args.mesh_dir is not None else locate_mesh_dir()

    cad_tree = UrdfZeroPoseTree(cad_urdf)
    sim_tree = UrdfZeroPoseTree(sim_urdf)

    print("[fit_planar_model] 输入")
    print(f"  CAD URDF : {cad_tree.path}")
    print(f"  sim URDF : {sim_tree.path}")
    print(f"  meshes   : {mesh_dir}")

    planar_data, raw = build_planar_data(cad_tree, sim_tree, mesh_dir)
    planar_out = Path(args.planar_out).resolve()
    planar_out.parent.mkdir(parents=True, exist_ok=True)
    planar_out.write_text(json.dumps(planar_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 打印提取结果
    left = planar_data["sides"]["left"]
    print("\n[fit_planar_model] 平面模型（left，右侧为精确镜像；单位 m/rad）")
    print(f"  hip_front A_f = {_fmt_vec(left['hip_front'])}")
    print(f"  hip_back  A_b = {_fmt_vec(left['hip_back'])}")
    print(f"  ham_front     = {_fmt_vec(left['ham_front'])}  len={left['derived']['ham_front_len']:.9f}  "
          f"zero={math.degrees(left['derived']['ham_front_zero']):.5f} deg")
    print(f"  ham_back      = {_fmt_vec(left['ham_back'])}  len={left['derived']['ham_back_len']:.9f}  "
          f"zero={math.degrees(left['derived']['ham_back_zero']):.5f} deg")
    print(f"  calf_front    = {_fmt_vec(left['calf_front'])}  len={left['derived']['calf_front_len']:.9f}  "
          f"zero={math.degrees(left['derived']['calf_front_zero']):.5f} deg")
    print(f"  calf_back     = {_fmt_vec(left['calf_back'])}  len={left['derived']['calf_back_len']:.9f}  "
          f"zero={math.degrees(left['derived']['calf_back_zero']):.5f} deg")
    print(f"  左右原始提取最大偏差 = {planar_data['symmetry']['raw_left_right_max_deviation']:.3e} "
          f"(CAD 导出舍入；模型已归一化为精确镜像)")
    print(f"  CAD 位闭链残差: left={planar_data['cad_pose']['closure_left']:.3e} m, "
          f"right(原始 CAD)={planar_data['cad_pose']['closure_right']:.3e} m")
    print(f"  轮半径（STL 实测）= {planar_data['wheel_radius']:.9f} m  "
          f"[{planar_data['wheel_radius_source']}]")
    print(f"  CAD 位 base_height = -axle_z + R = {planar_data['cad_pose']['base_height']:.9f} m")
    print(f"  输出: {planar_out}")

    # squat 搜索
    models = {side: LegPlanarModel.from_dict(planar_data, side) for side in ("left", "right")}
    squat = build_squat_data(models, planar_data)
    squat_out = Path(args.squat_out).resolve()
    squat_out.parent.mkdir(parents=True, exist_ok=True)
    squat_out.write_text(json.dumps(squat, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n[fit_planar_model] squat 姿态搜索结果")
    print(f"  delta = {squat['hips_delta']:.9f} rad")
    print(f"  hips  = front {squat['hips']['left_front_hip']:+.9f} / back {squat['hips']['left_back_hip']:+.9f}"
          f"（左右相同）")
    print(f"  knees = front {squat['knees']['left_front_knee']:+.9f} / back {squat['knees']['left_back_knee']:+.9f}"
          f"（左右相同）")
    print(f"  base_height = {squat['base_height']:.9f} m")
    print("  各关节限位 margin (rad):")
    for joint_name, margin in squat["margins"].items():
        lower, upper = planar_data["joint_limits"][joint_name]
        value = squat["hips"].get(joint_name, squat["knees"].get(joint_name))
        print(f"    {joint_name:18s} q={value:+.9f}  limit=[{lower:+.6f}, {upper:+.6f}]  margin={margin:.9f}")
    print(f"  termination: base_too_low={squat['termination']['base_too_low']:.9f}, "
          f"base_too_high={squat['termination']['base_too_high']:.9f}, "
          f"tilt_rad={squat['termination']['tilt_rad']}")
    print(f"  闭链残差: left={squat['closure_residual']['left']:.3e} m, "
          f"right={squat['closure_residual']['right']:.3e} m")
    print(f"  输出: {squat_out}")
    print("\n[fit_planar_model] 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
