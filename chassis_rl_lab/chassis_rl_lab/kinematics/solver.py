#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""torch 批量五连杆闭链解算器（全向量化，CPU / GPU 通用）。

作用
----
仿真（Isaac Lab / PhysX）只加载开链 URDF，4 个 knee 不参与动力学求解；
环境每个控制步由实测 hip 角解析解出 knee 目标，再写入位置目标让 PD 跟踪。
本模块就是该解析解算的 torch 批量实现：

* 输入 ``q_hips`` (N, 4)，顺序 ``HIP_ORDER``
  = (left_front_hip, left_back_hip, right_front_hip, right_back_hip)；
* 输出 knees (N, 4)，顺序 ``KNEE_ORDER``
  = (left_front_knee, left_back_knee, right_front_knee, right_back_knee)，
  以及闭链残差 residual (N,)（米）；
* 全向量化、无 Python 循环；闭链两个装配支路用 warm start（``prev_knees``）选择，
  首帧（``prev_knees=None``）用 squat 姿态的 knee 解作为参考分支；
* 内部用 float64 计算以保证残差精度，返回时保持输入的 dtype / device
  （训练时输入 float32，则输出 float32；校验脚本用 float64 可拿到 <1e-9 的残差）。

关节顺序与角度约定见 ``.planar_model``；数据来自 ``data/chassis_rl_planar.json``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .planar_model import (
    DEFAULT_PLANAR_PATH,
    DEFAULT_SQUAT_PATH,
    HIP_ORDER,
    KNEE_ORDER,
    LEG_SIDES,
    WHEEL_ORDER,
    LegPlanarModel,
    load_planar_model,
    load_squat_pose,
)


# ---------------------------------------------------------------------------
# 向量化小工具
# ---------------------------------------------------------------------------
def _wrap(x: torch.Tensor) -> torch.Tensor:
    """向量化角度归一化到 (-pi, pi]（对 2*pi 的周期安全）。"""
    return torch.atan2(torch.sin(x), torch.cos(x))


def _rotate(x: torch.Tensor, z: torch.Tensor, angle: torch.Tensor):
    """绕世界 +y 轴旋转（phi -> phi + angle）：返回旋转后的 (x', z')。"""
    c, s = torch.cos(angle), torch.sin(angle)
    return x * c + z * s, z * c - x * s


def _as_float64_batch(tensor, name: str, width: int) -> tuple[torch.Tensor, bool, torch.device, torch.dtype]:
    """把输入整理成 (N, width) 的 float64；返回 (batch, squeezed, 原 device, 原 dtype)。"""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor, dtype=torch.float32)
    if tensor.dim() == 1:
        squeezed = True
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2:
        squeezed = False
    else:
        raise ValueError(f"{name} 必须是 (N,{width}) 或 ({width},)，实际 shape={tuple(tensor.shape)}")
    if tensor.shape[-1] != width:
        raise ValueError(f"{name} 最后一维必须是 {width}，实际 shape={tuple(tensor.shape)}")
    out = tensor.to(dtype=torch.float64)
    return out, squeezed, tensor.device, (tensor.dtype if tensor.is_floating_point() else torch.float32)


class FiveBarSolver:
    """五连杆闭链批量解算器。

    Parameters
    ----------
    device : str | torch.device
        计算设备，默认 "cuda:0"（环境里用）；纯 CPU 工具传 "cpu"。
    data_path : str | Path | None
        ``data/chassis_rl_planar.json`` 路径；None 用包内默认数据（同目录下查找
        ``squat_pose.json`` 作为 warm start 首帧参考）。
    """

    def __init__(self, device: str | torch.device = "cuda:0", data_path: str | Path | None = None):
        self.device = torch.device(device)
        self.data_path = Path(data_path).resolve() if data_path is not None else DEFAULT_PLANAR_PATH
        self.planar_data = load_planar_model(self.data_path)
        self.models: dict[str, LegPlanarModel] = {
            side: LegPlanarModel.from_dict(self.planar_data, side) for side in LEG_SIDES
        }
        self.wheel_radius: float = float(self.planar_data["wheel_radius"])

        joint_limits = self.planar_data["joint_limits"]
        #: knee 关节限位：{关节名: (lower, upper)}，供 env / 工具 clamp 用
        self.knee_limits: dict[str, tuple[float, float]] = {
            name: (float(joint_limits[name][0]), float(joint_limits[name][1]))
            for name in KNEE_ORDER
        }

        # squat 姿态数据（warm start 首帧参考分支 / 默认姿态）
        squat_path = self.data_path.parent / "squat_pose.json"
        if not squat_path.is_file():
            squat_path = DEFAULT_SQUAT_PATH
        self.squat_pose: Optional[dict] = load_squat_pose(squat_path) if squat_path.is_file() else None
        if self.squat_pose is None:
            default_knee = [0.0] * 4
        else:
            default_knee = [float(self.squat_pose["knees"][name]) for name in KNEE_ORDER]

        self._build_buffers(default_knee)

    # ------------------------------------------------------------------
    # 参数张量化
    # ------------------------------------------------------------------
    def _side_param(self, getter) -> torch.Tensor:
        """把某标量参数按 (left, right) 顺序打包成 (2,) float64 张量。"""
        return torch.tensor(
            [float(getter(self.models[side])) for side in LEG_SIDES],
            dtype=torch.float64,
            device=self.device,
        )

    def _build_buffers(self, default_knee: list[float]) -> None:
        self._af_x = self._side_param(lambda m: m.hip_front[0])
        self._af_z = self._side_param(lambda m: m.hip_front[1])
        self._ab_x = self._side_param(lambda m: m.hip_back[0])
        self._ab_z = self._side_param(lambda m: m.hip_back[1])
        self._hamf_x = self._side_param(lambda m: m.ham_front[0])
        self._hamf_z = self._side_param(lambda m: m.ham_front[1])
        self._hamb_x = self._side_param(lambda m: m.ham_back[0])
        self._hamb_z = self._side_param(lambda m: m.ham_back[1])
        self._calff_x = self._side_param(lambda m: m.calf_front[0])
        self._calff_z = self._side_param(lambda m: m.calf_front[1])
        self._calfb_x = self._side_param(lambda m: m.calf_back[0])
        self._calfb_z = self._side_param(lambda m: m.calf_back[1])
        self._calf_f_len = torch.hypot(self._calff_x, self._calff_z)
        self._calf_b_len = torch.hypot(self._calfb_x, self._calfb_z)
        self._calf_f_zero = self._side_param(lambda m: m.calf_front_zero)
        self._calf_b_zero = self._side_param(lambda m: m.calf_back_zero)
        self._knee_lower = torch.tensor(
            [self.knee_limits[name][0] for name in KNEE_ORDER],
            dtype=torch.float64,
            device=self.device,
        )
        self._knee_upper = torch.tensor(
            [self.knee_limits[name][1] for name in KNEE_ORDER],
            dtype=torch.float64,
            device=self.device,
        )
        #: warm start 首帧参考（squat 的 knee 解），形状 (4,)，KNEE_ORDER
        self._default_prev = torch.tensor(default_knee, dtype=torch.float64, device=self.device)

    # ------------------------------------------------------------------
    # 核心（内部 float64）
    # ------------------------------------------------------------------
    def _solve_core(self, q_hips, prev_knees=None) -> dict:
        q, squeezed, device, dtype = _as_float64_batch(q_hips, "q_hips", 4)
        q = q.to(self.device)
        n = q.shape[0]

        q_hf = q[:, [0, 2]]  # (N,2) (left, right)
        q_hb = q[:, [1, 3]]

        # 两个膝点 K_f / K_b
        rx, rz = _rotate(self._hamf_x, self._hamf_z, q_hf)
        kfx, kfz = self._af_x + rx, self._af_z + rz
        rx, rz = _rotate(self._hamb_x, self._hamb_z, q_hb)
        kbx, kbz = self._ab_x + rx, self._ab_z + rz

        # 两圆交点（候选轮轴 W），h < 0 时退化为 h = 0（几何不可达，残差 > 0）
        dx, dz = kbx - kfx, kbz - kfz
        d_raw = torch.sqrt(dx * dx + dz * dz)
        d = d_raw.clamp_min(1e-9)
        ex = torch.where(d_raw > 1e-8, dx / d, torch.ones_like(dx))
        ez = torch.where(d_raw > 1e-8, dz / d, torch.zeros_like(dz))
        a = (self._calf_f_len * self._calf_f_len - self._calf_b_len * self._calf_b_len + d * d) / (2.0 * d)
        h_sq = self._calf_f_len * self._calf_f_len - a * a
        h = torch.sqrt(h_sq.clamp_min(0.0))
        nx, nz = -ez, ex
        px, pz = kfx + a * ex, kfz + a * ez

        # warm start 参考：squat 解（prev=None）或上一帧解
        if prev_knees is None:
            prev = self._default_prev.unsqueeze(0).expand(n, 4)
        else:
            prev, _, _, _ = _as_float64_batch(prev_knees, "prev_knees", 4)
            prev = prev.to(self.device)
            if prev.shape[0] != n:
                if prev.shape[0] == 1:
                    prev = prev.expand(n, 4)
                else:
                    raise ValueError(f"prev_knees 的 batch={prev.shape[0]} 与 q_hips 的 batch={n} 不一致")
        prev_front = prev[:, [0, 2]]
        prev_back = prev[:, [1, 3]]

        def _knee_angles(wx: torch.Tensor, wz: torch.Tensor):
            phi_f = torch.atan2(-(wz - kfz), wx - kfx)
            phi_b = torch.atan2(-(wz - kbz), wx - kbx)
            qk_f = _wrap(phi_f - self._calf_f_zero - q_hf)
            qk_b = _wrap(phi_b - self._calf_b_zero - q_hb)
            return qk_f, qk_b

        cands = []
        for sign in (1.0, -1.0):
            wx = px + sign * h * nx
            wz = pz + sign * h * nz
            qk_f, qk_b = _knee_angles(wx, wz)
            dist = _wrap(qk_f - prev_front) ** 2 + _wrap(qk_b - prev_back) ** 2
            cands.append((wx, wz, qk_f, qk_b, dist))

        # 逐 (env, side) 选择离参考更近的装配支路
        use_0 = cands[0][4] <= cands[1][4]
        wx = torch.where(use_0, cands[0][0], cands[1][0])
        wz = torch.where(use_0, cands[0][1], cands[1][1])
        qk_f = torch.where(use_0, cands[0][2], cands[1][2])
        qk_b = torch.where(use_0, cands[0][3], cands[1][3])

        # 由解出的膝角重算两条链的端点在 x-z 平面的距离（闭链残差，米）
        rx, rz = _rotate(self._calff_x, self._calff_z, q_hf + qk_f)
        wfx, wfz = kfx + rx, kfz + rz
        rx, rz = _rotate(self._calfb_x, self._calfb_z, q_hb + qk_b)
        wbx, wbz = kbx + rx, kbz + rz
        res_side = torch.sqrt((wfx - wbx) ** 2 + (wfz - wbz) ** 2)
        residual = res_side.max(dim=-1).values

        knees = torch.stack([qk_f[:, 0], qk_b[:, 0], qk_f[:, 1], qk_b[:, 1]], dim=-1)

        if squeezed:
            knees, residual, wx, wz = knees.squeeze(0), residual.squeeze(0), wx.squeeze(0), wz.squeeze(0)
        return {
            "knees": knees,
            "residual": residual,
            "axle_x": wx,
            "axle_z": wz,
            "squeezed": squeezed,
            "device": device,
            "dtype": dtype,
        }

    @staticmethod
    def _to_out(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return tensor.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def solve(self, q_hips: torch.Tensor, prev_knees: Optional[torch.Tensor] = None):
        """闭链解算。

        Parameters
        ----------
        q_hips : (N, 4) 或 (4,)，HIP_ORDER 顺序的 hip 角。
        prev_knees : (N, 4) 或 (4,)，KNEE_ORDER 顺序的上一帧 knee 解（warm start）；
            None 时用 squat 姿态解作为参考分支（首帧）。

        Returns
        -------
        (knees, residual) : knees (N,4)（KNEE_ORDER，dtype/device 与输入一致）、
            residual (N,)<单位 m>（闭链残差，0 表示严格闭合）。
        """
        core = self._solve_core(q_hips, prev_knees)
        return (
            self._to_out(core["knees"], core["device"], core["dtype"]),
            self._to_out(core["residual"], core["device"], core["dtype"]),
        )

    def closure_residual(self, q_hips: torch.Tensor, q_knees: torch.Tensor) -> torch.Tensor:
        """给定 hip + knee，直接前向计算两条链末端的 x-z 距离（米），形状 (N,)。

        与 ``solve`` 的 residual 含义相同，但不对 knee 做任何解算，可用于校验
        “实测关节角是否满足闭链”。返回 dtype/device 与输入 q_hips 一致。
        """
        q, squeezed, device, dtype = _as_float64_batch(q_hips, "q_hips", 4)
        k, _, _, _ = _as_float64_batch(q_knees, "q_knees", 4)
        q, k = q.to(self.device), k.to(self.device)
        q_hf, q_hb = q[:, [0, 2]], q[:, [1, 3]]
        qk_f, qk_b = k[:, [0, 2]], k[:, [1, 3]]

        rx, rz = _rotate(self._hamf_x, self._hamf_z, q_hf)
        kfx, kfz = self._af_x + rx, self._af_z + rz
        rx, rz = _rotate(self._hamb_x, self._hamb_z, q_hb)
        kbx, kbz = self._ab_x + rx, self._ab_z + rz

        rx, rz = _rotate(self._calff_x, self._calff_z, q_hf + qk_f)
        wfx, wfz = kfx + rx, kfz + rz
        rx, rz = _rotate(self._calfb_x, self._calfb_z, q_hb + qk_b)
        wbx, wbz = kbx + rx, kbz + rz
        residual = torch.sqrt((wfx - wbx) ** 2 + (wfz - wbz) ** 2).max(dim=-1).values
        if squeezed:
            residual = residual.squeeze(0)
        return self._to_out(residual, device, dtype)

    def axle_z(self, q_hips: torch.Tensor) -> torch.Tensor:
        """轮轴中心 z（base 系，米），形状 (N,)。

        取左右两条链解出的轮轴 z 的平均（base 是单一刚体，对称姿态下二者严格相等；
        非对称 hip 输入时该值只是“等效轮轴高度”参考）。
        """
        core = self._solve_core(q_hips, None)
        axle_z = core["axle_z"]
        if axle_z.dim() == 2:
            axle_z = axle_z.mean(dim=-1)
        return self._to_out(axle_z, core["device"], core["dtype"])

    def base_height(self, q_hips: torch.Tensor, wheel_radius: Optional[float] = None) -> torch.Tensor:
        """base 高度（米）= -axle_z + 轮半径，形状 (N,)。"""
        radius = self.wheel_radius if wheel_radius is None else float(wheel_radius)
        return -self.axle_z(q_hips) + radius

    def knee_targets_clamped(self, q_hips: torch.Tensor, margin: float = 0.02) -> torch.Tensor:
        """解算 knee 并 clamp 到关节限位内、每侧留 ``margin``（rad），形状 (N,4)。

        若某关节限位宽度不足 2*margin，则退化为该关节限位中点。
        """
        knees, _ = self.solve(q_hips)
        knees64 = knees.to(device=self.device, dtype=torch.float64)
        lo = self._knee_lower + margin
        hi = self._knee_upper - margin
        mid = 0.5 * (self._knee_lower + self._knee_upper)
        lo_safe = torch.where(lo <= hi, lo, mid)
        hi_safe = torch.where(lo <= hi, hi, mid)
        clamped = torch.clamp(knees64, min=lo_safe, max=hi_safe)
        return clamped.to(device=knees.device, dtype=knees.dtype)

    def solve_numpy(self, q_hip_front: float, q_hip_back: float,
                    prev: Optional[tuple[float, float]] = None, side: str = "left"):
        """单侧 numpy 解算（工具/测试用），返回 (前膝角, 后膝角, 闭链残差[m])。"""
        return self.models[side].ik_knees(float(q_hip_front), float(q_hip_back), prev=prev)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (f"FiveBarSolver(device={self.device}, data={self.data_path}, "
                f"wheel_radius={self.wheel_radius:.6f})")


__all__ = ["FiveBarSolver", "HIP_ORDER", "KNEE_ORDER", "WHEEL_ORDER"]
