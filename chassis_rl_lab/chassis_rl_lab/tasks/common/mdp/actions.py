"""chassis-rl 轮腿机器人动作项：4 hip 位置 + 2 轮（速度或力矩），4 knee 由五连杆闭链解析随动。

动作语义（三种模式）
--------------------
1) ``use_hip_actions=True`` + ``hip_control='position'``（6 维，位置控制）::

    [0:4] 4 个 hip 的位置目标偏移 (rad, × hip_scale)，顺序 HIP_ORDER
    [4:6] 2 个轮的控制目标，顺序 WHEEL_ORDER
          - wheel_control="velocity"（默认）：轮速目标 (rad/s, × wheel_scale)
          - wheel_control="torque"：轮力矩 (Nm, × wheel_torque_scale)

2) ``use_hip_actions=True`` + ``hip_control='torque'``（6 维，纯力矩）::

    [0:4] 4 个 hip 力矩 (Nm, × hip_torque_scale)，夹到 ±HIP_EFFORT_LIMIT
    [4:6] 2 个轮力矩 (Nm, × wheel_torque_scale)，夹到 ±WHEEL_EFFORT_LIMIT
    （需要 hips/wheels 执行器 stiffness=damping=0）

3) ``use_hip_actions=False``（髋关节无输出/锁死，动作 2 维）::

    [0:2] 2 个轮的控制目标（速度或力矩）
    hip 关节不写目标（配合零刚度/零阻尼执行器 → 无力被动）；或 lock_hips=True 时
    由 PD 锁死在 squat 姿态。

``wheel_unpowered_steps>0`` 时，每个 episode 开始的前 N 个控制步轮子输出强制为 0 力矩
（"轮子无力"阶段，机器人自然倒地），之后才接管策略输出。

- 4 个 knee 是五连杆闭链的**被动随动关节**：Isaac Lab / PhysX 不约束 loop_joint，
  所以每个控制步用**实测 hip 角**通过 ``FiveBarSolver`` 解出 knee 目标，夹到限位内后写位置目标。
  只有当 10 个关节都跟踪目标时闭链才近似成立（knee PD 增益较高，实测残差 ~mm 级）。

关节符号约定：所有关节绕世界 +y 轴右手定则正转为正（左右同号）；轮子正转 = 车体 +x 前进。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from chassis_rl_lab.kinematics import (
    HIP_ORDER,
    KNEE_ORDER,
    WHEEL_ORDER,
    FiveBarSolver,
    load_squat_pose,
)
from chassis_rl_lab.robots.chassis_rl_cfg import (
    HIP_ACTION_SCALE,
    HIP_EFFORT_LIMIT,
    WHEEL_ACTION_SCALE,
    WHEEL_EFFORT_LIMIT,
    WHEEL_VELOCITY_LIMIT,
)


def _ordered(values: Mapping[str, float] | Sequence[float], order: Sequence[str]) -> tuple[float, ...]:
    """把 ``{"joint_name": value}`` 或按 ``order`` 排好的序列统一成 tuple。"""
    if isinstance(values, Mapping):
        return tuple(float(values[name]) for name in order)
    values = tuple(float(v) for v in values)
    if len(values) != len(order):
        raise ValueError(f"expected {len(order)} values, got {len(values)}")
    return values


class ChassisWheelLegAction(ActionTerm):
    """chassis-rl 轮腿动作项（hip 位置 + 轮速度；knee 五连杆闭链随动）。"""

    cfg: ChassisWheelLegActionCfg
    _asset: Articulation

    def __init__(self, cfg: ChassisWheelLegActionCfg, env):  # noqa: ANN001 - Isaac Lab ManagerBasedEnv
        super().__init__(cfg, env)

        # ---- 关节按名字解析（绝不依赖 articulation 内部顺序） ---- #
        self._hip_ids, self._hip_names = self._asset.find_joints(
            list(self.cfg.hip_joint_names), preserve_order=True
        )
        self._knee_ids, self._knee_names = self._asset.find_joints(
            list(self.cfg.knee_joint_names), preserve_order=True
        )
        self._wheel_ids, self._wheel_names = self._asset.find_joints(
            list(self.cfg.wheel_joint_names), preserve_order=True
        )
        if list(self._hip_names) != list(self.cfg.hip_joint_names):
            raise RuntimeError(f"hip 关节解析顺序不一致: {self._hip_names}")
        if list(self._knee_names) != list(self.cfg.knee_joint_names):
            raise RuntimeError(f"knee 关节解析顺序不一致: {self._knee_names}")
        if list(self._wheel_names) != list(self.cfg.wheel_joint_names):
            raise RuntimeError(f"wheel 关节解析顺序不一致: {self._wheel_names}")
        self._num_hips = len(self._hip_ids)
        self._num_knees = len(self._knee_ids)
        self._num_wheels = len(self._wheel_ids)
        #: 是否把 4 个 hip 位置作为策略动作。False = 髋关节无输出（配合零刚度执行器 = 无力被动），
        #: 策略动作只剩 2 个轮速（此时 hip 目标不写入，髋关节自由摆动）。
        self._use_hip_actions = bool(self.cfg.use_hip_actions)
        self._num_hip_actions = self._num_hips if self._use_hip_actions else 0
        self._action_dim = self._num_hip_actions + self._num_wheels
        #: 屏蔽髋输出（动作维度保留，但前 4 维髋动作被忽略；髋由 PD 保持在 squat）
        self._mask_hip_actions = bool(self.cfg.mask_hip_actions)
        if self._mask_hip_actions and not self._use_hip_actions:
            raise ValueError("mask_hip_actions=True 需要 use_hip_actions=True（保持 6 维动作接口）")

        # ---- 轮子控制模式 ----
        if self.cfg.wheel_control not in ("velocity", "torque"):
            raise ValueError(f"wheel_control 必须是 'velocity' 或 'torque'，收到 {self.cfg.wheel_control!r}")
        self._wheel_control = str(self.cfg.wheel_control)
        self._wheel_unpowered_steps = int(self.cfg.wheel_unpowered_steps)
        if self._wheel_unpowered_steps < 0:
            raise ValueError(f"wheel_unpowered_steps 必须 >= 0，收到 {self._wheel_unpowered_steps}")

        # ---- 髋关节控制模式 ----
        if self.cfg.hip_control not in ("position", "torque"):
            raise ValueError(f"hip_control 必须是 'position' 或 'torque'，收到 {self.cfg.hip_control!r}")
        self._hip_control = str(self.cfg.hip_control)
        if self._hip_control == "torque" and not self._use_hip_actions:
            raise ValueError("hip_control='torque' 需要 use_hip_actions=True（髋力矩是策略动作的一部分）")
        #: 髋力矩目标 (N, 4)，仅 hip_control='torque' 时使用
        self._hip_torques = torch.zeros(self.num_envs, self._num_hips, device=self.device)

        # ---- squat 姿态（左右相同） ---- #
        squat = load_squat_pose(self.cfg.squat_pose_path)
        self._squat_hips = torch.tensor(
            _ordered(squat["hips"], self.cfg.hip_joint_names), device=self.device, dtype=torch.float32
        )
        self._squat_knees = torch.tensor(
            _ordered(squat["knees"], self.cfg.knee_joint_names), device=self.device, dtype=torch.float32
        )

        # ---- 五连杆闭链解算器（GPU 批量） ---- #
        self._solver = FiveBarSolver(device=str(self.device))

        # ---- 关节限位（来自 USD/URDF，按名字取；兜底用运动学模型数据） ---- #
        planar_limits = self._solver.planar_data["joint_limits"]
        hip_fallback = {
            name: (float(planar_limits[name][0]), float(planar_limits[name][1]))
            for name in self.cfg.hip_joint_names
        }
        self._hip_lo, self._hip_hi = self._joint_limits(self._hip_ids, hip_fallback, self.cfg.hip_joint_names)
        self._knee_lo, self._knee_hi = self._joint_limits(
            self._knee_ids, self._solver.knee_limits, self.cfg.knee_joint_names
        )

        # ---- 动作 / 目标 / 解算缓存 ---- #
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._hip_targets = self._squat_hips.unsqueeze(0).repeat(self.num_envs, 1)
        self._wheel_targets = torch.zeros(self.num_envs, self._num_wheels, device=self.device)
        #: 力矩模式下的轮力矩目标 (N, 2)
        self._wheel_torques = torch.zeros(self.num_envs, self._num_wheels, device=self.device)
        #: 每个 env 已执行的控制步数（用于 wheel_unpowered_steps "自然倒地"阶段）
        self._step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._knee_targets = self._squat_knees.unsqueeze(0).repeat(self.num_envs, 1)
        #: 上一控制步的 knee 解（闭链解算分支的 warm start）
        self._prev_knees = self._squat_knees.unsqueeze(0).repeat(self.num_envs, 1)
        #: 最近一次闭链残差 [m]（前链轮轴点与后链轴孔锚点距离）
        self._closure_residual = torch.zeros(self.num_envs, device=self.device)

        # ---- 平衡辅助（PD 辅助力矩 + 线性退火，起立任务"扶着学"） ---- #
        self._assist_enabled = bool(self.cfg.balance_assist)
        self._assist_kp = float(self.cfg.assist_kp)
        self._assist_kd = float(self.cfg.assist_kd)
        self._assist_start = float(self.cfg.assist_scale_start)
        self._assist_end = float(self.cfg.assist_scale_end)
        self._assist_anneal_steps = int(self.cfg.assist_anneal_steps)
        self._assist_cycle_steps = int(self.cfg.assist_cycle_steps)
        self._assist_phases = int(self.cfg.assist_phases)
        self._assist_scale = self._assist_start
        #: 全局控制步计数（跨 episode 累积，用于辅助力矩退火；不随 reset 归零）
        self._global_steps = 0
        #: 最近一次实际施加的辅助力矩 (N, 2)，调试/日志用
        self._assist_torques = torch.zeros(self.num_envs, self._num_wheels, device=self.device)

        # ---- 开局"收腿稳定"阶段（倒地起始：先在最短腿长姿态稳住，再交给策略） ---- #
        self._settle_steps = int(self.cfg.settle_steps)
        if self._settle_steps < 0:
            raise ValueError(f"settle_steps 必须 >= 0，收到 {self._settle_steps}")
        settle_positions = tuple(float(v) for v in self.cfg.settle_hip_positions)
        if len(settle_positions) != self._num_hips:
            raise ValueError(
                f"settle_hip_positions 长度 {len(settle_positions)} != hip 数 {self._num_hips}"
            )
        self._settle_hips = torch.tensor(settle_positions, device=self.device, dtype=torch.float32)
        self._settle_kp = float(self.cfg.settle_hip_kp)
        self._settle_kd = float(self.cfg.settle_hip_kd)
        #: 本控制步是否处于收腿稳定阶段 (N,)
        self._settling = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ---- 鲁棒性：动作延迟 / 力矩缩放 / 力矩一阶低通（模拟电机与信号链的非理想特性） ---- #
        self._dt = float(env.step_dt)
        self._action_delay_max = max(0, int(getattr(self.cfg, "action_delay_max", 0)))
        self._action_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._action_history = (
            torch.zeros(
                self.num_envs, self._action_delay_max + 1, self._action_dim, device=self.device
            )
            if self._action_delay_max > 0
            else None
        )
        self._history_index = 0
        self._torque_scale_range = tuple(
            float(v) for v in getattr(self.cfg, "torque_scale_range", (1.0, 1.0))
        )
        self._torque_scale = torch.ones(self.num_envs, 1, device=self.device)
        self._torque_lpf_tau_range = tuple(
            float(v) for v in getattr(self.cfg, "torque_lpf_tau_range", (0.0, 0.0))
        )
        self._torque_lpf_tau = torch.zeros(self.num_envs, 1, device=self.device)
        self._hip_tau_filt = torch.zeros(self.num_envs, self._num_hips, device=self.device)
        self._wheel_tau_filt = torch.zeros(self.num_envs, self._num_wheels, device=self.device)

    # ------------------------------------------------------------------ #
    # 关节限位
    # ------------------------------------------------------------------ #
    def _joint_limits(
        self, joint_ids: Sequence[int], fallback: Mapping[str, tuple[float, float]], names: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """优先用资产实测限位（USD/URDF），否则退回运动学模型给出的限位。"""
        lo_fb = torch.tensor([fallback[n][0] for n in names], device=self.device, dtype=torch.float32)
        hi_fb = torch.tensor([fallback[n][1] for n in names], device=self.device, dtype=torch.float32)
        try:
            limits = self._asset.data.joint_pos_limits
            if limits is None:
                return lo_fb.unsqueeze(0).repeat(self.num_envs, 1), hi_fb.unsqueeze(0).repeat(self.num_envs, 1)
            limits = limits[:, list(joint_ids)]
            if limits.shape[0] != self.num_envs:
                limits = limits.expand(self.num_envs, -1, -1)
            lo = torch.nan_to_num(limits[..., 0].float(), nan=-1.0e9, posinf=1.0e9, neginf=-1.0e9)
            hi = torch.nan_to_num(limits[..., 1].float(), nan=1.0e9, posinf=1.0e9, neginf=1.0e9)
            # 限制为有限值时以资产为准；若资产限位退化（±inf/0 宽）则用模型值
            degenerate = (hi - lo) < 1.0e-6
            lo = torch.where(degenerate, lo_fb.unsqueeze(0).expand_as(lo), lo)
            hi = torch.where(degenerate, hi_fb.unsqueeze(0).expand_as(hi), hi)
            return lo, hi
        except Exception:  # noqa: BLE001 - 数据还没初始化时退回模型值
            return lo_fb.unsqueeze(0).repeat(self.num_envs, 1), hi_fb.unsqueeze(0).repeat(self.num_envs, 1)

    # ------------------------------------------------------------------ #
    # ActionTerm 接口
    # ------------------------------------------------------------------ #
    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """最近一次送入动作项的原始动作 (N, 6)（已按 cfg.clip_actions 裁剪）。"""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """最近一次处理后的动作 (N, 6)：前 4 维为 hip 位置目标，后 2 维为轮速目标。"""
        return self._processed_actions

    @property
    def applied_hip_targets(self) -> torch.Tensor:
        """最近一次写入 PhysX 的 hip 位置目标 (N, 4)。"""
        return self._hip_targets

    @property
    def applied_knee_targets(self) -> torch.Tensor:
        """最近一次写入 PhysX 的 knee 位置目标 (N, 4)。"""
        return self._knee_targets

    @property
    def applied_wheel_targets(self) -> torch.Tensor:
        """最近一次写入 PhysX 的轮速目标 (N, 2)。"""
        return self._wheel_targets

    @property
    def closure_residual(self) -> torch.Tensor:
        """五连杆闭链残差 (N,) [m]。"""
        return self._closure_residual

    def process_actions(self, actions: torch.Tensor) -> None:
        """把 [-clip, clip] 的策略动作映射成 hip 位置目标与轮速目标。

        ``use_hip_actions=True``：动作 ``[4 hip, 2 wheel]``；
        ``use_hip_actions=False``（髋关节无输出/无力）：动作只有 ``[2 wheel]``，
        hip 目标保持 squat 但**不写入**（执行器 stiffness=damping=0，髋关节自由）。
        """
        actions = actions.to(self.device, dtype=torch.float32)
        if self.cfg.clip_actions is not None:
            actions = torch.clamp(actions, -self.cfg.clip_actions, self.cfg.clip_actions)
        # NaN 防护：策略输出若出现 NaN/Inf（数值爆炸）直接置 0 并夹到范围内
        actions = torch.nan_to_num(actions, nan=0.0, posinf=float(self.cfg.clip_actions or 8.0), neginf=-float(self.cfg.clip_actions or 8.0))
        self._raw_actions[:] = actions
        # ---- 鲁棒性：动作延迟（模拟信号/总线延迟；每 env 延迟在 reset 时随机 0~max） ---- #
        if self._action_delay_max > 0 and self._action_history is not None:
            self._history_index = (self._history_index + 1) % (self._action_delay_max + 1)
            self._action_history[:, self._history_index] = actions
            pick = (self._history_index - self._action_delay) % (self._action_delay_max + 1)
            env_index = torch.arange(self.num_envs, device=self.device)
            actions = self._action_history[env_index, pick]
        # 开局"收腿稳定"阶段标记（在 step_counter 自增之前计算）
        self._settling = self._step_counter < self._settle_steps

        if self._use_hip_actions:
            if self._mask_hip_actions:
                # 阶段1：屏蔽髋输出——忽略前 4 维动作，髋目标固定 squat（由 PD 保持）
                hip_targets = self._squat_hips.unsqueeze(0).repeat(self.num_envs, 1)
                self._hip_torques = torch.zeros_like(self._hip_torques)
            elif self._hip_control == "torque":
                # 髋力矩模式：输出 Nm，夹到髋力矩上限；不写位置目标
                hip_torques = torch.clamp(
                    self.cfg.hip_torque_scale * actions[:, : self._num_hips],
                    -HIP_EFFORT_LIMIT,
                    HIP_EFFORT_LIMIT,
                )
                self._hip_torques = hip_torques
                hip_targets = self._squat_hips.unsqueeze(0).repeat(self.num_envs, 1)
            else:
                hip_targets = self._squat_hips.unsqueeze(0) + self.cfg.hip_scale * actions[:, : self._num_hips]
                hip_targets = torch.maximum(torch.minimum(hip_targets, self._hip_hi), self._hip_lo)
        else:
            # 髋关节无输出：目标仅记录（不写入 PhysX），保持 squat 姿态参考
            hip_targets = self._squat_hips.unsqueeze(0).repeat(self.num_envs, 1)

        wheel_actions = actions[:, self._num_hip_actions :]
        if self._wheel_control == "velocity":
            wheel_targets = torch.clamp(
                self.cfg.wheel_scale * wheel_actions, -WHEEL_VELOCITY_LIMIT, WHEEL_VELOCITY_LIMIT
            )
            self._wheel_targets = wheel_targets
        else:
            # 力矩模式：输出 Nm，夹到轮力矩上限
            wheel_torques = torch.clamp(
                self.cfg.wheel_torque_scale * wheel_actions, -WHEEL_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT
            )
            # "轮子无力"阶段：每个 episode 前 wheel_unpowered_steps 个控制步强制 0 力矩（自然倒地）
            if self._wheel_unpowered_steps > 0:
                unpowered = self._step_counter < self._wheel_unpowered_steps
                wheel_torques = torch.where(
                    unpowered.unsqueeze(-1), torch.zeros_like(wheel_torques), wheel_torques
                )
            # 开局收腿稳定阶段：轮子也无力（先稳在最短腿长姿态，不滚动）
            if self._settle_steps > 0:
                wheel_torques = torch.where(
                    self._settling.unsqueeze(-1), torch.zeros_like(wheel_torques), wheel_torques
                )
            self._wheel_torques = wheel_torques
            wheel_targets = wheel_torques  # processed_actions 记录实际施加的力矩

        self._step_counter += 1
        self._hip_targets = hip_targets
        self._wheel_targets = wheel_targets
        if self._use_hip_actions:
            if self._hip_control == "torque":
                self._processed_actions = torch.cat([self._hip_torques, wheel_targets], dim=-1)
            else:
                self._processed_actions = torch.cat([hip_targets, wheel_targets], dim=-1)
        else:
            self._processed_actions = wheel_targets.clone()

    def apply_actions(self) -> None:
        """解五连杆闭链得到 knee 目标，并写入 hip/knee 位置目标与轮速目标。"""
        # 用实测 hip 角解算 knee（闭链约束），warm start 上一帧解
        measured_hips = self._asset.data.joint_pos[:, self._hip_ids]
        measured_hips = torch.nan_to_num(measured_hips, nan=0.0, posinf=1.0e9, neginf=-1.0e9)
        knees, residual = self._solver.solve(measured_hips, self._prev_knees)
        knees = torch.nan_to_num(knees, nan=0.0, posinf=1.0e9, neginf=-1.0e9)  # NaN 防护
        self._prev_knees = knees.detach()
        self._closure_residual = residual.detach()

        margin = float(self.cfg.knee_margin)
        knee_targets = torch.maximum(
            torch.minimum(knees, self._knee_hi - margin), self._knee_lo + margin
        ).detach()
        self._knee_targets = torch.nan_to_num(
            knee_targets, nan=0.0, posinf=1.0e9, neginf=-1.0e9
        )

        self._asset.set_joint_position_target(self._knee_targets, joint_ids=self._knee_ids)
        if self._wheel_control == "velocity":
            self._asset.set_joint_velocity_target(self._wheel_targets, joint_ids=self._wheel_ids)
        else:
            # 力矩模式：隐式执行器 stiffness=damping=0 时，effort target 直接作为 PhysX actuation force
            torques = self._wheel_torques
            if self._assist_enabled:
                # 叠加 PD 平衡辅助（可退火）→ 早期帮策略"从倒地起立"，后期撤掉
                torques = self._apply_balance_assist()
            if self._settle_steps > 0:
                # 开局收腿稳定阶段：轮子不给力矩（already 0），也不给辅助
                torques = torch.where(
                    self._settling.unsqueeze(-1), torch.zeros_like(torques), torques
                )
            # 鲁棒性：力矩缩放 + 一阶低通（模拟电机响应滞后/力矩误差）
            torques = self._shape_torque(torques, self._wheel_tau_filt)
            self._wheel_tau_filt = torques.detach()
            self._asset.set_joint_effort_target(torques, joint_ids=self._wheel_ids)
        # 髋关节写目标的条件：
        # - mask_hip_actions=True（阶段1"只学轮平衡"）：忽略策略髋动作，髋位置目标固定 squat（PD 保持）；
        # - hip_control='torque'（use_hip_actions=True）：写髋力矩目标（纯力矩控制，无 PD）；
        #   其中开局 settle_steps 步用 PD 力矩把 4 个 hip 稳定在"收腿姿态"，再交给策略；
        # - hip_control='position' 且 use_hip_actions=True：策略输出 hip 位置偏移；
        # - use_hip_actions=False 且 lock_hips=True：锁死在 squat 姿态（PD 保持，策略不管）；
        # - 两者都 False：不写目标（髋执行器须配零刚度/零阻尼 → 无力被动）。
        if self._use_hip_actions and self._mask_hip_actions:
            self._asset.set_joint_position_target(self._hip_targets, joint_ids=self._hip_ids)
        elif self._use_hip_actions and self._hip_control == "torque":
            hip_effort = self._hip_torques
            if self._settle_steps > 0 and bool(self._settling.any()):
                q = self._asset.data.joint_pos[:, self._hip_ids]
                qd = self._asset.data.joint_vel[:, self._hip_ids]
                settle_tau = torch.clamp(
                    self._settle_kp * (self._settle_hips.unsqueeze(0) - q) - self._settle_kd * qd,
                    -HIP_EFFORT_LIMIT,
                    HIP_EFFORT_LIMIT,
                )
                hip_effort = torch.where(
                    self._settling.unsqueeze(-1), settle_tau, self._hip_torques
                )
            # 鲁棒性：力矩缩放 + 一阶低通（模拟电机响应滞后/力矩误差）
            hip_effort = self._shape_torque(hip_effort, self._hip_tau_filt)
            self._hip_tau_filt = hip_effort.detach()
            self._asset.set_joint_effort_target(hip_effort, joint_ids=self._hip_ids)
        elif self._use_hip_actions or self.cfg.lock_hips:
            self._asset.set_joint_position_target(self._hip_targets, joint_ids=self._hip_ids)

    def _apply_balance_assist(self) -> torch.Tensor:
        """在策略轮力矩上叠加 PD 平衡辅助，返回总力矩（夹到 ±WHEEL_EFFORT_LIMIT）。

        ``tau_assist = clamp(+(kp*pitch + kd*pitch_rate), ±limit)``（左右轮相同），
        总力矩 = clamp(tau_policy + scale * tau_assist, ±limit)。
        ``scale`` 随全局控制步从 ``assist_scale_start`` **线性退火**到 ``assist_scale_end``；
        ``assist_apply_after_warmup=True`` 时，前 ``wheel_unpowered_steps`` 步（自然倒地阶段）
        不施加辅助，保证"自然倒地"过程不变。
        """
        # ---- 退火（按全局控制步数，不随 episode 重置） ----
        if self._assist_cycle_steps > 0:
            # 分阶段课程：每个阶段内 1→0 退火；前 assist_phases 个阶段有辅助，之后为 0
            cycle = int(self._assist_cycle_steps)
            phase = self._global_steps // cycle
            if phase >= int(self._assist_phases):
                self._assist_scale = self._assist_end
            else:
                frac = (self._global_steps % cycle) / float(cycle)
                self._assist_scale = self._assist_start + (self._assist_end - self._assist_start) * frac
        elif self._assist_anneal_steps > 0:
            frac = min(1.0, self._global_steps / float(self._assist_anneal_steps))
            self._assist_scale = self._assist_start + (self._assist_end - self._assist_start) * frac
        else:
            self._assist_scale = self._assist_start
        self._global_steps += 1

        # ---- PD 辅助力矩 ----
        # 注意符号：实测"正确的平衡方向"是 tau = +(kp*pitch + kd*pitch_rate)
        # （符号取反会让机体保持在 17° 倾斜不动，见 tools 可行性实验）。
        g_x = torch.clamp(self._asset.data.projected_gravity_b[:, 0], -1.0, 1.0)
        pitch = torch.asin(g_x)
        pitch_rate = self._asset.data.root_ang_vel_b[:, 1]
        tau = torch.clamp(
            self._assist_kp * pitch + self._assist_kd * pitch_rate,
            -WHEEL_EFFORT_LIMIT,
            WHEEL_EFFORT_LIMIT,
        ).unsqueeze(-1).expand(-1, self._num_wheels) * self._assist_scale

        # 自然倒地阶段不施加辅助
        if self.cfg.assist_apply_after_warmup and self._wheel_unpowered_steps > 0:
            active = (self._step_counter > self._wheel_unpowered_steps).unsqueeze(-1).float()
            tau = tau * active

        self._assist_torques = tau.detach()
        total = self._wheel_torques + tau
        return torch.clamp(total, -WHEEL_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT)

    def _shape_torque(self, cmd: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
        """鲁棒性：力矩缩放（每 env 随机）+ 一阶低通滤波（每 env 随机时间常数）。

        - ``torque_scale_range`` 模拟电机力矩标定误差（如 ±10%）；
        - ``torque_lpf_tau_range`` 模拟电机/驱动响应滞后：``filt += dt/tau * (cmd - filt)``。
        """
        if self._torque_scale_range != (1.0, 1.0):
            cmd = cmd * self._torque_scale
        if float(self._torque_lpf_tau.max()) > 0.0:
            alpha = self._dt / (self._torque_lpf_tau + 1.0e-6)
            cmd = filt + alpha * (cmd - filt)
        return cmd

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """重置时把闭链解、目标恢复成 squat 姿态（事件会再把关节状态写成一致值）。"""
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return

        if self._use_hip_actions:
            self._hip_targets[ids] = self._squat_hips
        self._hip_torques[ids] = 0.0
        self._wheel_targets[ids] = 0.0
        self._wheel_torques[ids] = 0.0
        self._step_counter[ids] = 0
        self._settling[ids] = False
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0

        # 鲁棒性随机量：每 episode 重新采样（动作延迟/力矩缩放/低通时间常数）
        n_ids = ids.numel()
        if self._action_delay_max > 0:
            self._action_delay[ids] = torch.randint(
                0, self._action_delay_max + 1, (n_ids,), device=self.device
            )
            if self._action_history is not None:
                self._action_history[ids] = 0.0
        scale_lo, scale_hi = self._torque_scale_range
        if scale_lo != 1.0 or scale_hi != 1.0:
            self._torque_scale[ids] = scale_lo + (scale_hi - scale_lo) * torch.rand(
                n_ids, 1, device=self.device
            )
        tau_lo, tau_hi = self._torque_lpf_tau_range
        if tau_hi > 0.0:
            self._torque_lpf_tau[ids] = tau_lo + (tau_hi - tau_lo) * torch.rand(
                n_ids, 1, device=self.device
            )
        self._hip_tau_filt[ids] = 0.0
        self._wheel_tau_filt[ids] = 0.0

        # 用重置后的 hip 实测值重新解一次 knee（事件先把 hip 写好了，顺序由 env 保证）
        measured_hips = self._asset.data.joint_pos[ids][:, self._hip_ids]
        prev = self._squat_knees.unsqueeze(0).repeat(ids.numel(), 1)
        knees, residual = self._solver.solve(measured_hips, prev)
        self._prev_knees[ids] = knees.detach()
        self._closure_residual[ids] = residual.detach()
        self._knee_targets[ids] = torch.maximum(
            torch.minimum(knees, self._knee_hi[ids] - self.cfg.knee_margin),
            self._knee_lo[ids] + self.cfg.knee_margin,
        ).detach()


@configclass
class ChassisWheelLegActionCfg(ActionTermCfg):
    """chassis-rl 轮腿动作项配置（见 :class:`ChassisWheelLegAction`）。"""

    class_type: type = ChassisWheelLegAction

    asset_name: str = "robot"
    """场景里机器人资产的名称。"""

    hip_joint_names: list[str] = list(HIP_ORDER)
    """4 个 hip 关节名（顺序即动作顺序）。"""

    knee_joint_names: list[str] = list(KNEE_ORDER)
    """4 个 knee 关节名（闭链随动，不作为策略动作）。"""

    wheel_joint_names: list[str] = list(WHEEL_ORDER)
    """2 个轮关节名（顺序即动作顺序）。"""

    hip_scale: float = HIP_ACTION_SCALE
    """hip 位置动作缩放 [rad / action unit]（hip_control='position' 时有效）。"""

    hip_control: str = "position"
    """髋关节控制模式：

    - ``'position'``（默认）：动作是 4 个 hip 位置偏移（× hip_scale）；
    - ``'torque'``：动作是 4 个 hip 力矩 [Nm]（× hip_torque_scale），
      要求 ``use_hip_actions=True`` 且 hips 执行器 stiffness=damping=0（纯力矩）。
    """

    hip_torque_scale: float = 5.0
    """力矩模式下髋力矩缩放 [Nm / action unit]（最终夹到 ±HIP_EFFORT_LIMIT=40 Nm）。"""

    wheel_scale: float = WHEEL_ACTION_SCALE
    """轮速动作缩放 [(rad/s) / action unit]。"""

    knee_margin: float = 0.01
    """knee 目标相对关节限位保留的余量 [rad]。"""

    clip_actions: float | None = 6.0
    """动作裁剪（None 表示不裁剪；训练器里的 clip_actions 优先）。"""

    use_hip_actions: bool = True
    """是否把 4 个 hip 位置偏移作为策略动作。

    False = "髋关节无输出/无力"：动作只有 2 个轮，hip 关节由零刚度/零阻尼执行器
    （见 env cfg 里的 actuators 覆盖）保持无力被动，不写 hip 位置目标。
    """

    wheel_control: str = "velocity"
    """轮子控制模式："velocity"（rad/s 速度目标，默认）或 "torque"（Nm 力矩目标）。"""

    wheel_torque_scale: float = 1.0
    """力矩模式下轮力矩缩放 [Nm / action unit]（最终夹到 ±WHEEL_EFFORT_LIMIT）。"""

    wheel_unpowered_steps: int = 0
    """每个 episode 开始的前 N 个控制步强制轮子 0 力矩（"轮子无力"自然倒地阶段）；0 = 禁用。"""

    lock_hips: bool = False
    """use_hip_actions=False 时是否把 4 个 hip **锁死**在 squat 姿态（PD 保持）。

    True  = 髋关节位置保持 squat（需要 hips 执行器有正常 stiffness/damping）；
    False = 不写髋目标（髋执行器应为零刚度/零阻尼 → 无力被动）。
    """

    mask_hip_actions: bool = False
    """**屏蔽髋关节输出**（阶段1"只学轮平衡"用）。

    True 时动作维度仍是 6（与放开髋的阶段2 接口一致，方便 --resume），但前 4 维髋动作
    **被忽略**：髋由执行器 PD 保持在 squat 姿态（需要 hips 有正常 stiffness/damping），
    策略实际只通过后 2 维轮力矩控制机器人。
    """

    # ---- 开局"收腿稳定"阶段（倒地起始：先用髋 PD 力矩把腿保持在最短姿态，N 步后再交给策略）---- #
    settle_steps: int = 0
    """每个 episode 开头用髋 PD 力矩把 hip 稳定在 ``settle_hip_positions`` 的控制步数（0 = 禁用）。

    用于"倒地状态下先把腿长收到最短、稳住，再开始起立"：这些步里髋关节由 PD 力矩保持收腿姿态，
    轮子无力、也不施加平衡辅助；之后正常由策略力矩控制。
    """

    settle_hip_positions: list[float] = [0.0, 0.0, 0.0, 0.0]
    """收腿稳定姿态（按 ``hip_joint_names`` 顺序的绝对角，rad）；默认 q=0 = 腿长最短（机械收回端）。"""

    settle_hip_kp: float = 300.0
    """收腿稳定阶段的髋 PD 比例增益 [Nm/rad]。"""

    settle_hip_kd: float = 10.0
    """收腿稳定阶段的髋 PD 微分增益 [Nm/(rad/s)]。"""

    squat_pose_path: str | None = None
    """squat 姿态 JSON 路径（None = 使用 chassis_rl_lab/data/squat_pose.json）。"""

    # ---- 平衡辅助（PD 辅助力矩 + 退火；仅力矩模式有效，用于"扶着学起立"） ---- #
    balance_assist: bool = False
    """是否在策略轮力矩上叠加 PD 平衡辅助力矩（默认关；起立任务可开）。"""

    assist_kp: float = 10.0
    """辅助 PD 的比例增益 [Nm / rad]（实测 10 能从 17° 倾斜 0.5 s 拉回 ~3°）。"""

    assist_kd: float = 1.0
    """辅助 PD 的微分增益 [Nm / (rad/s)]。"""

    assist_scale_start: float = 1.0
    """辅助强度初始比例（0~1；1 = 辅助 PD 全量叠加）。"""

    assist_scale_end: float = 0.0
    """辅助强度退火终值（0 = 训练后期完全撤掉辅助，策略自己扛）。"""

    assist_anneal_steps: int = 0
    """退火长度 [全局控制步]（0 = 不退火，恒为 start；48 步/iter ⇒ 600 iter ≈ 28800）。"""

    assist_cycle_steps: int = 0
    """**分阶段课程退火**：>0 时按"每 ``assist_cycle_steps`` 步一个阶段"循环退火。

    每个阶段内辅助强度从 ``assist_scale_start`` 线性退到 ``assist_scale_end``；
    前 ``assist_phases`` 个阶段有辅助，之后的阶段辅助强度恒为 ``assist_scale_end``（通常 0）。
    （用于"前 300 iter 正方向倒地+退火、301~600 iter 负方向倒地+退火、之后无辅助"的课程。）
    """

    assist_phases: int = 2
    """有辅助的阶段数（配合 ``assist_cycle_steps``；之后阶段辅助为 0）。"""

    assist_apply_after_warmup: bool = True
    """True = 前 wheel_unpowered_steps 步（自然倒地阶段）不施加辅助，保持自然倒地过程。"""

    # ---- 鲁棒性：动作延迟 / 力矩缩放 / 力矩一阶低通（模拟电机与信号链非理想特性）---- #
    action_delay_max: int = 0
    """动作延迟最大步数（控制步）：每 env 在 reset 时随机 0~该值；0 = 无延迟。
    模拟真机的总线/控制周期延迟（1~2 步 = 20~40 ms）。"""

    torque_scale_range: tuple[float, float] = (1.0, 1.0)
    """执行器力矩缩放范围（每 env 在 reset 时随机），模拟电机力矩误差/标定误差。"""

    torque_lpf_tau_range: tuple[float, float] = (0.0, 0.0)
    """力矩一阶低通时间常数范围 [s]（每 env 在 reset 时随机），模拟电机/驱动响应滞后。"""


__all__ = ["ChassisWheelLegAction", "ChassisWheelLegActionCfg"]
