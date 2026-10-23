"""Locomotion (velocity-command tracking) task for the infantry-rl five-bar robot.

This task turns the *balance* environment into a drivable one.  The difference is
where the target comes from:

* balance task -- a fixed target (heading 0, position 0): "stand there and do not
  move";
* this task -- a **velocity command** ``(vx, wz)`` produced by a command term: "drive
  forward at vx and turn at wz".

Everything else (robot, loop-closure action term, PhysX side) is shared with the
balance task; only the observation layout and the reward set differ.

Why ``(vx, wz)`` and not ``(vx, vy, wz)``
----------------------------------------
The robot is a two-wheel differential drive: forward/backward comes from the mean
wheel speed and turning from the difference.  There is **no lateral actuator**, so a
``vy`` command could never be tracked; it is pinned to 0 and dropped from the
observation instead of feeding the policy a constant dimension.

Why ``heading_command=True``
----------------------------
With heading commands the manager converts the heading error into the yaw-rate
command every control step::

    wz_cmd = clip(heading_control_stiffness * wrap(heading_target - heading))

So "hold the current direction" and "turn to a new direction" become the same
mechanism, and the policy only ever has to track a yaw rate.  This is what makes
in-place rotation (``vx = 0, heading_target moving``) and turning while driving
(``vx != 0``) the same skill.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaac_ws_infantry.mdp as mdp
from isaac_ws_infantry.robots import INFANTRY_RL_CFG
from isaac_ws_infantry.tasks.common.robot_env_cfg import (
    FALL_ANGLE,
    FALL_HEIGHT,
    NOMINAL_HEIGHT,
    ActionsCfg,
    RobotSceneCfg,
    TerminationsCfg,
)


def _env_float(name: str, default: float) -> float:
    """读取浮点环境变量；未设置（或为空）时返回默认值。"""
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else float(default)


def _env_flag(name: str, default: str = "0") -> bool:
    """读取布尔开关环境变量（1 / true / yes / on）。"""
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# ============================================================================
# ★★★ 速度相关常数 —— 决定「速度怎么换算成动作」★★★
# ============================================================================
# 这一块和下面的 TRAIN_MODE 一起，构成训练目标的两个输入：
#   1) 这里：机器人有多快（轮半径、动作量程）—— 一般不用改
#   2) TRAIN_MODE：想练什么速度（6 种模式）—— 改这一个字符串
#
# 速度 → 动作的三条换算关系（都由下面的常数决定，改速度目标时自动重算）：
#
#   wheel_scale      = max(REF_MAX_VX, |vx|max) / (WHEEL_RADIUS × CLIP_ACTIONS × ACTION_HEADROOM)
#                      意义：策略输出 1.0 对应多少轮速（rad/s）
#                      作用：保证最高速指令一定落在动作量程内，否则机器人跑不到目标速度
#                            而且**不会报错**（策略被 clip 卡住，看起来只是"学不会"）
#
#   track_lin_std    = 0.125 × max(REF_MAX_VX, |vx|max)
#                      意义：速度跟踪奖励 exp 核的宽度（m/s）
#                      作用：核越窄，"多接近才算跟得上"的要求越严
#
#   wheel_spin_weight = -3e-5 × (1.2 / max(REF_MAX_VX, |vx|max))²
#                      意义：轮子空转惩罚权重
#                      作用：合法轮速² 随速度变大而变大，权重必须反向缩放
#
# 关键：三者的公式都用 REF_MAX_VX（而不是当前模式的最高速）打底，
# 所以只要各模式的速度目标都不超过 REF_MAX_VX，动作尺度就完全相同，
# 模式之间可以互相 --resume 继续训练（实测都是 5.013 / 0.150 / -3.0e-05）。
# 只有把 vx 目标设到超过 REF_MAX_VX（比如要跑 2.5 m/s）才会整体放大、旧 checkpoint 作废。
# ----------------------------------------------------------------------------

#: 动作尺度的参考速度（m/s）。各模式的速度目标不超过它 -> wheel_scale 恒定
REF_MAX_VX = 1.2
#: 轮半径（m）。与 isaac_ws_infantry.robot_model.WHEEL_RADIUS 必须一致
WHEEL_RADIUS = 0.068
#: agent cfg 里的 clip_actions：策略输出每维最大到 ±这个值
CLIP_ACTIONS = 4.0
#: 最高指令只占动作满量程的比例，留 (1-这个值) 的余量给加速/纠偏
ACTION_HEADROOM = 0.88

# ----------------------------------------------------------------------------
# 由上面四个常数算出的实际可达上限，方便核对（不需要改）：
#   动作量程上限 = CLIP_ACTIONS × wheel_scale × WHEEL_RADIUS
#                = REF_MAX_VX / ACTION_HEADROOM = 1.36 m/s
#   也就是说 vx 目标只要 ≤ 1.2，就一定跑得到。
# ----------------------------------------------------------------------------


# ============================================================================
# ★★★ 训练目标（任务模式）—— 整个训练只改下面这一个字符串 ★★★
# ============================================================================
#
# 可选值（只有这 6 个，写错会直接报错并列出可选项）：
#
#   "straight"   直线速度跟踪                                        ← 默认值
#                训练什么 : 在直线上把速度跟准，航向保持 0
#                vx 范围  : [+0.2, +1.2] m/s（只前进）
#                航向目标 : 锁定 0。航向环仍在跑，负责抵消漂移，但不会主动转向
#                指令保持 : 5~10 s 换一条速度指令
#                站立环境 : 10%（唯一的学习"刹车"的样本）
#                主导奖励 : track_lin_vel_x(+8.0) / track_lin_vel_x_l1/l2
#                什么时候用: 第一次训练就选它；线性速度要从这里练起
#
#   "turning"    转弯 / 转向控制
#                训练什么 : 转到指定航向，同时保持速度跟踪
#                vx 范围  : [+0.2, +1.0] m/s（仍然发速度指令 → 直线能力不丢）
#                航向目标 : 每 3~6 s 在 [-pi, pi] 随机采一个新目标
#                指令保持 : 3~6 s（换向比直线模式更频繁）
#                站立环境 : 10%（其余 90% 照常转弯，不影响学习转向）
#                主导奖励 : track_ang_vel_z(+3.0) / heading_error(-0.6)
#                           —— 奖励权重不用改，跟踪类奖励是"跟什么就奖什么"
#                什么时候用: 直线练完之后；可从直线 checkpoint 直接 --resume
#
#   "reverse"    前进 + 倒车
#                训练什么 : 前进和倒车都以 1.2 m/s 跟准，航向保持 0
#                vx 范围  : [-1.2, +1.2] m/s（负值 = 倒车）
#                航向目标 : 锁定 0
#                指令保持 : 5~10 s
#                站立环境 : 10%
#                主导奖励 : track_lin_vel_x(+8.0)（对符号不敏感，正负一视同仁）
#                什么时候用: 需要倒车能力时；可从直线 checkpoint 续训
#
#   "spin"       原地旋转到指定航向（位置式转向）
#                训练什么 : 车身不动，转到指令航向；wz 由航向环按角度差算出来
#                vx 范围  : [0, 0]（完全不给前进速度 → 只能原地转）
#                航向目标 : 每 2~4 s 在 [-pi, pi] 随机采一个新目标
#                航向环   : heading_command=True，增益 2.0（比直线/转弯更硬）
#                wz 钳位  : ±2.0 rad/s
#                站立环境 : 0%（站立环境会把 wz 也清零，那就没旋转指令了）
#                主导奖励 : track_ang_vel_z(+3.0) / heading_error(-0.6)
#                什么时候用: 转弯练完、想专门补"原地转到位"这个能力
#
#   "yaw_rate"   原地按指令角速度旋转（速度式转向）
#                训练什么 : 车身不动，以指令的角速度连续旋转（可以连转多圈）
#                vx 范围  : [0, 0]
#                角速度   : wz 直接采样 ±2.0 rad/s（heading_command=False，不走航向环）
#                指令保持 : 每 2~4 s 换一次目标角速度
#                站立环境 : 0%（同上）
#                主导奖励 : track_ang_vel_z(+3.0)
#                什么时候用: 需要"匀速转、转到哪算哪"的能力时
#                和 spin 的区别: spin 是"转到某个角度"，这是"以某个角速度转"
#
#   "stand"      原地站住 / 平衡
#                训练什么 : 不动，只保持平衡和航向（学"站着别抖"）
#                vx 范围  : [0, 0]（不下发速度指令）
#                航向目标 : 锁定 0
#                指令保持 : 5~10 s
#                站立环境 : 100%（全部环境都在学站住）
#                主导奖励 : alive(+0.5) / pitch(-4.0) / roll / base_height
#                什么时候用: 调试平衡、查抖动；或作为后续任务的热启动
#
# ----------------------------------------------------------------------------
# 改这一个值之后，下面这些**全部自动跟着变**，不需要改别的地方：
#   * 速度指令范围（vx 上下界）
#   * 航向目标采样范围（锁定 0 还是 [-pi, pi] 随机）
#   * wz 是走航向环（heading_command=True）还是直接采样（False）
#   * wz 钳位 / 采样范围（默认 ±2.5，spin/yaw_rate 用 ±2.0）
#   * 指令重采样间隔（5~10 s 还是 3~6 s）
#   * 站立环境比例（10% / 10% / 0% / 0% / 10% / 100%）
#   * wheel_scale          —— 动作尺度，按固定 reference_max_vx=1.2 反推
#   * track_lin_vel_x 的 std —— 跟踪核宽度，同上
#   * wheel_spin 权重       —— 轮子空转惩罚，同上
#
# 因为最后三项都按固定的 REF_MAX_VX 算，**六种模式的动作语义完全一致**
# （实测都是 wheel_scale=5.013 / std=0.150 / wheel_spin=-3.0e-05），
# 所以模式之间可以互相 `--resume` 继续训练，不会互相破坏。
#
# 想加新模式：在下面 _MODE_PRESETS 里照抄一段改数字即可，键名就是可选值。
# 想改某个模式的数字：直接改 _MODE_PRESETS 里对应的 dict（不会影响别的模式）。
# ----------------------------------------------------------------------------
TRAIN_MODE = "straight"  # 可选：straight / turning / spin / yaw_rate / reverse / stand

# ============================================================================
# ★★★ 速度指令表 —— 所有速度数字只在这张表里出现 ★★★
# ============================================================================
# 每行 = 一种模式，所有速度量都显式写出来，方便一眼对比、直接改。
# 字段含义与 LocomotionTargets 一一对应（数据类里**没有默认值**，数字只有这里一份）：
#
#   vx_min/vx_max        前进速度范围（m/s），负值=倒车
#   vy_min/vy_max        侧向速度范围（m/s），差速轮无侧向执行器 → 恒为 0
#   wz_min/wz_max        偏航角速度范围（rad/s）
#                          heading_command=True  -> 这是航向环输出的「钳位」
#                          heading_command=False -> 这是直接采样的「范围」
#   enable_turning       航向目标是否随机采样 [-pi, pi]（False = 锁定 0）
#   heading_command      True = wz 由航向差算出（闭环转到目标角）
#                        False = wz 直接采样（按指令角速度转）
#   heading_stiffness    航向误差 → 偏航角速度的增益（1/s）
#   resample_time_s      指令保持时间（s），到点换一条新指令
#   rel_standing_envs    站立环境比例（整条指令被清零，用于学刹车）
#   rel_heading_envs     航向环境比例（heading_command=True 时通常 1.0）
# ----------------------------------------------------------------------------
#             模式名        vx (m/s)        vy       wz (rad/s)   转弯  航向环  增益   保持(s)      站立   航向环境
_MODE_PRESETS: dict[str, dict] = {
    "straight": dict(
        vx_min=0.2,   vx_max=1.2,   vy_min=0.0, vy_max=0.0, wz_min=-2.5, wz_max=2.5,
        enable_turning=False, heading_command=True,  heading_stiffness=1.5,
        resample_time_s=(5.0, 10.0), rel_standing_envs=0.1, rel_heading_envs=1.0, rel_spin_envs=0.0,
    ),
    "turning": dict(
        # 仍然下发 vx 指令 -> 直线速度跟踪照旧生效，直线能力不会丢
        vx_min=0.2,   vx_max=1.0,   vy_min=0.0, vy_max=0.0, wz_min=-2.5, wz_max=2.5,
        enable_turning=True,  heading_command=True,  heading_stiffness=1.5,
        resample_time_s=(3.0, 6.0),  rel_standing_envs=0.1, rel_heading_envs=1.0, rel_spin_envs=0.0,
    ),
    "spin": dict(
        # 原地转到指定航向：vx=0，wz 由航向环按角度差算出
        vx_min=0.0,   vx_max=0.0,   vy_min=0.0, vy_max=0.0, wz_min=-2.0, wz_max=2.0,
        enable_turning=True,  heading_command=True,  heading_stiffness=2.0,
        resample_time_s=(2.0, 4.0),  rel_standing_envs=0.0, rel_heading_envs=1.0, rel_spin_envs=0.0,
    ),
    "yaw_rate": dict(
        # 原地按指令角速度连续旋转：vx=0，wz 直接采样，不走航向环
        vx_min=0.0,   vx_max=0.0,   vy_min=0.0, vy_max=0.0, wz_min=-2.0, wz_max=2.0,
        enable_turning=False, heading_command=False, heading_stiffness=1.5,
        resample_time_s=(2.0, 4.0),  rel_standing_envs=0.0, rel_heading_envs=1.0, rel_spin_envs=0.0,
    ),
    "reverse": dict(
        vx_min=-1.2,  vx_max=1.2,   vy_min=0.0, vy_max=0.0, wz_min=-2.5, wz_max=2.5,
        enable_turning=False, heading_command=True,  heading_stiffness=1.5,
        resample_time_s=(5.0, 10.0), rel_standing_envs=0.1, rel_heading_envs=1.0, rel_spin_envs=0.0,
    ),
    "stand": dict(
        # 不给速度指令，只学保持平衡和航向
        vx_min=0.0,   vx_max=0.0,   vy_min=0.0, vy_max=0.0, wz_min=-2.5, wz_max=2.5,
        enable_turning=False, heading_command=True,  heading_stiffness=1.5,
        resample_time_s=(5.0, 10.0), rel_standing_envs=1.0, rel_heading_envs=1.0, rel_spin_envs=0.0,
    ),
    "full": dict(
        # 全部技能合成一个任务：前进/倒车/边跑边转/原地转/站住
        vx_min=-1.2,  vx_max=1.2,   vy_min=0.0, vy_max=0.0, wz_min=-2.5, wz_max=2.5,
        enable_turning=True,  heading_command=True,  heading_stiffness=1.5,
        resample_time_s=(3.0, 6.0),  rel_standing_envs=0.1, rel_heading_envs=1.0, rel_spin_envs=0.2,
    ),
}

if TRAIN_MODE not in _MODE_PRESETS:
    raise ValueError(f"TRAIN_MODE={TRAIN_MODE!r} 不认识；可选：{sorted(_MODE_PRESETS)}")
_MODE = _MODE_PRESETS[TRAIN_MODE]

# ============================================================================
# 下面由 TRAIN_MODE + _MODE_PRESETS 自动生成，**不需要手动改**
# ============================================================================

# 环境变量（进程级，只对当前 cmd 窗口有效；优先级高于 _MODE_PRESETS）
#   set INFANTRY_RL_VX_MIN=0.2          速度下界（m/s；负值=倒车）
#   set INFANTRY_RL_VX_MAX=1.2          速度上界（m/s）
#   set INFANTRY_RL_ENABLE_TURNING=1    强制打开随机航向目标
#
# 硬件依据：轮速上限 61.471 rad/s x 0.068 m = 4.18 m/s；轮推力 2x3.69/0.068 = 108 N
# -> 约 8.5 m/s^2，轮胎抓地约 7.8 m/s^2。1.2 m/s 只用到轮速上限的约 29%，留足余量。
_SPEED = dict(_MODE)
_SPEED["vx_min"] = _env_float("INFANTRY_RL_VX_MIN", _SPEED["vx_min"])
_SPEED["vx_max"] = _env_float("INFANTRY_RL_VX_MAX", _SPEED["vx_max"])
if _env_flag("INFANTRY_RL_ENABLE_TURNING"):
    _SPEED["enable_turning"] = True

@dataclass(frozen=True)
class LocomotionTargets:
    """速度目标的只读容器 + 派生公式。

    这里**故意不给任何字段默认值** —— 所有速度数字只有文件开头的
    ``_MODE_PRESETS`` 那一张表一处来源。本类只负责：
      1. 装下当前模式选中的那组数字；
      2. 提供派生的 ``wheel_scale`` / ``track_lin_std`` / ``wheel_spin_weight``。

    想改速度：改 ``TRAIN_MODE`` 或 ``_MODE_PRESETS``，**不要动这个类**。
    """

    # -- 速度指令（由 _MODE_PRESETS 传入，此处不写默认值）--------------------
    vx_min: float            # 前进速度下界（m/s）
    vx_max: float            # 前进速度上界（m/s）
    vy_min: float            # 侧向速度下界（m/s），差速轮恒 0
    vy_max: float            # 侧向速度上界（m/s）
    wz_min: float            # 偏航角速度下界（rad/s）
    wz_max: float            # 偏航角速度上界（rad/s）
    enable_turning: bool     # 航向目标是否随机采样 [-pi, pi]
    heading_command: bool    # True=航向环算 wz，False=直接采样 wz
    heading_stiffness: float # 航向误差 -> 偏航角速度增益（1/s）
    resample_time_s: tuple[float, float]  # 指令保持时间（s）
    rel_standing_envs: float # 站立环境比例
    rel_heading_envs: float  # 航向环境比例
    rel_spin_envs: float     # 原地旋转环境比例（vx 强制 0，保留 wz）

    # -- 当前模式名（只用于打印/记录）---------------------------------------
    mode: str

    # -- 机器人常数（来自文件最开头的常数块，这里不写数字）-------------------
    wheel_radius: float      # 轮半径（m）
    clip_actions: float      # 策略输出每维上限
    action_headroom: float   # 最高指令占动作量程的比例
    reference_max_vx: float  # 动作尺度的参考速度（m/s）

    # ========================================================================
    # 派生量（不需要改，改 TRAIN_MODE / _MODE_PRESETS 后自动跟着变）
    # ========================================================================
    @property
    def lin_vel_x(self) -> tuple[float, float]:
        """前进速度范围，给 UniformVelocityCommandCfg.Ranges 用。"""
        return (self.vx_min, self.vx_max)

    @property
    def lin_vel_y(self) -> tuple[float, float]:
        return (self.vy_min, self.vy_max)

    @property
    def ang_vel_z(self) -> tuple[float, float]:
        return (self.wz_min, self.wz_max)

    @property
    def heading(self) -> tuple[float, float]:
        """航向目标采样范围：转弯/旋转模式随机，其余锁定 0。"""
        return (-math.pi, math.pi) if self.enable_turning else (0.0, 0.0)

    @property
    def max_abs_vx(self) -> float:
        """速度目标里最大的绝对值（m/s）；倒车更慢时也按绝对值算。"""
        return max(abs(self.vx_min), abs(self.vx_max))

    def _scale_v(self) -> float:
        """参与动作尺度换算的速度（m/s）：不低于 REF_MAX_VX。"""
        return max(self.reference_max_vx, self.max_abs_vx)

    @property
    def wheel_scale(self) -> float:
        """动作 1.0 对应多少轮速（rad/s）。

        按 ``max(reference_max_vx, max_abs_vx)`` 反推，保证最高速指令落在动作量程内::

            v_max = clip_actions * wheel_scale * wheel_radius * action_headroom

        * 速度目标 <= reference_max_vx（默认所有模式都是）-> wheel_scale 恒定，
          模式之间可以互相 --resume；
        * 速度目标超过它（比如要跑 2.5 m/s）-> 自动放大，旧 checkpoint 作废。
        """
        return self._scale_v() / (self.wheel_radius * self.clip_actions * self.action_headroom)

    @property
    def track_lin_std(self) -> float:
        """线速度跟踪 exp 核的宽度（m/s）= 0.125 x 换算速度。

        同样按 reference_max_vx 打底，所以各模式用的核宽度相同，resume 时奖励尺度不变。
        """
        return 0.125 * self._scale_v()

    @property
    def wheel_spin_weight(self) -> float:
        """轮子空转惩罚权重：按换算速度的平方反向缩放，保持相对惩罚不变。"""
        return -3.0e-5 * (REF_MAX_VX / self._scale_v()) ** 2


TARGETS = LocomotionTargets(
    mode=TRAIN_MODE,
    reference_max_vx=REF_MAX_VX,
    wheel_radius=WHEEL_RADIUS,
    clip_actions=CLIP_ACTIONS,
    action_headroom=ACTION_HEADROOM,
    **_SPEED,
)

# 兼容旧代码/外部脚本的模块级别名（值都来自 TARGETS，不要在别处重复定义）
VX_MIN_MPS = TARGETS.vx_min
VX_MAX_MPS = TARGETS.vx_max
LIN_VEL_X_RANGE = TARGETS.lin_vel_x
ANG_VEL_Z_RANGE = TARGETS.ang_vel_z
HEADING_STIFFNESS = TARGETS.heading_stiffness
REL_STANDING_ENVS = TARGETS.rel_standing_envs
_ENABLE_TURNING = TARGETS.enable_turning
HEADING_RANGE = TARGETS.heading
HEADING_COMMAND = TARGETS.heading_command


@configclass
class LocomotionCommandsCfg:
    """Velocity commands for the locomotion task（数值全部来自 TARGETS）。"""

    base_velocity = mdp.SpinAwareVelocityCommandCfg(
        asset_name="robot",
        heading_command=TARGETS.heading_command,
        heading_control_stiffness=TARGETS.heading_stiffness,
        rel_standing_envs=TARGETS.rel_standing_envs,
        rel_heading_envs=TARGETS.rel_heading_envs,
        rel_spin_envs=TARGETS.rel_spin_envs,
        # a new command every 5-10 s: long enough to actually track it, short enough
        # to see several commands per episode
        resampling_time_range=TARGETS.resample_time_s,
        debug_vis=False,
        ranges=mdp.SpinAwareVelocityCommandCfg.Ranges(
            lin_vel_x=TARGETS.lin_vel_x,
            lin_vel_y=TARGETS.lin_vel_y,  # fixed 0: not actuated on a differential drive
            ang_vel_z=TARGETS.ang_vel_z,
            heading=TARGETS.heading,
        ),
    )


@configclass
class LocomotionObservationsCfg:
    """Observation specifications for the MDP (35 dims)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """What the policy sees.

        Compared with the balance task, the fixed ``heading`` and ``position`` dims are
        replaced by the ``velocity_commands``: the target is no longer "stay at 0" but
        "track this (vx, wz)", so the command *is* the goal.

        ``leg_rates`` was added so the hips can be the attitude actuator.  The firmware
        controller this robot is based on (``infantry-chassis``: LQR + VMC) uses the
        state ``[theta, dtheta, xb, dxb, phi, dphi]``, i.e. it feeds back the leg-angle
        RATE explicitly (the ``Tp`` row of its gain matrix has a gain of ~5.9 on
        ``dphi``).  An MLP has no memory, so without ``dL0``/``dphi`` as inputs the legs
        can only do proportional feedback on their pose and will lag when they start
        carrying the pitch/roll loop.
        """

        # base state
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, scale=0.25)
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)

        # the goal: commanded forward speed and yaw rate
        velocity_commands = ObsTerm(
            func=mdp.velocity_commands, params={"command_name": "base_velocity"}, scale=1.0
        )

        # leg state, relative to the commanded / solved values
        hip_pos_rel_nominal = ObsTerm(func=mdp.hip_pos_rel_nominal, params={"action_name": "joints"})
        knee_pos_rel_solved = ObsTerm(func=mdp.knee_pos_rel_solved, params={"action_name": "joints"})
        wheel_vel = ObsTerm(func=mdp.wheel_vel, params={"action_name": "joints"}, scale=0.1)

        # last action and the solved leg length / angle
        last_action = ObsTerm(func=mdp.last_action_raw, params={"action_name": "joints"})
        solved_leg_state = ObsTerm(func=mdp.solved_leg_state, params={"action_name": "joints"})

        # leg-length / leg-angle rates (dL0_l, dphi_l, dL0_r, dphi_r):
        # the "dphi" of the reference LQR state, needed to damp the leg attitude loop
        leg_rates = ObsTerm(func=mdp.leg_rates, params={"action_name": "joints"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class LocomotionRewardsCfg:
    """Reward terms for the locomotion task.

    The objective, in the order the terms enforce it:

    1. **stay up** -- ``alive`` + ``fall_penalty``;
    2. **hold the stance** -- ``base_height`` (L1, strong) keeps the body at the
       measured nominal height instead of crouching, ``pitch`` (L1, strong) keeps the
       body level nose-up/nose-down, ``lin_vel_z`` stops bouncing;
    3. **walk upright** -- ``roll`` plus ``roll_when_straight_abs``.  A *permanent*
       sideways lean is not acceptable; the second term is the one that removes it,
       because it charges the full roll price only while the commanded yaw rate is
       near zero and lets roll go free when actually cornering;
    4. **track the command** -- the two exp kernels;
    5. **turn fast enough** -- ``wz`` is the dominant tracking term and the command
       range is wide, because a slow heading loop is what made the previous policy feel
       sluggish.

    Why a *small* amount of roll is still fine: a two-wheel robot has no lateral force
    source, so cornering needs a bank of ``atan(v*w/g)`` -- about 2.6 deg at the trained
    speed/turn-rate limits.  That is cheap under these weights, while the 10-20 deg lean
    the previous policy settled into is not.
    """

    # -- staying up ------------------------------------------------------
    alive = RewTerm(func=base_mdp.is_alive, weight=0.5)
    fall_penalty = RewTerm(func=mdp.fall_penalty, weight=-50.0)

    # -- hold the stance -------------------------------------------------
    # L1 terms: a constant pull from any distance, see mdp.rewards.pitch_abs /
    # base_height_abs for why the squared versions were not enough.
    #
    # base_height is deliberately relaxed to -4 (was -10): a leg-based balance strategy
    # HAS to change the stance -- leaning the leg by phi drops the body by
    # L0*(1-cos(phi)) (~1.5 cm at 20 deg) and lengthening/shortening the leg changes it
    # even more.  Penalising that at -10 pushes the policy straight back to "rigid legs".
    base_height = RewTerm(
        func=mdp.base_height_abs, weight=-4.0, params={"target_height": NOMINAL_HEIGHT}
    )
    # pitch stays the strict quantity: this is what the legs are supposed to regulate.
    pitch = RewTerm(func=mdp.pitch_abs, weight=-4.0)

    # -- walk upright, but bank freely while cornering --------------------
    # The previous -0.3 "token" weight was what let the policy settle into a permanent
    # sideways lean.  The pair below fixes that without blocking cornering: the global
    # term keeps a sane bound, the conditional one pays in full only when the commanded
    # yaw rate is ~0 (i.e. when driving straight).
    roll = RewTerm(func=mdp.roll_abs, weight=-1.5)
    roll_straight = RewTerm(
        func=mdp.roll_when_straight_abs,
        weight=-2.5,
        params={"command_name": "base_velocity", "wz_scale": TARGETS.wz_max},
    )
    roll_rate = RewTerm(func=mdp.ang_vel_x_l2, weight=-0.1)

    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    # -- tracking the command --------------------------------------------
    # WEIGHT REBALANCE (v1.0.2).  This used to be lin 2.0 / yaw 2.5, and that asymmetry
    # produced a "stand still and rotate" local optimum that 2000 iterations could not
    # escape -- measured on the 2026-10-21 run:
    #
    #     track_ang_vel_z  2.2185 / 2.5  = 89% of its ceiling
    #     track_lin_vel_x  0.3959 / 2.0  = 20%
    #     Policy/mean_std  0.497 -> 0.067  (exploration collapsed)
    #
    # Why a stationary robot scores so well on the YAW term: `wz*` is produced by the
    # heading controller and decays to 0 once the target heading is reached; a robot that
    # does not move has `wz = 0`, so `exp(0) = 1` -- it collects the full 2.5 for free,
    # without ever translating.  Standing still therefore paid ~0.5 + 2.5 + 0.47 = 2.7
    # against ~4.5 for actually walking, i.e. 60% of the score with none of the risk.
    #
    # After the rebalance the gap is ~3.06 per step instead of ~1.8, and translation --
    # the actual goal -- is the dominant term.
    # v1.0.7 REBALANCE (measured on the 2026-10-22 00:00 run, 650 iterations):
    #   error_vel_xy      = 1.204 m/s   (the command mean is only 0.7 m/s!)
    #   track_lin_vel_x   = 0.620 / 4.0 = 15.5%
    #   track_ang_vel_z   = 1.359 / 1.5 = 91%   <- free while standing still
    #   mean_episode_length = 1000 (never falls), mean_std = 0.031 (exploration dead)
    # i.e. the policy settled on "stand still and be safe".  Two causes:
    #
    #  (a) standing still still PAID: 0.5 alive + 1.5 yaw + 0.23 exp - 0.57 L2 = +1.66/step;
    #  (b) the exp kernel is dead at large errors: with cmd 1.2 and v 0,
    #      exp(-(1.2)^2/0.25^2) = exp(-23) ~ 0, so it supplies NO gradient to start moving.
    #
    # The fix raises the cost of being at the wrong speed (exp +6.0, L2 -3.0 -- the L2
    # gradient -2*3*e grows with the error, so it still pulls when the exp is saturated)
    # and lowers the free yaw reward (1.5 -> 1.0).  Standing still now pays ~+0.12/step
    # against ~7.5 for walking, instead of +1.66 vs 6.0.
    # v1.0.8 PRECISION: target is `Metrics/base_velocity/error_vel_xy` and `..._yaw`
    # below 0.05.  NOTE the scale of that metric: the command term accumulates
    # `error / (resampling_time_max/step_dt) = error/500` every step and only zeroes it at
    # the EPISODE reset, so over a 1000-step episode the logged number is
    # `sum(error)/500 = 2 * mean_error`.  A logged 0.05 therefore means a true mean error
    # of 0.025 m/s.  Getting there needs three things, hence the changes below:
    #   (a) a reward that still discriminates at a few cm/s -> narrower exp kernel;
    #   (b) a term that keeps pulling at cm/s errors -> the L1 terms (constant gradient);
    #   (c) less friction on the fast corrections -> relaxed action_jerk / wheel_torques.
    track_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_x_exp, weight=8.0, params={"command_name": "base_velocity", "std": 0.15}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.15}
    )
    # reaching the commanded heading, not merely matching the yaw rate
    heading_error = RewTerm(
        func=mdp.heading_error_l2, weight=-0.6, params={"command_name": "base_velocity"}
    )

    # -- steady-state stiffness the exp kernels cannot provide ------------
    # `exp(-e^2/std^2)` has ZERO gradient at e = 0, so small deviations oscillating at
    # the control rate are almost free: the policy is pushed to "roughly the commanded
    # speed" but not to a *steady* one.  Measured symptom: the robot advances in bursts
    # ("move, stop, move") and keeps a visible yaw wobble, because a two-wheel robot is
    # continuously making small steering corrections and nothing charged for them.
    #
    #   * L2 (gradient -2*w*e) grows with the error -> it is what breaks the "stand still"
    #     optimum and cleans up large errors (v1.0.7: -3.0);
    #   * L1 (gradient -w, CONSTANT) is the one that fixes the last few cm/s: at e = 0.02
    #     the L2 gradient is only 2*6*0.02 = 0.24, while L1 still pulls with its full 2.0.
    #     Without L1 the policy has no reason to go from 10 cm/s to 2 cm/s (v1.0.8).
    track_lin_vel_x_l1 = RewTerm(
        func=mdp.track_lin_vel_x_l1, weight=-2.0, params={"command_name": "base_velocity"}
    )
    track_ang_vel_z_l1 = RewTerm(
        func=mdp.track_ang_vel_z_l1, weight=-1.0, params={"command_name": "base_velocity"}
    )
    track_lin_vel_x_l2 = RewTerm(
        func=mdp.track_lin_vel_x_l2, weight=-6.0, params={"command_name": "base_velocity"}
    )
    track_ang_vel_z_l2 = RewTerm(
        func=mdp.track_ang_vel_z_l2, weight=-2.0, params={"command_name": "base_velocity"}
    )

    # no lateral actuator -> skidding sideways is never the right answer
    lin_vel_y = RewTerm(func=mdp.lin_vel_y_l2, weight=-0.5)

    # -- keep the two wheels on a common axle ----------------------------
    # `phi` is measured per leg and NOTHING tied the left and right values together, so
    # the policy learned to lean one leg forward and the other one back and use that as
    # an extra yaw actuator.  The physical result is a SKEWED axle: the two wheels no
    # longer sit at the same fore-aft position, so the robot drives with one wheel ahead
    # of the other and scrubs sideways instead of going straight.
    #
    # The term is |L0_l*sin(phi_l) - L0_r*sin(phi_r)| in metres -- literally "how many
    # millimetres are the wheels out of line" -- and it is L1 so it pulls with a constant
    # force at any offset.
    #
    # Deliberately NOT penalised: the COMMON part of phi.  Both legs leaning together is
    # the inverted-pendulum CG shift that produces forward acceleration, so it must stay
    # free.  Only the left-right difference is constrained, which is also what the
    # reference firmware does (its VMC + LQR drives both leg angles to the same
    # reference).
    #
    # Scale check: 3 deg of asymmetry per leg at L0 = 0.25 m gives 26 mm of skew, and
    # 26 mm * 8 = 0.21 per step against an alive bonus of 0.5 -- a clear signal without
    # dominating the tracking terms.
    wheel_axle = RewTerm(
        func=mdp.leg_axle_skew_abs, weight=-8.0, params={"action_name": "joints"}
    )

    # -- make the LEGS the balancing actuator ----------------------------
    # Pitch is controllable by two redundant actuators on this robot:
    #   (a) wheel torque -- the reaction torque twists the body (Segway style);
    #   (b) the hips    -- changing the leg angle phi slides the body's centre of
    #                     gravity fore/aft over the wheel contact (inverted-pendulum
    #                     style), which needs no wheel torque at all.
    # Nothing in the old reward set charged anything for (a): with joint_torques at
    # -1e-4 a 2 Nm wheel torque costs 8e-6 per step against an alive bonus of 1e-2,
    # i.e. 1250x cheaper than staying alive.  PPO therefore kept the legs rigid and
    # balanced entirely with the wheels.
    #
    # `wheel_torques` charges only the WHEEL joints, so it does not discourage the
    # hips.  Inverted-pendulum locomotion by leaning needs almost no wheel torque
    # (the wheels only have to roll under the body), while wheel-torque balancing
    # spikes into the 1-3 Nm range.  This term is what flips the trade-off.
    #
    # It uses a DEADBAND at 1.0 Nm rather than a plain L2: the wheels are also the only
    # longitudinal actuator, and with the speed range now up to 2 m/s a plain sum(tau^2)
    # would charge normal acceleration (~0.9 Nm per wheel) and command steps (~1.7 Nm).
    # Only the attitude-correction spikes are penalised -- see mdp.rewards.
    wheel_torques = RewTerm(
        func=mdp.wheel_torques_excess_l2,
        weight=-5.0e-2,
        # v1.0.8: dead band 1.0 -> 1.5 N*m.  The maximum wheel torque is 3.69 N*m, so a
        # 1.0 dead band left only 2.7 N*m of free authority and charged for every
        # acceleration burst -- exactly what tight velocity tracking needs.  The term
        # still keeps the wheels from becoming the balance actuator (the reason it exists).
        params={"threshold": 1.5, "action_name": "joints"},
    )

    # -- smooth actuation ------------------------------------------------
    # action_rate is relaxed (was -1e-2): the legs must be allowed to move FAST to
    # track pitch/roll, otherwise the penalty alone keeps them stiff.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-4.0e-3)
    # NEW in v1.0.5: penalise the SECOND difference of the action, not the first.
    #
    # The legs still tremble a little after the reward/dynamics fixes.  Raising
    # `action_rate` cannot fix that -- it charges every change, including the fast leg
    # motion that keeps the body upright -- so the tremor needs a term that is blind to
    # smooth motion.  For a constant / ramp / accelerating trajectory
    #     a[t] - 2a[t-1] + a[t-2]
    # is ~0, while for a step-alternating (dithering) action it is 4x the amplitude.
    #
    # Scale: a tremor of +/-0.05 in every action dim gives sum(jerk^2) ~ 0.24, so -0.5
    # charges 0.12 per step against an alive bonus of 0.5 -- clearly worth removing,
    # while a smooth leg swing is essentially free.
    action_jerk = RewTerm(func=mdp.action_jerk_l2, weight=-0.3, params={"action_name": "joints"})
    # joint_torques covers the hips too, so keep it light (was -1e-4) -- penalising
    # hip torque directly fights the behaviour we want.
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    # was -1e-7 (effectively zero).  Joint acceleration is the most direct measure of the
    # tremor, so give it a real weight; 1e-6 still stays well below the tracking terms.
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)
    # weight lowered from -1e-4: at 2 m/s the wheels legitimately spin at 29 rad/s, so a
    # plain sum(w^2) costs 0.17 per step -- a 3.4% drag that biases the policy against
    # actually using the top of the speed range.
    wheel_spin = RewTerm(func=mdp.wheel_spin_l2, weight=-3.0e-5, params={"action_name": "joints"})

    # -- the closed loop must stay closed --------------------------------
    closure_residual = RewTerm(
        func=mdp.closure_residual_l2, weight=-1.0e3, params={"action_name": "joints"}
    )


@configclass
class LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Velocity-command tracking environment."""

    #: 1024 is the largest environment count this machine can run without running the
    #: 8 GB GPU (and PhysX' fabric path) out of memory; ``--num_envs`` overrides it.
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: LocomotionObservationsCfg = LocomotionObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: LocomotionRewardsCfg = LocomotionRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: LocomotionCommandsCfg = LocomotionCommandsCfg()

    #: name of the action term that owns the loop-closing knee targets
    joint_action_name: str = "joints"

    def __post_init__(self):
        # ---- 由 TARGETS 反推的派生量：改上面的速度目标后，这里自动跟着变 ----------
        # 1) 动作尺度：wheel_scale 决定“策略输出 1.0 对应多少轮速”。速度目标变高而
        #    wheel_scale 不变时，所需轮速会超过 clip_actions × wheel_scale，机器人根本
        #    跑不到目标速度，而且不会报错 -- 所以必须由目标反推。
        self.actions.joints.wheel_scale = TARGETS.wheel_scale
        # 2) 跟踪奖励的 exp 核宽度：随速度量程缩放
        self.rewards.track_lin_vel_x.params["std"] = TARGETS.track_lin_std
        # 3) 轮子空转惩罚：按 vmax² 反向缩放，保持相对惩罚不变
        self.rewards.wheel_spin.weight = TARGETS.wheel_spin_weight

        # general settings
        self.decimation = 4  # 50 Hz control on a 200 Hz physics step
        self.episode_length_s = 20.0  # long enough for 2-4 commands per episode
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Isaac Lab's default is 5 * 2**15; this scene only has a flat ground plane and
        # 1024 small robots (~11 links each), so the doubled buffer bought nothing and
        # only added GPU memory pressure -- the exact resource whose exhaustion shows up
        # as "CUDA error: an illegal memory access" inside the PhysX GPU allocator.
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15

        import os as _os

        # NOTE: these are per-process environment variables, NOT persistent settings.
        #   * they only live in the terminal window that exported them;
        #   * `set INFANTRY_RL_USE_FABRIC=` (empty) or closing the window reverts them;
        #   * nothing is written to the system/user environment.
        # The line below always prints the resolved state so a run can be checked.
        _fabric_off = _os.environ.get("INFANTRY_RL_USE_FABRIC", "1").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        )
        if _fabric_off:
            self.sim.use_fabric = False
        print(f"[cfg] PhysX fabric        : {'DISABLED (INFANTRY_RL_USE_FABRIC)' if _fabric_off else 'enabled (default)'}")

        _iters = _os.environ.get("INFANTRY_RL_PHYSX_POS_ITERS")
        if _iters:
            try:
                self.sim.physx.max_position_iteration_count = int(_iters)
                print(f"[cfg] PhysX position iters: {int(_iters)} (INFANTRY_RL_PHYSX_POS_ITERS)")
            except ValueError:
                pass

        # 训练目标（速度指令）的解析结果：启动时打印一次，便于核对“到底在训什么速度”。
        # 这些值全部来自文件顶部的 TARGETS = LocomotionTargets(...)（唯一配置入口）。
        print(
            f"[cfg] train mode          : {TARGETS.mode.upper()}  "
            f"(TRAIN_MODE @ locomotion_env_cfg.py; 改这一个值即可切换训练目标)"
        )
        print(
            f"[cfg] derived from TARGETS: wheel_scale {TARGETS.wheel_scale:.2f} "
            f"(动作量程上限 {TARGETS.clip_actions * TARGETS.wheel_scale * TARGETS.wheel_radius:.2f} m/s), "
            f"track_lin std {TARGETS.track_lin_std:.3f} m/s, "
            f"wheel_spin {TARGETS.wheel_spin_weight:+.2e}"
        )
        print(
            f"[cfg] command ranges      : vx [{TARGETS.vx_min:+.2f}, {TARGETS.vx_max:+.2f}] m/s "
            f"(INFANTRY_RL_VX_MIN / INFANTRY_RL_VX_MAX), "
            f"standing envs {TARGETS.rel_standing_envs:.0%} (vx=0), "
            f"spin envs {TARGETS.rel_spin_envs:.0%} (vx=0, wz kept), "
            f"wz clamp +/-{TARGETS.wz_max:.2f} rad/s, "
            f"heading stiffness {TARGETS.heading_stiffness:.2f}, "
            f"resample {TARGETS.resample_time_s[0]:.0f}-{TARGETS.resample_time_s[1]:.0f} s "
            f"[TARGETS @ locomotion_env_cfg.py]"
        )
        # 转向/旋转的三种形态必须分开打印，否则 yaw_rate 模式会被误标成"直线任务"
        if not TARGETS.heading_command:
            print(
                "[cfg] turning             : wz SAMPLED directly in "
                f"[{TARGETS.wz_min:+.1f}, {TARGETS.wz_max:+.1f}] rad/s "
                "(heading_command=False, no heading target -- yaw_rate task)"
            )
        elif TARGETS.enable_turning:
            print(
                "[cfg] turning             : ENABLED -- heading targets sampled in [-pi, pi], "
                "wz comes from the heading loop (turn-in-place / turning task)"
            )
        else:
            print(
                "[cfg] turning             : DISABLED -- heading TARGET pinned to 0, the "
                "heading loop still cancels yaw drift (straight-line task)"
            )

        # viewer (only used in non-headless play)
        self.viewer.eye = (2.0, -2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.2)


@configclass
class LocomotionEnvCfg_PLAY(LocomotionEnvCfg):
    """Variant used for replay / scripted evaluation: no random command resampling."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 1.0e6  # never time out during a replay
        # no observation corruption at play time
        self.observations.policy.enable_corruption = False
        # a fixed command owns the task: never resample it, never force "stand still"
        self.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        self.commands.base_velocity.rel_standing_envs = 0.0


__all__ = [
    "ANG_VEL_Z_RANGE",
    "HEADING_STIFFNESS",
    "LIN_VEL_X_RANGE",
    "LocomotionCommandsCfg",
    "LocomotionEnvCfg",
    "LocomotionEnvCfg_PLAY",
    "LocomotionObservationsCfg",
    "LocomotionRewardsCfg",
    "LocomotionTargets",
    "TARGETS",
    "ActionsCfg",
    "RobotSceneCfg",
    "TerminationsCfg",
    "FALL_ANGLE",
    "FALL_HEIGHT",
    "NOMINAL_HEIGHT",
    "CommandTermCfg",
    "DoneTerm",
    "INFANTRY_RL_CFG",
]
