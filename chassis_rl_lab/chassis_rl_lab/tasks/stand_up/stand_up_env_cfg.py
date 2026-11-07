"""chassis-rl 五连杆轮腿机器人「两阶段」训练环境配置（方案 A：阶段1 屏蔽髋、阶段2 放开髋）。

阶段1（``Chassis-Stand-Up-v0``）：倒地起立 + 保持平衡
------------------------------------------------------
- **屏蔽髋关节输出**：动作维度仍是 6 维（为了和阶段2 接口一致、可 ``--resume``），
  但前 4 维髋动作被忽略；髋由执行器 PD（300/10）保持在 squat 姿态，腿不动；
  策略实际只控制 2 个轮力矩 → 学"从倒地姿态用轮子起立并保持平衡"。
- 起始：倒地姿态（pitch 0.24~0.34 rad ≈14~19°，squat 腿型）；
- **倒地方向课程**：前 300 iter 用 pitch **正**方向倒地 + 辅助力（300 iter 内 1→0 退火）；301~600 iter 用 pitch **负**方向倒地 + 辅助力（同样 300 iter 内退火）；之后**无辅助力**，pitch 正/负每 episode **随机**。

阶段2（``Chassis-Stand-Up-Height-v0``）：放开髋 + 按指令改变机体高度
-------------------------------------------------------------------
- 髋关节改为**纯力矩**控制（动作 6 维全部生效：4 髋力矩 + 2 轮力矩），腿可以伸展/收腿；
- 起始：倒地 + 髋收到**最短腿长**（q=0 机械收回端），开局 0.4 s「收腿稳定」后交给策略；
- 高度指令：每个 episode 随机目标高度 ∈ [0.172, 0.190] m，obs 里给策略、奖励跟踪；
- 从阶段1 checkpoint ``--resume`` 继续训练（两阶段观测 25 维 / 动作 6 维完全一致）。

控制/仿真
---------
- 物理 200 Hz（``sim.dt = 1/200``），decimation=4 → 控制 50 Hz；episode 20 s（1000 控制步）；
- 关节符号约定：所有关节绕世界 +y 轴右手定则正转为正（左右同号）；轮子正转/正力矩 = +x 前进。
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
)
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
)
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
)
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
)
from isaaclab.managers import (
    SceneEntityCfg,
)
from isaaclab.managers import (
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from chassis_rl_lab.kinematics import HIP_ORDER, KNEE_ORDER, WHEEL_ORDER, load_squat_pose
from chassis_rl_lab.robots.chassis_rl_cfg import CHASSIS_RL_CFG, SQUAT_BASE_HEIGHT
from chassis_rl_lab.tasks.common import mdp
from chassis_rl_lab.tasks.common.mdp.actions import ChassisWheelLegActionCfg
from chassis_rl_lab.tasks.common.mdp.sensor_noise import ACC_NOISE_CFG, GYRO_NOISE_CFG


def _joint_cfg(order) -> SceneEntityCfg:
    """按名字选关节（顺序严格等于 order，不依赖 articulation 内部顺序）。"""
    return SceneEntityCfg("robot", joint_names=list(order), preserve_order=True)


#: 名义平衡高度 [m]（squat 附近；squat 运动学高度 0.1756 实测站住约 0.1716）
NOMINAL_HEIGHT: float = SQUAT_BASE_HEIGHT - 0.004
#: **腿长最短**的 hip 姿态：机械收回端 q=0（CAD 姿态；关节限位下限端，squat 距它仅 0.035 rad）
RETRACTED_HIPS: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
#: 腿长最短时的 base 高度 [m]（运动学：0.171707）
RETRACTED_BASE_HEIGHT: float = 0.1717070043
#: 倒地起始姿态的 pitch 范围 [rad]（≈14~19°，实测"歪着靠在腿上"的稳定姿态）
FALLEN_PITCH_RANGE: tuple[float, float] = (0.24, 0.34)
#: 目标高度指令范围 [m]：
#: - 下限 = 腿长最短（收回端）0.1717 m；
#: - 上限 0.270 m 对应髋伸展 ~0.60 rad（运动学最高 0.2825 m @ 0.66 rad，再大闭链换分支）。
HEIGHT_COMMAND_RANGE: tuple[float, float] = (0.172, 0.270)
#: 高度指令 curriculum 长度 [全局控制步]：前期从名义高度逐步放开到完整范围
#: （0 = 不启用 curriculum，直接随机；48 步/iter ⇒ 800 iter ≈ 38400）。
#: 范围从 0.19 扩到 0.27 后拉长 curriculum，避免续训时突然要求更高姿态。
HEIGHT_CURRICULUM_STEPS: int = 48 * 800
#: 阶段1"收腿稳定/自然倒地"总时长（控制步）：阶段1 不需要，置 0
STAGE1_SETTLE_STEPS: int = 0
#: 阶段2 开局"收腿稳定"步数（0.4 s）：髋 PD 把腿保持在最短姿态后再交给策略
STAGE2_SETTLE_STEPS: int = 20
#: **倒地方向课程**阶段长度 [全局控制步]：每 300 iter（48 步/iter）一个阶段。
#: 阶段0：pitch 正方向倒地 + 辅助力退火；阶段1：pitch 负方向倒地 + 辅助力退火；
#: 阶段2 起：无辅助力，pitch 正/负每个 episode 随机。
CURRICULUM_PHASE_STEPS: int = 48 * 300


# --------------------------------------------------------------------------- #
# 场景
# --------------------------------------------------------------------------- #
@configclass
class SceneCfg(InteractiveSceneCfg):
    """**阶段2 场景**：地面 + chassis-rl（髋/轮纯力矩、knee PD 闭链随动）+ 灯。"""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # 阶段2：hips 纯力矩（stiffness=0）+ 被动阻尼 2.0；wheels 0/0；knee 保持默认 PD
    robot = CHASSIS_RL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=CHASSIS_RL_CFG.init_state.replace(pos=(0.0, 0.0, RETRACTED_BASE_HEIGHT)),
        actuators={
            **CHASSIS_RL_CFG.actuators,
            # 髋：纯力矩 + 被动阻尼（等效电机反电动势；阻尼=0 时腿塌陷会"弹飞"）
            "hips": CHASSIS_RL_CFG.actuators["hips"].replace(stiffness=0.0, damping=2.0),
            "wheels": CHASSIS_RL_CFG.actuators["wheels"].replace(stiffness=0.0, damping=0.0),
        },
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class SceneCfgHipLocked(SceneCfg):
    """**阶段1 场景**：髋关节用默认 PD（300/10）锁在 squat 姿态；wheels 纯力矩。"""

    robot = CHASSIS_RL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=CHASSIS_RL_CFG.init_state.replace(pos=(0.0, 0.0, NOMINAL_HEIGHT)),
        actuators={
            **CHASSIS_RL_CFG.actuators,
            "wheels": CHASSIS_RL_CFG.actuators["wheels"].replace(stiffness=0.0, damping=0.0),
        },
    )


# --------------------------------------------------------------------------- #
# 动作
# --------------------------------------------------------------------------- #
@configclass
class ActionsCfg:
    """**阶段1 动作**：6 维接口，但前 4 维髋输出**被屏蔽**（髋 PD 锁 squat），只实际控制 2 轮力矩。"""

    wheel_leg = ChassisWheelLegActionCfg(
        asset_name="robot",
        use_hip_actions=True,  # 保持 6 维动作接口（阶段2 可 resume）
        mask_hip_actions=True,  # 但屏蔽髋输出：髋由 PD 保持在 squat，腿不动
        hip_control="position",
        lock_hips=False,
        wheel_control="torque",
        wheel_torque_scale=1.0,  # 1 action unit = 1 Nm（夹 ±4.92 Nm）
        clip_actions=8.0,
        wheel_unpowered_steps=0,
        settle_steps=STAGE1_SETTLE_STEPS,
        # 平衡辅助（"扶着学"）：轮上 PD 辅助力矩；按倒地方向课程分阶段退火
        balance_assist=True,
        assist_kp=15.0,
        assist_kd=1.0,
        assist_scale_start=1.0,
        assist_scale_end=0.0,
        # 每 CURRICULUM_PHASE_STEPS 步（=300 iter）一个阶段，阶段内 1→0 退火；
        # 前 2 个阶段有辅助，之后（第 3 阶段起）辅助恒为 0
        assist_cycle_steps=CURRICULUM_PHASE_STEPS,
        assist_phases=2,
        assist_apply_after_warmup=True,
    )


@configclass
class ActionsCfgStage2:
    """**阶段2 动作**：6 维全部生效（4 髋力矩 + 2 轮力矩），开局 0.4 s 收腿稳定。"""

    wheel_leg = ChassisWheelLegActionCfg(
        asset_name="robot",
        use_hip_actions=True,
        mask_hip_actions=False,  # 放开髋
        hip_control="torque",
        hip_torque_scale=2.5,  # 1 action unit = 2.5 Nm（clip 8 → ±20 Nm 持续上限）
        lock_hips=False,
        wheel_control="torque",
        wheel_torque_scale=1.0,
        clip_actions=8.0,
        wheel_unpowered_steps=0,
        # 开局收腿稳定：先用髋 PD 把腿保持在最短姿态 0.4 s，再交给策略
        settle_steps=STAGE2_SETTLE_STEPS,
        settle_hip_positions=list(RETRACTED_HIPS),
        settle_hip_kp=300.0,
        settle_hip_kd=10.0,
        # 阶段1 已学会轮平衡，阶段2 不再重新施加辅助
        balance_assist=False,
    )


# --------------------------------------------------------------------------- #
# 观测（25 维，Fudan 式布局；两阶段相同）
# --------------------------------------------------------------------------- #
@configclass
class ObservationsCfg:
    """观测组：policy（25 维，Fudan 式布局，actor/critic 共用）。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测（顺序即拼接顺序：ang_vel → gravity → commands → hip_pos_rel → dof_vel → actions）。"""

        base_ang_vel = ObsTerm(
            func=base_mdp.base_ang_vel, scale=0.25, noise=GYRO_NOISE_CFG
        )  # [0:3] 陀螺仪噪声（rad/s）
        projected_gravity = ObsTerm(
            func=base_mdp.projected_gravity, scale=1.0, noise=ACC_NOISE_CFG
        )  # [3:6] 加速度计噪声 σ = 实测量程噪声/9.81 g
        commands = ObsTerm(func=mdp.commands_obs)  # [6:9] [lin_vel_x, yaw, height_cmd×5]
        hip_pos_rel = ObsTerm(
            func=mdp.hip_pos_rel, params={"asset_cfg": _joint_cfg(HIP_ORDER)}, scale=1.0
        )  # [9:13]
        dof_vel = ObsTerm(func=mdp.actuated_dof_vel, scale=0.05)  # [13:19] 4 hip + 2 wheel
        last_action = ObsTerm(func=base_mdp.last_action)  # [19:25]

        def __post_init__(self) -> None:
            self.enable_corruption = True  # 训练期开启 BMI088 IMU 噪声
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# --------------------------------------------------------------------------- #
# 事件（阶段1：倒地 + squat 腿型 + 固定名义高度）
# --------------------------------------------------------------------------- #
@configclass
class EventCfg:
    """**阶段1 事件**：按课程切换倒地方向的 reset（squat 腿型）+ 固定名义高度 + 启动 DR + 推力。"""

    # 1) 机身位姿：**按训练阶段切换倒地方向**（阶段0 pitch 正、阶段1 pitch 负、之后随机），
    #    带 x/y/z/roll/yaw 与速度小随机
    reset_root = EventTerm(
        func=mdp.reset_root_fallen_directional,
        mode="reset",
        params={
            "pitch_range": FALLEN_PITCH_RANGE,
            "phase_steps": CURRICULUM_PHASE_STEPS,
            "x_range": (-0.05, 0.05),
            "y_range": (-0.05, 0.05),
            "z_range": (-0.004, 0.004),
            "roll_range": (-0.05, 0.05),
            "yaw_range": (-0.1, 0.1),
            "velocity_range": (-0.05, 0.05),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 2) 4 个 hip：squat 默认位 ±0.02 rad（阶段1 髋由 PD 锁 squat，腿不动）
    reset_hips = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": _joint_cfg(HIP_ORDER),
        },
    )

    # 3) 2 个轮：位置回到 0、速度 0
    reset_wheels = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": _joint_cfg(WHEEL_ORDER),
        },
    )

    # 4) knee：读上一步写入的 hip 实测角，解 knee 并写入（保证初始闭链一致）
    reset_closure = EventTerm(
        func=mdp.reset_closure,
        mode="reset",
        params={
            "asset_cfg": _joint_cfg(HIP_ORDER),
            "knee_cfg": _joint_cfg(KNEE_ORDER),
            "knee_margin": 0.01,
        },
    )

    # 5) 高度指令：阶段1 固定在名义高度（不做高度跟踪）
    reset_height_command = EventTerm(
        func=mdp.reset_height_command,
        mode="reset",
        params={
            "height_range": (NOMINAL_HEIGHT, NOMINAL_HEIGHT),
            "nominal": NOMINAL_HEIGHT,
            "curriculum_steps": 0,
        },
    )

    # ------------------------------- startup（只随机化一次） ------------------------------- #
    randomize_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel.*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )

    randomize_base_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )

    randomize_actuator_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # ------------------------------- interval（周期性推力，考验平衡鲁棒性） ------------------------------- #
    push_robot = EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "yaw": (-0.2, 0.2),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class EventCfgHeight(EventCfg):
    """**阶段2 事件**：髋收到最短腿长 reset + 每 episode 随机高度指令。"""

    # 髋：**收到最短腿长**（q=0 机械收回端）+ 小噪声（reset_closure 会按实测角解 knee）
    reset_hips = EventTerm(
        func=mdp.reset_hips_to_pose,
        mode="reset",
        params={
            "positions": RETRACTED_HIPS,
            "noise": 0.01,
            "asset_cfg": _joint_cfg(HIP_ORDER),
        },
    )

    # 高度指令：每 episode 在可行区间随机（带 curriculum）
    reset_height_command = EventTerm(
        func=mdp.reset_height_command,
        mode="reset",
        params={
            "height_range": HEIGHT_COMMAND_RANGE,
            "nominal": NOMINAL_HEIGHT,
            "curriculum_steps": HEIGHT_CURRICULUM_STEPS,
            "curriculum_start_steps": 0,
        },
    )


# --------------------------------------------------------------------------- #
# 奖励
# --------------------------------------------------------------------------- #
@configclass
class RewardsCfg:
    """**阶段1 奖励**：起立 + 平衡（高度跟踪关闭）。"""

    alive = RewTerm(func=mdp.is_alive, weight=0.5)  # 存活
    upright = RewTerm(func=mdp.upright, weight=2.0, params={"denom": 0.02})  # 姿态竖直（窄核）
    tilt = RewTerm(func=mdp.tilt_l2, weight=-3.0)  # 倾斜度平方惩罚（全程有梯度）
    pitch_progress = RewTerm(func=mdp.pitch_progress, weight=-1.5)  # 姿态改善过程奖励
    # 高度跟踪（阶段1 关闭）
    base_height_track = RewTerm(func=mdp.base_height_track, weight=0.0, params={"sigma": 0.001})
    height_rate = RewTerm(func=mdp.height_rate_l2, weight=0.0)
    base_lin_vel = RewTerm(func=mdp.base_lin_vel_xy_l2, weight=-1.0)
    base_ang_vel_xy = RewTerm(func=mdp.base_ang_vel_xy_l2, weight=-0.1)
    yaw_rate = RewTerm(func=mdp.yaw_rate_l2, weight=-0.3)
    dof_vel = RewTerm(func=mdp.dof_vel_l2, weight=-0.0005)
    dof_acc = RewTerm(func=mdp.dof_acc_l2, weight=-2.5e-7)
    torques = RewTerm(func=mdp.torques_l2, weight=-1.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.dof_pos_limits, weight=-1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-50.0)  # 实际 -1.0/回合


@configclass
class RewardsCfgHeight(RewardsCfg):
    """**阶段2 奖励**：在阶段1基础上打开高度跟踪。"""

    base_height_track = RewTerm(func=mdp.base_height_track, weight=2.0, params={"sigma": 0.001})
    height_rate = RewTerm(func=mdp.height_rate_l2, weight=-0.5)


# --------------------------------------------------------------------------- #
# 终止：超时 + 未恢复（宽限期后仍倒着 → 终止）
# --------------------------------------------------------------------------- #
@configclass
class TerminationsCfg:
    """终止项：time_out + 超时未恢复。"""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    failed_to_recover = DoneTerm(
        func=mdp.failed_to_recover, params={"limit_rad": 0.25, "grace_steps": 300}
    )
    # NaN/数值爆炸防护：根状态或关节状态出现 NaN/Inf、或速度超限时立即终止该 env
    state_invalid = DoneTerm(
        func=mdp.state_invalid,
        params={"max_lin_vel": 30.0, "max_ang_vel": 60.0, "max_joint_vel": 200.0},
    )


# --------------------------------------------------------------------------- #
# 阶段1 环境配置
# --------------------------------------------------------------------------- #
@configclass
class ChassisStandUpEnvCfg(ManagerBasedRLEnvCfg):
    """**阶段1**：从倒地开始 → 用轮子起立 → 保持平衡（髋输出被屏蔽，腿不动）。"""

    scene: SceneCfg = SceneCfgHipLocked(num_envs=2048, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    #: 参考值（日志/工具用）
    nominal_height: float = NOMINAL_HEIGHT
    fallen_pitch_range: tuple[float, float] = FALLEN_PITCH_RANGE
    height_command_range: tuple[float, float] = HEIGHT_COMMAND_RANGE

    def __post_init__(self) -> None:
        self.decimation = 4  # 50 Hz
        self.episode_length_s = 20.0  # 1000 个控制步
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.sim.device = "cuda:0"
        self.sim.physics_material = self.scene.ground.spawn.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        squat = load_squat_pose()
        self.nominal_height = float(NOMINAL_HEIGHT)
        self.target_base_height = float(squat["base_height"])
        self.wheel_unpowered_steps = int(self.actions.wheel_leg.wheel_unpowered_steps)


@configclass
class ChassisStandUpEnvCfg_PLAY(ChassisStandUpEnvCfg):
    """阶段1 play/评估用：小场景、关观测噪声、关随机化与推力。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.randomize_friction = None
        self.events.randomize_base_mass = None
        self.events.randomize_actuator_gains = None
        self.events.push_robot = None


# --------------------------------------------------------------------------- #
# 阶段2 环境配置（在阶段1 checkpoint 上 --resume）
# --------------------------------------------------------------------------- #
@configclass
class ChassisStandUpHeightEnvCfg(ChassisStandUpEnvCfg):
    """**阶段2**：放开髋关节 → 从收腿倒地状态起立 + 在保持平衡前提下按指令改变高度。

    与阶段1 的观测/动作维度完全一致（25 维 / 6 维），可直接：

        scripts\\train.py --task Chassis-Stand-Up-Height-v0 --resume --load_run ".*stage1" ...
    """

    scene: SceneCfg = SceneCfg(num_envs=2048, env_spacing=4.0)
    actions: ActionsCfgStage2 = ActionsCfgStage2()
    rewards: RewardsCfgHeight = RewardsCfgHeight()
    events: EventCfgHeight = EventCfgHeight()


# --------------------------------------------------------------------------- #
# 中间任务：倒地自起（随机方向）+ 平衡；髋自由，但**不跟踪高度/腿长**
# （从阶段1的 model_750 继续训练用；观测 25 / 动作 6 与阶段1、阶段2 完全一致）
# --------------------------------------------------------------------------- #
@configclass
class EventCfgRecovery(EventCfg):
    """恢复任务事件：**随机方向**倒地（pitch 正负随机）+ 髋收到最短腿长 + 固定高度指令。"""

    # 随机方向：phase_steps=0 ⇒ 阶段号恒 >=2 ⇒ 每个 episode 正/负随机
    reset_root = EventTerm(
        func=mdp.reset_root_fallen_directional,
        mode="reset",
        params={
            "pitch_range": FALLEN_PITCH_RANGE,
            "phase_steps": 0,
            "x_range": (-0.05, 0.05),
            "y_range": (-0.05, 0.05),
            "z_range": (-0.004, 0.004),
            "roll_range": (-0.05, 0.05),
            "yaw_range": (-0.1, 0.1),
            "velocity_range": (-0.05, 0.05),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 髋收到最短腿长（q=0，机械收回端）+ 小噪声；reset_closure 会按实测角解 knee
    reset_hips = EventTerm(
        func=mdp.reset_hips_to_pose,
        mode="reset",
        params={
            "positions": RETRACTED_HIPS,
            "noise": 0.01,
            "asset_cfg": _joint_cfg(HIP_ORDER),
        },
    )
    # reset_height_command 继承 EventCfg（固定名义高度，不做高度跟踪）


@configclass
class ChassisStandUpRecoveryEnvCfg(ChassisStandUpEnvCfg):
    """**倒地自起（随机方向）+ 保持平衡**；髋关节自由，用于从倒地状态站起来。

    与阶段2（Height）的区别：**不跟踪高度/不改变腿长**——高度指令固定名义值、
    高度奖励权重为 0；髋关节只为"站起来"服务。
    """

    scene: SceneCfg = SceneCfg(num_envs=2048, env_spacing=4.0)
    actions: ActionsCfgStage2 = ActionsCfgStage2()  # 髋力矩 + 开局 0.4 s 收腿稳定
    events: EventCfgRecovery = EventCfgRecovery()
    # rewards 用基类 RewardsCfg（高度跟踪权重 0），terminations 同基类
