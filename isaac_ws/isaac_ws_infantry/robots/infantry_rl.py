"""Articulation configuration for the infantry-rl closed-chain wheel-leg robot.

Verified facts this file encodes (see ``isaac_ws/docs/VERIFIED_RESULTS.md`` for the
measurements and how to reproduce them):

* the converted USD exposes **10** joints -- 4 hips, 4 knees, 2 wheels; the two
  ``loop_joints/*_calf_axle`` are *not* articulation DOFs, so the closed loop is
  re-imposed by control (see :mod:`isaac_ws_infantry.mdp.actions`);
* joint values keep the hardware sign convention (positive = CCW seen from the
  motor output shaft); no axis is ever flipped here;
* the knees are passive but must still belong to an actuator group, otherwise
  Isaac Lab never writes a position target for them into PhysX
  (``Articulation._apply_actuator_model`` only fills ``_joint_pos_target_sim`` for
  joints covered by an actuator).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from ..model import (
    HIP_ORDER,
    KNEE_ORDER,
    USD_PATH,
    WHEEL_ORDER,
    nominal_base_height,
    nominal_pose,
)

__all__ = [
    "INFANTRY_RL_CFG",
    "DEFAULT_JOINT_POS",
    "HIP_JOINT_REGEX",
    "KNEE_JOINT_REGEX",
    "WHEEL_JOINT_REGEX",
    "HIP_ORDER",
    "KNEE_ORDER",
    "WHEEL_ORDER",
    "NOMINAL_BASE_HEIGHT",
]

HIP_JOINT_REGEX = [".*_front_hip", ".*_back_hip"]
KNEE_JOINT_REGEX = [".*_front_knee", ".*_back_knee"]
WHEEL_JOINT_REGEX = [".*_wheel"]

#: reset / default joint positions, straight from the measured nominal stance
DEFAULT_JOINT_POS: dict[str, float] = {}
for _side, _names in (
    ("left", ("left_front_hip", "left_back_hip", "left_front_knee", "left_back_knee")),
    ("right", ("right_front_hip", "right_back_hip", "right_front_knee", "right_back_knee")),
):
    _vals = nominal_pose()[_side]
    DEFAULT_JOINT_POS[_names[0]] = float(_vals["front_hip"])
    DEFAULT_JOINT_POS[_names[1]] = float(_vals["back_hip"])
    DEFAULT_JOINT_POS[_names[2]] = float(_vals["front_knee"])
    DEFAULT_JOINT_POS[_names[3]] = float(_vals["back_knee"])
for _wheel in WHEEL_ORDER:
    DEFAULT_JOINT_POS[_wheel] = 0.0

#: root height (m) at which the nominal stance rests with the wheels on the ground
NOMINAL_BASE_HEIGHT = nominal_base_height()


INFANTRY_RL_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, NOMINAL_BASE_HEIGHT),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=dict(DEFAULT_JOINT_POS),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        # the two hip motors per leg: position controlled
        "hips": ImplicitActuatorCfg(
            joint_names_expr=HIP_JOINT_REGEX,
            effort_limit_sim=40.0,
            velocity_limit_sim=16.755,
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        # passive knee followers, driven every control step by the loop solver
        "knees": ImplicitActuatorCfg(
            joint_names_expr=KNEE_JOINT_REGEX,
            effort_limit_sim=40.0,
            velocity_limit_sim=16.755,
            stiffness=400.0,
            damping=10.0,
            armature=0.01,
        ),
        # wheels: velocity controlled, damping acts as the motor brake
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINT_REGEX,
            effort_limit_sim=3.69,
            velocity_limit_sim=61.471,
            stiffness=0.0,
            damping=2.0,
            armature=0.001,
        ),
    },
)
