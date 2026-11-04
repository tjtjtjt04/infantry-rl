"""Project-level constants and the robot asset configuration.

This module is the flat-layout equivalent of the original

* ``isaac_ws_infantry/robots/infantry_rl.py``  (the articulation config),
* ``isaac_ws_infantry/terrains/__init__.py``   (offline ground USD),
* the task-name / log-path glue that used to live in ``scripts/train.py``.

It imports Isaac Lab, so it must be imported **after** ``AppLauncher`` has
started Kit.  Anything that has to be read before that (CLI parsing) lives in
``train.py`` / ``play.py`` itself.

The task-specific MDP configs live in ``env_squat.py``; the shared scene/action/
loop-closure environment lives in ``env_base.py``; the PPO runner configs live in
``agent.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from robot_model import (
    HIP_ORDER,
    KNEE_ORDER,
    USD_PATH,
    WHEEL_ORDER,
    nominal_base_height,
    nominal_pose,
)

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
PROJECT_DIR: Path = Path(__file__).resolve().parent
ASSETS_DIR: Path = PROJECT_DIR / "assets"
GROUND_USD_PATH: Path = ASSETS_DIR / "ground" / "ground.usda"
DEFAULT_LOG_ROOT: Path = PROJECT_DIR / "logs" / "rsl_rl"

# --------------------------------------------------------------------------- #
# Task names
# --------------------------------------------------------------------------- #
#: canonical task names used by train.py / play.py
TASKS: tuple[str, ...] = ("squat", "squat_torque")

#: old gym IDs still accepted on the command line
TASK_ALIASES: dict[str, str] = {
    "Infantry-RL-Squat-v0": "squat",
    "Infantry-RL-Squat-Play-v0": "squat",
    "Infantry-RL-SquatTorque-v0": "squat_torque",
    "Infantry-RL-SquatTorque-Play-v0": "squat_torque",
}

#: rsl_rl experiment folder names
EXPERIMENT_NAMES: dict[str, str] = {
    "squat": "infantry_rl_squat",
    "squat_torque": "infantry_rl_squat_torque",
}

DEFAULT_ITERATIONS: dict[str, int] = {
    "squat": 1000,
    "squat_torque": 1000,
}
DEFAULT_HORIZON: dict[str, int] = {
    "squat": 24,
    "squat_torque": 24,
}
DEFAULT_NUM_ENVS: dict[str, int] = {
    "squat": 1024,
    "squat_torque": 1024,
}
DEFAULT_PLAY_NUM_ENVS: dict[str, int] = {
    "squat": 16,
    "squat_torque": 16,
}


def normalize_task(name: str | None) -> str:
    """Map a CLI task name / old gym ID to a canonical task name."""
    if name is None or name == "":
        return "squat"
    if name in TASK_ALIASES:
        return TASK_ALIASES[name]
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; expected one of {TASKS} or a legacy Infantry-RL-* ID")
    return name


# --------------------------------------------------------------------------- #
# Offline ground plane
# --------------------------------------------------------------------------- #
# ``TerrainImporterCfg(terrain_type="plane")`` (and ``spawn_ground_plane``) fetches a
# Nucleus-hosted USD, which fails offline.  ``terrain_type="usd"`` just spawns a local
# file, so this module ships the same one-plane ground (friction 1.0 / 0.9) and rebuilds
# it if it is missing.
#: Marker written into the ground USD.  ``ensure_ground_usd`` regenerates an existing
#: file that predates the geometric grid.
GROUND_USD_MARKER = "grid_major"


def _grid_line_mesh(
    name: str, spacing: float, width: float, z: float, material: str, half: float = 100.0
) -> str:
    """USDA for a line grid: lines every ``spacing`` m, ``width`` m wide, at height ``z``.

    Built from real (thin) quads rather than a texture: the RTX material conversion in
    this setup renders a constant ``UsdPreviewSurface`` colour correctly, but the
    ``UsdUVTexture`` network came out black.  A 200 m plane needs only ~480 quads, so
    this costs nothing worth worrying about.
    """
    n = int(round(half / spacing))
    points: list[tuple[float, float, float]] = []
    indices: list[int] = []

    def add_quad(quad: list[tuple[float, float, float]]) -> None:
        base = len(points)
        points.extend(quad)
        indices.extend((base, base + 1, base + 2, base + 3))

    for k in range(-n, n + 1):
        c = k * spacing
        # line parallel to Y at x = c
        add_quad(
            [
                (c - width / 2, -half, z),
                (c + width / 2, -half, z),
                (c + width / 2, half, z),
                (c - width / 2, half, z),
            ]
        )
        # line parallel to X at y = c
        add_quad(
            [
                (-half, c - width / 2, z),
                (half, c - width / 2, z),
                (half, c + width / 2, z),
                (-half, c + width / 2, z),
            ]
        )

    pts = ",\n            ".join(f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, _ in points)
    idx = ", ".join(str(i) for i in indices)
    counts = ", ".join("4" for _ in range(len(points) // 4))
    return f"""    def Mesh "{name}" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{idx}]
        point3f[] points = [
            {pts}
        ]
        float3[] extent = [({-half}, {-half}, {z}), ({half}, {half}, {z})]
        uniform token subdivisionScheme = "none"
        bool doubleSided = true
        rel material:binding = </ground/{material}>
    }}
"""


def _material(name: str, rgb: tuple[float, float, float]) -> str:
    """A constant-colour ``UsdPreviewSurface`` material (no texture network)."""
    r, g, b = rgb
    return f"""    def Material "{name}"
    {{
        token outputs:surface.connect = </ground/{name}/PreviewSurface.outputs:surface>

        def Shader "PreviewSurface"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = ({r}, {g}, {b})
            float inputs:roughness = 0.9
            float inputs:metallic = 0.0
            token outputs:surface
        }}
    }}
"""


def _ground_usda() -> str:
    return f"""#usda 1.0
(
    defaultPrim = "ground"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "ground"
{{
    # Physics collider: unchanged PhysX plane, so the contact dynamics are identical to
    # the old untextured ground.  Everything else is visual only.
    def Plane "plane" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]
    )
    {{
        uniform token axis = "Z"
        float3 xformOp:scale = (200, 200, 1)
        uniform token[] xformOpOrder = ["xformOp:scale"]
        rel material:binding:physics = </ground/groundMaterial>
    }}

    # Visual base, 1 mm above the collider: hides the dark physics plane and receives
    # the robot's shadow.
    def Mesh "grid_base" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [
            (-100, -100, 0.001),
            (100, -100, 0.001),
            (100, 100, 0.001),
            (-100, 100, 0.001)
        ]
        float3[] extent = [(-100, -100, 0.001), (100, 100, 0.001)]
        uniform token subdivisionScheme = "none"
        bool doubleSided = true
        rel material:binding = </ground/gridBaseMaterial>
    }}

{_grid_line_mesh("grid_fine", 1.0, 0.02, 0.002, "gridFineMaterial")}
{_grid_line_mesh("grid_major", 5.0, 0.05, 0.003, "gridMajorMaterial")}
{_material("gridBaseMaterial", (0.42, 0.43, 0.46))}
{_material("gridFineMaterial", (0.18, 0.18, 0.20))}
{_material("gridMajorMaterial", (0.02, 0.02, 0.03))}
    def Material "groundMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
    )
    {{
        float physics:dynamicFriction = 0.9
        float physics:staticFriction = 1.0
        float physics:restitution = 0.0
    }}
}}
"""


def ensure_ground_usd(path: Path | str | None = None) -> str:
    """Return the path to the local ground USD, (re)writing it when needed.

    The file is regenerated when it is missing OR predates the geometric grid (it has
    no ``GROUND_USD_MARKER``).  The physics plane itself is unchanged; only the visual
    base / line meshes are added.
    """
    target = Path(path) if path is not None else GROUND_USD_PATH
    stale = target.is_file() and GROUND_USD_MARKER not in target.read_text(encoding="utf-8", errors="ignore")
    if not target.is_file() or stale:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_ground_usda(), encoding="utf-8")
        print(f"[ground] wrote {target}" + (" (upgraded to the geometric grid)" if stale else ""))
    return str(target)


# --------------------------------------------------------------------------- #
# Robot articulation
# --------------------------------------------------------------------------- #
# Verified facts encoded here (see isaac_ws/docs/VERIFIED_RESULTS.md in the original
# workspace for the measurements):
#   * the converted USD exposes **10** joints -- 4 hips, 4 knees, 2 wheels; the two
#     ``loop_joints/*_calf_axle`` are not articulation DOFs, so the loop is re-imposed
#     by control (``mdp.actions``);
#   * joint values keep the hardware sign convention (positive = CCW seen from the
#     motor output shaft); no axis is flipped here;
#   * knees are passive but still belong to an actuator group, otherwise Isaac Lab
#     never writes a position target for them into PhysX.
#: built from the resolved name tuples because ``chassis-rl`` calls its wheel
#: joints ``wheel_left`` / ``wheel_right`` (the old ``.*_wheel`` regex misses them)
HIP_JOINT_REGEX = [f".*{name}" for name in HIP_ORDER]
KNEE_JOINT_REGEX = [f".*{name}" for name in KNEE_ORDER]
WHEEL_JOINT_REGEX = [f".*{name}" for name in WHEEL_ORDER]

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

#: hardware effort limits used by both the implicit actuators and the torque action
HIP_EFFORT_LIMIT = 40.0
WHEEL_EFFORT_LIMIT = 3.69


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
            effort_limit_sim=HIP_EFFORT_LIMIT,
            velocity_limit_sim=16.755,
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        # passive knee followers, driven every control step by the loop solver
        "knees": ImplicitActuatorCfg(
            joint_names_expr=KNEE_JOINT_REGEX,
            effort_limit_sim=HIP_EFFORT_LIMIT,
            velocity_limit_sim=16.755,
            stiffness=400.0,
            damping=10.0,
            armature=0.01,
        ),
        # wheels: velocity controlled, damping acts as the motor brake
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINT_REGEX,
            effort_limit_sim=WHEEL_EFFORT_LIMIT,
            velocity_limit_sim=61.471,
            stiffness=0.0,
            damping=2.0,
            armature=0.001,
        ),
    },
)

#: Pure-torque variant for the `squat_torque` task: hips and wheels have no implicit
#: PD (``set_joint_effort_target`` is then the joint torque); the knees keep their
#: position servo because they are the closed-chain followers.
INFANTRY_RL_TORQUE_CFG = INFANTRY_RL_CFG.replace(
    actuators={
        "hips": INFANTRY_RL_CFG.actuators["hips"].replace(stiffness=0.0, damping=0.0),
        "knees": INFANTRY_RL_CFG.actuators["knees"].replace(),
        "wheels": INFANTRY_RL_CFG.actuators["wheels"].replace(stiffness=0.0, damping=0.0),
    }
)


__all__ = [
    "ASSETS_DIR",
    "DEFAULT_HORIZON",
    "DEFAULT_ITERATIONS",
    "DEFAULT_JOINT_POS",
    "DEFAULT_LOG_ROOT",
    "DEFAULT_NUM_ENVS",
    "DEFAULT_PLAY_NUM_ENVS",
    "EXPERIMENT_NAMES",
    "GROUND_USD_MARKER",
    "GROUND_USD_PATH",
    "HIP_EFFORT_LIMIT",
    "HIP_JOINT_REGEX",
    "HIP_ORDER",
    "INFANTRY_RL_CFG",
    "INFANTRY_RL_TORQUE_CFG",
    "KNEE_JOINT_REGEX",
    "KNEE_ORDER",
    "NOMINAL_BASE_HEIGHT",
    "PROJECT_DIR",
    "TASKS",
    "TASK_ALIASES",
    "WHEEL_EFFORT_LIMIT",
    "WHEEL_JOINT_REGEX",
    "WHEEL_ORDER",
    "ensure_ground_usd",
    "normalize_task",
]
