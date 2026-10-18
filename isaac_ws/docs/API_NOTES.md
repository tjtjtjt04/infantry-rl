# Isaac Lab 2.3.2 / Isaac Sim 5.1.0 — Verified API Notes

Extracted by reading the **installed source tree** at `D:\IsaacLab\source\...`. Every claim below carries a
`path:line` citation to a line that was actually read.

## Provenance / environment

| Fact | Value | Evidence |
|---|---|---|
| Isaac Lab version | `2.3.2` | `D:\IsaacLab\VERSION:1` |
| `isaaclab` python dist | `0.54.4` (editable from `D:\IsaacLab\source\isaaclab`) | `D:\IsaacLab\source\isaaclab\pyproject.toml:1-4` + queried metadata |
| `isaacsim` python dist | `5.1.0.0` | queried metadata |
| `rsl-rl-lib` python dist | `5.0.1` | queried metadata |
| `isaaclab_rl` / `isaaclab_tasks` | `0.5.2` / `0.11.16` | queried metadata |

Source-relative root used below: `D:\IsaacLab\source\`. Citations are shortened as
`isaaclab/isaaclab/...` meaning `D:\IsaacLab\source\isaaclab\isaaclab\...`.

**IMPORTANT:** `isaaclab.app.AppLauncher` at 2.3.2 **requires** `isaacsim` on path and imports
`from isaacsim import SimulationApp` (`isaaclab/isaaclab/app/app_launcher.py:24-27`). It no longer wraps
`omni.isaac.kit.SimulationApp`.

---

## 1. `isaaclab.app.AppLauncher`

### 1.1 Constructor

`isaaclab/isaaclab/app/app_launcher.py:63`

```python
def __init__(self, launcher_args: argparse.Namespace | dict | None = None, **kwargs):
```

- `launcher_args`: dict or `argparse.Namespace` (converted via `.__dict__`). `None` == `{}`. — `app_launcher.py:97-100`
- `**kwargs`: merged into `launcher_args`. Overlapping keys raise `ValueError`. — `app_launcher.py:103-111`
- Properties exposed for training scripts: `self.app` (`SimulationApp`), `self.device_id`, `self.local_rank`,
  `self.global_rank`. — `app_launcher.py:120-122`, `app_launcher.py:169-175`

### 1.2 Canonical headless training-script form

```python
# standard Isaac Lab 2.3.2 script prologue
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent.")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)      # adds --headless, --device, --enable_cameras, ...
args_cli = parser.parse_args()

# or, equivalently, directly:
app_launcher = AppLauncher(headless=True, device="cuda:0")   # headless: bool, device: str
simulation_app = app_launcher.app
```

### 1.3 Args added by `AppLauncher.add_app_launcher_args(parser)`

Static method — `app_launcher.py:181-378`. Added flags:

| Flag | Type / action | Default | Citation |
|---|---|---|---|
| `--headless` | `store_true` (bool `False`) | `False` | `app_launcher.py:275-280` |
| `--livestream` | `int`, choices `{0,1,2}` | `-1` | `app_launcher.py:281-287` |
| `--enable_cameras` | `store_true` (bool `False`) | `False` | `app_launcher.py:288-293` |
| `--xr` | `store_true` (bool `False`) | `False` | `app_launcher.py:294-299` |
| `--device` | `str`, `ExplicitAction` | `"cuda:0"` | `app_launcher.py:300-306` |
| `--cpu` | `store_true`, `argparse.SUPPRESS` (deprecated, raises) | — | `app_launcher.py:308`, `:677-679` |
| `--verbose` | `store_true` | — | `app_launcher.py:309-313` |
| `--info` | `store_true` | — | `app_launcher.py:314-318` |
| `--experience` | `str` | `""` | `app_launcher.py:319-328` |
| `--rendering_mode` | `str`, choices `{"performance","balanced","quality"}` | `"balanced"` | `app_launcher.py:329-339` |
| `--kit_args` | `str` (space-separated) | `""` | `app_launcher.py:340-348` |
| `--anim_recording_enabled` | `store_true` | — | `app_launcher.py:349-353` |
| `--anim_recording_start_time` | `float` | `0` | `app_launcher.py:354-362` |
| `--anim_recording_stop_time` | `float` | `10` | `app_launcher.py:363-371` |

Authoritative default table: `app_launcher.py:384-392`

```python
_APPLAUNCHER_CFG_INFO: dict[str, tuple[list[type], Any]] = {
    "headless":        ([bool], False),
    "livestream":      ([int],  -1),
    "enable_cameras":  ([bool], False),
    "xr":              ([bool], False),
    "device":          ([str],  "cuda:0"),
    "experience":      ([str],  ""),
    "rendering_mode":  ([str],  "balanced"),
}
```

### 1.4 `kwargs` actually forwarded to `SimulationApp`

Only keys present in both your kwargs and this whitelist land in `self._sim_app_config`.
`app_launcher.py:404-428` + `:508-510`:

```python
_SIM_APP_CFG_TYPES = {
    "headless": [bool], "hide_ui": [bool, type(None)], "active_gpu": [int, type(None)],
    "physics_gpu": [int], "multi_gpu": [bool], "sync_loads": [bool],
    "width": [int], "height": [int], "window_width": [int], "window_height": [int],
    "display_options": [int], "subdiv_refinement_level": [int], "renderer": [str],
    "anti_aliasing": [int], "samples_per_pixel_per_frame": [int], "denoiser": [bool],
    "max_bounces": [int], "max_specular_transmission_bounces": [int], "max_volume_bounces": [int],
    "open_usd": [str, type(None)], "livesync_usd": [str, type(None)],
    "fast_shutdown": [bool], "experience": [str],
}
```

So `AppLauncher(headless=True, device="cuda:0", num_envs=4096)` is valid: `num_envs` is simply ignored by
`SimulationApp` but retained in `launcher_args` for your script. — `app_launcher.py:97-111`, `:508-510`

### 1.5 Behaviour details worth knowing

- **Environment-variable overrides**, resolved in `_config_resolution` (`app_launcher.py:480-510`):
  `LIVESTREAM` (`:514`), `PUBLIC_IP` (`:541`), `HEADLESS` (`:573`), `ENABLE_CAMERAS` (`:607`),
  `XR` (`:626`), `EXP_PATH` (`:714`), `LOCAL_RANK`/`JAX_LOCAL_RANK` (`:683`), `RANK`/`JAX_RANK` (`:685`),
  `WORLD_SIZE` (`:693`).
- `livestream` >= 1 forces `headless = True`. — `app_launcher.py:585-598`
- `enable_cameras=True` + `headless=True` implies `self._offscreen_render = True`. — `app_launcher.py:620-622`
- `headless=True` and no livestream disables the viewport: `self._render_viewport = False`. — `app_launcher.py:643-645`
- `launcher_args["create_new_stage"] = False` is forced for startup performance. — `app_launcher.py:653`
- `launcher_args["hide_ui"] = True` when headless without livestream. — `app_launcher.py:648-650`
- `--cpu` raises `ValueError("The '--cpu' flag is deprecated. Please use '--device cpu' instead.")` — `app_launcher.py:677-679`
- Distributed (`launcher_args["distributed"] == True`) sets `local_rank`, `global_rank`, `device_id`
  and forces `multi_gpu=False`. — `app_launcher.py:681-698`
- Experience `.kit` selection when `--experience` is empty — `app_launcher.py:721-741`:
  headless+rendering → `apps/isaaclab.python.headless.rendering.kit`; rendering → `apps/isaaclab.python.rendering.kit`;
  xr headless → `apps/isaaclab.python.xr.openxr.headless.kit`; xr → `apps/isaaclab.python.xr.openxr.kit`;
  headless → `apps/isaaclab.python.headless.kit`; else → `apps/isaaclab.python.kit`.
- `is_isaac_sim_version_4_5()` public method exists (returns `False` on your 5.1 install). — `app_launcher.py:945-967`

---

## 2. `ManagerBasedRLEnv` / `ManagerBasedRLEnvCfg`

### 2.1 Class definition and constructor

```python
# isaaclab/isaaclab/envs/manager_based_rl_env.py:65
def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
```

`render_mode=None` is "similar to `human`". — `manager_based_rl_env.py:65-72`
`self.render_mode = render_mode`; `self.metadata["render_fps"] = 1 / self.step_dt`. — `:82`, `:87`

Key derived quantities:

```python
# isaaclab/isaaclab/envs/manager_based_rl_env.py:96-103
max_episode_length_s  == self.cfg.episode_length_s
max_episode_length    == math.ceil(self.max_episode_length_s / self.step_dt)
```

### 2.2 `ManagerBasedRLEnvCfg` — full field list

Declaration: `isaaclab/isaaclab/envs/manager_based_rl_env_cfg.py:14-80`.
It **inherits** `ManagerBasedEnvCfg` (`manager_based_env_cfg.py:38-150`). There is **no** `randomization`
field anywhere in this hierarchy — see GOTCHAS.

**Declared directly on `ManagerBasedRLEnvCfg`** (`manager_based_rl_env_cfg.py`):

```python
@configclass
class ManagerBasedRLEnvCfg(ManagerBasedEnvCfg):
    ui_window_class_type: type | None = ManagerBasedRLEnvWindow          # :19
    is_finite_horizon: bool = False                                      # :22
    episode_length_s: float = MISSING                                    # :44
    rewards: object = MISSING                                            # :58
    terminations: object = MISSING                                       # :64
    curriculum: object | None = None                                     # :70
    commands: object | None = None                                       # :76
```

**Inherited from `ManagerBasedEnvCfg`** (`manager_based_env_cfg.py`):

```python
@configclass
class ManagerBasedEnvCfg:
    viewer: ViewerCfg = ViewerCfg()                                      # :43
    sim: SimulationCfg = SimulationCfg()                                 # :46
    ui_window_class_type: type | None = BaseEnvWindow                    # :50  (shadowed above)
    seed: int | None = None                                              # :62
    decimation: int = MISSING                                            # :70
    scene: InteractiveSceneCfg = MISSING                                 # :78
    recorders: object = DefaultEmptyRecorderManagerCfg()                 # :84
    observations: object = MISSING                                       # :90
    actions: object = MISSING                                            # :96
    events: object = DefaultEventManagerCfg()                            # :102
    rerender_on_reset: bool = False               # DEPRECATED 2.3.1      # :108
    num_rerenders_on_reset: int = 0                                      # :126
    wait_for_textures: bool = True                                       # :137
    xr: XrCfg | None = None                                              # :140
    teleop_devices: DevicesCfg = field(default_factory=DevicesCfg)       # :143
    export_io_descriptors: bool = False                                  # :146
    log_dir: str | None = None                                           # :149
```

#### Which fields are `@configclass` subclasses?

| Field | Annotation | Actual type used at runtime | Citation |
|---|---|---|---|
| `viewer` | `ViewerCfg` (**@configclass**) | `isaaclab.envs.common.ViewerCfg` | `manager_based_env_cfg.py:43`; `envs/common.py:20-21` |
| `sim` | `SimulationCfg` (**@configclass**) | `isaaclab.sim.SimulationCfg` | `manager_based_env_cfg.py:46`; `sim/simulation_cfg.py:348-349` |
| `scene` | `InteractiveSceneCfg` (**@configclass**) | subclassed by user, e.g. `MySceneCfg` | `manager_based_env_cfg.py:78`; `scene/interactive_scene_cfg.py:11-12` |
| `observations` | `object` | **your own plain `@configclass`** holding `ObservationGroupCfg` attrs | `manager_based_env_cfg.py:90` |
| `actions` | `object` | **your own plain `@configclass`** holding `ActionTermCfg` attrs | `manager_based_env_cfg.py:96` |
| `rewards` | `object` | **your own plain `@configclass`** holding `RewardTermCfg` attrs | `manager_based_rl_env_cfg.py:58` |
| `terminations` | `object` | **your own plain `@configclass`** holding `TerminationTermCfg` attrs | `manager_based_rl_env_cfg.py:64` |
| `events` | `object` | `DefaultEventManagerCfg` (**@configclass**) by default | `manager_based_env_cfg.py:102`; `:27-35` |
| `commands` | `object \| None` | your own plain `@configclass` holding `CommandTermCfg` attrs | `manager_based_rl_env_cfg.py:76` |
| `curriculum` | `object \| None` | your own plain `@configclass` holding `CurriculumTermCfg` attrs | `manager_based_rl_env_cfg.py:70` |
| `recorders` | `object` | `RecorderManagerBaseCfg` (**@configclass**) | `manager_based_env_cfg.py:84`; `:18` |
| `xr` | `XrCfg \| None` (**@configclass**) | `isaaclab.devices.openxr.XrCfg` | `manager_based_env_cfg.py:140`; `:16` |
| `teleop_devices` | `DevicesCfg` (**@configclass**) | via `field(default_factory=DevicesCfg)` | `manager_based_env_cfg.py:143`; `:15` |

**Critical:** `observations`, `actions`, `rewards`, `terminations`, `commands`, `curriculum` are typed
`object`, NOT `ObservationGroupCfg`/etc. `@configclass` validates them by `isinstance` against the *annotation*,
so the loose `object` type is what allows arbitrary user sub-config classes (`manager_based_env_cfg.py:90-100`).

### 2.3 `DefaultEventManagerCfg` (the default reset behaviour)

```python
# isaaclab/isaaclab/envs/manager_based_env_cfg.py:27-35
@configclass
class DefaultEventManagerCfg:
    reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
```

With `@configclass`, the annotation-less assignment is inferred from the value's type. Same pattern works for
your own scene/observation/reward config classes.

### 2.4 `ViewerCfg` — full field list

`isaaclab/isaaclab/envs/common.py:20-68`

```python
@configclass
class ViewerCfg:
    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)                     # :24
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)                   # :27
    cam_prim_path: str = "/OmniverseKit_Persp"                             # :30
    resolution: tuple[int, int] = (1280, 720)                              # :35
    origin_type: Literal["world", "env", "asset_root", "asset_body"] = "world"   # :40
    env_index: int = 0                                                     # :52
    asset_name: str | None = None                                          # :58
    body_name: str | None = None                                           # :64
```

**There are no `viewer.origin_type`-free / flat `viewer.*` fields.** `ViewerCfg` is not `@configclass`-nested
inside itself; you assign `cfg.viewer.origin_type = "asset_root"` in `__post_init__`.

### 2.5 `SimulationCfg` — full field list

`isaaclab/isaaclab/sim/simulation_cfg.py:348-444`

```python
@configclass
class SimulationCfg:
    physics_prim_path: str = "/physicsScene"                       # :352
    device: str = "cuda:0"                                          # :355
    dt: float = 1.0 / 60.0                                          # :365
    render_interval: int = 1                                        # :368
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)         # :371
    enable_scene_query_support: bool = False                        # :377
    use_fabric: bool = True                                         # :392
    physx: PhysxCfg = PhysxCfg()                                    # :412
    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg() # :415
    render: RenderCfg = RenderCfg()                                 # :424
    create_stage_in_memory: bool = False                            # :427
    logging_level: Literal["DEBUG","INFO","WARNING","ERROR","CRITICAL"] = "WARNING"  # :433
    save_logs_to_file: bool = True                                  # :436
    log_dir: str | None = None                                      # :439
```

`PhysxCfg` defaults (`sim/simulation_cfg.py:20-195`):

```python
solver_type: Literal[0, 1] = 1                       # :37
solve_articulation_contact_last: bool = False        # :46   <-- NEW in 2.3, needs Isaac Sim 5.1
min_position_iteration_count: int = 1                # :66
max_position_iteration_count: int = 255              # :76
min_velocity_iteration_count: int = 0                # :86
max_velocity_iteration_count: int = 255              # :96
enable_ccd: bool = False                             # :106
enable_stabilization: bool = False                   # :110
enable_external_forces_every_iteration: bool = False # :123
enable_enhanced_determinism: bool = False            # :136
bounce_threshold_velocity: float = 0.5               # :144
friction_offset_threshold: float = 0.04              # :147
friction_correlation_distance: float = 0.025         # :150
gpu_max_rigid_contact_count: int = 2**23             # :153
gpu_max_rigid_patch_count: int = 5 * 2**15           # :156
gpu_found_lost_pairs_capacity: int = 2**21           # :159
gpu_found_lost_aggregate_pairs_capacity: int = 2**25 # :165
gpu_total_aggregate_pairs_capacity: int = 2**21      # :172
gpu_collision_stack_size: int = 2**26                # :175
gpu_heap_capacity: int = 2**26                       # :178
gpu_temp_buffer_capacity: int = 2**24                # :182
gpu_max_num_partitions: int = 8                      # :185
gpu_max_soft_body_contacts: int = 2**20              # :191
gpu_max_particle_contacts: int = 2**20               # :194
```

### 2.6 How the time-steps relate (needed to set `decimation`)

`isaaclab/isaaclab/envs/manager_based_env.py:120-130`

```python
self.physics_dt                 == cfg.sim.dt
self.step_dt                    == cfg.decimation * cfg.sim.dt
cfg.sim.render_interval         # should normally equal cfg.decimation
# warning logged if cfg.sim.render_interval < cfg.decimation
```

`manager_based_rl_env.py:182` confirms `for _ in range(self.cfg.decimation)` physics steps per env step.

---

## 3. Scene, articulation, spawner configs

### 3.1 `InteractiveSceneCfg`

`isaaclab/isaaclab/scene/interactive_scene_cfg.py:11-126`

```python
@configclass
class InteractiveSceneCfg:
    num_envs: int = MISSING                 # :70
    env_spacing: float = MISSING            # :73
    lazy_sensor_update: bool = True         # :80
    replicate_physics: bool = True          # :87
    filter_collisions: bool = True          # :101
    clone_in_fabric: bool = False           # :114
```

Entity creation order matters: terrain → physics assets (articulations, rigid bodies) → sensors → lights.
— `interactive_scene_cfg.py:18-23`

Terrain and `env_spacing` wiring (this is why `terrain_type="plane"` works without you setting
`TerrainImporterCfg.env_spacing`): `InteractiveScene` overwrites the terrain cfg fields with the scene's own:

```python
# isaaclab/isaaclab/scene/interactive_scene.py:739-742
# terrains are special entities since they define environment origins
asset_cfg.num_envs = self.cfg.num_envs
asset_cfg.env_spacing = self.cfg.env_spacing
self._terrain = asset_cfg.class_type(asset_cfg)
```

`env_origins` property returns `self._terrain.env_origins`. — `interactive_scene.py:368-370`

### 3.2 `AssetBaseCfg` (parent of `ArticulationCfg`)

`isaaclab/isaaclab/assets/asset_base_cfg.py:15-77`

```python
@configclass
class AssetBaseCfg:
    @configclass
    class InitialStateCfg:                                       # :23
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)        # :35
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # :37  (w, x, y, z)

    class_type: type[AssetBase] = None                           # :42
    prim_path: str = MISSING                                     # :49
    spawn: SpawnerCfg | None = None                              # :59
    init_state: InitialStateCfg = InitialStateCfg()              # :66
    collision_group: Literal[0, -1] = 0                          # :69
    debug_vis: bool = False                                      # :76
```

`prim_path` supports `{ENV_REGEX_NS}`, e.g. `"{ENV_REGEX_NS}/Robot"` → `/World/envs/env_.*/Robot`. — `asset_base_cfg.py:52-57`

### 3.3 `ArticulationCfg` + `InitialStateCfg`

`isaaclab/isaaclab/assets/articulation/articulation_cfg.py:15-71`

```python
@configclass
class ArticulationCfg(AssetBaseCfg):
    @configclass
    class InitialStateCfg(AssetBaseCfg.InitialStateCfg):          # :20
        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)     # :24
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)     # :26
        joint_pos: dict[str, float] = {".*": 0.0}                 # :30  regex-keyed dict
        joint_vel: dict[str, float] = {".*": 0.0}                 # :32  regex-keyed dict

    class_type: type = Articulation                                             # :39
    articulation_root_prim_path: str | None = None                              # :41
    init_state: InitialStateCfg = InitialStateCfg()                             # :53
    soft_joint_pos_limit_factor: float = 1.0                                    # :56
    actuators: dict[str, ActuatorBaseCfg] = MISSING                             # :66
    actuator_value_resolution_debug_print = False                               # :69
```

- `soft_joint_pos_limit_factor` scales the parsed joint limits into a safety region; results land in
  `ArticulationData.soft_joint_pos_limits`. — `articulation_cfg.py:56-64`
- `articulation_root_prim_path` must start with `/` and is relative to `prim_path`; `None` = search for
  `ArticulationRootAPI`. — `articulation_cfg.py:41-51`
- **Note** `actuator_value_resolution_debug_print` is declared with no annotation (line 69). It is not
  `@configclass`-converted to a real field via annotation, but it still exists as a class attribute.

Real usage (regex-keyed `joint_pos`):

```python
# isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/digit/rough_env_cfg.py:233
# Don't randomize the initial joint positions because we have closed loops.
```

### 3.4 `UsdFileCfg` / `UrdfFileCfg` / `MjcfFileCfg` — how `asset_path` / `usd_path` is expressed

**Both field names exist at 2.3.2 but on different objects:**

- `UsdFileCfg.usd_path` — the USD reference path. (`from_files_cfg.py:101`) **`UsdFileCfg` has no
  `asset_path`.**
- `UrdfFileCfg.asset_path` / `MjcfFileCfg.asset_path` — inherited **from `AssetConverterBaseCfg`**, not from
  `FileCfg`. (`converters/asset_converter_base_cfg.py:15`) **`UrdfFileCfg` has no `usd_path`.**

`UrdfFileCfg` is declared as `class UrdfFileCfg(FileCfg, converters.UrdfConverterCfg)`
(`from_files_cfg.py:114`), so it inherits the converter's source-file fields *and* the spawner's
spawn-modification fields.

#### `AssetConverterBaseCfg` (source path + output USD location) — full field list

`isaaclab/isaaclab/sim/converters/asset_converter_base_cfg.py:11-48`

```python
@configclass
class AssetConverterBaseCfg:
    asset_path: str = MISSING            # :15   absolute path to the source asset (.urdf / .xml)
    usd_dir: str | None = None           # :18   output dir; None -> /tmp/IsaacLab/usd_{date}_{time}_{random}
    usd_file_name: str | None = None     # :25   None -> derived from asset file name
    force_usd_conversion: bool = False   # :35   True -> always regenerate (overwrites)
    make_instanceable: bool = True       # :41
```

`isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py:20-219`

```python
@configclass
class FileCfg(RigidObjectSpawnerCfg, DeformableObjectSpawnerCfg):     # :21
    scale: tuple[float, float, float] | None = None                  # :36
    articulation_props: schemas.ArticulationRootPropertiesCfg | None = None   # :39
    fixed_tendons_props: schemas.FixedTendonPropertiesCfg | None = None       # :42
    spatial_tendons_props: schemas.SpatialTendonPropertiesCfg | None = None   # :45
    joint_drive_props: schemas.JointDrivePropertiesCfg | None = None          # :48
    visual_material_path: str = "material"                           # :58
    visual_material: materials.VisualMaterialCfg | None = None       # :65


@configclass
class UsdFileCfg(FileCfg):                                           # :74
    func: Callable = from_files.spawn_from_usd                       # :99
    usd_path: str = MISSING                                          # :101   <-- the path field
    variants: object | dict[str, str] | None = None                  # :104


@configclass
class UrdfFileCfg(FileCfg, converters.UrdfConverterCfg):             # :114
    func: Callable = from_files.spawn_from_urdf                      # :132


@configclass
class MjcfFileCfg(FileCfg, converters.MjcfConverterCfg):             # :136
    func: Callable = from_files.spawn_from_mjcf                      # :154


@configclass
class GroundPlaneCfg(SpawnerCfg):                                    # :198
    func: Callable = from_files.spawn_ground_plane                   # :204
    usd_path: str = f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd"  # :206
    color: tuple[float, float, float] | None = (0.0, 0.0, 0.0)       # :209
    size: tuple[float, float] = (100.0, 100.0)                       # :215
    physics_material: materials.RigidBodyMaterialCfg = materials.RigidBodyMaterialCfg()  # :218
```

Exported publicly as `isaaclab.sim.UrdfFileCfg` etc. — `isaaclab/isaaclab/sim/spawners/from_files/__init__.py:16-23`

`UrdfConverterCfg` own fields (all inherited by `UrdfFileCfg`) —
`isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py:90-132`:

```python
fix_base: bool = MISSING                       # :90
root_link_name: str | None = None              # :93
link_density: float = 0.0                      # :99
merge_fixed_joints: bool = True                # :104
convert_mimic_joints_to_normal_joints: bool = False   # :107
joint_drive: JointDriveCfg | None = JointDriveCfg()   # :110
collision_from_visuals = False                 # :116  (no annotation)
collider_type: Literal["convex_hull", "convex_decomposition"] = "convex_hull"  # :119
self_collision: bool = False                   # :128
replace_cylinders_with_capsules: bool = False  # :131
```

Nested `UrdfConverterCfg.JointDriveCfg` (`urdf_converter_cfg.py:17-88`):

```python
@configclass
class JointDriveCfg:
    @configclass
    class PDGainsCfg:                                    # :22
        stiffness: dict[str, float] | float = MISSING    # :25  regex dict allowed
        damping: dict[str, float] | float | None = None  # :33  regex dict allowed

    @configclass
    class NaturalFrequencyGainsCfg:                      # :42
        natural_frequency: dict[str, float] | float = MISSING   # :57
        damping_ratio: dict[str, float] | float = 0.005         # :64

    drive_type: dict[str, Literal["acceleration", "force"]] | Literal["acceleration", "force"] = "force"  # :71
    target_type: dict[str, Literal["none", "position", "velocity"]] | Literal["none", "position", "velocity"] = "position"  # :79
    gains: PDGainsCfg | NaturalFrequencyGainsCfg = PDGainsCfg()   # :87
```

URL fallback: a missing `.../4.5/...` USD path is retried as `/5.0/` — `from_files.py:300-307`.

### 3.5 `ArticulationRootPropertiesCfg` (what `spawn=UsdFileCfg(articulation_props=...)` accepts)

`isaaclab/isaaclab/sim/schemas/schemas_cfg.py:14-54`

```python
@configclass
class ArticulationRootPropertiesCfg:
    articulation_enabled: bool | None = None                 # :25
    enabled_self_collisions: bool | None = None              # :28
    solver_position_iteration_count: int | None = None       # :31
    solver_velocity_iteration_count: int | None = None       # :34
    sleep_threshold: float | None = None                     # :37
    stabilization_threshold: float | None = None             # :40
    fix_root_link: bool | None = None                        # :43
```

**There is no `exclude_from_articulation` field here** — see section 12.

### 3.6 Complete, usable floating-base robot asset block

```python
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

MY_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="D:/Robomaster/RL/isaac_ws/assets/infantry-rl/infantry_rl.usd",
        activate_contact_sensors=True,            # see RigidObjectSpawnerCfg
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
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            fix_root_link=False,                  # <-- floating base
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={".*": 0.0},                    # regex-keyed
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_wheel"],
            effort_limit_sim=20.0,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=2.5,
            friction=0.0,
            saturation_effort=20.0,
        ),
        "legs": DCMotorCfg(
            joint_names_expr=[".*_(ham|calf)"],
            effort_limit_sim=40.0,
            velocity_limit_sim=20.0,
            stiffness=60.0,
            damping=2.0,
            armature=0.01,
            saturation_effort=40.0,
        ),
    },
)
```

To spawn **from URDF** instead (`asset_path`, not `usd_path`):

```python
spawn=sim_utils.UrdfFileCfg(
    # --- from AssetConverterBaseCfg (converters/asset_converter_base_cfg.py:15-48) ---
    asset_path="D:/Robomaster/RL/isaac_ws/assets/infantry-rl/urdf/infantry-rl-sim.urdf",
    usd_dir="D:/Robomaster/RL/isaac_ws/assets/infantry-rl/converted",   # None -> /tmp/IsaacLab/usd_{date}_{time}_{random}
    usd_file_name="infantry_rl.usd",                                   # None -> derived from asset file name
    force_usd_conversion=True,
    make_instanceable=True,
    # --- from UrdfConverterCfg (converters/urdf_converter_cfg.py:90-132) ---
    fix_base=False,                            # floating base
    root_link_name=None,
    link_density=0.0,
    merge_fixed_joints=False,                  # keep fixed-joint links separate
    convert_mimic_joints_to_normal_joints=True,   # flatten mimic joints -> normal joints
    collider_type="convex_hull",
    self_collision=False,
    replace_cylinders_with_capsules=False,
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
    ),
    # --- from FileCfg (spawners/from_files/from_files_cfg.py:20-70) ---
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False),
    joint_drive_props=None,
)
```

> **Verified:** `asset_path` (`asset_converter_base_cfg.py:15`), `usd_dir` (`:18`), `usd_file_name` (`:25`),
> `force_usd_conversion` (`:35`), `make_instanceable` (`:41`), and all `UrdfConverterCfg` fields above.
> `collision_from_visuals` and `actuator_value_resolution_debug_print` are declared **without annotations**
> (`urdf_converter_cfg.py:116`; `articulation_cfg.py:69`) — they still exist as class attributes but are not
> regular typed config fields.

---

## 4. `isaaclab.actuators`

### 4.1 Public exports

`isaaclab/isaaclab/actuators/__init__.py:25-36`

```python
from .actuator_base import ActuatorBase
from .actuator_base_cfg import ActuatorBaseCfg
from .actuator_net import ActuatorNetLSTM, ActuatorNetMLP
from .actuator_net_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg
from .actuator_pd import DCMotor, DelayedPDActuator, IdealPDActuator, ImplicitActuator, RemotizedPDActuator
from .actuator_pd_cfg import (
    DCMotorCfg, DelayedPDActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg, RemotizedPDActuatorCfg,
)
```

### 4.2 `ActuatorBaseCfg` — exact fields

`isaaclab/isaaclab/actuators/actuator_base_cfg.py:11-165`

```python
@configclass
class ActuatorBaseCfg:
    class_type: type = MISSING                                    # :15
    joint_names_expr: list[str] = MISSING                         # :21
    effort_limit:         dict[str, float] | float | None = None  # :28
    velocity_limit:       dict[str, float] | float | None = None  # :51
    effort_limit_sim:     dict[str, float] | float | None = None  # :74
    velocity_limit_sim:   dict[str, float] | float | None = None  # :91
    stiffness:            dict[str, float] | float | None = MISSING  # :107
    damping:              dict[str, float] | float | None = MISSING  # :117
    armature:             dict[str, float] | float | None = None  # :127
    friction:             dict[str, float] | float | None = None  # :138
    dynamic_friction:     dict[str, float] | float | None = None  # :156
    viscous_friction:     dict[str, float] | float | None = None  # :163
```

**Regex-keyed dict support:** every one of `effort_limit`, `velocity_limit`, `effort_limit_sim`,
`velocity_limit_sim`, `stiffness`, `damping`, `armature`, `friction`, `dynamic_friction`,
`viscous_friction` is annotated `dict[str, float] | float | None`, i.e. it accepts **either a scalar or a
regex-keyed dict**. `joint_names_expr` is a `list[str]` of joint names **or regexes**.
— `actuator_base_cfg.py:21-26`, `:28`, `:51`, `:74`, `:91`, `:107`, `:117`, `:127`, `:138`, `:156`, `:163`

Semantics (from the docstrings):

- `effort_limit` clips the **actuator model output**, only for **explicit** actuators. — `:29-49`
- `effort_limit_sim` goes to the **physics solver**; if `None`, implicit actuators use the USD joint prim value and
  explicit actuators default to `1.0e9`. — `:74-89`
- `velocity_limit` is **not used** by implicit actuators (backwards compat). — `:66-72`
- `velocity_limit_sim` goes to the physics solver; if `None`, USD joint prim value is used for both kinds. — `:91-105`
- `friction` is a *coefficient* in Isaac Sim 4.5 but an *effort (torque/force)* in Isaac Sim 5.0+. — `:152-154`

### 4.3 The four config classes and their `class_type`

`isaaclab/isaaclab/actuators/actuator_pd_cfg.py:18-81`

```python
@configclass
class ImplicitActuatorCfg(ActuatorBaseCfg):        # :19
    class_type: type = actuator_pd.ImplicitActuator   # :26

@configclass
class IdealPDActuatorCfg(ActuatorBaseCfg):         # :35
    class_type: type = actuator_pd.IdealPDActuator    # :38

@configclass
class DCMotorCfg(IdealPDActuatorCfg):              # :42
    class_type: type = actuator_pd.DCMotor            # :45
    saturation_effort: float = MISSING                # :47   N-m

@configclass
class DelayedPDActuatorCfg(IdealPDActuatorCfg):    # :52
    class_type: type = actuator_pd.DelayedPDActuator  # :55
    min_delay: int = 0                                # :57   physics steps
    max_delay: int = 0                                # :60   physics steps

@configclass
class RemotizedPDActuatorCfg(DelayedPDActuatorCfg):  # :65
    class_type: type = actuator_pd.RemotizedPDActuator  # :74
    joint_parameter_lookup: list[list[float]] = MISSING  # :76  shape (N, 3): (angle rad, ratio, torque N*m)
```

All of the above are proper `@configclass` classes and all inherit `ActuatorBaseCfg`, so all base fields apply.

**`DCMotorCfg` is the only one that adds `saturation_effort`** — `actuator_pd_cfg.py:47-48`.
`IdealPDActuatorCfg` adds **nothing**; it is a pure `class_type` override (`:35-38`).

### 4.4 Deprecations in 2.3.2 (exact behaviour)

1. **`isaaclab.actuators.actuator_cfg` module is deprecated as a whole.** Importing any name from it emits a
   `DeprecationWarning` and forwards to the new module:
   `isaaclab/isaaclab/actuators/actuator_cfg.py:12-36`
   ```python
   warnings.warn(
       f"The module actuator_cfg.py is deprecated. Please import {name} directly from the isaaclab.actuators"
       f" package, or from its new module {new_module.__name__}.",
       DeprecationWarning, stacklevel=2,
   )
   ```
   **Use `from isaaclab.actuators import DCMotorCfg`, never `from isaaclab.actuators.actuator_cfg import DCMotorCfg`.**

2. **`ImplicitActuatorCfg.effort_limit` is deprecated in favour of `effort_limit_sim`.**
   `isaaclab/isaaclab/actuators/actuator_pd.py:60-78`
   ```python
   if cfg.effort_limit_sim is None and cfg.effort_limit is not None:
       logger.warning(
           "The <ImplicitActuatorCfg> object has a value for 'effort_limit'."
           " This parameter will be removed in the future."
           " To set the effort limit, please use 'effort_limit_sim' instead."
       )
       cfg.effort_limit_sim = cfg.effort_limit
   elif cfg.effort_limit_sim is not None and cfg.effort_limit is None:
       cfg.effort_limit = cfg.effort_limit_sim
   elif cfg.effort_limit_sim is not None and cfg.effort_limit is not None:
       if cfg.effort_limit_sim != cfg.effort_limit:
           raise ValueError(...)   # "Please only set 'effort_limit_sim' for implicit actuators."
   ```

3. **`ImplicitActuatorCfg.velocity_limit` is deprecated and is actively NULLED OUT.**
   `actuator_pd.py:81-102`
   ```python
   if cfg.velocity_limit_sim is None and cfg.velocity_limit is not None:
       logger.warning(
           "The <ImplicitActuatorCfg> object has a value for 'velocity_limit'."
           " Previously, although this value was specified, it was not getting used by implicit"
           " actuators. ... To set the velocity limit, please use 'velocity_limit_sim' instead."
       )
       cfg.velocity_limit = None          # <-- the value you set is DISCARDED
   ```
   Setting `velocity_limit` on an **implicit** actuator silently has no effect. Use `velocity_limit_sim`.

4. **Deprecated in 2.3.1** (adjacent, relevant to training scripts):
   `ManagerBasedEnvCfg.rerender_on_reset` → `num_rerenders_on_reset`
   (`envs/manager_based_env_cfg.py:118-124`; warning at `envs/manager_based_env.py:203`).
   `DirectRLEnvCfg.rerender_on_reset` likewise (`envs/direct_rl_env.py:230`).

5. **Deprecated in 2.3**: `Articulation/RigidObject/RigidObjectCollection.set_external_force_and_torque`
   in favour of `...permanent_wrench_composer.set_forces_and_torques` (`isaaclab/docs/CHANGELOG.rst:90-92`).

6. **Not deprecated** at 2.3.2: `actuator_net_cfg.py` (`ActuatorNetLSTMCfg`, `ActuatorNetMLPCfg`) — no
   deprecation path exists for them in `actuator_cfg.py`'s forwarding logic other than the module warning.

---

## 5. `isaaclab.managers` — term cfg signatures

### 5.1 What `isaaclab.managers` exports

`isaaclab/isaaclab/managers/__init__.py:13-34`

```python
from .action_manager import ActionManager, ActionTerm
from .command_manager import CommandManager, CommandTerm
from .curriculum_manager import CurriculumManager
from .event_manager import EventManager
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import (
    ActionTermCfg, CommandTermCfg, CurriculumTermCfg, EventTermCfg, ManagerTermBaseCfg,
    ObservationGroupCfg, ObservationTermCfg, RecorderTermCfg, RewardTermCfg, TerminationTermCfg,
)
from .observation_manager import ObservationManager
from .recorder_manager import DatasetExportMode, RecorderManager, RecorderManagerBaseCfg, RecorderTerm
from .reward_manager import RewardManager
from .scene_entity_cfg import SceneEntityCfg
from .termination_manager import TerminationManager
```

### 5.2 `ManagerTermBaseCfg` (base for most term cfgs)

`isaaclab/isaaclab/managers/manager_term_cfg.py:29-53`

```python
@configclass
class ManagerTermBaseCfg:
    func: Callable | ManagerTermBase = MISSING                        # :33
    params: dict[str, Any | SceneEntityCfg] = dict()                  # :46
```

`params` values may be `SceneEntityCfg` objects; the manager resolves them against the scene and injects
`joint_ids` / `body_ids`. — `manager_term_cfg.py:48-53`

### 5.3 `ObservationTermCfg` / `ObservationGroupCfg`

```python
# isaaclab/isaaclab/managers/manager_term_cfg.py:146-195
@configclass
class ObservationTermCfg(ManagerTermBaseCfg):
    func: Callable[..., torch.Tensor] = MISSING                       # :150  -> (num_envs, obs_term_dim)
    modifiers: list[ModifierCfg] | None = None                        # :158
    noise: NoiseCfg | NoiseModelCfg | None = None                     # :169
    clip: tuple[float, float] | None = None                           # :172
    scale: tuple[float, ...] | float | None = None                     # :176
    history_length: int = 0                                           # :184
    flatten_history_dim: bool = True                                  # :193
```

```python
# isaaclab/isaaclab/managers/manager_term_cfg.py:198-242
@configclass
class ObservationGroupCfg:
    concatenate_terms: bool = True            # :202
    concatenate_dim: int = -1                 # :211
    enable_corruption: bool = False           # :221
    history_length: int | None = None         # :228   overrides per-term history
    flatten_history_dim: bool = True          # :236
```

Pipeline order inside a term: **modifiers → noise → clip → scale** — `managers/observation_manager.py:354-357`.

### 5.4 `ActionTermCfg` — **there is NO `ActionGroupCfg`**

`isaaclab/isaaclab/managers/manager_term_cfg.py:77-98`

```python
@configclass
class ActionTermCfg:
    class_type: type[ActionTerm] = MISSING    # :81
    asset_name: str = MISSING                 # :87
    debug_vis: bool = False                   # :94
    clip: dict[str, tuple] | None = None      # :97
```

The `ActionManager` takes a **flat** mapping `dict[str, ActionTermCfg]` (or a config object whose `__dict__`
holds them) — there is no grouping layer. — `managers/action_manager.py:198-213`, `:443-445`

**`default_name` does not exist at 2.3.2.** Exhaustive grep for `default_name` across
`D:\IsaacLab\source\isaaclab\isaaclab\**\*.py` returns **zero matches**. Use `asset_name`.
`ActionTerm.__init__` does `self._asset = self._env.scene[self.cfg.asset_name]`.
— `managers/action_manager.py:54-55`

### 5.5 `RewardTermCfg`

`isaaclab/isaaclab/managers/manager_term_cfg.py:310-330`

```python
@configclass
class RewardTermCfg(ManagerTermBaseCfg):
    func: Callable[..., torch.Tensor] = MISSING   # :314   -> (num_envs,)
    weight: float = MISSING                       # :322
```

### 5.6 `TerminationTermCfg`

`isaaclab/isaaclab/managers/manager_term_cfg.py:338-355`

```python
@configclass
class TerminationTermCfg(ManagerTermBaseCfg):
    func: Callable[..., torch.Tensor] = MISSING   # :342   -> bool (num_envs,)
    time_out: bool = False                        # :350
```

### 5.7 `EventTermCfg`

`isaaclab/isaaclab/managers/manager_term_cfg.py:250-302`

```python
@configclass
class EventTermCfg(ManagerTermBaseCfg):
    func: Callable[..., None] = MISSING                        # :254
    mode: str = MISSING                                        # :261
    interval_range_s: tuple[float, float] | None = None         # :269  (mode == "interval")
    is_global_time: bool = False                               # :280  (mode == "interval")
    min_step_count_between_reset: int = 0                     # :291  (mode == "reset")
```

`mode` values seen in source: `"startup"`, `"reset"`, `"interval"`, `"custom"`, `"prestartup"`.
Modes available are discovered via `event_manager.available_modes` (`envs/manager_based_env.py:161`,
`envs/manager_based_rl_env.py:134`).

### 5.8 `CommandTermCfg`, `CurriculumTermCfg`, `RecorderTermCfg`

```python
# isaaclab/isaaclab/managers/manager_term_cfg.py:106-119
@configclass
class CommandTermCfg:
    class_type: type[CommandTerm] = MISSING          # :109
    resampling_time_range: tuple[float, float] = MISSING  # :116
    debug_vis: bool = False                          # :118

# :127-138
@configclass
class CurriculumTermCfg(ManagerTermBaseCfg):
    func: Callable[..., float | dict[str, float] | None] = MISSING   # :131

# :61-69
@configclass
class RecorderTermCfg:
    class_type: type[RecorderTerm] = MISSING         # :65
```

### 5.9 `SceneEntityCfg` — full field list

`isaaclab/isaaclab/managers/scene_entity_cfg.py:20-116`

```python
@configclass
class SceneEntityCfg:
    name: str = MISSING                                        # :28
    joint_names: str | list[str] | None = None                 # :35   regex allowed
    joint_ids: list[int] | slice = slice(None)                 # :44
    fixed_tendon_names: str | list[str] | None = None          # :52   regex allowed
    fixed_tendon_ids: list[int] | slice = slice(None)          # :61
    body_names: str | list[str] | None = None                  # :69   regex allowed
    body_ids: list[int] | slice = slice(None)                  # :78
    object_collection_names: str | list[str] | None = None     # :86   regex allowed
    object_collection_ids: list[int] | slice = slice(None)     # :95
    preserve_order: bool = False                               # :102
```

- `joint_ids` defaults to `slice(None)` = all joints. — `:44-50`
- If `*_names` is given, `*_ids` is filled automatically; if both are given and inconsistent → `ValueError`.
  — `scene_entity_cfg.py:152-184`, `:240-245`
- Performance optimization: if the resolved names equal the entity's full ordered list, `joint_ids` is
  reset to `slice(None)`. — `:176-179`
- **`preserve_order=True`** keeps your regex/name order instead of the entity's ascending joint order
  (delegates to `isaaclab.utils.string.resolve_matching_names`). — `:102-116`

### 5.10 `ObsTerm` / `RewTerm` / `TermCfg` aliases — how they are actually exposed

**They are NOT exported by `isaaclab.managers` and not by `isaaclab.envs.mdp`.** Every occurrence in the
tree is a *local* `import ... as ...` alias inside the task file. The canonical aliases are:

```python
# isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:12-18
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
```

There is **no `TermCfg`** name at all. Aliases found in the source tree:
`ObsGroup` (`velocity_env_cfg.py:14`), `ObsTerm` (`:15`), `RewTerm` (`:16`), `DoneTerm` (`:18`),
`EventTerm` (`:13`), `CurrTerm` (`:12`). `ActionTermCfg` is imported un-aliased
(`test/envs/check_manager_based_env_floating_cube.py:43`).

`isaaclab.envs.mdp` re-exports the term *functions and action classes*, not the cfgs:

```python
# isaaclab/isaaclab/envs/mdp/__init__.py:18-25
from .actions import *
from .commands import *
from .curriculums import *
from .events import *
from .observations import *
from .recorders import *
from .rewards import *
from .terminations import *
```

```python
# isaaclab/isaaclab/envs/mdp/actions/__init__.py:8-13
from .actions_cfg import *
from .binary_joint_actions import *
from .joint_actions import *
from .joint_actions_to_limits import *
from .non_holonomic_actions import *
from .surface_gripper_actions import *
```

So `mdp.JointPositionActionCfg` / `mdp.JointVelocityActionCfg` / `mdp.JointEffortActionCfg` **exist**
(because `actions_cfg.py` is star-imported), and `mdp.base_ang_vel` etc. exist. You must import the *term
cfgs* from `isaaclab.managers`.

---

## 6. `isaaclab.envs.mdp` — available terms

### 6.1 Observation terms (`isaaclab/isaaclab/envs/mdp/observations.py`)

| Term | Line | Signature | Returns |
|---|---|---|---|
| `base_pos_z` | `:44` | `(env, asset_cfg=SceneEntityCfg("robot"))` | root z, `(N, 1)`, m |
| `base_lin_vel` | `:54` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `root_lin_vel_b`, `(N, 3)`, m/s |
| `base_ang_vel` | `:64` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `root_ang_vel_b`, `(N, 3)`, rad/s |
| `projected_gravity` | `:74` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `projected_gravity_b`, `(N, 3)`, unit vector |
| `root_pos_w` | `:84` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `root_pos_w - env.scene.env_origins`, `(N, 3)`, m |
| `root_quat_w` | `:94` | `(env, make_quat_unique=False, asset_cfg=SceneEntityCfg("robot"))` | `(N, 4)` `(w,x,y,z)` |
| `root_lin_vel_w` | `:114` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `(N, 3)`, m/s |
| `root_ang_vel_w` | `:124` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `(N, 3)`, rad/s |
| `body_pose_w` | `:137` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `(N, 7*num_bodies)`, `[x,y,z,qw,qx,qy,qz]` per body, env-frame |
| `body_projected_gravity_b` | `:165` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `(N, 3*num_bodies)` |
| `joint_pos` | `:197` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `(N, num_joints)`, rad |
| `joint_pos_rel` | `:212` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `joint_pos - default_joint_pos`, `(N, J)`, rad |
| `joint_pos_limit_normalized` | `:223` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `(N, J)` in `[-1, 1]` |
| `joint_vel` | `:242` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `(N, J)`, rad/s |
| `joint_vel_rel` | `:257` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `joint_vel - default_joint_vel`, `(N, J)`, rad/s |
| `joint_effort` | `:270` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `applied_torque`, `(N, J)`, N·m |
| `height_scan` | `:292` | `(env, sensor_cfg: SceneEntityCfg, offset: float = 0.5)` | sensor z − ray hit z − offset, `(N, num_rays)`, m |
| `body_incoming_wrench` | `:303` | `(env, asset_cfg: SceneEntityCfg)` | `(N, 6*num_bodies)`, world frame |
| `imu_orientation` | `:315` | `(env, asset_cfg=SceneEntityCfg("imu"))` | `(N, 4)` |
| `imu_projected_gravity` | `:331` | `(env, asset_cfg=SceneEntityCfg("imu"))` | `(N, 3)` |
| `imu_ang_vel` | `:346` | `(env, asset_cfg=SceneEntityCfg("imu"))` | `(N, 3)` |
| `imu_lin_acc` | `:362` | `(env, asset_cfg=SceneEntityCfg("imu"))` | `(N, 3)`, m/s² |
| `image` | `:376` | `(env, sensor_cfg=SceneEntityCfg("tiled_camera"), data_type="rgb", convert_perspective_to_orthogonal=False, normalize=True)` | camera image |
| `image_features` | `:427` | class `image_features(ManagerTermBase)` | features |
| `last_action` | `:657` | `(env, action_name: str \| None = None)` | `action_manager.action` or `get_term(name).raw_actions` |
| `generated_commands` | `:675` | `(env, command_name: str \| None = None)` | `command_manager.get_command(name)` |
| `current_time_s` | `:685` | `(env)` | `episode_length_buf * step_dt`, `(N,)` |
| `remaining_time_s` | `:690` | `(env)` | `max_episode_length_s - elapsed`, `(N,)` |

Note `last_action`: with `action_name=None` you get the **processed/raw concatenation of all terms**
(`env.action_manager.action`), not per-term raw actions — `observations.py:663-666`.

### 6.2 Action terms (`isaaclab/isaaclab/envs/mdp/actions/`)

Classes (used as `class_type` inside the cfg classes):

| Class | File:line |
|---|---|
| `JointAction` (base) | `joint_actions.py:28` |
| `JointPositionAction` | `joint_actions.py:184` |
| `RelativeJointPositionAction` | `joint_actions.py:202` |
| `JointVelocityAction` | `joint_actions.py:235` |
| `JointEffortAction` | `joint_actions.py:253` |
| `JointPositionToLimitsAction` | `joint_actions_to_limits.py` |
| `EMAJointPositionToLimitsAction` | `joint_actions_to_limits.py` |
| `BinaryJointPositionAction`, `BinaryJointVelocityAction`, `AbsBinaryJointPositionAction` | `binary_joint_actions.py` |
| `NonHolonomicAction` | `non_holonomic_actions.py` |
| `DifferentialInverseKinematicsAction`, `OperationalSpaceControllerAction` | `task_space_actions.py` |
| `SurfaceGripperBinaryAction` | `surface_gripper_actions.py` |

Cfg classes, all in `isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py`:

```python
JointActionCfg                     # :27
JointPositionActionCfg             # :44   <-- the alias EXISTS (from isaaclab.envs.mdp import JointPositionActionCfg)
RelativeJointPositionActionCfg     # :62
JointVelocityActionCfg             # :78
JointEffortActionCfg               # :95
JointPositionToLimitsActionCfg     # :110
EMAJointPositionToLimitsActionCfg  # :139
BinaryJointActionCfg               # :160
BinaryJointPositionActionCfg       # :175
BinaryJointVelocityActionCfg       # :185
AbsBinaryJointPositionActionCfg    # :195
NonHolonomicActionCfg              # :235
DifferentialInverseKinematicsActionCfg  # :263
OperationalSpaceControllerActionCfg     # :300
SurfaceGripperBinaryActionCfg      # :366
```

### 6.3 Reward terms (`isaaclab/isaaclab/envs/mdp/rewards.py`)

| Term | Line | Signature | Returns |
|---|---|---|---|
| `is_alive` | `:32` | `(env)` | `(~terminated).float()`, `(N,)` |
| `is_terminated` | `:37` | `(env)` | `terminated.float()`, `(N,)` |
| `is_terminated_term` | `:42` | class `(ManagerTermBase)`; `__call__(env, term_keys: str \| list[str] = ".*")` | `(N,)`, count of non-timeout terminations |
| `lin_vel_z_l2` | `:77` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `root_lin_vel_b[:,2]**2`, `(N,)` |
| `ang_vel_xy_l2` | `:84` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `sum(root_ang_vel_b[:,:2]**2)`, `(N,)` |
| `flat_orientation_l2` | `:91` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `sum(projected_gravity_b[:,:2]**2)`, `(N,)` |
| `base_height_l2` | `:101` | `(env, target_height: float, asset_cfg=SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg \| None = None)` | `(root_pos_w[:,2] - target)**2`, `(N,)` |
| `body_lin_acc_l2` | `:126` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `(N,)` |
| `joint_torques_l2` | `:137` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `sum(applied_torque[:,jids]**2)`, `(N,)` |
| `joint_vel_l1` | `:149` | `(env, asset_cfg: SceneEntityCfg)` | `sum(abs(joint_vel))`, `(N,)` |
| `joint_vel_l2` | `:156` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `sum(joint_vel**2)`, `(N,)` |
| `joint_acc_l2` | `:168` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `sum(joint_acc**2)`, `(N,)` |
| `joint_deviation_l1` | `:180` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `sum(abs(joint_pos - default))`, `(N,)` |
| `joint_pos_limits` | `:189` | `(env, asset_cfg=SceneEntityCfg("robot"))` | soft-limit violation sum, `(N,)` |
| `joint_vel_limits` | `:206` | `(env, soft_ratio: float, asset_cfg=SceneEntityCfg("robot"))` | violation clipped to `[0,1]` per joint, `(N,)` |
| `applied_torque_limits` | `:233` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `(N,)`; **explicit actuators only** |
| `action_rate_l2` | `:252` | `(env)` | `sum((action - prev_action)**2)`, `(N,)` |
| `action_l2` | `:257` | `(env)` | `sum(action**2)`, `(N,)` |
| `undesired_contacts` | `:267` | `(env, threshold: float, sensor_cfg: SceneEntityCfg)` | count of violating bodies, `(N,)` |
| `desired_contacts` | `:278` | `(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0)` | `1.0` if no desired contact, `(N,)` |
| `contact_forces` | `:288` | `(env, threshold: float, sensor_cfg: SceneEntityCfg)` | sum of force violations, `(N,)` |
| `track_lin_vel_xy_exp` | `:304` | `(env, std: float, command_name: str, asset_cfg=SceneEntityCfg("robot"))` | `exp(-err/std²)`, `(N,)` |
| `track_ang_vel_z_exp` | `:318` | `(env, std: float, command_name: str, asset_cfg=SceneEntityCfg("robot"))` | `exp(-err/std²)`, `(N,)` |

**There is NO `base_height` and NO `upright` reward at 2.3.2.** Grep of `isaaclab/envs/mdp` for
`^def base_height` / `^def upright` yields only `rewards.py:101: def base_height_l2`. Use
`base_height_l2(target_height=...)` and `flat_orientation_l2`.

### 6.4 Termination terms (`isaaclab/isaaclab/envs/mdp/terminations.py`)

| Term | Line | Signature | Returns |
|---|---|---|---|
| `time_out` | `:31` | `(env)` | `episode_length_buf >= max_episode_length`, `(N,)` bool |
| `command_resample` | `:36` | `(env, command_name: str, num_resamples: int = 1)` | `(N,)` bool |
| `bad_orientation` | `:51` | `(env, limit_angle: float, asset_cfg=SceneEntityCfg("robot"))` | `(N,)` bool |
| `root_height_below_minimum` | `:63` | `(env, minimum_height: float, asset_cfg=SceneEntityCfg("robot"))` | `(N,)` bool; **flat terrain only** |
| `joint_pos_out_of_limit` | `:81` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `(N,)` bool, soft limits |
| `joint_pos_out_of_manual_limit` | `:94` | `(env, bounds: tuple[float,float], asset_cfg=SceneEntityCfg("robot"))` | `(N,)` bool |
| `joint_vel_out_of_limit` | `:112` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `(N,)` bool, soft vel limits |
| `joint_vel_out_of_manual_limit` | `:121` | `(env, max_velocity: float, asset_cfg=SceneEntityCfg("robot"))` | `(N,)` bool |
| `joint_effort_out_of_limit` | `:131` | `(env, asset_cfg=SceneEntityCfg("robot"))` | `(N,)` bool |
| `illegal_contact` | `:154` | `(env, threshold: float, sensor_cfg: SceneEntityCfg)` | `(N,)` bool |

**There is NO termination term called `base_contact`.** The 2.3.2 convention is to name your own term
`base_contact` and point it at `mdp.illegal_contact`:

```python
# isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:271-274
base_contact = DoneTerm(
    func=mdp.illegal_contact,
    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
)
```

### 6.5 Event terms (`isaaclab/isaaclab/envs/mdp/events.py`)

```python
randomize_rigid_body_scale                   # :45
randomize_rigid_body_material   (class)      # :155
randomize_rigid_body_mass       (class)      # :286
randomize_rigid_body_com                     # :400
randomize_rigid_body_collider_offsets        # :441
randomize_physics_scene_gravity              # :498
randomize_actuator_gains        (class)      # :541
randomize_joint_parameters      (class)      # :652
randomize_fixed_tendon_parameters (class)    # :838
apply_external_force_torque                  # :1010
push_by_setting_velocity                     # :1046
reset_root_state_uniform                     # :1074
reset_root_state_with_random_orientation     # :1119
reset_root_state_from_terrain                # :1171
reset_joints_by_scale                        # :1238
reset_joints_by_offset                       # :1278
reset_nodal_state_uniform                    # :1318
reset_scene_to_default                       # :1360
randomize_visual_texture_material (class)    # :1400
randomize_visual_color          (class)      # :1567
```

`reset_scene_to_default(env, env_ids, reset_joint_targets: bool = False)` — `events.py:1360`

### 6.6 Curriculum terms (`isaaclab/isaaclab/envs/mdp/curriculums.py`)

```python
modify_reward_weight  (class)     # :24
modify_env_param      (class)     # :50
modify_term_cfg       (class)     # :259   subclasses modify_env_param
```

`modify_env_param.NO_CHANGE` sentinel — `curriculums.py:122`

### 6.7 Command terms (`isaaclab/isaaclab/envs/mdp/commands/commands_cfg.py`)

```python
NullCommandCfg              # :21
UniformVelocityCommandCfg   # :33    (+ nested Ranges at :81)
NormalVelocityCommandCfg    # :100   (+ Ranges at :128)
UniformPoseCommandCfg       # :133   (+ Ranges at :172)
UniformPose2dCommandCfg     # :189   (+ Ranges at :219)
TerrainBasedPose2dCommandCfg # :232
```

`UniformVelocityCommandCfg` fields (`commands_cfg.py:33-98`):

```python
class_type: type = UniformVelocityCommand        # :36
asset_name: str = MISSING                        # :38
heading_command: bool = False                    # :41
heading_control_stiffness: float = 1.0           # :49
rel_standing_envs: float = 0.0                   # :52
rel_heading_envs: float = 1.0                    # :55
ranges: Ranges = MISSING                         # :81
goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(...)  # :84
current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(...) # :89
```

(Range sub-fields are at `commands_cfg.py` between lines 55 and 81; read that block before relying on them.)

---

## 7. `isaaclab.terrains`

### 7.1 `TerrainImporterCfg` — full field list

`isaaclab/isaaclab/terrains/terrain_importer_cfg.py:20-112`

```python
@configclass
class TerrainImporterCfg:
    class_type: type = TerrainImporter                                  # :24
    collision_group: int = -1                                           # :30
    prim_path: str = MISSING                                            # :33
    num_envs: int = 1                                                   # :39  overwritten by scene
    terrain_type: Literal["generator", "plane", "usd"] = "generator"    # :46
    terrain_generator: TerrainGeneratorCfg | None = None                # :52  generator only
    usd_path: str | None = None                                         # :58  "usd" only
    env_spacing: float | None = None                                    # :64  "plane"/"usd"/grid
    use_terrain_origins: bool = True                                    # :72  "generator" only
    visual_material: sim_utils.VisualMaterialCfg | None = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,0.0,0.0))  # :80
    physics_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg()  # :91
    max_init_terrain_level: int | None = None                           # :100 sub-terrain origins
    debug_vis: bool = False                                             # :111
```

### 7.2 What `terrain_type="plane"` actually requires — confirmed from source

```python
# isaaclab/isaaclab/terrains/terrain_importer.py:108-114
elif self.cfg.terrain_type == "plane":
    # load the plane
    self.import_ground_plane("terrain")
    # configure the origins in a grid
    self.configure_env_origins()
else:
    raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")
```

`configure_env_origins()` with no `terrain_origins` requires `cfg.env_spacing`:

```python
# isaaclab/isaaclab/terrains/terrain_importer.py:306-312
else:
    self.terrain_origins = None
    # check if env spacing is valid
    if self.cfg.env_spacing is None:
        raise ValueError("Environment spacing must be specified for configuring grid-like origins.")
    # compute environment origins
    self.env_origins = self._compute_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing)
```

**Resolution:** `TerrainImporterCfg(prim_path=..., terrain_type="plane", physics_material=...)` is
**sufficient**, because `InteractiveScene` injects `scene.cfg.env_spacing` into the terrain cfg before
constructing it:

```python
# isaaclab/isaaclab/scene/interactive_scene.py:740-742
asset_cfg.num_envs = self.cfg.num_envs
asset_cfg.env_spacing = self.cfg.env_spacing
self._terrain = asset_cfg.class_type(asset_cfg)
```

So the only hard requirement is that your `InteractiveSceneCfg` subclass sets `env_spacing`.
Setting `TerrainImporterCfg.env_spacing` yourself is redundant (and would be overwritten).

### 7.3 Flat-ground pattern (exact, working form)

```python
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class MyFlatSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",                       # -> TerrainImporter.import_ground_plane()
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        debug_vis=False,
    )
    # robot = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

The plane is spawned by

```python
# isaaclab/isaaclab/terrains/terrain_importer.py:209-224
color = (0.0, 0.0, 0.0)
if self.cfg.visual_material is not None:
    material = self.cfg.visual_material.to_dict()
    if "diffuse_color" in material:
        color = material["diffuse_color"]
    else:
        logger.warning("Visual material specified for ground plane but no diffuse color found. ...")
ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=self.cfg.physics_material, size=size, color=color)
ground_plane_cfg.func(prim_path, ground_plane_cfg)
```

**`physics_material` is used for `"plane"` and `"generator"`; NOT for `"usd"`.** — `terrain_importer_cfg.py:91-98`

Be aware: `visual_material` must expose a `diffuse_color` key or you get a warning and a black plane
(`terrain_importer.py:211-220`). `PreviewSurfaceCfg(diffuse_color=...)` (the default) does.

### 7.4 Switching an existing rough task to flat (real example)

```python
# isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py:13-28
def __post_init__(self):
    # post init of parent
    super().__post_init__()

    # override rewards
    self.rewards.flat_orientation_l2.weight = -2.5
    self.rewards.feet_air_time.weight = 0.25

    # change terrain to flat
    self.scene.terrain.terrain_type = "plane"
    self.scene.terrain.terrain_generator = None
    # no height scan
    self.scene.height_scanner = None
    self.observations.policy.height_scan = None
    # no terrain curriculum
    self.curriculum.terrain_levels = None
```

Note the ordering subtlety: `self.scene.height_scanner = None` is only legal *after* `super().__post_init__()`
and for an `InteractiveSceneCfg` subclass that declared `height_scanner` (setting a declared attribute to
`None` removes the entity). — `interactive_scene.py:686-690`

---

## 8. Reward manager: `weight` handling and `get_term_cfg` / `set_term_cfg`

### 8.1 How `weight` is applied

`isaaclab/isaaclab/managers/reward_manager.py:129-159`

```python
def compute(self, dt: float) -> torch.Tensor:
    # reset computation
    self._reward_buf[:] = 0.0
    # iterate over all the reward terms
    for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
        # skip if weight is zero (kind of a micro-optimization)
        if term_cfg.weight == 0.0:
            self._step_reward[:, term_idx] = 0.0
            continue
        # compute term's value
        value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
        # update total reward
        self._reward_buf += value
        # update episodic sum
        self._episode_sums[name] += value

        # Update current reward for this step.
        self._step_reward[:, term_idx] = value / dt

    return self._reward_buf
```

**Effective reward = `func(env, **params) * weight * dt`, accumulated per env step.** `weight == 0.0`
short-circuits the term entirely. Episode logging normalizes by the episode length in seconds:

```python
# isaaclab/isaaclab/managers/reward_manager.py:119-120
episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
extras["Episode_Reward/" + key] = episodic_sum_avg / self._env.max_episode_length_s
```

### 8.2 `RewardManager` accessors — exact signatures

```python
# isaaclab/isaaclab/managers/reward_manager.py:165
def set_term_cfg(self, term_name: str, cfg: RewardTermCfg):
    if term_name not in self._term_names:              # :175
        raise ValueError(f"Reward term '{term_name}' not found.")
    self._term_cfgs[self._term_names.index(term_name)] = cfg      # :178

# isaaclab/isaaclab/managers/reward_manager.py:180
def get_term_cfg(self, term_name: str) -> RewardTermCfg:
    if term_name not in self._term_names:              # :192
        raise ValueError(f"Reward term '{term_name}' not found.")
    return self._term_cfgs[self._term_names.index(term_name)]     # :195
```

Sibling managers have identical-shaped accessors:
`EventManager.set_term_cfg` (`managers/event_manager.py:282`), `get_term_cfg` (`:304`);
`TerminationManager.set_term_cfg` (`managers/termination_manager.py:216`), `get_term_cfg` (`:231`).

### 8.3 The real Isaac Lab call site (this is the canonical pattern)

**It is NOT in `isaaclab_tasks` — it is in `isaaclab.envs.mdp.curriculums`:**

```python
# isaaclab/isaaclab/envs/mdp/curriculums.py:24-47
class modify_reward_weight(ManagerTermBase):
    """Curriculum that modifies the reward weight based on a step-wise schedule."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # obtain term configuration
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        weight: float,
        num_steps: int,
    ) -> float:
        # update term settings
        if env.common_step_counter > num_steps:
            self._term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)

        return self._term_cfg.weight
```

Real **task-side** usage of that curriculum term (exhaustive grep over `isaaclab_tasks` for
`modify_reward_weight` found these; there are **none** in the locomotion velocity configs):

```python
# isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py:175-185
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )
```

Other task-side call sites: `isaaclab_tasks/.../manipulation/reach/reach_env_cfg.py:175,179`, and the
`openarm` configs under `manipulation/reach/config/openarm/...`.

**In-script equivalent** (this is the pattern you will actually use from a training/play script):

```python
env = gym.make(task, cfg=env_cfg)
env_unwrapped = env.unwrapped

# read
cfg = env_unwrapped.reward_manager.get_term_cfg("track_lin_vel_xy_exp")
print(cfg.weight)

# write
cfg.weight = 2.0
env_unwrapped.reward_manager.set_term_cfg("track_lin_vel_xy_exp", cfg)
```

**Alternatively, the more common runtime pattern in Isaac Lab scripts is to reach into the env cfg
directly BEFORE constructing the env, since `RewardTermCfg` objects are plain config instances:**

```python
env_cfg = parse_env_cfg(task)
env_cfg.rewards.track_lin_vel_xy_exp.weight = 2.0   # works because rewards is a @configclass object
```

---

## 9. `isaaclab_rl.rsl_rl`

> Installed `rsl-rl-lib` is **5.0.1** → the `actor`/`critic` model-config path is the live one and `policy`
> is deprecated (`isaaclab_rl/isaaclab_rl/rsl_rl/utils.py:17-18`, `:27-32`, `:69-75`).

### 9.1 `RslRlVecEnvWrapper` — exact signature

`isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py:26`

```python
class RslRlVecEnvWrapper(VecEnv):
    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
```

- Must be the **last** wrapper in the chain (does not follow the `gym.Wrapper` interface). — `:17-20`
- Raises `ValueError` unless `env.unwrapped` is `ManagerBasedRLEnv` or `DirectRLEnv`. — `:41-45`
- Sets `self.env`, `self.clip_actions`, `self.num_envs`, `self.device`, `self.max_episode_length`. — `:48-54`
- `num_actions` = `unwrapped.action_manager.total_action_dim` if it has an action manager, else
  `gym.spaces.flatdim(single_action_space)`. — `:57-60`
- **Calls `self.env.reset()` in the constructor** since the RSL-RL runner does not reset. — `:65-66`
- `cfg` property returns the env cfg; `unwrapped` returns `self.env.unwrapped`; `episode_length_buf`
  delegates to the env. — `:80-83`, `:105-111`, `:117-120`

Canonical use — `isaaclab_rl/isaaclab_rl/rsl_rl/__init__.py:10-14`:

```python
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

env = RslRlVecEnvWrapper(env)
```

### 9.2 `RslRlOnPolicyRunnerCfg` + base runner fields

`isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:229-347`

```python
@configclass
class RslRlBaseRunnerCfg:
    seed: int = 42                                   # :233
    device: str = "cuda:0"                           # :236
    num_steps_per_env: int = MISSING                 # :239
    max_iterations: int = MISSING                    # :242
    empirical_normalization: bool = MISSING          # :245  DEPRECATED
    obs_groups: dict[str, list[str]] = MISSING       # :252
    clip_actions: float | None = None                # :274
    check_for_nan: bool = True                       # :281
    save_interval: int = MISSING                     # :284
    experiment_name: str = MISSING                   # :287
    run_name: str = ""                               # :290
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"  # :298
    neptune_project: str = "isaaclab"                # :301
    wandb_project: str = "isaaclab"                  # :304
    resume: bool = False                             # :307
    load_run: str = ".*"                             # :313
    load_checkpoint: str = "model_.*.pt"             # :319


@configclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):   # :327
    class_name: str = "OnPolicyRunner"               # :330
    actor: RslRlMLPModelCfg = MISSING                # :333
    critic: RslRlMLPModelCfg = MISSING               # :336
    algorithm: RslRlPpoAlgorithmCfg = MISSING        # :339
    policy: RslRlPpoActorCriticCfg = MISSING         # :342  DEPRECATED for rsl-rl >= 4.0.0
```

### 9.3 `RslRlMLPModelCfg` — the live model config

`isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:21-93`

```python
@configclass
class RslRlMLPModelCfg:
    class_name: str = "MLPModel"                                   # :25
    hidden_dims: list[int] = MISSING                               # :28
    activation: str = MISSING                                      # :31
    obs_normalization: bool = False                                # :34
    distribution_cfg: DistributionCfg | None = None                # :37

    @configclass
    class DistributionCfg:                                         # :41
        class_name: str = MISSING

    @configclass
    class GaussianDistributionCfg(DistributionCfg):                # :48
        class_name: str = "GaussianDistribution"                   # :52
        init_std: float = MISSING                                  # :54
        std_type: Literal["scalar", "log"] = "scalar"              # :57

    @configclass
    class HeteroscedasticGaussianDistributionCfg(GaussianDistributionCfg):  # :61
        class_name: str = "HeteroscedasticGaussianDistribution"    # :65

    stochastic: bool = MISSING                 # :67  DEPRECATED (rsl-rl >= 5.0.0)
    init_noise_std: float = MISSING            # :74  DEPRECATED (rsl-rl >= 5.0.0)
    noise_std_type: Literal["scalar","log"] = "scalar"  # :81  DEPRECATED (rsl-rl >= 5.0.0)
    state_dependent_std: bool = False          # :88  DEPRECATED (rsl-rl >= 5.0.0)
```

Also: `RslRlRNNModelCfg` (`:97`, adds `rnn_type`, `rnn_hidden_dim`, `rnn_num_layers`),
`RslRlCNNModelCfg` (`:114`, adds `cnn_cfg`).

### 9.4 `RslRlPpoAlgorithmCfg` — full field list

`isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:161-221`

```python
@configclass
class RslRlPpoAlgorithmCfg:
    class_name: str = "PPO"                                          # :165
    num_learning_epochs: int = MISSING                               # :168
    num_mini_batches: int = MISSING                                  # :171
    learning_rate: float = MISSING                                   # :174
    schedule: str = MISSING                                          # :177
    gamma: float = MISSING                                           # :180
    lam: float = MISSING                                             # :183
    entropy_coef: float = MISSING                                    # :186
    desired_kl: float = MISSING                                      # :189
    max_grad_norm: float = MISSING                                   # :192
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"   # :195
    value_loss_coef: float = MISSING                                 # :198
    use_clipped_value_loss: bool = MISSING                           # :201
    clip_param: float = MISSING                                      # :204
    normalize_advantage_per_mini_batch: bool = False                 # :207
    share_cnn_encoders: bool = False                                 # :214
    rnd_cfg: RslRlRndCfg | None = None                               # :217
    symmetry_cfg: RslRlSymmetryCfg | None = None                     # :220
```

**`optimizer` is new and only supported for rsl-rl >= 4.0.0** — `isaaclab_rl/isaaclab_rl/rsl_rl/utils.py:52-61`.

### 9.5 `RslRlPpoActorCriticCfg` — the DEPRECATED path (still present)

`isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:355-407`

```python
@configclass
class RslRlPpoActorCriticCfg:
    """For rsl-rl >= 4.0.0, this configuration is deprecated. Please use `RslRlMLPModelCfg` instead."""  # :359

    class_name: str = "ActorCritic"                  # :362
    init_noise_std: float = MISSING                  # :365
    noise_std_type: Literal["scalar", "log"] = "scalar"   # :368
    state_dependent_std: bool = False                # :371
    actor_obs_normalization: bool = MISSING          # :374
    critic_obs_normalization: bool = MISSING         # :377
    actor_hidden_dims: list[int] = MISSING           # :380
    critic_hidden_dims: list[int] = MISSING          # :383
    activation: str = MISSING                        # :386


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):   # :391
    class_name: str = "ActorCriticRecurrent"         # :397
    rnn_type: str = MISSING                          # :400
    rnn_hidden_dim: int = MISSING                    # :403
    rnn_num_layers: int = MISSING                    # :406
```

**On this install (`rsl-rl-lib 5.0.1`) `policy` is cleared and its values are used only to infer missing
`actor`/`critic` configs** — `isaaclab_rl/isaaclab_rl/rsl_rl/utils.py:69-80`. **Write `actor` + `critic`.**

### 9.6 Real `runner_cfg` in the locomotion velocity tasks (2.3.2 style)

`isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py:8-50`

```python
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
    RslRlSymmetryCfg,
)

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symmetry import anymal


@configclass
class AnymalDRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "anymal_d_rough"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AnymalDFlatPPORunnerCfg(AnymalDRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "anymal_d_flat"
        self.actor.hidden_dims = [128, 128, 128]
        self.critic.hidden_dims = [128, 128, 128]
```

Recurrent variant (`:65-82`) swaps `actor`/`critic` for `RslRlRNNModelCfg(..., rnn_type="lstm",
rnn_hidden_dim=256, rnn_num_layers=1)`. Symmetry variant (`:86-105`) adds
`symmetry_cfg=RslRlSymmetryCfg(use_data_augmentation=True, data_augmentation_func=...)` to `algorithm`.

### 9.7 Reading the runner cfg in a train script

```python
import os
import gymnasium as gym
import torch
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

task = "Isaac-Velocity-Flat-Unitree-Go2-v0"
env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=4096)
# the rsl_rl agent cfg is loaded by your launcher from the gym kwargs key
# "rsl_rl_cfg_entry_point", e.g. via isaaclab_tasks.utils.parse_cfg / load_cfg_from_registry
```

The gym registration provides the fully-qualified agent-cfg string under the kwargs key
`"rsl_rl_cfg_entry_point"` — see section 10.2.

---

## 10. A complete, real, working floating-base manager-based task config

The canonical floating-base task in 2.3.2 is the loco-velocity task. `isaaclab_tasks/.../locomotion/velocity/`
contains `a1, anymal_b, anymal_c, anymal_d, cassie, digit, g1, go1, go2, h1, spot` — all floating-base
quadrupeds/humanoids.

### 10.1 The shared floating-base env cfg (`LocomotionVelocityRoughEnvCfg`)

`isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:289-329` — quoted in full:

```python
##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
```

**Per-robot `__post_init__` (floating-base robot, flat terrain) in full:**
`isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py:16-85`

```python
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
```

The **FLAT** variant in full — `isaaclab_tasks/.../config/go2/flat_env_cfg.py:11-43`:

```python
@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
```

Note the idiom `self.rewards.undesired_contacts = None` / `self.events.push_robot = None` /
`self.curriculum.terrain_levels = None`: **assigning `None` to a declared `@configclass` attribute removes
that term** (the manager iterates the config's `__dict__` and skips `None`).

### 10.2 A real `gym.register` block

`isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py:1-56` — complete file:

```python
import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
```

**Confirmed patterns (all four kwarg keys are `str` entry-point specs, not objects):**
`"env_cfg_entry_point"` → `"{module}:{ClassName}"`; `"rsl_rl_cfg_entry_point"` →
`"{module}:{ClassName}"`; `"skrl_cfg_entry_point"` → `"{module}:{file.yaml}"` (note the **colon** for a
YAML file, not a dot).

> **Correction to the task brief:** there is **no** `play_env_cfg_entry_point` key at 2.3.2 — the play
> variant is a *separate gym id* (suffix `-Play-v0`) that overrides `env_cfg_entry_point` with a
> `*_PLAY` cfg class. Exhaustive grep over `isaaclab_tasks` finds no `play_env_cfg_entry_point`.

---

## 11. `JointPositionAction` / `JointVelocityAction`: scale, offset, `use_default_offset`, `preserve_order`, `joint_names`

### 11.1 `JointActionCfg` — the shared affine-transform base

`isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py:26-40`

```python
@configclass
class JointActionCfg(ActionTermCfg):
    """Configuration for the base joint action term."""

    joint_names: list[str] = MISSING                                # :33
    scale: float | dict[str, float] = 1.0                           # :35
    offset: float | dict[str, float] = 0.0                          # :37
    preserve_order: bool = False                                    # :39
```

Inherited from `ActionTermCfg` (`managers/manager_term_cfg.py:77-98`): `class_type: type[ActionTerm] = MISSING`,
`asset_name: str = MISSING`, `debug_vis: bool = False`, `clip: dict[str, tuple] | None = None`.

### 11.2 Subclasses — `JointPositionActionCfg` EXISTS as an alias

`isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py:43-101`

```python
@configclass
class JointPositionActionCfg(JointActionCfg):                        # :44
    """Configuration for the joint position action term."""
    class_type: type[ActionTerm] = joint_actions.JointPositionAction  # :50
    use_default_offset: bool = True                                   # :52


@configclass
class RelativeJointPositionActionCfg(JointActionCfg):                 # :62
    class_type: type[ActionTerm] = joint_actions.RelativeJointPositionAction  # :68
    use_zero_offset: bool = True                                      # :70


@configclass
class JointVelocityActionCfg(JointActionCfg):                         # :78
    class_type: type[ActionTerm] = joint_actions.JointVelocityAction   # :84
    use_default_offset: bool = True                                   # :86


@configclass
class JointEffortActionCfg(JointActionCfg):                           # :95
    class_type: type[ActionTerm] = joint_actions.JointEffortAction     # :101
```

**Yes — `JointPositionActionCfg` is the real class name** (it is not an alias for something else); and
because `isaaclab/envs/mdp/actions/__init__.py:8` star-imports `actions_cfg`, it is reachable as
`from isaaclab.envs.mdp import JointPositionActionCfg` and as `mdp.JointPositionActionCfg`.
There is **no** `JointPositionAction` *cfg* alias without the `Cfg` suffix on the cfg side.

### 11.3 Where scale/offset are actually applied

The affine law is `action = offset + scaling * input_action`. — `envs/mdp/actions/joint_actions.py:34-47`

```python
# isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:60-112  (JointAction.__init__)
def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
    super().__init__(cfg, env)
    # resolve the joints over which the action term is applied
    self._joint_ids, self._joint_names = self._asset.find_joints(
        self.cfg.joint_names, preserve_order=self.cfg.preserve_order
    )
    self._num_joints = len(self._joint_ids)
    ...
    # Avoid indexing across all joints for efficiency
    if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
        self._joint_ids = slice(None)

    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self.raw_actions)

    # parse scale
    if isinstance(cfg.scale, (float, int)):
        self._scale = float(cfg.scale)
    elif isinstance(cfg.scale, dict):
        self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
        self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
        raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
    # parse offset
    if isinstance(cfg.offset, (float, int)):
        self._offset = float(cfg.offset)
    elif isinstance(cfg.offset, dict):
        self._offset = torch.zeros_like(self._raw_actions)
        index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
        self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
        raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
    # parse clip
    if self.cfg.clip is not None:
        if isinstance(cfg.clip, dict):
            self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                self.num_envs, self.action_dim, 1
            )
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
            self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")
```

```python
# isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:169-181
def process_actions(self, actions: torch.Tensor):
    # store the raw actions
    self._raw_actions[:] = actions
    # apply the affine transformations
    self._processed_actions = self._raw_actions * self._scale + self._offset
    # clip actions
    if self.cfg.clip is not None:
        self._processed_actions = torch.clamp(
            self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
        )

def reset(self, env_ids: Sequence[int] | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
```

`use_default_offset` **overwrites `offset` wholesale** with the articulation's default joint state:

```python
# isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:190-199
def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg, env)
    # use default joint positions as offset
    if cfg.use_default_offset:
        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

def apply_actions(self):
    # set position targets
    self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
```

```python
# isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:241-250
def __init__(self, cfg: actions_cfg.JointVelocityActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg, env)
    # use default joint velocity as offset
    if cfg.use_default_offset:
        self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()

def apply_actions(self):
    self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)
```

```python
# isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:259-264
def __init__(self, cfg: actions_cfg.JointEffortActionCfg, env: ManagerBasedEnv):
    super().__init__(cfg, env)

def apply_actions(self):
    self._asset.set_joint_effort_target(self.processed_actions, joint_ids=self._joint_ids)
```

**Rules confirmed:**
1. `use_default_offset=True` (default for both position and velocity) makes `offset` irrelevant — do not set both.
2. `preserve_order=False` (default) sorts `joint_ids` ascending in the asset's joint order; with
   `preserve_order=True` the action-vector layout follows your `joint_names` order. When all joints are
   selected and `preserve_order=False`, `_joint_ids` is collapsed to `slice(None)`. — `:65-67`, `:76-77`
3. `scale` / `offset` accept **either a scalar or a regex-keyed dict** (e.g. `{".*_wheel": 5.0, ".*_calf": 0.25}`).
4. `clip` must be a **dict** `{regex: (lo, hi)}`; a tuple raises `ValueError("Unsupported clip type")`.
   — `:104-112`
5. `raw_actions` reset to `0.0` on reset, so `offset` fully determines the post-reset target. — `:180-181`

Real 2.3.2 usage:

```python
# isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:108-112
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
```

```python
# isaaclab_tasks/.../config/go2/rough_env_cfg.py:29-30
# reduce action scale
self.actions.joint_pos.scale = 0.25
```

---

## 12. Closed-loop / loop joints / constrained articulations — exhaustive grep result

**Exhaustive search performed** over every file in `D:\IsaacLab\source` (**1648 files total**: 971 `.py`,
475 `.pyc`, 118 `.yaml`, 22 `.txt`, 12 `.toml`, 11 `.json`, 7 `.md`, 6 `.rst`, 6 `.yml`, 4 `.urdf`,
3 `.npz`, 12 extension-less, 1 `.gitkeep`) using `Select-String -SimpleMatch` across the whole tree
(not just `*.py`):

| Pattern searched | Matches found |
|---|---|
| `loop_joint` | **0** |
| `loopJoint` | **0** |
| `ArticulationConstraint` | **0** |
| `excludeFromArticulation` | **0** |
| `exclude_from_articulation` | **0** |
| `JointConstraint` | **0** |
| `PhysxLimitAPI` | **0** |
| `mimic` | 30 matches (see below) |
| `closed` / `closed-loop` / `closed_loop` / `closed loop` | (see below) |
| `PhysxArticulation` | 14 matches (see below) |

### 12.1 Conclusion

**Isaac Lab 2.3.2 contains NO first-class API for closed-loop / loop-closure joints, articulation
constraints, or `excludeFromArticulation`. There is no `loop_joint`, no `ArticulationConstraint`,
no `JointConstraint`, and no `excludeFromArticulation` anywhere in the tree.** There is no
`schemas.*` config class for them either: `isaaclab/isaaclab/sim/schemas/schemas_cfg.py` (679 lines) exposes
only `ArticulationRootPropertiesCfg`, `RigidBodyPropertiesCfg`, `CollisionPropertiesCfg`,
`MassPropertiesCfg`, `JointDrivePropertiesCfg`, `FixedTendonPropertiesCfg`, `SpatialTendonPropertiesCfg`,
`DeformableBodyPropertiesCfg`, and the mesh-collision family — none of which mention loop joints or
`excludeFromArticulation` (verified `:14-679`). The four URDF files in the tree are asset fixtures, not
closed-loop examples.

### 12.2 Everything that *does* exist, verbatim

**Mimic joints (only URDF-converter level):**

```
isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py:107
    convert_mimic_joints_to_normal_joints: bool = False
isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py:108
    """Convert mimic joints to normal joints. Defaults to False."""
isaaclab/isaaclab/sim/converters/urdf_converter.py:149
    # convert mimic joints to normal joints
isaaclab/isaaclab/sim/converters/urdf_converter.py:150
    import_config.set_parse_mimic(self.cfg.convert_mimic_joints_to_normal_joints)
```

This is the **only** mechanism: during URDF→USD conversion, mimic joints can be *flattened into normal
joints* (i.e. the mimic constraint is dropped, not modelled). This is exactly what you need for a
mechanically-parallel leg linkage: set `convert_mimic_joints_to_normal_joints=True` and drive both joints
from the action term.

`mimic` also appears in the **unrelated** Isaac Lab Mimic imitation-learning subsystem
(`isaaclab/isaaclab/envs/manager_based_rl_mimic_env.py`, `mimic_env_cfg.py`, `isaaclab_mimic/`,
`devices/openxr/retargeters/.../*.py`, and `isaaclab/docs/CHANGELOG.rst`). Those are **not** joint
constraints and must not be confused with them.

**Closed-loop mentions (none are joint constraints):**

```
isaaclab/isaaclab/controllers/__init__.py:8
    Controllers or motion generators are responsible for closed-loop tracking of a given command.
isaaclab/isaaclab/controllers/operational_space_cfg.py:71
    """The proportional gain for determining operational space command forces for closed-loop contact force control.
isaaclab/isaaclab/controllers/operational_space.py:384,461,466
    closed-loop force control error paths / feedforward term
isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/config/franka/osc_env_cfg.py:33
    # If closed-loop contact force control is desired, contact sensors should be enabled for the robot
isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/digit/rough_env_cfg.py:233
    # Don't randomize the initial joint positions because we have closed loops.
```

The **Digit** config is the only place in the whole tree that acknowledges a robot with kinematic closed
loops, and its only accommodation is to *not* randomize initial joint positions. Its `__post_init__` region
around line 233 is the closest thing in Isaac Lab to closed-loop guidance:

```python
# isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/digit/rough_env_cfg.py:233
# Don't randomize the initial joint positions because we have closed loops.
```

(Read the surrounding block in that file if you intend to model a closed-loop leg; the takeaway is that
Isaac Lab does not simulate the loop — the USD/URDF must already represent it as normal joints.)

**`PhysxArticulation` occurrences (all are articulation-root handling, none are constraints):**

```
isaaclab/isaaclab/assets/rigid_object/rigid_object.py:507
    if articulation_prims[0].GetAttribute("physxArticulation:articulationEnabled").Get():
isaaclab/isaaclab/assets/rigid_object_collection/rigid_object_collection.py:658
    if articulation_prims[0].GetAttribute("physxArticulation:articulationEnabled").Get():
isaaclab/isaaclab/envs/ui/base_env_window.py:332
    prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
isaaclab/isaaclab/markers/visualization_markers.py:388
    child_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
isaaclab/isaaclab/sim/schemas/schemas.py:116,135,159,161,208,217,224
isaaclab/test/deps/isaacsim/check_floating_base_made_fixed.py:131,140,147
isaaclab/test/sim/test_schemas.py:259
```

### 12.3 What to do for a closed-loop / parallel-linkage robot (e.g. a 5-bar or wheeled-leg infantry)

1. **Flatten the loop at conversion time.** In your URDF, mark the passive link's joint as a `mimic` joint
   and convert with `UrdfFileCfg(convert_mimic_joints_to_normal_joints=True, ...)`
   (`urdf_converter_cfg.py:107`, `urdf_converter.py:149-150`). The resulting USD has no loop closure and
   PhysX treats it as a tree — this is what the bundled assets do.
2. **Convert to USD once, cache it, then use `UsdFileCfg(usd_path=...)`** at runtime. Isaac Lab will reuse
   that USD directly (`from_files_cfg.py:99-102`).
3. **Do not randomize initial joint positions** on a looped mechanism — see the Digit precedent
   (`digit/rough_env_cfg.py:233`).
4. If you truly need runtime loop closure, it is **not available in this Isaac Lab version** and you must go
   directly to USD/PhysX (`UsdPhysics.Joint` with `excludeFromArticulation`, or
   `PhysxSchema.PhysxArticulationConstraintAPI`) *outside* Isaac Lab — nothing in `D:\IsaacLab\source`
   provides or documents it.

---

## GOTCHAS FOR 2.3.2

Differences from widely-known Isaac Lab 1.x / 2.0 conventions:

1. **`viewer` is a nested `@configclass`, not flat fields.** There is no `cfg.viewer_eye`,
   `cfg.viewer_lookat`, `cfg.viewer_origin_type`. Use `cfg.viewer.eye`, `cfg.viewer.lookat`,
   `cfg.viewer.origin_type`, `cfg.viewer.resolution`, `cfg.viewer.cam_prim_path`, `cfg.viewer.env_index`,
   `cfg.viewer.asset_name`, `cfg.viewer.body_name`. — `envs/common.py:20-68`

2. **THERE IS NO `randomization` FIELD** in `ManagerBasedRLEnvCfg` or `ManagerBasedEnvCfg`.
   Randomization is expressed entirely through the `events` config object (`EventTerm` with
   `mode="startup" | "reset" | "interval" | "custom" | "prestartup"`). — `envs/manager_based_rl_env_cfg.py:14-80`,
   `envs/manager_based_env_cfg.py:38-150`, `managers/manager_term_cfg.py:250-302`

3. **`observations` / `actions` / `rewards` / `terminations` / `commands` / `curriculum` / `events` are all
   typed `object`, not their manager cfg types.** You *cannot* rely on the base class to type-check them;
   they are just containers whose `__dict__` holds the terms. — `envs/manager_based_env_cfg.py:90-106`,
   `envs/manager_based_rl_env_cfg.py:58-80`

4. **`default_name` DOES NOT EXIST.** `ActionTermCfg` uses `asset_name: str = MISSING`.
   Exhaustive grep for `default_name` across `isaaclab/isaaclab/**/*.py` returns **zero** matches.
   `ActionTerm.__init__` does `self._asset = self._env.scene[self.cfg.asset_name]`.
   — `managers/manager_term_cfg.py:87`, `managers/action_manager.py:55`

5. **There is NO `ActionGroupCfg`.** Actions are a flat `dict[str, ActionTermCfg]` fed to `ActionManager`.
   Only observations have a group concept (`ObservationGroupCfg`). — `managers/manager_term_cfg.py:77-98`,
   `:198-242`; `managers/action_manager.py:198-202`

6. **`ObsTerm` / `RewTerm` / `DoneTerm` / `EventTerm` / `CurrTerm` / `ObsGroup` are NOT library exports.**
   They are local `import ... as ...` aliases defined in each task file, e.g.
   `from isaaclab.managers import ObservationTermCfg as ObsTerm`. There is no `TermCfg` name at all.
   — `isaaclab_tasks/.../velocity/velocity_env_cfg.py:12-18`;
   `managers/__init__.py:13-34` (full export list)

7. **Actuator limits were renamed and the old names are actively neutered.**
   Use `effort_limit_sim` (not `effort_limit`) and `velocity_limit_sim` (not `velocity_limit`).
   On `ImplicitActuatorCfg`: setting `velocity_limit` logs a warning and then **sets it to `None`**
   (silently no effect); setting `effort_limit` logs a deprecation warning and copies to `effort_limit_sim`;
   setting **both** with different values **raises `ValueError`**.
   — `actuators/actuator_pd.py:60-78`, `:81-102`

8. **`isaaclab.actuators.actuator_cfg` is a deprecated shim module.** `from isaaclab.actuators import DCMotorCfg`
   is correct; `from isaaclab.actuators.actuator_cfg import DCMotorCfg` emits `DeprecationWarning`.
   — `actuators/actuator_cfg.py:12-36`

9. **`UsdFileCfg.usd_path` vs `UrdfFileCfg.asset_path` — two different names on two different bases.**
   `UsdFileCfg` has `usd_path` and **no** `asset_path`; `UrdfFileCfg`/`MjcfFileCfg` have `asset_path`
   (from `AssetConverterBaseCfg`) and **no** `usd_path`. Passing the wrong one is a `TypeError`.
   — `sim/spawners/from_files/from_files_cfg.py:101`, `:114`, `:136`;
   `sim/converters/asset_converter_base_cfg.py:15`

10. **`UrdfFileCfg` EXISTS** (`sim/spawners/from_files/from_files_cfg.py:114`) and is exported as
    `isaaclab.sim.UrdfFileCfg`. URDF conversion has `convert_mimic_joints_to_normal_joints: bool = False`
    (`sim/converters/urdf_converter_cfg.py:107`) — this is the **only** closed-loop mitigation in the tree,
    see section 12.

11. **rsl-rl runner cfg: use `actor` + `critic` (`RslRlMLPModelCfg`), NOT `policy`
    (`RslRlPpoActorCriticCfg`).** `policy` is present but explicitly marked deprecated for rsl-rl >= 4.0.0
    and is *cleared* at load time. Stochasticity moved from `init_noise_std` to
    `distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=...)`. `RslRlPpoAlgorithmCfg` gained
    `optimizer` (default `"adam"`). Your installed `rsl-rl-lib` is **5.0.1**, so the new path is mandatory.
    — `isaaclab_rl/rsl_rl/rl_cfg.py:21-93`, `:161-221`, `:327-347`, `:355-359`;
    `isaaclab_rl/rsl_rl/utils.py:17-32`, `:69-75`

12. **No `play_env_cfg_entry_point` kwarg.** Play variants are separate gym ids with a `-Play-v0` suffix and a
    separate `*_PLAY` cfg class. Kwarg keys are `env_cfg_entry_point`, `rsl_rl_cfg_entry_point`,
    `skrl_cfg_entry_point` (YAML uses `module:file.yaml`). — `.../config/go2/__init__.py:14-56`

13. **`base_height` and `upright` rewards do not exist.** They are `base_height_l2(target_height: float,
    ...)` and `flat_orientation_l2(...)`. — `envs/mdp/rewards.py:101`, `:91`

14. **`base_contact` is a user-chosen term *name*, not a library term.** The 2.3.2 idiom is
    `base_contact = DoneTerm(func=mdp.illegal_contact, params={...})`.
    — `.../velocity/velocity_env_cfg.py:271-274`

15. **`rerender_on_reset` is deprecated (since 2.3.1) in favour of `num_rerenders_on_reset: int = 0`.**
    `rerender_on_reset=True` ≙ `num_rerenders_on_reset=1`.
    — `envs/manager_based_env_cfg.py:108-135`; `envs/manager_based_env.py:203`;
    `envs/direct_rl_env.py:230`

16. **`AppLauncher` is backed by `isaacsim.SimulationApp`, not `omni.isaac.kit`.** `--cpu` raises; use
    `--device cpu`. New-ish flags: `--rendering_mode {performance,balanced,quality}` and the
    `--anim_recording_*` trio. `create_new_stage` is forced to `False`.
    — `app/app_launcher.py:27`, `:677-679`, `:329-339`, `:349-371`, `:653`

17. **`PhysxCfg.solve_articulation_contact_last` is NEW in 2.3** and marks itself "only available with
    Isaac Sim 5.1". Relevant if you tune a stiff gripper/contact constraint system.
    — `sim/simulation_cfg.py:46-64`

18. **`SimulationCfg` gained `create_stage_in_memory`, `logging_level`, `save_logs_to_file`, `log_dir`.**
    `physics_material` and `physx` are nested `@configclass` objects. — `sim/simulation_cfg.py:412-444`

19. **`TerrainImporterCfg` uses `terrain_type: Literal["generator","plane","usd"]`, and
    `env_spacing` is auto-injected** from `InteractiveSceneCfg.env_spacing` — so
    `terrain_type="plane"` needs *only* `prim_path`, and never set `TerrainImporterCfg.env_spacing` yourself
    (it will be overwritten). `physics_material` applies to `"plane"`/`"generator"` only; a
    `visual_material` without a `diffuse_color` key produces a warning + black plane.
    — `terrains/terrain_importer_cfg.py:46`, `:80-98`; `terrains/terrain_importer.py:108-114`, `:209-224`,
    `:306-312`; `scene/interactive_scene.py:740-742`

20. **Assigning `None` to a declared config attribute removes that term** — this is the supported way to
    disable a reward/event/observation/curriculum term in `__post_init__`
    (e.g. `self.rewards.undesired_contacts = None`). — `.../config/go2/rough_env_cfg.py:49,54,84,85`

21. **`RewardTermCfg` is implemented as an ordinary `@configclass`**, so the most common runtime weight change
    is a direct attribute write on the cfg (`env_cfg.rewards.<name>.weight = w`) *before* env construction.
    After construction, go through `env.reward_manager.get_term_cfg(name)` → mutate `.weight` →
    `env.reward_manager.set_term_cfg(name, cfg)`.
    — `managers/reward_manager.py:165-195`; `envs/mdp/curriculums.py:24-47`

22. **No closed-loop / loop-joint / articulation-constraint API exists** (see section 12). Flatten mimic
    joints at URDF conversion (`convert_mimic_joints_to_normal_joints=True`) and do not randomize initial
    joint positions on looped mechanisms (Digit precedent).

23. **`SceneEntityCfg.preserve_order` (default `False`) is essential for wheeled/parallel mechanisms** where
    the action-vector layout must match a hand-written joint order: it flows into both
    `entity.find_joints(..., preserve_order=...)` and `Articulation.find_bodies/find_fixed_tendons/...`.
    — `managers/scene_entity_cfg.py:102-116`, `:162`, `:175`, `:197`, `:211`, `:237`, `:250`, `:271`, `:285`

24. **`isaaclab/isaaclab/actuators/actuator_cfg.py` still exists, so old imports "work" but warn.** Same
    pattern exists for `isaaclab/sim/utils/legacy` (`isaaclab/docs/CHANGELOG.rst:216`). Grep your imports
    before trusting them.

---

## Files read for this document

- `D:\IsaacLab\VERSION`
- `D:\IsaacLab\source\isaaclab\pyproject.toml`
- `D:\IsaacLab\source\isaaclab\isaaclab\app\app_launcher.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\manager_based_env_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\manager_based_rl_env_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\manager_based_env.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\manager_based_rl_env.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\common.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\mdp\__init__.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\mdp\observations.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\mdp\rewards.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\mdp\terminations.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\mdp\events.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\mdp\curriculums.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\mdp\actions\__init__.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\mdp\actions\actions_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\mdp\actions\joint_actions.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\envs\mdp\commands\commands_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\scene\interactive_scene_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\scene\interactive_scene.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\assets\asset_base_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\assets\articulation\articulation_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\actuators\__init__.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\actuators\actuator_base_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\actuators\actuator_pd_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\actuators\actuator_pd.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\actuators\actuator_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\managers\__init__.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\managers\manager_term_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\managers\scene_entity_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\managers\reward_manager.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\managers\action_manager.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\managers\manager_base.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\sim\simulation_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\sim\spawners\from_files\__init__.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\sim\spawners\from_files\from_files_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\sim\converters\asset_converter_base_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\sim\converters\urdf_converter_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\sim\schemas\schemas_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\terrains\__init__.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\terrains\terrain_importer_cfg.py`
- `D:\IsaacLab\source\isaaclab\isaaclab\terrains\terrain_importer.py`
- `D:\IsaacLab\source\isaaclab\docs\CHANGELOG.rst`
- `D:\IsaacLab\source\isaaclab_rl\isaaclab_rl\rsl_rl\__init__.py`
- `D:\IsaacLab\source\isaaclab_rl\isaaclab_rl\rsl_rl\rl_cfg.py`
- `D:\IsaacLab\source\isaaclab_rl\isaaclab_rl\rsl_rl\utils.py`
- `D:\IsaacLab\source\isaaclab_rl\isaaclab_rl\rsl_rl\vecenv_wrapper.py`
- `D:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\velocity_env_cfg.py`
- `D:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2\__init__.py`
- `D:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2\rough_env_cfg.py`
- `D:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\go2\flat_env_cfg.py`
- `D:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomotion\velocity\config\anymal_d\agents\rsl_rl_ppo_cfg.py`
- `D:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\lift\lift_env_cfg.py`
