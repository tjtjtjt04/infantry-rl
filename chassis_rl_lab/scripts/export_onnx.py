"""把训练好的蹲姿平衡策略导出为 ONNX + 元数据 JSON（chassis_rl_lab 规格 §7）。

用法（Windows，`D:\\miniconda\\envs\\RM\\python.exe`）::

    # 默认导出最新 run 的最新 checkpoint -> exports/chassis_standup_policy.onnx(+.json)
    python chassis_rl_lab\\scripts\\export_onnx.py

    # 指定 checkpoint / 输出路径
    python chassis_rl_lab\\scripts\\export_onnx.py ^
        --checkpoint logs\\rsl_rl\\chassis_rl_stand_up\\<run>\\model_500.pt ^
        --output exports\\chassis_standup_policy.onnx

关键事实（详见 `_design/01_reference_notes.md` §2.15）：
    * rsl_rl 5.0.1 的 actor 是 ``MLPModel``（不是旧版 ``ActorCritic``/``Sequential``），
      **必须**用 ``OnPolicyRunner.export_policy_to_onnx()``；isaaclab_rl 0.5.2 的
      ``export_policy_as_onnx`` 是 rsl_rl < 4.0 的旧接口，会在本机报错。
    * 导出图的输入名 ``obs``、输出名 ``actions``、batch 固定为 1、opset 18。
    * actor 配置 ``obs_normalization=True`` 时，**obs 归一化（EmpiricalNormalization）在 ONNX 图内部**，
      部署侧喂"按 obs term scale 缩放后的原始观测"，不要再做一次归一化。
    * 输出是动作分布的**均值**（确定性输出）。

元数据 JSON（与 onnx 同名）包含：
    obs 各项名称/维度/scale/顺序、动作维度/scale/映射、关节顺序（HIP/KNEE/WHEEL）、
    squat 姿态、关节符号约定（**统一世界 +y 右手定则正转、左右同号**、轮子正转=车体前进）、
    sim.dt / decimation / 控制频率、部署注意事项。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1) 项目根加入 sys.path
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# 2) argparse 阶段：只能 import AppLauncher
# --------------------------------------------------------------------------- #
from isaaclab.app import AppLauncher  # noqa: E402

DEFAULT_TASK = "Chassis-Stand-Up-v0"
DEFAULT_OUTPUT = "exports/chassis_standup_policy.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained chassis-rl stand-up policy to ONNX with metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="gym 注册的任务 id")
    parser.add_argument("--checkpoint", type=str, default="-1", help="checkpoint 路径/正则；'-1' = 最新")
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="ONNX 输出路径（相对路径按项目根解析；同名 .json 写元数据）",
    )
    parser.add_argument("--experiment_name", type=str, default=None, help="覆盖 agent cfg 的 experiment_name")
    parser.add_argument("--load_run", type=str, default=None, help="run 目录（正则/目录名）；配合 --checkpoint -1")
    parser.add_argument("--log_root", type=str, default=None, help="日志根目录（默认相对当前工作目录 logs/rsl_rl）")
    # 注意：不要自己加 --verbose——AppLauncher.add_app_launcher_args() 已定义同名参数，
    # 重复定义会抛 argparse.ArgumentError 导致脚本启动即崩；torch.onnx.export 直接复用 args_cli.verbose。
    parser.add_argument("--gui", action="store_true", default=False, help="用 GUI 启动（默认 headless，导出不需要窗口）")
    # AppLauncher 提供 --headless / --device 等
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
# 导出只需要构建一个 env 拿配置信息，默认 headless，避免弹窗口
if not args_cli.gui:
    args_cli.headless = True

# --------------------------------------------------------------------------- #
# 3) 启动 Isaac Sim
# --------------------------------------------------------------------------- #
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
# 4) Kit 启动后的 import
# --------------------------------------------------------------------------- #
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

try:
    import chassis_rl_lab  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "无法 import `chassis_rl_lab`：请确认 chassis_rl_lab/chassis_rl_lab/__init__.py 已实现 gym.register。"
    ) from exc


# --------------------------------------------------------------------------- #
# 5) 工具函数
# --------------------------------------------------------------------------- #
def load_cfg_from_registry(task: str, entry_point_key: str):
    import importlib

    try:
        spec = gym.spec(task.split(":")[-1])
    except gym.error.Error as exc:
        raise RuntimeError(f"任务 {task!r} 尚未在 gym 注册。") from exc
    entry_point = spec.kwargs.get(entry_point_key)
    if entry_point is None:
        raise ValueError(f"任务 {task!r} 缺少 registry kwarg {entry_point_key!r}")
    if callable(entry_point):
        return entry_point()
    module_name, class_name = entry_point.split(":")
    return getattr(importlib.import_module(module_name), class_name)()


def _checkpoint_sort_key(filename: str) -> tuple[int, str]:
    import re

    match = re.search(r"(\d+)", filename)
    return (int(match.group(1)) if match else -1, filename)


def resolve_checkpoint(log_root: str, load_run: str = ".*", load_checkpoint: str = "model_.*.pt") -> str:
    import re

    if load_checkpoint and os.path.isfile(load_checkpoint):
        return os.path.abspath(load_checkpoint)
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"日志根目录不存在：{log_root}")
    run_pattern = ".*" if load_run in (None, "", "-1") else load_run
    runs = [
        os.path.join(log_root, name)
        for name in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, name)) and re.match(run_pattern, name)
    ]
    if not runs:
        raise FileNotFoundError(f"{log_root} 下没有匹配 {run_pattern!r} 的 run 目录")
    run_dir = sorted(runs)[-1]
    checkpoint_pattern = "model_.*.pt" if load_checkpoint in (None, "", "-1") else load_checkpoint
    files = [name for name in os.listdir(run_dir) if re.match(checkpoint_pattern, name)]
    if not files:
        raise FileNotFoundError(f"{run_dir} 下没有匹配 {checkpoint_pattern!r} 的 checkpoint")
    files.sort(key=_checkpoint_sort_key)
    return os.path.join(run_dir, files[-1])


def _callable_name(func: object) -> str:
    """把 obs term 的 func 变成可读字符串（函数/可调用类/partial 都兼容）。"""
    module = getattr(func, "__module__", None)
    name = getattr(func, "__name__", None)
    if module and name:
        return f"{module}.{name}"
    return repr(func)


def _obs_term_cfgs(env_cfg: object, group: str) -> dict[str, object]:
    """从 env cfg 的 observation group 里取 ``term_name -> ObservationTermCfg`` 映射。"""
    group_cfg = getattr(env_cfg.observations, group, None)
    term_cfgs: dict[str, object] = {}
    if group_cfg is None:
        return term_cfgs
    for field in dataclasses.fields(group_cfg):
        value = getattr(group_cfg, field.name, None)
        # ObservationTermCfg 有 func/params/scale 等属性；group 级配置没有 func
        if value is not None and hasattr(value, "func") and hasattr(value, "scale"):
            term_cfgs[field.name] = value
    return term_cfgs


def _action_term_cfg(env_cfg: object) -> tuple[str, object | None]:
    """从 env cfg 的 actions 里取第一个动作项 (name, cfg)。"""
    for field in dataclasses.fields(env_cfg.actions):
        value = getattr(env_cfg.actions, field.name, None)
        if value is not None and hasattr(value, "asset_name") and hasattr(value, "class_type"):
            return field.name, value
    return "", None


def _collect_obs_metadata(env) -> dict:
    """按 observation manager 的真实顺序/维度 + cfg 里的 scale 收集 obs 描述。"""
    manager = env.unwrapped.observation_manager
    active_terms: dict[str, list[str]] = manager.active_terms  # type: ignore[assignment]
    if not active_terms:
        raise RuntimeError("observation manager 没有任何 obs group")
    group = "policy" if "policy" in active_terms else next(iter(active_terms))
    names = list(active_terms[group])
    dims = [int(math.prod(dim)) for dim in manager.group_obs_term_dim[group]]  # type: ignore[index]

    cfgs = _obs_term_cfgs(env.unwrapped.cfg, group)
    terms: list[dict] = []
    for name, dim in zip(names, dims):
        cfg = cfgs.get(name)
        scale = getattr(cfg, "scale", None) if cfg is not None else None
        func = getattr(cfg, "func", None) if cfg is not None else None
        terms.append(
            {
                "name": name,
                "dim": dim,
                "scale": 1.0 if scale is None else float(scale) if isinstance(scale, (int, float)) else scale,
                "func": _callable_name(func) if func is not None else None,
                "params": _json_safe(getattr(cfg, "params", {}) or {}) if cfg is not None else {},
            }
        )
    return {
        "group": group,
        "dim": int(sum(dims)),
        "order": names,
        "terms": terms,
        "note": (
            "obs = 按 order 顺序把各 term 的输出拼接（concatenate_terms=True）；"
            "每个 term 先乘它自己的 scale。"
        ),
    }


def _collect_joint_convention() -> dict:
    """关节顺序 + squat 姿态 + 符号约定（优先从 kinematics/robots 读，失败则降级）。"""
    hip_order = knee_order = wheel_order = None
    squat: dict | None = None
    wheel_radius = None
    warnings: list[str] = []

    try:
        from chassis_rl_lab.kinematics import HIP_ORDER, KNEE_ORDER, WHEEL_ORDER, load_squat_pose

        hip_order = list(HIP_ORDER)
        knee_order = list(KNEE_ORDER)
        wheel_order = list(WHEEL_ORDER)
        squat = load_squat_pose()
        wheel_radius = squat.get("wheel_radius") if isinstance(squat, dict) else None
    except Exception as exc:  # noqa: BLE001 - 元数据不应因可选依赖失败而中断导出
        warnings.append(f"无法从 chassis_rl_lab.kinematics 读取关节顺序/squat 姿态：{exc}")

    convention = {
        "frame": "base_link：x 前 / y 左 / z 上",
        "joint_positive": (
            "所有关节统一以【绕世界 +y 轴右手定则旋转为正】（从 +y 看向 -y 逆时针）；"
            "左右腿同号（不再有镜像符号）。"
        ),
        "hip": {
            "positive": {
                "front": "前 hip 正转 = 前大腿向下摆（腿伸长方向）",
                "back": "后 hip 负转 = 后大腿向下摆（腿伸长方向）",
            },
        },
        "knee": {
            "positive": {"front": "前膝负转 = 伸长（行程约 114.5°）", "back": "后膝正转 = 伸长（行程约 114.5°）"},
            "passive": (
                "4 个 knee 是五连杆闭链的被动随动关节（真实执行器只有 4 hip + 2 wheel）；"
                "部署端必须用同一套解析闭链解算（chassis_rl_lab.kinematics.FiveBarSolver）"
                "由实测 hip 角求出 knee 位置目标，并保留与训练相同的 knee_margin。"
            ),
        },
        "wheel_positive": "轮子正转 = 车体向 +x（前进）；前进需要轮子正速度。",
        "q_zero": "q = 0 = CAD 装配位 = 机械行程的收回端；squat 姿态在 q≈0 附近。",
    }
    return {
        "hip_order": hip_order,
        "knee_order": knee_order,
        "wheel_order": wheel_order,
        "squat_pose": squat,
        "wheel_radius": wheel_radius,
        "convention": convention,
        "warnings": warnings,
    }


def _json_safe(value: object) -> object:
    """把 numpy/torch 标量等转成 JSON 可序列化对象。"""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:  # noqa: BLE001
        pass
    try:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:  # noqa: BLE001
        pass
    return str(value)


def main() -> int:
    import importlib.metadata as metadata

    installed_version = metadata.version("rsl-rl-lib")

    # ---------------- 1. 读取 cfg ----------------
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run

    device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.device = device
    agent_cfg.device = device
    # 导出只需要构建 env 读取维度/配置；1 个环境即可
    env_cfg.scene.num_envs = 1
    env_cfg.log_dir = None

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # ---------------- 2. 解析 checkpoint ----------------
    if args_cli.log_root is not None:
        log_root_path = os.path.abspath(args_cli.log_root)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = resolve_checkpoint(log_root_path, agent_cfg.load_run, args_cli.checkpoint)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)

    # ---------------- 3. 输出路径 ----------------
    output_path = Path(args_cli.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    if output_path.suffix.lower() != ".onnx":
        output_path = output_path.with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".json")

    print("=" * 78)
    print(f"[INFO] 任务              : {args_cli.task}")
    print(f"[INFO] 设备              : {device}")
    print(f"[INFO] rsl-rl 版本       : {installed_version}")
    print(f"[INFO] checkpoint        : {resume_path}")
    print(f"[INFO] ONNX 输出         : {output_path}")
    print("=" * 78)

    # ---------------- 4. 构建 env / 加载策略 ----------------
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.alg.get_policy()

    obs_dim = int(policy.obs_dim)
    action_dim = int(env.num_actions)

    # ---------------- 5. 导出 ONNX ----------------
    # rsl_rl 5.0.1: OnPolicyRunner.export_policy_to_onnx -> MLPModel.as_onnx(_OnnxMLPModel)
    #   input names = ["obs"], output names = ["actions"], opset=18, batch 固定 1
    runner.export_policy_to_onnx(
        path=str(output_path.parent),
        filename=output_path.name,
        verbose=bool(args_cli.verbose),
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ONNX 导出失败：{output_path} 不存在或为空")
    print(f"[INFO] ONNX 已导出：{output_path} ({output_path.stat().st_size / 1024:.1f} KiB)")

    # 注意：不要在 Isaac Sim 进程内调用 onnx.checker.check_model()。
    # onnx 1.22 的 Windows native checker 在这个进程里会 segfault（access violation），
    # 导致后面的元数据 JSON 写不出来。ONNX 结构校验请用独立进程：
    #   python -c "import onnx; m=onnx.load('exports/chassis_standup_policy.onnx'); onnx.checker.check_model(m); print('OK')"
    onnx_check = "skipped_in_process"
    print("[INFO] 跳过进程内 onnx.checker（已知 Windows 崩溃问题）；请用独立进程校验 ONNX")

    # ---------------- 6. 元数据 ----------------
    obs_meta = _collect_obs_metadata(env)
    if obs_meta["dim"] != obs_dim:
        raise RuntimeError(
            f"obs 维度不一致：observation_manager 合计 {obs_meta['dim']}，policy.obs_dim={obs_dim}。"
            " 请检查 obs_groups / observation 配置。"
        )

    action_term_name, action_cfg = _action_term_cfg(env.unwrapped.cfg)
    hip_scale = float(getattr(action_cfg, "hip_scale", 1.0)) if action_cfg is not None else 1.0
    wheel_scale = float(getattr(action_cfg, "wheel_scale", 1.0)) if action_cfg is not None else 1.0
    hip_control = str(getattr(action_cfg, "hip_control", "position")) if action_cfg is not None else "position"
    hip_torque_scale = float(getattr(action_cfg, "hip_torque_scale", 1.0)) if action_cfg is not None else 1.0
    wheel_control = str(getattr(action_cfg, "wheel_control", "velocity")) if action_cfg is not None else "velocity"
    wheel_torque_scale = float(getattr(action_cfg, "wheel_torque_scale", 1.0)) if action_cfg is not None else 1.0
    clip_actions = getattr(action_cfg, "clip_actions", agent_cfg.clip_actions) if action_cfg is not None else agent_cfg.clip_actions

    # 力气/高度相关的部署常量
    try:
        from chassis_rl_lab.robots.chassis_rl_cfg import (
            HIP_EFFORT_LIMIT,
            HIP_PEAK_EFFORT,
            WHEEL_EFFORT_LIMIT,
        )

        hip_effort_limit = float(HIP_EFFORT_LIMIT)
        hip_peak_effort = float(HIP_PEAK_EFFORT)
        wheel_effort_limit = float(WHEEL_EFFORT_LIMIT)
    except Exception:  # noqa: BLE001 - 读不到就降级成 None，不阻断导出
        hip_effort_limit = hip_peak_effort = wheel_effort_limit = None
    nominal_height = getattr(env_cfg, "nominal_height", None)
    height_range = getattr(env_cfg, "height_command_range", None)

    joints = _collect_joint_convention()
    hip_order = joints["hip_order"] or list(getattr(action_cfg, "hip_joint_names", []) or [])
    knee_order = joints["knee_order"] or list(getattr(action_cfg, "knee_joint_names", []) or [])
    wheel_order = joints["wheel_order"] or list(getattr(action_cfg, "wheel_joint_names", []) or [])

    control_hz = 1.0 / (float(env_cfg.sim.dt) * int(env_cfg.decimation))
    squat_pose = joints["squat_pose"]
    squat_base_height = None
    if isinstance(squat_pose, dict):
        squat_base_height = squat_pose.get("base_height")
    if squat_base_height is None:
        try:
            squat_base_height = float(env_cfg.scene.robot.init_state.pos[2])
        except Exception:  # noqa: BLE001
            squat_base_height = None

    metadata: dict = {
        "format": "chassis_rl_lab_policy_metadata",
        "format_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": args_cli.task,
        "checkpoint": os.path.abspath(resume_path),
        "rsl_rl_version": installed_version,
        "onnx": {
            "file": output_path.name,
            "path": str(output_path),
            "checker": onnx_check,
            "input_names": ["obs"],
            "output_names": ["actions"],
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "batch": 1,
            "opset": 18,
            "output_is_mean": True,
            "obs_normalization_included": bool(getattr(policy, "obs_normalization", False)),
            "note": (
                "onnx.checker 结果见 checker；batch 固定为 1（无 dynamic axes）。"
                "obs_normalization_included=True 表示 ONNX 图内已包含 EmpiricalNormalization，"
                "部署侧不要再次归一化。"
            ),
        },
        "observations": obs_meta,
        "actions": {
            "dim": action_dim,
            "term_name": action_term_name,
            "term_type": getattr(action_cfg.class_type, "__name__", str(action_cfg.class_type)) if action_cfg is not None else None,
            "hip_control": hip_control,
            "wheel_control": wheel_control,
            "hip_scale": hip_scale,
            "wheel_scale": wheel_scale,
            "hip_torque_scale_nm_per_unit": hip_torque_scale,
            "wheel_torque_scale_nm_per_unit": wheel_torque_scale,
            "hip_effort_limit_nm": hip_effort_limit,
            "hip_peak_effort_nm": hip_peak_effort,
            "wheel_effort_limit_nm": wheel_effort_limit,
            "clip_actions": clip_actions,
            "mapping": (
                {
                    "a[0:4]": (
                        f"hip_torque[i] = clamp(hip_torque_scale * a[i], ±{hip_effort_limit} Nm)"
                        "（顺序 = hip_order；髋电机峰值 40 Nm 但持续力矩按 20 Nm 计，训练/部署都用 20 Nm）"
                    ),
                    "a[4:6]": (
                        f"wheel_torque[i] = clamp(wheel_torque_scale * a[4+i], ±{wheel_effort_limit} Nm)"
                        "（顺序 = wheel_order；正力矩 = 车体 +x 前进）"
                    ),
                    "knee": (
                        "knee 不在动作里：每个控制步用 FiveBarSolver 由实测 hip 解出 knee 目标"
                        "（warm start + 限位内留 knee_margin）后作为位置目标下发"
                    ),
                }
                if hip_control == "torque" and wheel_control == "torque"
                else {
                    "a[0:4]": "hip_target[i] = squat_hip[i] + hip_scale * a[i]（顺序 = hip_order，最后 clamp 到关节限位）",
                    "a[4:6]": "wheel_target[i] = wheel_scale * a[4+i]（顺序 = wheel_order，单位 rad/s）",
                    "knee": "knee 不在动作里：每个控制步用 FiveBarSolver 由实测 hip 解出 knee 目标后作为位置目标下发",
                }
            ),
            "note": "策略动作先被 clip 到 ±clip_actions，再按上表的 scale/limit 映射为关节力矩。",
        },
        "height_command": {
            "channel": "observations.commands[2]（commands = [lin_vel_x_cmd, ang_vel_yaw_cmd, height_cmd]）",
            "unit": "m",
            "obs_scale": 5.0,
            "range_m": list(height_range) if height_range is not None else None,
            "nominal_height_m": nominal_height,
            "note": (
                "部署端把目标机体高度填到 commands[2]（观测里乘 5），前两维固定 0（本任务不做速度跟踪）。"
                "策略输出不含高度目标，高度跟踪由观测中的高度指令驱动。"
            ),
        },
        "joints": {
            "hip_order": hip_order,
            "knee_order": knee_order,
            "wheel_order": wheel_order,
            "convention": joints["convention"],
        },
        "squat_pose": {
            "data": squat_pose,
            "base_height": squat_base_height,
            "note": "hips/knees 为 squat 姿态绝对关节角（rad），左右相同；data 由数据文件原样转储。",
        },
        "wheel": {
            "radius": joints["wheel_radius"],
            "positive_direction": "轮子正转 = 车体向 +x（前进）；前进需要正轮速。",
        },
        "sim": {
            "dt": float(env_cfg.sim.dt),
            "decimation": int(env_cfg.decimation),
            "control_hz": control_hz,
            "episode_length_s": float(env_cfg.episode_length_s),
            "physics_engine": "PhysX (Isaac Sim 5.1 / Isaac Lab 2.3.2)",
        },
        "deployment_notes": [
            "观测：按 observations.order 顺序拼接各 term，并逐项乘 observations.terms[i].scale。",
            "若 onnx.obs_normalization_included=true，ONNX 内部已归一化，部署端不得重复归一化。",
            "动作：ONNX 输出为动作均值（[1, action_dim]），按 actions.mapping 换算 hip 位置目标与轮速目标。",
            "knee 必须用与训练一致的解析闭链解算（同一份 chassis_rl_planar.json / squat_pose.json）。",
            "关节符号：所有关节绕世界 +y 轴右手定则正转为正，左右同号；轮子正转 = 车体前进（+x）。",
        ],
        "warnings": joints["warnings"],
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, ensure_ascii=False, indent=2)
    print(f"[INFO] 元数据已写入：{metadata_path}")

    env.close()

    # 摘要打印
    print("-" * 78)
    print(f"[SUMMARY] obs dim = {obs_dim}  terms = " + ", ".join(f"{t['name']}({t['dim']})" for t in obs_meta["terms"]))
    print(f"[SUMMARY] act dim = {action_dim}  hip_scale = {hip_scale} rad  wheel_scale = {wheel_scale} rad/s")
    print(f"[SUMMARY] control = {control_hz:.1f} Hz (sim.dt={env_cfg.sim.dt:.5f}, decimation={env_cfg.decimation})")
    print(f"[SUMMARY] squat base_height = {squat_base_height}")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        exit_code = 1
    finally:
        # 注意：Windows 上 Isaac Sim 的 close() 偶发挂起/崩溃，只 flush 后强制退出；
        # 元数据 JSON 在 main() 里已经写完，不受影响。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
