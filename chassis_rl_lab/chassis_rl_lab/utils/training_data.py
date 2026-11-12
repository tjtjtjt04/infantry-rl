"""训练数据导出：把一次训练 run 的指标整理成 CSV + 图像 + 摘要，存到 ``data/<时间戳_run名>/``。

用法
----
训练结束时 ``scripts/train.py`` 会自动调用（默认导出到项目根的 ``data/`` 目录）；
也可以对已有 run 单独导出::

    D:\\miniconda\\envs\\RM\\python.exe tools\\export_training_data.py ^
        --log_dir logs\\rsl_rl\\chassis_rl_stand_up\\<run>

产物
----
``data/<run 目录名>/``（run 目录名自带时间戳，例如 ``2026-11-06_12-30-00_v1``）::

    metrics.csv         长格式：iteration,tag,value（TensorBoard 全部标量）
    summary.json        run 摘要：每个标量的 first/last/min/max、checkpoint 列表、配置
    README.md           人读摘要
    env.yaml / agent.yaml   训练用的环境/算法配置（从 run 目录拷贝）
    plots/
        00_height_tracking.png  Height 任务：目标高度 vs 当前高度 / 误差
    01_train.png            平均奖励 / 平均 episode 长度
        02_episode_reward.png   Episode_Reward/* 各分项
        03_termination.png      Episode_Termination/* 终止比例
        04_losses.png           value/surrogate/entropy loss
        05_policy_lr.png        learning rate / action std
        06_perf.png             FPS / 采样与学习耗时
        07_base_state.png       机体高度 / pitch-roll / 速度
        08_joint_positions.png  各关节位置（hip/knee/wheel）
        09_joint_velocities.png 各关节速度
        10_joint_torques.png    各关节力矩

图表使用 matplotlib Agg 后端（无窗口），tag 名直接做坐标轴标签（英文，避免字体问题）。
"""

from __future__ import annotations

import fnmatch

import json
import shutil
import time
from pathlib import Path

__all__ = ["export_training_data"]

#: 每个图文件包含的 TensorBoard tag 列表（tag 不存在时自动跳过）
_PLOT_GROUPS: dict[str, dict[str, list[str]]] = {
    # "目标 vs 当前"：Height 任务的指令高度与实际机体高度（其它任务无这些 tag 时自动跳过）
    "00_height_tracking.png": {
        "Height target / now [m]": ["Height/target", "Height/now"],
        "Height error [m]": ["Height/error"],
    },
    "01_train.png": {
        "Train/mean_reward": ["Train/mean_reward"],
        "Train/mean_episode_length": ["Train/mean_episode_length"],
    },
    "02_episode_reward.png": {
        "Episode_Reward": ["Episode_Reward/*"],
    },
    "03_termination.png": {
        "Episode_Termination": ["Episode_Termination/*"],
    },
    "04_losses.png": {
        "Loss": ["Loss/value", "Loss/surrogate", "Loss/entropy"],
    },
    "05_policy_lr.png": {
        "Learning rate": ["Loss/learning_rate"],
        "Action std": ["Policy/mean_std"],
    },
    "06_perf.png": {
        "FPS": ["Perf/total_fps"],
        "Collection / learning time": ["Perf/collection_time", "Perf/learning_time"],
    },
    # 机体姿态/速度（实际表现）
    "07_base_state.png": {
        "Base height [m]": ["State/base_height"],
        "Pitch / roll [deg]": ["State/pitch_deg", "State/roll_deg"],
        "Base lin vel xy / ang vel z": ["State/lin_vel_xy", "State/ang_vel_z"],
        "Hip split (left-right mismatch) [rad]": ["State/hip_split"],
    },
    # 每个关节的实际行为（位置/速度/力矩）
    "08_joint_positions.png": {
        "Hip positions [rad]": ["Joint/*hip_pos"],
        "Knee positions [rad]": ["Joint/*knee_pos"],
        "Wheel positions [rad]": ["Joint/wheel*_pos"],
    },
    "09_joint_velocities.png": {
        "Hip velocities [rad/s]": ["Joint/*hip_vel"],
        "Knee velocities [rad/s]": ["Joint/*knee_vel"],
        "Wheel velocities [rad/s]": ["Joint/wheel*_vel"],
    },
    "10_joint_torques.png": {
        "Hip torques [Nm]": ["Joint/*hip_tau"],
        "Knee torques [Nm]": ["Joint/*knee_tau"],
        "Wheel torques [Nm]": ["Joint/wheel*_tau"],
    },
}

#: 长格式 CSV 列
_CSV_HEADER = "iteration,tag,value\n"
#: 高度跟踪（目标 vs 当前）专用 CSV 列
_HEIGHT_CSV_HEADER = "iteration,height_target,height_now,height_error\n"


def _write_height_tracking_csv(
    scalars: dict[str, list[tuple[int, float]]], out_dir: Path
) -> str | None:
    """写 ``height_tracking.csv``（目标高度/当前高度/误差，按 iteration 对齐）。

    Height 任务训练时由精简 logger 每轮写 ``Height/target|now|error`` 到 TensorBoard；
    这里把它们导出成一张宽表，方便直接看"目标 vs 当前"。没有这些 tag 时跳过。
    """
    target = dict(scalars.get("Height/target", []))
    now = dict(scalars.get("Height/now", []))
    error = dict(scalars.get("Height/error", []))
    steps = sorted(set(target) | set(now) | set(error))
    if not steps:
        return None
    path = out_dir / "height_tracking.csv"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_HEIGHT_CSV_HEADER)
        for step in steps:
            handle.write(
                f"{step},{target.get(step, float('nan')):.6f},{now.get(step, float('nan')):.6f},"
                f"{error.get(step, float('nan')):.6f}\n"
            )
    return path.name


def _find_event_files(log_dir: Path) -> list[Path]:
    files = sorted(log_dir.glob("events.out.tfevents*"))
    if not files:
        # 某些版本会写在子目录
        files = sorted(log_dir.rglob("events.out.tfevents*"))
    return files


def _load_scalars(log_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """读取 run 目录里所有 TensorBoard 标量，返回 ``{tag: [(step, value), ...]}``。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = _find_event_files(log_dir)
    if not files:
        raise FileNotFoundError(f"{log_dir} 下没有 TensorBoard events 文件（events.out.tfevents*）")

    scalars: dict[str, list[tuple[int, float]]] = {}
    for event_file in files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags()["scalars"]:
            points = scalars.setdefault(tag, [])
            points.extend((int(event.step), float(event.value)) for event in accumulator.Scalars(tag))
    for points in scalars.values():
        points.sort(key=lambda p: p[0])
    return scalars


def _expand_group(scalars: dict[str, list[tuple[int, float]]], patterns: list[str]) -> list[str]:
    """把 ``Episode_Reward/*``、``Joint/*hip_pos`` 这类通配展开成实际存在的 tag 列表。"""
    tags: list[str] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            tags.extend(sorted(fnmatch.filter(scalars.keys(), pattern)))
        elif pattern in scalars:
            tags.append(pattern)
    # 去重并保持顺序
    seen: set[str] = set()
    unique: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique


def _write_plots(scalars: dict[str, list[tuple[int, float]]], plot_dir: Path) -> list[str]:
    """生成各组 PNG，返回实际生成的文件名列表。"""
    import matplotlib

    matplotlib.use("Agg")  # 无窗口后端（必须在 pyplot 之前设置）
    import matplotlib.pyplot as plt

    written: list[str] = []
    for filename, panels in _PLOT_GROUPS.items():
        panels_resolved = {
            title: _expand_group(scalars, patterns) for title, patterns in panels.items()
        }
        panels_resolved = {title: tags for title, tags in panels_resolved.items() if tags}
        if not panels_resolved:
            continue

        fig, axes = plt.subplots(len(panels_resolved), 1, figsize=(10, 3.2 * len(panels_resolved)))
        if len(panels_resolved) == 1:
            axes = [axes]
        for ax, (title, tags) in zip(axes, panels_resolved.items()):
            for tag in tags:
                steps = [p[0] for p in scalars[tag]]
                values = [p[1] for p in scalars[tag]]
                ax.plot(steps, values, label=tag, linewidth=1.2)
            ax.set_title(title)
            ax.set_xlabel("iteration")
            ax.grid(True, alpha=0.3)
            if len(tags) <= 8:
                ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        out_path = plot_dir / filename
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        written.append(filename)
    return written


def _tag_summary(points: list[tuple[int, float]]) -> dict:
    values = [v for _, v in points]
    return {
        "n": len(points),
        "first_iter": points[0][0],
        "last_iter": points[-1][0],
        "first_value": values[0],
        "last_value": values[-1],
        "min": min(values),
        "max": max(values),
    }


def export_training_data(
    log_dir: str | Path,
    data_root: str | Path = "data",
    extra: dict | None = None,
    snapshot: bool = False,
) -> Path:
    """导出一次训练 run 的指标 → ``data/<run 目录名>/``，返回导出目录。

    Args:
        log_dir: 训练 run 目录（含 TensorBoard events / env.yaml / agent.yaml）。
        data_root: 数据根目录（默认相对当前工作目录的 ``data/``）。
        extra: 额外写入 summary.json 的字段（如 checkpoint 路径、训练耗时等）。
        snapshot: True 时在 ``data/<run 目录名>/snapshot_<时间戳>/`` 下保存本次快照
            （用于训练中周期性导出"从开始到当前"的累计数据，**不覆盖历史快照**）；
            False 时若 ``data/<run 目录名>/`` 已存在则加时间戳后缀。

    Returns:
        导出目录路径。
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        raise FileNotFoundError(f"run 目录不存在：{log_dir}")
    scalars = _load_scalars(log_dir)

    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    run_dir = data_root / log_dir.name
    if snapshot:
        # 训练中的周期导出：在 run 目录下按时间戳建快照，历史数据全部保留
        run_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = run_dir / f"snapshot_{stamp}"
        suffix = 1
        while out_dir.exists():
            out_dir = run_dir / f"snapshot_{stamp}_{suffix}"
            suffix += 1
    else:
        out_dir = run_dir
        if out_dir.exists():  # 同一 run 重复导出：加后缀避免覆盖
            out_dir = data_root / f"{log_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    # 1) metrics.csv（长格式，便于 pandas/Excel 直接读）
    with open(out_dir / "metrics.csv", "w", encoding="utf-8") as handle:
        handle.write(_CSV_HEADER)
        for tag in sorted(scalars):
            for step, value in scalars[tag]:
                handle.write(f"{step},{tag},{value:.8g}\n")

    # 2) 高度跟踪专用 CSV（目标 vs 当前）
    height_csv = _write_height_tracking_csv(scalars, out_dir)

    # 3) 图
    plot_files = _write_plots(scalars, out_dir / "plots")

    # 3) 拷贝配置
    configs: list[str] = []
    for name in ("env.yaml", "agent.yaml"):
        src = log_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)
            configs.append(name)

    # 4) checkpoint 列表
    checkpoints = sorted(
        (p.name for p in log_dir.glob("model_*.pt")),
        key=lambda n: int("".join(ch for ch in Path(n).stem if ch.isdigit()) or 0),
    )

    # 5) summary.json / README.md
    summary = {
        "run_name": log_dir.name,
        "log_dir": str(log_dir.resolve()),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(out_dir.resolve()),
        "num_tags": len(scalars),
        "tags": {tag: _tag_summary(points) for tag, points in sorted(scalars.items())},
        "plots": plot_files,
        "height_tracking_csv": height_csv,
        "configs": configs,
        "checkpoints": checkpoints,
        "latest_checkpoint": checkpoints[-1] if checkpoints else None,
    }
    if extra:
        summary["extra"] = extra
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    train_reward = summary["tags"].get("Train/mean_reward")
    train_length = summary["tags"].get("Train/mean_episode_length")
    readme = [
        f"# 训练数据：{log_dir.name}",
        "",
        f"- 导出时间：{summary['exported_at']}",
        f"- 来源 run 目录：`{log_dir}`",
        f"- 指标数量：{len(scalars)} 个 TensorBoard 标量",
        f"- 最新 checkpoint：`{summary['latest_checkpoint']}`",
        "",
        "## 关键曲线终点",
    ]
    if train_reward:
        readme.append(
            f"- `Train/mean_reward`：{train_reward['last_value']:.4f}"
            f"（iter {train_reward['last_iter']}）"
        )
    if train_length:
        readme.append(
            f"- `Train/mean_episode_length`：{train_length['last_value']:.1f}"
            f"（iter {train_length['last_iter']}）"
        )
    readme += ["", f"## 文件", "", "- `metrics.csv`：iteration,tag,value 长格式", "- `plots/`：" + ", ".join(plot_files)]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    return out_dir
