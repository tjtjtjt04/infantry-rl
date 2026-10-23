# -*- coding: utf-8 -*-
"""
make_report_data.py
===================
从两个 rsl_rl TensorBoard 事件文件（只读）提取标量，生成训练报告所需的数据文件：

  report/data/run_A_scalars.csv      阶段A 宽表（step + 每个标量 tag 一列）
  report/data/run_B_scalars.csv      阶段B 宽表（step + 每个标量 tag 一列）
  report/data/run_ranges.json        每个 run 的精确 step 范围、事件数、tag 列表、连续性判断
  report/data/metrics_summary.json   窗口均值 / 变化量 / 最佳百轮 / 阶段B 终值（机器可读）
  report/data/metrics_summary.md     同上，中文表格（人类可读）

运行方式：
  D:\\miniconda\\envs\\RM\\python.exe report\\data\\make_report_data.py

注意：
- 仅读取 logs/ 下的 events 文件；只写 report/data/ 下文件。
- Train/mean_reward/time、Train/mean_episode_length/time 两个 tag 的 TensorBoard step
  是累计训练秒数而不是迭代号（rsl_rl 的 logger 用 tot_time 作 step），
  无法与迭代号对齐，因此不写入宽表 CSV，在 run_ranges.json 中单独报告。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = Path(r"D:\Robomaster\RL")
RUN_ROOT = BASE / "logs" / "rsl_rl" / "infantry_rl_locomotion"
OUT_DIR = BASE / "report" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "A": {
        "label_zh": "阶段A（第1次训练，v1.0.7）",
        "dir": RUN_ROOT / "2026-10-22_00-28-38",
        "event_file": "events.out.tfevents.1792600120.CMD.10812.0",
        "resume": False,
    },
    "B": {
        "label_zh": "阶段B（续训，v1.0.8）",
        "dir": RUN_ROOT / "2026-10-22_01-17-16",
        "event_file": "events.out.tfevents.1792603039.CMD.12320.0",
        "resume": True,
    },
}

TZ = timezone(timedelta(hours=8))

# ---------------------------------------------------------------- 指标分组（报告表格用）
GROUPS = [
    ("训练信号", [
        "Train/mean_reward",
        "Train/mean_episode_length",
    ]),
    ("速度跟踪奖励项", [
        "Episode_Reward/track_lin_vel_x",
        "Episode_Reward/track_lin_vel_x_l1",
        "Episode_Reward/track_lin_vel_x_l2",
        "Episode_Reward/track_ang_vel_z",
        "Episode_Reward/track_ang_vel_z_l1",
        "Episode_Reward/track_ang_vel_z_l2",
    ]),
    ("姿态 / 高度奖励项", [
        "Episode_Reward/base_height",
        "Episode_Reward/pitch",
        "Episode_Reward/roll",
        "Episode_Reward/roll_straight",
    ]),
    ("平滑 / 能耗 / 其它奖励项", [
        "Episode_Reward/action_rate",
        "Episode_Reward/action_jerk",
        "Episode_Reward/joint_acc",
        "Episode_Reward/joint_torques",
        "Episode_Reward/wheel_torques",
        "Episode_Reward/wheel_axle",
        "Episode_Reward/wheel_spin",
        "Episode_Reward/closure_residual",
        "Episode_Reward/fall_penalty",
        "Episode_Reward/alive",
        "Episode_Reward/heading_error",
        "Episode_Reward/lin_vel_y",
        "Episode_Reward/lin_vel_z",
        "Episode_Reward/roll_rate",
    ]),
    ("速度跟踪误差（越低越好）", [
        "Metrics/base_velocity/error_vel_xy",
        "Metrics/base_velocity/error_vel_yaw",
    ]),
    ("PPO 内部量", [
        "Loss/value",
        "Loss/surrogate",
        "Loss/entropy",
        "Loss/learning_rate",
        "Policy/mean_std",
    ]),
    ("回合终止率", [
        "Episode_Termination/time_out",
        "Episode_Termination/bad_orientation",
        "Episode_Termination/base_too_low",
    ]),
    ("训练性能", [
        "Perf/total_fps",
        "Perf/collection_time",
        "Perf/learning_time",
    ]),
]

SUMMARY_TAGS = [t for _, tags in GROUPS for t in tags]


# ---------------------------------------------------------------- 工具
def fmt_value(tag: str, v) -> str:
    """按 tag 类型给出可读数值；None -> —"""
    if v is None:
        return "—"
    if tag == "Train/mean_reward":
        return f"{v:.3f}"
    if tag == "Train/mean_episode_length":
        return f"{v:.2f}"
    if tag.startswith("Metrics/base_velocity/"):
        return f"{v:.4f}"
    if tag == "Loss/learning_rate":
        return f"{v:.3e}"
    if tag in ("Loss/value",):
        return f"{v:.4f}"
    if tag == "Loss/surrogate":
        return f"{v:.5f}"
    if tag == "Loss/entropy":
        return f"{v:.3f}"
    if tag == "Policy/mean_std":
        return f"{v:.4f}"
    if tag.startswith("Episode_Termination/"):
        return f"{v:.4f}"
    if tag == "Perf/total_fps":
        return f"{v:.0f}"
    if tag.startswith("Perf/"):
        return f"{v:.3f}"
    if tag.startswith("Episode_Reward/"):
        if v == 0:
            return "0"
        if abs(v) < 1e-4:
            return f"{v:.3e}"
        return f"{v:.5f}"
    return f"{v:.5g}"


def to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=TZ).strftime("%Y-%m-%d %H:%M:%S")


def load_run(run: dict) -> dict:
    """读取一个 run 的所有标量：返回 tags / events(DataFrame) 等。"""
    path = run["dir"] / run["event_file"]
    if not path.exists():
        raise FileNotFoundError(path)
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    scalars = ea.Tags()["scalars"]
    frames = {}
    meta = {}
    for tag in sorted(scalars):
        events = ea.Scalars(tag)
        df = pd.DataFrame(
            {
                "step": [e.step for e in events],
                "value": [e.value for e in events],
                "wall_time": [e.wall_time for e in events],
            }
        )
        frames[tag] = df
        meta[tag] = {
            "count": len(events),
            "first_step": int(df["step"].iloc[0]),
            "last_step": int(df["step"].iloc[-1]),
        }
    return {"path": path, "frames": frames, "meta": meta, "tags": sorted(scalars)}


def classify_tags(run_data: dict):
    """把 tag 分成 '迭代轴' 与 '时间轴' 两类。"""
    time_tags, iter_tags = [], []
    for tag in run_data["tags"]:
        # rsl_rl 把累计训练时长写进 step 的 tag 以 '/time' 结尾
        if tag.endswith("/time"):
            time_tags.append(tag)
        else:
            iter_tags.append(tag)
    return iter_tags, time_tags


def build_wide(run_data: dict, iter_tags) -> pd.DataFrame:
    """按 TensorBoard step 对齐，生成宽表（first column: step）。"""
    wide = None
    for tag in iter_tags:
        df = run_data["frames"][tag][["step", "value"]].rename(columns={"value": tag})
        wide = df if wide is None else wide.merge(df, on="step", how="outer", validate="one_to_one")
    if wide is None:
        raise RuntimeError("no iteration tags")
    wide = wide.sort_values("step").reset_index(drop=True)
    # 检查是否有 tag 缺 step（产生 NaN）
    nan_cols = [c for c in wide.columns if c != "step" and wide[c].isna().any()]
    if nan_cols:
        print("[WARN] 宽表存在缺 step 的 tag:", nan_cols[:5], "... 共", len(nan_cols))
    return wide


def window_stats(df: pd.DataFrame, tag: str, start_idx: int, end_idx: int) -> dict:
    """闭区间 [start_idx, end_idx]（行号，0-based）窗口的均值与 step 范围。"""
    sub = df.iloc[start_idx : end_idx + 1]
    vals = sub[tag].to_numpy(dtype=float)
    return {
        "num": int(len(sub)),
        "start_step": int(sub["step"].iloc[0]),
        "end_step": int(sub["step"].iloc[-1]),
        "mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else None,
    }


def window_index(n: int, s: int, e: int) -> list:
    """把窗口定义 (s, e) 转成 0-based 行号列表。s/e 为负表示从末尾数。"""
    if s >= 0:
        return list(range(s, n if e == -1 else e + 1))
    return list(range(n + s, n))


def best_rolling_mean(df: pd.DataFrame, tag: str, window: int) -> dict:
    vals = df[tag].to_numpy(dtype=float)
    steps = df["step"].to_numpy(dtype=int)
    n = len(vals)
    if n < window:
        return {"window": window, "windows": 0, "mean": None}
    roll = pd.Series(vals).rolling(window=window, min_periods=window).mean().to_numpy()
    valid = np.where(np.isfinite(roll))[0]
    if len(valid) == 0:
        return {"window": window, "windows": 0, "mean": None}
    # 并列最佳时取最早窗口
    best_end = int(valid[np.argmax(roll[valid])])
    best_start = best_end - window + 1
    return {
        "window": window,
        "windows": int(len(valid)),
        "mean": float(roll[best_end]),
        "start_step": int(steps[best_start]),
        "end_step": int(steps[best_end]),
        "center_step": int(round((steps[best_start] + steps[best_end]) / 2)),
    }


def config_diff() -> dict:
    """对比两个 run 的 params/env.yaml，提取真正变化的 reward 配置项。"""
    def load_env(run):
        p = run["dir"] / "params" / "env.yaml"
        if not p.exists():
            return {}
        with open(p, encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.UnsafeLoader)

    def flat(d, prefix=""):
        out = {}
        if isinstance(d, dict):
            for k, v in d.items():
                out.update(flat(v, prefix + "/" + str(k)))
        elif isinstance(d, list):
            for i, v in enumerate(d):
                out.update(flat(v, prefix + "/" + str(i)))
        else:
            out[prefix] = d
        return out

    fa, fb = flat(load_env(RUNS["A"])), flat(load_env(RUNS["B"]))
    keys = sorted(set(fa) | set(fb))
    changes = {}
    for k in keys:
        va, vb = fa.get(k, None), fb.get(k, None)
        if str(va) != str(vb):
            changes[k] = {"A": va, "B": vb}
    return changes


def clean_float(v):
    if v is None:
        return None
    v = float(v)
    return v


# ---------------------------------------------------------------- 主流程
def main() -> int:
    print("=" * 78)
    print("Extract TensorBoard scalars -> report/data/")
    print("=" * 78)

    loaded = {}
    for key, run in RUNS.items():
        print(f"[load] 阶段{key}: {run['dir'] / run['event_file']}")
        loaded[key] = load_run(run)
        print(f"        tags={len(loaded[key]['tags'])}, "
              f"scalars events={sum(m['count'] for m in loaded[key]['meta'].values())}")

    # ---------------- 宽表 CSV ----------------
    ranges = {"runs": {}, "tag_counts": {}}
    iter_tags = {}
    wide = {}
    for key, run in RUNS.items():
        rd = loaded[key]
        it_tags, time_tags = classify_tags(rd)
        iter_tags[key] = it_tags
        w = build_wide(rd, it_tags)
        wide[key] = w
        out_csv = OUT_DIR / f"run_{key}_scalars.csv"
        w.to_csv(out_csv, index=False, float_format="%.9g")
        print(f"[csv ] {out_csv.name}: rows={len(w)}, cols={len(w.columns)}, "
              f"step {int(w['step'].iloc[0])}..{int(w['step'].iloc[-1])}")

        reward_tags = [t for t in it_tags if t.startswith("Episode_Reward/")]
        term_tags = [t for t in it_tags if t.startswith("Episode_Termination/")]
        all_steps = sorted({int(s) for t in it_tags for s in rd["frames"][t]["step"]})
        wall_min = min(float(rd["frames"][t]["wall_time"].iloc[0]) for t in it_tags)
        wall_max = max(float(rd["frames"][t]["wall_time"].iloc[-1]) for t in it_tags)
        ranges["runs"][key] = {
            "label_zh": run["label_zh"],
            "resume": run["resume"],
            "run_dir": str(run["dir"].relative_to(BASE)),
            "event_file": run["event_file"],
            "first_step": int(w["step"].iloc[0]),
            "last_step": int(w["step"].iloc[-1]),
            "iteration_count": int(len(w)),
            "distinct_steps": len(all_steps),
            "contiguous_steps": bool(len(all_steps) == (all_steps[-1] - all_steps[0] + 1)),
            "tag_count_total": len(rd["tags"]),
            "iteration_tag_count": len(it_tags),
            "time_axis_tag_count": len(time_tags),
            "episode_reward_tag_count": len(reward_tags),
            "episode_termination_tag_count": len(term_tags),
            "episode_reward_tags": reward_tags,
            "excluded_time_axis_tags": time_tags,
            "nan_inf_tags": [
                t for t in it_tags
                if not np.isfinite(rd["frames"][t]["value"].to_numpy(dtype=float)).all()
            ],
            "first_event_wall_time": wall_min,
            "last_event_wall_time": wall_max,
            "first_event_local": to_iso(wall_min),
            "last_event_local": to_iso(wall_max),
            "wall_duration_hours": (wall_max - wall_min) / 3600.0,
        }
        ranges["tag_counts"][key] = {
            t: {
                "count": int(len(rd["frames"][t])),
                "first_step": int(rd["frames"][t]["step"].iloc[0]),
                "last_step": int(rd["frames"][t]["step"].iloc[-1]),
            }
            for t in rd["tags"]
        }

    # 连续性
    a_last = ranges["runs"]["A"]["last_step"]
    b_first = ranges["runs"]["B"]["first_step"]
    b_last = ranges["runs"]["B"]["last_step"]
    overlap = sorted(set(wide["A"]["step"].tolist()) & set(wide["B"]["step"].tolist()))
    union_tags = sorted(set(iter_tags["A"]) | set(iter_tags["B"]))
    ranges["step_continuity"] = {
        "phase_B_continues_phase_A": bool(b_first == a_last),
        "A_last_step": int(a_last),
        "B_first_step": int(b_first),
        "B_last_step": int(b_last),
        "overlap_steps": overlap,
        "overlap_count": len(overlap),
        "B_new_iterations_after_A": int(b_last - a_last),
        "note_zh": (
            f"阶段B 的 step 编号接着阶段A 继续（B 首步 {b_first} = A 末步 {a_last}），"
            f"因此两阶段在 step {overlap[0]} 处有 {len(overlap)} 个重叠步；"
            f"阶段B 共 {ranges['runs']['B']['iteration_count']} 条事件，"
            f"即续训后新增 {b_last - a_last} 个训练迭代。"
        ),
    }
    ranges["tag_union"] = {
        "union": union_tags,
        "only_in_A": sorted(set(iter_tags["A"]) - set(iter_tags["B"])),
        "only_in_B": sorted(set(iter_tags["B"]) - set(iter_tags["A"])),
    }
    ranges["transformations"] = [
        "宽表 CSV 以 iter 标量的 TensorBoard step 为行索引；float_format='%.9g'。",
        "Train/mean_reward/time 与 Train/mean_episode_length/time 的 step 是累计训练秒数"
        "（非迭代号），不写入宽表 CSV，其事件数/首末 step 见 tag_counts。",
        "未做任何平滑、重采样或裁剪；文件内容为 events 中的原始标量值。",
    ]

    # ---------------- 窗口统计 ----------------
    WINDOWS = [
        ("first_100", 0, 99),
        ("last_100", -100, -1),
        ("last_50", -50, -1),
    ]
    summary = {
        "generated_by": "report/data/make_report_data.py",
        "data_sources": {
            key: {
                "label_zh": run["label_zh"],
                "event_file": str(run["dir"] / run["event_file"]),
            }
            for key, run in RUNS.items()
        },
        "phases": {},
        "phase_A_start_vs_end_delta": {},
        "best_100_iteration_mean_reward": {},
        "final_values_phase_B": {},
        "config_changes_A_to_B": config_diff(),
        "notes_zh": [],
    }

    for key in RUNS:
        w = wide[key]
        rd = loaded[key]
        n = len(w)
        phase = {
            "label_zh": RUNS[key]["label_zh"],
            "first_step": int(w["step"].iloc[0]),
            "last_step": int(w["step"].iloc[-1]),
            "iterations": int(n),
            "wall_duration_hours": ranges["runs"][key]["wall_duration_hours"],
            "window_bounds": {
                name: {
                    "num": int(len(window_index(n, s, e))),
                    "start_step": int(w["step"].iloc[window_index(n, s, e)[0]]),
                    "end_step": int(w["step"].iloc[window_index(n, s, e)[-1]]),
                }
                for name, s, e in WINDOWS
            },
            "metrics": {},
        }
        for tag in sorted(rd["tags"]):
            if tag.endswith("/time"):
                continue
            vals = w[tag].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            row = {
                "first_value": clean_float(vals[0]) if finite[0] else None,
                "last_value": clean_float(vals[-1]) if finite[-1] else None,
                "min": clean_float(np.nanmin(vals)) if finite.any() else None,
                "max": clean_float(np.nanmax(vals)) if finite.any() else None,
                "mean_all": clean_float(np.nanmean(vals)) if finite.any() else None,
            }
            for name, s, e in WINDOWS:
                idx = window_index(n, s, e)
                sub = vals[idx]
                row[name] = clean_float(np.nanmean(sub)) if np.isfinite(sub).any() else None
                row[name + "_start_step"] = int(w["step"].iloc[idx[0]])
                row[name + "_end_step"] = int(w["step"].iloc[idx[-1]])
            phase["metrics"][tag] = row
        summary["phases"][key] = phase

    # 阶段A 前100 vs 后100 变化
    for tag in sorted(set(loaded["A"]["tags"])):
        if tag.endswith("/time"):
            continue
        m = summary["phases"]["A"]["metrics"][tag]
        f, l = m["first_100"], m["last_100"]
        if f is None or l is None:
            continue
        delta = l - f
        pct = (delta / abs(f) * 100.0) if abs(f) > 1e-12 else None
        summary["phase_A_start_vs_end_delta"][tag] = {
            "first_100_mean": f,
            "last_100_mean": l,
            "delta": delta,
            "pct_change": pct,
        }

    # 最佳 100 轮（滚动均值）
    for key in RUNS:
        best = best_rolling_mean(wide[key], "Train/mean_reward", 100)
        best["note_zh"] = (
            "以 100 个连续迭代为窗口对 Train/mean_reward 做滚动均值，取均值最大的窗口；"
            "阶段A 窗口数充足；阶段B 只有 125 个迭代，滚动窗口仅 "
            f"{best.get('windows', 0)} 个，且与起点重叠，仅作参考。"
        )
        summary["best_100_iteration_mean_reward"][key] = best

    # 阶段B 终值
    b = wide["B"]
    for tag in sorted(loaded["B"]["tags"]):
        if tag.endswith("/time"):
            continue
        summary["final_values_phase_B"][tag] = {
            "step": int(b["step"].iloc[-1]),
            "value": clean_float(b[tag].iloc[-1]),
        }

    # 终止率说明（base_too_low 全 0 等）
    for key in RUNS:
        if (wide[key]["Episode_Termination/base_too_low"].abs() < 1e-12).all():
            summary["notes_zh"].append(
                f"阶段{key}: Episode_Termination/base_too_low 全程恒为 0（该终止条件未触发）。"
            )
    if (wide["B"]["Episode_Reward/pitch"].abs() < 1e-12).all():
        summary["notes_zh"].append(
            "阶段B: Episode_Reward/pitch 全程恒为 0；params/env.yaml 显示续训时 pitch 奖励权重 "
            "由 -4.0 改为 -0.0，因此该曲线为零不是数据缺失。"
        )
    summary["notes_zh"].extend([
        ranges["step_continuity"]["note_zh"],
        "阶段B 事件流在 step %d 处直接结束（无 NaN/Inf），目录中最后一个 checkpoint 为 model_1100.pt，"
        "没有 model_1150.pt，因此属于未跑满/中断，而非数值发散。" % b_last,
        "续训时奖励配置由 v1.0.7 升级为 v1.0.8（见 config_changes_A_to_B）："
        "track_lin_vel_x 6→8、track_lin_vel_x_l2 -3→-6、track_ang_vel_z 1→3、"
        "track_ang_vel_z_l2 -1→-2、新增 L1 项（lin -2.0 / yaw -1.0）、action_jerk -0.5→-0.3、"
        "wheel_torques 死区 1.0→1.5、pitch -4.0→-0.0。",
        "由于奖励项权重在续训时改变，Episode_Reward/* 与 Train/mean_reward 的绝对值"
        "不能跨阶段直接比较，只能比较趋势；阶段B 新增的两个 L1 项只存在于阶段B。",
        "窗口定义：first_100=该阶段前100个迭代，last_100=后100个迭代，last_50=后50个迭代；"
        "阶段B 只有125个迭代，first_100 与 last_100 有 75 个迭代重叠，解释时需注意。",
    ])

    # ---------------- 写 run_ranges.json ----------------
    with open(OUT_DIR / "run_ranges.json", "w", encoding="utf-8") as f:
        json.dump(ranges, f, ensure_ascii=False, indent=2)
    print(f"[json] run_ranges.json")
    with open(OUT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[json] metrics_summary.json")

    # ---------------- 写 metrics_summary.md ----------------
    write_markdown(summary, ranges, wide)
    print(f"[md  ] metrics_summary.md")

    print("-" * 78)
    print("完成。")
    return 0


def write_markdown(summary, ranges, wide):
    A = summary["phases"]["A"]
    B = summary["phases"]["B"]
    L = []
    add = L.append

    add("# 训练指标汇总（阶段A 第1次训练 / 阶段B 续训）")
    add("")
    add("> 本文件由 `report/data/make_report_data.py` 从 TensorBoard 事件文件只读解析生成。")
    add("> 数值为原始标量（未平滑）；表中均值均为对应窗口内逐迭代值的算术平均。")
    add("")
    add("## 1. 训练规模")
    add("")
    add("| 项目 | 阶段A（第1次训练） | 阶段B（续训） |")
    add("|---|---|---|")
    add(f"| 运行目录 | `{ranges['runs']['A']['run_dir']}` | `{ranges['runs']['B']['run_dir']}` |")
    add(f"| 迭代 step 范围 | {A['first_step']} – {A['last_step']} | {B['first_step']} – {B['last_step']} |")
    add(f"| 迭代数（事件数） | {A['iterations']} | {B['iterations']} |")
    add(f"| 标量 tag 总数 / 按迭代 / 时间轴 tag | "
        f"{ranges['runs']['A']['tag_count_total']} / {ranges['runs']['A']['iteration_tag_count']} / "
        f"{ranges['runs']['A']['time_axis_tag_count']} | "
        f"{ranges['runs']['B']['tag_count_total']} / {ranges['runs']['B']['iteration_tag_count']} / "
        f"{ranges['runs']['B']['time_axis_tag_count']} |")
    add(f"| Episode_Reward 项数 | {ranges['runs']['A']['episode_reward_tag_count']} | "
        f"{ranges['runs']['B']['episode_reward_tag_count']} |")
    add(f"| 事件墙钟时长 | {A['wall_duration_hours']:.3f} h | {B['wall_duration_hours']:.3f} h |")
    add(f"| NaN / Inf 的 tag 数 | {len(ranges['runs']['A']['nan_inf_tags'])} | "
        f"{len(ranges['runs']['B']['nan_inf_tags'])} |")
    add("")
    add(f"**连续性**：{ranges['step_continuity']['note_zh']}")
    add("")
    add(f"**阶段B 结束情况**：事件流止于 step {B['last_step']}（无 NaN/Inf）；"
        f"目录内 checkpoint 为 model_1000/1050/1100，没有 model_1150，说明训练未跑满即结束（中断/手动停止），"
        f"不是数值发散。")
    add("")
    add("**奖励配置变化（v1.0.7 → v1.0.8，来自两个 run 的 `params/env.yaml` 差异）**：")
    add("")
    add("| 配置项 | 阶段A | 阶段B |")
    add("|---|---|---|")
    name_map = {
        "/rewards/track_lin_vel_x/weight": "track_lin_vel_x 权重",
        "/rewards/track_lin_vel_x/params/std": "track_lin_vel_x 核宽 std",
        "/rewards/track_lin_vel_x_l2/weight": "track_lin_vel_x_l2 权重",
        "/rewards/track_lin_vel_x_l1/weight": "track_lin_vel_x_l1 权重（新增）",
        "/rewards/track_ang_vel_z/weight": "track_ang_vel_z 权重",
        "/rewards/track_ang_vel_z/params/std": "track_ang_vel_z 核宽 std",
        "/rewards/track_ang_vel_z_l2/weight": "track_ang_vel_z_l2 权重",
        "/rewards/track_ang_vel_z_l1/weight": "track_ang_vel_z_l1 权重（新增）",
        "/rewards/action_jerk/weight": "action_jerk 权重",
        "/rewards/wheel_torques/params/threshold": "wheel_torques 死区阈值",
        "/rewards/pitch/weight": "pitch 权重",
    }
    for k, v in summary["config_changes_A_to_B"].items():
        if k not in name_map:
            continue
        va = "（无）" if v["A"] is None else str(v["A"])
        vb = "（无）" if v["B"] is None else str(v["B"])
        add(f"| {name_map[k]} | {va} | {vb} |")
    add("")
    add("由于奖励权重改变，**Episode_Reward/* 与 Train/mean_reward 的绝对值不能跨阶段直接比较**，"
        "只能比较同一阶段内的趋势；阶段B 的 L1 项为新增项，阶段A 无对应数据。")
    add("")

    # 窗口均值表
    add("## 2. 窗口均值（前100 / 后100 / 后50 迭代）")
    add("")
    add("窗口按各阶段内部迭代顺序切分：阶段A = step 0–99 / 900–999 / 950–999；"
        "阶段B = step 999–1098 / 1024–1123 / 1074–1123。")
    add("")
    for gi, (gname, tags) in enumerate(GROUPS, start=1):
        rows = [t for t in tags if t in summary["phases"]["A"]["metrics"] or t in summary["phases"]["B"]["metrics"]]
        if not rows:
            continue
        add(f"### 2.{gi} {gname}")
        add("")
        add("| 指标 | A 前100 | A 后100 | A 后50 | B 前100 | B 后100 | B 后50 |")
        add("|---|---|---|---|---|---|---|")
        for tag in rows:
            ma = summary["phases"]["A"]["metrics"].get(tag)
            mb = summary["phases"]["B"]["metrics"].get(tag)
            short = tag
            cells = []
            for phase_metrics in (ma, mb):
                if phase_metrics is None:
                    cells += ["—", "—", "—"]
                else:
                    cells += [fmt_value(tag, phase_metrics[k]) for k in ("first_100", "last_100", "last_50")]
            add(f"| `{short}` | " + " | ".join(cells) + " |")
        add("")

    # 阶段A 变化
    add("## 3. 阶段A 起点 vs 终点变化（前100均值 → 后100均值）")
    add("")
    add("| 指标 | 前100均值 | 后100均值 | 变化量 | 相对变化 |")
    add("|---|---|---|---|---|")
    for tag in SUMMARY_TAGS:
        if tag not in summary["phase_A_start_vs_end_delta"]:
            continue
        d = summary["phase_A_start_vs_end_delta"][tag]
        pct = "—" if d["pct_change"] is None else f"{d['pct_change']:+.1f}%"
        add(f"| `{tag}` | {fmt_value(tag, d['first_100_mean'])} | {fmt_value(tag, d['last_100_mean'])} | "
            f"{fmt_value(tag, d['delta'])} | {pct} |")
    add("")

    # 最佳百轮
    add("## 4. 最佳 100 迭代平均回报（滚动均值最大窗口）")
    add("")
    add("| 阶段 | 迭代数 | 100 轮窗口数 | 最佳100轮均值 | 窗口起始 step | 窗口结束 step | 窗口中心 step |")
    add("|---|---|---|---|---|---|---|")
    for key in ("A", "B"):
        best = summary["best_100_iteration_mean_reward"][key]
        add(f"| 阶段{key} | {summary['phases'][key]['iterations']} | {best['windows']} | "
            f"{best['mean']:.4f} | {best['start_step']} | {best['end_step']} | {best['center_step']} |")
    add("")
    best_a = summary["best_100_iteration_mean_reward"]["A"]["mean"]
    last_a = summary["phases"]["A"]["metrics"]["Train/mean_reward"]["last_100"]
    best_b = summary["best_100_iteration_mean_reward"]["B"]["mean"]
    last_b = summary["phases"]["B"]["metrics"]["Train/mean_reward"]["last_100"]
    add(
        f"阶段A：最佳 100 轮均值 {best_a:.4f}，末 100 轮均值 {last_a:.4f}，"
        f"差 {best_a - last_a:+.4f}；"
        f"阶段B：最佳 100 轮均值 {best_b:.4f}，末 100 轮均值 {last_b:.4f}，"
        f"差 {best_b - last_b:+.4f}（阶段B 仅 125 个迭代，窗口重叠，仅供参考）。"
    )
    add("")

    # 阶段B 终值
    add(f"## 5. 阶段B 最终值（step {B['last_step']}，也是全部数据的最后一条）")
    add("")
    add("| 指标 | 最终值 |")
    add("|---|---|")
    for tag in SUMMARY_TAGS:
        if tag in summary["final_values_phase_B"]:
            v = summary["final_values_phase_B"][tag]["value"]
            add(f"| `{tag}` | {fmt_value(tag, v)} |")
    add("")

    # 备注
    add("## 6. 数据说明 / 注意事项")
    add("")
    for note in summary["notes_zh"]:
        add(f"- {note}")
    add("")
    add("宽表 CSV 列名与 TensorBoard tag 完全一致（`run_A_scalars.csv` / `run_B_scalars.csv`），"
        "`step` 列为 TensorBoard step（= rs_rl 训练迭代号）。")
    add("")

    (OUT_DIR / "metrics_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
