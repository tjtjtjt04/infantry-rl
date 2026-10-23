# -*- coding: utf-8 -*-
"""
plot_report_figures.py
======================
读取 report/data/run_{A,B}_scalars.csv，生成报告所需的 9 张 PNG 图（200 dpi）到 report/figures/。

 01_training_reward.png       平均回报（原始 + EMA），两阶段 + 续训边界
 02_episode_length.png        平均回合长度 + 回合终止率
 03_tracking_rewards.png      速度跟踪奖励项（+ L1/L2 惩罚项）
 04_attitude_height.png       姿态 / 高度奖励项 2x2
 05_velocity_error.png        速度跟踪误差 2 panel（对数轴）
 06_ppo_losses.png            PPO 内部量 2x2（value / surrogate / entropy / policy std）
 07_termination.png           3 种终止率 + 回合长度叠加
 08_mechanical_smoothness.png 平滑 / 能耗 2x2
 09_performance.png           训练性能 3 panel（fps / collection / learning）

运行方式：
  D:\\miniconda\\envs\\RM\\python.exe report\\data\\plot_report_figures.py

约定：
- 阶段A 颜色 #1f77b4（蓝），阶段B 颜色 #d62728（红）。
- 阶段B 起点 step=999 用虚线竖线 + 浅红底色标出。
- 中文标签使用 C:\\Windows\\Fonts\\msyh.ttc（Microsoft YaHei）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator

BASE = Path(r"D:\Robomaster\RL")
DATA_DIR = BASE / "report" / "data"
FIG_DIR = BASE / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 字体 / 全局样式
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 13,
    "axes.titlesize": 15.5,
    "axes.labelsize": 13.5,
    "axes.titlepad": 8,
    "legend.fontsize": 11.5,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "figure.autolayout": False,
})

COL = {"A": "#1f77b4", "B": "#d62728"}
COL_SOFT = {"A": "#8fb8d8", "B": "#e79a9a"}
LAB = {"A": "阶段A（第1次训练）", "B": "阶段B（续训）"}
BOUNDARY = 999      # 阶段B 首条日志的 step（= 阶段A 最后一步）
B_END = 1123        # 阶段B 最后一条日志的 step
X_MAX = 1135

# 续训时奖励配置变化（来自两个 run 的 params/env.yaml 差异，见 run_ranges/metrics_summary）
REWARD_NOTE = (
    # "注：续训时奖励权重由 v1.0.7 升级到 v1.0.8：\n"
    # "track_lin_vel_x 权重 6→8（核宽 0.25→0.15）、新增 L1 跟踪惩罚（lin −2.0 / yaw −1.0）、\n"
    # "L2 惩罚加严（−3→−6 / −1→−2）、track_ang_vel_z 1→3、action_jerk −0.5→−0.3、\n"
    # "wheel_torques 死区 1.0→1.5、pitch −4.0→0。\n"
    # "因此两阶段的 Episode_Reward / 总回报绝对值不可直接比较，只能比较阶段内趋势。"
)


# ---------------------------------------------------------------- 工具
def load_phase() -> dict:
    out = {}
    for key in ("A", "B"):
        df = pd.read_csv(DATA_DIR / f"run_{key}_scalars.csv")
        out[key] = df
    return out


def ema(s: pd.Series, span: int = 20) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def add_boundary(ax, legend_label: str | None = None, y_frac: float = 0.5):
    """续训边界：虚线 + 阶段B 浅红底色。"""
    ax.axvspan(BOUNDARY, B_END, color=COL["B"], alpha=0.055, zorder=0, linewidth=0)
    line = ax.axvline(BOUNDARY, color="#444444", linestyle="--", linewidth=1.3, zorder=2)
    if legend_label:
        line.set_label(legend_label)
    return line


def style_axis(ax, xlabel: bool = False):
    ax.set_xlim(-10, X_MAX)
    if xlabel:
        ax.set_xlabel("训练迭代（TensorBoard step）")
    ax.grid(True, alpha=0.3)


def phase_handles(lw: float = 2.2):
    return [
        Line2D([0], [0], color=COL["A"], lw=lw, label=LAB["A"]),
        Line2D([0], [0], color=COL["B"], lw=lw, label=LAB["B"]),
    ]


def save(fig, name: str):
    out = FIG_DIR / name
    fig.tight_layout(pad=1.2)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)
    print(f"[fig ] {out.name}")


def win(df: pd.DataFrame, tag: str, n: int) -> float:
    return float(df[tag].iloc[-n:].mean())


# ---------------------------------------------------------------- 01 平均回报
def fig_reward(data):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    a, b = data["A"], data["B"]
    ax.plot(a["step"], a["Train/mean_reward"], color=COL["A"], lw=0.8, alpha=0.30,
            label=f"{LAB['A']} 原始值")
    ax.plot(a["step"], ema(a["Train/mean_reward"]), color=COL["A"], lw=2.2,
            label=f"{LAB['A']} EMA(span=20)")
    ax.plot(b["step"], b["Train/mean_reward"], color=COL["B"], lw=0.8, alpha=0.30,
            label=f"{LAB['B']} 原始值")
    ax.plot(b["step"], ema(b["Train/mean_reward"]), color=COL["B"], lw=2.2,
            label=f"{LAB['B']} EMA(span=20)")

    add_boundary(ax, legend_label="续训起点（第999轮）")

    # 阶段A 早期最低点
    i_min = int(a["Train/mean_reward"].idxmin())
    ax.annotate(
        f"第{int(a.loc[i_min, 'step'])}轮最低 {a.loc[i_min, 'Train/mean_reward']:.1f}\n（探索期大量摔倒）",
        xy=(int(a.loc[i_min, "step"]), float(a.loc[i_min, "Train/mean_reward"])),
        xytext=(70, -228), fontsize=11,
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1),
    )

    # 两阶段末值标注：放在坐标区内，避免压住标题
    ylim = (-275, 205)
    ax.set_ylim(*ylim)
    a_last = float(a["Train/mean_reward"].iloc[-1])
    b_last = float(b["Train/mean_reward"].iloc[-1])
    ax.scatter([BOUNDARY], [a_last], color=COL["A"], s=55, zorder=6)
    ax.scatter([B_END], [b_last], color=COL["B"], s=55, zorder=6)
    ax.annotate(f"阶段A 末值 {a_last:.1f}", xy=(BOUNDARY, a_last),
                xytext=(-8, -15), textcoords="offset points", ha="right", va="top",
                fontsize=11, color=COL["A"], fontweight="bold")
    ax.annotate(f"阶段B 末值 {b_last:.1f}", xy=(B_END, b_last),
                xytext=(0, 12), textcoords="offset points", ha="right", va="bottom",
                fontsize=11, color=COL["B"], fontweight="bold")

    ax.text(0.015, 0.975, REWARD_NOTE, transform=ax.transAxes, va="top", ha="left",
            fontsize=10.5, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#cccccc",
                      alpha=0.93))
    ax.set_ylabel("平均回合回报  Train/mean_reward")
    ax.set_title("训练平均回报（Train/mean_reward）")
    style_axis(ax, xlabel=True)
    ax.legend(loc="lower right", framealpha=0.95, ncol=2, fontsize=10.5)
    save(fig, "01_training_reward.png")


# ---------------------------------------------------------------- 02 回合长度 + 终止率
def fig_episode_length(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.6), sharex=True)

    for key in ("A", "B"):
        df = data[key]
        ax1.plot(df["step"], df["Train/mean_episode_length"], color=COL[key], lw=1.5,
                 label=LAB[key])
    ax1.axhline(1000, color="#888888", ls=":", lw=1.2)
    ax1.text(8, 1006, "回合上限 1000 步", fontsize=10.5, color="#666666")
    add_boundary(ax1)
    ax1.set_ylabel("平均回合长度（策略步）")
    ax1.set_title("平均回合长度（上）与回合终止率（下）")
    ax1.legend(loc="lower right", framealpha=0.95, fontsize=11)
    ax1.set_ylim(0, 1070)

    styles = {"time_out": "-", "bad_orientation": "--", "base_too_low": ":"}
    term_handles, term_labels = [], []
    for tag, ls in styles.items():
        for key in ("A", "B"):
            df = data[key]
            line, = ax2.plot(df["step"], df["Episode_Termination/" + tag], color=COL[key],
                             ls=ls, lw=1.4, alpha=0.95,
                             label=f"{LAB[key]}：{tag}")
            term_handles.append(line)
            term_labels.append(f"{LAB['A'] if key == 'A' else LAB['B']}：{tag}")
    ax2.annotate("阶段B 第999轮为首条日志\n（回合统计尚未积累，数值偏低）",
                 xy=(BOUNDARY, 0.02), xytext=(640, 0.38),
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0), fontsize=10.5)
    add_boundary(ax2)
    ax2.set_ylabel("终止原因占比（每回合比例）")
    ax2.set_xlabel("训练迭代（TensorBoard step）")
    ax2.set_ylim(-0.03, 1.06)
    # 终止原因图例放到图外底部（x 轴标签下方），避免遮挡曲线或坐标轴标签
    fig.legend(term_handles, term_labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.10), fontsize=10.5, framealpha=0.95)
    style_axis(ax2, xlabel=True)
    ax1.set_xlim(-10, X_MAX)
    save(fig, "02_episode_length.png")


# ---------------------------------------------------------------- 03 跟踪奖励项
def fig_tracking_rewards(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.8), sharex=True)

    def panel(ax, main, l2, l1, title, main_anchor=(1.0, 0.50)):
        ax2t = ax.twinx()
        main_handles, main_labels = [], []
        pen_handles, pen_labels = [], []
        for key in ("A", "B"):
            df = data[key]
            line, = ax.plot(df["step"], df[main], color=COL[key], lw=1.8)
            main_handles.append(line)
            main_labels.append(f"{LAB[key]}：{main.split('/')[-1]} 奖励（左轴）")
            line2, = ax2t.plot(df["step"], df[l2], color=COL[key], lw=1.2, ls="--", alpha=0.85)
            pen_handles.append(line2)
            pen_labels.append(f"{LAB[key]}：{l2.split('/')[-1]} 惩罚（右轴）")
            if l1 in df.columns:
                line3, = ax2t.plot(df["step"], df[l1], color=COL[key], lw=1.4, ls=":", alpha=0.95)
                pen_handles.append(line3)
                pen_labels.append(f"{LAB[key]}：{l1.split('/')[-1]} 惩罚（新增，右轴）")
        ax.axvspan(BOUNDARY, B_END, color=COL["B"], alpha=0.055, zorder=0, linewidth=0)
        ax.axvline(BOUNDARY, color="#444444", linestyle="--", lw=1.3)
        ax.set_ylabel("奖励项（加权后）")
        ax2t.set_ylabel("L1/L2 惩罚项（负值）")
        ax2t.grid(False)
        ax.legend(main_handles, main_labels, loc="center right", bbox_to_anchor=main_anchor,
                  fontsize=10.5, framealpha=0.95)
        ax2t.legend(pen_handles, pen_labels, loc="lower right", fontsize=10.0, framealpha=0.95)
        ax.set_title(title)
        style_axis(ax)

    panel(ax1, "Episode_Reward/track_lin_vel_x", "Episode_Reward/track_lin_vel_x_l2",
          "Episode_Reward/track_lin_vel_x_l1",
          "前进速度跟踪 track_lin_vel_x（左轴：奖励；右轴：L1/L2 惩罚，负值）",
          main_anchor=(1.0, 0.46))
    panel(ax2, "Episode_Reward/track_ang_vel_z", "Episode_Reward/track_ang_vel_z_l2",
          "Episode_Reward/track_ang_vel_z_l1",
          "航向角速度跟踪 track_ang_vel_z（左轴：奖励；右轴：L1/L2 惩罚，负值）",
          main_anchor=(1.0, 0.66))
    ax2.set_xlabel("训练迭代（TensorBoard step）")
    style_axis(ax2, xlabel=True)
    fig.text(0.5, 0.005,
             "说明：L1/L2 惩罚为 v1.0.8 新增/加严项，阶段A 无 L1 数据；"
             "奖励绝对值受权重影响，跨阶段只比较趋势。",
             ha="center", fontsize=10.5, color="#444444")
    save(fig, "03_tracking_rewards.png")


# ---------------------------------------------------------------- 04 姿态 / 高度
def fig_attitude_height(data):
    terms = ["base_height", "pitch", "roll", "roll_straight"]
    zh = {
        "base_height": "高度保持 base_height",
        "pitch": "俯仰惩罚 pitch",
        "roll": "横滚惩罚 roll",
        "roll_straight": "直线横滚惩罚 roll_straight",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    for ax, term in zip(axes.ravel(), terms):
        tag = "Episode_Reward/" + term
        for key in ("A", "B"):
            df = data[key]
            ax.plot(df["step"], df[tag], color=COL[key], lw=1.2, label=LAB[key])
        add_boundary(ax)
        ax.set_title(zh[term])
        ax.set_ylabel("奖励项（加权后）")
        ax.tick_params(axis="y", labelsize=10.5)
        if term == "pitch":
            ax.annotate("阶段B pitch 权重为 0\n（曲线恒为 0，非数据缺失）",
                        xy=(B_END, 0), xytext=(430, -0.75),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
                        fontsize=10.5,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="#cccccc", alpha=0.9))
        if term == "base_height":
            ax.legend(loc="center right", fontsize=10.5)
    for ax in axes[-1]:
        ax.set_xlabel("训练迭代（TensorBoard step）")
        style_axis(ax, xlabel=True)
    for ax in axes[0]:
        style_axis(ax)
    save(fig, "04_attitude_height.png")


# ---------------------------------------------------------------- 05 速度误差
def fig_velocity_error(data):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    specs = [
        ("Metrics/base_velocity/error_vel_xy", "线速度误差  error_vel_xy（m/s）", "越低越好"),
        ("Metrics/base_velocity/error_vel_yaw", "偏航角速度误差  error_vel_yaw（rad/s）", "越低越好"),
    ]
    for ax, (tag, ylabel, hint) in zip(axes, specs):
        for key in ("A", "B"):
            df = data[key]
            ax.plot(df["step"], df[tag], color=COL[key], lw=1.2, label=LAB[key])
        add_boundary(ax)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel + "\n（对数轴）")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1)))
        # 末100轮均值标注（放在曲线上方空白处）
        a_m = win(data["A"], tag, 100)
        b_m = win(data["B"], tag, 100)
        ax.annotate(f"A 末100轮均值 {a_m:.4f}", xy=(999, a_m), xytext=(520, a_m * 2.4),
                    fontsize=10.5, color=COL["A"],
                    arrowprops=dict(arrowstyle="->", color=COL["A"], lw=0.9, alpha=0.7))
        ax.annotate(f"B 末100轮均值 {b_m:.4f}", xy=(B_END, b_m), xytext=(700, b_m * 0.38),
                    fontsize=10.5, color=COL["B"],
                    arrowprops=dict(arrowstyle="->", color=COL["B"], lw=0.9, alpha=0.7))
        ax.legend(loc="center left", fontsize=10.5, framealpha=0.95)
        style_axis(ax)
    axes[0].set_title("基座速度跟踪误差（越低越好；纵轴为对数坐标）")
    axes[-1].set_xlabel("训练迭代（TensorBoard step）")
    style_axis(axes[-1], xlabel=True)
    save(fig, "05_velocity_error.png")


# ---------------------------------------------------------------- 06 PPO
def fig_ppo_losses(data):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True)
    specs = [
        ("Loss/value", "价值损失 Loss/value", "value loss"),
        ("Loss/surrogate", "策略替代损失 Loss/surrogate", "surrogate"),
        ("Loss/entropy", "策略熵项 Loss/entropy", "entropy"),
        ("Policy/mean_std", "策略标准差 Policy/mean_std", "std"),
    ]
    for ax, (tag, title, _) in zip(axes.ravel(), specs):
        for key in ("A", "B"):
            df = data[key]
            ax.plot(df["step"], df[tag], color=COL[key], lw=1.0, alpha=0.75,
                    label=LAB[key])
            ax.plot(df["step"], ema(df[tag], span=20), color=COL[key], lw=2.0, alpha=0.95)
        add_boundary(ax)
        ax.set_title(title)
        ax.set_ylabel("数值")
        ax.tick_params(axis="y", labelsize=10.5)
        if tag == "Policy/mean_std":
            ax.axhline(0.06, color="#888888", ls="--", lw=1.0)
            ax.axhline(0.40, color="#888888", ls="--", lw=1.0)
            ax.text(8, 0.032, "灰色虚线：策略 std 限幅 [0.06, 0.4]（v1.0.7 起生效）",
                    fontsize=10, color="#555555")
            ax.set_ylim(0.02, 0.46)
            ax.legend(loc="upper right", fontsize=10.5)
        if tag == "Loss/entropy":
            ax.text(180, 0.7, "熵项随策略 std 下降由正转负\n（6 维高斯策略的微分熵）",
                    fontsize=10.5, color="#444444",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#dddddd", alpha=0.9))
    for ax in axes[-1]:
        ax.set_xlabel("训练迭代（TensorBoard step）")
        style_axis(ax, xlabel=True)
    for ax in axes[0]:
        style_axis(ax)
    fig.text(0.5, 0.002, "细线=每轮原始值，粗线=EMA(span=20)",
             ha="center", fontsize=10.5, color="#444444")
    save(fig, "06_ppo_losses.png")


# ---------------------------------------------------------------- 07 终止原因
def fig_termination(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True)
    styles = {"time_out": "-", "bad_orientation": "--", "base_too_low": ":"}
    zh = {"time_out": "超时 time_out", "bad_orientation": "姿态失控 bad_orientation",
          "base_too_low": "基座过低 base_too_low"}
    for tag, ls in styles.items():
        for key in ("A", "B"):
            df = data[key]
            ax1.plot(df["step"], df["Episode_Termination/" + tag], color=COL[key], ls=ls,
                     lw=1.5, alpha=0.95, label=f"{LAB[key]}：{zh[tag]}")
    add_boundary(ax1)
    ax1.set_ylabel("终止原因占比")
    ax1.set_title("回合终止原因（上）与平均回合长度（下）")
    ax1.set_ylim(-0.03, 1.05)
    ax1.legend(loc="center right", fontsize=9.6, ncol=1, framealpha=0.95)
    # 动态定位阶段A 姿态失控率峰值
    bo = data["A"]["Episode_Termination/bad_orientation"]
    i_peak = int(bo.idxmax())
    peak_step = int(data["A"].loc[i_peak, "step"])
    peak_val = float(bo.iloc[i_peak])
    ax1.annotate(
        f"阶段A 第{peak_step}轮姿态失控率最高 {peak_val:.2f}，\n"
        f"训练后期降至约 {win(data['A'], 'Episode_Termination/bad_orientation', 100):.2f}；"
        f"阶段B 末100轮均值 {win(data['B'], 'Episode_Termination/bad_orientation', 100):.3f}",
        xy=(peak_step, peak_val), xytext=(120, 0.60), fontsize=10.5,
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))

    for key in ("A", "B"):
        df = data[key]
        ax2.plot(df["step"], df["Train/mean_episode_length"], color=COL[key], lw=1.6,
                 label=LAB[key])
    add_boundary(ax2)
    ax2.set_ylabel("平均回合长度（策略步）")
    ax2.set_xlabel("训练迭代（TensorBoard step）")
    ax2.set_ylim(0, 1060)
    ax2.legend(loc="lower left", fontsize=10.5, framealpha=0.95)
    ax2.annotate("续训首条日志（第999轮）\n数值偏低", xy=(BOUNDARY, 28), xytext=(700, 250),
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0), fontsize=10.5)
    style_axis(ax1)
    style_axis(ax2, xlabel=True)
    save(fig, "07_termination.png")


# ---------------------------------------------------------------- 08 机械平滑 / 能耗
def fig_mechanical(data):
    specs = [
        ("action_rate", "动作变化率惩罚 action_rate", "续训前后权重未变（−4.0e-3）"),
        ("action_jerk", "动作二阶差分惩罚 action_jerk", "权重 −0.5→−0.3（绝对值不可比）"),
        ("wheel_torques", "轮扭矩惩罚 wheel_torques", "死区阈值 1.0→1.5 N·m（不可比）"),
        ("wheel_axle", "轮轴惩罚 wheel_axle", "权重续训前后未变"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    for ax, (term, title, note) in zip(axes.ravel(), specs):
        tag = "Episode_Reward/" + term
        for key in ("A", "B"):
            df = data[key]
            ax.plot(df["step"], df[tag], color=COL[key], lw=1.2, label=LAB[key])
        add_boundary(ax)
        ax.set_title(title)
        ax.set_ylabel("奖励项（加权后）")
        ax.tick_params(axis="y", labelsize=10.5)
        ax.text(0.02, 0.04, note, transform=ax.transAxes, fontsize=10, color="#444444",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#dddddd", alpha=0.85))
    axes[0, 0].legend(loc="lower right", fontsize=10.5)
    for ax in axes[-1]:
        ax.set_xlabel("训练迭代（TensorBoard step）")
        style_axis(ax, xlabel=True)
    for ax in axes[0]:
        style_axis(ax)
    save(fig, "08_mechanical_smoothness.png")


# ---------------------------------------------------------------- 09 训练性能
def fig_performance(data):
    specs = [
        ("Perf/total_fps", "采样吞吐  Perf/total_fps（环境步/秒）", False),
        ("Perf/collection_time", "采样耗时  Perf/collection_time（秒/轮）", False),
        ("Perf/learning_time", "学习耗时  Perf/learning_time（秒/轮）", False),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.0), sharex=True)
    perf_handles, perf_labels = [], []
    for ax, (tag, ylabel, _) in zip(axes, specs):
        for key in ("A", "B"):
            df = data[key]
            line_raw, = ax.plot(df["step"], df[tag], color=COL[key], lw=1.0, alpha=0.55,
                                label=f"{LAB[key]} 原始值")
            line_ema, = ax.plot(df["step"], ema(df[tag], span=20), color=COL[key], lw=2.1,
                                label=f"{LAB[key]} EMA(span=20)")
            if key == "A" and tag == specs[0][0]:
                perf_handles += [line_raw, line_ema]
                perf_labels += [f"{LAB['A']} 原始值", f"{LAB['A']} EMA(span=20)"]
            if key == "B" and tag == specs[0][0]:
                perf_handles += [line_raw, line_ema]
                perf_labels += [f"{LAB['B']} 原始值", f"{LAB['B']} EMA(span=20)"]
        add_boundary(ax)
        ax.set_ylabel(ylabel, fontsize=12.5)
        ax.tick_params(axis="y", labelsize=10.5)
    axes[0].set_title("训练性能：采样吞吐、采样耗时与学习耗时")
    # 图例放到图外底部，避免遮挡数据
    fig.legend(perf_handles, perf_labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.035), fontsize=10.5, framealpha=0.95)
    axes[-1].set_xlabel("训练迭代（TensorBoard step）")
    style_axis(axes[-1], xlabel=True)
    for ax in axes[:-1]:
        style_axis(ax)
    save(fig, "09_performance.png")


# ---------------------------------------------------------------- main
def main():
    data = load_phase()
    print(f"[data] A={len(data['A'])} rows, B={len(data['B'])} rows")
    fig_reward(data)
    fig_episode_length(data)
    fig_tracking_rewards(data)
    fig_attitude_height(data)
    fig_velocity_error(data)
    fig_ppo_losses(data)
    fig_termination(data)
    fig_mechanical(data)
    fig_performance(data)
    print("全部 9 张图已生成 ->", FIG_DIR)


if __name__ == "__main__":
    main()
