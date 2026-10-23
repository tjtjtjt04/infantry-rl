# -*- coding: utf-8 -*-
"""生成《项目进展汇报材料 —— 训练曲线与关键数据讲解》DOCX。

用途：组会 / 导师进展汇报时逐图讲解的“讲稿文档”。
数据来源：report/figures 下 9 张训练曲线（由两个 run 的 TensorBoard 事件文件生成）；
数字来源：report/data/metrics_summary.md。

运行：<bundled python> report/build_progress_brief.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_report import (  # noqa: E402
    BLUE,
    FIG,
    FONT_HEI,
    FONT_SONG,
    GRAY,
    Report,
    set_run_font,
    setup_section,
)

ROOT = Path(__file__).resolve().parent
OUT_DOCX = ROOT / "进展汇报_训练曲线与关键数据讲解_2026-10-22.docx"


def title_block(r: Report):
    doc = r.doc
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    rr = p.add_run("项目进展汇报材料")
    set_run_font(rr, size=22, bold=True, east=FONT_HEI, color=BLUE)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    rr = p.add_run("训练曲线与关键数据讲解")
    set_run_font(rr, size=18, bold=True, east=FONT_HEI, color=BLUE)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = p.add_run("轮腿式 RoboMaster 步兵底盘 · 强化学习运动控制（Isaac Lab + PPO）")
    set_run_font(rr, size=12, east=FONT_HEI, color=GRAY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = p.add_run("2026 年 10 月 22 日  ·  数据来源：两段连续训练（阶段 A / 阶段 B）")
    set_run_font(rr, size=10, east=FONT_SONG, color=GRAY)

    r.note("本材料用于组会 / 导师进展汇报：每张图给出“看什么 → 怎么讲 → 关键数字 → 别误读”，"
           "数字均可追溯到两个 run 的 TensorBoard 事件文件；完整技术报告见 report/ 目录。")


def section_mainline(r: Report):
    r.h1("1  汇报主线与建议结构")
    r.p("整场汇报用三句话串起来，其余内容都是给这三句话提供证据：")
    r.bullets([
        [("① 做了什么：", True),
         ("对轮腿式步兵底盘完成两段连续 PPO 训练（阶段 A 从零训练 1000 轮；阶段 B 升级精度奖励后从检查点续训 124 轮）。", False)],
        [("② 结果如何：", True),
         ("策略已能稳定站立、按 0.2–1.2 m/s 前进速度行走；回报从 −135 升到 +116，回合长度 890 → 985/1000 步，"
          "真实速度误差约 0.069 m/s，姿态保持在 1° 以内。", False)],
        [("③ 卡在哪 / 下一步：", True),
         ("精度距目标 0.025 m/s 还差约 2.5 倍；阶段 B 训练量不足且中断；转弯/倒车/地形与真机迁移尚未开始。", False)],
    ])

    r.table(
        ["页序", "内容", "建议时长", "主要素材"],
        [
            ["1", "一句话进展 + 汇报路线", "1 min", "图 1（回报曲线）"],
            ["2", "目标与训练对象（机器人 / 任务 / 接口 / 不包含什么）", "2 min", "指标口径"],
            ["3", "训练策略（PPO、奖励设计、两段式课程）", "2–3 min", "奖励函数表"],
            ["4", "训练过程（关键节点）", "4–5 min", "图 1 / 2 / 7"],
            ["5", "结果（能站、能走、走得准不准）", "3–4 min", "图 3 / 4 / 5 + 数字表"],
            ["6", "问题与风险", "2 min", "图 6 / 9"],
            ["7", "下一步计划 + 想请老师定的点", "2 min", "计划表"],
        ],
        widths=[1.2, 8.0, 2.2, 4.4],
        caption="建议汇报结构（约 10–15 分钟）",
        align_first_left=True,
    )
    r.note("提示：完整技术报告（report/Infantry-RL_…docx，30 页）作为附录/备份，汇报只取上面 7 页；"
           "老师追问细节再翻报告。")


def insert_figure(r: Report, filename: str, width_cm: float = 15.2):
    """只插图、不加图题（图题用上面的 h2 标题，避免重复）。"""
    par = r.doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = par.paragraph_format
    pf.space_before = Pt(4)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.0
    run = par.add_run()
    run.add_picture(str(FIG / filename), width=Cm(width_cm))


def explain_figure(r: Report, filename: str, caption: str, look: str, talk: str,
                   numbers: str, warning: str, width_cm: float = 15.2):
    r.h2(caption)
    insert_figure(r, filename, width_cm=width_cm)
    r.rich([("看什么：", True), (look, False)])
    r.rich([("怎么讲：", True), (talk, False)])
    r.rich([("关键数字：", True), (numbers, False)])
    r.rich([("别误读：", True), (warning, False)])


def section_figures(r: Report):
    r.h1("2  九张图逐张讲解")

    explain_figure(
        r, "01_training_reward.png", "图 1  平均回报：先摔后立，阶段 A 基本收敛",
        "横轴是训练迭代；蓝色为阶段 A、红色为阶段 B；粗线是 EMA 平滑，细线是原始值；"
        "虚线是续训起点第 999 轮，浅红底色表示阶段 B；图内文字注明续训时奖励权重重定义。",
        "第 0–22 轮策略接近随机，大量摔倒，回报探底到 −249；前 100 轮均值 −135.4，之后持续上升，"
        "约 700 轮后进入平台，末 100 轮均值 115.8（最佳 100 轮 116.6，只差 0.78，说明阶段 A 已基本收敛）。"
        "阶段 B 首条日志 0.44 是续训瞬态，约 20 轮后回到稳态，末 100 轮均值 161.1。",
        "−249（第 22 轮最低）→ 前 100 轮 −135.4 → 末 100 轮 +115.8；峰值 119.7（第 941 轮）；阶段 B 末 100 轮 +161.1。",
        "两段奖励权重不同（v1.0.7 → v1.0.8，9 处变化），蓝色和红色的绝对高度不能直接比较，只能看各自阶段内的趋势。",
    )

    explain_figure(
        r, "02_episode_length.png", "图 2  回合长度与终止率：很快学会“不摔”",
        "上图是平均回合长度（上限 1000 步 = 20 s），下图是三种终止原因的占比（超时 / 姿态失控 / 高度过低）。",
        "约 25 轮后回合长度就接近 1000 步，末 100 轮 984.7 步、超时率 97.5%；中段第 400–600 轮回合长度"
        "跌到约 760 步，对应下方姿态失控率峰值 0.31，随后恢复；高度过低终止全程为 0（从未“塌下去”）。",
        "回合长度 890.6 → 984.7/1000；超时率 89.6% → 97.5%；姿态失控率 0.06% → 2.5%（中段峰值 31%）；"
        "高度过低终止 0。",
        "回合长度接近满值 ≠ 学会走——前 450 轮“站着不动”也能跑满回合；这张图只能说明“不倒”，不能说明“会走”。",
    )

    explain_figure(
        r, "05_velocity_error.png", "图 3  速度误差：第 470–570 轮出现“台阶式”下降",
        "上图是线速度跟踪误差，下图是偏航角速度误差，纵轴为对数坐标（越低越好）；虚线仍是续训起点。",
        "误差在第 470–570 轮之间出现台阶式下降（约 1.15 → 0.15），这就是“从站着不动到真正走起来”的直接证据；"
        "之后继续缓慢下降，阶段 A 末 100 轮为 0.1386 / 0.3211，阶段 B 末 100 轮为 0.1306 / 0.3081。"
        "把日志值除以 2 得到真实平均误差：约 0.069 → 0.065 m/s；目标 0.025 m/s，还差约 2.5 倍。",
        "阶段 A：1.147 → 0.1386（线速度）、1.891 → 0.3211（yaw）；阶段 B 末 100 轮：0.1306 / 0.3081，"
        "最终 0.1221 / 0.3061。",
        "阶段 B 第 999 轮出现极小值（0.018 / 0.012），是续训瞬态，不能当成绩；对数轴会让“看起来变化不大”"
        "的曲线实际下降了一个数量级。",
    )

    explain_figure(
        r, "03_tracking_rewards.png", "图 4  跟踪奖励项：跟踪奖励确实起来了",
        "上图是前进速度跟踪奖励，下图是航向角速度跟踪奖励；左轴是奖励值（越大越好），右轴是 L1/L2 惩罚项（负值，越小越好）。",
        "阶段 A 前 450 轮前进跟踪奖励只在 1–3/6 徘徊（说明策略选择“站着不动”），450 轮后快速抬升，"
        "末 100 轮达到 5.56/6；阶段 B 末 100 轮为 7.27/8。L2 惩罚从早期的 −1.37 / −1.69 收敛到 −0.05 量级；"
        "阶段 B 新增的两条 L1 惩罚为 −0.082 / −0.153。",
        "阶段 A：track_lin 0.86 → 5.56 / 6；track_ang 0.20 → 0.77 / 1；阶段 B 末 100 轮：7.27 / 8、1.44 / 3。",
        "跟踪核的宽度（std）和权重都变了，奖励项数值不能跨阶段比较；要比较精度请看速度误差图（图 3）。",
    )

    explain_figure(
        r, "04_attitude_height.png", "图 5  姿态与高度：俯仰、高度在阶段 A 内稳住",
        "2×2 四个面板：高度项 base_height、俯仰项 pitch、横滚项 roll、直行横滚项 roll_straight，均为加权后的每步奖励值。",
        "阶段 A 的俯仰惩罚从 −2.35 收敛到 −0.068（反推平均 |pitch|≈0.068/4≈1.0°），高度项从 −0.093 到 "
        "−0.029（偏差约 7 mm），横滚项从 −0.036 到 −0.013（约 0.5°）——俯仰、高度、姿态在阶段 A 内学会稳住。",
        "阶段 A 末 100 轮：base_height −0.0287、pitch −0.0678、roll −0.0135、roll_straight −0.0205。",
        "阶段 B 的 pitch 曲线恒为 0：因为续训时把 pitch 奖励权重改成了 0（疑似误置，VERSION.md 仍记 −4.0），"
        "不是姿态更好，也不是数据缺失；阶段 B 的 roll 惩罚略大是走得更快带来的姿态代价。",
    )

    explain_figure(
        r, "07_termination.png", "图 6  终止原因：唯一的非超时终止是姿态失控",
        "上图三种终止原因占比（超时 / 姿态失控 / 高度过低），下图平均回合长度（与图 2 对照）。",
        "阶段 A 第 400–600 轮姿态失控率抬升到峰值 0.31，是“从站着不动转向走”的探索代价，之后回落到末 100 轮 0.025；"
        "阶段 B 稳定在 0.031 左右。高度过低终止全程 0，说明没有“坍塌式”失败；超时率与回合长度互补。",
        "姿态失控率峰值 31.17%（第 547 轮）；阶段 A 末 100 轮 2.5%、阶段 B 末 100 轮 3.1%。",
        "中段失稳是探索行为，不是训练发散——同一时期价值损失同时在跳升，之后都恢复了；汇报时不要只说“摔得多”，"
        "要说明这是学会走之前的代价。",
    )

    explain_figure(
        r, "08_mechanical_smoothness.png", "图 7  机械平滑性：动作抖动与歪腿都明显改善",
        "2×2 四个面板：一阶动作差分 action_rate、二阶差分 action_jerk、轮力矩 wheel_torques、轮轴歪斜 wheel_axle。",
        "action_jerk 从 −1.256 降到 −0.114（阶段 A，约 91% 抖动被消除），wheel_axle 从 −0.213 到 −0.097"
        "（左右轮轴歪斜减轻），action_rate 一直很小。说明策略动作平滑、腿不再高频抖动、轮子没有变成主要平衡执行器。",
        "阶段 A 末 100 轮：action_rate −0.00038、action_jerk −0.114、wheel_torques −0.167、wheel_axle −0.097。",
        "阶段 B 的 action_jerk 绝对值变小，部分原因是权重从 −0.5 放宽到了 −0.3；wheel_torques 死区也从 1.0 改到 1.5 N·m"
        "——这些项不能跨阶段直接比绝对值，要按权重反推原始量（A/B 的原始抖动水平接近）。",
    )

    explain_figure(
        r, "06_ppo_losses.png", "图 8  PPO 内部量：训练本身是健康的",
        "2×2 四个面板：价值损失、策略替代损失、策略熵、策略标准差；细线原始值、粗线平滑。",
        "价值损失从 3.25 降到 0.37（critic 跟上了回报）；策略标准差从 0.285 收敛到 0.074，已经贴近 0.06 的硬下限"
        "（探索空间接近最小）；熵由正转负只是 6 维高斯分布 std 收缩的数学结果，不是训练坏了。"
        "阶段 B 换奖励的瞬间价值损失从 0.37 跳到 6.4，再回落到 0.46——这是价值函数重新适配的代价。",
        "阶段 A：value loss 3.253 → 0.368、std 0.285 → 0.074；阶段 B 末 100 轮：value 0.464、std 0.0726。",
        "std 贴近下限意味着“不再自己探索精度”；想继续提升精度不能只靠接着训，需要调奖励/课程（或放宽探索）。",
    )

    explain_figure(
        r, "09_performance.png", "图 9  训练性能：长训练在硬件上越跑越慢",
        "三个面板：总吞吐（样本/秒）、采样耗时（秒/轮）、学习耗时（秒/轮）。",
        "阶段 A 的 fps 从约 2.8 万降到 1.6 万，采样耗时从 1.6 s/轮升到 3.1 s/轮；学习耗时稳定在 0.15–0.3 s/轮。"
        "日志没有记录变慢原因，本材料只如实呈现该现象，不做因果解释。",
        "fps 28076 → 16038；collection_time 1.604 → 3.130 s/轮；learning_time 0.154 → 0.278 s/轮。",
        "这不是算法问题，而是硬件/环境现象；汇报时用它解释“为什么 1000 轮要 41 分钟、为什么长训练容易崩”，"
        "不要展开成技术细节。",
    )


def section_numbers(r: Report):
    r.h1("3  关键数字速查表（可直接背）")
    r.table(
        ["指标", "阶段 A 前 100 轮", "阶段 A 末 100 轮", "阶段 B 末 100 轮", "说明"],
        [
            ["平均回报 mean_reward", "−135.44", "115.82", "161.10", "阶段 B 奖励权重不同，不可直接比"],
            ["平均回合长度（步）", "890.60", "984.65", "985.13", "满值 1000 步（20 s）"],
            ["track_lin_vel_x 奖励", "0.860", "5.558", "7.271", "权重 6 → 8，核宽 0.25 → 0.15"],
            ["track_ang_vel_z 奖励", "0.202", "0.767", "1.435", "权重 1 → 3，核宽 0.35 → 0.15"],
            ["error_vel_xy（日志值）", "1.147", "0.1386", "0.1306", "≈ 真实平均误差 × 2"],
            ["→ 真实线速度误差（m/s）", "0.574", "0.0693", "0.0653", "日志值 ÷ 2"],
            ["error_vel_yaw（日志值）", "1.891", "0.3211", "0.3081", "≈ 真实平均误差 × 2"],
            ["→ 真实 yaw 误差（rad/s）", "0.946", "0.1606", "0.1541", "日志值 ÷ 2"],
            ["超时率 time_out", "0.896", "0.975", "0.969", "越高越稳"],
            ["姿态失控率 bad_orientation", "0.0006", "0.0250", "0.0309", "中段峰值 0.312（第 547 轮）"],
            ["高度过低终止 base_too_low", "0", "0", "0", "全程未触发"],
            ["策略标准差 Policy/mean_std", "0.285", "0.0742", "0.0726", "硬限幅 [0.06, 0.4]"],
            ["价值损失 Loss/value", "3.253", "0.368", "0.464", "阶段 B 换奖励时曾到 6.4"],
            ["吞吐 total_fps", "28 076", "16 038", "18 426", "样本/秒"],
        ],
        widths=[4.0, 2.7, 2.7, 2.7, 3.7],
        caption="关键指标速查表（窗口均值）",
        align_first_left=True,
        font_size=8.5,
        header_size=8.5,
        highlight_rows=[5, 7],
    )
    r.note("注：error_vel_* 是 Isaac Lab 命令管理器按回合累积的日志量，数值约为真实平均误差的 2 倍；"
           "真实误差以“日志值 ÷ 2”估算。阶段 B 只有 124 个迭代，其“末 100 轮”窗口与“前 100 轮”大量重叠，解释时注意。")


def section_rewards(r: Report):
    r.h1("4  奖励函数速查（老师追问时用）")
    r.p("奖励 = 每步 Σ(权重 × 奖励项)，Isaac Lab 再乘 dt=0.02 s 累加进 PPO 回报；共 24 项"
        "（阶段 A 为 22 项）。配置在 locomotion_env_cfg.py 的 LocomotionRewardsCfg，公式实现在 mdp/rewards.py。"
        "两阶段完整权重如下（标 ★ 为阶段 B 新增/变化较大的项）：")
    r.table(
        ["奖励项", "公式（简写）", "阶段 A 权重", "阶段 B 权重", "作用"],
        [
            ["alive", "1（存活）", "+0.5", "+0.5", "每步基础奖励"],
            ["fall_penalty", "1[摔倒终止]", "−50", "−50", "摔倒重罚"],
            ["base_height", "|h − 0.189775|", "−4.0", "−4.0", "保持站立高度"],
            ["pitch", "|pitch|", "−4.0", "−0.0", "俯仰归零（B 实际失效）"],
            ["roll", "|roll|", "−1.5", "−1.5", "侧倾基础约束"],
            ["roll_straight", "|roll|·(1−min(|wz*|/2.5,1))", "−2.5", "−2.5", "直行时重罚侧倾"],
            ["roll_rate", "ωx²", "−0.1", "−0.1", "侧倾速率阻尼"],
            ["lin_vel_z", "vz²", "−2.0", "−2.0", "抑制上下弹跳"],
            ["track_lin_vel_x", "exp(−(vx−vx*)²/std²)", "+6.0 (0.25)", "+8.0 (0.15)", "前进速度跟踪（主项）"],
            ["track_ang_vel_z", "exp(−(ωz−wz*)²/std²)", "+1.0 (0.35)", "+3.0 (0.15)", "航向角速度跟踪"],
            ["heading_error", "wrap(ψ*−ψ)²", "−0.6", "−0.6", "真正到达目标航向"],
            ["track_lin_vel_x_l1 ★", "|vx−vx*|", "—", "−2.0", "恒定梯度压最后几 cm/s"],
            ["track_ang_vel_z_l1 ★", "|ωz−wz*|", "—", "−1.0", "yaw 恒定梯度"],
            ["track_lin_vel_x_l2", "(vx−vx*)²", "−3.0", "−6.0", "大误差主导"],
            ["track_ang_vel_z_l2", "(ωz−wz*)²", "−1.0", "−2.0", "yaw 抖动抑制"],
            ["lin_vel_y", "vy²", "−0.5", "−0.5", "禁止侧滑"],
            ["wheel_axle", "|L0_l·sinφ_l − L0_r·sinφ_r|", "−8.0", "−8.0", "左右轮轴对齐"],
            ["wheel_torques", "Σ max(|τ_wheel|−τd, 0)²", "−0.05 (τd=1.0)", "−0.05 (τd=1.5)", "只罚轮力矩尖峰"],
            ["action_rate", "Σ(a_t−a_{t−1})²", "−4e-3", "−4e-3", "动作平滑"],
            ["action_jerk", "Σ(a_t−2a_{t−1}+a_{t−2})²", "−0.5", "−0.3", "只杀高频抖动"],
            ["joint_torques", "Στ²", "−2e-5", "−2e-5", "关节力矩轻罚"],
            ["joint_acc", "Σq̈²", "−1e-6", "−1e-6", "关节加速度"],
            ["wheel_spin", "Σω_wheel²", "−3e-5", "−3e-5", "轮子空转轻罚"],
            ["closure_residual", "Σ|W−E|²", "−1000", "−1000", "五连杆闭环硬约束"],
        ],
        widths=[3.2, 4.3, 2.2, 2.2, 3.9],
        caption="两阶段奖励权重速查（A = v1.0.7，B = v1.0.8）",
        align_first_left=True,
        font_size=8.0,
        header_size=8.0,
    )
    r.p("设计逻辑用四句话说明即可：")
    r.bullets([
        [("① 姿态用 L1：", True), ("|x| 的梯度恒定，能从任何距离把姿态拉回；x² 在 0 附近没梯度，实测会“蹲低 5 cm 跑更快”。", False)],
        [("② 速度跟踪三段式：", True), ("exp 负责大误差有界收敛，L2 负责跳出“站着不动”局部最优，L1 负责把最后几 cm/s 压下去。", False)],
        [("③ 轮力矩带死区：", True), ("轮子也是唯一纵向执行器，普通 L2 会罚正常加速；死区后只罚姿态纠正尖峰，把平衡“交还”给髋关节。", False)],
        [("④ 平滑分一阶/二阶：", True), ("action_rate 放宽让腿敢动，action_jerk（二阶差分）专杀高频抖动，平滑运动基本免费。", False)],
    ])


def section_qa(r: Report):
    r.h1("5  老师可能追问的问题与回答")
    qa = [
        ("为什么阶段 B 只有 124 轮？",
         "8 GB 单卡 + PhysX GPU 周期性崩溃（约 16 分钟一次），阶段 B 在第 1123 轮中断，没来得及保存 model_1150；"
         "属于硬件中断而不是算法发散（无 NaN/Inf）。机制上每 50 轮存检查点、支持 --resume_latest 续训，后续可接着训。"),
        ("为什么误差只到约 0.07 m/s，离 0.025 还差 2.5 倍？",
         "阶段 A 的奖励只追“大致对”（exp + L2），误差停在 0.07 左右；阶段 B 刚加入 L1 精度项、只训了 124 轮，"
         "L1 项数值仍很小（−0.08 / −0.15），说明还没进入到厘米级区间。另外探索 std 已贴近下限 0.06。"
         "下一步继续训练并配合确定性回放评测。"),
        ("阶段 B 的 pitch 权重为什么是 0？",
         "v1.0.8 改版时把 pitch 权重置成了 −0.0（疑似误置，VERSION.md 仍记录 −4.0），报告附录 C 已列为待核对项。"
         "所以阶段 B 对俯仰没有奖励约束，评估俯仰需要用 play.py 回放单独测。"),
        ("训练为什么会崩？怎么解决的？",
         "定位为 PhysX GPU 分配器/fabric 的缺陷，不是显存不足（崩溃时显存只用了约 1 GB/8 GB）。缓解手段："
         "并行环境限制在 1024、网络回到 128-128-64、patch 缓冲回默认、NaN 防护、每 50 轮存检查点并续训；"
         "长时间训练前建议重启机器。"),
        ("为什么两阶段的回报不能直接比较？",
         "续训时奖励权重改了 9 处（v1.0.7 → v1.0.8），回报的“量纲”变了，绝对数不可比；只能看阶段内趋势，"
         "或者比较不受权重影响的指标（速度误差、回合长度、超时率）。"),
        ("这套策略能直接上真机吗？",
         "不能直接上。当前训练是平地、无域随机化、无电机执行器建模（饱和/延迟/摩擦），只覆盖 0.2–1.2 m/s 前进。"
         "上真机前需要：加域随机化与执行器模型、做仿真-实车同指令对比（站立 / 0.3 / 0.8 m/s）、再逐步放开。"),
        ("为什么没有训练转弯和倒车？",
         "本阶段先聚焦“前进 + 站立”这个基础能力；转弯需要随机航向目标（单独一个 run 以隔离变量）；"
         "倒车按当前要求不启用（速度范围保持非负）。"),
        ("下一步计划是什么？需要老师什么支持？",
         "计划：继续 v1.0.8 配置训练到 2000–3000 轮 → 每 200 轮做确定性回放评测 → 修配置/文档一致性 → "
         "再扩转弯、域随机化、sim2real。需要老师定：① 0.025 m/s 精度目标是否现实、② 优先级（先补能力还是先上真机）、"
         "③ 训练硬件是否升级（现在崩溃太频繁）。"),
    ]
    for q, a in qa:
        r.rich([("Q：" + q, True)])
        r.p(a)


def section_short(r: Report):
    r.h1("6  3 分钟 / 10 分钟两种讲法")
    r.bullets([
        [("3 分钟版（只讲 4 张图）：", True),
         ("图 1 回报从 −135 → +116、阶段 A 基本收敛；图 2 回合长度 890 → 985、超时率 97.5%；"
          "图 3 误差台阶式下降、真实误差约 0.07 m/s（目标 0.025）；图 8 PPO 健康但探索接近最小。", False)],
        [("10 分钟版：", True), ("按第 1 节的 7 页结构讲，结果部分用图 1/2/3/5（姿态）和数字速查表。", False)],
        [("收尾一句话：", True),
         ("“回报和回合长度证明能站能走；误差图说明已到 0.07 m/s，离目标 0.025 还差 2.5 倍；"
          "PPO 诊断说明训练本身健康，只是探索快耗尽——下一步继续训精度，再补转弯和真机。”", False)],
    ])
    r.note("完整技术报告与全部原始数据：report/Infantry-RL_轮腿机器人运动控制训练报告_2026-10-22.docx、"
           "report/figures/、report/data/metrics_summary.md、report/analysis/training_facts.md。")


def main():
    r = Report()
    setup_section(r.doc.sections[0],
                  header_text="项目进展汇报材料 · 训练曲线与关键数据讲解", footer_numbers=True)
    title_block(r)
    section_mainline(r)
    section_figures(r)
    section_numbers(r)
    section_rewards(r)
    section_qa(r)
    section_short(r)

    props = r.doc.core_properties
    props.title = "项目进展汇报材料 —— 训练曲线与关键数据讲解"
    props.subject = "轮腿式 RoboMaster 步兵底盘 · 强化学习运动控制（Isaac Lab + PPO）"
    props.author = "Infantry-RL 项目组"
    r.doc.save(str(OUT_DOCX))
    print(f"[OK] saved: {OUT_DOCX}")
    print(f"[OK] paragraphs: {len(r.doc.paragraphs)}, tables: {len(r.doc.tables)}, "
          f"figures: {len(r.doc.inline_shapes)}")


if __name__ == "__main__":
    main()
