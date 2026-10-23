# 报告数据与图说明（report/data/README.md）

本目录保存机器人轮腿底盘 RL 运动训练（Isaac Lab + rsl_rl）两个阶段的数据提取结果与指标汇总，
供报告正文引用；9 张成图在 `report/figures/` 下。所有数值均由 TensorBoard 事件文件只读解析得到，
未做平滑、插值或人工修改（图上的 EMA 仅用于显示）。

数据来源（仅这两个 run）：

| 阶段 | run 目录 | events 文件 | 奖励配置 | 迭代范围 | 事件数 | 墙钟时长 |
|---|---|---|---|---|---|---|
| 阶段A（第1次训练） | `logs/rsl_rl/infantry_rl_locomotion/2026-10-22_00-28-38` | `events.out.tfevents.1792600120.CMD.10812.0` | v1.0.7 | step 0–999 | 1000 | 0.689 h |
| 阶段B（续训） | `logs/rsl_rl/infantry_rl_locomotion/2026-10-22_01-17-16` | `events.out.tfevents.1792603039.CMD.12320.0` | v1.0.8 | step 999–1123 | 125 | 0.102 h |

## 0. 目录内容

| 文件 | 内容 |
|---|---|
| `run_A_scalars.csv` / `run_B_scalars.csv` | 宽表：一行一个训练迭代，首列 `step`（TensorBoard step），其后每列一个标量 tag（列名与 tag 完全一致）。A：1000 行 × 38 列（37 个标量 + step）；B：125 行 × 40 列（39 个标量 + step）。无 NaN/空列。 |
| `run_ranges.json` | 两个 run 的精确首末 step、迭代数、各 tag 事件数与首末 step、标量 tag 总数、Episode_Reward 项数、墙钟起止时间、`nan_inf_tags`、两阶段 step 连续性判断（B 首步 999 = A 末步 999，重叠 1 步；B 新增 124 个迭代）。 |
| `metrics_summary.json` | 机器可读的完整指标：每个阶段的 first_100 / last_100 / last_50 窗口均值、全段均值、首末/极值，阶段A 前100→后100 变化量与相对变化，最佳 100 轮滚动均值及窗口位置，阶段B 最终值，以及两个 run 的 `params/env.yaml` 奖励配置差异。 |
| `metrics_summary.md` | 同上内容的中文表格版（报告可直接引用）。 |
| `make_report_data.py` | 数据提取脚本：从 events 生成上述 CSV/JSON/MD，可重复运行。 |
| `plot_report_figures.py` | 绘图脚本：读 CSV 生成 9 张 200 dpi PNG，可重复运行。 |

口径说明：`Train/mean_reward/time` 与 `Train/mean_episode_length/time` 两个 tag 的 TensorBoard step 是**累计训练秒数**（rsl_rl 用 `tot_time` 作 step），无法与迭代号对齐，因此不写入宽表 CSV；其事件数/首末 step 见 `run_ranges.json` 的 `tag_counts`。

## 1. 续训时的奖励配置变化（v1.0.7 → v1.0.8）

来自两个 run 的 `params/env.yaml` 实际差异（`metrics_summary.md` 第 1 节有表）：

- `track_lin_vel_x` 权重 6 → 8，指数核 std 0.25 → 0.15；`track_lin_vel_x_l2` 权重 −3 → −6；**新增** `track_lin_vel_x_l1`（−2.0）。
- `track_ang_vel_z` 权重 1 → 3，核 std 0.35 → 0.15；`track_ang_vel_z_l2` 权重 −1 → −2；**新增** `track_ang_vel_z_l1`（−1.0）。
- `action_jerk` 权重 −0.5 → −0.3；`wheel_torques` 死区阈值 1.0 → 1.5 N·m；`pitch` 权重 −4.0 → −0.0。

**结论：`Episode_Reward/*` 与 `Train/mean_reward` 的绝对值不能跨阶段直接比较**，只能比较各阶段内部的趋势；
阶段B 新增的两个 L1 项在阶段A 无对应数据。另：B 的 `Episode_Reward/pitch` 全程恒为 0，是权重改为 0 所致，不是数据缺失。

数据质量：两个 run 的全部标量 tag 均无 NaN/Inf；`Episode_Termination/base_too_low` 两阶段全程为 0（未触发）。
阶段B 事件流在 step 1123 处直接结束，目录内 checkpoint 为 model_1000/1050/1100（没有 model_1150），
属于未跑满的中断/主动停止，不是数值发散。

## 2. 逐图说明

### 01_training_reward.png — 平均回报
两阶段 `Train/mean_reward` 原始值（细线）+ EMA(span=20)（粗线）；虚线=续训起点第 999 轮，浅红色区域=阶段B；图内注明奖励权重重定义。
关键数字：阶段A 从第 0 轮 −3.167 起步，第 22 轮最低 **−249.102**（探索期大量摔倒），前100轮均值 −135.436，之后持续上升，
约 700 轮后进入平台：末100轮均值 **115.817**、末50轮 115.691、峰值 **119.713**（第 941 轮）、最佳100轮均值 **116.6011**（853–952 轮）。
阶段B 首条日志（第 999 轮）为 **0.443**（续训边界：奖励重定义 + 回合统计刚重启，不代表真实水平），随后快速升至稳态，末100轮均值 **161.103**、末50轮 160.975、最终 **163.511**（第 1123 轮），全段最高 **165.749**（第 1058 轮）。

### 02_episode_length.png — 回合长度与终止率
上图 `Train/mean_episode_length`，下图三种 `Episode_Termination/*` 占比（time_out 实线、bad_orientation 虚线、base_too_low 点线）。
关键数字：阶段A 前100轮 890.60 步 → 末100轮 **984.65** 步（回合上限 1000 步），最终 995.78；阶段B 末100轮 **985.13**、最终 **986.98**。
终止率：阶段A time_out 0.8959 → 0.9750（末100轮）、bad_orientation 0.0006 → 0.0250（第 547 轮峰值 0.3117）；
阶段B 末100轮 time_out 0.9691、bad_orientation 0.0309（末50轮 0.0365，略高于阶段A 的 0.0182）。

### 03_tracking_rewards.png — 速度跟踪奖励项（含 L1/L2）
上=前进速度 `track_lin_vel_x`，下=航向角速度 `track_ang_vel_z`；左轴为奖励项，右轴为 L1/L2 惩罚项（负值）。
关键数字（末100轮均值）：阶段A `track_lin_vel_x` 5.55772（权重 6）、`track_ang_vel_z` 0.76737（权重 1）、
`track_lin_vel_x_l2` −0.04901、`track_ang_vel_z_l2` −0.05108；
阶段B `track_lin_vel_x` 7.27096（权重 8）、`track_ang_vel_z` 1.43511（权重 3）、
`track_lin_vel_x_l1`（新增）−0.08219、`track_ang_vel_z_l1`（新增）−0.15298、
`track_lin_vel_x_l2` −0.08812、`track_ang_vel_z_l2` −0.07702。阶段A 早期 L2 惩罚为 −1.37/−1.69，随跟踪改善快速收敛到 −0.05 量级。
注意：奖励项绝对值随权重变化，跨阶段只看趋势；L1 只有阶段B 有数据。

### 04_attitude_height.png — 姿态 / 高度奖励项（2×2）
`base_height`、`pitch`、`roll`、`roll_straight` 四项的加权奖励值。
关键数字（末100轮均值）：阶段A base_height −0.02874、pitch −0.06783、roll −0.01348、roll_straight −0.02048；
阶段B base_height −0.03176、pitch **0**（权重已改为 0，曲线恒零，非缺失）、roll −0.02087、roll_straight −0.03267。
阶段A 的 pitch 惩罚从 −2.349（前100轮）收敛到 −0.068，说明姿态控制在该项上学到；阶段B roll/roll_straight 惩罚略大于阶段A，
与阶段B 更激进的前进速度跟踪（姿态代价增加）一致，但因权重未变、绝对值可粗略对比。

### 05_velocity_error.png — 速度跟踪误差（对数轴，越低越好）
上=`error_vel_xy`（m/s），下=`error_vel_yaw`（rad/s），纵轴对数。
关键数字：阶段A `error_vel_xy` 前100轮 1.1471 → 末100轮 **0.1386**、最终 0.1420；`error_vel_yaw` 1.8910 → **0.3211**、最终 0.3234。
阶段B 末100轮 `error_vel_xy` **0.1306**（比阶段A 末100轮低约 5.8%）、末50轮 0.1296、最终 **0.1221**；`error_vel_yaw` 末100轮 0.3081、最终 **0.3061**。
阶段B 第 999 轮的极小值（0.0181/0.0123）是续训首条日志的瞬态，不能当作稳态误差。

### 06_ppo_losses.png — PPO 内部量（2×2）
`Loss/value`、`Loss/surrogate`、`Loss/entropy`、`Policy/mean_std`，细线原始值、粗线 EMA(span=20)。
关键数字：阶段A 价值损失 3.2526（前100轮）→ 0.3684（末100轮），策略 std 0.2846 → 0.0742，熵项 0.870 → −7.200；
阶段B 末100轮价值损失 0.4639、std 0.0726、熵项 −7.361（最终 −7.443）。
`Loss/entropy` 是 6 维高斯策略的微分熵：std 由 0.393 降到约 0.07 后，熵由正转负（第 75 轮首次为负，第 74 轮为 0.0202）。
灰色虚线为策略 std 限幅 [0.06, 0.4]（v1.0.7 起）；两阶段末段 std 都贴近 0.06 下限（A 0.0742、B 0.0726），说明探索已接近最小。

### 07_termination.png — 回合终止原因与回合长度
上图为 `time_out` / `bad_orientation` / `base_too_low` 两阶段曲线，下图为平均回合长度（与图 02 对照）。
关键数字：阶段A 姿态失控率在 400–600 轮抬升，**第 547 轮最高 0.3117**，随后回落，末100轮 0.0250；阶段B 稳定在 0.03 左右（末50轮 0.0365）。
`base_too_low` 两阶段全程 0。time_out 与回合长度互补：阶段A 末100轮 0.9750 / 984.65 步，阶段B 末100轮 0.9691 / 985.13 步。

### 08_mechanical_smoothness.png — 平滑与能耗（2×2）
`action_rate`、`action_jerk`、`wheel_torques`、`wheel_axle` 四项加权奖励。
关键数字（末100轮均值）：阶段A action_rate −0.00038、action_jerk −0.11402、wheel_torques −0.16722、wheel_axle −0.09739；
阶段B action_rate −0.00038、action_jerk −0.06993、wheel_torques −0.09901、wheel_axle −0.06457。
注意：`action_jerk` 权重由 −0.5 改为 −0.3，按权重反推的原始惩罚约 0.228（A）vs 0.233（B），即原始抖动水平接近；
`wheel_torques` 死区由 1.0 改为 1.5 N·m，两者绝对值不可直接比较。

### 09_performance.png — 训练性能（3 panel）
`Perf/total_fps`、`Perf/collection_time`、`Perf/learning_time`，细线原始值、粗线 EMA(span=20)。
关键数字：阶段A fps 前100轮 28076 → 末100轮 16038、collection_time 1.604 → 3.130 s/轮、learning_time 0.154 → 0.278 s/轮；
阶段B fps 21816 → 18426、collection 2.426 → 2.878 s/轮、learning 0.214 → 0.251 s/轮。
阶段A 中后段采样耗时明显上升、吞吐下降；事件日志没有记录原因，本报告只如实呈现该现象，不做因果解释。

## 3. 学习趋势总结（一段话）

阶段A 从随机策略起步，前 50 轮因大量摔倒出现最低 −249.10 的回报，随后在约 700 轮内持续改善并进入 115–119 的平台
（末100轮均值 115.82，最佳100轮 116.60，出现在 853–952 轮，与末100轮仅差 0.78，说明阶段A 已基本收敛）；
同期平均回合长度由 890.60 步升到 984.65 步（上限 1000），超时率由 0.896 升到 0.975，姿态失控率在 500–560 轮出现 0.31 的峰值后
回落到 0.025，速度跟踪误差 xy 由 1.147 降到 0.139 m/s、yaw 由 1.891 降到 0.321 rad/s，策略 std 由 0.285 收敛到 0.074，
价值损失由 3.25 降到 0.37——"站稳、跑满回合、把速度跟踪住"这三件事在阶段A 内基本学到位。
阶段B 是基于 v1.0.8 加严奖励（新增 L1 跟踪惩罚、L2 加严、跟踪权重提高）的 124 轮续训：受奖励重定义影响，第 999 轮首条日志
只有 0.443 且前 ~20 轮是热启动瞬态，不能当作真实水平；约 20 轮后回到稳态，末100轮回报 161.10、末50轮 160.98、
最佳100轮即末100轮（161.10，窗口 1024–1123），说明这 124 轮内回报已进入平台。可横向比较的跟踪误差确实进一步下降：
末100轮 error_vel_xy 0.1306（阶段A 0.1386，−5.8%）、error_vel_yaw 0.3081（阶段A 0.3211，−4.0%），末值 0.1221 / 0.3061 更低；
但**尚未收敛/尚未解决**的问题同样明显：稳态跟踪残差仍有 ~0.12 m/s 与 ~0.31 rad/s，姿态失控率反而从阶段A 末100轮的 0.025
升到阶段B 的 0.031（末50轮 0.036），roll / roll_straight 的加权惩罚绝对值变大（更快前进带来更大的姿态代价），
策略 std 已贴近 0.06 下限（0.0726，探索接近最小），且阶段B 只训练 124 轮便在 step 1123 处中断（无 NaN/Inf，最后一个
checkpoint 为 model_1100）——因此阶段B 的结果是"奖励改版后的短期验证"，不是最终收敛结论。

## 4. 复现方法

```powershell
D:\miniconda\envs\RM\python.exe D:\Robomaster\RL\report\data\make_report_data.py     # 重新提取 CSV / JSON / MD
D:\miniconda\envs\RM\python.exe D:\Robomaster\RL\report\data\plot_report_figures.py  # 重新生成 9 张 PNG
```

绘图使用 `C:\Windows\Fonts\msyh.ttc`（注册为 Microsoft YaHei），`axes.unicode_minus=False`。
阶段A 蓝色 `#1f77b4`，阶段B 红色 `#d62728`；图中虚线竖线与浅红底色标记续训起点（第 999 轮）。
