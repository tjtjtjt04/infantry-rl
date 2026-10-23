# 训练指标汇总（阶段A 第1次训练 / 阶段B 续训）

> 本文件由 `report/data/make_report_data.py` 从 TensorBoard 事件文件只读解析生成。
> 数值为原始标量（未平滑）；表中均值均为对应窗口内逐迭代值的算术平均。

## 1. 训练规模

| 项目 | 阶段A（第1次训练） | 阶段B（续训） |
|---|---|---|
| 运行目录 | `logs\rsl_rl\infantry_rl_locomotion\2026-10-22_00-28-38` | `logs\rsl_rl\infantry_rl_locomotion\2026-10-22_01-17-16` |
| 迭代 step 范围 | 0 – 999 | 999 – 1123 |
| 迭代数（事件数） | 1000 | 125 |
| 标量 tag 总数 / 按迭代 / 时间轴 tag | 39 / 37 / 2 | 41 / 39 / 2 |
| Episode_Reward 项数 | 22 | 24 |
| 事件墙钟时长 | 0.689 h | 0.102 h |
| NaN / Inf 的 tag 数 | 0 | 0 |

**连续性**：阶段B 的 step 编号接着阶段A 继续（B 首步 999 = A 末步 999），因此两阶段在 step 999 处有 1 个重叠步；阶段B 共 125 条事件，即续训后新增 124 个训练迭代。

**阶段B 结束情况**：事件流止于 step 1123（无 NaN/Inf）；目录内 checkpoint 为 model_1000/1050/1100，没有 model_1150，说明训练未跑满即结束（中断/手动停止），不是数值发散。

**奖励配置变化（v1.0.7 → v1.0.8，来自两个 run 的 `params/env.yaml` 差异）**：

| 配置项 | 阶段A | 阶段B |
|---|---|---|
| action_jerk 权重 | -0.5 | -0.3 |
| pitch 权重 | -4.0 | -0.0 |
| track_ang_vel_z 核宽 std | 0.35 | 0.15 |
| track_ang_vel_z 权重 | 1.0 | 3.0 |
| track_ang_vel_z_l1 权重（新增） | （无） | -1.0 |
| track_ang_vel_z_l2 权重 | -1.0 | -2.0 |
| track_lin_vel_x 核宽 std | 0.25 | 0.15 |
| track_lin_vel_x 权重 | 6.0 | 8.0 |
| track_lin_vel_x_l1 权重（新增） | （无） | -2.0 |
| track_lin_vel_x_l2 权重 | -3.0 | -6.0 |
| wheel_torques 死区阈值 | 1.0 | 1.5 |

由于奖励权重改变，**Episode_Reward/* 与 Train/mean_reward 的绝对值不能跨阶段直接比较**，只能比较同一阶段内的趋势；阶段B 的 L1 项为新增项，阶段A 无对应数据。

## 2. 窗口均值（前100 / 后100 / 后50 迭代）

窗口按各阶段内部迭代顺序切分：阶段A = step 0–99 / 900–999 / 950–999；阶段B = step 999–1098 / 1024–1123 / 1074–1123。

### 2.1 训练信号

| 指标 | A 前100 | A 后100 | A 后50 | B 前100 | B 后100 | B 后50 |
|---|---|---|---|---|---|---|
| `Train/mean_reward` | -135.436 | 115.817 | 115.691 | 142.852 | 161.103 | 160.975 |
| `Train/mean_episode_length` | 890.60 | 984.65 | 988.39 | 879.08 | 985.13 | 984.26 |

### 2.2 速度跟踪奖励项

| 指标 | A 前100 | A 后100 | A 后50 | B 前100 | B 后100 | B 后50 |
|---|---|---|---|---|---|---|
| `Episode_Reward/track_lin_vel_x` | 0.85993 | 5.55772 | 5.59206 | 6.50379 | 7.27096 | 7.27067 |
| `Episode_Reward/track_lin_vel_x_l1` | — | — | — | -0.07677 | -0.08219 | -0.08216 |
| `Episode_Reward/track_lin_vel_x_l2` | -1.37133 | -0.04901 | -0.05206 | -0.08455 | -0.08812 | -0.08708 |
| `Episode_Reward/track_ang_vel_z` | 0.20234 | 0.76737 | 0.76699 | 1.28437 | 1.43511 | 1.43785 |
| `Episode_Reward/track_ang_vel_z_l1` | — | — | — | -0.13891 | -0.15298 | -0.15237 |
| `Episode_Reward/track_ang_vel_z_l2` | -1.68766 | -0.05108 | -0.06214 | -0.07451 | -0.07702 | -0.07652 |

### 2.3 姿态 / 高度奖励项

| 指标 | A 前100 | A 后100 | A 后50 | B 前100 | B 后100 | B 后50 |
|---|---|---|---|---|---|---|
| `Episode_Reward/base_height` | -0.09305 | -0.02874 | -0.02895 | -0.02795 | -0.03176 | -0.03221 |
| `Episode_Reward/pitch` | -2.34940 | -0.06783 | -0.07244 | 0 | 0 | 0 |
| `Episode_Reward/roll` | -0.03615 | -0.01348 | -0.01453 | -0.01741 | -0.02087 | -0.02297 |
| `Episode_Reward/roll_straight` | -0.03359 | -0.02048 | -0.02154 | -0.02718 | -0.03267 | -0.03592 |

### 2.4 平滑 / 能耗 / 其它奖励项

| 指标 | A 前100 | A 后100 | A 后50 | B 前100 | B 后100 | B 后50 |
|---|---|---|---|---|---|---|
| `Episode_Reward/action_rate` | -0.00358 | -0.00038 | -0.00039 | -0.00035 | -0.00038 | -0.00040 |
| `Episode_Reward/action_jerk` | -1.25049 | -0.11402 | -0.11597 | -0.06270 | -0.06993 | -0.07226 |
| `Episode_Reward/joint_acc` | -0.30490 | -0.24907 | -0.25258 | -0.22516 | -0.25162 | -0.24780 |
| `Episode_Reward/joint_torques` | -0.11007 | -0.11489 | -0.11624 | -0.10343 | -0.11408 | -0.11287 |
| `Episode_Reward/wheel_torques` | -0.18043 | -0.16722 | -0.16813 | -0.08884 | -0.09901 | -0.09866 |
| `Episode_Reward/wheel_axle` | -0.21266 | -0.09739 | -0.09286 | -0.05940 | -0.06457 | -0.06613 |
| `Episode_Reward/wheel_spin` | -0.00036 | -0.00815 | -0.00818 | -0.00725 | -0.00815 | -0.00803 |
| `Episode_Reward/closure_residual` | -0.00040 | -0.00062 | -0.00062 | -0.00057 | -0.00063 | -0.00063 |
| `Episode_Reward/fall_penalty` | -2.604e-05 | -0.00158 | -0.00110 | -0.00136 | -0.00154 | -0.00174 |
| `Episode_Reward/alive` | 0.44782 | 0.49196 | 0.49483 | 0.44269 | 0.49288 | 0.49246 |
| `Episode_Reward/heading_error` | -0.58830 | -0.03824 | -0.04381 | -0.03361 | -0.03666 | -0.04149 |
| `Episode_Reward/lin_vel_y` | -0.00539 | -0.00107 | -0.00113 | -0.00091 | -0.00099 | -0.00098 |
| `Episode_Reward/lin_vel_z` | -0.01753 | -0.00624 | -0.00625 | -0.00572 | -0.00628 | -0.00633 |
| `Episode_Reward/roll_rate` | -0.01377 | -0.00263 | -0.00274 | -0.00228 | -0.00250 | -0.00249 |

### 2.5 速度跟踪误差（越低越好）

| 指标 | A 前100 | A 后100 | A 后50 | B 前100 | B 后100 | B 后50 |
|---|---|---|---|---|---|---|
| `Metrics/base_velocity/error_vel_xy` | 1.1471 | 0.1386 | 0.1415 | 0.1206 | 0.1306 | 0.1296 |
| `Metrics/base_velocity/error_vel_yaw` | 1.8910 | 0.3211 | 0.3337 | 0.2795 | 0.3081 | 0.3069 |

### 2.6 PPO 内部量

| 指标 | A 前100 | A 后100 | A 后50 | B 前100 | B 后100 | B 后50 |
|---|---|---|---|---|---|---|
| `Loss/value` | 3.2526 | 0.3684 | 0.5351 | 0.5727 | 0.4639 | 0.4594 |
| `Loss/surrogate` | -0.00892 | -0.00260 | -0.00301 | -0.00094 | -0.00208 | -0.00242 |
| `Loss/entropy` | 0.870 | -7.200 | -7.096 | -7.307 | -7.361 | -7.357 |
| `Loss/learning_rate` | 3.808e-03 | 7.684e-05 | 4.982e-05 | 1.346e-04 | 1.482e-04 | 1.466e-04 |
| `Policy/mean_std` | 0.2846 | 0.0742 | 0.0754 | 0.0732 | 0.0726 | 0.0728 |

### 2.7 回合终止率

| 指标 | A 前100 | A 后100 | A 后50 | B 前100 | B 后100 | B 后50 |
|---|---|---|---|---|---|---|
| `Episode_Termination/time_out` | 0.8959 | 0.9750 | 0.9818 | 0.8724 | 0.9691 | 0.9635 |
| `Episode_Termination/bad_orientation` | 0.0006 | 0.0250 | 0.0182 | 0.0257 | 0.0309 | 0.0365 |
| `Episode_Termination/base_too_low` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### 2.8 训练性能

| 指标 | A 前100 | A 后100 | A 后50 | B 前100 | B 后100 | B 后50 |
|---|---|---|---|---|---|---|
| `Perf/total_fps` | 28076 | 16038 | 12791 | 21816 | 18426 | 12953 |
| `Perf/collection_time` | 1.604 | 3.130 | 3.542 | 2.426 | 2.878 | 3.570 |
| `Perf/learning_time` | 0.154 | 0.278 | 0.306 | 0.214 | 0.251 | 0.306 |

## 3. 阶段A 起点 vs 终点变化（前100均值 → 后100均值）

| 指标 | 前100均值 | 后100均值 | 变化量 | 相对变化 |
|---|---|---|---|---|
| `Train/mean_reward` | -135.436 | 115.817 | 251.252 | +185.5% |
| `Train/mean_episode_length` | 890.60 | 984.65 | 94.05 | +10.6% |
| `Episode_Reward/track_lin_vel_x` | 0.85993 | 5.55772 | 4.69779 | +546.3% |
| `Episode_Reward/track_lin_vel_x_l2` | -1.37133 | -0.04901 | 1.32232 | +96.4% |
| `Episode_Reward/track_ang_vel_z` | 0.20234 | 0.76737 | 0.56503 | +279.2% |
| `Episode_Reward/track_ang_vel_z_l2` | -1.68766 | -0.05108 | 1.63658 | +97.0% |
| `Episode_Reward/base_height` | -0.09305 | -0.02874 | 0.06431 | +69.1% |
| `Episode_Reward/pitch` | -2.34940 | -0.06783 | 2.28158 | +97.1% |
| `Episode_Reward/roll` | -0.03615 | -0.01348 | 0.02267 | +62.7% |
| `Episode_Reward/roll_straight` | -0.03359 | -0.02048 | 0.01312 | +39.0% |
| `Episode_Reward/action_rate` | -0.00358 | -0.00038 | 0.00320 | +89.3% |
| `Episode_Reward/action_jerk` | -1.25049 | -0.11402 | 1.13648 | +90.9% |
| `Episode_Reward/joint_acc` | -0.30490 | -0.24907 | 0.05584 | +18.3% |
| `Episode_Reward/joint_torques` | -0.11007 | -0.11489 | -0.00483 | -4.4% |
| `Episode_Reward/wheel_torques` | -0.18043 | -0.16722 | 0.01321 | +7.3% |
| `Episode_Reward/wheel_axle` | -0.21266 | -0.09739 | 0.11527 | +54.2% |
| `Episode_Reward/wheel_spin` | -0.00036 | -0.00815 | -0.00779 | -2144.3% |
| `Episode_Reward/closure_residual` | -0.00040 | -0.00062 | -0.00021 | -53.0% |
| `Episode_Reward/fall_penalty` | -2.604e-05 | -0.00158 | -0.00156 | -5973.3% |
| `Episode_Reward/alive` | 0.44782 | 0.49196 | 0.04413 | +9.9% |
| `Episode_Reward/heading_error` | -0.58830 | -0.03824 | 0.55006 | +93.5% |
| `Episode_Reward/lin_vel_y` | -0.00539 | -0.00107 | 0.00432 | +80.1% |
| `Episode_Reward/lin_vel_z` | -0.01753 | -0.00624 | 0.01128 | +64.4% |
| `Episode_Reward/roll_rate` | -0.01377 | -0.00263 | 0.01114 | +80.9% |
| `Metrics/base_velocity/error_vel_xy` | 1.1471 | 0.1386 | -1.0085 | -87.9% |
| `Metrics/base_velocity/error_vel_yaw` | 1.8910 | 0.3211 | -1.5698 | -83.0% |
| `Loss/value` | 3.2526 | 0.3684 | -2.8842 | -88.7% |
| `Loss/surrogate` | -0.00892 | -0.00260 | 0.00632 | +70.8% |
| `Loss/entropy` | 0.870 | -7.200 | -8.070 | -927.1% |
| `Loss/learning_rate` | 3.808e-03 | 7.684e-05 | -3.731e-03 | -98.0% |
| `Policy/mean_std` | 0.2846 | 0.0742 | -0.2103 | -73.9% |
| `Episode_Termination/time_out` | 0.8959 | 0.9750 | 0.0791 | +8.8% |
| `Episode_Termination/bad_orientation` | 0.0006 | 0.0250 | 0.0244 | +3991.8% |
| `Episode_Termination/base_too_low` | 0.0000 | 0.0000 | 0.0000 | — |
| `Perf/total_fps` | 28076 | 16038 | -12038 | -42.9% |
| `Perf/collection_time` | 1.604 | 3.130 | 1.526 | +95.1% |
| `Perf/learning_time` | 0.154 | 0.278 | 0.124 | +80.2% |

## 4. 最佳 100 迭代平均回报（滚动均值最大窗口）

| 阶段 | 迭代数 | 100 轮窗口数 | 最佳100轮均值 | 窗口起始 step | 窗口结束 step | 窗口中心 step |
|---|---|---|---|---|---|---|
| 阶段A | 1000 | 901 | 116.6011 | 853 | 952 | 902 |
| 阶段B | 125 | 26 | 161.1033 | 1024 | 1123 | 1074 |

阶段A：最佳 100 轮均值 116.6011，末 100 轮均值 115.8167，差 +0.7844；阶段B：最佳 100 轮均值 161.1033，末 100 轮均值 161.1033，差 +0.0000（阶段B 仅 125 个迭代，窗口重叠，仅供参考）。

## 5. 阶段B 最终值（step 1123，也是全部数据的最后一条）

| 指标 | 最终值 |
|---|---|
| `Train/mean_reward` | 163.511 |
| `Train/mean_episode_length` | 986.98 |
| `Episode_Reward/track_lin_vel_x` | 7.39837 |
| `Episode_Reward/track_lin_vel_x_l1` | -0.07266 |
| `Episode_Reward/track_lin_vel_x_l2` | -0.07298 |
| `Episode_Reward/track_ang_vel_z` | 1.46128 |
| `Episode_Reward/track_ang_vel_z_l1` | -0.15205 |
| `Episode_Reward/track_ang_vel_z_l2` | -0.07606 |
| `Episode_Reward/base_height` | -0.03221 |
| `Episode_Reward/pitch` | 0 |
| `Episode_Reward/roll` | -0.02518 |
| `Episode_Reward/roll_straight` | -0.03913 |
| `Episode_Reward/action_rate` | -0.00041 |
| `Episode_Reward/action_jerk` | -0.07464 |
| `Episode_Reward/joint_acc` | -0.22809 |
| `Episode_Reward/joint_torques` | -0.11369 |
| `Episode_Reward/wheel_torques` | -0.09938 |
| `Episode_Reward/wheel_axle` | -0.07097 |
| `Episode_Reward/wheel_spin` | -0.00738 |
| `Episode_Reward/closure_residual` | -0.00064 |
| `Episode_Reward/fall_penalty` | -0.00052 |
| `Episode_Reward/alive` | 0.49612 |
| `Episode_Reward/heading_error` | -0.03251 |
| `Episode_Reward/lin_vel_y` | -0.00098 |
| `Episode_Reward/lin_vel_z` | -0.00600 |
| `Episode_Reward/roll_rate` | -0.00242 |
| `Metrics/base_velocity/error_vel_xy` | 0.1221 |
| `Metrics/base_velocity/error_vel_yaw` | 0.3061 |
| `Loss/value` | 0.5597 |
| `Loss/surrogate` | -0.00232 |
| `Loss/entropy` | -7.443 |
| `Loss/learning_rate` | 2.563e-04 |
| `Policy/mean_std` | 0.0718 |
| `Episode_Termination/time_out` | 0.9673 |
| `Episode_Termination/bad_orientation` | 0.0327 |
| `Episode_Termination/base_too_low` | 0.0000 |
| `Perf/total_fps` | 28004 |
| `Perf/collection_time` | 1.601 |
| `Perf/learning_time` | 0.154 |

## 6. 数据说明 / 注意事项

- 阶段A: Episode_Termination/base_too_low 全程恒为 0（该终止条件未触发）。
- 阶段B: Episode_Termination/base_too_low 全程恒为 0（该终止条件未触发）。
- 阶段B: Episode_Reward/pitch 全程恒为 0；params/env.yaml 显示续训时 pitch 奖励权重 由 -4.0 改为 -0.0，因此该曲线为零不是数据缺失。
- 阶段B 的 step 编号接着阶段A 继续（B 首步 999 = A 末步 999），因此两阶段在 step 999 处有 1 个重叠步；阶段B 共 125 条事件，即续训后新增 124 个训练迭代。
- 阶段B 事件流在 step 1123 处直接结束（无 NaN/Inf），目录中最后一个 checkpoint 为 model_1100.pt，没有 model_1150.pt，因此属于未跑满/中断，而非数值发散。
- 续训时奖励配置由 v1.0.7 升级为 v1.0.8（见 config_changes_A_to_B）：track_lin_vel_x 6→8、track_lin_vel_x_l2 -3→-6、track_ang_vel_z 1→3、track_ang_vel_z_l2 -1→-2、新增 L1 项（lin -2.0 / yaw -1.0）、action_jerk -0.5→-0.3、wheel_torques 死区 1.0→1.5、pitch -4.0→-0.0。
- 由于奖励项权重在续训时改变，Episode_Reward/* 与 Train/mean_reward 的绝对值不能跨阶段直接比较，只能比较趋势；阶段B 新增的两个 L1 项只存在于阶段B。
- 窗口定义：first_100=该阶段前100个迭代，last_100=后100个迭代，last_50=后50个迭代；阶段B 只有125个迭代，first_100 与 last_100 有 75 个迭代重叠，解释时需注意。

宽表 CSV 列名与 TensorBoard tag 完全一致（`run_A_scalars.csv` / `run_B_scalars.csv`），`step` 列为 TensorBoard step（= rs_rl 训练迭代号）。

