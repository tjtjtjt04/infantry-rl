# 两阶段奖励表（可直接粘贴进报告）

数据来源（权威）：`logs\rsl_rl\infantry_rl_locomotion\2026-10-22_00-28-38\params\env.yaml`（阶段 A，v1.0.7）
与 `logs\rsl_rl\infantry_rl_locomotion\2026-10-22_01-17-16\params\env.yaml`（阶段 B，v1.0.8）。
形状/实现见 `isaac_ws\isaac_ws_infantry\mdp\rewards.py`。Isaac Lab 按 `value × weight × dt` 累加
（dt=0.02 s），因此报告引用"配置权重"即可，绝对值跨阶段不可比。

## 阶段 A（v1.0.7，22 项，全部有效）

| # | 项 | 权重 | 形状 | 说明 |
|---|---|---|---|---|
| 1 | alive | +0.5 | 常数 | 存活奖励 |
| 2 | fall_penalty | −50.0 | 常数（摔倒步） | 摔倒重罚 |
| 3 | base_height | −4.0 | L1，目标 0.189775 m | 保持站高 |
| 4 | pitch | −4.0 | L1 | 抑制俯仰 |
| 5 | roll | −1.5 | L1 | 抑制侧倾 |
| 6 | roll_straight | −2.5 | L1 × wz* 门控 | 直行时不许侧倾 |
| 7 | roll_rate | −0.1 | L2 | 侧倾速率阻尼 |
| 8 | lin_vel_z | −2.0 | L2 | 抑制上下弹跳 |
| 9 | track_lin_vel_x | **+6.0**（std 0.25） | exp 核 | 前进速度跟踪 |
| 10 | track_ang_vel_z | **+1.0**（std 0.35） | exp 核 | 角速度跟踪 |
| 11 | heading_error | −0.6 | L2 | 到达目标航向 |
| 12 | track_lin_vel_x_l2 | −3.0 | L2 | 大误差压速 |
| 13 | track_ang_vel_z_l2 | −1.0 | L2 | yaw 抖动抑制 |
| 14 | lin_vel_y | −0.5 | L2 | 禁止侧滑 |
| 15 | wheel_axle | −8.0 | L1（米） | 轮轴对齐 |
| 16 | wheel_torques | −0.05（死区 **1.0 N·m**） | 带死区 L2 | 逼腿主导姿态 |
| 17 | action_rate | −4.0e-3 | L2 一阶差分 | 动作平滑 |
| 18 | action_jerk | **−0.5** | L2 二阶差分 | 只罚抖动 |
| 19 | joint_torques | −2.0e-5 | L2 | 关节力矩轻罚 |
| 20 | joint_acc | −1.0e-6 | L2 | 关节加速度 |
| 21 | wheel_spin | −3.0e-5 | L2 | 轮速轻罚 |
| 22 | closure_residual | −1000 | L2（米²） | 闭环硬约束 |

## 阶段 B（v1.0.8，24 项；pitch 权重 0 → 实际贡献 23 项）

| # | 项 | 权重 | 形状 | 说明 |
|---|---|---|---|---|
| 1 | alive | +0.5 | 常数 | 存活奖励 |
| 2 | fall_penalty | −50.0 | 常数（摔倒步） | 摔倒重罚 |
| 3 | base_height | −4.0 | L1，目标 0.189775 m | 保持站高 |
| 4 | pitch | **−0.0（=0）** | L1 | **本阶段实际不贡献** |
| 5 | roll | −1.5 | L1 | 抑制侧倾 |
| 6 | roll_straight | −2.5 | L1 × wz* 门控 | 直行时不许侧倾 |
| 7 | roll_rate | −0.1 | L2 | 侧倾速率阻尼 |
| 8 | lin_vel_z | −2.0 | L2 | 抑制上下弹跳 |
| 9 | track_lin_vel_x | **+8.0**（std **0.15**） | exp 核 | 前进速度跟踪（收窄核） |
| 10 | track_ang_vel_z | **+3.0**（std **0.15**） | exp 核 | 角速度跟踪（收窄核） |
| 11 | heading_error | −0.6 | L2 | 到达目标航向 |
| 12 | track_lin_vel_x_l1 | **−2.0（新增）** | L1 | 恒定梯度压最后几 cm/s |
| 13 | track_ang_vel_z_l1 | **−1.0（新增）** | L1 | yaw 恒定梯度项 |
| 14 | track_lin_vel_x_l2 | **−6.0** | L2 | 大误差压速（加严） |
| 15 | track_ang_vel_z_l2 | **−2.0** | L2 | yaw 抖动抑制（加严） |
| 16 | lin_vel_y | −0.5 | L2 | 禁止侧滑 |
| 17 | wheel_axle | −8.0 | L1（米） | 轮轴对齐 |
| 18 | wheel_torques | −0.05（死区 **1.5 N·m**） | 带死区 L2 | 逼腿主导姿态（放宽死区） |
| 19 | action_rate | −4.0e-3 | L2 一阶差分 | 动作平滑 |
| 20 | action_jerk | **−0.3** | L2 二阶差分 | 只罚抖动（放宽） |
| 21 | joint_torques | −2.0e-5 | L2 | 关节力矩轻罚 |
| 22 | joint_acc | −1.0e-6 | L2 | 关节加速度 |
| 23 | wheel_spin | −3.0e-5 | L2 | 轮速轻罚 |
| 24 | closure_residual | −1000 | L2（米²） | 闭环硬约束 |

## 一句话说明（可直接写进报告）

> 阶段 A 为 22 项奖励（v1.0.7）；阶段 B 为"精度跟踪"改版（v1.0.8）：跟踪核从
> std 0.25/0.35 收窄到 0.15、exp 主项权重 6.0→8.0 / 1.0→3.0、新增两条 L1 恒定梯度项
> （lin −2.0 / yaw −1.0）、L2 项加严（−3.0→−6.0 / −1.0→−2.0），同时放宽
> `action_jerk`（−0.5→−0.3）与轮力矩死区（1.0→1.5 N·m），`pitch` 权重被置 0
> （v1.0.8 变更记录未写这一条）。两次 run 的奖励项数分别为 **22 / 24**；
> 因权重不同，**`Train/mean_reward` 与 `Episode_Reward/*` 不能跨阶段直接比较**。
