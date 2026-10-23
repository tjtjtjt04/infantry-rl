# 版本说明（Version Notes）

本文件记录**代码版本**。每次改动训练逻辑、观测、奖励、超参或机器人参数，都在
「变更记录」里追加一条，并把上面的「当前冻结配置」更新为新值。

> 规则：**配置表必须与代码一致**。改完代码如果没同步这里，这个文件就失去意义。
> 核对方法见 §9「如何确认当前代码版本」。

## ⚠️ 改动后必须同步检查的两个文档

**每次改完代码，这两个文件都要过一遍**（缺一个都算没改完）：

| 文档 | 什么时候必须改 |
|---|---|
| **`VERSION.md`**（本文件） | 版本号 + 「冻结配置」对应小节 + 「变更记录」追加一行 |
| **`TRAINING_GUIDE.md`** | §0 速查 / §5 任务与配置 / §3.1 控制台示例 / 相关操作章节 |

### 对照清单

| 改了什么 | `VERSION.md` 要改 | `TRAINING_GUIDE.md` 要改 |
|---|---|---|
| 奖励项、权重、exp 核 `std` | §3.5 表 | §5 奖励表 |
| 速度范围 / 训练模式 / `_MODE_PRESETS` | §3.6 + 变更记录 | §5「改训练目标」整节（含 **行号**） |
| PPO 超参 / 网络 / std 限幅 / 熵 | §3.9 表 | §5 配置表 |
| 观测项 / 动作维度 | §3.3 / §3.4 | §5 任务表的「观测维度」+ 配置表 |
| 环境变量增删改 | §6 表 | §0 环境变量表 |
| 新增/删除文件、函数 | §4 文件清单与结构树 | §8 相关脚本 |
| 删除某个功能 | 变更记录里写明"已移除" | 删掉对应章节，别留死链接 |
| 控制台输出格式 | — | §3.1 的示例输出 |
| 机器人参数（质量、轮半径、执行器） | §3.1 / §3.2 | §5 配置表 |

### 收尾动作

改完后**跑一次 1 轮训练**，把启动日志里的 `[cfg] ...` 行和文档里贴的示例逐字对比：

```
[cfg] train mode          : STRAIGHT  (...)
[cfg] derived from TARGETS: wheel_scale 5.01 (动作量程上限 1.36 m/s), track_lin std 0.150 m/s, wheel_spin -3.00e-05
[cfg] command ranges      : vx [+0.20, +1.20] m/s ..., standing envs 10% (vx=0), wz clamp +/-2.50 rad/s, resample 5-10 s
[cfg] turning             : DISABLED -- heading TARGET pinned to 0 ...
```

**对不上就是文档过期了。**

---

---

# v1.0.27 — 2026-10-22

## 1. 版本定位

**首个「腿主导平衡」的完整 locomotion 实现。**

| 项目 | 值 |
|---|---|
| 基线提交 | `f99cddd`（git HEAD，提交信息 `latest`） |
| 本版本 | `f99cddd` + 9 个文件改动 |
| 改动量 | +370 行 / −25 行 |
| 训练任务 | `Infantry-RL-Locomotion-v0` |
| 回放任务 | `Infantry-RL-Locomotion-Play-v0` |
| 状态 | 可用（训练会因 PhysX GPU 缺陷周期性崩溃，见 §8） |

## 2. 与上一版本的核心差异

| # | 改动 | 旧 | 新 | 目的 |
|---|---|---|---|---|
| 1 | **观测加腿速度** | 31 维 | **35 维** | 让髋关节能做阻尼；对齐固件 LQR 的 `dφ` 状态 |
| 2 | **奖励加轮力矩惩罚** | 无 | **−0.05（1.0 N·m 死区）** | 逼髋关节接管姿态，而不是轮子 |
| 3 | **速度指令范围** | ±0.5 m/s | **±2.0 m/s** | 贴合实际（有快有慢） |
| 4 | **PPO 超参** | γ=0.99, epochs=5, `save_interval`=100 | **γ=0.995, epochs=8, `save_interval`=50** | 长视界 + 更高样本利用率 + 崩溃时少丢进度 |
| 5 | **稳定性加固** | 无 | patch 缓冲回默认、3 处 NaN 防护、脚本自动适配 checkpoint 结构 | 应对比特币级 GPU 瓶颈 |

### 2.1 为什么这些改动是"一套"的

这不是独立调参，而是围绕**一个物理事实**重做目标函数：

> 这台机器人的 **pitch 有两个冗余执行器**：
> - **轮子力矩** → 反作用力矩直接扭机体（Segway 式）
> - **髋关节** → 改变腿轴角 φ，让机体 CG 相对轮子前后移动（倒立摆式）

旧奖励里 `joint_torques = −1e-4`，轮子力矩 2 N·m 只罚 `8e-6/步`，而 `alive` 是
`1e-2/步` —— **轮子平衡便宜了 1250 倍**。PPO 因此选了"腿绷直 + 轮子扭动"。

本版本把轮子力矩变贵、给腿补上速度反馈、把姿态惩罚加严，
让 **腿成为姿态的便宜解**。这正是参考固件（`infantry-chassis`：LQR + VMC）的做法。

## 3. 冻结配置（本版本）

### 3.1 机器人

| 项目 | 值 |
|---|---|
| 总质量 | 12.776 kg（URDF 11 个 link 汇总） |
| 自由度 | 10（4 髋 + 4 膝 + 2 轮） |
| 闭环 | `<loop_joint>` 不是 DOF → **靠控制维持**，每控制步解一次 |
| 轮半径 | 0.068 m |
| 髋轴高度 | 0.128 m（相对 base_link） |
| 名义腿长 L0 | 0.249775 m |
| **名义站立高度** | **0.189775 m** |
| L0 可达范围 | 0.1356 … 0.2843 m |
| φ（腿轴角）范围 | −31.2° … +33.2°（**与 L0 强耦合**，L0=0.136 时只有 ±0.6°） |

### 3.2 执行器

| 组 | 关节数 | effort | velocity | stiffness | damping | armature |
|---|---|---|---|---|---|---|
| 髋 | 4 | 40 N·m | 16.755 rad/s | **200** | 5 | 0.01 |
| 膝（被动跟随） | 4 | 40 N·m | 16.755 rad/s | **400** | 10 | 0.01 |
| 轮 | 2 | **3.69 N·m** | **61.471 rad/s** | 0 | 2 | 0.001 |

### 3.3 动作空间（6 维）

| 维度 | 含义 | 缩放 | 限幅 |
|---|---|---|---|
| **v1.0.0** | 2026-10-20 | 首个腿主导平衡的 locomotion 实现 | 需要（观测 31→35） |
| **v1.0.1** | 2026-10-20 | 新增 `wheel_axle` 约束，禁止左右腿角不对称（轮轴歪斜） | 结构兼容可续训，**建议重训**（策略已学会歪着走） |
| **v1.0.2** | 2026-10-21 | 修复"站着不动"局部最优：`track_lin` 2.0→4.0、`track_ang` 2.5→1.5、`entropy_coef` 0.005→0.015 | **必须重训**（策略已锁死在"站着+原地转"） |
| **v1.0.3** | 2026-10-21 | 修 `entropy_coef` 0.015 导致的 std 失控（168）；加 std 硬限幅 [0.02,1.0]；加两个 L2 跟踪项抑制 yaw 抖动和速度波动 | **必须重训**（v1.0.2 的策略输出纯噪声） |
| **v1.0.4** | 2026-10-21 | **关闭转弯**：航向目标锁定为 0（`INFANTRY_RL_ENABLE_TURNING=0` 默认），任务简化为纯直线移动；`heading_error` −0.4→−0.6、`track_ang_vel_z_l2` −0.5→−1.0 | 建议重训（任务定义变了） |
| **v1.0.5** | 2026-10-21 | 压制腿抖：`entropy_coef` 0.005→0.001、std 限幅 [0.02,1.0]→[0.03,0.4]、新增 `action_jerk`（二阶差分）−0.5、`joint_acc` 1e-7→1e-6 | 建议重训（训练噪声条件变了） |
| **v1.0.6** | 2026-10-21 | 速度指令范围改为 **仅前进 [+0.2, +1.2] m/s**（`INFANTRY_RL_VX_MIN/MAX` 可调）；停止仍由 10% 站立环境训练 | 建议重训（指令分布变了） |
| **v1.0.7** | 2026-10-22 | 打破"静止不动"局部最优：`track_lin_vel_x` +4.0→+6.0、`track_lin_vel_x_l2` −1.0→−3.0、`track_ang_vel_z` +1.5→+1.0；`entropy_coef` 0.001→0.003、std 下限 0.03→0.06 恢复探索 | **必须重训** |
| **v1.0.8** | 2026-10-22 | **精度跟踪**：`track_lin_vel_x` +6.0→+8.0（std 0.25→0.15）、新增 L1 跟踪项（lin −2.0 / yaw −1.0）、L2 加严（−3.0→−6.0 / −1.0→−2.0）、`track_ang_vel_z` +3.0（std 0.35→0.15）；放宽 `action_jerk` −0.5→−0.3、`wheel_torques` 死区 1.0→1.5 | **必须重训** |
| **v1.0.10** | 2026-10-22 | **加入倒车**：速度指令范围 `[+0.2, +1.2]` → `[-1.2, +1.2]` m/s（对称，前进/倒车同速）；10% 站立环境保留作为 vx=0 刹车样本 | **必须重训**（指令分布变了） |
| **v1.0.11** | 2026-10-22 | **动态奖励**：新增 `mdp/curriculums.py`，`adaptive_tracking_weights`（按实测误差 → 权重 ×1.05/÷1.05，上限 4 倍）与 `modify_tracking_weights_scheduled`（2000 轮线性 1→3 倍）；由 `INFANTRY_RL_ADAPTIVE_REWARD` 开关，**默认关闭**（**v1.0.15 已整体移除**）。另修：控制台过滤器改为跨帧取 key 并集（否则中途出现的 `Curriculum/*` 不显示） | 可续训（初始 scale=1.0） |
| **v1.0.12** | 2026-10-22 | **控制台输出**：隐藏全部 `track_*` 行；新增实测物理量 `vel_x`（含 `cmd` 对照）、`pitch`/`yaw` 角（由 `root_quat_w` 反算） | 不需要（只改输出） |
| **v1.0.13** | 2026-10-22 | **控制台按“可对比项同行”重排**：`vel_x`/`cmd`/`diff` 一行，`pitch`/`roll`/`yaw` 一行，`error_vel_xy`/`error_vel_yaw` 一行（开启自适应时右侧并排显示 `curriculum_*` 权重倍数），惩罚项按一阶/二阶、力矩/加速度成对排列；`roll` 角加入姿态行 | 不需要（只改输出） |
| **v1.0.14** | 2026-10-22 | 动态奖励开关易用性：`INFANTRY_RL_ADAPTIVE_REWARD` 改为**前缀匹配**（`adapt*` → adaptive、`sched*` → schedule，之前必须写全 `adaptive`）；启动时打印 `[cfg] adaptive reward : ADAPTIVE / SCHEDULE / OFF`（**v1.0.15 已整体移除**） | 不需要 |
| **v1.0.15** | 2026-10-22 | **移除动态奖励功能**：删除 `mdp/curriculums.py`、`LocomotionCurriculumCfg`、`INFANTRY_RL_ADAPTIVE_REWARD` 开关及控制台 `curriculum_*` 列；奖励权重回到纯静态配置。保留：倒车速度范围、控制台分组排版 | 不需要（行为等同关闭） |
| **v1.0.16** | 2026-10-22 | **速度目标单点配置**：`LocomotionTargets` 新增 `wheel_scale` / `track_lin_std` / `wheel_spin_weight` 三个派生属性，在 `LocomotionEnvCfg.__post_init__` 里写回动作项与奖励权重 —— 只改 `TARGETS` 里的 `vx_min`/`vx_max` 即可，动作尺度与跟踪核宽度自动跟随；启动日志新增 `[cfg] derived from TARGETS` 行 | **换速度范围后必须重训** |
| **v1.0.17** | 2026-10-22 | **训练目标单点切换**：新增 `TRAIN_MODE`（`straight` / `turning` / `reverse` / `stand`）—— 改这一个字符串即可同时切换速度范围、航向目标、指令重采样间隔、站立环境比例，以及由速度反推的动作尺度/跟踪核宽度/wheel_spin 权重；启动日志新增 `[cfg] train mode` 行 | **切模式必须重训** |
| **v1.0.18** | 2026-10-22 | **模式间可续训**：`wheel_scale` / `track_lin_std` / `wheel_spin_weight` 改为按固定 `reference_max_vx=1.2` 计算（此前按各模式最高速算，导致 straight→turning 的动作尺度不同、无法 resume）；`turning` 预设保留 vx 指令 `[0.2,1.0]` 与 10% 站立环境，使直线与刹车能力在转弯训练中得以保留 | 同尺度模式间可续训 |
| **v1.0.19** | 2026-10-22 | **配置区前置 + 补齐注释**：速度常数（`REF_MAX_VX`/`WHEEL_RADIUS`/`CLIP_ACTIONS`/`ACTION_HEADROOM`）与`TRAIN_MODE`/`_MODE_PRESETS` 统一放到文件最开头（L74–270）；新增 `spin`（原地转到指定航向）与 `yaw_rate`（原地按指令角速度旋转）两种模式，共 6 种；`LocomotionTargets` 新增 `heading_command` 字段 | 见下表 |
| **v1.0.20** | 2026-10-22 | **速度数字单点化**：`_MODE_PRESETS` 写成完整表格（6 模式 x 12 个速度量，含 vy/wz/heading_command/rel_heading_envs），`LocomotionTargets` **移除全部字段默认值**（改为只读容器 + 派生公式），`TARGETS` 由表格解包构造 —— 全库速度数字只剩文件开头一张表 | 无行为变化 |
| **v1.0.21** | 2026-10-22 | 修启动日志误标：`[cfg] turning` 原先只看 `enable_turning`，导致 `yaw_rate` 模式被打印成 "DISABLED ... straight-line task"。改为按 `heading_command` 分三种形态打印（锁定 0 / 航向随机 / wz 直接采样） | 无行为变化 |
| **v1.0.22** | 2026-10-22 | **全部技能合成一个任务**：新增 `mdp/commands.py::SpinAwareVelocityCommand`（按 `rel_spin_envs` 比例把 vx 强制 0、保留航向环算出的 wz，用于学原地旋转）；新增 `TRAIN_MODE="full"`（vx ±1.2 + 航向随机 + 20% 原地旋转 + 10% 站立）；命令项统一换用该子类（`rel_spin_envs=0` 时行为与原先一致，已实测） | 同尺度可续训 |
| **v1.0.23** | 2026-10-22 | `play.py` 新增 **`--cmd_wz`**（rad/s，直接给偏航角速度，绕过航向环，用于 `yaw_rate` checkpoint）；同时修复固定指令只维持一个重采样窗口就被覆盖的问题（现在把 `time_left` 顶到 1e9） | 无需重训 |
| **v1.0.24** | 2026-10-22 | **修复 `pitch` 奖励权重笔误**：代码写成 `weight=-.0`（等于 0，pitch 惩罚完全失效），导致 `2026-10-22_01-17-16` 与 `15-22-23` 两次训练在无 pitch 约束下进行；已恢复 `-4.0`（与 §3.5 表一致） | **建议重训**（旧策略未受 pitch 约束） |
| **v1.0.25** | 2026-10-22 | **修回放 yaw 统计**：`net turn` 原为 `heading[末]−heading[首]`（角度已 wrap 到 ±180°），转过半圈以上完全错（实测转 4 圈 1466° 却报 +26°）→ 改为逐点累加 wrap 增量；新增 **`NET rate`**（累计转角/时长，判断净转速的唯一正确指标）；yaw rate 从世界系改为**机体系**（与指令同坐标系），同时打印世界系交叉检查 | 无需重训 |
| **v1.0.26** | 2026-10-22 | 回放新增**播放速率统计**：`[play] timing: sim=..s wall=..s -> playback N% of real time`。用于解释"数据转得对但动画看着慢"——实测 1 环境 headless 只有 **61%** 实时，`--real-time` 只能压慢、无法加速；推荐用 `--video` 录帧后按 50 fps 播放，或直接看 `NET rate` | 无需重训 |
| **v1.0.27** | 2026-10-22 | 修 `RecordVideo` 缺 `fps`：gymnasium 默认 30 fps，而 1 帧 = 1 控制步 = 20 ms，导致录像慢 1.67 倍；`play.py` / `train.py` 现显式传 `fps = 1/step_dt`（50）。实测 200 步 → 视频 4.02 s，播放即真实速度（解决"动画看着比数据慢"） | 无需重训 |
| 0–3 | 髋关节**位置**目标（相对名义姿态） | `hip_scale = 0.25` rad | 名义 ± `hip_range 0.45` |
| 4–5 | 轮**速度**目标 | `wheel_scale = 5.0` rad/s | — |

- 膝角不是动作：每控制步用牛顿迭代从实测髋角反解（热启动）
- 闭环求解：**解析雅可比 + 2 次牛顿迭代**（实测残差 6.0e-7 m，仿真真值约 7.5e-4 m）

### 3.4 观测空间（35 维，10 项）

| # | 项 | 维度 | 说明 |
|---|---|---|---|
| 1 | `projected_gravity` | 3 | 机体系重力投影 |
| 2 | `base_ang_vel` | 3 | ×0.25 |
| 3 | `base_lin_vel` | 3 | |
| 4 | `velocity_commands` | 2 | `(vx*, wz*)` |
| 5 | `hip_pos_rel_nominal` | 4 | 髋角 − 名义 |
| 6 | `knee_pos_rel_solved` | 4 | 实测膝角 − 目标膝角 |
| 7 | `wheel_vel` | 2 | ×0.1 |
| 8 | `last_action` | 6 | 上一步动作（未缩放） |
| 9 | `solved_leg_state` | 4 | `(L0_l, φ_l, L0_r, φ_r)`，**目标值** |
| 10 | **`leg_rates`** | 4 | `(dL0_l, dφ_l, dL0_r, dφ_r)`，**实测一步差分** |

- 归一化：`EmpiricalNormalization`（`obs_normalization=True`，actor/critic 各自一份）
- `leg_rates` 是新加的：MLP 无记忆，没有腿部速度就无法对姿态环做阻尼
- 复位后第一步 `leg_rates` 强制为 0（避免姿态跳变产生假速度）

### 3.5 奖励（24 项）

| # | 项 | 权重 | 形状 | 实现 |
|---|---|---|---|---|
| 1 | `alive` | **+0.5** | 常数 | `is_alive` |
| 2 | `fall_penalty` | **−50** | 常数 | `fall_penalty` |
| 3 | `base_height` | **−4.0** | L1 | `base_height_abs`（目标 0.189775） |
| 4 | `pitch` | **−4.0** | L1 | `pitch_abs` |
| 5 | `roll` | **−1.5** | L1 | `roll_abs` |
| 6 | `roll_straight` | **−2.5** | L1 × 门控 | `roll_when_straight_abs`（`wz_scale=2.5`） |
| 7 | `roll_rate` | −0.1 | L2 | `ang_vel_x_l2` |
| 8 | `lin_vel_z` | −2.0 | L2 | `lin_vel_z_l2` |
| 9 | **`track_lin_vel_x`** | **+8.0** | exp | `track_lin_vel_x_exp`（`std=0.15`）— v1.0.8 提高权重、收窄核 |
| 10 | `track_ang_vel_z` | **+3.0** | exp | `track_ang_vel_z_exp`（`std=0.15`）— v1.0.8 收窄核 |
| 11 | `heading_error` | **−0.6** | L2 | `heading_error_l2`（v1.0.4 由 −0.4 加严） |
| 11b | **`track_lin_vel_x_l1`** | **−2.0** | L1 | `\|vx − vx_cmd\|` — **恒定梯度**，把最后几 cm/s 压下去（v1.0.8 新增） |
| 11c | **`track_lin_vel_x_l2`** | **−6.0** | L2 | `(vx − vx_cmd)²` — 大误差主导，梯度随误差增长（v1.0.8 由 −3.0 加严） |
| 11d | **`track_ang_vel_z_l1`** | **−1.0** | L1 | `\|ωz − wz_cmd\|`，yaw 的恒定梯度项（v1.0.8 新增） |
| 11e | **`track_ang_vel_z_l2`** | **−2.0** | L2 | `(ωz − wz_cmd)²`（v1.0.8 由 −1.0 加严） |
| 12 | `lin_vel_y` | −0.5 | L2 | `lin_vel_y_l2` |
| 13 | **`wheel_torques`** | **−0.05** | L2（**死区 1.0 N·m**） | `wheel_torques_excess_l2`，**只作用于两个轮关节** |
| 14 | `action_rate` | −4.0e-3 | L2 | `action_rate_l2`（一阶差分） |
| 14b | **`action_jerk`** | **−0.5** | L2 | `(a[t] − 2a[t−1] + a[t−2])²` — 只罚**抖动**，平滑运动免费（v1.0.5） |
| 15 | `joint_torques` | −2.0e-5 | L2 | `joint_torques_l2`（全部 10 关节） |
| 16 | `joint_acc` | **−1.0e-6** | L2 | `joint_acc_l2`（v1.0.5 由 −1e-7 提高 10 倍） |
| 17 | `wheel_spin` | −3.0e-5 | L2 | `wheel_spin_l2` |
| 18 | **`wheel_axle`** | **−8.0** | L1（米） | `leg_axle_skew_abs`：`\|L0_l sinφ_l − L0_r sinφ_r\|`，**约束轮轴不歪**（v1.0.1 新增） |
| 19 | `closure_residual` | **−1000** | L2 | `closure_residual_l2`（硬约束） |

#### 关键设计决定（含"故意不约束"清单）

**① `pitch` / `roll` / `base_height` 用 L1 而不是 L2**

`|x|` 从任何距离都给出恒定拉力；`x²` 在 0 附近梯度消失、远处爆炸。
实测 L2 版本会让机器人**蹲低 5 cm**。

**② `roll` 拆成「全局 + 直行加权」两项**

两轮车过弯需要侧倾（`atan(vω/g)` ≈ 2.6°），但常驻 10~20° 侧倾是白费力。
门控用的是**指令**的 `wz*`，不是实测角速度（否则会被侧倾本身骗过去）。

**③ `wheel_torques` 带 1.0 N·m 死区**

轮子是**唯一纵向执行器**。匀速 2 m/s 只需约 0.2 N·m，但加速需约 0.9 N·m、
指令从 +2 跳到 −2 会到约 1.7 N·m。无死区会惩罚正常行驶；
死区之后，平滑推进免费，**只罚姿态纠正的力矩尖峰（2~3 N·m）**。

**④ 故意不给「左右腿长差 L0_l − L0_r」加约束** ⚠️ 设计决定，不是遗漏

`L0` 左右不一致是**合法且必需**的自由度：

| 场景 | 需要的左右腿差 |
|---|---|
| 侧倾坡面（左右轮高差） | **必须**，否则机体无法保持水平 |
| 单轮压上台阶 | **必须** |
| 过弯侧倾 | 需要（轻微） |

所以**永远不要**加 `\|L0_l − L0_r\|` 这类惩罚项。目前 `L0` 只被 `roll` / `roll_straight`
**间接**约束（因为腿长差会产生机体 roll），没有直接约束。

> `wheel_axle`（约束轮轴对齐）**不会**误伤坡面工况 —— 实测：左右腿长差 20 mm、
> 腿竖直（φ=0）时，轮轴歪斜 = **0 mm，零惩罚**；只有"腿长差 + 腿整体前倾"才会产生
> 真实的几何歪斜（φ=10°、差 20 mm → 3.5 mm → 0.028/步，仅占 `alive` 的 5.6%）。
> 这是真实几何，不是耦合副作用。

**⑤ 目前故意不约束的其他量**

| 量 | 为什么不约束 |
|---|---|
| 共同 φ（两腿一起前倾） | 这就是倒立摆移 CG 加速的机制，必须自由 |
| 左右腿长差 `ΔL0` | 坡面/台阶必需（见 ④） |
| 髋关节绝对角度 | 只约束腿长/腿角的几何量，不限制关节怎么摆 |

### 3.6 速度指令

| 项目 | 值 | 可否调整 |
|---|---|---|
| 速度 `vx*` | **[−1.2, +1.2] m/s**（**含倒车**，v1.0.10） | ✅ `INFANTRY_RL_VX_MIN` / `INFANTRY_RL_VX_MAX` |
| **航向目标** | **固定 0（转弯关闭）** | ✅ `INFANTRY_RL_ENABLE_TURNING=1` 恢复 −π…+π |
| `wz*` 命令 | `clip(1.5 · wrap(0 − ψ), ±2.5)` | 改 `HEADING_STIFFNESS` / `ANG_VEL_Z_RANGE` |
| `wz` 的 **±2.5** | **只作为纠正漂移的钳位**，不是转向指令 | 改 `ANG_VEL_Z_RANGE` |
| 指令保持 | 每 5~10 s 重新采样 | 改 `resampling_time_range` |
| 站住不动的环境比例 | 10%（刹车样本） | 改 `REL_STANDING_ENVS` |

**`wz*` 不是直接采样的**，而是由航向内环算出：

```
wz* = clip( 1.5 · wrap(ψ_target − ψ), −2.5, +2.5 )
```

v1.0.4 起 `ψ_target` 恒为 0，所以 `wz*` **只用来抵消航向漂移**（`vx=0` 时也照样纠偏）；
主动转弯、原地旋转都不在训练范围内。

**速度上限的物理依据**：

| 限制 | 计算 | 结果 |
|---|---|---|
| 轮速上限 | 61.471 × 0.068 | 4.18 m/s |
| 轮推力 | 2 × 3.69 / 0.068 | 108 N → 8.5 m/s² |
| 轮胎抓地 | μg ≈ 0.8 × 9.81 | 7.8 m/s² |

→ **1.2 m/s 只用轮速上限的 29%**（17.6 rad/s），留足打滑/坡度/力矩上限余量。

### 3.7 终止条件

| 项 | 条件 |
|---|---|
| `time_out` | 20 s（1000 个控制步） |
| `bad_orientation` | 倾斜角 > **1.2 rad（69°）** |
| `base_too_low` | 高度 < **0.11 m**（名义 0.19） |

### 3.8 仿真与场景

| 项目 | 值 |
|---|---|
| 环境数 | **1024** |
| 环境间距 | 2.5 m |
| 物理步长 | 0.005 s（200 Hz） |
| 控制频率 | **50 Hz**（`decimation=4`） |
| `gpu_max_rigid_patch_count` | **5 × 2¹⁵**（Isaac Lab 默认） |
| PhysX fabric | 默认开启（可用 `INFANTRY_RL_USE_FABRIC=0` 关闭） |

### 3.9 PPO 训练

| 参数 | 值 |
|---|---|
| `num_steps_per_env` | **48** |
| 每轮样本 | 48 × 1024 = **49,152** |
| `num_mini_batches` | 4（每批 12,288） |
| `num_learning_epochs` | **8** → 32 次梯度更新/轮 |
| `learning_rate` | 1e-3，`schedule="adaptive"`，`desired_kl=0.01` |
| `gamma` | **0.995**（≈ 4 s 视界） |
| `lam` | 0.95 |
| `clip_param` | 0.2 |
| `entropy_coef` | **0.003**（v1.0.7；0.001 时 std 掉到下限、探索停摆） |
| **policy std 硬限幅** | **[0.06, 0.4]**（`train.py::_clamp_policy_std`；v1.0.7 下限 0.03→0.06，防止再次彻底确定化） |
| `value_loss_coef` | 1.0，`use_clipped_value_loss=True` |
| `max_grad_norm` | 1.0 |
| `clip_actions` | 4.0 |
| actor / critic | **[128, 128, 64]**，ELU |
| 高斯标准差 | `init_std = 0.5`（状态无关，6 维向量） |
| `save_interval` | **50** 轮 |
| 实测每轮耗时 | **约 1.94 s**（1024 环境 / 128 网络） |
| 吞吐 | 约 25,300 样本/秒 |

## 4. 文件清单与职责

### 4.1 本版本改动的文件

| 文件 | 改动 |
|---|---|
| `isaac_ws/isaac_ws_infantry/mdp/actions.py` | 新增 `update_leg_rates()`；3 处 NaN 防护（`process_actions` / `update_knee_targets` / `update_leg_rates`） |
| `isaac_ws/isaac_ws_infantry/mdp/rewards.py` | 新增 `wheel_torques_excess_l2()`（带死区） |
| `isaac_ws/isaac_ws_infantry/mdp/__init__.py` | 导出 `leg_rates` / `wheel_torques_excess_l2` |
| `isaac_ws/isaac_ws_infantry/tasks/locomotion/locomotion_env_cfg.py` | 观测加 `leg_rates`；奖励加 `wheel_torques` 并调整 5 项权重；速度范围 ±2.0 可配置；打印指令范围；patch 缓冲回默认 |
| `isaac_ws/isaac_ws_infantry/tasks/locomotion/agents/rsl_rl_ppo_cfg.py` | γ 0.995、epochs 8、`save_interval` 50、网络 128-128-64 |
| `isaac_ws/isaac_ws_infantry/tasks/balance/balance_env_cfg.py` | patch 缓冲回默认 |
| `isaac_ws/scripts/train.py` | `--resume` 时自动匹配 checkpoint 的网络结构 |
| `isaac_ws/scripts/play.py` | 同样自动匹配；输出腿部指标（`hip dev` / `φ` / `dφ` / `L0 差`） |
| `TRAINING_GUIDE.md` | 重写指令部分；新增 §6 规则、§7.10 腿主导平衡、§7.11 崩溃恢复 |
| `VERSION.md` | 本文件 |

### 4.2 核心代码结构

```
isaac_ws/
├── isaac_ws_infantry/
│   ├── robot_model.py           # 机器人模型：关节顺序/限位、名义姿态、批量求解器
│   ├── kinematics/
│   │   ├── planar_leg.py        # 单腿平面五连杆：定位/逆解/解析雅可比牛顿求解
│   │   ├── urdf_tree.py         # URDF 解析 + 精确 FK（所有模型的验证基准）
│   │   ├── reference_impl.py    # numpy 参考实现（仅测试对照，运行时不使用）
│   │   ├── check_planar_leg.py  # 校验：模型 vs URDF 精确 FK
│   │   └── check_reference_impl.py  # 校验：参考实现 vs URDF 精确 FK
│   ├── mdp/
│   │   ├── actions.py           # FiveBarJointAction：6 维动作 → 髋角+轮速+膝目标
│   │   ├── observations.py      # 10 个观测项
│   │   ├── rewards.py           # 全部自定义奖励项
│   ├── robots/infantry_rl.py    # 执行器、关节名、初始姿态
│   ├── terrains/                # 本地地面平面（离线，不拉 Nucleus 资产）
│   └── tasks/
│       ├── common/              # ★ 两个任务共享：FlatScene/RobotSceneCfg、ActionsCfg、
│       │                        #   TerminationsCfg、FiveBarEnv 基类
│       ├── balance/             # Balance 任务（旧任务：原地保持航向/位置，观测 33 维）
│       └── locomotion/          # **Locomotion 任务（主任务，观测 35 维）**
└── scripts/
    ├── train.py                 # 训练入口（--resume_latest / 自动匹配网络结构）
    ├── play.py                  # 回放/评估（--cmd_vx / --cmd_heading_deg / 腿部指标）
    ├── check_robot.py           # 不训练也能验证闭环与站立
    └── tools/                   # 一次性工具（资产准备 / 关节映射 / 工作空间分析）
```

### 4.3 配置与数据

```
isaac_ws/config/
├── joint_map.json       # URDF 关节名 ↔ 内部索引
├── leg_model.json       # 五连杆几何（A/B/P/Q/S/T 坐标）
├── nominal_pose.json    # 名义站立位（L0=0.249775, φ≈0）
├── feasible_poses.json  # 可达工作空间采样
└── proxy_geometry.json  # 代理碰撞体
```

## 5. 兼容性矩阵

| 变化 | 旧 checkpoint 能否续训 |
|---|---|
| 奖励权重 | ✅ 能 |
| `gamma` / `lam` / epochs / lr | ✅ 能（最优性会变，但结构兼容） |
| 速度范围 | ✅ 能 |
| 环境数 | ✅ 能 |
| **观测维度或顺序** | ❌ **不能**（网络输入层尺寸不符） |
| **网络 `hidden_dims`** | ⚠️ **交给我们自己的自动适配**：脚本会读取 checkpoint 的实际宽度并覆盖配置 |
| **`obs_normalization` 开关** | ❌ 不能（state_dict 会多/少 `obs_normalizer._mean/_var/_std/count`） |
| 任务（Balance ↔ Locomotion） | ❌ 不能（观测 33 vs 35 维） |

> `train.py` / `play.py` 会自动从 checkpoint 读回网络宽度，所以**改 `hidden_dims` 后
> 旧权重仍可回放/续训**（会打印一行 `hidden_dims: config [...] -> checkpoint [...]`）。
> 但**观测布局改了就必须从零开始**。

## 6. 环境变量开关

全部是**进程级**（只在该 cmd 窗口有效，`set VAR=` 或关闭窗口即失效）：

| 变量 | 默认 | 作用 | 适用任务 |
|---|---|---|---|
| `INFANTRY_RL_VX_MIN` | `0.2` | `vx` 指令下界（m/s），负值可恢复后退（v1.0.6） | Locomotion |
| `INFANTRY_RL_VX_MAX` | `1.2` | `vx` 指令上界（m/s）（v1.0.6） | Locomotion |
| `INFANTRY_RL_ENABLE_TURNING` | `0` | 设为 `1` 恢复随机航向目标（转弯） | Locomotion |
| `INFANTRY_RL_USE_FABRIC` | `1` | 设 `0` 关闭 PhysX fabric | 两者 |
| `INFANTRY_RL_PHYSX_POS_ITERS` | 引擎默认 | PhysX 位置迭代次数 | 两者 |
| `INFANTRY_RL_ASSET` | 内置 | 换机器人资产路径 | 两者 |
| `INFANTRY_RL_HEADING_CONTROL` | `0` | 外挂航向环（Balance） | Balance |
| `INFANTRY_RL_POSITION_CONTROL` | `0` | 外挂位置环（**实测会摔，勿用**） | Balance |
| `INFANTRY_RL_NO_HEADING_OBS` | `0` | 去掉航向观测 | Balance |
| `INFANTRY_RL_NO_POSITION_OBS` | `0` | 去掉位置观测 | Balance |
| `INFANTRY_RL_TARGET_X` | `0` | 位置环目标 x | Balance |

## 7. 已知限制

### 7.1 单 GPU，无法多卡

- 只有 1 块 RTX 4060 Laptop（8 GB）
- Windows 版 PyTorch **没有 NCCL**（只有 gloo），而 rsl_rl 硬编码 `backend="nccl"`
- 结论：**本机不可能多 GPU 并行**

### 7.2 环境数上限 1024

| 环境数 | 结果 |
|---|---|
| 1 | ✅ 可用（只用于回放） |
| 2 ~ 256 | ❌ 卡死在 `gym.make()` |
| **1024** | ✅ **推荐** |
| 2048 | ⚠️ 能跑，约 24 分钟崩 |
| 4096 | ❌ 约 5 分钟崩 |

### 7.3 外挂位置环不可行

实测：航向环有效（差动轮速与平衡正交）；**位置环会摔机器人**
（共模轮速就是平衡输入，无法同时用于位置控制）。位置跟踪只能靠训练。

## 8. 已知问题：PhysX GPU 崩溃（未解决）

### 8.1 症状

```
RuntimeError: CUDA error: an illegal memory access was encountered
RuntimeError: CUDA error: an illegal instruction was encountered
[omni.physx.fabric.plugin] CUDA error: ... DirectGpuHelper.cpp: 563-587
[omni.physx.plugin] PhysX warning: PxgCudaMemoryAllocator.cpp, LINE 68
```

**Python traceback 指向哪一行都是假线索**（CUDA 异步，报错点只是"下一个碰 GPU 的地方"）。
真正出错点是 PhysX GPU 分配器。崩溃时显存通常只用 1 GB / 8 GB。

### 8.2 实测存活时间

| 配置 | 存活 |
|---|---|
| 4096 环境 | ~5 分钟 |
| 2048 环境 | ~24 分钟 |
| 1024 环境 + 256 网络 + patch 缓冲翻倍（未重启） | 2.9 分钟 |
| 1024 环境 + 128 网络 + patch 缓冲默认（重启后） | **16.2 分钟** |

同一次开机内反复崩溃，存活时间会**持续缩短**（实测 9 → 3.1 → 2.9 分钟）。

### 8.3 已做的缓解

1. `save_interval = 50` → 崩溃最多丢 2 分钟，配合 `--resume` 效率约 90%
2. 网络回到 128-128-64（单轮 2.4 → 1.94 s，GPU 压力下降）
3. `gpu_max_rigid_patch_count` 回到默认 `5*2**15`
4. 3 处 NaN 防护，NaN 绝不进入 PhysX 关节目标
5. 建议：**长时间训练前重启电脑**

### 8.4 还可尝试（按顺序）

```cmd
set INFANTRY_RL_USE_FABRIC=0     :: 关掉报错的 fabric 插件，代价约 4% 速度
set INFANTRY_RL_VX_MAX=0.8       :: 降低速度/冲击负载
--num_envs 768                   :: 降低负载（512/768 未验证）
```

## 9. 如何确认当前代码版本

逐项核对，全部对上才是 v1.0.0：

```cmd
cd /d D:\Robomaster\RL
```

| # | 检查项 | 期望输出 |
|---|---|---|
| 1 | `git rev-parse --short HEAD` | `f99cddd` |
| 2 | 观测维度 | 启动日志里 `Active Observation Terms in Group: 'policy' (shape: (35,))` |
| 3 | 动作维度 | `Active Action Terms (shape: 6)` |
| 4 | 奖励项数 | `Reward Manager: contains 22 active terms` |
| 5 | 指令范围 | `[cfg] command ranges : vx +/-2.00 m/s ...` |
| 6 | 网络结构 | `Actor Model:` 里 `Linear(in_features=35, out_features=128)` |
| 7 | 归一化 | `(obs_normalizer): EmpiricalNormalization()` |

最省事的办法：跑一条 1 轮的测试命令（**记得加 `--run_name ZZZTEST`，跑完删目录**）：

```cmd
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 1024 --max_iterations 1 --no_code_state --run_name ZZZTEST
```

看启动日志里第 2~6 项。

## 10. 变更记录（后续版本往这里加）

> 格式：`## vX.Y.Z — YYYY-MM-DD` + 一句话定位 + 差异表 + 是否需要重训

| 版本 | 日期 | 一句话 | 是否需重训 |
|---|---|---|---|
| | | | |
