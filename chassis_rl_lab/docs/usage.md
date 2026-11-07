# chassis_rl_lab 使用手册（训练 / 续训 / 回放 / 导出 / 数据）

> 所有命令都在**项目根**执行。8 GB 显存建议 `--num_envs 256`（1024 也能跑，显存更紧）。
>
> 当前是**三个任务**（可按需组合训练）：
> - **阶段1** `Chassis-Stand-Up-v0`：从倒地 → 用轮子起立 → 保持平衡；**屏蔽髋关节输出**（髋 PD 锁 squat、腿不动，策略实际只控 2 个轮子）；
> - **恢复任务** `Chassis-Stand-Up-Recovery-v0`：**随机方向倒地 → 髋自由起立 → 保持平衡**，不跟踪高度/腿长（从阶段1续训用）；
> - **阶段2** `Chassis-Stand-Up-Height-v0`：在恢复任务基础上**按指令改变机体高度**（高度指令随机 + 高度奖励）。
>
> 三个任务观测（25 维）/动作（6 维接口）完全一致，可直接互相 `--resume`。

## 0. 先进入项目根（以下所有命令的前提）

```powershell
cd D:\Robomaster\RL\chassis_rl_lab
```

## 1. 最常用三条（先跑起来）

```powershell
# 阶段1训练（数据会自动存到 data\<时间戳>_<run名>\）
D:\miniconda\envs\RM\python.exe scripts\train.py --headless --num_envs 1024 --max_iterations 1500 --run_name stage1

# 回放最新模型（默认带 IMU 噪声、带平衡辅助）
D:\miniconda\envs\RM\python.exe scripts\play.py --headless --num_envs 1 --episodes 5 --checkpoint -1

# 看训练曲线（TensorBoard）
D:\miniconda\envs\RM\python.exe -m tensorboard.main --logdir logs\rsl_rl
```

## 2. 两阶段训练流程（核心）

### 2.1 阶段1：倒地起立 + 平衡（`Chassis-Stand-Up-v0`）

```powershell
D:\miniconda\envs\RM\python.exe scripts\train.py --headless --num_envs 1024 --max_iterations 1500 --run_name stage1
```

- 环境：从倒地姿态（squat 腿型）开始；动作保持 6 维接口但**前 4 维髋输出被屏蔽**（髋 PD 锁 squat、腿不动），策略实际只控 **2 个轮力矩（≤±4.92 Nm）**；
- 高度指令固定在名义高度（0.1716 m），高度奖励权重为 0 → 只学"站起来并保持平衡"；
- **倒地方向课程 + 分阶段辅助退火**（每阶段 300 iter）：
  - iter 0–300：pitch **正**方向倒地，辅助力在 300 iter 内 1→0 退火；
  - iter 301–600：pitch **负**方向倒地，辅助力在 300 iter 内 1→0 退火；
  - iter >600：**无辅助力**，pitch 正/负每 episode **随机**。

**阶段1 成功判据**（TensorBoard）：

- `Train/mean_episode_length` → 接近 **1000**（20 s 全程都在平衡）；
- `Episode_Termination/failed_to_recover` → 从 1.0 降到 **0**；
- `Episode_Reward/upright` 上升；
- **iter > 600（无辅助）后** `mean_episode_length` 不回落，两个方向都能站起来。

### 2.2 阶段2：在阶段1基础上按指令变高度（`Chassis-Stand-Up-Height-v0`）

```powershell
# 1) 先看阶段1最新 checkpoint 名（也可用 model_<数字>.pt 指定最好的那次）
dir logs\rsl_rl\chassis_rl_stand_up\*stage1\model_*.pt

# 2) 续训（--max_iterations 必须大于阶段1已训迭代数）
D:\miniconda\envs\RM\python.exe scripts\train.py --headless --task Chassis-Stand-Up-Height-v0 --resume --load_run ".*stage1" --load_checkpoint model_1000.pt --num_envs 1024 --max_iterations 1000 --run_name stage2_height
```

- 阶段2 **放开髋**（4 髋力矩 ≤±20 Nm + 2 轮力矩全部生效）；起始改为"收腿倒地"（髋收到最短 q=0，开局 0.4 s 收腿稳定）；高度指令每 episode 在 **0.172~0.270 m** 随机（带 curriculum，续训时从续训点起慢慢放开）+ 高度跟踪奖励 `exp(-(h-h_cmd)²/0.001)`；
- 观测/动作维度与阶段1相同，所以 `--resume` 直接加载阶段1网络继续训练（迭代数接着算）；
- 训练数据同样自动落到 `data/<时间戳>_stage2_height/`。

**阶段2 成功判据**：`Episode_Reward/base_height_track` 上升（最高 2.0）、`failed_to_recover` 保持 0、`mean_episode_length` 接近 1000。

### 2.3 恢复任务：随机方向倒地自起（`Chassis-Stand-Up-Recovery-v0`）

只练"**随机方向倒地 → 站起来 → 保持平衡**"，**不跟踪高度/腿长**（高度指令固定、高度奖励权重 0）。
髋关节自由（6 维力矩全部生效），用于从上个阶段继续训练"倒地自起"：

```powershell
# 从阶段1的 model_750 继续训练
D:\miniconda\envs\RM\python.exe scripts\train.py --headless --task Chassis-Stand-Up-Recovery-v0 ^
    --resume --load_checkpoint "logs\rsl_rl\chassis_rl_stand_up\2026-11-06_22-29-04_stage1\model_750.pt" ^
    --num_envs 256 --max_iterations 750 --run_name recovery
```

- 起始：倒地姿态（pitch **正/负每 episode 随机**）+ 髋收到最短腿长（q=0）+ 开局 0.4 s 收腿稳定；
- 奖励：只有起立/平衡/速度/平滑等项，**无高度跟踪**；
- 无辅助力（阶段1 已学会轮平衡）；
- 成功判据：`Train/mean_episode_length` → 1000、`Episode_Termination/failed_to_recover` → 0。

### 2.4 评估不同阶段/效果

```powershell
# 阶段1（倒地起立/平衡）：纯策略（关辅助）+ 带噪声
D:\miniconda\envs\RM\python.exe scripts\play.py --num_envs 2 --episodes 5 --no-assist --checkpoint -1

# 阶段2（高度跟踪）：纯策略 + 曲线（打印 pitch/roll/高度）
D:\miniconda\envs\RM\python.exe scripts\play.py --task Chassis-Stand-Up-Height-v0 --num_envs 1 --episodes 1 --no-assist --trace --checkpoint -1

# 指定某个阶段的 run
D:\miniconda\envs\RM\python.exe scripts\play.py --load_run ".*stage1" --checkpoint -1 --episodes 5 --no-assist
D:\miniconda\envs\RM\python.exe scripts\play.py --task Chassis-Stand-Up-Height-v0 --load_run ".*stage2_height" --checkpoint -1 --episodes 5 --no-assist
```

### 2.5 查看指定的 run / 指定的 .pt 文件

**① 列出所有 checkpoint（按时间排序）**

```powershell
Get-ChildItem logs\rsl_rl\chassis_rl_stand_up -Recurse -Filter "model_*.pt" |
  Sort-Object LastWriteTime | Select-Object FullName, LastWriteTime
```

**② 看某个 run 里有哪些 .pt**

```powershell
dir logs\rsl_rl\chassis_rl_stand_up\2026-11-06_22-29-04_stage1\model_*.pt
```

**③ 回放"指定 run 的最新 checkpoint"**

```powershell
D:\miniconda\envs\RM\python.exe scripts\play.py --load_run ".*22-29-04_stage1" --checkpoint -1 --episodes 5 --no-assist
```

**④ 回放"指定 run 的指定 .pt"**（run 用 `--load_run` 正则，文件名用 `--checkpoint`）

```powershell
D:\miniconda\envs\RM\python.exe scripts\play.py --task Chassis-Stand-Up-v0 ^
  --load_run ".*22-29-04_stage1" --checkpoint model_750.pt --episodes 5 --no-assist
```

**⑤ 直接用 .pt 的完整路径**（不依赖 run 匹配，最稳）

```powershell
D:\miniconda\envs\RM\python.exe scripts\play.py --task Chassis-Stand-Up-v0 ^
  --checkpoint "logs\rsl_rl\chassis_rl_stand_up\2026-11-06_22-29-04_stage1\model_750.pt" ^
  --episodes 5 --no-assist
```

**⑥ 看 .pt 里存了什么（迭代数 / 网络键名）**

```powershell
D:\miniconda\envs\RM\python.exe -c "import torch; d=torch.load(r'logs\rsl_rl\chassis_rl_stand_up\2026-11-06_22-29-04_stage1\model_750.pt', map_location='cpu', weights_only=False); print(list(d.keys())); print('iter =', d.get('iter'))"
```

**⑦ 批量评估（headless）**

```powershell
D:\miniconda\envs\RM\python.exe scripts\play.py --headless --num_envs 16 --episodes 20 ^
  --task Chassis-Stand-Up-v0 --checkpoint "logs\rsl_rl\chassis_rl_stand_up\2026-11-06_22-29-04_stage1\model_750.pt"
```

> 说明：
> - `--checkpoint -1` = 在匹配 `--load_run` 的**最新 run** 里取**迭代数最大**的 .pt；
> - `--load_run` 是正则；**每次启动训练都会新建 run 目录**，同一个正则可能匹配到多个 run（取按名排序的最后一个）。要精确回放请写带时间戳的 run 名，或直接用方式⑤的完整路径；
> - 回放的 `--task` 要与 checkpoint 的训练任务一致（三种任务观测 25/动作 6 维度兼容，但物理/控制语义不同；旧的 2 维 action 检查点不能用）；
> - GUI 看效果：去掉 `--headless`，加 `--num_envs 1 --episodes 1`；`--trace` 会打印 pitch/roll/base_z 以及 Height 任务的 `h_cmd`（指令高度）/`h_err`（误差）。
> - **回放/评估时高度指令会自动随机**（播放会关闭 curriculum；否则新进程的指令恒为名义高度 0.1716，看起来"机器人不会变高度"）。部署 ONNX 时也必须由外部给高度指令（obs 中 `commands[2] = height_cmd × 5`），否则策略只会保持默认高度。

## 3. 指令替换对照表

### 3.1 训练 / 续训

| 我想… | 加入/替换的参数 | 说明 |
|---|---|---|
| 换阶段 | `--task Chassis-Stand-Up-v0` / `--task Chassis-Stand-Up-Height-v0` | 阶段1 / 阶段2 |
| 换并行环境数 | `--num_envs 256` / `--num_envs 1024` | 8 GB 显存 256 稳 |
| 换训练长度 | `--max_iterations 1500` | 默认取 agent cfg |
| 区分实验 | `--run_name stage1` / `--run_name test_v2` | 日志目录后缀（带时间戳） |
| 数据保存频率 | `--export_interval 100` | 训练中每 N 轮导出一次累计数据（默认 100；0=只在结束时导出） |
| 换实验目录 | `--experiment_name chassis_rl_stand_up2` | 默认 `chassis_rl_stand_up` |
| 换随机种子 | `--seed 17`（默认已改为 17） | 之前 seed=42 会在 1024 envs、iter 169 触发可复现崩溃 |
| 改保存间隔 | `--save_interval 100` | 默认 50 |
| 不导出数据 | `--no_data_export` | |
| 换数据目录 | `--data_root D:\rl_data` | 默认 `<项目根>\data` |
| 续训最新 | `--resume` | 自动找该实验最新 run 的最新 checkpoint |
| 阶段2 续阶段1 | `--task Chassis-Stand-Up-Height-v0 --resume --load_run ".*stage1" --load_checkpoint model_1000.pt` | `--max_iterations` 要大于阶段1已训迭代数 |
| 续训指定文件 | `--resume --load_checkpoint <完整路径>` | |

### 3.2 回放（看不同效果）

| 我想看… | 参数 |
|---|---|
| 默认效果（带噪声+辅助） | 什么都不加 |
| **纯策略**（判断真学会没有） | `--no-assist` |
| 理想观测（无噪声对照） | `--no-noise` |
| 逐步姿态曲线（pitch/roll/高度） | `--trace` |
| 指定 run 的最新 .pt | `--load_run ".*22-29-04_stage1" --checkpoint -1` |
| 指定 run 的指定 .pt | `--load_run ".*22-29-04_stage1" --checkpoint model_750.pt` |
| 指定 .pt 完整路径 | `--checkpoint "logs\...\model_750.pt"`（不依赖 run 匹配） |
| 看"变腿高"明显起伏 | `--height_sweep --sweep_interval 100`（Height 任务：每 100 步在 0.172↔0.270 切换） |
| 量化高度跟踪误差 | `... tools\check_height_tracking.py --checkpoint <pt> --num_envs 3` |
| 指定阶段2任务 | `--task Chassis-Stand-Up-Height-v0` |
| 统计数 / 单回合步数 | `--episodes 5` / `--steps 500` |
| 开窗口 | 去掉 `--headless` |

### 3.3 导出 / 数据 / 自检

| 我想… | 命令 |
|---|---|
| 导出 ONNX（最新，自动按任务取） | `... scripts\export_onnx.py --checkpoint -1` |
| 导出阶段2模型 | `... scripts\export_onnx.py --task Chassis-Stand-Up-Height-v0 --load_run ".*stage2_height" --checkpoint -1` |
| 校验 ONNX | `... -c "import onnx;m=onnx.load(r'exports\chassis_standup_policy.onnx');onnx.checker.check_model(m);print('OK')"` |
| 导出某 run 训练数据 | `... tools\export_training_data.py --log_dir logs\rsl_rl\chassis_rl_stand_up\<run>` |
| 环境自检 | `... tools\check_env.py --num_envs 16 --steps 200`（阶段2加 `--task Chassis-Stand-Up-Height-v0`） |
| 展示用 USD | `... tools\make_display_usd.py --verify` |

## 4. 任务说明（两阶段对比）

| 项 | 阶段1 `Chassis-Stand-Up-v0` | 阶段2 `Chassis-Stand-Up-Height-v0` |
|---|---|---|
| 目标 | 倒地 → 起立 → 保持平衡 | 保持平衡 + 按指令改变高度 |
| 起始状态 | 倒地姿态（pitch 14~19°，**squat 腿型、髋被 PD 锁住**）；方向按课程切换：0–300 iter **正**、301–600 iter **负**、之后**随机** | 倒地姿态 + **髋收到最短腿长（q=0）**；开局 0.4 s「收腿稳定」后交给策略（方向同样按课程切换） |
| 动作（6 维） | 6 维接口，但**前 4 维髋输出被屏蔽**（髋 PD 锁 squat）；实际只控 2 轮力矩（×1 Nm，夹 ±4.92） | **6 维全部生效**：4 髋力矩（×2.5 Nm，夹 ±20）+ 2 轮力矩（×1 Nm，夹 ±4.92） |
| 观测（25 维） | ang_vel×0.25 → gravity → commands(固定名义高度) → hip_pos_rel → dof_vel×0.05 → actions | 同左（commands[2] 每 episode 随机高度×5） |
| 高度指令 | 固定 0.1716 m | 随机 ∈ [0.172, 0.270] m（下限=收腿最短，上限≈髋伸展0.60rad；运动学极限0.2825），curriculum 800 iter |
| 高度奖励 | 关闭（权重 0） | `exp(-(h-h_cmd)²/0.001)` 权重 2.0 |
| 终止 | 超时 + 未恢复（300 步后倾角 >0.25 rad） | 同左 |
| 平衡辅助 | 轮上 PD：每阶段（300 iter）内 1→0 退火，前 2 阶段有辅助，**iter>600 无辅助** | 关闭（阶段1 已学会轮平衡） |

## 5. 训练数据（`data/`）

**控制台输出**：训练时每轮打印精简信息——第一行是训练轮次 `[iter 当前/总]`，之后从 `Mean value loss` 开始（去掉了 Total steps/速度/耗时等标题行），并在 `Mean action std` 后追加三行：`Height target`（目标高度）、`Height now`（当前高度）、`Height error`（误差）；完整曲线照常写入 TensorBoard 与 `data/<run>/`。

训练中每隔 `--export_interval` 轮（默认 **100**）自动导出一次**从训练开始到当前**的累计数据（覆盖同一个 `data/<run>/` 目录），训练结束 / Ctrl+C 时再导出最终版：

```
data\<时间戳>_<run名>\
    metrics.csv        全部指标（iteration,tag,value 长格式）
    summary.json       每指标 first/last/min/max + checkpoint 列表 + 训练配置
    README.md          人读摘要（含最终 reward / episode 长度）
    env.yaml / agent.yaml
    plots\
        01_train.png  02_episode_reward.png  03_termination.png
        04_losses.png 05_policy_lr.png       06_perf.png
```

补导出已有 run：`... tools\export_training_data.py --log_dir <run> [--data_root D:\rl_data]`

## 6. 想改训练行为：改 env cfg 开关

| 想改变的行为 | 改哪个字段 | 默认值 → 可试 |
|---|---|---|
| **阶段1 屏蔽髋输出** | `ActionsCfg.wheel_leg.mask_hip_actions` | `True`（阶段1）；阶段2 `False` |
| 髋执行器（阶段1/2） | `SceneCfg.robot` 的 hips `stiffness/damping` | 阶段1：默认 PD `300/10`（锁 squat）；阶段2：`0/2.0`（纯力矩+被动阻尼） |
| 髋位置控制（非力矩） | `hip_control` | `"torque"` → `"position"`（+ actuators 恢复 hips 增益） |
| 髋持续力矩上限 | `robots/chassis_rl_cfg.py` 的 `HIP_EFFORT_LIMIT` | `20` Nm（峰值 40 只留档） |
| 平衡辅助开关/退火 | `balance_assist / assist_cycle_steps / assist_phases` | `True / 48*300 / 2`（每阶段 300 iter 内 1→0 退火，之后无辅助） |
| 倒地方向课程长度 | `CURRICULUM_PHASE_STEPS` | `48*300`（=300 iter/阶段；若改 num_steps_per_env 需同步换算） |
| 恢复任务方向随机 | `EventCfgRecovery.reset_root.phase_steps` | `0`（每个 episode 正/负随机，不按课程） |
| 高度区间 / curriculum | `HEIGHT_COMMAND_RANGE` / `HEIGHT_CURRICULUM_STEPS` | `(0.172, 0.270)` / `48*800`（0.270 已接近运动学极限 0.2825，再大闭链换分支） |
| 起始姿态（倒地↔站立） | `EventCfg.reset_root.pose_range.pitch` | 倒地 `(0.24,0.34)`；站立改 `(-0.05,0.05)` |
| 阶段2 倒地时腿长 | `EventCfgHeight.reset_hips`（`RETRACTED_HIPS`/`noise`） | 默认收到最短（q=0）；阶段1 用 squat ±0.02 |
| 阶段2 开局收腿稳定 | `ActionsCfgStage2.wheel_leg.settle_steps / settle_hip_kp / settle_hip_kd` | 20 步（0.4 s）/ 300 / 10；阶段1 为 0 |
| 阶段2 髋被动阻尼 | `SceneCfg.robot` 的 hips `damping` | 2.0（0 时腿塌陷会"弹飞"） |
| 站立→自然倒地→起立 | `wheel_unpowered_steps` | `0`（直接倒地）→ `40`（先自然倒地） |
| 终止阈值 | `failed_to_recover.limit_rad / grace_steps` | `0.25 rad / 300` |
| 推力扰动 | `EventCfg.push_robot` | 默认每 2–4 s 推一次；`None` 关掉 |
| PPO 超参 | `tasks/stand_up/agents/rsl_rl_ppo_cfg.py` | `num_steps_per_env=48`、`entropy_coef=0.01`、`init_std=1.8`、`clip_actions=8.0` |

> 改配置训练时换一个 `--run_name`，`logs/`、`data/` 能区分；续训按 `--load_run ".*<run名>"` 找回。

## 7. 常见坑

1. **没 cd 到项目根** → `scripts\train.py` 找不到（日志也会写错位置）。
2. `--task` 必须是注册 id（阶段1 `Chassis-Stand-Up-v0` / 阶段2 `Chassis-Stand-Up-Height-v0`），不是 run 名。
3. 回放默认**带辅助**（辅助退火从 1.0 重新开始）；判断策略真实水平请加 `--no-assist`。
4. **旧 checkpoint（2 维动作 / 旧 25 维布局）不能用于新任务**；两阶段之间可以 resume（观测 25 / 动作 6 相同，阶段1 髋输出虽被屏蔽但维度保留）。
5. 导出 ONNX 后**不要在 Isaac Sim 进程内**做 `onnx.checker`（Windows 会崩），用独立进程校验。

## 8. 稳定性：TDR 调参与 NaN 防护

### 8.1 Windows TDR（显卡超时检测与恢复）

训练偶发崩溃、事件查看器里出现 `nvlddmkm` + `LiveKernelEvent` / `LKD_0x141_Tdr` 时，
是 Windows 的 TDR 机制在"显卡内核超时（默认约 2 s）"后复位驱动。**需以管理员身份**执行
（重启后生效）：

```powershell
$k = 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers'

# 把 TDR 超时从默认 2 s 调到 60 s（避免长内核被误杀）
New-ItemProperty -Path $k -Name TdrDelay     -Value 60 -PropertyType DWord -Force
New-ItemProperty -Path $k -Name TdrDdiDelay  -Value 60 -PropertyType DWord -Force

# 可选：关闭"硬件加速 GPU 计划"（HAGS），减少 TDR 兼容性问题（1=关闭, 2=开启）
New-ItemProperty -Path $k -Name HwSchMode    -Value 1  -PropertyType DWord -Force

# 重启生效
Restart-Computer
```

还原默认（同样管理员，重启生效）：

```powershell
$k = 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers'
Remove-ItemProperty -Path $k -Name TdrDelay    -ErrorAction SilentlyContinue
Remove-ItemProperty -Path $k -Name TdrDdiDelay -ErrorAction SilentlyContinue
Set-ItemProperty    -Path $k -Name HwSchMode   -Value 2 -Type DWord   # 恢复 HAGS
```

> 说明：调大 TDR 只是避免"误杀"，**不修复真正的 GPU 挂死**；仍建议同时用 `--num_envs 256`、
> 关渲染/关其他 GPU 程序。显存 8 GB 低于 Isaac Sim 5.1 最低 16 GB，1024 envs 风险高。

### 8.2 NaN 防护（三层）

1. **环境层终止**：`TerminationsCfg.state_invalid` —— 根状态/关节状态出现 NaN/Inf，或线速度
   >30 m/s、角速度 >60 rad/s、关节速度 >200 rad/s 时，立即终止并重置该 env。
   可在 TensorBoard 看 `Episode_Termination/state_invalid` 判断 NaN/爆炸频率。
2. **obs/reward 过滤**：`chassis_rl_lab/utils/wrappers.py::SanitizeWrapper`，`train.py`/`play.py`
   自动包在环境最外层，把 obs（±100）与 reward（±1e4）里的 NaN/Inf 替换为 0 并限幅；
   内部计数 `num_obs_sanitized / num_reward_sanitized`。
3. **动作层防护**：动作项对策略动作、实测髋角、闭链 knee 目标做 `nan_to_num`，
   防止坏值进入 PhysX 造成状态污染。

> 若仍出现 NaN：先看 `Episode_Termination/state_invalid` 是否增长；可减小 `--num_envs`，
> 或在 `terminations.py::state_invalid` 里调低 `max_lin_vel / max_ang_vel / max_joint_vel` 阈值。
