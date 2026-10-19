# 训练与查看结果 —— 操作手册

本文件里的命令**全部在本机实测过**，可直接复制粘贴到 **cmd** 运行。

> **所有操作手动执行。** 项目里没有任何 `.bat` 自动化脚本 ——
> 开始、停止、续训、回放都由你自己敲命令。

主任务是 **Locomotion**（前进后退 / 转弯 / 原地旋转）。

---

## 0. 速查

```cmd
:: ───── 开始训练（从零）────────────────────────────────────
cd /d D:\Robomaster\RL
taskkill /F /IM python.exe
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 1024 --max_iterations 8000 --no_code_state

:: ───── 停止训练（另开一个 cmd 窗口）──────────────────────
taskkill /F /IM python.exe

:: ───── 续训（崩溃后 / 手动停止后）────────────────────────
cd /d D:\Robomaster\RL
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 1024 --max_iterations 3000 --no_code_state --resume --resume_latest

:: ───── 看效果（必须先停训练！）───────────────────────────
cd /d D:\Robomaster\RL
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\play.py --task Infantry-RL-Locomotion-Play-v0 --num_envs 1 --load_run .* --checkpoint model_5000.pt --cmd_vx 0.3 --cmd_heading_deg 0 --steps 3000 --real-time
```

**参数速记**

| 参数 | 含义 |
|---|---|
| `--max_iterations N` | **再训 N 轮**（不是"总共到 N 轮"，见 §1.5） |
| `--resume --resume_latest` | 从最新 checkpoint 续训 |
| `--headless` | 不开窗口（训练必加） |
| `--no_code_state` | **必加**（见规则 3） |
| `--num_envs` | 训练用 **1024**（见规则 2） |
| `--cmd_vx` | 回放时的前进指令，单位 m/s |
| `--cmd_heading_deg` | 回放时的目标航向，单位度 |
| `--steps` | 回放跑多少个控制步（50 步 = 1 秒） |
| `--real-time` | 按真实时间播放（不加则全速跑完就退出） |

---

## ⚠️ 五条必须知道的规则

### 1. 训练和回放**不能同时跑**

两个都要 Kit + 同一块 GPU（8 GB 显存）。
**回放前必须先 `taskkill /F /IM python.exe`**，否则卡死或显存不足。

### 2. `--num_envs` 用 **1024**

| 环境数 | 结果 |
|---|---|
| 1 | ✅ 可用（只用于回放看效果） |
| 2 ~ 256 | ❌ **卡死在 `gym.make()`**，无限阻塞 |
| **1024** | ✅ **推荐**，约 1.8 s/轮，长期稳定 |
| 2048 | ⚠️ 能跑，但约 **24 分钟崩一次** |
| 4096 | ❌ 约 **5 分钟崩一次**（BAR1 显存窗口耗尽） |

崩溃原因见 §7.9。**训练就用 1024。**

### 3. 训练必须加 `--no_code_state`

否则在第一次迭代前就崩：

```
UnicodeEncodeError: 'utf-8' codec can't encode characters: surrogates not allowed
```

原因：本仓库 git 索引里有个已删除的 1 MB GBK 日志 `infantry-rl/export.log`，
rsl_rl 存 git diff 时解码失败。`--no_code_state` 只跳过这个存档，
**不影响 TensorBoard 和 checkpoint**。

### 4. 同一时间只能有一个训练进程

两个训练进程会抢同一块 8 GB 显卡：**两边都变慢，而且更容易触发崩溃**。

开始前先确认：

```cmd
tasklist /FI "IMAGENAME eq python.exe"
```

输出 `信息: 没有运行的任务匹配指定标准。` 才说明干净。

### 5. `rmdir` 必须在 `taskkill` 之后执行

训练**正在跑**的时候删日志目录，会导致"进程还在、但日志/checkpoint 写不进去"的混乱状态。

```cmd
:: 正确顺序
taskkill /F /IM python.exe
rmdir /s /q "D:\Robomaster\RL\logs\rsl_rl\infantry_rl_locomotion"
```

---

## 1. 开始训练

### 1.1 从零开始（推荐首次）

```cmd
cd /d D:\Robomaster\RL
taskkill /F /IM python.exe
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 1024 --max_iterations 8000 --no_code_state
```

- **不加 `--resume`** → 从第 0 轮开始
- 每次 `train.py` 启动都会**新建一个 run 目录**（`logs\rsl_rl\infantry_rl_locomotion\<时间戳>\`），这是 rsl_rl 的正常行为
- 预计速度 **约 1.8 s/轮** → 8000 轮约 **4 小时**
- 每 **100 轮**存一次 checkpoint（`model_0.pt`、`model_100.pt`、…）

### 1.2 快速验证（约 2 分钟，只确认链路通）

```cmd
cd /d D:\Robomaster\RL
taskkill /F /IM python.exe
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 1024 --max_iterations 100 --no_code_state --run_name quick
```

`--run_name quick` 会让目录名带 `_quick` 后缀，方便和正式训练区分。

### 1.3 断点续训

```cmd
cd /d D:\Robomaster\RL
taskkill /F /IM python.exe
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 1024 --max_iterations 3000 --no_code_state --resume --resume_latest
```

- `--resume_latest` 自动找**最新 run 目录里轮次最高的** checkpoint，不用手写文件名
- ⚠️ **不要只写 `--resume`** —— Isaac Lab 自带的排序是按字符串的，会把 `model_0.pt` 当成最新，结果从没训练过的权重开始

也可以指定某个 run / checkpoint：

```cmd
:: 先看看有哪些 run
dir /ad /o-d /b "D:\Robomaster\RL\logs\rsl_rl\infantry_rl_locomotion"

:: 指定 run + checkpoint
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 1024 --max_iterations 3000 --no_code_state --resume --load_run 2026-10-19_21-30-00 --checkpoint model_5000.pt
```

### 1.4 常用参数

| 参数 | 说明 |
|---|---|
| `--task` | 训练用 `Infantry-RL-Locomotion-v0`；回放用 `Infantry-RL-Locomotion-Play-v0` |
| `--num_envs` | **1024**（见规则 2） |
| `--max_iterations` | 迭代数；1024 环境下约 1.8 s/轮 |
| `--no_code_state` | **必加**（见规则 3） |
| `--run_name` | 给本次 run 加后缀，方便区分 |
| `--seed` | 随机种子，默认 42 |
| `--headless` | 不开窗口（训练必加） |
| `--resume --resume_latest` | 从最新 checkpoint 续训 |

### 1.5 `--max_iterations` 的真实含义

它是「**再训多少轮**」，不是「总共训到多少轮」。rsl_rl 的循环：

```python
start_it = self.current_learning_iteration              # 从 checkpoint 恢复的轮次
total_it = start_it + num_learning_iterations           # ← 加，不是赋值
for it in range(start_it, total_it):
```

举例：已经训到 5000 轮，想再练 3000 轮：

```cmd
:: ✅ 正确
... --max_iterations 3000 --resume --resume_latest     → 从 5000 续到 8000

:: ❌ 错了
... --max_iterations 8000 --resume --resume_latest     → 从 5000 一路跑到 13000
```

`--resume` 会恢复：网络权重 + **Adam 优化器状态** + **已完成轮次**，是真正的续训。

---

## 2. 停止训练

### 方式 1：训练窗口按 `Ctrl+C`

### 方式 2：另开一个 cmd 窗口

```cmd
taskkill /F /IM python.exe
```

### 停止后确认

```cmd
tasklist /FI "IMAGENAME eq python.exe"
```

输出 `信息: 没有运行的任务匹配指定标准。` 就干净了。

---

## 3. 查看训练进度

### 3.1 前台看（输出直接刷在训练窗口）

每次迭代都会打印一块：

```
Learning iteration 1200/8000

      Mean reward: -3.42
      Mean episode length: 1000.00
      Mean value loss: 0.0213
      Steps per second: 2712
      Collection time: 1.520s
      Learning time: 0.271s

Episode_Reward/track_lin_vel_x: 0.8421
Episode_Reward/track_ang_vel_z: 0.7834
Episode_Reward/roll_straight: -0.0912
Episode_Reward/pitch: -0.0311
```

重点看这四项：

| 指标 | 应该怎样 |
|---|---|
| `Mean reward` | **持续上升** |
| `Episode_Reward/track_lin_vel_x` | 上升 → 前进指令跟得越来越准 |
| `Episode_Reward/track_ang_vel_z` | 上升 → 转弯跟得越来越准 |
| `Episode_Reward/roll_straight` | 从较大的负值逐步接近 0 → 侧倾在改善 |

### 3.2 TensorBoard 曲线

```cmd
cd /d D:\Robomaster\RL
D:\miniconda\envs\RM\python.exe -m tensorboard.main --logdir logs\rsl_rl\infantry_rl_locomotion
```

浏览器打开它打印的地址（通常 `http://localhost:6006`）。
可以看 `Episode_Reward/*` 每一项和 `Episode_Termination/*` 的曲线。

### 3.3 看已存了哪些 checkpoint

**列出所有 run**：

```cmd
dir /ad /o-d /b "D:\Robomaster\RL\logs\rsl_rl\infantry_rl_locomotion"
```

**列出某个 run 的全部 checkpoint**：

```cmd
dir /b "D:\Robomaster\RL\logs\rsl_rl\infantry_rl_locomotion\<run目录名>\model_*.pt"
```

**找当前最新的轮次**（跨所有 run）：

```cmd
powershell -NoProfile -Command "Get-ChildItem 'D:\Robomaster\RL\logs\rsl_rl\infantry_rl_locomotion' -Recurse -Filter model_*.pt | Sort-Object {[int]($_.BaseName -replace 'model_','')} | Select-Object -Last 1 | ForEach-Object { $_.FullName }"
```

每 **100 轮**存一次，文件名 `model_<轮次>.pt`（数字越大越新）。

---

## 4. 看效果（回放）

### ⚠️ 回放前务必先停训练

```cmd
taskkill /F /IM python.exe
```

### 4.1 开窗口实时看

```cmd
cd /d D:\Robomaster\RL
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\play.py --task Infantry-RL-Locomotion-Play-v0 --num_envs 1 --load_run .* --checkpoint model_5000.pt --cmd_vx 0.3 --cmd_heading_deg 0 --steps 3000 --real-time
```

- **不要加 `--headless`**，否则没有窗口
- `--steps 3000` = 60 秒仿真时间（控制频率 50 Hz）
- **`--real-time` 必须加**，否则几秒就把 3000 步全速跑完然后退出
- 窗口里：鼠标左键旋转、右键平移、滚轮缩放

### 4.2 固定指令评测（推荐）

`play.py` 现在支持锁死速度指令，否则会用**随机指令**，看不出单项能力：

| 想看什么 | `--cmd_vx` | `--cmd_heading_deg` |
|---|---|---|
| 站住不动 | `0` | `0` |
| 直行 0.3 m/s | `0.3` | `0` |
| 后退 0.3 m/s | `-0.3` | `0` |
| 原地左转 90° | `0` | `90` |
| 边走边转 45° | `0.3` | `45` |

```cmd
:: 例：直行
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\play.py --task Infantry-RL-Locomotion-Play-v0 --num_envs 1 --load_run .* --checkpoint model_5000.pt --cmd_vx 0.3 --cmd_heading_deg 0 --steps 3000 --real-time

:: 例：原地旋转
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\play.py --task Infantry-RL-Locomotion-Play-v0 --num_envs 1 --load_run .* --checkpoint model_5000.pt --cmd_vx 0 --cmd_heading_deg 90 --steps 1500 --real-time
```

### 4.3 `--checkpoint` 三种写法

```cmd
:: ① 纯文件名（自动在新 run 目录里找）
--load_run .* --checkpoint model_5000.pt

:: ② 相对 logs\rsl_rl\<实验名>\ 的路径
--checkpoint 2026-10-19_21-30-00\model_5000.pt

:: ③ 绝对路径
--checkpoint D:\Robomaster\RL\logs\rsl_rl\infantry_rl_locomotion\2026-10-19_21-30-00\model_5000.pt
```

> ⚠️ 有多个 run 都存了同名 checkpoint 时（比如都叫 `model_5000.pt`），
> `--load_run .*` 挑哪个是不确定的。**想精确就写 ② 或 ③。**

### 4.4 回放输出怎么读

```
[play] step    0  base_z=0.1912  x=+0.000m y=-0.000m vx=+0.015m/s  heading=  +0.02 deg ...
[play] step   50  base_z=0.1905  x=+0.152m y=+0.003m vx=+0.298m/s  heading=  -0.51 deg ...

[play] steps=3000  base_z mean=0.1898 m  max closure residual=9.7e-04 m
[play] heading: net turn=-1.10 deg  max |deviation|=1.29 deg  mean |yaw rate|=0.148 rad/s
[play] attitude: mean |pitch|=1.23 deg (max 3.10)  mean |roll|=2.05 deg (max 5.40)
[play] position: net displacement=(+8.950, +0.002) m  max |x|=8.950 m  mean |vx|=0.298 m/s
```

| 指标 | 含义 | 合格标准 |
|---|---|---|
| `base_z` | 机体离地高度 | **站立约 0.19 m**；掉到 0.14~0.15 m 就是倒了 |
| `mean \|vx\|` | 实际前进速度 | 应**接近你设的 `--cmd_vx`**（设 0.3 → 约 0.30） |
| `net displacement` 的 x | 净位移 | 直行时应**持续增大** |
| `mean \|roll\|` | 侧倾角 | 直行时应 **< 5°**（这是之前"斜着走"问题的关键指标） |
| `mean \|pitch\|` | 俯仰角 | 应 **< 5°** |
| `net turn` | 航向净变化 | 目标 0° 时应**接近 0** |
| `closure_max` | 五连杆闭环残差 | **< 1e-3 m** 说明闭环控制正常 |

### 4.5 无头模式快速评估（不开窗口，快）

```cmd
cd /d D:\Robomaster\RL
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\play.py --task Infantry-RL-Locomotion-Play-v0 --num_envs 1 --load_run .* --checkpoint model_5000.pt --cmd_vx 0.3 --cmd_heading_deg 0 --steps 2000 --headless
```

不加 `--real-time` 会全速跑完，几秒出结果，适合批量对比不同 checkpoint。

### 4.6 录制视频

```cmd
cd /d D:\Robomaster\RL
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\play.py --task Infantry-RL-Locomotion-Play-v0 --num_envs 1 --load_run .* --checkpoint model_5000.pt --cmd_vx 0.3 --cmd_heading_deg 0 --headless --video --steps 500
```

视频输出到 `logs\rsl_rl\infantry_rl_locomotion\videos\play\`。

---

## 5. 任务与当前配置

### 任务

| 任务名 | 用途 | 观测维度 |
|---|---|---|
| `Infantry-RL-Locomotion-v0` | **训练（主任务）** | 31 |
| `Infantry-RL-Locomotion-Play-v0` | 回放 | 31 |
| `Infantry-RL-Balance-v0` | 训练（旧任务：原地保持航向/位置） | 33 |
| `Infantry-RL-Balance-Play-v0` | 回放（旧任务） | 33 |

> Locomotion 和 Balance 的观测维度不同，**checkpoint 不能互相加载**。

### 当前 Locomotion 配置

| 项目 | 值 |
|---|---|
| 环境数 | 1024 |
| 动作 | 6 维（4 髋位置 + 2 轮速） |
| 观测 | **35 维** + `EmpiricalNormalization` 运行时归一化 |
| 控制频率 | 50 Hz（物理 200 Hz，decimation 4） |
| 回合长度 | 20 s |
| 每轮样本 | 48 × 1024 = **49,152** |
| 梯度更新 | 5 epochs × 4 minibatches = 20 次/轮 |
| `save_interval` | 100 轮 |
| 每轮耗时 | 约 **1.8 s** |
| 闭环求解 | 解析雅可比 + 2 次牛顿迭代 |

**指令范围**：`vx` ±0.5 m/s，`wz` ±2.5 rad/s，航向目标 −π~+π（每 5~10 s 换一次）。

**18 项奖励**：

| 项 | 权重 | 作用 |
|---|---|---|
| `alive` | +0.5 | 活着 |
| `fall_penalty` | −50 | 摔了重罚 |
| `base_height` | −4（L1） | 保持站立高度（从 −10 放宽，给腿部动作留空间） |
| **`pitch`** | **−4（L1）** | **pitch 严格归零** |
| `roll` | −1.5（L1） | 基础侧倾约束 |
| `roll_straight` | −2.5 | 直行时重罚侧倾；转弯时放开 |
| `roll_rate` | −0.1 | 抑制侧向摆动 |
| `lin_vel_z` | −2 | 不许上下颠 |
| `track_lin_vel_x` | +2.0 | 前进速度跟踪 |
| `track_ang_vel_z` | +2.5 | 角速度跟踪 |
| `heading_error` | −0.4 | 真的转到目标航向 |
| `lin_vel_y` | −0.5 | 不许横向滑 |
| **`wheel_torques`** | **−0.05** | **只罚轮子力矩 → 逼髋关节主导平衡** |
| `action_rate` | −0.004 | 从 −0.01 放宽，允许腿部快速响应 |
| `joint_torques` | −2e-5 | 从 −1e-4 大幅放宽，不压制髋关节出力 |
| `joint_acc` | −1e-7 | 平滑 |
| `wheel_spin` | −1e-4 | 轮子空转 |
| `closure_residual` | −1000 | 五连杆闭环必须闭合 |

> ⭐ **`wheel_torques` 是关键改动**（详见 §7.10）。它让"用轮子扭动来平衡"变贵，
> 于是髋关节（改变腿角 φ 移动 CG）成为更划算的平衡方式。

> 奖励设计的推导见 §7.8；腿部主导平衡的设计见 §7.10。

---

## 6. 出问题怎么办

| 现象 | 原因 / 处理 |
|---|---|
| 卡在 `Starting the simulation` 不动 | `--num_envs` 用了 2~256。改成 **1 或 1024** |
| `UnicodeEncodeError ... surrogates` | 忘了加 `--no_code_state` |
| `No checkpoints ... match` | `--checkpoint` 写错。用 §3.3 的命令列一下实际文件 |
| 训练 / 回放卡死、显存不足 | 还有一个在跑。先 `taskkill /F /IM python.exe` |
| `CUDA error: an illegal instruction` | 已知的累计型显存问题，见 §7.9。重启用 `--resume --resume_latest` 接着训 |
| 训练很慢 / 有两个进程 | `tasklist /FI "IMAGENAME eq python.exe"` 确认只有一个 |
| 日志目录空了但 GPU 还占着 | 训练运行时删了日志。`taskkill` 后重来 |
| 报 `No module named isaac_ws_infantry` | 必须在 `D:\Robomaster\RL` 下运行（先 `cd /d D:\Robomaster\RL`） |
| 续训后显示 `Learning iteration 0/...` | 漏了 `--resume`，或用了不带 `--resume_latest` 的 `--resume`（排序 bug） |

**任何卡死都先做这一步**：

```cmd
taskkill /F /IM python.exe
```

---

## 7. 实测记录（供对照）

> 下面带 "4096 环境" 的行是**早期历史数据** —— 当时 4096 还能跑（约 5 分钟崩）。
> 现在请用 **1024 环境**，速度约 1.8 s/轮（48 步 rollout）。

| 项目 | 实测值 |
|---|---|
| **1024 环境 迭代速度** | **约 1.8 s/轮**（48 步 rollout × 1024 环境） |
| **1024 环境 8000 轮预计** | 约 **4 小时** |
| 4096 环境启动时间（历史） | 约 2 s（**不卡**） |
| 4096 迭代速度（历史） | 2.65 ~ 2.7 s/迭代 |
| 100 迭代（1 环境）结果 | **摔倒**，`base_z` 从 0.19 掉到 0.145 后不动 |
| 100 迭代的闭环残差 | 6.8e-4 m（**闭环控制始终正常**） |
| 4096 训练第 100 迭代时（历史） | `Mean reward -6.4`，`Mean episode length 500.00`（满分，还没开始摔） |
| 闭环控制验证（`smoke_test.py`） | 残差 **0.750 mm**，200 步零跌倒 |
| 运动学验证（`test_leg_model.py`） | 与精确 FK 误差 **1.8e-16 m** |

> **提醒**：100 迭代 × 1 环境 = 2400 步经验，离学会平衡差 4 个数量级，摔倒完全正常。
> 参考仓库那种效果是**几万次迭代 × 几千环境**训出来的。本机请用 1024 × ≥8000。

### 怎么判断"学会了没有"

| `Mean episode length` | `base_z`（回放） | 结论 |
|---|---|---|
| 500.00 | 稳定在 0.19 m | ✅ 学会了 |
| 500.00 | 慢慢往下掉 | 🟡 在学，继续训 |
| 明显 < 500 | 掉到 0.14 ~ 0.15 m 就不动 | ❌ 还在摔，需要更多迭代 |

### 训练到什么程度才值得看

| 迭代数 | 约需时间 | 预期 |
|---|---|---|
| < 500 | < 25 min | 基本还在摔，别急 |
| 1000 ~ 2000 | 45 ~ 90 min | 可能开始"撑一下" |
| ≥ 3000 | ≥ 2.2 h | 有望站稳，这时候回放才有意义 |

---

## 7.5 让机器人保持航向（不原地打转）

### 为什么会打转

第一版奖励里**完全没有航向项**：

- `flat_orientation_l2` 只看投影重力的 **x/y 分量**，也就是只约束 roll/pitch；
  **纯 yaw 旋转完全不影响这两个分量**，所以在它眼里"打转"和"站着不动"一样好；
- 观测里也没有绝对航向（`base_ang_vel` 只有 yaw *角速度*，没有角度）；
- `wheel_spin` 权重只有 `-1e-4`，相对 `alive=+0.5` 可以忽略。

所以 PPO 很自然地找到了"站着慢慢转"这个同样高分的解。已在实测中确认：
两个轮轴在机体坐标系里**同向**（都是 `-Y`），所以不存在转向反接的硬件 bug。

### 两种修法（二选一）

#### 方法 A：重训，让策略自己保持航向（推荐，长期更好）

已经改好了，直接按原来的命令重训即可：

- 观测 **29 → 31 维**（新增 `heading` = 航向误差的 sin/cos）
- 新增 3 个奖励项：

| 奖励项 | 权重 | 作用 |
|---|---|---|
| `heading` | −0.5 | 二次惩罚航向误差（1 rad 误差约等于全部 alive 奖励） |
| `yaw_rate` | −0.25 | 抑制快速旋转，防止绕目标来回摆 |
| `heading_alive` | +0.25 | 误差 < 0.35 rad 时给固定奖励，近目标处梯度更明确 |

```cmd
cd /d D:\Robomaster\RL
taskkill /F /IM python.exe
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Balance-v0 --headless --num_envs 1024 --max_iterations 5000 --no_code_state
```

> ⚠️ **必须从零重训**。观测维度变了（29→31），旧的 checkpoint 无法续训。

#### 方法 B：不重训，外挂一个航向环（立刻可用）

保留你已经训好的 29 维策略，在它的轮速指令外面套一个**航向 PD 控制器**。
原理：把策略输出的**轮速均值**（负责前后平衡）保留，把**差速分量**（负责转向）换成控制器算的，
所以策略学到的平衡能力不受影响。

```cmd
cd /d D:\Robomaster\RL
taskkill /F /IM python.exe
set INFANTRY_RL_HEADING_CONTROL=1
set INFANTRY_RL_NO_HEADING_OBS=1
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\play.py --task Infantry-RL-Balance-Play-v0 --num_envs 1 --checkpoint 2026-10-17_10-45-58\model_2650.pt --debug_heading
```

| 环境变量 | 作用 |
|---|---|
| `INFANTRY_RL_HEADING_CONTROL=1` | 打开航向环 |
| `INFANTRY_RL_NO_HEADING_OBS=1` | 去掉 heading 观测，恢复 29 维（旧权重才能加载） |

调参（在 `balance_env_cfg.py` 的 `ActionsCfg.joints` 里）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `heading_kp` | 3.0 | 每 rad 航向误差对应的目标偏航角速度 |
| `heading_kd` | 0.35 | 阻尼，抑制摆动 |
| `heading_max_yaw_rate` | 3.0 | 目标偏航角速度上限 (rad/s) |
| `heading_mode` | `"override"` | `override` 替换策略差速；`additive` 叠加在策略差速上 |
| `target_heading` | 0.0 | 要保持的航向（rad），所有环境初始 yaw = 0 |

### 实测效果（`model_2650` + 方法 B，400 步 = 8 秒）

| 指标 | 修之前 | 修之后 |
|---|---|---|
| 净转角 | −179.4° | **−1.10°** |
| 航向最大偏差 | 179.97° | **1.29°** |
| 机体高度 | 0.2039 m | **0.2029 m**（平衡未受影响） |
| 闭环残差 | 9.4e-4 m | **6.5e-4 m** |

残余的 1.3° 是**约 1 Hz 的偏航小幅摆动**，不是持续漂移
（用四元数逐步核对：航向稳定后 50 步内几乎不变）。

### 判断有没有修好

回放时看新增的输出：

```
[play] heading: net turn=-1.10 deg  max |deviation|=1.29 deg
```

- **`net turn` 接近 0** → 修好了（原来是 -180° 左右）
- `net turn` 还是 ±180° → 航向环没生效，检查两个环境变量是否都设了

---

## 7.6 让机器人不乱跑（保持位置 / 控制位移）

### 为什么修好航向后又开始往后跑

和航向是**完全同一类漏洞**：第一版奖励里也没有任何约束前后位移的项。

| 奖励项 | 为什么管不住前后跑 |
|---|---|
| `alive` / `flat_orientation` / `base_height` | 只管"站着"，不管"站在哪" |
| `wheel_spin` (−1e-4) | 权重太小，可忽略 |
| 观测 | 有 `base_lin_vel`（速度），但**没有位置** —— 策略不知道自己跑出去多远了 |

实测：修好航向后，机器人 10 秒内**稳定后退 −0.245 m**（约 2.5 cm/s），
高度始终 0.2008 m —— 也就是"平衡完美，但一直在漂"。

### ⚠️ 为什么这次不能用外挂环（和航向不同）

| | 航向 | 前后位移 |
|---|---|---|
| 对应轮速分量 | **差速**（两轮反向） | **共模**（两轮同向） |
| 与平衡的关系 | **正交**，覆盖它不影响平衡 | **就是平衡的控制输入本身** |
| 能否外挂环 | ✅ 可以（已验证，见 7.5） | ❌ **不行** |

航向环能成功是因为差速只产生偏航力矩，不改变前进/后退力；
而前后位移的共模轮速**正是**平衡控制器用来前后调整的那个量。
从外面硬给共模轮速 = 像"不给 Segway 倾角、只给轮速"，必然推倒。

**实测验证**（`model_350` + 位置环 kp=1.5/kd=0.6，500 步）：

| | 无位置环 | 有位置环 |
|---|---|---|
| 后退位移 | −0.245 m | −0.445 m（更差） |
| 机体高度 | 0.2008 m 稳定 | **掉到 0.16 m → 摔了** |

所以位移**必须靠训练**，让策略学会"一边倾斜一边移动"这一整个技能。

### 方法：重训（观测 31 → 33 维）

已改好，直接重训：

- 观测新增 **`position`（2 维）** = 位置误差在机体偏航系下的 (前, 左) 分量
- 新增 3 个奖励项：

| 奖励项 | 权重 | 作用 |
|---|---|---|
| `position` | −4.0 | 二次惩罚离目标点的距离（0.25 m 偏差约消耗一半 alive 奖励） |
| `lin_vel` | −1.0 | 阻尼，防止在目标点附近来回振荡 |
| `position_alive` | +0.2 | 距目标 < 0.2 m 时给固定奖励 |

```cmd
cd /d D:\Robomaster\RL
taskkill /F /IM python.exe
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Balance-v0 --headless --num_envs 1024 --max_iterations 5000 --no_code_state
```

> ⚠️ **必须从零重训**。观测维度 31→33，之前那个 31 维的 checkpoint 无法续训。

### 怎么"控制位移"（让机器人走到指定地方）

改 `balance_env_cfg.py` 顶部的两个常量即可：

```python
TARGET_X = 0.0   # 目标位置（m），+X 是机器人正前方
TARGET_Y = 0.0   # +Y 是左侧
```

比如 `TARGET_X = 0.5` 就是"往前 0.5 m 处停住"。
这是**每个环境相对自己出生点**的目标，因为训练时 1024 个环境共用同一套配置。

> 想做"变目标点"的训练（课程学习），需要把目标改成每个环境随机采样的
> command，参考 `isaaclab.envs.mdp.commands` 里的 `UniformVelocityCommand` 之类。
> 当前是固定目标，够用且更简单。

### 回放时怎么看

```cmd
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\play.py --task Infantry-RL-Balance-Play-v0 --num_envs 1 --checkpoint ^<你的run^>\model_xxxx.pt
```

看新增的两行：

```
[play] position: net displacement=(-0.245, +0.002) m  max |x|=0.245 m  mean |vx|=0.026 m/s
```

- **`net displacement` 的 x 接近 0** → 不再往后跑
- 还是 −0.2 m 以上 → 训练轮数不够，或位置奖励权重需要加大

---

## 7.7 让机器人跑起来（前进后退 / 转弯 / 原地旋转）

这是一个**新任务** `Infantry-RL-Locomotion-v0`，和前面的 Balance 任务并列，
互不影响（Balance 的代码没动）。

### 和 Balance 任务的区别

| | Balance | Locomotion |
|---|---|---|
| 目标 | 站在原地，保持航向/位置 | 跟踪速度指令 `(vx, wz)` |
| 目标的来源 | 固定常量（0） | CommandManager 产生 |
| 观测维度 | 33 | **31** |
| 能不能按指令动 | 不能 | **能**（前进/后退/转弯/原地旋转） |

### 指令是怎么定义的

机器人是**两轮差速**：前进后退靠两轮**共模**转速，转弯靠**差速**。
所以指令是二维的：

- `vx` = 前进速度（前后）
- `wz` = 偏航角速度（左右转）

**没有 `vy`**（侧向）—— 这个机器人没有侧向驱动轮，侧移指令永远跟不了，
所以直接固定成 0，也不喂给策略（避免浪费一个观测维度）。

用的是 Isaac Lab 的 `UniformVelocityCommand`，并开启 `heading_command=True`：

```python
wz_cmd = clip(0.9 * wrap(heading_target - heading), -1.5, 1.5)
```

也就是说**策略只跟踪角速度，航向目标由上层持有**。这样三个动作是同一套机制：

| 操作 | 怎么实现 |
|---|---|
| 前进/后退 | 直接给 `vx` 指令 |
| 转弯 | `vx ≠ 0` 且航向目标持续变化 |
| 原地旋转 | `vx = 0` 且航向目标持续变化 |

### 训练范围

| 指令 | 训练范围 |
|---|---|
| `vx` | −0.5 ~ +0.5 m/s |
| `wz` | −1.5 ~ +1.5 rad/s |
| 航向目标 | −π ~ +π |
| 指令保持时长 | 5 ~ 10 秒（每集 20 秒，能看到 2~4 个指令） |

另外 10% 的环境会被指令为"原地站住"，用来学刹车。

### 怎么训练

```cmd
cd /d D:\Robomaster\RL
taskkill /F /IM python.exe
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 1024 --max_iterations 8000 --no_code_state
```

> 运动比站立难得多，**8000 轮**才算比较稳（约 6 小时）。
> 前 500 轮可能看着像"乱蹬腿"，这是正常的 —— 先学会不摔，再学会走。

### 崩溃续训（这个任务训练时间长，很可能用得上）

之前的 GPU 崩溃大约在 200 轮 / 5 分钟左右发生。现在支持**自动找最新 checkpoint 续训**：

```cmd
cd /d D:\Robomaster\RL
taskkill /F /IM python.exe
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 1024 --max_iterations 8000 --no_code_state --resume --resume_latest
```

`--resume_latest` 会按**数字大小**挑最新的 `model_N.pt`。
（内置的 `--resume` 是按**字符串**排序的，会把 `model_0.pt` 当成最新的，
所以一定要加 `--resume_latest`。）

### 训练轮次不够，怎么接着练（重要）

#### 规则：`--max_iterations` 是"**再训多少轮**"，不是"总共训到多少轮"

rsl-rl 的循环是这么写的（`on_policy_runner.py` L77-79）：

```python
start_it = self.current_learning_iteration              # 从 checkpoint 恢复的轮次
total_it = start_it + num_learning_iterations           # ← 加，不是赋值
for it in range(start_it, total_it):
```

举例：已经训到 **2000 轮**，想再练 2000 轮：

```cmd
:: ✅ 正确：--max_iterations 2000  → 从 2000 续到 4000
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 1024 --max_iterations 2000 --no_code_state --resume --resume_latest

:: ❌ 错误：--max_iterations 4000 → 会从 2000 一路跑到 6000
```

#### 恢复的是什么

`--resume` 是**真正的续训**，不是"拿权重重新微调"：

| 恢复内容 | 说明 |
|---|---|
| actor / critic 网络权重 | ✅ |
| **优化器状态**（Adam 动量） | ✅ 所以学习率自适应会接着来 |
| **已完成轮次** `iter` | ✅ 所以轮次编号会接着往上数 |

#### 续训后目录会变，轮次编号不变

续训会**新建一个时间戳目录**，但里面的 checkpoint 编号**接着原来的数**：

```
logs/rsl_rl/infantry_rl_locomotion/
├── 2026-10-17_14-25-45/          ← 第一次训练
│   ├── model_1950.pt
│   └── model_2000.pt
└── 2026-10-17_15-30-00/          ← 续训产生的新目录
    ├── model_2000.pt             ← 续训起点（会重新存一个）
    ├── model_2050.pt
    └── ...
```

`--resume_latest` 已经处理了这点：它**跨所有目录按轮次数字取最大**，
所以无论续训多少次，永远接在"练得最多的那个"后面。

#### 怎么确认续训正确启动了

看训练日志第一行迭代输出：

```
Learning iteration 2000/4000     ← 起点是 2000，总数是 4000 = 2000 + 2000
```

- 起点是 **2000**（不是 0）→ 恢复成功
- 总共是 **4000** → `--max_iterations 2000` 起了作用

如果显示 `Learning iteration 0/...`，说明加载的还是初始权重，检查是否漏了 `--resume`。

#### 也可以指定从哪个 checkpoint 续

不想用"最新的"，想从某个特定点续：

```cmd
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 1024 --max_iterations 2000 --no_code_state --resume --load_run 2026-10-17_14-25-45 --checkpoint model_1000.pt
```

> ⚠️ **续训前必须先停掉当前训练**（`taskkill /F /IM python.exe`）。
> 两个训练同时跑会抢同一块 GPU（8 GB），既慢又容易触发之前的 CUDA 崩溃。

### 训练中怎么看

除了常规的 `Mean reward`，重点看这四项：

```
Episode_Reward/track_lin_vel_x     ← 前进速度跟得准不准，满值约 2.0*20s
Episode_Reward/track_ang_vel_z     ← 转弯角速度跟得准不准
Episode_Reward/heading_error       ← 航向到了没有
Metrics/base_velocity/error_vel_xy ← 速度误差（越小越好）
Metrics/base_velocity/error_vel_yaw← 角速度误差（越小越好）
```

### 怎么看训练效果（回放，固定指令）

键盘遥控脚本已删除，现在只用 `play.py` 按固定指令回放：

```cmd
cd /d D:\Robomaster\RL
taskkill /F /IM python.exe

:: 原地站住（vx=0，航向 0）
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\play.py --task Infantry-RL-Locomotion-Play-v0 --num_envs 1 ^
    --checkpoint <run>\model_xxxx.pt --steps 2000 --cmd_vx 0 --cmd_heading_deg 0

:: 直行 0.3 m/s
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\play.py --task Infantry-RL-Locomotion-Play-v0 --num_envs 1 ^
    --checkpoint <run>\model_xxxx.pt --steps 2000 --cmd_vx 0.3 --cmd_heading_deg 0

:: 原地左转 90 度（vx=0，航向目标 +90°）
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\play.py --task Infantry-RL-Locomotion-Play-v0 --num_envs 1 ^
    --checkpoint <run>\model_xxxx.pt --steps 2000 --cmd_vx 0 --cmd_heading_deg 90

:: 边走边转（前进 0.3 m/s，同时转向 45 度）
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\play.py --task Infantry-RL-Locomotion-Play-v0 --num_envs 1 ^
    --checkpoint <run>\model_xxxx.pt --steps 2000 --cmd_vx 0.3 --cmd_heading_deg 45
```

不加 `--cmd_*` 就用环境的随机指令。

回放看这几行：

```
[play] base_z mean=0.19xx m                     ← 应接近 0.19（不再蹲着跑）
[play] heading: net turn=... max |deviation|   ← 转向是否到达目标
[play] position: net displacement=(x, y)       ← 走了多远
```

---

## 7.8 运动任务的目标设计（为什么这么配权重）

> 这一节解释 7.7 里那些权重是怎么来的，改行为时照着调。

### 需求

1. base 保持平衡
2. **pitch 尽量为 0**（前进、后退、原地旋转时都如此）
3. 行进中转弯**可以靠倾斜机身**实现
4. 转弯要够快

### 之前的问题（实测）

`model_4999`（旧奖励，5000 轮）跑起来：

```
base_z mean = 0.1399 m      ← 正常站立约 0.19，蹲了 5 cm
```

原因：旧奖励用 `base_height_l2`（平方，权重 −10）。在 5 cm 偏差下
每步只贡献约 `0.05² × 10 × 0.02 = 0.0005`，而速度跟踪奖励有 `0.04` ——
**差了 80 倍**，所以"蹲着跑更快"这个交易永远划算。

### 新奖励设计

| 奖励项 | 权重 | 形式 | 作用 |
|---|---|---|---|
| `alive` | +0.5 | — | 活着 |
| `fall_penalty` | −50 | — | 摔了重罚 |
| **`base_height`** | **−10** | **L1** `\|h−h0\|` | 锁住站立高度，不再蹲 |
| **`pitch`** | **−3** | **L1** `\|pitch\|` | **pitch 归零** |
| `roll` | −0.3 | L1 `\|roll\|` | 只做轻微约束，**允许压弯** |
| `lin_vel_z` | −2 | L2 | 不许上下颠 |
| `track_lin_vel_x` | +2.0 | exp | 前进速度跟踪 |
| **`track_ang_vel_z`** | **+2.5** | exp | 角速度跟踪（权重更高 → 转得更利落） |
| `heading_error` | −0.4 | L2 | 真的转到目标航向 |
| `lin_vel_y` | −0.5 | L2 | 不许横向滑 |
| `action_rate` / `joint_torques` / `joint_acc` / `wheel_spin` | 极小 | L2 | 平滑 |
| `closure_residual` | −1000 | L2 | 五连杆闭环必须闭合 |

### 三个关键设计决定

**① 用 L1（绝对值）而不是 L2（平方）**

`x²` 的梯度在 0 附近趋于 0、远处才变大，所以要么压不住小偏差、
要么一大就爆。`|x|` 的梯度是常数，从任何距离都以恒定力往回拉 ——
这正好是"pitch 保持在 0"想要的。

**② 把 `flat_orientation_l2` 拆成 pitch 和 roll**

原来的 `flat_orientation_l2` 惩罚 `投影重力 x²+y²`，**同时约束 roll 和 pitch**。
但两轮机器人**没有横向力来源**，高速转弯必须靠**侧倾（roll）**让重力提供向心力
—— 这就是你说的"倾斜机身转弯"。所以：

- **pitch**：强力拉回 0（−3）
- **roll**：只给 −0.3 的象征性约束（防止侧翻），不阻止压弯

> 侧倾角度不需要专门奖励去"教"——物理会自然逼出来：
> 轮子不提供横向力，不压弯就转不了弯。把 roll 放开即可。

**③ 转弯要快：加大指令范围 + 加大 heading P 增益**

| 参数 | 旧值 | 新值 | 效果 |
|---|---|---|---|
| `ANG_VEL_Z_RANGE` | ±1.5 rad/s | **±2.5 rad/s** | 最大 143°/s |
| `HEADING_STIFFNESS` | 0.9 | **1.5** | 1 rad 航向误差直接要满舵 |
| `track_ang_vel_z` 权重 | +2.0 | **+2.5** | 角速度跟踪优先于平移 |

### 观测维度**没变**（仍 31 维）→ 可以直接续训

这一点很重要：这次只改了奖励和指令范围，**没有改观测**，
所以 `model_4999` 可以直接接着训，不用从零开始。

```cmd
cd /d D:\Robomaster\RL
taskkill /F /IM python.exe
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 1024 --max_iterations 3000 --no_code_state --resume --resume_latest
```

> 改成新奖励后，**头几百轮奖励会明显下跌**（`base_z` 从 0.14 被拉向 0.19，
> 走路变"别扭"），这是策略在重新学站姿，正常。看 `base_z` 是否回到 0.19 附近。

---

## 7.9 训练加速（不改硬件）

### 先看清瓶颈：不是 GPU 算力，是 CPU 发射

实测 1024 环境训练时：

```
GPU 利用率   : 35% ~ 43%     ← 只用了四成
显存占用     : 3.6 GB / 8.2 GB
每轮耗时     : 1.18 s（24 个控制步）→ 每控制步 49 ms
```

GPU 大半时间在**等 CPU 派活**。原因是每个控制步要发**成百上千个小 CUDA kernel**，
而每个 kernel 的 Python/驱动开销约 10~30 µs，加起来就是几十毫秒。

### 用两轮数据拟合出的耗时模型

| 环境数 | 实测每轮耗时 |
|---|---|
| 1024 | 1.18 s |
| 4096 | 2.65 s（历史数据） |

拟合 `t = a + b·N`：

```
b（每环境成本）= 4.8e-4 s
a（与环境数无关的固定成本）= 0.69 s   ← 占 1024 时的 58%！
```

**这 0.69 s 就是"每步 CPU 开销 × 24 步"**，它与环境数无关 —— 所以
**加环境数几乎不增加这 0.69 s，却能线性增加数据量**，这是最大的杠杆。

### 已经做掉的优化（代码已改，不改变数值行为）

| 优化 | 说明 | 依据 |
|---|---|---|
| **消除每步 GPU 同步** | `knee_clamped_total` 从 `int(x.sum().item())` 改成 GPU 标量；从 `extras` 里删掉每步的 `max().item()` | 每个 `.item()` 都强制 CUDA→CPU 同步，会卡住流水线 |
| **去掉重复计算** | `measured_closure_norm()` 每步被算了 2 次（奖励一次、日志一次），改成缓存复用 | 少一半相关 kernel |
| **牛顿求解器 3 → 2 次迭代** | 400 组可行 hip 组合实测：2 次迭代最大残差 **3.6 µm**，而仿真实际只能做到 **750 µm** | 精度完全够，少 ~150 个 kernel/步 |
| **解析雅可比替代有限差分** | 原来每次迭代要算 3 次闭环函数（12 次旋转），现在只需 1 次闭环 + 解析导数（约 4 次旋转） | A/B 实测：2 次迭代残差从 3.6e-6 降到 **6.0e-7 m**，更准 |

合并效果：求解器 kernel 数量约为原来的 **1/4.5**，且不再有每步同步点。

### 最大的杠杆：提高 `--num_envs`

| 环境数 | 吞吐（步/秒） | 相对 1024 |
|---|---|---|
| 1024 | 20,800 | — |
| 1536 | 25,900 | **+24%** |
| **2048** | **29,400** | **+41%** |
| 3072 | 34,100 | **+64%** |
| 4096 | 37,100 | +78%（但会崩，见下） |

4096 崩溃的原因是 **BAR1（PCIe 显存窗口）耗尽**（8164/8192 MiB），不是显存不够。
现在有开关可以试：

```cmd
:: 关掉 fabric（崩溃就发生在它里面），试 2048
set INFANTRY_RL_USE_FABRIC=0
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 2048 --max_iterations 100 --no_code_state --run_name speed_test
```

**建议按 1536 → 2048 → 3072 阶梯试**，每个跑 100 轮看是否稳定、记下 `Mean reward` 是否正常。

### 还可以试的（有风险，需要验证）

| 手段 | 预期收益 | 风险 |
|---|---|---|
| `INFANTRY_RL_PHYSX_POS_ITERS=4`（默认 8） | 减少 PhysX 求解器负载 | 物理精度下降，可能影响行走 |
| `sim.dt` 0.005→0.01，`decimation` 4→2 | 物理子步减半，**最多快 2×** | 轮子高速摩擦可能不稳定，需重新验证 |
| 关闭其他占 GPU 的程序（浏览器/EDA 软件） | 腾出显存给更多环境 | 无 |

### 怎么测速

```cmd
cd /d D:\Robomaster\RL
set INFANTRY_RL_USE_FABRIC=0
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\train.py --task Infantry-RL-Locomotion-v0 --headless --num_envs 2048 --max_iterations 100 --no_code_state --run_name speed_test
```

训练日志里找 `Learning iteration` 的时间戳，或者看每 50 轮的 checkpoint 文件时间间隔：
**50 轮的实际耗时**就是最准的指标。对比公式：

```
吞吐（步/秒） = 48 × num_envs / 每轮耗时
```

---

## 7.10 让腿主导平衡（而不是轮子）

### 问题：腿完全不动，全靠轮子扭

实测回放：`mean |hip dev|` 长期接近 0 —— 髋关节绷在名义姿态，
机器人是靠**轮子力矩反作用**保持 pitch 的（Segway 式）。

### 物理原因：pitch 有两个冗余执行器

| 执行器 | 机制 | 现在是否被罚 |
|---|---|---|
| **轮子力矩** | 反作用力矩直接扭机体 | ❌ 几乎免费 |
| **髋关节** | 改腿角 φ → 机体 CG 相对轮子前后移动（倒立摆） | ✅ 被多处惩罚 |

旧奖励下轮子力矩 2 N·m 的惩罚是 `2² × 1e-4 × 0.02 = 8e-6/步`，
而 `alive` 是 `0.01/步` —— **差 1250 倍，等于免费**。PPO 当然选更简单的轮子方案。

同时还有三项在机械地压制腿部动作：

| 旧奖励项 | 问题 |
|---|---|
| `base_height −10` | 腿角 φ 一动高度就降（φ=20° 约降 1.5 cm，因为 `Δh = L0(1−cosφ)`） |
| `action_rate −0.01` | 腿需要快速响应，动作变化被罚 |
| `joint_torques −1e-4` | 髋关节出力也被罚 |

### 改动

| 奖励项 | 旧 | 新 | 理由 |
|---|---|---|---|
| **`wheel_torques`**（新增） | — | **−0.05** | 只罚两个轮关节的 L2 力矩。倒立摆式行走几乎不需要轮子力矩（轮子只要跟着滚），而轮子平衡会飙到 1~3 N·m → 这个交易被翻转 |
| `base_height` | −10 | **−4** | 给腿角变化留空间 |
| `pitch` | −3 | **−4** | pitch 是腿要调节的量，加严 |
| `action_rate` | −1e-2 | **−4e-3** | 允许腿快速动 |
| `joint_torques` | −1e-4 | **−2e-5** | 不压制髋关节出力 |

> 实现：`wheel_torques` 用内置的 `joint_torques_l2`，只传轮关节
> `SceneEntityCfg("robot", joint_names=["left_wheel", "right_wheel"])`，
> 所以它**只罚轮子，不罚髋**。

### 怎么验证腿在动

`play.py` 新增了腿部的完整状态输出：

```
[play] legs: mean |hip dev|=X.XX deg (max ...)  mean |phi|=X.XX deg (max ...)  mean |dphi|=X.XX rad/s (max ...)
[play] stance: mean |dL0|=X.XXX m/s  mean |L0_left - L0_right|=X.XX mm
```

| 指标 | 含义 | 判定 |
|---|---|---|
| `mean \|hip dev\|` | 髋关节偏离名义姿态的角度 | **腿主导 ≈ 5~20°**；轮子主导 ≈ 0~1° |
| `mean \|phi\|` | 腿轴角（前后倾，改 CG 用） | 腿主导时明显非零 |
| `mean \|dphi\|` | 腿轴角速度（腿在摆动的速率） | 腿主动工作时 > 0.2 rad/s |
| `mean \|dL0\|` | 腿长变化率 | 主动调高度时明显非零 |
| `mean \|L0_left − L0_right\|` | 左右腿长差 | >0 说明在用**腿部差动调 roll** |

判断标准：

```cmd
D:\miniconda\envs\RM\python.exe isaac_ws\scripts\play.py --task Infantry-RL-Locomotion-Play-v0 --num_envs 1 --load_run .* --checkpoint model_5000.pt --cmd_vx 0.3 --cmd_heading_deg 0 --steps 3000 --real-time
```

- `mean |hip dev|` **> 3°** → 腿在参与平衡 ✅
- 仍接近 0° → 还是轮子平衡，把 `wheel_torques` 权重从 −0.05 加大到 −0.1
- 如果腿动得厉害但站不稳 → 权重过大了，往回调

### 观测补上腿速度（31 → 35 维）

固件 LQR 的状态是 `x = [θ, dθ, xb, dxb, φ, dφ]` —— **`dφ` 是显式控制状态**
（它的 `Tp` 行在 `dφ` 上的增益约 5.9，用来给腿角环做阻尼）。

原来的 31 维观测里只有 `φ`（在 `solved_leg_state` 内），**没有 `dφ`**。
MLP 没有记忆，只靠位置反馈的话腿一旦开始承担姿态环就会滞后/振荡。

所以新增 4 维观测：

```
leg_rates = (dL0_l, dφ_l, dL0_r, dφ_r)
```

取自实测 `(L0, φ)` 的一步差分（`actions.py` 的 `update_leg_rates()`），
复位的那一步强制为 0（避免姿态跳变产生假速度）。

> ⚠️ 观测 31 → **35 维**，旧 checkpoint **不能续训**，必须从零开始。

> 奖励改动**不影响 checkpoint 兼容性**，但观测改了就不行。这一轮必须从零训练。

---

## 8. 相关脚本

| 脚本 | 用途 |
|---|---|
| `scripts\train.py` | 训练入口 |
| `scripts\play.py` | 回放/评估 |
| `scripts\smoke_test.py` | 闭环 + 站立验证（不开训练也能跑） |
| `scripts\diag_clone.py` | 诊断多环境卡死 |
| `scripts\build_proxy_asset.py` | 生成轻量代理资产 |
| `isaac_ws_infantry\kinematics\test_leg_model.py` | 运动学验证（秒级，不需要 GPU） |

完整背景见 [`VERIFIED_RESULTS.md`](VERIFIED_RESULTS.md) 与 [`DESIGN.md`](DESIGN.md)。