# 轮腿 RoboMaster 底盘 RL locomotion 训练事实档案（配置 / 源码 / 时间线）

> 本档案供撰写中文训练报告直接引用。每条数字尽量给出 `相对路径:行号` 出处。
> **快照时间：2026-10-22 11:30（Asia/Shanghai）**。写作期间工作树中的 `VERSION.md`、
> `TRAINING_GUIDE.md`、`isaac_ws/.../locomotion_env_cfg.py`、`isaac_ws/scripts/train.py`
> 仍在继续更新（最终已到 v1.0.15，全部属于**训练之后**的改动；相关文件行号以 11:30 快照为准），因此：
>
> * **描述"这两次 run 跑了什么"时，一律以 run 目录里的 `params\*.yaml` 为权威**（文件生成后不再变）；
> * 以 `git show be7cc15:...` 引用的行号表示"训练时实际代码/文档"快照；工作树行号为 11:30 快照；
> * 文档与 yaml/源码冲突处逐一列出，并注明哪一方权威（§3.3、§4.3）。
>
> **阶段定义**：
> * **A = 2026-10-22_00-28-38**（v1.0.7 配置，从零训练）；
> * **B = 2026-10-22_01-17-16**（v1.0.8 配置，从 A 的 `model_999.pt` 续训）。
>
> 两条 run 的 `params\env.yaml` 别名：`A/env.yaml` =
> `logs\rsl_rl\infantry_rl_locomotion\2026-10-22_00-28-38\params\env.yaml`，
> `B/env.yaml` = 同结构下 `2026-10-22_01-17-16\params\env.yaml`；
> `A/agent.yaml` / `B/agent.yaml` 同理。
>
> 其它简写：`rewards.py` / `actions.py` / `observations.py` / `curriculums.py` 均指
> `isaac_ws\isaac_ws_infantry\mdp\` 下的同名文件（其中 `curriculums.py` 在 v1.0.15 已被删除，
> 档案中只作为历史事实提及）；`locomotion_env_cfg.py` /
> `locomotion_env.py` 指 `isaac_ws\isaac_ws_infantry\tasks\locomotion\` 下的文件；
> §4.1 表格内的 `A:xxx` / `B:xxx` 是 `A/env.yaml:xxx` / `B/env.yaml:xxx` 的紧凑写法。

---

## 1. 任务与平台

### 1.1 任务注册

| 项目 | 值 | 出处 |
|---|---|---|
| 训练任务 id | `Infantry-RL-Locomotion-v0` | `isaac_ws\isaac_ws_infantry\tasks\locomotion\__init__.py:12` |
| 环境类 / 配置入口 | `locomotion_env:FiveBarLocomotionEnv` / `locomotion_env_cfg:LocomotionEnvCfg` | 同上 `:13-16` |
| Agent 入口 | `agents.rsl_rl_ppo_cfg:InfantryLocomotionPPORunnerCfg` | 同上 `:17` |
| 回放任务 id | `Infantry-RL-Locomotion-Play-v0`（`LocomotionEnvCfg_PLAY`，`num_envs=1`、无超时） | `isaac_ws\isaac_ws_infantry\tasks\locomotion\__init__.py:20-29`；`locomotion_env_cfg.py:500-512` |

### 1.2 机器人本体（URDF 与实测配置）

资产：`isaac_ws\assets\infantry-rl\urdf\infantry-rl-sim.urdf`（USD 由它转换，
`isaac_ws\assets\infantry-rl\config.yaml:1-3`；训练实际加载 USD：
`A/env.yaml:168` = `...\isaac_ws\assets\infantry-rl\infantry_rl.usd`）。

| 项目 | 值 | 出处 |
|---|---|---|
| 连杆数 | **11**（base_link + 左右 front/back_ham + 左右 front/back_calf + 左右 wheel） | URDF `link name=` 行：`:3,25,54,83,112,141,170,199,228,257,286` |
| 总质量 | **12.7761 kg**（=9.3897 + 2×0.31578 + 2×0.24697 + 2×0.69516 + 2×0.33194 + 2×0.10335；`VERSION.md:56` 记 12.776 kg） | 质量行：URDF `:6,28,57,86,115,144,173,202,231,260,289` |
| 自由度 | **10** = 4 髋（revolute）+ 4 膝（revolute）+ 2 轮（continuous） | `VERSION.md:57`；URDF 关节类型行 `:47,76,105,134,163,192,221,250,279,308` |
| 闭环 | 2 个 `<loop_joint type="spherical">`（`left_calf_axle` / `right_calf_axle`），**不是 DOF**，靠控制维持，每个控制步解一次 | URDF `:315-322`；`isaac_ws\isaac_ws_infantry\robots\infantry_rl.py:6-8`；`VERSION.md:58` |
| 轮半径 | **0.068 m** | `isaac_ws\isaac_ws_infantry\robot_model.py:126-128`；`A/env.yaml:446`（动作项 `wheel_radius`）；`VERSION.md:59` |
| 髋轴高度 | **0.128 m**（相对 base_link；URDF 髋关节 origin z=0.128） | URDF `:48,135,193,280`；`isaac_ws\config\planar_leg.json`（A/B 点 y=0.128） |
| 名义腿长 L0 | **0.24977467 m**（左右实测均值 0.249775） | `isaac_ws\config\nominal_pose.json`（"L0": 0.24977467318622631 / 0.2497746654962643）；`robot_model.py:142-145`；`VERSION.md:61` |
| 名义站立高度 | **0.18977467 m** = 0.068 + 0.24977467 − 0.128 | 公式 `robot_model.py:148-150`；`A/env.yaml:174`（初始 z=0.18977466934124532）；`VERSION.md:62` |
| L0 可达范围 / φ 范围 | 0.1356…0.2843 m / −31.2°…+33.2° | `VERSION.md:63-64` |

### 1.3 关节限位与执行器

URDF 关节限位（rad，effort/velocity 为电机额定）：

| 关节组 | 限位 | effort / velocity | 出处 |
|---|---|---|---|
| 前髋（left/right_front_hip） | −0.184047 … 1.305207 | 40 N·m / 16.755 rad/s | URDF `:52`（右 `:197`） |
| 后髋（left/right_back_hip） | −1.822593 … 0.184012 | 40 / 16.755 | URDF `:139`（右 `:284`） |
| 前膝（left/right_front_knee） | −0.162714 … 1.836403 | 40 / 16.755 | URDF `:81`（右 `:226`） |
| 后膝（left/right_back_knee） | −1.836403 … 0.162697 | 40 / 16.755 | URDF `:168`（右 `:313`） |
| 轮（continuous） | 无限位 | **3.69 N·m / 61.471 rad/s** | URDF `:110,255` |

实测电机极限（`底盘数据.txt:15-19`：v8009pmax 16.755、v_wheel max 61.4705、T_wheel max 3.69），与 URDF 一致。

仿真执行器参数（**以 run 时 yaml 为准**；与源码 `robots\infantry_rl.py:93-121` 一致）：

| 组 | 关节数 | effort_limit_sim | velocity_limit_sim | stiffness | damping | armature | 出处 |
|---|---|---|---|---|---|---|---|
| hips | 4 | 40.0 | 16.755 | **200** | 5.0 | 0.01 | `A/env.yaml:206-220` |
| knees（被动跟随） | 4 | 40.0 | 16.755 | **400** | 10.0 | 0.01 | `A/env.yaml:221-235` |
| wheels | 2 | **3.69** | **61.471** | 0.0 | 2.0 | 0.001 | `A/env.yaml:236-249` |

### 1.4 五连杆闭环（膝不是自由度）

* 动作只有髋与轮，**膝角由实测髋角每控制步反解**：`mdp\actions.py:348-377`
  （`update_knee_targets()`，热启动自上一帧解）。
* 求解器：**解析雅可比 + 固定 2 次牛顿迭代，批量 torch 实现**（`solver_iters=2`：
  `A/env.yaml:439`、`mdp\actions.py:469-478`、`kinematics\planar_leg.py:398-412,432-433`）；
  2 次迭代最坏残差 3.6e-6 m，3 次为 7.7e-8 m（`mdp\actions.py:472-477`）。
* 仿真中闭环实际残差约 **7.5e-4 m**（PhysX 柔顺 + 被动膝，`VERSION.md:97`）；
  该量作为 `closure_residual` 奖励项（权重 −1000）被硬约束。
* 膝目标会被夹到 URDF 膝限位内、NaN 回退到名义膝角（`mdp\actions.py:361-370`）。
* 动作项默认关闭外挂航向/位置环：`heading_control: false`、`position_control: false`
  （`A/env.yaml:440,448`）；航向由指令项生成的 `wz*` 负责（§3）。

---

## 2. 观测与动作

### 2.1 35 维观测（10 项）

以 `A/env.yaml:326-428`（`observations.policy`）为准，运行时代码与
`locomotion_env_cfg.py:142-190`、`mdp\observations.py` 一致；项顺序即拼接顺序：

| # | 项 | 维度 | 缩放/处理 | 说明（实现行） |
|---|---|---|---|---|
| 1 | `projected_gravity` | 3 | — | 机体系重力投影（`A/env.yaml:332-340`） |
| 2 | `base_ang_vel` | 3 | **×0.25** | 机体角速度（`:341-349`） |
| 3 | `base_lin_vel` | 3 | — | 机体系线速度（`:350-358`） |
| 4 | `velocity_commands` | 2 | ×1.0 | `(vx*, wz*)`，取自 command manager（`:359-368`；`observations.py:159-171`） |
| 5 | `hip_pos_rel_nominal` | 4 | — | 实测髋角 − 名义（`:369-378`；`observations.py:98-101`） |
| 6 | `knee_pos_rel_solved` | 4 | — | 实测膝角 − 反解目标（`:379-388`；`observations.py:104-111`） |
| 7 | `wheel_vel` | 2 | **×0.1** | 轮关节速度（`:389-398`；`observations.py:114-117`） |
| 8 | `last_action` | 6 | 未缩放 | 上一步原始动作（`:399-408`；`observations.py:120-122`） |
| 9 | `solved_leg_state` | 4 | — | 目标 `(L0_l, φ_l, L0_r, φ_r)`（`:409-418`；`observations.py:125-127`） |
| 10 | `leg_rates` | 4 | — | 实测一步差分 `(dL0_l, dφ_l, dL0_r, dφ_r)`（`:419-428`；`observations.py:135-145`） |

合计 3+3+3+2+4+4+2+6+4+4 = **35**；项数 10；文档对照 `VERSION.md:99-116`。

* 归一化：**EmpiricalNormalization**，actor 与 critic 各一份
  （`A/agent.yaml:30,42` 的 `obs_normalization: true`；`VERSION.md:114`）。
  检查点证据：`model_999.pt` 的 actor state_dict 含 `obs_normalizer._mean/_std (1,35)`、
  `count`；首层 `mlp.0.weight` 形状 `(128,35)`。
* 观测**无噪声**（各 term `noise: null`，`enable_corruption: false`，`A/env.yaml:329`）。

### 2.2 6 维动作

| 维度 | 含义 | 缩放 | 夹取路径 | 出处 |
|---|---|---|---|---|
| 0–3 | 4 个髋关节的**位置偏移**（相对名义姿态），顺序 `left_front_hip, left_back_hip, right_front_hip, right_back_hip` | **×0.25 rad** | 目标 = 名义 + 0.25a，再夹到 `[max(URDF下限+0.01, 名义−0.45), min(URDF上限−0.01, 名义+0.45)]` | `mdp\actions.py:34-40,73-86,196-199`；`robot_model.py:79`；`A/env.yaml:435-438` |
| 4–5 | 2 个轮关节**速度目标**（rad/s），顺序 `left_wheel, right_wheel` | **×5.0 rad/s** | 无二次夹取；受执行器 61.471 rad/s 上限约束 | `mdp\actions.py:34-40,73-86,200`；`A/env.yaml:436` |

* 策略输出在 env wrapper 层先被 `clip_actions = 4.0` 夹到 ±4（`A/agent.yaml:11`；
  `isaac_ws\scripts\train.py:615`；裁剪实现为 Isaac Lab `RslRlVecEnvWrapper`，外部文件
  `D:\IsaacLab\source\isaaclab_rl\isaaclab_rl\rsl_rl\vecenv_wrapper.py:153-154`）；
  对应髋偏移 ±1.0 rad、轮速 ±20 rad/s，髋再被 `hip_range=0.45` 进一步限制。
* 膝角**不是动作**；`clip: null` 的动作项参数见 `A/env.yaml:434`。
* 动作项为 `FiveBarJointAction`（`A/env.yaml:431`；实现 `mdp\actions.py:31-41`）。

### 2.3 控制频率、规模与回合

| 项目 | 值 | 出处 |
|---|---|---|
| 物理步长 | **0.005 s（200 Hz）** | `A/env.yaml:21` |
| decimation | **4** → 控制 **50 Hz**（dt=0.02 s） | `A/env.yaml:84`；`locomotion_env_cfg.py:436` |
| 并行环境数 | **1024** | `A/env.yaml:86`；`locomotion_env_cfg.py:424` |
| 环境间距 | 2.5 m | `A/env.yaml:87` |
| 回合长度 | **20 s = 1000 控制步** | `A/env.yaml:471` |
| 随机种子 | 42 | `A/env.yaml:83`；`A/agent.yaml:1` |
| 地面 | 单一平地 `terrains\ground.usda`（无地形课程） | `A/env.yaml:92-99` |
| 关节求解迭代 | 位置 8 / 速度 1 | `A/env.yaml:158-159` |

### 2.4 终止条件

| 项 | 条件 | 出处 |
|---|---|---|
| `time_out` | 20 s（1000 步），`time_out: true` | `A/env.yaml:578-581`（B `:588-591`） |
| `bad_orientation` | 倾斜角 > **1.2 rad（≈69°）** | `A/env.yaml:582-586`（B `:592-596`）；`tasks\common\robot_env_cfg.py:37,117` |
| `base_too_low` | 根高度 < **0.11 m**（名义 0.1898 的 58%） | `A/env.yaml:587-591`（B `:597-601`）；`robot_env_cfg.py:35,118-120` |

---

## 3. 指令与课程

### 3.1 速度指令（以 run 时 yaml 为准，A/B 完全相同）

| 参数 | A 值 / 出处 | B 值 / 出处 |
|---|---|---|
| 指令项类型 | `UniformVelocityCommand`，`heading_command: true` | `A/env.yaml:595,601`；`B/env.yaml:605,611` |
| `vx*` 范围 | **[+0.2, +1.2] m/s（仅前进）** `:606-608` | **[+0.2, +1.2] m/s** `:616-618` |
| `vy*` 范围 | 0（差速底盘无侧向执行器，观测里也丢弃） | `:609-611` / `:619-621` |
| `ang_vel_z` 范围 | ±2.5 rad/s（**只作航向纠偏钳位**，不是随机转向指令） | `:612-614` / `:622-624` |
| `heading` 目标 | **[0.0, 0.0]（转弯关闭，目标航向恒 0）** | `:615-617` / `:625-627` |
| 航向刚度 | 1.5（1/s） | `:602` / `:612` |
| 指令重采样 | 每 **5–10 s** | `:596-598` / `:606-608` |
| 站立环境比例 | **10%**（`rel_standing_envs=0.1`，作为 vx=0 刹车样本） | `:603` / `:613` |
| 航向环境比例 | `rel_heading_envs: 1.0` | `:604` / `:614` |

### 3.2 `wz*` 的生成方式（不是直接采样）

`heading_command=True` 时，command term 每个控制步由航向内环计算：

```
wz* = clip( 1.5 · wrap(ψ_target − ψ), −2.5, +2.5 )        ψ_target ≡ 0（转弯关闭）
```

出处：`VERSION.md:197,205,208-209`；实现机制说明 `locomotion_env_cfg.py:21,95-114`。
**含义**：训练中的 `wz*` 只用于抵消航向漂移；主动转弯、原地旋转都不在训练分布内
（`VERSION.md:208-209`）。

### 3.3 文档与 yaml 冲突（重要，报告引用时以 yaml 为准）

| # | 文档说法 | 实际（两次 run） | 结论 |
|---|---|---|---|
| 1 | `VERSION.md:32`："速度指令范围 ±0.5 → **±2.0 m/s**" | A/B 的 yaml 都是 **[0.2, 1.2]**（`A/env.yaml:606-608`、`B/env.yaml:616-618`） | 文档为 v1.0.0 时期旧描述/规划值；**yaml 权威** |
| 2 | `VERSION.md:195`："速度 vx* = [−1.2, +1.2] m/s（含倒车，v1.0.10）" | 两次 run 均为 [0.2, 1.2] | 该行描述的是**当前工作树代码默认值**（v1.0.10 之后，`locomotion_env_cfg.py:77-78` 现为 −1.2/1.2），不是这两次 run |
| 3 | `VERSION.md:344-345`：`INFANTRY_RL_VX_MIN/MAX` 默认 0.2 / 1.2 | 与 run 时 yaml 一致 | 对 run 时代码正确；对当前工作树已过时（现默认 −1.2/1.2） |
| 4 | `VERSION.md:434`（§9 检查项）："vx +/-2.00 m/s" | 与 yaml 不符 | 同 #1，检查项过时 |
| 5 | `VERSION.md:26-34`（§2）把"观测 31→35、新增 wheel_torques、速度 ±2.0、γ 0.99→0.995"写成"本版本差异" | 这些改动**早于本轮两次 run**：A 的 yaml 已是 35 维（`A/env.yaml:419-428` 有 `leg_rates`）、已有 `wheel_torques`（`A/env.yaml:544-549`）、γ 已是 0.995（`A/agent.yaml:50`） | §2/§4.1 的"上一版本"基线是 v1.0.0，**不能用来归因本轮训练** |

### 3.4 课程（自适应奖励）模块：两次 run 时不存在，也从未启用

* 历史事实：课程/自适应奖励功能于 **v1.0.11** 被加入（`mdp\curriculums.py`，
  包含 `adaptive_tracking_weights` 与 `modify_tracking_weights_scheduled`），
  训练之后又于 **v1.0.15 被整体移除**（当前工作树已无该文件、无
  `LocomotionCurriculumCfg`、无 `INFANTRY_RL_ADAPTIVE_REWARD` 开关；
  `VERSION.md:88,92`）。写作档案时该文件已不在工作树中，故无法给出行号级引用，
  只能引用版本变更记录。
* **这两次 run 未启用**，证据（均为 run 时产物，权威）：
  1. 两份 run 的 `env.yaml` 都是 `curriculum: null`（`A/env.yaml:592`、`B/env.yaml:602`）；
  2. 两个 TensorBoard 事件文件里 `Curriculum/*` tag 数均为 **0**（A：39 个标量 / 22 个
     Episode_Reward；B：41 / 24；无 Curriculum tag）；
  3. 两次 run 的奖励权重全程与静态 yaml 一致（事件标量无 Curriculum 记录）。
* 结论：**本轮训练是"静态奖励 + PPO"训练，没有课程/自适应奖励，也没有其它被训练的
  模块（只训练策略网络，价值网络为 PPO critic）**。课程功能属于训练之后的代码演进，
  不应写进"本次训练使用了课程"的叙述（同理 v1.0.10 之后的倒车默认值也不属于这两次 run）。

---

## 4. 奖励设计

### 4.1 两次 run 的完整奖励表

权重来自各 run 的 `env.yaml`（权威）；形状/实现见 `mdp\rewards.py`。A 共 **22 项**，
B 共 **24 项**（新增 2 个 L1 跟踪项）。顺序按 A 的 yaml；B 的 yaml 行号见括号内
（`A:xxx` = `A/env.yaml:xxx`，`B:xxx` = `B/env.yaml:xxx`）。

| # | 奖励项 | A 权重 | B 权重 | 形状 / 参数 | 作用（实现行） |
|---|---|---|---|---|---|
| 1 | `alive` | +0.5 | +0.5 | 常数 1.0/步（存活） | 基础存活奖励（`A:473-476`/`B:473-476`；Isaac Lab `is_alive`） |
| 2 | `fall_penalty` | −50.0 | −50.0 | 摔倒终止步 1.0 | 重罚摔倒（`A:477-480`/`B:477-480`；`rewards.py:156-161`） |
| 3 | `base_height` | −4.0 | −4.0 | **L1** `\|z−0.18977467\|` | 维持名义站高（`A:481-485`/`B:481-485`；`rewards.py:482-496`） |
| 4 | `pitch` | **−4.0** | **−0.0** | **L1** `\|pitch\|` | 抑制俯仰；**B 权重为 0 → 实际不贡献**（`A:486-489`/`B:486-489`；`rewards.py:399-419`） |
| 5 | `roll` | −1.5 | −1.5 | **L1** `\|roll\|` | 抑制常驻侧倾（`A:490-493`/`B:490-493`；`rewards.py:422-440`） |
| 6 | `roll_straight` | −2.5 | −2.5 | **L1 × 门控** `\|roll\|·(1−min(\|wz*\|/2.5,1))`，`wz_scale=2.5` | 直行时严厉、转弯时允许侧倾（`A:494-499`/`B:494-499`；`rewards.py:443-466`） |
| 7 | `roll_rate` | −0.1 | −0.1 | L2 `ωx²` | 阻尼左右晃动（`A:500-503`/`B:500-503`；`rewards.py:469-479`） |
| 8 | `lin_vel_z` | −2.0 | −2.0 | L2 `vz²` | 抑制上下弹跳（`A:504-507`/`B:504-507`；`rewards.py:379-390`） |
| 9 | `track_lin_vel_x` | **+6.0**, std=0.25 | **+8.0**, std=0.15 | exp 核 `exp(−e²/std²)` | 前进速度跟踪主奖励（`A:508-513`/`B:508-513`；`rewards.py:254-269`） |
| 10 | `track_ang_vel_z` | **+1.0**, std=0.35 | **+3.0**, std=0.15 | exp 核 `exp(−e²/std²)` | 角速度跟踪（航向纠偏）（`A:514-519`/`B:514-519`；`rewards.py:272-286`） |
| 11 | `heading_error` | −0.6 | −0.6 | L2 `wrap(ψ_target−ψ)²` | 真正到达目标航向（`A:520-524`/`B:520-524`；`rewards.py:499-515`） |
| 12 | `track_lin_vel_x_l1` | —（无） | **−2.0** | **L1** `\|e\|`（恒定梯度） | 压掉最后几 cm/s 误差（B:525-529；`rewards.py:333-351`） |
| 13 | `track_ang_vel_z_l1` | —（无） | **−1.0** | **L1** `\|e\|` | yaw 的恒定梯度项（B:530-534；`rewards.py:354-362`） |
| 14 | `track_lin_vel_x_l2` | −3.0 | **−6.0** | L2 `e²` | 大误差主导、打破"站着不动"（`A:525-529`/`B:535-539`；`rewards.py:289-312`） |
| 15 | `track_ang_vel_z_l2` | −1.0 | **−2.0** | L2 `e²` | 抑制 yaw 抖动（`A:530-534`/`B:540-544`；`rewards.py:315-330`） |
| 16 | `lin_vel_y` | −0.5 | −0.5 | L2 `vy²` | 禁止横向拖滑（`A:535-538`/`B:545-548`；`rewards.py:365-376`） |
| 17 | `wheel_axle` | −8.0 | −8.0 | **L1(米)** `\|L0_l sinφ_l − L0_r sinφ_r\|` | 轮轴对齐、禁止歪着走（`A:539-543`/`B:549-553`；`rewards.py:104-140`） |
| 18 | `wheel_torques` | −0.05, **死区 1.0 N·m** | −0.05, **死区 1.5 N·m** | `Σ(max(\|τ_wheel\|−thr,0)²)`，仅 2 个轮 | 逼髋关节而非轮子做姿态平衡（`A:544-549`/`B:554-559`；`rewards.py:73-95`） |
| 19 | `action_rate` | −4.0e-3 | −4.0e-3 | L2 一阶差分 `Σ(Δa)²` | 抑制动作突变（`A:550-553`/`B:560-563`；`rewards.py:29-31`） |
| 20 | `action_jerk` | **−0.5** | **−0.3** | L2 二阶差分 `Σ(a[t]−2a[t−1]+a[t−2])²` | 只罚抖动、平滑运动免费（`A:554-558`/`B:564-568`；`rewards.py:34-49`） |
| 21 | `joint_torques` | −2.0e-5 | −2.0e-5 | L2 `Στ²`（全部 10 关节） | 轻罚关节力矩（`A:559-562`/`B:569-572`；`rewards.py:52-54`） |
| 22 | `joint_acc` | −1.0e-6 | −1.0e-6 | L2 `Σq̈²` | 抑制腿抖（`A:563-566`/`B:573-576`；`rewards.py:57-59`） |
| 23 | `wheel_spin` | −3.0e-5 | −3.0e-5 | L2 `Σv_wheel²` | 轻罚轮速（`A:567-571`/`B:577-581`；`rewards.py:67-70`） |
| 24 | `closure_residual` | −1000 | −1000 | L2 `Σ\|W−E\|²`（米） | 闭环硬约束（`A:572-576`/`B:582-586`；`rewards.py:143-153`） |

> 注意：Isaac Lab 的 RewardManager 实际按 `value × weight × dt` 累加
> （`D:\IsaacLab\source\isaaclab\isaaclab\managers\reward_manager.py:150`），
> 报告引用配置权重即可；TensorBoard 的 `Episode_Reward/*` 已含该缩放。

### 4.2 v1.0.7 → v1.0.8 变更集（A → B，逐项来自 yaml diff）

| 项 | A | B | 是否在 `VERSION.md:86` 的 v1.0.8 变更记录中 |
|---|---|---|---|
| `track_lin_vel_x` 权重 / std | +6.0 / 0.25 | **+8.0 / 0.15** | ✅ |
| `track_ang_vel_z` 权重 / std | +1.0 / 0.35 | **+3.0 / 0.15** | ✅ |
| 新增 `track_lin_vel_x_l1` | — | **−2.0** | ✅ |
| 新增 `track_ang_vel_z_l1` | — | **−1.0** | ✅ |
| `track_lin_vel_x_l2` | −3.0 | **−6.0** | ✅ |
| `track_ang_vel_z_l2` | −1.0 | **−2.0** | ✅ |
| `action_jerk` | −0.5 | **−0.3** | ✅ |
| `wheel_torques` 死区 | 1.0 | **1.5** | ✅ |
| **`pitch` 权重** | −4.0 | **−0.0** | ❌ **变更记录未提；VERSION §3.5 表仍写 −4.0** |

其余配置（指令、观测、PPO 超参、terminations、actuator）A/B 完全相同；
`agent.yaml` 只有 `resume: false → true` 一个差异（§5.1）。

### 4.3 有效项数与 VERSION 冲突

* **A**：yaml 奖励项 **22 项**，全部非零 → 22 项实际贡献。
* **B**：yaml 奖励项 **24 项**；其中 `pitch` 权重为 `−0.0`（=0）。Isaac Lab 的
  RewardManager 会把所有配置项都加入 `active_terms`（`reward_manager.py:243-244`），
  但 `compute()` 中跳过 weight==0 的项（`reward_manager.py:144-148`），因此 B 的
  启动日志会打印 24 个 active terms，而**实际每步贡献的只有 23 项**；
  `Episode_Reward/pitch` 在 B 中恒为 0（`report/data/metrics_summary.md:67` 实测为 0）。
* `VERSION.md:118` 的表题"奖励（24 项）"对应 **B 的 yaml 项数**（也与当前工作树
  `locomotion_env_cfg.py` 的配置一致），**不适用于 A**（A 只有 22 项）。
* `VERSION.md:433`（§9 检查项）"Reward Manager: contains 22 active terms" 只对应 **A**，
  与同文件 §3.5 的 24 项自相矛盾 → 报告若引用"22/24"必须说明是哪一次 run。
* `VERSION.md` §3.5 表（当前行 118-140）还有 3 处数值与 run B 的 yaml / 源码不一致：
  `pitch` 写 −4.0（`:125`，B 为 0）、`wheel_torques` 死区写 1.0（`:138`，B 为 1.5）、
  `action_jerk` 写 −0.5（`:140`，B 为 −0.3）。**yaml 与源码权威**：B 的 yaml
  `:489`（0.0）、`:557`（1.5）、`:568`（−0.3）；当前工作树
  `locomotion_env_cfg.py:233`（`weight=-.0`）、`:381`（threshold 1.5）、`:400`（−0.3）。

### 4.4 设计动机（可直接引用，均有源码注释支撑）

* **腿主导平衡**：pitch 有两个冗余执行器——轮力矩（Segway 式反作用力矩）与髋关节
  （倒立摆式移 CG）。旧奖励下轮子平衡"便宜 1250 倍"，策略学会"腿绷直+轮子扭动"；
  v1.0.7 起用 `wheel_torques`（只罚 2 个轮，带死区）+ `leg_rates` 观测（dL0/dφ 阻尼）
  把姿态任务交给腿（`rewards.py:400-428`；`VERSION.md:38-48`、`:114-116`）。
* **L1 vs L2**：exp 核在误差大时饱和、在 0 附近梯度为零；L2 梯度随误差增长（管大误差），
  L1 梯度恒定（管最后几 cm/s）；姿态量（pitch/roll/height）用 L1 因为 `|x|` 从任意距离
  都给恒定拉力（`rewards.py:289-351`、`:399-419`、`:482-496`；`VERSION.md:149-152`）。
* **wheel_torques 死区**：轮是唯一纵向执行器；匀速几乎不需要力矩，但加速/指令阶跃
  需 0.9–1.7 N·m；死区让平滑推进免费、只罚 2–3 N·m 的姿态纠正尖峰（`rewards.py:78-95`；
  `VERSION.md:159-163`）。
* **wheel_axle**：φ 原本左右独立 → 策略用"一腿前一腿后"当额外 yaw 执行器，轮轴歪斜、
  横向刮擦；该项是物理量 `|L0_l sinφ_l − L0_r sinφ_r|`（米，L1）（`rewards.py:104-140`；
  `VERSION.md:178-181`）。
* **roll 拆两项**：两轮车过弯需要侧倾 `atan(vω/g)≈2.6°`，但常驻 10–20° 侧倾无益；
  `roll_straight` 用**指令**的 wz* 门控（不是实测角速度）（`rewards.py:443-466`；
  `VERSION.md:154-157`）。
* **action_jerk**：一阶差分罚所有运动（包括救姿态的快速腿部运动），二阶差分对平滑轨迹
  ≈0、只罚抖动（`rewards.py:34-49`；`mdp\actions.py:202-214`）。
* **动作范围/自由度**：故意不约束"左右腿长差"和"共同 φ"（坡面/台阶/倒立摆移 CG 必需）
  （`VERSION.md:165-189`）。

---

## 5. PPO 超参

### 5.1 配置（`A/agent.yaml` 与 `B/agent.yaml` 完全相同，唯一差异是 `resume`）

| 参数 | 值 | 行号（A=B，特殊注明除外） |
|---|---|---|
| seed / device | 42 / `cuda:0` | `:1-2` |
| num_steps_per_env | **48**（每轮 48×1024=**49,152** 样本） | `:3` |
| max_iterations | **1000**（CLI 覆盖值） | `:4` |
| save_interval | **50** | `:13` |
| clip_actions / check_for_nan | 4.0 / true | `:11-12` |
| 观测分组 | actor/critic 都用 `policy` | `:6-10` |
| actor / critic | `MLPModel`，hidden **[128,128,64]**，activation **ELU**，`obs_normalization: true` | `:24-30`、`:36-42` |
| 高斯策略 | `GaussianDistribution`，**init_std=0.5**，`std_type: scalar`（6 维） | `:32-34` |
| num_learning_epochs / num_mini_batches | **8 / 4** → 每轮 32 次梯度更新（minibatch 12,288 样本） | `:46-47` |
| learning_rate / schedule / desired_kl | 1e-3 / **adaptive** / 0.01 | `:48-49,53` |
| gamma / lam / clip_param | **0.995** / 0.95 / 0.2 | `:50-51,58` |
| entropy_coef | **0.003** | `:52` |
| max_grad_norm / optimizer | 1.0 / Adam | `:54-55` |
| value_loss_coef / clipped value loss | 1.0 / true | `:56-57` |
| normalize_advantage_per_mini_batch | false | `:59` |
| **resume** | A: **false**（`:19`）；B: **true**（`B/agent.yaml:19`） | 唯一差异 |

> `empirical_normalization: {}`（`agent.yaml:5`）为空（v1.0.11 之前的残留键），
> 实际归一化由 actor/critic 的 `obs_normalization: true` 生效（§2.1）。

### 5.2 网络 / 归一化 / 检查点证据

* actor：`Linear(35→128) → ELU → 128 → ELU → 64 → ELU → 6`；critic 同构但输出 1
  （检查点 `model_999.pt` 的 `actor_state_dict`：`mlp.0.weight (128,35)`、
  `mlp.2.weight (128,128)`、`mlp.4.weight (64,128)`、`mlp.6.weight (6,64)`、
  `distribution.std_param (6,)`）。
* `EmpiricalNormalization`：state_dict 含 `obs_normalizer._mean/_std (1,35)`；
  `count` 每迭代 +49,152（与 48×1024 一致）。
* 策略 std 硬限幅 **`_clamp_policy_std(runner, low=0.06, high=0.4)`**：
  实现 `isaac_ws\scripts\train.py:285-350`，在 `runner.learn()` 之前调用
  （`train.py:635`）。因此 `init_std=0.5` 实际从 **0.4** 起步（`model_0.pt` 实测
  std 均值 0.3928），A 末轮 `model_999.pt` 实测 0.0760，接近下限 0.06；
  A 最后 50 轮 `Policy/mean_std=0.0742`、B 最后 50 轮 0.0728
  （`report/data/metrics_summary.md:105`）。
* 限幅的文档表述：`VERSION.md:253` 写 `[0.06, 0.4]`（与代码一致）；
  但 `train.py:309` 的 docstring 仍写 "Since v1.0.5 the window is [0.03, 0.4]"
  （过时注释），`agents\rsl_rl_ppo_cfg.py:69` 注释同样写 `[0.03, 0.4]`（过时）。

### 5.3 续训 / max_iterations / no_code_state 语义

* **`--max_iterations` 是"再训多少轮"**：rsl_rl 的循环为
  `start_it = current_learning_iteration; total_it = start_it + num_learning_iterations`
  （`D:\miniconda\envs\RM\Lib\site-packages\rsl_rl\runners\on_policy_runner.py:77-79`），
  最终保存到 `model_{current_learning_iteration}.pt`（同文件 `:128-133`）。
  `train.py` 用 CLI 覆盖 cfg（`train.py:42` 默认 None、`:537-540`），然后
  `runner.learn(num_learning_iterations=agent_cfg.max_iterations)`（`train.py:673`）。
  因此：A 用 1000 → 跑全局迭代 0–999；B 用 1000 → 目标是 999–1998，但**实际只到 1123**（§6）。
  过程说明见 `TRAINING_GUIDE.md:193-213`。
* **`--resume` 恢复**：网络权重 + Adam 优化器状态 + 已完成轮次（
  `on_policy_runner.py:145-161`）；`train.py:590-598` 选择 checkpoint
  （`--resume_latest` → `train.py:207-275` 的 `_newest_checkpoint`，按数字取最大，
  规避 Isaac Lab `get_checkpoint_path` 字符串排序把 `model_0.pt` 排在 `model_900.pt` 之后的坑；
  `--resume` 不带 `--resume_latest` 时走 `get_checkpoint_path`）。
  续训前 `_match_network_to_checkpoint` 会从 checkpoint 读回 hidden_dims
  （`train.py:170-204`）。
* **`--no_code_state`**：跳过 rsl_rl 用 GitPython 写 git-diff 的步骤，避免仓库中一个
  删除的非 UTF-8 blob 触发 `UnicodeEncodeError` 而无法开训（`train.py:47-57`，
  清除逻辑 `:664-668`）。两次 run 的 `git` 子目录均为**空目录**（若有该步骤会写
  `git\<repo>.diff`，`rsl_rl\utils\logger.py:276-307`），说明两次都用了 `--no_code_state`；
  同时**训练时的工作树 diff 没有被保存下来**，A 的精确代码状态不可完整复原（只有 yaml 权威）。
* 因为 `--no_code_state` 在 learn 前清空 `git_status_repos`，所以 run 目录里只有空 `git/`。

---

## 6. 训练过程事实与时间线

### 6.1 时间线（文件时间戳 / git 提交 / 事件流）

| 时间（2026-10-22, +08:00） | 事件 | 证据 |
|---|---|---|
| **10-19 23:55:22** | 提交 `f99cddd`（"latest"，A 之前最后一次提交） | `git log --date=iso` |
| **00:28:38** | A 的 log 目录名生成（`datetime.now()`） | `train.py:566` |
| 00:28:40 | A `params\env.yaml`、`agent.yaml` 写出；`git\` 空目录 | 目录 mtime；`train.py:642-643` |
| 00:28:43 | A `model_0.pt`（iter=0）+ 第一条 TB 标量 | 文件时间；事件文件首条 wall time |
| 00:30:12 / … / 01:06:50 | A `model_50.pt` … `model_950.pt`（每 50 轮） | 文件时间 |
| **01:10:03** | A `model_999.pt`（终局保存，iter=999）；事件流止于 step 999 | 文件/事件 |
| **01:15:52** | 提交 `be7cc15`（"latest"，v1.0.8 代码+文档；两次 run 之间） | `git log --date=iso` |
| **01:17:16** | B 的 log 目录名生成（续训） | `train.py:566` |
| 01:17:18 | B `params\env.yaml`、`agent.yaml`（`resume: true`）写出 | 目录 mtime |
| 01:17:19 | B 事件文件创建（文件名 unix ts 1792603039）、空 `git\` | 文件名 |
| 01:17:24 | B `model_1000.pt`（iter=1000） | 文件时间 |
| 01:18:55 | B `model_1050.pt` | 文件时间 |
| **01:22:00** | B `model_1100.pt` | 文件时间 |
| **01:23:30** | B 最后一条事件（step 1123）；此后无任何新文件；**无 `model_1150.pt`** | 事件/目录清单 |

* A 目录存活 **41 min 25 s**（00:28:38→01:10:03）；B **约 6 min 14 s**（01:17:16→01:23:30，
  事件首末 01:17:22→01:23:30）。
* A 的 checkpoint 共 **21 个**（0,50,…,950 + 999）；B 共 **3 个**（1000/1050/1100）。

### 6.2 迭代次数 / 检查点硬件证据

* A 事件文件：标量 tag **39** 个；`Episode_Reward/*` **22** 个；step **0–999**（1000 个、
  连续）；**无 `Curriculum/*`**。
* B 事件文件：标量 tag **41** 个；`Episode_Reward/*` **24** 个；step **999–1123**（125 个、
  连续）；**无 `Curriculum/*`**；无 NaN/Inf。
* 检查点只读检查（`torch.load(..., weights_only=False)`）：
  * `model_999.pt`：`iter=999`，`obs_normalizer.count=49,152,000 = 1000×49,152`；
  * `model_1000.pt`：`iter=1000`，`count=49,250,304 = 1002×49,152`；
  * `model_1100.pt`：`iter=1100`，`count=54,165,504 = 1102×49,152`。
* 两次 run 合计覆盖全局迭代 **0–1123**（1124 个不同迭代；B 在 step 999 与 A 重叠 1 个，
  即 B 加载后**重跑了第 999 轮**再继续），共执行 **1125 次 PPO 迭代更新**。

### 6.3 续训事实

* B 的 `agent.yaml:19` 为 `resume: true`；**B 从 A 的 `model_999.pt` 续训**，证据链：
  1. B 事件流第一个 step 是 **999**（rsl_rl 从 checkpoint 的 `iter` 继续）；
  2. B 的第一个 checkpoint 是 `model_1000.pt`（iter=1000），说明循环从 999 开始、在
     it=1000 处首次满足 `it % 50 == 0`；
  3. `model_1000.pt` 的 `obs_normalizer.count` 比 `model_999.pt` 多 98,304 = 2×49,152
     （重跑 999 + 1000 两轮），与"从 iter=999 加载"完全吻合。
* 网络结构、观测布局（35 维）、`obs_normalization` 开关 A/B 相同，因此可结构兼容续训
  （`VERSION.md:321-328` 兼容性矩阵：观测维度变了才不能续）。
* `model_1000.pt` 与 `model_999.pt` 文件大小 734,709 vs 734,627 B（差在 obs_normalizer
  计数/统计量），不是重新初始化。

### 6.4 为什么 B 在 ~100 轮后停止（证据边界）

* 客观事实：事件流在 **step 1123 / 01:23:30** 突然结束；最后保存的 checkpoint 是
  `model_1100`（01:22:00）；目标终局 `model_1998` 不存在；run 目录内**没有**
  traceback/日志文件；事件标量**无 NaN/Inf**。
* 因此可确定：**不是数值发散导致的中断**；更符合"进程被终止/崩溃"（无正常收尾保存）。
  与 `VERSION.md:379-410` §8 记录的 **PhysX GPU 崩溃**（"CUDA error: an illegal memory
  access" 等，实际是 PhysX GPU 分配器问题；1024 环境 + 128 网络 + 默认 patch 缓冲实测
  存活 **16.2 分钟**，`VERSION.md:400`）模式一致；但**产物无法证明**具体崩溃类型，
  也无法区分"崩溃"与"手动 kill"——报告应表述为"因已知的 PhysX GPU 问题中断（无终局保存）"，
  不要断言 traceback 内容。
* 与存活时间对照：A 跑满 41 min 未崩，B 仅 6.2 min 就中断 → 该缺陷的存活时间不稳定
  （`VERSION.md:402` 也记录同开机内反复崩溃会持续缩短）。
* `save_interval=50` 的损失上界：A 正常结束无损失；B 最后保存 1100、事件止于 1123，
  即损失 ≤23 轮（最坏情况 ≤50 轮；`VERSION.md:406` 的"最多丢 2 分钟"为早期估计）。

### 6.5 未提交改动：不属于这两次 run

分析时 `git status`（HEAD=`be7cc15`）显示以下工作树改动，均为**训练之后**的
v1.0.9–v1.0.15 演进（`VERSION.md:87-92` 变更记录），不参与 A/B：

* `M VERSION.md`、`M TRAINING_GUIDE.md`（文档已更新到 v1.0.15）；
* `M isaac_ws/isaac_ws_infantry/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`
  （注释/文档随版本更新，超参不变）；
* `M isaac_ws/isaac_ws_infantry/tasks/locomotion/locomotion_env_cfg.py`
  （v1.0.10 倒车默认值、v1.0.11–v1.0.14 课程配置、v1.0.15 移除课程、pitch 0 等）；
* `M isaac_ws/scripts/train.py`（v1.0.11/12/13 控制台输出重排）；
* 说明：v1.0.11 新增的 `mdp/curriculums.py` 与 `mdp/__init__.py` 的课程导出在
  v1.0.15 中**已被删除/还原**（`VERSION.md:92`），故当前 `git status` 不再出现它们；
  写作过程中确实短暂出现过 `?? mdp/curriculums.py`。
* 变更记录中 **v1.0.9 缺失**（从 v1.0.8 直接跳到 v1.0.10，`VERSION.md:86-87`）——事实如此，
  报告不必展开。
* A 在 `be7cc15` 之前启动，运行的是后来被提交为 `be7cc15` 的工作树中**尚未包含 v1.0.8
  奖励改动**（22 项版本、pitch −4.0）的状态；因 `--no_code_state` 未存 diff，
  唯一权威记录就是 `A/env.yaml`。

### 6.6 速度 / 耗时事实

* 由事件墙钟：A `1000 轮 / 2480 s ≈ 2.48 s/轮`；B `124 个间隔 / 369 s ≈ 2.98 s/轮`。
* `Perf/collection_time`（A 前 100 → 后 50）**1.604 s → 3.542 s**；B 后 50 为 3.570 s；
  `Perf/total_fps` A 28086 → 12791（`report/data/metrics_summary.md:119-120`）→
  训练后期明显变慢（与 GPU/PhysX 压力累积一致）。
* 文档标称"约 1.94 s/轮"（`VERSION.md:260`、`TRAINING_GUIDE.md:186,547`）是早期/理想值，
  **不要用它推算这两次 run 的时长**；报告可用实测 2.5–3.0 s/轮。

---

## 7. 已知限制与风险

1. **单 GPU / 无法多卡**：只有 1 块 **RTX 4060 Laptop 8 GB**；Windows 版 PyTorch 无 NCCL
   （只有 gloo），而 rsl_rl 硬编码 `backend="nccl"` → 本机不可能多 GPU 并行
   （`VERSION.md:358-363`）。
2. **环境数上限 1024**：1 可用（仅回放）；**2–256 会卡死在 `gym.make()`**；1024 推荐；
   2048 约 24 min 崩；4096 约 5 min 崩（`VERSION.md:364-373`；`TRAINING_GUIDE.md:62-72`）。
3. **PhysX GPU 崩溃（未解决）**：见 §6.4；1024 环境建议重启机器后再长训
   （`VERSION.md:379-410`）。
4. **训练时长/轮数不足**：PPO 配置的 `max_iterations` 默认是 **8000**
   （`agents\rsl_rl_ppo_cfg.py:24`），实际 A 只跑 1000、B 续跑 125（§6），
   约为设计长度的 1/7；报告应把"最终策略"描述为**阶段性策略**。
5. **训练分布很窄**（这两次 run）：
   * `vx*` 仅 **[0.2, 1.2] m/s**，无倒车、无 0 速常规指令（0 速只来自 10% 站立环境，
     且其指令被置零）；`vy*≡0`；航向目标恒 0，**转弯/原地旋转不在训练分布**
     （§3.1-3.2）；
   * 平地单地形、无域随机化：`events` 只有 `reset_scene_to_default`（`A/env.yaml:454-461`），
     观测无噪声、无 corruption（`A/env.yaml:329`）；执行器参数固定 → sim2real 风险
     需要在报告中明说。
6. **评估协议受限**：回放/评估只能经 `play.py`，用 `--cmd_vx` / `--cmd_heading_deg`
   固定指令（`isaac_ws\scripts\play.py:56-71,269-290`），且 PLAY 配置 `num_envs=1`、
   不超时（`locomotion_env_cfg.py:500-512`）。**固定指令与训练时的随机指令/随机站立不同**，
   且超出训练范围（vx<0.2、vx>1.2、任意转弯、heading≠0）属于**分布外（OOD）**，
   失败不能直接归因于策略能力。
7. **指标解释注意**：`Metrics/base_velocity/error_vel_xy` / `error_vel_yaw` 的日志值
   是真实平均误差的 **2 倍**（command term 每步累积 `error/500`，`locomotion_env_cfg.py:282-288`；
   `TRAINING_GUIDE.md:252,284`）。另外 A/B 奖励权重不同（§4.2），
   **跨阶段的 `Train/mean_reward`、`Episode_Reward/*` 不可直接比较**，只能看同阶段趋势
   （`report/data/metrics_summary.md:38`）。
8. **只训练策略网络**：没有课程（§3.4）、没有感知/视觉输入、没有对价值网络之外的辅助任务；
   `empirical_normalization` 顶层键为空。
9. **roll 角可解释性**：两轮车允许过弯侧倾，但训练里没有"主动转弯"，所以 roll 主要来自
   扰动/建模误差，报告不宜把 roll 小解读为"学会了过弯"。

---

## 8. 报告可直接引用的要点（按章节）

**平台/任务**
1. 任务为 `Infantry-RL-Locomotion-v0`（Isaac Lab 管理器式 RL 环境 + rsl_rl PPO），机器人是
   五连杆闭链轮腿式 RoboMaster 步兵底盘：**11 个连杆、10 个自由度（4 髋 + 4 膝 + 2 轮）、
   总质量 12.776 kg**；膝不是自由度，闭环靠控制维持。
2. 关键几何：轮半径 **0.068 m**、名义腿长 **L0=0.249775 m**、名义站立高度 **0.189775 m**、
   髋轴高度 0.128 m；执行器：髋 40 N·m/16.755 rad/s、刚度 200/阻尼 5，膝 40 N·m、
   刚度 400/阻尼 10，轮 3.69 N·m/61.471 rad/s、阻尼 2、armature 0.001。
3. 仿真：**1024 个并行环境、物理 200 Hz（dt=5 ms）、控制 50 Hz（decimation=4）、
   20 s/回合（1000 控制步）**；终止条件为倾斜 >1.2 rad、高度 <0.11 m 或超时。
4. 观测 **35 维（10 项）**：重力投影 3 + 机体角速度 3（×0.25）+ 线速度 3 + 指令 2 +
   髋相对角 4 + 膝跟踪误差 4 + 轮速 2（×0.1）+ 上一步动作 6 + 目标腿状态 4 +
   腿长/腿角变化率 4；动作 **6 维**（4 髋位置偏移 ×0.25 rad + 2 轮速度 ×5 rad/s），
   输出先整体夹到 ±4。
5. 策略网络 **35→128→128→64→6**（ELU），actor/critic 均带 EmpiricalNormalization。

**指令/奖励**
6. 两次 run 的指令范围都是 **vx∈[+0.2,+1.2] m/s、vy=0、航向目标恒 0（转弯关闭）**，
   `wz* = clip(1.5·wrap(0−ψ), ±2.5)` 只用于纠偏；指令每 5–10 s 重采样，10% 环境为站立
   （刹车）样本。
7. 阶段 A（v1.0.7）奖励 **22 项**；阶段 B（v1.0.8）奖励 **24 项**（新增
   `track_lin_vel_x_l1=-2.0`、`track_ang_vel_z_l1=-1.0`；但 `pitch` 权重改为 **0**，
   实际贡献 23 项）。
8. 主要奖励权重（B）：`alive +0.5`、`fall_penalty −50`、`base_height −4.0`、`track_lin_vel_x
   +8.0(std 0.15)`、`track_ang_vel_z +3.0(std 0.15)`、`track_lin_vel_x_l2 −6.0`、
   `track_ang_vel_z_l2 −2.0`、`wheel_axle −8.0`、`wheel_torques −0.05(死区1.5 N·m)`、
   `action_jerk −0.3`、`closure_residual −1000`。
9. A→B 的 v1.0.8 改动是"**精度跟踪**"：跟踪核从 std 0.25/0.35 收窄到 0.15、主跟踪权重
   6.0→8.0、1.0→3.0，新增两条 L1 恒定梯度项，L2 加严 −3.0→−6.0、−1.0→−2.0，同时放宽
   `action_jerk`(−0.5→−0.3) 与轮力矩死区(1.0→1.5 N·m)。（另有一条文档未记录的
   `pitch −4.0→0` 改动。）
10. 证据表明两次 run **没有使用课程/自适应奖励**（`curriculum: null`、事件文件 0 条
    `Curriculum/*`），奖励是静态的；课程模块是训练之后才加入的。

**PPO/训练过程**
11. PPO：每轮 48 步 ×1024 环境 = **49,152 样本**，4 个 minibatch、8 个 epoch（32 次梯度
    更新），lr=1e-3（adaptive，desired_kl=0.01），γ=0.995、λ=0.95、clip=0.2、
    entropy=0.003、max_grad_norm=1.0；策略 std 硬限幅 **[0.06, 0.4]**（初始 0.5 → 0.4）。
12. 阶段 A：`2026-10-22 00:28:38` 启动，**迭代 0–999 全部完成**，`model_999.pt` 于
    **01:10:03** 写入（正常收尾），历时约 41 分钟，平均约 2.5 s/轮。
13. 两次 run 之间（**01:15:52**）提交了 `be7cc15`（v1.0.8）；阶段 B 于 **01:17:16**
    启动、`resume: true`，**从 `model_999.pt` 续训**（事件流首步 999、`model_1000.pt`
    iter=1000）。
14. 阶段 B 实际只执行了 **125 轮迭代**（全局迭代 999–1123，其中 999 为加载后重跑），
    在 **01:23:30** 事件流突然中断；最后 checkpoint 为 `model_1100.pt`（01:22:00），
    无 `model_1150.pt`，也无目标终局保存。中断时无 NaN/Inf，符合已知 PhysX GPU 崩溃
    特征（非数值发散）；因 `save_interval=50`，最多损失 50 轮（本次损失 ≤23 轮）。
15. 合计有效训练：**1125 次 PPO 迭代更新、约 5.5×10⁷ 环境步**（1125×49,152≈5.53e7），
    但对 8000 轮的配置目标只完成约 1/7。
16. 训练/评估硬件限制：单卡 **RTX 4060 Laptop 8 GB**，Windows 无 NCCL 不能多卡；
    2–256 环境会卡死，1024 为上限推荐值；长训会因 PhysX GPU 缺陷周期性崩溃，
    必须靠 `save_interval=50` + `--resume` 续训。

**评估/结论引用**
17. 评估只能通过 `play.py --cmd_vx <v> --cmd_heading_deg <deg>` 固定指令跑（默认 1 个环境），
    与训练时的随机指令分布不同；训练分布仅 **vx∈[0.2,1.2]、航向恒 0**，
    因此倒车、停车、转弯、超速测试都属于**分布外**，其失败不能直接归因于策略。
18. 跟踪误差指标 `error_vel_xy/error_vel_yaw` 的日志值是真实均值的 2 倍（如日志 0.05
    = 真实 0.025 m/s），引用时需注明。
19. 训练后期速度明显下降（`collection_time` 1.6 s→3.5 s/轮），说明训练吞吐受 GPU/PhysX
    资源约束，长训的时间成本被低估。
20. 这两次 run 的最终策略是"**直线速度跟踪 + 保持航向的阶段策略**"，不是完整的转向/倒车
    能力；报告结论应限定在此范围内。

---

## 9. 引用文件清单（可溯源）

| 文件 | 用途 |
|---|---|
| `logs\rsl_rl\infantry_rl_locomotion\2026-10-22_00-28-38\params\env.yaml` | A 的全部环境配置（权威） |
| `logs\rsl_rl\infantry_rl_locomotion\2026-10-22_01-17-16\params\env.yaml` | B 的全部环境配置（权威） |
| 两 run 的 `params\agent.yaml` | PPO/网络超参（权威） |
| `logs\...\2026-10-22_00-28-38\events.out.tfevents.1792600120.CMD.10812.0` | A 的 1000 轮标量（step 0–999） |
| `logs\...\2026-10-22_01-17-16\events.out.tfevents.1792603039.CMD.12320.0` | B 的 125 轮标量（step 999–1123） |
| `logs\...\model_999.pt` / `model_1000.pt` / `model_1100.pt` | iter / obs_normalizer 证据 |
| `isaac_ws\assets\infantry-rl\urdf\infantry-rl-sim.urdf` | 质量/连杆/关节限位/闭环定义 |
| `isaac_ws\config\nominal_pose.json`、`planar_leg.json` | L0=0.249775、髋轴高度 0.128、名义关节角 |
| `isaac_ws\isaac_ws_infantry\robot_model.py` | 轮半径、名义站高公式、关节顺序 |
| `isaac_ws\isaac_ws_infantry\robots\infantry_rl.py` | 执行器配置、初始姿态 |
| `isaac_ws\isaac_ws_infantry\mdp\actions.py`、`observations.py`、`rewards.py` | 实现细节（`curriculums.py` 已于 v1.0.15 删除，本档案仅作历史提及） |
| `isaac_ws\isaac_ws_infantry\tasks\locomotion\locomotion_env_cfg.py` | 任务配置（当前工作树行号） |
| `isaac_ws\scripts\train.py`、`play.py`、`cli_args.py` | 训练/评估流程、std 限幅、resume |
| `VERSION.md`、`TRAINING_GUIDE.md` | 版本/流程文档（工作树已更新至 v1.0.15，含 §3.3 列出的过时项） |
| `report\data\metrics_summary.md`、`run_ranges.json`、`run_A_scalars.csv`、`run_B_scalars.csv` | 训练曲线/指标（数据组产出，本档案引用其汇总值） |

> 版本快照命令：`git show be7cc15:VERSION.md`、`git show be7cc15:isaac_ws/scripts/train.py`；
> `git log --format='%h %ad %s' --date=iso`。
