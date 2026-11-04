# squat 实机部署方案：无 base_height、无膝关节编码器

本文是 `exports/chassis_squat_model_598.onnx`（chassis 资产、squat 任务、model 598）
的下位机部署定稿。目标：

- 下位机不安装高度传感器；
- 下位机不读取膝关节角度；
- 保持与训练时完全一致的 30 维观测接口；
- 为后续开放髋关节留出扩展路径，但不在当前版本提前引入解算开销。

---

## 1. 最终方案（结论）

1. **ONNX 接口保持不变**：单输入 `obs[1,30]`，单输出 `action[1,2]`。
   - 归一化器已经折叠进第一层 `Gemm`，下位机送**原始观测值**，不要再做归一化。
   - 输出是原始策略均值，后处理为：
     `wheel_target = 5.0 * clamp(action, -6.0, +6.0)`（rad/s）。
   - 控制频率 50 Hz。

2. **当前锁髋 squat：缺失的 9 维全部用常量填充**，下位机不做五连杆解算。
   - `17:21 knee_pos_rel_solved`：常量；
   - `21:25 solved_leg_state`：常量；
   - `29 base_height_rel`：常量。
   - 常量取**训练均值**，不是 0。填 0 会引入可测的动作偏置；填均值等价于把这几维
     mask 在训练分布中心，标准化输入为 0。

3. **下位机只保留必要计算**：
   - IMU：重力投影、角速度、yaw；
   - 4 个髋编码器：`hip_meas - NOMINAL_HIP`；
   - 2 个轮编码器：`wheel_vel * 0.1`；
   - 上一帧动作；
   - 常量表。

4. **不建议**把常量折进 ONNX 第一层 bias。
   - 只省几十次乘加，收益很小；
   - 后续开放髋后观测接口会变，维护两套导出不划算；
   - 等锁髋版本接口冻结、准备量产时再考虑。

5. **后续开放髋关节时**：
   - 五连杆解算放在**下位机 C 代码**里，用**闭式解**（圆-圆交点），不要放进 ONNX；
   - 保留 `HIPS_LOCKED / HIPS_OPEN` 模式开关：锁髋时跳过解算，开放髋时才运行；
   - 开放髋后观测接口大概率扩到 34–35 维，需要重新训练/导出，是另一个 ONNX；
   - 膝关节跟踪误差在实机上仍然 ≈0（机械闭链），但建议用仿真微调消除 sim2real 差。

6. **上线前建议做一次仿真微调**：
   - 在仿真里把缺失维替换成固件实际会喂的常量/估计值，加噪声和 1 步延迟，
     从 `model_598` 微调 100–300 iter；
   - 目的是消除“训练时有真实值、部署时喂常量”的分布差；
   - 微调后重新导出 ONNX，并重新提取常量表。

7. **安全终止不依赖 base_height**：
   - 沿用 `pitch/roll/bad_orientation`；
   - 增加髋跟踪误差、轮电流、角速度、超时等替代条件；
   - 首次上机使用吊带/台架和急停。

---

## 2. 当前 ONNX 的事实

文件：`exports/chassis_squat_model_598.onnx`

```text
INPUT : obs    [1, 30]
OUTPUT: action [1, 2]
NODES : Gemm -> Elu -> Gemm -> Elu -> Gemm -> Elu -> Gemm
OPSET : 17
```

来源 checkpoint：

```text
logs/rsl_rl/infantry_rl_squat/2026-11-03_23-02-16_chassis_squat_s1_cont/model_598.pt
```

关键语义：

- 输出是**确定性策略均值**，不含环境后处理；
- squat 的 hips 是锁定的常量姿态；knees 由五连杆闭链确定；
- 动作只有 2 维轮速目标，不是 6 维。

---

## 3. 30 维观测布局与下位机实现

| slice | 观测项 | 训练 scale | 下位机实现 |
|---|---|---:|---|
| 0:3 | projected_gravity | 1 | IMU 重力投影 |
| 3:6 | base_ang_vel | 0.25 | 陀螺仪 × 0.25 |
| 6:9 | base_lin_vel | 1 | 线速度估计（IMU/轮速） |
| 9:11 | heading `(sin, cos)` | 1 | `sin(yaw), cos(yaw)`，目标 0 |
| 11:13 | position error `(fwd, left)` | 1 | `目标 - 当前`，旋转到机体系；目标 `(0,0)` |
| **13:17** | hip_pos_rel | 1 | `hip_meas - NOMINAL_HIP` |
| **17:21** | knee_pos_rel_solved | 1 | **常量** |
| **21:25** | solved_leg_state | 1 | **常量** |
| 25:27 | wheel_vel | 0.1 | 轮编码器速度 × 0.1 |
| 27:29 | last_action | 1 | 上一帧 ONNX 输出（clip ±6 后） |
| **29:30** | base_height_rel | 1 | **常量** |

### 3.1 关节顺序与符号

```text
HIP_ORDER  = (left_front_hip,  left_back_hip,  right_front_hip,  right_back_hip)
KNEE_ORDER = (left_front_knee, left_back_knee, right_front_knee, right_back_knee)
WHEEL_ORDER= (left_wheel, right_wheel)
```

chassis 资产：

```text
right_hip_sign = -1
```

右腿的关节值相对左腿镜像，固件必须使用与训练相同的符号约定。

### 3.2 标称髋角

```text
NOMINAL_HIP = [-0.68246, +0.682780375, +0.68246, -0.682780375]  # lf, lb, rf, rb
```

注意：`13:17` 在 squat 配置里虽然项名叫 `hip_pos_rel_locked`，实际实现是
`hip_pos_rel_nominal`，即 **`measured_hip - nominal_hip`**，不是
`measured_hip - squat_hip`。两者相差约 0.67 rad，弄错是严重的分布偏移。

### 3.3 `last_action` 语义

`last_action` 是上一帧经过训练同款 clip（squat 为 ±6）后的**原始动作单位**，
不是乘 5 之后的轮速目标：

```text
last_action = clamp(previous_onnx_action, -6.0, +6.0)
wheel_target = 5.0 * last_action
```

---

## 4. 缺失量处理

### 4.1 膝关节（17:21）

实机每条腿只有 2 个髋电机，膝盖是被动关节、机械闭链，膝角由两个髋角唯一确定。
仿真里的 `knee_pos_rel_solved` 是“实测膝角 − 解算膝角”，本质是 PhysX 位置伺服的
跟踪误差，在训练中近似常量：

```text
mean ≈ ±0.006 rad
std  ≈ 0.0008 rad
```

实机刚性闭链下真实误差 ≈ 0，因此可以：

- 填训练均值（推荐，标准化偏移为 0）；
- 或填 0；
- 不需要膝编码器，也不需要在下位机运行解算。

squat 姿态的膝角常量（`robot_data/chassis-rl/squat_pose.json`）：

```text
left : front_knee = +0.00992, back_knee = -0.00995 rad
right: front_knee = -0.00992, back_knee = +0.00995 rad
```

### 4.2 base_height（29:30）

轮子接地、地面水平时，高度由腿几何和姿态唯一确定，不需要高度传感器。

用髋编码器得到每侧 `(L0, phi)` 后：

```text
r_body = ( x_mid + L0*sin(phi),  y_wheel,  h_hip - L0*cos(phi) )
base_z = wheel_radius - [ R(roll,pitch,yaw) * r_body ]_z
base_height_rel = base_z - SQUAT_BASE_HEIGHT
```

chassis 几何常量：

```text
wheel_radius = 0.068 m
h_hip        = 0.030406 m
x_mid        = -0.003475 m
y_wheel      = ±0.219 m
SQUAT_BASE_HEIGHT = 0.172990 m
```

水平、无姿态修正时的简化式：

```text
base_z ≈ wheel_radius + L0*cos(phi) - h_hip
squat 姿态: 0.068 + 0.1354 - 0.0304 = 0.1730 m
```

在本任务的安全姿态范围内（pitch ≤ 0.24、roll ≤ 0.30 rad），纯运动学高度变化只有
约 2–5 mm；训练中该维 std 仅 1 mm。因此：

- 最简：直接填训练均值常量；
- 需要高度信息时：用上面的公式 + IMU 姿态；
- **不要**用 IMU 加速度二次积分求高度（漂移）。

### 4.3 常量表（model_598 / chassis 专用）

以下均为**原始观测单位**，直接填入 ONNX 输入：

```text
# 17:21  knee_pos_rel_solved  [lf, lb, rf, rb]  (rad)
KNEE_ERR = [-0.006292638, +0.005624739, +0.006758041, -0.006178977]

# 21:25  solved_leg_state  [L0_l, phi_l, L0_r, phi_r]  (m, rad)
SOLVED_LEG = [0.135391533, -0.000005311, 0.135391921, -0.000008040]

# 29     base_height_rel  (m)
BASE_HEIGHT_REL = -0.002473299
```

这些值来自 `model_598.pt` 的 `obs_normalizer._mean`，不是物理标定值。
**换 checkpoint 或换资产必须重新提取。**

### 4.4 为什么不用 0 填充

在训练均值状态附近做过一次线性化探测：

- 5 维全填 0：动作偏移约 `[-0.066, -0.165]` 原始单位，
  即约 `[-0.33, -0.83]` rad/s 的轮速目标；
- 填训练均值：标准化输入偏移为 0，没有额外偏置。

该数字只是数量级参考，最终应以仿真消融为准；但结论是：

> **填常量和填 0 的运行时代价完全一样，所以优先填训练均值。**

### 4.5 重新提取常量的方法

```python
import torch

ckpt = torch.load(
    "logs/rsl_rl/infantry_rl_squat/2026-11-03_23-02-16_chassis_squat_s1_cont/model_598.pt",
    map_location="cpu",
    weights_only=False,
)
mean = ckpt["actor_state_dict"]["obs_normalizer._mean"].flatten()
std = ckpt["actor_state_dict"]["obs_normalizer._std"].flatten()

knee_err = mean[17:21]
solved_leg = mean[21:25]
base_height_rel = mean[29:30]
```

---

## 5. 后续开放髋关节

### 5.1 接口变化

开放髋后，以下内容会变化：

| 内容 | 锁髋 squat | 开放髋后 |
|---|---|---|
| action | 2 维轮速 | 6 维（4 hips + 2 wheels） |
| last_action | 2 维 | 6 维 |
| solved_leg_state | 常量 | **动态，必须解算** |
| hip_pos_rel | 实测 − nominal | 动态，但仍是减法 |
| knee_pos_rel_solved | 常量 | 实机仍 ≈0（机械闭链）；仿真训练误差需单独处理 |
| leg_rates | 无 | 如果采用 locomotion 风格观测，需要差分 |
| 观测维度 | 30 | 约 34–35，需重新训练/导出 |

当前常量表**不能**直接用于开放髋版本。

### 5.2 固件架构

用一个模式开关分开，固件只维护一套观测拼装逻辑：

```text
if HIPS_LOCKED:
    knee_err      = KNEE_ERR_CONST
    solved_leg    = SOLVED_LEG_CONST
    base_height   = BASE_HEIGHT_REL_CONST
    # 不运行五连杆解算
else:
    knee_est      = five_bar_solve(hip_meas)      # 闭式解，每周期一次
    solved_leg    = leg_state(hip_meas, knee_est)
    knee_err      = knee_est - knee_cmd           # 实机 ≈0
    leg_rates     = (solved_leg - prev) / dt      # 需要保留上一帧
    base_height   = 运动学估计或保留常量
```

### 5.3 解算器选择

- 优先**闭式解**（两圆交点，`atan2/acos/sqrt`），不要用迭代 Newton；
- 闭式解无收敛失败、算力固定、分支显式；
- 用 `kinematics/planar_leg.py` 作为参考实现生成测试向量；
- 随机采样髋角范围（约 ±0.7 rad），固件与 Python 参考误差应 < 1e-5 rad；
- 闭链有两个装配分支，必须固定在训练使用的分支（squat 姿态的膝角可作为 warm start / 查表基准）。

### 5.4 不要把解算放进 ONNX

- ONNX 可以有多输入，也可以包含三角函数，但这不会减少总计算量；
- Cube.AI 对 `atan2/acos` 等算子的支持和性能不确定；
- 迭代/分支逻辑放进计算图后很难调试；
- 五连杆解算在 MCU 上只有微秒级开销，留在 C 代码里更可测试。

---

## 6. 算力分析

以 model_598（30 → 128 → 128 → 64 → 2）为例：

| 计算 | 每周期开销 |
|---|---|
| MLP 推理 | 约 28.5k 次乘加（≈57 kFLOP） |
| 五连杆闭式解 | 每腿几次 `sqrt/atan2/acos` + 基本运算 |
| `leg_rates` 差分 | 每腿几次减法 |
| 常量填充（当前锁髋） | 0 次解算 |

结论：

- 锁髋版本几乎零解算；
- 开放髋后解算通常小于 NN 推理的 1%；
- 瓶颈是 NN、三角函数库和内存带宽，不是五连杆；
- 50 Hz 下无需为解算做特殊优化；如无 FPU 再考虑查表。

---

## 7. 安全与终止

仿真的 `base_too_low` 依赖 base 高度：

```text
BASE_FLOOR(chassis) = 0.13299 m
```

实机没有高度传感器时，不要直接依赖该条件，改用组合判据：

1. 现有 `pitch_limit = 0.24 rad`、`roll_limit = 0.30 rad`、`bad_orientation = 1.2 rad`；
2. pitch/roll 角速度阈值；
3. 髋跟踪误差 `|hip_meas - hip_cmd|`，建议从 0.1 rad 起标定；
4. 轮子指令接近 0 但电流持续偏大（堵转/打滑/趴地）；
5. 有运动学高度估计时，低通后再比较 floor；
6. 控制循环超时/看门狗；
7. 首次上机：吊带/台架 + 急停；限制轮速 scale 或动作 clip。

---

## 8. 验证计划

### 8.1 仿真消融（不改策略）

分别用以下填充方式跑 1000+ 步：

1. 填 0；
2. 填训练均值；
3. 填运动学估计值。

记录：

- 与原始策略的动作 L2 差；
- episode 存活时间；
- base_height / knee_err 的真实分布。

判断标准（建议）：

- 动作差小于 clip 范围的约 5%，且存活率不下降 → 常量填充可用；
- 否则进入 8.2 的微调。

### 8.2 部署一致微调（推荐）

在仿真里把缺失维替换成固件实际喂的值，并加入：

- base_height 噪声约 ±1–2 mm；
- knee_err 噪声约 ±0.002–0.005 rad；
- 1 步观测延迟；
- 以 p≈0.5 随机 mask 这几维。

从 `model_598` 微调 100–300 iter，学习率约 1e-4，奖励不变。
导出后：

- `onnx.checker` 通过；
- 与 checkpoint 参考实现误差 < 1e-4；
- 若微调改变了常量，重新提取常量表。

### 8.3 下位机单元测试

- 闭式五连杆解与 `kinematics/planar_leg.py` 对比，误差 < 1e-5 rad；
- 观测拼装逐维和 Python 参考对比；
- 用一组固定输入跑 ONNX golden vector，逐位/定点对比。

### 8.4 实机观测检查

- 静止/手动摆姿态，记录 30 维观测；
- 用 checkpoint 的 `obs_normalizer._mean/_std` 计算 z-score；
- 各维 |z| 应在约 3 以内；
- 特别检查 13:17（nominal 参考系）、9:11（yaw 零位）、27:29（clip 语义）。

### 8.5 台架闭环

- 吊带/台架，先小轮速 scale，再逐步放开；
- 监控姿态、轮速、电流、髋跟踪误差；
- 确认异常时安全终止能及时触发。

---

## 9. 实施步骤

| 阶段 | 内容 | 产出 |
|---|---|---|
| 0 | 从 checkpoint 提取常量，按本文档实现固件观测拼装 | 30 维观测 buffer |
| 1 | 仿真消融（0 / 均值 / 估计） | 是否可直接上机的结论 |
| 2 | （推荐）仿真微调 + 重新导出 | 部署一致的新 ONNX |
| 3 | 台架/吊带验证 | 观测 z-score、闭环稳定 |
| 4 | 开放髋时增加闭式五连杆 + 模式开关 | 34/35 维新 ONNX |

---

## 10. 风险与对齐点

| 风险 | 说明 |
|---|---|
| 髋角参考系 | `13:17` 必须用 `measured_hip - NOMINAL_HIP`，不是减 squat 姿态；差值约 0.67 rad |
| 右腿镜像 | chassis `right_hip_sign = -1`，关节顺序必须一致 |
| 闭链分支 | 五连杆有两个装配解，开放髋后必须固定在训练分支 |
| yaw/位置原点 | 上电/复位时 IMU yaw 和里程计原点清零 |
| last_action | 上一帧 **clip 后原始动作**，不是轮速目标 |
| 归一化 | 已折叠进 ONNX，送原始观测值，不要重复归一化 |
| 常量绑定 | 常量来自 model_598/chassis，换 checkpoint/资产必须重新提取 |
| 高度假设 | 运动学估计假设轮子接地、平地、连杆刚性；软地面/轮胎压缩会偏 |
| 其他观测 | 若 `base_lin_vel`、`position` 也无法获取，需要另做里程计/估计方案 |
| 微调一致性 | 若做了部署一致微调，常量必须改成微调时实际使用的值 |

---

## 11. 一句话总结

**当前锁髋版本：30 维单输入 ONNX + 固件常量填充，几乎不做解算。
开放髋后：同一套固件加闭式五连杆和模式开关，解算留在 C，NN 留在 ONNX。
上线前建议做一次部署一致的仿真微调。**
