# squat_torque ONNX 实机部署说明（25 维观测）

任务只有 `squat_torque` 一个：

```text
6 个执行器 = 4 个髋电机 + 2 个轮电机，全部力矩控制
膝盖 = 被动机械闭链，不参与动作，也不需要观测
观测 = 25 维，全部是实机可直接测量的量
```

---

## 1. 导出

```bat
C:/Users/32560/.conda/envs/isaaclab/python.exe tools/export_policy_onnx.py ^
  --checkpoint logs/rsl_rl/infantry_rl_squat_torque/<run>/model_<N>.pt ^
  --output exports/chassis_squat_torque.onnx
```

### 1.1 ONNX 接口

```text
INPUT : history [1, 125] float32      # 5 帧 × 25 维，oldest -> newest
OUTPUT: action  [1, 6]   float32
GRAPH : Sub/Div -> encoder(125->128->64->3) + current 25 -> actor(28->...->6)
OPS   : Concat / Div / Elu / Gemm / Slice / Sub
```

- 归一化器以显式 Sub/Div 保留在图中，固件送**原始（已乘 scale）的 125 维历史**；
- 输出是确定性策略均值，**原始动作单位**；
- 输出不含 clip、不含 Nm 换算、不含电流换算。

旧版 25/42 维前馈 checkpoint 与本文不兼容，导出脚本会直接报错。

### 1.1.1 历史怎么组

固件维护 5 帧环形缓冲，每帧 25 维：

```text
history = [o_{t-4}, o_{t-3}, o_{t-2}, o_{t-1}, o_t]      # 125
```

上电/复位时用当前帧填满 5 帧。ONNX 内部：

```text
x       = (history - mean) / (std + eps)
current = x[-25:]                    # 当前帧
latent  = encoder(x)                 # 3 维
action  = actor([current, latent])   # 28 -> 6
```

### 1.2 动作后处理（参考 wheel_legged 关节空间）

```text
a = clamp(action, -4.0, 4.0)                 # PPO clip_actions = 4.0

# 髋：位置目标 + PD -> 力矩
q_ref = SQUAT_HIPS + a[0:4] * 0.5                     # rad
tau_hip = 40.0 * (q_ref - q_hip) - 1.0 * qdot_hip     # Nm
tau_hip = clamp(tau_hip, -40.0, +40.0)
I_hip   = clamp(tau_hip / Kt_eff_hip, I_MIN, I_MAX)
send_current_hip(I_hip)

# 轮：速度目标 + 阻尼 -> 力矩
v_ref = a[4:6] * 10.0                                 # rad/s
tau_wheel = 0.5 * (v_ref - wheel_vel)                 # Nm
tau_wheel = clamp(tau_wheel, -3.69, +3.69)
I_wheel = clamp(tau_wheel / Kt_eff_wheel, I_MIN, I_MAX)
send_current_wheel(I_wheel)

last_action = a                              # 下一帧反馈用的是 clip 后的 a
```

动作顺序：

```text
[hip_offset_lf, hip_offset_lb, hip_offset_rf, hip_offset_rb, wheel_vel_l, wheel_vel_r]
```

`Kt_eff` 是折算到关节侧的力矩常数：

```text
直驱：Kt_eff = Kt_motor
减速箱：Kt_eff = Kt_motor * N * eta
```

**重要**：`last_action` 是 `a`（clip 后的原始动作），不是 `tau`，也不是电流。

---

## 2. 25 维/帧 × 5 帧历史

| slice（每帧内） | 项 | dim | 训练 scale | 下位机实现 |
|---|---|---:|---:|---|
| 0:3 | base_ang_vel | 3 | 0.25 | 陀螺仪 × 0.25 |
| 3:6 | projected_gravity | 3 | 1 | IMU 重力投影（机体系） |
| 6:9 | command | 3 | 1 | `[vx, vy, wz]`，squat 任务恒为 0 |
| 9:13 | hip_pos_rel_locked | 4 | 1 | `hip_meas - LOCKED_HIPS` |
| 13:19 | joint_vel | 6 | 0.05 | 4 个髋 + 2 个轮速度，位置差分 × 0.05 |
| 19:25 | last_action | 6 | 1 | 上一帧 clip ±4 后的原始动作 |

### 2.1 关节顺序与符号

```text
HIP_ORDER   = (lf, lb, rf, rb) = (left_front, left_back, right_front, right_back)
WHEEL_ORDER = (left_wheel, right_wheel)
joint_vel   = [hip_lf, hip_lb, hip_rf, hip_rb, wheel_left, wheel_right] * 0.05
```

chassis 资产：

```text
right_hip_sign = -1
```

右腿相对左腿镜像，固件必须使用与训练相同的符号。

### 2.2 `LOCKED_HIPS` 常量

来自 `robot_data/chassis-rl/squat_pose.json`（运行环境会打印实际值）：

```text
LOCKED_HIPS = [-0.011721, +0.011800, +0.011721, -0.011800]   # lf, lb, rf, rb
```

`hip_pos_rel_locked` 是 **`hip_meas - LOCKED_HIPS`**，不是相对于 nominal 髋角。

### 2.3 command

squat 任务的命令项恒为 `[0, 0, 0]`，固件直接填 0 即可。以后如果训练速度
跟踪任务，这一项就是目标 `[vx, vy, wz]`（机体系）。

### 2.4 关节速度的算法

训练默认和参考仓库一致，用**编码器位置差分**，而不是驱动器内部速度：

```text
q      = [hip_lf, hip_lb, hip_rf, hip_rb, wheel_left, wheel_right]
q_dot  = (q - q_prev) / dt            # dt = 控制周期 10 ms
obs[13:19] = q_dot * 0.05
q_prev = q

# 编码器噪声大时可加一阶低通（训练时用同一 alpha）：
# q_filt = alpha * q_filt + (1 - alpha) * q_dot
# obs[13:19] = q_filt * 0.05
```

上电/复位时先令 `q_prev = q`，第一帧速度填 0。轮子是连续关节，位置不回绕；
髋有机械限位，也不需要处理 ±π 跳变。

### 2.5 训练噪声

训练时会对观测加均匀噪声（IMU、髋角、关节速度），**部署时不需要加**：

```text
ang_vel    ±0.2 rad/s   -> 观测中 ±0.05
gravity    ±0.05
hip_pos    ±0.01 rad
joint_vel  ±1.5 rad/s   -> 观测中 ±0.075
command / last_action    不加噪声
```

### 2.6 训练时的域随机化（部署端不需要实现）

策略训练时已经见过以下扰动，部署端不需要额外注入，但**不要假设真实机器人
一定比训练范围更理想**：

```text
摩擦 0.4-1.2 / 恢复系数 0-0.2 / 基座质量 -2..+3 kg / 质心 ±5 cm
膝 PD 增益 ×0.9-1.1 / 每电机力矩 ×0.9-1.1
随机推力 0.3 m/s，每 5-8 s
动作延迟 0-1 个控制步（50 Hz，即 0-20 ms）
reset 髋角 ±0.01 rad
```

实机的通信/电流环延迟和电机力矩误差天然存在，正好落在训练范围内。
如果实机延迟明显大于 20 ms，应把 `INFANTRY_RL_ACTION_DELAY_MAX` 调大后重训。

### 2.7 不需要的量

以下内容**已经从观测中删除**，固件不需要实现：

- 线速度估计、heading sin/cos、位置误差；
- 膝角 / 膝跟踪误差 / 五连杆 `L0, phi` / `leg_rates`；
- 机体高度。

---

## 3. 固件主循环伪代码

```text
# 100 Hz
hip      = read_hip_pos()                    # 4, 编码器
wheel    = read_wheel_pos()                  # 2, 编码器（连续关节）
q        = [hip, wheel]                      # 6
q_dot    = (q - q_prev) / dt                 # 位置差分，dt = 10 ms
# q_filt = alpha * q_filt + (1 - alpha) * q_dot   # 可选一阶低通

frame[0:3]   = gyro * 0.25
frame[3:6]   = gravity_projection(imu_quat)
frame[6:9]   = [0.0, 0.0, 0.0]               # squat 任务
frame[9:13]  = hip - LOCKED_HIPS
frame[13:19] = q_dot * 0.05                  # 或 q_filt * 0.05
frame[19:25] = last_action

# 5 帧环形缓冲，oldest -> newest
history = push_history(frame)                # 125

a     = onnx_run(history)                    # 6, 原始动作单位
a     = clamp(a, -4, 4)

q_ref = SQUAT_HIPS + a[0:4] * 0.5             # rad
v_ref = a[4:6] * 10.0                         # rad/s

tau_hip = clamp(40.0*(q_ref - hip) - 1.0*hip_vel, -40, 40)
tau_whl = clamp(0.5*(v_ref - wheel_vel), -3.69, 3.69)
I_hip = clamp(tau_hip / KT_EFF_HIP, I_MIN, I_MAX)
I_whl = clamp(tau_whl / KT_EFF_WHEEL, I_MIN, I_MAX)
send_current([I_hip, I_whl])
last_action = a
```

上电/复位时：

```text
last_action = 0
q_prev      = q
history     = [frame, frame, frame, frame, frame]
```

髋角反馈与 `LOCKED_HIPS` 必须使用同一套关节零位和符号。

---

## 4. Python 端快速验证

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(
    "exports/chassis_squat_torque.onnx",
    providers=["CPUExecutionProvider"],
)

history = np.zeros((1, 125), dtype=np.float32)   # 5 x 25
action = sess.run(None, {"history": history})[0]

print(action.shape)      # (1, 6)
print(action)
```

注意：

- 输入名 `"history"`，输出名 `"action"`；
- 输入必须是 `float32`，顺序 oldest -> newest；
- 归一化已经在模型里，**不要**再减 mean/除 std；
- 输出是原始动作单位，不是 Nm。

### 4.1 用 sim 的观测做 golden vector

1. 在 `play.py --task squat_torque` 里记录一帧真实的 125 维历史和策略输出；
2. 把同一帧历史喂给 ONNX；
3. 比较 ONNX 输出与 sim 内策略输出；
4. 再把动作按 `clip -> Nm` 后处理，确认力矩在范围内。

---

## 5. 扭矩到电流

```text
I = tau / Kt_eff
```

需要标定：

| 量 | 方法 |
|---|---|
| `Kt_eff` | 台架/堵转，测不同电流下的关节力矩，线性拟合 |
| 死区 | 从小电流逐步增加，记录起动电流 |
| 摩擦 | 自由轮/自由髋在不同速度下的阻力矩 |
| 电流环延迟 | 电流阶跃响应，拟合一阶滞后 + 延迟 |
| 电压饱和 | 不同转速下最大可达电流 |
| 符号 | 正电流必须对应关节正方向 |

轮子 3.69 Nm 的限幅要换算成电流，确认驱动器能持续/峰值输出。

---

## 6. 安全

底层 PD 失效时没有隐式阻尼，**零力矩不是安全状态**：

- 看门狗：通信超时/姿态超限/过流/过温 → 切到简单阻尼
  `tau = -kd*qdot` 或抱闸；
- 力矩 slew limit；
- 电流/温度限幅；
- 终止判据：pitch/roll、髋跟踪误差、轮电流、角速度、超时；
- 首次上机吊带/台架 + 急停；
- 异常时不要直接切零力矩，先确认机械支撑。

---

## 7. 上机前检查清单

### 模型

- [ ] ONNX 输入 `[1,125]`、输出 `[1,6]`；
- [ ] `onnx.checker` 通过；
- [ ] Torch vs ONNX 误差 < 1e-4；
- [ ] checkpoint 已收敛；
- [ ] 低层 PD 增益与训练一致（hip 40/1，wheel 0.5）；
- [ ] sidecar `deploy_kind = squat_torque`、`clip_actions = 4.0`。

### 固件

- [ ] `last_action` 存的是 clip ±4 后的原始动作，不是 Nm/电流；
- [ ] `9:13` 用 `LOCKED_HIPS`，不是 nominal 髋角；
- [ ] `13:19` 的顺序是 4 髋 + 2 轮，且乘 0.05；
- [ ] 右腿符号与训练一致；
- [ ] `joint_vel` 的滤波/差分时间常数一致；
- [ ] 观测各维 z-score 在 ±3 内（用 checkpoint 的 mean/std 检查）；
- [ ] 力矩 → 电流的 `Kt_eff` 已标定；
- [ ] 看门狗、过流/过温、力矩 slew limit、急停有效。

### 台架

- [ ] 先吊带/台架，限制轮速/力矩；
- [ ] 零力矩时能安全支撑；
- [ ] 异常时能切到阻尼/抱闸；
- [ ] 逐步放开到完整闭环。

---

## 8. 常见错误

| 错误 | 后果 |
|---|---|
| 用了旧 42 维观测的 ONNX | 输入维度不符，动作语义完全不同 |
| 忘记 clip ±4 就乘 scale | 力矩超出训练分布，策略失控 |
| 把 `tau` 或电流当 `last_action` 反馈 | 观测分布错误 |
| `9:13` 用了 nominal 髋角 | 偏移约 0.67 rad，直接失效 |
| 对输入再做一次归一化 | 双重归一化，动作错误 |
| 导出辅助未衰减完的 checkpoint | 不是纯力矩策略 |
| `Kt_eff` 用了电机手册值而非关节实测 | 力矩偏低/偏高，平衡失败 |
| 驱动器只支持 PWM 却按力矩用 | 力矩随转速变化，sim2real 崩 |

---

## 9. 相关文件

```text
训练/任务说明 : docs/squat_torque.md
导出脚本 : tools/export_policy_onnx.py
观测定义 : env_squat.py  SquatObservationsCfg
动作项 : mdp/actions.py  FiveBarJointAction
五连杆参考实现 : kinematics/planar_leg.py
```
