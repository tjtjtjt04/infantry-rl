# squat_torque ONNX 实机部署说明

本文对应已经导出的纯力矩策略：

```text
checkpoint : logs/rsl_rl/infantry_rl_squat_torque/2026-11-04_17-56-09_chassis_squat_torque_s3_pure/model_1798.pt
ONNX       : exports/chassis_squat_torque.onnx
sidecar    : exports/chassis_squat_torque.json
```

范围：

```text
6 个执行器 = 4 个髋电机 + 2 个轮电机，全部力矩控制
膝盖 = 被动机械闭链，不参与动作，不需要膝电机/编码器
```

如果实机膝盖也有电机，或髋驱动器只能位置/速度模式，这份说明不适用。

---

## 1. 导出结果

### 1.1 ONNX 接口

```text
INPUT : obs    [1, 42]  float32
OUTPUT: action [1, 6]   float32
OPS   : Gemm / Elu
```

- 归一化器已经折叠进第一层，固件送**原始观测值**；
- 输出是确定性策略均值，**原始动作单位**；
- 输出不含 clip、不含 Nm 换算、不含电流换算。

### 1.2 训练收敛情况

来自 `s3_pure` 最后一个 checkpoint：

```text
iter                 1798
mean_episode_length  485.17 / 500 step
time_out             0.970
pitch_limit          0.030
roll_limit           0.000
base_too_low         0.000
bad_orientation      0.000
hip assist           0（已关闭）
wheel assist         0（已关闭）
teacher              无
```

这是纯力矩策略，辅助已经全部为 0。

### 1.3 导出校验

```text
onnx.checker           OK
ops                    ['Elu', 'Gemm']
torch ref vs ONNX      max |diff| = 2.289e-05
```

---

## 2. 动作后处理（固件必须实现）

ONNX 输出 `action[6]` 后：

```text
a = clamp(action, -4.0, 4.0)                 # PPO clip_actions = 4.0

tau = a * [10, 10, 10, 10, 1, 1]             # Nm
tau[0:4] = clamp(tau[0:4], -40.0, +40.0)     # 髋
tau[4:6] = clamp(tau[4:6], -3.69, +3.69)    # 轮

I = tau / Kt_eff                             # 电流参考
I = clamp(I, I_min, I_max)
send_current(I)
```

动作顺序：

```text
[tau_lf_hip, tau_lb_hip, tau_rf_hip, tau_rb_hip, tau_left_wheel, tau_right_wheel]
```

`Kt_eff` 是折算到关节侧的力矩常数：

```text
直驱：Kt_eff = Kt_motor
减速箱：Kt_eff = Kt_motor * N * eta
```

**重要**：下一帧喂给网络的 `last_action` 是 `a`（clip 后的原始动作），
不是 `tau`，也不是电流。

---

## 3. 42 维观测怎么组

观测定义在 `env_squat.py` 的 `SquatTorqueObservationsCfg`（约 267–312 行）。

| slice | 项 | 训练 scale | 下位机实现 |
|---|---|---:|---|
| 0:3 | projected_gravity | 1 | IMU 重力投影 |
| 3:6 | base_ang_vel | 0.25 | 陀螺仪 × 0.25 |
| 6:9 | base_lin_vel | 1 | 线速度估计 |
| 9:11 | heading sin/cos | 1 | `sin(yaw), cos(yaw)`，目标 0 |
| 11:13 | position error | 1 | 目标 `(0,0)` 减去当前位置，转到机体系 |
| 13:17 | hip_pos_rel_locked | 1 | `hip_meas - LOCKED_HIPS` |
| 17:21 | hip_vel | 0.1 | 髋编码器速度 × 0.1 |
| 21:25 | knee_pos_rel_solved | 1 | **常量 / 0** |
| 25:29 | measured_leg_state | 1 | **髋角 + 闭式五连杆解算** |
| 29:33 | leg_rates | 0.5 | 对 `leg_state` 差分 × 0.5 |
| 33:35 | wheel_vel | 0.1 | 轮编码器速度 × 0.1 |
| 35:41 | last_action | 1 | 上一帧 clip ±4 后的 6 维动作 |
| 41:42 | base_height_rel | 1 | 运动学估计 − `SQUAT_BASE_HEIGHT` |

### 3.1 关节顺序与符号

```text
HIP_ORDER   = (lf, lb, rf, rb) = (left_front, left_back, right_front, right_back)
KNEE_ORDER  = (lf, lb, rf, rb)
WHEEL_ORDER = (left_wheel, right_wheel)
```

chassis 资产：

```text
right_hip_sign = -1
```

右腿相对左腿镜像，固件必须使用与训练相同的符号。

### 3.2 `LOCKED_HIPS` 常量

来自 `robot_data/chassis-rl/squat_pose.json`：

```text
LOCKED_HIPS = [-0.011721, +0.011800, +0.011721, -0.011800]   # lf, lb, rf, rb
```

注意：

- 42 维 torque 观测的 `hip_pos_rel_locked` 是 **`hip_meas - LOCKED_HIPS`**；
- 30 维 wheel-only squat 的 `hip_pos_rel` 是 **`hip_meas - NOMINAL_HIP`**；
- 两者相差约 0.67 rad，**不要混用固件代码路径**。

### 3.3 膝关节误差（21:25）

训练里是：

```text
measured_knee - solved_knee_target
```

实机没有膝编码器，但膝盖是机械闭链，真实膝角由两个髋角决定，
因此这个跟踪误差 ≈ 0。

本 checkpoint 的训练均值（原始观测单位）：

```text
KNEE_ERR_MEAN = [-0.008032605, +0.007064347, +0.008130094, -0.008227251]
```

固件可以：

- 填 `KNEE_ERR_MEAN`（推荐，标准化偏移为 0）；
- 或填 0（如果机械闭链足够刚性）。

不要用“上一帧膝角”或滤波硬凑，因为没有真实测量。

### 3.4 measured_leg_state（25:29）

顺序：

```text
[L0_left, phi_left, L0_right, phi_right]
```

用髋编码器和闭式五连杆解算：

```text
knee_est  = five_bar_closed_form(hip_meas)
leg_state = leg_state_from(hip_meas, knee_est)
```

不要用迭代 Newton；闭式解（圆-圆交点 + `atan2/acos/sqrt`）更简单、可测试。
用 `kinematics/planar_leg.py` 生成测试向量，固件误差目标 < 1e-5 rad。

本 checkpoint 的参考均值（用于检查观测是否在训练分布内）：

```text
LEG_STATE_MEAN = [0.134446979, +0.000604810, 0.134364784, -0.001266676]
```

### 3.5 leg_rates（29:33）

```text
leg_rates = (leg_state_now - leg_state_prev) / dt
obs[29:33] = leg_rates * 0.5
```

需要保留上一控制步的 `leg_state`；复位/刚上电时先填 0。

### 3.6 base_height（41:42）

髋在力矩模式下不是位置伺服，但 `hip_lock` 奖励会把它们约束在锁定姿态附近。
建议用运动学估计，而不是写死：

```text
r_body = ( x_mid + L0*sin(phi),  y_wheel,  h_hip - L0*cos(phi) )
base_z = wheel_radius - [ R(roll,pitch,yaw) * r_body ]_z
base_height_rel = base_z - SQUAT_BASE_HEIGHT
```

chassis 参数：

```text
wheel_radius = 0.068 m
h_hip        = 0.030406 m
x_mid        = -0.003475 m
y_wheel      = ±0.219 m
SQUAT_BASE_HEIGHT = 0.172990 m
```

水平地面简化式：

```text
base_z ≈ 0.068 + L0*cos(phi) - 0.030406
```

本 checkpoint 的参考均值：

```text
BASE_HEIGHT_MEAN = -0.002935953   # m
```

如果暂时不做姿态修正，可以填这个常量；不要用 IMU 加速度二次积分。

### 3.7 hip_vel（17:21）

用髋编码器速度（或位置差分 + 低通）：

```text
obs[17:21] = hip_vel * 0.1
```

本 checkpoint 的参考均值：

```text
HIP_VEL_MEAN = [+0.003526536, -0.007256237, -0.009363689, -0.000805129]
```

### 3.8 heading / position

- `TARGET_HEADING = 0.0`、`TARGET_X = TARGET_Y = 0.0`；
- 上电/复位时把 IMU yaw 和里程计原点清零；
- `position_error` 是 `目标 - 当前`，再旋转到机体 yaw 系。

---

## 4. 固件主循环伪代码

```text
# 50 Hz
hip      = read_hip_pos()                    # 4, 编码器
hip_vel  = filter(diff(hip))                 # 4
knee_est = five_bar_closed_form(hip)         # 每腿一次
leg      = leg_state(hip, knee_est)          # [L0_l, phi_l, L0_r, phi_r]
leg_rate = (leg - prev_leg) / dt             # 4
height   = base_height_kinematics(hip, imu)

obs[0:3]   = gravity
obs[3:6]   = gyro * 0.25
obs[6:9]   = lin_vel
obs[9:11]  = [sin(yaw), cos(yaw)]
obs[11:13] = body_frame(-position)
obs[13:17] = hip - LOCKED_HIPS
obs[17:21] = hip_vel * 0.1
obs[21:25] = KNEE_ERR_MEAN                   # 或 0
obs[25:29] = leg
obs[29:33] = leg_rate * 0.5
obs[33:35] = wheel_vel * 0.1
obs[35:41] = last_action
obs[41]    = height - SQUAT_BASE_HEIGHT

a   = onnx_run(obs)                          # 6, 原始动作单位
a   = clamp(a, -4, 4)
tau = clamp(a * TORQUE_SCALE, TAU_MIN, TAU_MAX)
I   = clamp(tau / KT_EFF, I_MIN, I_MAX)

send_current(I)
last_action = a
prev_leg    = leg
```

---

## 5. Python 端快速验证

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(
    "exports/chassis_squat_torque.onnx",
    providers=["CPUExecutionProvider"],
)

obs = np.zeros((1, 42), dtype=np.float32)
action = sess.run(None, {"obs": obs})[0]

print(action.shape)      # (1, 6)
print(action)
```

注意：

- 输入名 `"obs"`，输出名 `"action"`；
- 输入必须是 `float32`；
- 归一化已经在模型里，**不要**再减 mean/除 std；
- 输出是原始动作单位，不是 Nm。

建议再用 checkpoint 的 Torch 模型跑同一组 `obs`，确认：

```text
max |action_onnx - action_torch| < 1e-4
```

### 5.1 用 sim 的观测做 golden vector

最可靠的验证方法：

1. 在 `play.py --task squat_torque` 里记录一帧真实的 42 维观测和策略输出；
2. 把同一帧观测喂给 ONNX；
3. 比较 ONNX 输出与 sim 内策略输出；
4. 再把动作按 `clip -> Nm` 后处理，确认力矩在范围内。

这样可以同时验证固件观测拼装和 ONNX 推理链路。

---

## 6. 扭矩到电流

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

## 7. 安全

纯力矩没有隐式阻尼，**零力矩不是安全状态**：

- 看门狗：通信超时/姿态超限/过流/过温 → 切到简单阻尼
  `tau = -kd*qdot` 或抱闸；
- 力矩 slew limit；
- 电流/温度限幅；
- 终止判据：pitch/roll、髋跟踪误差、轮电流、角速度、超时；
- 首次上机吊带/台架 + 急停；
- 异常时不要直接切零力矩，先确认机械支撑。

---

## 8. 上机前检查清单

### 模型

- [x] ONNX 输入 `[1,42]`、输出 `[1,6]`；
- [x] `onnx.checker` 通过；
- [x] Torch vs ONNX 误差 `2.289e-05`；
- [x] checkpoint 已收敛（485/500 step，97% timeout）；
- [x] 训练辅助已全部为 0（纯力矩）；
- [x] sidecar `deploy_kind = squat_torque`、`clip_actions = 4.0`。

### 固件

- [ ] `last_action` 存的是 clip ±4 后的原始动作，不是 Nm/电流；
- [ ] `13:17` 用 `LOCKED_HIPS`，不是 nominal 髋角；
- [ ] 右腿符号与训练一致；
- [ ] 五连杆闭式解与 `kinematics/planar_leg.py` 对齐（< 1e-5 rad）；
- [ ] `hip_vel`、`leg_rates` 的滤波/差分时间常数一致；
- [ ] 观测各维 z-score 在 ±3 内（用 checkpoint 的 mean/std 检查）；
- [ ] 力矩 → 电流的 `Kt_eff` 已标定；
- [ ] 看门狗、过流/过温、力矩 slew limit、急停有效。

### 台架

- [ ] 先吊带/台架，限制轮速/力矩；
- [ ] 零力矩时能安全支撑；
- [ ] 异常时能切到阻尼/抱闸；
- [ ] 逐步放开到完整闭环。

---

## 9. 常见错误

| 错误 | 后果 |
|---|---|
| 用了 30 维 wheel-only squat 的 ONNX | 输入维度不符，动作语义完全不同 |
| 忘记 clip ±4 就乘 scale | 力矩超出训练分布，策略失控 |
| 把 `tau` 或电流当 `last_action` 反馈 | 观测分布错误 |
| `13:17` 用了 nominal 髋角 | 偏移约 0.67 rad，直接失效 |
| 对输入再做一次归一化 | 双重归一化，动作错误 |
| 膝角用上一帧/滤波硬凑 | 没有真实测量，噪声会污染策略 |
| 导出辅助未衰减完的 checkpoint | 不是纯力矩策略 |
| `Kt_eff` 用了电机手册值而非关节实测 | 力矩偏低/偏高，平衡失败 |
| 驱动器只支持 PWM 却按力矩用 | 力矩随转速变化，sim2real 崩 |

---

## 10. 相关文件

```text
训练/任务说明 : docs/squat_torque.md
30 维 wheel-only 部署方案 : docs/squat_deployment_plan.md
导出脚本 : tools/export_policy_onnx.py
观测定义 : env_squat.py  SquatTorqueObservationsCfg
动作项 : mdp/actions.py  FiveBarJointAction
五连杆参考实现 : kinematics/planar_leg.py
```
