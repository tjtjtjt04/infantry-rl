# infantry-rl 并联腿 URDF 闭环构建

> 把 SolidWorks 导出的**树形** URDF 还原成**闭环（并联五连杆）**模型，用于 Isaac Lab 训练。

---

## 1. 问题定位

SW2URDF 导出器**只能生成树形结构**（每个 link 只能有一个父关节），所以并联腿的闭环在导出时被切断了：

```
base_link
 ├─ left_front_hip  → left_front_ham  ─ left_front_knee → left_front_calf ─ left_wheel
 ├─ left_back_hip   → left_back_ham   ─ left_back_knee  → left_back_calf   ← 末端悬空
 ├─ right_front_hip → right_front_ham ─ right_front_knee → right_front_calf ─ right_wheel
 └─ right_back_hip  → right_back_ham  ─ right_back_knee  → right_back_calf  ← 末端悬空
```

`left_back_calf` / `right_back_calf` 就是"后膝盖连杆末端悬空"的那两个零件。
你说过：**小腿末端最大的那个孔就是与驱动轮连接的孔** —— 下面的实测结果和这一点完全吻合。

---

## 2. 实测的几何结论

全部由 URDF 关节原点 + STL 网格实测得到。复现：`tools/mobility_reference.py`（会打印下表）。

### 2.1 每条腿是一个等腰五连杆（与固件 `FiveBar/five_bar.c` 一致）

| 量 | 实测 | 固件默认值 |
|---|---|---|
| 前大腿 `l1` | 0.145 m | 0.145 |
| 前小腿 `l2` | 0.271019 m | 0.270 |
| 后小腿 `l3` | 0.271497 m | 0.270 |
| 后大腿 `l4` | 0.145 m | 0.145 |
| 机架 `b` | 0.150 m | 0.150 |

`l1=l4`、`|l2−l3| = 0.48 mm` —— 标准的等腰五连杆。

### 2.2 后小腿末端的大孔就在轮轴上

以前小腿/后小腿自身的坐标系为基准，网格顶点到轮轴线的距离：

| 基准 | 到轮轴的距离 |
|---|---|
| `*_front_calf` | 径向 0.00 mm（就在轮子中面） |
| `*_back_calf` | 径向 4.1 mm，轴向偏差 27.5 mm（在轮子内侧） |

后小腿末端端面法向实测为 `(0,∓1,0)`，与轮轴**平行** —— 说明该处是绕轮轴的**转动副**。

---

## 3. 生成的文件

只用**一个**闭环文件：

| 文件 | 说明 |
|---|---|
| `urdf/infantry-rl.urdf` | 你的原始 SW 导出，**一字未改**，保留作对比 |
| `urdf/infantry-rl-closed.urdf` | **要用的就是这个**：原树 + 2 个闭环关节，并修正了右腿的导出一致性问题 |
| `urdf/infantry-rl.csv` | SW 导出附带的 CSV，未改 |

`infantry-rl-closed.urdf` 里实际写进去的两个关节：

```xml
<joint name="left_back_calf_axle" type="revolute">
  <origin xyz="-0.22 -0.028445 0.15653" rpy="0 0 0" />
  <parent link="left_wheel" />  <child link="left_back_calf" />
  <axis xyz="0 -1 0" />  <limit lower="-3.14159" upper="3.14159" .../>
</joint>

<joint name="right_back_calf_axle" type="revolute">
  <origin xyz="0.22 -0.028446 0.15653" rpy="0 0 3.14159" />
  <parent link="right_wheel" /> <child link="right_back_calf" />
  <axis xyz="0 -1 0" />  <limit lower="-3.14159" upper="3.14159" .../>
</joint>
```

**这两个 `<origin>` 是关键，不能填 `(0,0,0)`**（我第一版就是这么填的，那个"闭环"其实是开的）：

- `<origin>` 表达在**父链接（轮子）自己的坐标系**里，等于
  `R_wheel^T · (p_后小腿 − p_轮子)`；
- `rpy` 同理，是 `R_wheel^T · R_后小腿`。左腿恰好是单位阵，**右腿的轮子 link
  坐标系被导出了 180° 绕 z 的旋转，所以右腿的 rpy 必须是 `(0,0,π)`**。

脚本已改成从模型自动推导这两个量（`build_closed_urdf.py` 的 `loop_frames()`），
不再手写。

### 3.1 顺带修掉的右腿导出问题

原文件右腿**不是**左腿的镜像，而且内部自相矛盾：

| 关节 | 原文件 | 应为 |
|---|---|---|
| `right_front_hip` 世界位置 | x = −0.075 | +0.075 |
| `right_front_knee` `origin` | `(-0.14273, 0.0275, +0.025549)` | `(-0.145, 0.0275, -0.001)` |
| `right_wheel` `origin` | `(0.20933, 0.028445, -0.17054)` | `(0.22, 0.023445, -0.15653)` |

修正后左右腿 `l1..l4` 完全一致（`|l2−l3| = 0.478 mm`，两腿相同），
链路 `|M·p_left − p_right| = 0`。

---

## 4. 验证结果（`tools/verify_closure.py`）

```
infantry-rl-closed.urdf
  links 11/11   joints 12   独立回路 2 个: [left_back_calf_axle, right_back_calf_axle]
  left  leg : 装配位形闭环残差 0.000e+00 ; 闭环 Jacobian rank 3 → 2 个独立运动
              沿运动方向走 1e-3/1e-2 rad : 残差 2.3e-08 / 2.3e-06
  right leg : 装配位形闭环残差 0.000e+00 ; 闭环 Jacobian rank 3 → 2 个独立运动
              沿运动方向走 1e-3/1e-2 rad : 残差 2.3e-08 / 2.3e-06
ALL STRUCTURAL CHECKS PASSED
```

闭环位置误差 ~1e-8 m、旋转误差 ~1e-22（数值零），**两条腿都确认是 2 自由度**，
和固件 `FiveBar_Solve(q1, q4, ...)` 把两个髋电机当独立输入的用法一致。

（说明：装配位形下两条小腿都竖直，约束梯度接近相关，闭环 Jacobian 会出现一个
接近零的奇异值，这是**真实位形退化**，不是模型错误；判据用 `1e-6·σmax` 的相对阈值。）

---

## 5. Isaac Lab 怎么用

### 5.1 关键限制

Isaac Sim / Isaac Lab 的 `Articulation`（PhysX reduced-coordinate 多刚体）
**不支持闭环运动链**，所以 `-closed*.urdf` 不能直接 import 成一个 articulation 资产。

### 5.2 三个方案

**方案 A（推荐）：树模型 + 被动关节跟随**

用树结构，把 3 个被动关节（`*_front_knee`、`*_back_knee`、`*_back_calf_axle`）
每步用闭环解算角度驱动，配高刚度 PD：

```python
q1, q4 = policy_hip_angles              # 两个髋电机指令
qkf, qkb = solve_loop_closure(q1, q4)   # 由闭环方程解出两个膝角
targets[:, i_knee_f] = qkf
targets[:, i_knee_b] = qkb
targets[:, i_loop]   = <闭环关节角>
```

稳定、快；接触瞬间被动关节会有微小形变，需要调 `stiffness`。

**方案 B：USD 层面用约束搭闭环**

改用 `RigidPrim` + `UsdPhysics.Joint` 让 PhysX 用约束闭合回路。物理最忠实，
但失去 articulation 的批量张量接口，训练吞吐明显下降。

**方案 C：`<mimic>`**

平面五连杆的闭环角**不是**髋角的线性函数，单个 `multiplier` 只是近似；
要用就先采样闭环解再拟合多项式。

### 5.3 导入

```bash
python scripts/tools/convert_urdf.py \
    infantry-rl/urdf/infantry-rl-closed.urdf \
    /path/to/assets/infantry_rl.usd \
    --merge-joints --joint-stiffness 0.0 --joint-damping 0.0 --fix-base
```

导入后**数关节数量**：应为 12（4 髋 + 4 膝 + 2 轮 + 2 闭环）。
若只有 10 个，说明导入器把闭环关节当 fixed 合并了 —— 回到方案 A。

### 5.4 训练建议

- 动作空间：两条腿各 2 个髋电机 + 2 个轮电机 = **8 维**（与实车一致）；
- 观测里给解算出的腿长 `L0` 和腿轴角 `φ`（对齐固件 `FiveBarState`），比给被动关节角更有用；
- 轮子 velocity 控制，髋电机 position 控制。

---

## 6. 工具（`tools/`）

| 脚本 | 作用 |
|---|---|
| `kinematics.py` | 公共库：URDF 解析、FK、STL 读取、腿长/腿轴角 |
| `build_closed_urdf.py` | **生成闭环 URDF**（闭环关节参数自动推导，右腿自动镜像） |
| `verify_closure.py` | **主验证**：回路数、闭环残差、自由度 |
| `mobility_reference.py` | **实测几何 + 自由度证据** |
| `validate_urdfs.py` | 所有 URDF 结构校验 |

```powershell
$py = "C:\Users\CMD\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
& $py "D:\Robomaster\RL\tools\build_closed_urdf.py"   # 重新生成（约 1 秒）
& $py "D:\Robomaster\RL\tools\validate_urdfs.py"      # 结构校验
& $py "D:\Robomaster\RL\tools\verify_closure.py"      # 运动学验证（约 5 秒）
& $py "D:\Robomaster\RL\tools\mobility_reference.py"  # 实测几何 + 自由度证据
```

> `build_closed_urdf.py` 依赖 `infantry-rl.urdf` 保持原样。
> 若重新从 SW 导出，原文件会被覆盖，需重跑上面 4 条命令复核。

---

## 7. 源头解决：SW 里怎么建

1. **把闭环关节涉及的活动零件做成独立 link**，不要和相邻零件合并；
2. 导出器里给每个闭环关节都建 Joint —— 导出器会写进 URDF，但**树结构里必然丢掉一个**；
3. 更省事：把整条腿的活动子装配导出成一个 link（刚体），只留髋/膝关节，
   再手工加闭环关节 —— 就是本仓库做的事。

**SW 导出器永远生成不了合法闭环 URDF**（URDF 规范要求 link 构成树），
无论如何都需要一步后处理。
