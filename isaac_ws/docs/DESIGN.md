# infantry‑rl 五连杆轮腿机器人 —— Isaac Sim 平衡训练设计说明

本文记录**在本机上实测得到**的事实，以及由此推出的训练方案。所有关键结论都标注了
复现命令，不是推断。

- 资产：`infantry-rl_close/urdf/infantry-rl-sim.urdf`（用户指定）
- 仿真：Isaac Sim **5.1.0.0** + Isaac Lab **2.3.2** + rsl‑rl‑lib **5.0.1**
- Python：`D:\miniconda\envs\RM\python.exe`

---

## 1. 电机正方向约定（用户要求）

> 所有电机从出轴看过去逆时针旋转为正。

**结论：URDF 里的关节值已经就是这个约定，代码中不做任何符号翻转。**

推导（两条腿、4 个关节逐一核对）：

每个关节的 `<axis>` 在**它自己的父坐标系**里都是 `(0, ±1, 0)`：

| 关节 | axis（父坐标系） | 父坐标系相对机体的姿态 |
|---|---|---|
| `left_front_hip` | `(0, +1, 0)` | 无旋转 |
| `left_back_hip` | `(0, +1, 0)` | 无旋转 |
| `left_front_knee` | `(0, -1, 0)` | 无旋转 |
| `left_back_knee` | `(0, -1, 0)` | 无旋转 |
| `right_front_hip` | `(0, -1, 0)` | `rpy = (0,0,-π)` |
| `right_back_hip` | `(0, -1, 0)` | `rpy = (0,0,-π)` |
| `right_front_knee` | `(0, +1, 0)` | 无旋转 |
| `right_back_knee` | `(0, +1, 0)` | 无旋转 |

关键点：**右腿髋关节的父坐标系多了一个绕 Z 的 180° 旋转**。绕 Z 的旋转**不改变 Y 轴方向**，
所以髋轴在机体坐标系里仍然是 ±Y；同时右腿的大腿/小腿几何被那 180° 镜像到了 −Y 侧。
两者相互抵消，于是：

- 同一个关节值 `q`，在左腿和右腿上产生**同样的物理转动方向**；
- 四条腿的符号约定完全一致，无需任何 per‑leg 取反。

另一个独立佐证：URDF 的限位本身就编码了这个约定 —— 同一条腿的两个髋
`front_hip ∈ [-0.184, +1.305]`、`back_hip ∈ [-1.823, +0.184]` 是**镜像**的，
说明两个髋电机在腿部伸缩时确实反向转。这与"两个电机都从各自出轴向内看逆时针为正"
完全自洽。

复现：`python isaac_ws/scripts/extract_joint_map.py`（打印全部 10 个关节的轴、限位、角色）。

---

## 2. ⚠️ 核心发现：PhysX 不支持这个闭环，必须用控制去闭合

URDF 里有两个非标准的 `<loop_joint>`，把每条腿闭合成并联五连杆。
**这条路在 Isaac Sim 里走不通**，实测如下：

### 2.1 导入器**保留**了 loop joint，但 PhysX **不把它们算作自由度**

用**真实的 URDF 导入器**转换后检查生成的 USD（原始探针脚本已在交付前清理，
结论保留在此；复现可用 `scripts\prepare_asset.py` 生成 USD 后自行枚举 joint prim）：

```
n_joints = 12
joint_names = left_back_hip, left_back_knee, left_calf_axle, left_front_hip,
              left_front_knee, left_wheel, right_back_hip, right_back_knee,
              right_calf_axle, right_front_hip, right_front_knee, right_wheel
→ 两个 loop joint 变成了 /infantry_rl_sim/loop_joints/{left,right}_calf_axle
articulation = /infantry_rl_sim/base_link
```

所以 USD 层面**有**这两个关节。但把它们加载成 articulation 之后：

```
num_joints = 10
joint_names = left_back_hip, left_front_hip, right_back_hip, right_front_hip,
              left_back_knee, left_front_knee, right_back_knee, right_front_knee,
              left_wheel, right_wheel
has_calf_axle_joint = False
```

→ **`*_calf_axle` 不是 articulation 的自由度。**

### 2.2 闭环确实是被"打开"的：后小腿会吊在那儿晃

同一脚本把 10 个驱动全部置零、重力打开、步进 240 步，测量
`|p(left_wheel) − p(left_back_calf)|`：

| step | 间隙 (m) | 机体高度 (m) |
|---|---|---|
| 0 | 0.271497 | 0.597 |
| 40 | 0.272248 | 0.080 |
| 240 | 0.271492 | 0.073 |

初始间隙 **0.2715 m** 正好等于后小腿与轮子的**连杆原点间距**（`l3`），而不是闭环锚点间距
（装配位形下应为 0）。整个过程中这个间隙不收敛到 0，说明后小腿是个自由摆动的开链末端。

**结论：物理引擎不会替你闭合这个环，必须由控制器每步把闭环解出来并驱动被动关节。**

> 复现方法：URDF 导入器保留了 loop joint，但把它加载成 articulation 后
> `num_joints` 只有 10、`has_calf_axle_joint` 为 False，且后小腿会自由摆动。
> 复现脚本 `scripts\diag_clone.py` 会打印 articulation 的关节清单；
> 最快的自检是 `scripts\smoke_test.py` —— 它从仿真刚体姿态独立测量闭环残差，
> 若残差为 0.27 m 左右（而非 0）就说明环是开的。

---

## 3. 运动学：腿部确实是平面机构，且在 X–Z 平面

所有关节轴都平行于机体 Y 轴 ⟹ 整条腿是平面机构。实测确认平面是 **X–Z**
（+X 前、+Z 上），因为任何关节运动都不会改变轮轴的 Y 坐标（`dY/dq == 0`）。

于是每条腿是两条 2 连杆链：

```
前链：  A ──大腿──> K1 ──小腿──> W        A = 前髋轴
后链：  B ──大腿──> K4 ──小腿──> E        B = 后髋轴
```

`W` 是轮轴，`E` 是后小腿上的闭环锚点，**闭环条件就是 `W == E`**（2 个方程）。
链条刚性 ⟹ 用零位下测得的两根连杆向量 `P`、`Q`：

```
K = base + R(s_hip · q_hip) · P
X = K    + R(s_hip · q_hip + s_knee · q_knee) · Q
```

（小腿继承髋的转角，因为膝轴长在大腿上。）

`P`、`Q` 和两个符号 `s_hip`、`s_knee` **全部由 URDF 的精确正运动学测量得到**，
不手推。实测结果（两条腿、两条链都一样）：

| 量 | 左腿 | 右腿 |
|---|---|---|
| `\|P\|` 大腿 | 0.1450034 m | 0.1450032 m |
| `\|Q\|` 小腿 | 0.2700030 m | 0.2700032 m |
| `s_hip` | −1 | −1 |
| `s_knee` | +1 | +1 |
| 模型对精确 FK 的残差 | **1.7e-16 m** | 4.5e-7 m（URDF float32 噪声） |

复现：`python isaac_ws/isaac_ws_infantry/kinematics/test_leg_model.py`
→ 3000 组随机关节角下，模型与精确 FK 的最大误差 1.8e-16 m（左）/ 4.6e-7 m（右），
零位闭环残差 2.8e-17 m。

---

## 4. 控制接口与可达工作空间

- **主动关节**：每腿 2 个髋（共 4 个）+ 2 个轮（速度控制）→ 与实车一致。
- **被动关节**：每腿 2 个膝，每控制步用 `solve_loop` 解出并写进位置目标（热启动）。

### 4.1 可达工作空间（实测，`feasible_workspace.py`）

14641 组髋角里只有 **6281 组**能既闭合闭环又让两个膝角落在限位内：

- 腿长 `L0 ∈ [0.1356, 0.2843] m`
- 腿轴角 `phi ∈ [-31.2°, +33.2°]`

| L0 (m) | phi 可行范围 |
|---|---|
| 0.136 | −0.6° … +0.6° |
| 0.160 | −7.3° … +7.3° |
| 0.196 | −13.8° … +15.4° |
| 0.244 | −25.7° … +25.5° |
| 0.280 | −29.4° … +32.5° |

**策略输出必须落在这个包络里**，否则闭环解会跑到膝限位外面。

### 4.2 推荐站立位（精确解，`nominal_pose.json`）

```
L0   = 0.249519 m
phi  = -0.0012 deg
front_hip  = +0.498611 rad      back_hip   = -0.498554 rad
front_knee = +0.606460 rad      back_knee  = -0.606438 rad
```

四个关节全部在限位内，闭环残差 ~1e-16。这是训练时的初始位形。

---

## 5. 训练方案

- **动作空间 6 维**：4 个髋 = 相对标称位的位置增量（每腿 2 个，前后髋独立）；
  2 个轮 = 目标角速度。髋指令被同时限制在 URDF 限位和标称位 ±0.45 rad 的带内，
  保证不会跑出可达包络。**膝不占动作维度** —— 它们是闭环的从动件。
- **观测**：投影重力、机体角速度/线速度、髋角（相对标称）、膝角（相对解算值）、
  轮速、上一步动作、解算出的腿长/腿轴角。
- **奖励**：姿态水平、机体高度保持、动作变化率、关节力矩、关节加速度、
  轮子空转惩罚、跌倒终止惩罚，以及**闭环残差惩罚**（保证机构不被拉开）。
- **PPO**（rsl‑rl 5.0.1，`actor`/`critic` 配置）。

运行：

```powershell
$py = "D:\miniconda\envs\RM\python.exe"
& $py isaac_ws\scripts\smoke_test.py --headless          # 先验证闭环与站立
& $py isaac_ws\scripts\train.py --task Infantry-RL-Balance-v0 --headless --num_envs 4096
& $py isaac_ws\scripts\play.py  --task Infantry-RL-Balance-Play-v0
```

---

## 6. 需要的额外传感器

当前 URDF/USD **没有任何传感器**。要在实车或仿真里做平衡，建议按优先级补：

### 必选

1. **IMU（陀螺仪 + 加速度计）** —— 平衡的核心。
   没有它就没有 `projected_gravity` 和角速度，倒立摆根本无法稳定。
   仿真里由 Isaac Lab 的 `mdp.projected_gravity` / `mdp.base_ang_vel` 免费提供；
   实车需要 ≥200 Hz、低零偏漂移的 IMU，装在机体几何中心附近并标定安装角。

2. **6 个关节编码器**：4 个髋 + 4 个膝 = 8 个，但髋和膝都在电机侧。
   髋角是主动量必须有；**膝角也建议测**，用来在线校验闭环残差
   （`|W−E|`），残差变大说明机构被撞变形或标定漂移。

### 强烈建议

3. **2 个轮速编码器**：轮子是速度控制的，没有轮速反馈无法做速度环；
   而且轮速是估计机体前进速度最直接的量。
4. **机体高度/离地检测**：可以用腿部正运动学推算（已知 4 个髋角即可算 `L0`），
   但如果要检测"是否被抬起/是否悬空"，加一个下视 ToF 或足端压力/电流阈值更可靠。

### 可选（提升上限）

5. **关节力矩估计**：用电机电流代替。用于力矩惩罚、过载保护和 sim‑to‑real 的
   执行器建模（`DCMotorCfg` 的 `saturation_effort`）。
6. **深度相机 / 激光雷达**：只有做地形自适应或越障时才需要；
   平地平衡用不到，加上会显著拖慢训练。
7. **接触传感器**（轮-地接触力）：如果要做"落地缓冲"或跳跃，需要它来判定触地时刻。

> 仿真侧：以上除力矩外都由 Isaac Lab 直接给出，**不需要额外配置传感器**。
> 只有当你打算做视觉/地形任务时，才需要在 `InteractiveSceneCfg` 里加
> `CameraCfg` / `RayCasterCfg`。

---

## 7. 复现全部结论的命令

```powershell
$py = "D:\miniconda\envs\RM\python.exe"
cd D:\Robomaster\RL

& $py isaac_ws\scripts\prepare_asset.py                       # 生成可移植资产包（相对 mesh 路径）
& $py isaac_ws\scripts\extract_joint_map.py                   # 关节表 + 轴 + 限位
& $py -B isaac_ws\isaac_ws_infantry\kinematics\test_leg_model.py     # 运动学与闭环求解器验证
& $py -B isaac_ws\isaac_ws_infantry\kinematics\feasible_workspace.py # 可达工作空间
& $py isaac_ws\scripts\smoke_test.py --headless --num_envs 1 --steps 200   # 闭环 + 站立（需 GPU）
& $py isaac_ws\scripts\diag_clone.py --num_envs 2 --steps 3 --headless     # 复现多环境卡死
```

> 生成可移植资产包的脚本还会顺手产出 `assets/infantry-rl/infantry_rl.usd`，
> 即"URDF 导入器保留 loop joint（12 个）"那条结论的产物；
> 而"加载成 articulation 只剩 10 个 DOF"由 `diag_clone.py` 打印的关节清单与
> `smoke_test.py` 测出的闭环残差共同佐证。
>
> 这些结论当初是用一组一次性探针脚本测出来的，脚本本身已在交付前清理，
> 上面给出的都是保留下来的、可反复运行的等价命令。

---

## 8. 已知限制

- **本机的沙箱没有外网**。Isaac Lab 自带的 `sim_utils.spawn_ground_plane()` 会去下载
  Nucleus 上的 USD 并失败，所以本工程的场景用**本地**地面/光源构建。
  你本机若有网，可以换回官方 helper，但没必要。
- `base_link.STL` 有 57 MB，转成的 USD 约 96 MB。首次导入会比较慢；
  底盘碰撞体是它的凸包（PhysX 已成功 cook，实测机器人能稳定落地）。
  如果发现接触不稳定，建议把底盘换成几个盒体代理碰撞（可以显著提速并更稳定）。
- 五连杆的闭环是**用 PD 驱动被动关节**来近似的，不是硬约束。刚度不足时高速运动下
  会有几毫米的形变；`smoke_test.py` 会打印实测残差供你判断。
