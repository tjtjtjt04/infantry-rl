# infantry‑rl 五连杆轮腿机器人 —— Isaac Sim 平衡训练

用 `infantry-rl_close/urdf/infantry-rl-sim.urdf` 做**平衡训练**的完整工程。
目标是让机器人用两条并联五连杆腿 + 两个驱动轮**站起来并保持平衡**。

> **先读这两份文档：**
> - [`docs/VERIFIED_RESULTS.md`](docs/VERIFIED_RESULTS.md) —— **实测结果与已知限制**（哪些跑通了、哪些没跑通、为什么）
> - [`docs/DESIGN.md`](docs/DESIGN.md) —— 设计说明（电机正方向、PhysX 不支持闭环的证据、运动学、传感器需求）

---

## 当前状态（一句话）

✅ **运动学已验证到机器精度**（1.8e‑16 m）；
✅ **闭环控制策略在真实物理仿真里验证通过**（200 步，闭环残差 **0.75 mm**，零跌倒）；
❌ **多环境训练在本沙箱里跑不起来**（`gym.make` 在 `num_envs ≥ 2` 时卡死，
已定位到 96 MB 资产，已排除复制/碰撞过滤两种原因）。
详见 [`docs/VERIFIED_RESULTS.md`](docs/VERIFIED_RESULTS.md)。

---

## 快速开始

环境：Isaac Sim **5.1.0** + Isaac Lab **2.3.2** + rsl‑rl‑lib **5.0.1**，
本机 Python 在 `D:\miniconda\envs\RM\python.exe`。

```powershell
$py = "D:\miniconda\envs\RM\python.exe"
cd D:\Robomaster\RL

# 0) 先清掉残留的 Kit 进程（会占住 GPU 和 Kit 的锁）
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# 1) 一次性：生成可移植资产包（把 package:// 换成相对路径）
& $py isaac_ws\scripts\prepare_asset.py

# 2) 验证运动学与闭环（秒级，不需要 GPU）→ 全部通过
& $py -B isaac_ws\isaac_ws_infantry\kinematics\test_leg_model.py

# 3) 冒烟测试：机器人站起来、闭环残差 < 1 mm（约 2 分钟，需要 GPU）→ PASS
& $py isaac_ws\scripts\smoke_test.py --headless --num_envs 1 --steps 200 `
     --report isaac_ws\scripts\smoke_test_report.json

# 4) 训练（当前只能用 1 个环境，原因见 VERIFIED_RESULTS.md）
& $py isaac_ws\scripts\train.py --task Infantry-RL-Balance-v0 --headless --num_envs 1
```

---

## 目录结构

```
isaac_ws/
├── assets/infantry-rl/              可移植资产包（由 prepare_asset.py 生成）
│   ├── meshes/*.STL                 从 infantry-rl_close 复制
│   ├── urdf/infantry-rl-sim.urdf    仅把 package:// 改成相对路径，物理一字未动
│   ├── infantry_rl.usd              完整 CAD 转换产物（96 MB，默认，已物理验证）
│   └── infantry_rl_proxy.usd        代理资产（几 KB，能多环境但碰撞不准）
├── config/
│   ├── joint_map.json               全部关节的轴 / 限位 / 角色
│   ├── leg_model.json               实测出的腿部平面模型常数
│   ├── nominal_pose.json            精确的标称站立位
│   ├── feasible_poses.json          可达工作空间包络
│   └── proxy_geometry.json          代理资产的实测几何
├── docs/
│   ├── VERIFIED_RESULTS.md          ★ 实测结果与已知限制
│   ├── DESIGN.md                    ★ 设计说明
│   └── API_NOTES.md                 Isaac Lab 2.3.2 精确 API 速查（带源码行号）
├── isaac_ws_infantry/
│   ├── kinematics/                  ★ 已验证的运动学与闭环求解器
│   │   ├── five_bar.py              URDF 精确正运动学
│   │   ├── leg_model.py             腿模型 + 闭环求解（numpy / torch 批量）
│   │   ├── test_leg_model.py        验证脚本（全通过）
│   │   └── feasible_workspace.py    工作空间扫描
│   ├── model.py                     模型/配置/几何的共享桥接
│   ├── robots/infantry_rl.py        ArticulationCfg
│   ├── mdp/                         观测 / 奖励 / 动作项
│   └── tasks/balance/               环境配置 + 环境类 + gym 注册
└── scripts/
    ├── prepare_asset.py             生成可移植资产包
    ├── build_proxy_asset.py         生成轻量代理资产
    ├── extract_joint_map.py         导出关节表
    ├── train.py / play.py           rsl-rl PPO 入口
    ├── smoke_test.py                闭环与站立验证
    └── diag_clone.py                复现"多环境卡死"的诊断脚本
```

> 训练产物（`logs/` 下的 checkpoint、TensorBoard 事件）已在交付前清理干净。
> 训练命令与查看效果的方法见 [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md)。

---

## 三个必须知道的实测结论

### 1. 电机正方向就是 URDF 的值，代码不翻转任何轴

用户要求"从出轴看逆时针为正"。核对全部 10 个关节后确认：

- 右腿髋关节的父坐标系多了 `rpy = 0 0 -π`，但**绕 Z 的旋转不改变 Y 轴方向**；
  同时右腿几何被镜像到 −Y 侧，两者抵消 ⟹ **左右腿符号约定一致**。
- URDF 限位本身就编码了这个约定：同一条腿的两个髋
  `front_hip ∈ [-0.184, +1.305]`、`back_hip ∈ [-1.823, +0.184]` 是镜像的，
  正是"伸缩时两个髋电机反向转"的体现。

### 2. ⚠️ PhysX **不会**替你闭合五连杆的环，必须用控制去闭合
URDF 导入器**保留**了两个 `<loop_joint>`（USD 里有
`/infantry_rl_sim/loop_joints/*_calf_axle`，共 12 个关节），
但**加载成 articulation 后只剩 10 个自由度**，`*_calf_axle` 不在其中：

```
num_joints = 10        has_calf_axle_joint = False
```

把驱动全部置零、重力打开步进 240 步，`|p(wheel) − p(back_calf)|` 始终停在
**0.2715 m**（= 两个连杆原点之间的距离，而不是闭环锚点间距 0）——后小腿就是个自由摆。

**所以本工程的做法是**：4 个髋 + 2 个轮做主动关节（**动作空间 6 维**），
**4 个膝做从动件**，每个控制步用解析闭环求解器解出膝角并写进位置目标
（热启动，1~2 次牛顿迭代收敛）。

实测效果（`smoke_test.py`，200 步随机动作）：闭环残差
**最大 0.79 mm / 平均 0.47 mm**，机体高度维持在 0.167–0.205 m，零次跌倒。

### 3. 腿是 X–Z 平面机构，模型精度到机器精度

所有关节轴都平行于机体 Y 轴 ⟹ 平面机构，且平面是 **X–Z**（实测 `dY/dq == 0`）。
模型常数全部由 URDF 精确正运动学**测量**得到，不手推：

| 量 | 左腿 | 右腿 |
|---|---|---|
| 大腿 `\|P\|` | 0.1450034 m | 0.1450032 m |
| 小腿 `\|Q\|` | 0.2700030 m | 0.2700032 m |
| 模型 vs 精确 FK（3000 组随机位形） | **1.8e-16 m** | 4.6e-7 m |

---

## 传感器需求（详见 DESIGN.md §6）

当前 URDF/USD **没有任何传感器**。仿真里大部分量 Isaac Lab 直接提供，**不需要额外配置**；
但要做到实车可用 / sim‑to‑real，按优先级：

| 优先级 | 传感器 | 为什么 |
|---|---|---|
| **必须** | IMU（陀螺 + 加速度计，≥200 Hz） | 平衡的根：没有姿态和角速度，倒立摆无法稳定 |
| **必须** | 6 个关节编码器（4 髋 + 4 膝） | 髋角是主动量；膝角用来在线校验闭环残差 |
| **强烈建议** | 2 个轮速编码器 | 轮子是速度控制，且轮速是估计前进速度最直接的量 |
| 建议 | 关节电流/力矩估计 | 力矩惩罚、过载保护、执行器建模 |
| 可选 | 下视 ToF / 足端压力 | 判断悬空、落地缓冲、跳跃 |
| 可选 | 深度相机 / 激光雷达 | **只有**做地形自适应才需要；平地平衡加了只会拖慢训练 |

> 换句话说：**做平地平衡，你只需要再补一个 IMU 和关节编码器**。
> 仿真侧一行额外配置都不用加。

---

## 已知限制

- **本机沙箱无外网**，Isaac Lab 的 `spawn_ground_plane()` 会去下载 Nucleus 资产并失败，
  所以场景用本地地面/光源构建。有网时可以换回官方 helper。
- `base_link.STL` 57 MB → USD 96 MB，首次导入慢；底盘碰撞体是它的凸包
  （PhysX 已成功 cook，机器人能稳定落地）。若接触不稳，建议换盒体代理碰撞。
- 闭环是用 PD 驱动被动关节**近似**的，不是硬约束。高速运动下会有几毫米形变，
  `smoke_test.py` 会打印实测残差。
