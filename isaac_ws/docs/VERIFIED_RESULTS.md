# 实测结论与已知限制（VERIFIED RESULTS & LIMITATIONS）

本文件只记录**在本机上真实跑出来的结果**和**真实的失败**，供你决定下一步。
所有数字都可以用文末的命令复现。

环境：Isaac Sim **5.1.0.0** + Isaac Lab **2.3.2** + rsl‑rl‑lib **5.0.1**，
Python `D:\miniconda\envs\RM\python.exe`，GPU RTX 4060 Laptop 8 GB。

---

## ✅ 1. 运动学：已验证到机器精度

`isaac_ws/isaac_ws_infantry/kinematics/check_planar_leg.py` → **exit 0，全部通过**

| 检查 | 结果 |
|---|---|
| 平面模型 vs URDF 精确 FK（前链，3000 组随机位形） | 左 **1.8e‑16 m** / 右 4.6e‑7 m |
| 平面模型 vs URDF 精确 FK（后链，3000 组） | 左 **1.4e‑16 m** / 右 5.7e‑7 m |
| URDF 装配位形闭环残差 | 左 **2.8e‑17 m** / 右 1.6e‑9 m |
| `solve_loop` 闭环（196 组髋角） | **196/196 收敛**，最差 9.97e‑13 m |
| torch 批量求解器（256 组） | 最差 **7.5e‑8 m** |
| 实测模型常数 | 大腿 0.1450034 m，小腿 0.2700030 m，`s_hip=‐1`，`s_knee=+1`（左右腿一致） |

**电机正方向**：URDF 的关节值已经就是"从出轴看逆时针为正"，
代码里**没有任何符号翻转**。推导见 `DESIGN.md` §1。

---

## ✅ 2. 闭环控制策略：在真实物理仿真里验证通过

`isaac_ws/scripts/check_robot.py --headless --num_envs 1 --steps 200`
→ **RESULT: PASS**（报告写到 `scripts/check_robot_report.json`；
它属于运行产物，已随训练痕迹一同清理，重跑该命令即可再生成）

| 指标 | 实测 | 容差 |
|---|---|---|
| 关节数 / 动作维度 | 10 DOF / 6 维 | — |
| **闭环残差（从仿真刚体姿态独立测量）** | **最大 0.750 mm** | < 5 mm ✅ |
| 闭环残差（从关节角独立计算，交叉校验） | 0.750 mm（两者吻合） | — |
| 机体高度 | 0.1674 – 0.2050 m（目标 0.1898） | — |
| 最大高度偏差 | **22.4 mm** | < 50 mm ✅ |
| 跌倒/重置次数（200 步） | **0** | — |
| 膝目标被限幅次数 | 0 | — |

这说明：**"用闭环求解器每步驱动 4 个被动膝"这条路是通的**，
被打开的并联环被控制重新闭合到了一毫米以内。

---

## ❌ 3. 未能完成：多环境训练在本沙箱里跑不起来

**这是本次交付唯一没有做到的事。** 现象与定位如下。

### 3.1 现象

| 场景 | 结果 |
|---|---|
| `check_robot.py --num_envs 1`（完整 CAD 资产） | ✅ 通过，200 步正常 |
| `check_robot.py --num_envs 4` | ❌ **卡死**在 `gym.make()` 里 |
| `train.py --num_envs 64 / 4 / 1` | ❌ **卡死**，同样位置 |
| `diag_clone.py --num_envs 2 --replicate-physics false` | ❌ 仍卡死 |
| `diag_clone.py --num_envs 2 --filter-collisions false` | ❌ 仍卡死 |
| 代理资产 `diag_clone.py --num_envs 256` | ✅ **2.9 秒**建好，可正常步进 |

卡死时进程 CPU 在 45 秒里只增长 0.6 秒 —— **是真的阻塞，不是在算**。
日志停在 Kit 的 `Starting the simulation`，紧接着出现一条
`carb.tasking SharedMutex::lockExclusive` 断言。

### 3.2 已排除的原因

- **不是** `replicate_physics` —— 关掉它照样卡死。
- **不是** `filter_collisions` —— 关掉它照样卡死。
- **不是**训练脚本的问题 —— 同一个环境用 `check_robot.py` 跑 4 个 env 也卡死。
- **不是** GPU 显存 —— 卡死时只用了 2.7 GB / 8 GB，且没有其他进程占用。

### 3.3 最可能的原因与已做的尝试

指向**那个 96 MB 的资产**：`base_link.STL` 有 57 MB，
PhysX 要为每个环境 clone 一份已经 cook 好的凸包。

我做了 `isaac_ws/scripts/build_proxy_asset.py`：从 URDF 重建**同样的 11 个连杆、
10 个关节**（名字/轴/原点/限位全部一致，所以运动学完全复用），
但把网格换成按包围盒生成的盒体/圆柱体。结果：

- ✅ 256 个环境 **2.9 秒**建好，不再卡死；
- ❌ 但它的碰撞体是粗糙的包围盒，机器人会摔倒：
  闭环残差 **489 mm**、200 步内 **448 次**重置。

所以代理资产**不是**能直接用的替代品，只是一个碰撞体调优的起点。
**默认配置已改回完整 CAD 资产**（已验证的那个），用 `--num_envs 1`。

### 3.4 建议你做的下一步（按性价比排序）

1. **把底盘碰撞体换成几个盒体代理**。57 MB 的 `base_link.STL` 是罪魁祸首，
   而且它的凸包对接触稳定性也没好处。做法：保留网格做视觉，
   只把 `base_link/collisions` 换成 2–3 个 `UsdGeom.Cube`。
   这大概率能同时解决卡死和接触抖动。
2. 腿部的代理碰撞体按**连杆轴向**生成细长盒体（当前是按整块包围盒，
   所以腿之间会互相穿模）。参考 `config/proxy_geometry.json` 里的实测尺寸。
3. 底盘碰撞体搞定后，逐步加环境数：1 → 4 → 16 → 64，确认卡死消失。
4. 你本机（有网、显卡更好）可能根本没有这个问题 —— 直接试
   `--num_envs 4096` 看看。这个卡死可能是本沙箱特有的
   （注意日志里的 `failed to open file .../isaacsim/kit/cache/...`：
   DSH 文件沙箱不允许写 Isaac Sim 的安装目录，这可能干扰了 Kit 的缓存/线程初始化）。

---

## ⚠️ 4. 关于 `wheeled-legged_RL` 参考仓库的重要结论

你让我参考 `D:\Robomaster\RL\wheeled-legged_RL`，我读了它的资产定义
（`source/agent_world/agent_world/assets/wheelbipe_V14_2.py`）。结论：

**那是另一个机构，预训练策略不能直接用在本 URDF 上。**

| | 本 URDF（infantry‑rl‑sim） | 参考仓库 wheelbipe V14_2 |
|---|---|---|
| 腿部机构 | **并联五连杆**（闭环） | **串联**（开链） |
| 关节命名 | `*_front_hip` / `*_back_hip` / `*_front_knee` / `*_back_knee` / `*_wheel` | `*_front1..4_joint` / `*_rear1..2_joint` / `*_spring1,2_joint` / `*_guide_joint` / `*_wheel_joint` |
| 主动关节 | 4 髋 + 2 轮 | `.*_rear1_joint` + `.*_front1_joint` + 轮 |
| 额外机构 | 无 | 弹簧 + 导向（`spring1/2`、`guide`） |
| 云台 | 无 | `gimbal_yaw_joint` / `gimbal_pitch_joint` |

**但它的奖励/控制设计有参考价值**，我确认了几点与本工程一致或值得借鉴的：

1. **它也只给少数关节位置控制**：`legs_act` 只含 `front1`/`rear1`，
   其余腿部关节放在 `legs_inact`（`stiffness=0, damping≈0`）。
   → 与本工程"髋主动、膝被动跟随"的思路一致。
2. **执行器参数**：腿部 `stiffness=60, damping=2, effort_limit=40, velocity_limit=17`；
   轮子 `stiffness=0, damping=0.2, effort_limit=5.0, velocity_limit=60`。
   本工程用的是 `stiffness=200/400, damping=5/10`（膝更硬，因为要顶住闭环约束），
   轮子 `effort_limit_sim=3.69`、`velocity_limit_sim=61.471`（**取自 URDF 实测值**）。
3. **轮速控制**：它用 `use_wheel_vel_control: true`、`wheel_vel_action_scale: 10.0`、
   `max_wheel_vel: 100.0`。本工程是 5.0 rad/s 每单位动作。
4. **仿真步长**：它 `dt=0.005`、`decimation=4` → **50 Hz 控制**，与本工程一致。
5. **奖励项命名**（可直接借来补全）：`wheel_acc`、`wheel_vel`、`wheel_power`、
   `wheel_air_spin`（腾空时别空转）、`action_smoothness_wheel`。
   本工程已有 `wheel_spin`，可以考虑补 `wheel_air_spin`。

**如果你想直接复现他们的成果**：他们的预训练权重在
`pretrained/26_infantry/**`（`.pt` / `.onnx`），
但必须用**他们仓库自己的环境**（`Robotics-Wheelbipe-V14-*-v0`）和
**他们自己的 USD**（`wheelbipeV14_2.usd`）来加载，
不能喂给本 URDF 的五连杆机器人 —— 观测/动作的维度和语义都对不上。

---

## 5. 复现命令

```powershell
$py = "D:\miniconda\envs\RM\python.exe"
cd D:\Robomaster\RL

# 1) 运动学验证（秒级，不需要 GPU）
& $py -B isaac_ws\isaac_ws_infantry\kinematics\check_planar_leg.py

# 2) 可达工作空间
& $py -B isaac_ws\isaac_ws_infantry\kinematics\tools/analyze_workspace.py

# 3) 闭环 + 站立验证（约 2 分钟，需要 GPU）→ PASS
& $py isaac_ws\scripts\check_robot.py --headless --num_envs 1 --steps 200 `
     --report isaac_ws\scripts\check_robot_report.json

# 4) 复现"多环境卡死"（这是本沙箱的限制）
& $py isaac_ws\scripts\diag_clone.py --num_envs 2 --steps 3 --headless

# 5) 代理资产（能建 256 环境，但会摔）
& $py isaac_ws\scripts\build_proxy_asset.py
& $py isaac_ws\scripts\diag_clone.py --num_envs 256 --steps 40 --asset proxy --headless

# 6) 训练（当前只能用 1 个环境）
& $py isaac_ws\scripts\train.py --task Infantry-RL-Balance-v0 --headless --num_envs 1
```

> 跑之前先确认没有残留的 Kit 进程：
> `Get-Process python | Stop-Process -Force`
> 残留进程会占住 GPU 和 Kit 的锁。

---

## 6. 资产选择开关

`isaac_ws_infantry/model.py` 里可以用环境变量切换资产：

```powershell
$env:INFANTRY_RL_ASSET = "full"    # 默认：完整 CAD 资产（已验证，只能 1 个环境）
$env:INFANTRY_RL_ASSET = "proxy"   # 代理资产（能多环境，但碰撞不准，会摔）
```

---

## 7. ✅ 训练与回放：已跑通（修复了两个真 bug）

修完之后，**训练和回放都验证通过**。

### 7.1 Bug ①：训练在第一次迭代前就崩 —— rsl_rl 记录 git diff 时编码失败

```
File "...\rsl_rl\utils\logger.py", line 305, in _store_code_state
    f.write(content)
UnicodeEncodeError: 'utf-8' codec can't encode characters in position 84951-84956: surrogates not allowed
```

**根因**（已实测复现，与 Kinematics/环境代码无关）：
本仓库的 git 索引里躺着 **48 个已删除文件**，其中
`infantry-rl/export.log` 是一个 1 MB 的 GBK 日志。
`rsl_rl` 的 `_store_code_state()` 用 `repo.git.diff(head_tree)` 取全量 diff，
GitPython 用 `surrogateescape` 解码非 UTF-8 的 blob，于是产生 **4797 个孤立 surrogate 字符**，
再用 `open(..., encoding="utf-8")` 写文件就必然抛 `UnicodeEncodeError`。

复现（不需要启动 Isaac Sim）：

```powershell
& $py -B -c "import git; r=git.Repo(r'D:\Robomaster\RL'); df=r.git.diff(r.head.commit.tree); print(sum(1 for l in df.split(chr(10)) if any(0xD800<=ord(c)<=0xDFFF for c in l)))"
# 输出 658 行含 surrogate
```

**修复**：给 `train.py` 加了 `--no_code_state` 开关，跳过这一步
（只影响那个 diff 存档文件，TensorBoard、checkpoint、params YAML 全部不受影响）。

### 7.2 Bug ②：`play.py` 直接崩 —— `--checkpoint` 参数重复注册

```
argparse.ArgumentError: argument --checkpoint: conflicting option string: --checkpoint
```

play.py 自己声明了 `--checkpoint`，而 `cli_args.add_rsl_rl_args(parser)` 也声明了一个。
**修复**：删掉 play.py 里的本地声明，改用 `agent_cfg.load_checkpoint` 来解析，
并支持三种写法（绝对路径 / 相对路径 / 纯文件名或 glob）。

### 7.3 实测结果

训练（1 环境，3 次迭代）：

```
[INFO] --no_code_state: skipped rsl_rl git-diff logging (2 repo path(s) dropped)
                             Learning iteration 0/3
...
                        Mean value loss: 0.0104
Training time: 4.57 seconds
[INFO] final closure residual (m): 7.552e-04
```

→ 正常跑完，**闭环残差仍然是 0.755 mm**（说明训练过程中闭环控制一直有效），
权重存到 `logs/rsl_rl/infantry_rl_balance/<时间戳>_<run_name>/model_*.pt`。

回放：

```
[INFO] Loading model checkpoint from: ...\model_2.pt
[play] step    0  base_z=0.1919  closure_max=2.370e-04 m
[play] steps=60  base_z mean=0.1943 m  max closure residual=5.795e-04 m
```

→ 正常加载并运行，**闭环残差 0.58 mm，机体高度 0.194 m**。

> 注意：3 次迭代当然训不出策略（`Episode_Reward/*` 全是 0.0000 是正常的 ——
> 回合长度 20 s = 1000 个控制步，太少迭代根本没有回合结束，reward buffer 是空的）。
> 要真正训练需要更多迭代 + 多环境，而多环境受第 3 节的限制。
