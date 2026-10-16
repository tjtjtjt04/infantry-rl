# infantry-rl — Isaac Sim 5.1.0 asset

闭链五连杆轮腿的仿真资产。源文件(`infantry-rl-main/`)一个字节都没动,这里全部是派生产物。

## 目录

| 路径 | 说明 |
|---|---|
| `urdf/infantry-rl-sim.urdf` | **生成物**,导入 Isaac Sim 用的 URDF(11 links / 10 joints,树) |
| `infantry_rl.usd` | 导入产物,IsaacLab 里 `UsdFileCfg` 引用的就是这个 |
| `configuration/*.usd` | 上面那个 USD 的 payload(几何 / 物理 / 传感器) |
| `meshes/*.STL` | 从源仓库拷来的网格,`package://infantry-rl/meshes/...` 靠这个相对结构解析 |
| `tools/build_sim_urdf.py` | 生成 URDF(唯一的权威脚本) |
| `tools/five_bar.py` | 单腿闭环运动学、装配姿态、初始姿态求解 |
| `tools/inspect_usd.py` | 打印 USD 里 PhysX 实际拿到的关节 / 限位 |
| `tools/test_loop_physx.py` | **验证闭链**:加零重力、给髋一个初速度,量闭环点的漂移 |
| `tools/check_axle_hole.py` | 后小腿网格与轮心的相对位置 |
| `tools/compare_right_leg.py` | 原始 vs 闭环两个 URDF 的右腿逐关节对比 |
| `tools/check_mesh_alignment.py` | 左右腿网格对镜像的偏差 |
| `tools/render_pose.py` | 把某个 URDF 的装配位形画成 PNG |

## 右腿:三个文件,谁对谁错

SW2URDF 导出的 `infantry-rl.urdf` 里,右前腿的**关节原点和网格文件是不自洽的**:
`right_front_knee` 的原点比左腿镜像值偏了 26.65 mm,`right_wheel` 偏了 15.88 mm,
而 `right_front_calf.STL` / `right_wheel.STL` 恰好在一个偏移了同样距离的坐标系里被导出。

于是就有了三方各错一半的局面:

| | 铰链位置 | 网格渲染 |
|---|---|---|
| `infantry-rl.urdf`(原始) | ✗ 偏 26.65 / 15.88 mm | ✓ 正确(网格补偿了) |
| `infantry-rl-closed.urdf` | ✓ 镜像值 | ✗ 网格跟着跑偏 |
| `infantry-rl-sim.urdf`(本文件) | ✓ 镜像值 | ✓ 给网格加 `<origin>` 补偿 |

**判据**是"网格到自己 link 铰链的最小距离",左腿作基准:

| | 前小腿 | 轮 |
|---|---|---|
| 左腿(基准) | 22.07 mm | 51.02 mm |
| 右腿 · 原始原点 | 4.41 mm ✗ | 35.98 mm ✗ |
| 右腿 · 镜像原点 + visual origin | **22.07 mm** ✓ | **51.13 mm** ✓ |

两条腿同一零件,这个距离必须一致。只有"镜像原点 + 网格补偿"能同时对上。

`infantry-rl-closed.urdf` 还额外把 `right_back_hip` 的 `rpy` 从 `0 0 -pi` 改成 `0 0 0`
而没转网格,导致整个右后腿网格绕 Z 转了 180°(对镜像的偏差 258 / 303 mm)——**后腿在原始文件里
本来就是对的,不该动**。本文件保留了原始的后腿。

闭环脚本的残差验证是自证的(loop 关节的 `<origin>` 由同一个模型推导),验不出这类问题。

## 对原始文件做的改动(仅此四类)

### 1. 修正右前腿的铰链 + 补偿网格

`right_front_knee` / `right_wheel` 的 `<origin>` 改成左腿的镜像值,同时给
`right_front_calf` / `right_wheel` 的 `<visual>` 和 `<collision>` 加 `<origin>`
(`+0.002270 0 +0.026549` 和 `-0.008400 +0.005000 +0.012539`),把网格留在 SolidWorks 摆的位置。
这两个补偿量由 `build_sim_urdf.py` 从两条运动链自动推导,不是手写的。

改完之后左右腿的 a0 和限位**逐项完全相同**,装配位形下两腿都是 `leg_len=0.15753 m`,解算器残差 0。

### 2. axis 符号统一到"写法 B"

`<axis>` 写在**关节自己的坐标系**里。两个右髋都带 `rpy="0 0 -pi"`,所以它们下面写 `0 -1 0`
在 base 系里其实是 `+Y`,和左腿一致。但 `right_front_knee` / `right_back_knee` / `right_wheel`
导出时**没做这个共轭**,在 base 系里变成 `+Y`,和左腿反号 —— 翻成 `0 1 0`。
髋轴的 `9.9049E-05` 是 SW 噪声(0.006°),清成 0。

写法 B 的含义:**左右腿同号**。实车右腿电机是镜像装配的,它的"逆时针为正"在世界坐标里跟左腿
相反,所以部署时右腿取负 —— 那是一张符号表,台架标定,不写进模型。

### 3. 右腿限位前后互换(真 bug)

原始导出里 `right_front_hip` 拿的是后髋的量程 `[1.3121, 3.3187]`,`right_back_hip` 拿的是前髋的
`[-0.17715, 1.3121]`,膝同理。按**机械角色**重新分配。

### 4. 限位重新参考到关节零位

URDF 的零位 = CAD 装配位形,但导出器写的 `<limit>` 是相对**下游连杆的绝对朝向**给的,不是相对
这个零位。照字面用,`q = 0` 会落在自己限位外,PhysX 第一步就把机器人拽回去。

证据:按这个约定,四个关节在装配位形下都落进区间,且距端点的余量两两精确相同
(髋 10.545°/10.543°,膝 9.323°/9.320°,误差 0.002°),换算后两条大腿分别从 0°→75°、
180°→75° 往下扫 —— 正是五连杆该有的运动。

最终限位(rad,左右腿相同):

| 关节 | lower | upper |
|---|---|---|
| front_hip | −0.184046 | +1.305204 |
| back_hip | −1.822596 | +0.184004 |
| front_knee | −0.162710 | +1.836400 |
| back_knee | −1.836398 | +0.162692 |

这只是**换个标签**,物理可达范围没变。

## 闭环:Isaac Sim 的 `<loop_joint>` URDF 扩展

**不要**用标准的 `<joint parent=... child=...>` 去接环。PhysX 直接报
`appears to be a part of a closed articulation, which is not supported`,然后**随机挑一个关节踢掉**
(实测那次踢掉的是 `left_front_hip`/`right_front_hip`,反而把 `*_back_calf_axle` 当成真关节,
拿到的是棵砍错边的树)。

正解是 Isaac Sim URDF 扩展(importer ≥ 2.1.0,2024-10)的 `<loop_joint>` 标签:

```xml
<loop_joint name="left_calf_axle" type="spherical">
  <link1 link="left_wheel"     rpy="0 0 0" xyz="0 0 0" />
  <link2 link="left_back_calf" rpy="0 0 0" xyz="0.22 0.028445 -0.15653" />
</loop_joint>
```

每个 link 在**自己的 link 坐标系**里给出一个点,约束这两个点重合。五连杆腿的这个点就是轮轴中心
(在轮坐标系里是原点,在后小腿坐标系里是上面那个值)。坐标由 `build_sim_urdf.py` 从模型自动算出,
不是手写的。

导入器把它变成 `/infantry_rl_sim/loop_joints/*` 下的 `PhysicsSphericalJoint` ——
**在 articulation 之外**,PhysX 当普通约束解,所以不会触发闭环错误。

**实测(Isaac Sim 5.1.0 + IsaacLab 2.3.2,`tools/test_loop_physx.py`):**

```
num_joints = 10   num_bodies = 11
joints: [8 个髋/膝 + 2 个轮]              ← 正确的 10 个自由度,闭环关节不在其中
left  |axle hole - wheel centre| at t=0 : 0.0000 mm
right |axle hole - wheel centre| at t=0 : 0.0000 mm
  t [s]   gap L [mm]   gap R [mm]   q_front_hip L   q_back_knee L
   0.00       0.0122       0.0124          0.0048          0.0019
   2.00       0.0015       0.0016          0.4062         -0.1797
```

机构在动(前髋扫了 0.4 rad,被动后膝自己跟着走到 −0.18 rad),闭环误差全程 **1.5 微米**。

**所以不需要 Python 侧的解算器。** 两个膝是真正的被动关节,PhysX 自动解出来,
policy 只需要输出 2 髋 + 2 轮 = 4 维/腿。`tools/five_bar.py` 现在只用来算初始姿态,
不再参与每步控制。

> 注意:每个 `<loop_joint>` 都**不产生驱动和限位**(官方文档原话),这是对的 —— 环里那个关节
> 本来就不该被驱动。

## 重新生成

```bash
PY="C:/Users/32560/.conda/envs/isaaclab/python.exe"

$PY assets/infantry-rl/tools/build_sim_urdf.py          # URDF
$PY scripts/tools/convert_urdf.py \                     # 在 IsaacLab-main 目录下跑
    assets/infantry-rl/urdf/infantry-rl-sim.urdf \
    assets/infantry-rl/infantry_rl.usd \
    --headless --merge-joints --joint-stiffness 0.0 --joint-damping 0.0
$PY assets/infantry-rl/tools/inspect_usd.py             # 核对关节
$PY assets/infantry-rl/tools/five_bar.py                # 闭环 + 初始姿态
```

## 在 GUI 里会看到的三件事

1. **闭链** —— 现在是**真的闭链**(`<loop_joint>` → PhysX 约束,实测 1.5 µm)。
2. **"启动后会散架"** —— 转换参数是 `--joint-stiffness 0.0 --joint-damping 0.0`,所有关节零刚度。
   GUI 里按 Play,重力直接把腿压塌。这是给 IsaacLab 用的正确参数(增益由
   `ArticulationCfg.actuators` 给),但裸 USD 在 GUI 里一定塌。想看它站着,在 GUI 里手动给
   关节加 stiffness,或者直接跑 IsaacLab 环境。
   > 注意闭环本身**不会**让它站着 —— 环只约束 3 个位置自由度,腿仍然会塌。
3. **右腿在旧版里位置不对** —— 见上面"右腿"一节。已修。

按 wiki 的验证方法,还可以:把某个 loop_joint 在 GUI 里 deactivate 再释放,闭环点会出现振动位移;
或者把关节 stiffness 全调 0 后释放,整条腿会被重力拉散 —— 这都说明环在起作用。

## 还没解决的

- **碰撞体**。视觉和碰撞都复用整块 STL,导入按 `convex_hull` 简化。`base_link.STL` 60 MB,
  前小腿 2.2 MB。凸包对这些零件是错的近似,训练前应换成球/胶囊/长方体。
- **自碰撞**默认关闭(`self_collision: false`)。闭环被切开后后小腿和轮子在几何上本来就重叠,
  必须保持关闭。
- **实车符号表**。写法 B 把左右同号定死,部署时需要每关节一个 ±1。左右互为相反数,所以独立
  标定量是 3 个(髋前 / 髋后 / 轮)。
