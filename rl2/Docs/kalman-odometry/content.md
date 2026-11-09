# 目 录

**第 1 章　卡尔曼滤波：完整公式与讲解**

1.1　问题描述与基本思想

1.2　状态空间模型与噪声假设

1.3　预测步：先验估计及其协方差

1.4　更新步：从联合高斯到条件高斯

1.5　增益 K 的独立推导：最小均方误差

1.6　五条核心公式汇总

1.7　连续时间模型与离散化

1.8　扩展卡尔曼滤波（EKF）

1.9　误差状态卡尔曼滤波（ESKF）

1.10　从通用 KF 到轮式里程计的对应关系

1.11　数值实现与一致性检验

**第 2 章　轮式里程计 ESKF：完整公式**

2.1　符号与坐标系

2.2　编码器 / 电机转速到轮速

2.3　滤波器状态定义

2.4　预测：IMU 机械编排

2.5　误差状态方程与协方差传播

2.6　观测方程

2.7　更新：增益、注入与重置

2.8　初始化

2.9　噪声换算与打滑判定

2.10　平衡底盘的杠杆臂修正

2.11　输出与场地系变换

2.12　每个控制周期的执行顺序

2.13　可选：在线标定的状态增广

# 第 1 章　卡尔曼滤波：完整公式与讲解

## 1.1　问题描述与基本思想

卡尔曼滤波解决的是同一个问题：一个动态系统的内部状态无法直接测量，只能得到与状态相关的含噪观测，如何由这些观测**递推**地给出状态的最优估计。

它需要三个已知条件：状态如何随时间演化（状态方程）、观测如何由状态产生（观测方程）、两处噪声的统计特性。在**线性**且**高斯**的假设下，卡尔曼滤波给出的估计是最优的：后验均值即最小均方误差估计，且递推可以只靠均值和协方差两阶矩闭式完成。

整个算法在结构上只有两步，交替执行：

- **预测**：用系统模型把上一时刻的后验估计外推到当前时刻，得到先验估计；不确定性因模型噪声而增大。
- **更新**：用当前时刻的观测修正先验估计，得到后验估计；不确定性因观测信息而减小。

## 1.2　状态空间模型与噪声假设

设离散时刻 $k=1,2,\dots$，状态 $\mathbf{x}_k\in\mathbb{R}^{n}$，控制输入 $\mathbf{u}_k\in\mathbb{R}^{m}$，观测 $\mathbf{z}_k\in\mathbb{R}^{p}$。

**状态方程**

$$
\mathbf{x}_k=\mathbf{F}_{k-1}\,\mathbf{x}_{k-1}+\mathbf{B}_{k-1}\,\mathbf{u}_{k-1}+\mathbf{w}_{k-1}
$$

**观测方程**

$$
\mathbf{z}_k=\mathbf{H}_k\,\mathbf{x}_k+\mathbf{v}_k
$$

其中 $\mathbf{F}$ 为状态转移矩阵，$\mathbf{B}$ 为控制输入矩阵，$\mathbf{H}$ 为观测矩阵，$\mathbf{w}$ 为过程噪声（模型不准确、外部扰动），$\mathbf{v}$ 为观测噪声（传感器误差）。

**噪声与初始状态的假设**（白噪声、零均值、高斯、互不相关）

$$
\mathbb{E}\left[\mathbf{w}_k\right]=\mathbf{0},\qquad
\mathbb{E}\left[\mathbf{w}_k\mathbf{w}_j^{\top}\right]=\mathbf{Q}_k\,\delta_{kj}
$$

$$
\mathbb{E}\left[\mathbf{v}_k\right]=\mathbf{0},\qquad
\mathbb{E}\left[\mathbf{v}_k\mathbf{v}_j^{\top}\right]=\mathbf{R}_k\,\delta_{kj}
$$

$$
\mathbb{E}\left[\mathbf{w}_k\mathbf{v}_j^{\top}\right]=\mathbf{0},\qquad
\mathbf{x}_0\sim\mathcal{N}\left(\hat{\mathbf{x}}_0,\ \mathbf{P}_0\right)
$$

$$
\mathbb{E}\left[\left(\mathbf{x}_0-\hat{\mathbf{x}}_0\right)\mathbf{w}_k^{\top}\right]=\mathbf{0},\qquad
\mathbb{E}\left[\left(\mathbf{x}_0-\hat{\mathbf{x}}_0\right)\mathbf{v}_k^{\top}\right]=\mathbf{0}
$$

$\mathbf{Q}_k$ 为过程噪声协方差（$n\times n$，半正定），$\mathbf{R}_k$ 为观测噪声协方差（$p\times p$，正定），$\delta_{kj}$ 为克罗内克函数。约定：上标 − 表示只用过去观测得到的**先验**，上标 + 表示加入当前观测后的**后验**。

## 1.3　预测步：先验估计及其协方差

预测步回答：若已知 $k-1$ 时刻的后验分布 $p(\mathbf{x}_{k-1}\,|\,\mathbf{z}_{1:k-1})$，则 $k$ 时刻的先验分布为

$$
p\left(\mathbf{x}_k\vert\mathbf{z}_{1:k-1}\right)
=\int p\left(\mathbf{x}_k\vert\mathbf{x}_{k-1}\right)p\left(\mathbf{x}_{k-1}\vert\mathbf{z}_{1:k-1}\right)\text{d}\mathbf{x}_{k-1}
$$

线性高斯系统下积分仍为高斯，只需传播前两阶矩。**均值**直接由状态方程传播：

$$
\hat{\mathbf{x}}_k^-=\mathbf{F}_{k-1}\,\hat{\mathbf{x}}_{k-1}^++\mathbf{B}_{k-1}\,\mathbf{u}_{k-1}
$$

**协方差**由误差传播得到。定义预测误差

$$
\mathbf{e}_k^-=\mathbf{x}_k-\hat{\mathbf{x}}_k^-=\mathbf{F}_{k-1}\left(\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1}^+\right)+\mathbf{w}_{k-1}
$$

两端取外积的期望，并利用 $\mathbf{w}$ 与 $\mathbf{x}_{k-1}-\hat{\mathbf{x}}_{k-1}^+$ 不相关：

$$
\mathbf{P}_k^-=\mathbb{E}\left[\mathbf{e}_k^-\left(\mathbf{e}_k^-\right)^{\top}\right]
=\mathbf{F}_{k-1}\,\mathbf{P}_{k-1}^+\,\mathbf{F}_{k-1}^{\top}+\mathbf{Q}_{k-1}
$$

**讲解**：先验协方差由两项叠加。第一项 $\mathbf{F}\mathbf{P}\mathbf{F}^{\top}$ 表示旧的估计误差被系统矩阵“拉伸”或“旋转”——系统把状态放大 10 倍，误差也放大 10 倍；第二项 $\mathbf{Q}$ 表示模型本身的不确定性。预测必然使不确定性**单调不减**（$\mathbf{Q}$ 半正定），这正是“没有观测信息时误差只会变大”的数学表达。

## 1.4　更新步：从联合高斯到条件高斯

更新步回答：已知先验 $\mathcal{N}\left(\hat{\mathbf{x}}_k^-,\mathbf{P}_k^-\right)$ 和观测 $\mathbf{z}_k=\mathbf{H}_k\mathbf{x}_k+\mathbf{v}_k$，求后验。

把状态与观测拼成联合高斯。由观测方程，$\mathbf{z}_k$ 的均值为 $\mathbf{H}_k\hat{\mathbf{x}}_k^-$，互协方差为 $\mathbf{P}_k^-\mathbf{H}_k^{\top}$，观测协方差为 $\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^{\top}+\mathbf{R}_k$：

$$
\begin{bmatrix}\mathbf{x}_k\\\mathbf{z}_k\end{bmatrix}
\sim\mathcal{N}\left(
\begin{bmatrix}\hat{\mathbf{x}}_k^-\\\mathbf{H}_k\hat{\mathbf{x}}_k^-\end{bmatrix},
\begin{bmatrix}
\mathbf{P}_k^- & \mathbf{P}_k^-\mathbf{H}_k^{\top}\\
\mathbf{H}_k\mathbf{P}_k^- & \mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^{\top}+\mathbf{R}_k
\end{bmatrix}\right)
$$

对任意联合高斯分布，条件分布的均值与协方差有闭式解：

$$
\mathbb{E}\left[\mathbf{x}\,|\,\mathbf{z}\right]=\boldsymbol{\mu}_x+\boldsymbol{\Sigma}_{xz}\boldsymbol{\Sigma}_{zz}^{-1}\left(\mathbf{z}-\boldsymbol{\mu}_z\right)
$$

$$
\text{Cov}\left(\mathbf{x}\,|\,\mathbf{z}\right)=\boldsymbol{\Sigma}_{xx}-\boldsymbol{\Sigma}_{xz}\boldsymbol{\Sigma}_{zz}^{-1}\boldsymbol{\Sigma}_{zx}
$$

把上面的分块代入，直接得到卡尔曼滤波的更新公式。先定义**新息**（innovation）与**新息协方差**：

$$
\boldsymbol{\nu}_k=\mathbf{z}_k-\mathbf{H}_k\hat{\mathbf{x}}_k^-
$$

$$
\mathbf{S}_k=\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^{\top}+\mathbf{R}_k
$$

则**卡尔曼增益**、**后验均值**与**后验协方差**为

$$
\mathbf{K}_k=\mathbf{P}_k^-\mathbf{H}_k^{\top}\mathbf{S}_k^{-1}
$$

$$
\hat{\mathbf{x}}_k^+=\hat{\mathbf{x}}_k^-+\mathbf{K}_k\,\boldsymbol{\nu}_k
$$

$$
\mathbf{P}_k^+=\left(\mathbf{I}-\mathbf{K}_k\mathbf{H}_k\right)\mathbf{P}_k^-
$$

后验协方差还有一个等价的对称形式（$\mathbf{K}_k$ 取最优值时两者严格相等）：

$$
\mathbf{P}_k^+=\mathbf{P}_k^--\mathbf{K}_k\mathbf{S}_k\mathbf{K}_k^{\top}
$$

**讲解**：这三项各有明确含义。

- $\boldsymbol{\nu}_k$ 是观测中“新”的部分——模型中已经预测到的成分被减掉了，剩下的才是需要修正的信息。
- $\mathbf{S}_k$ 是新息的方差：预测得越不准（$\mathbf{P}^-$ 大）、传感器越差（$\mathbf{R}$ 大），新息就越不可靠。
- $\mathbf{K}_k$ 是“状态-观测互协方差”与“新息协方差”的比值，即**该信谁**的比例。它自动完成两件事：把观测空间（$p$ 维）的修正量变换到状态空间（$n$ 维）；按不确定性加权。

极限情形可以检验直觉：$\mathbf{R}_k\to\mathbf{0}$ 时 $\mathbf{K}_k\to\mathbf{H}_k^{-1}$（完全相信观测）；$\mathbf{P}_k^-\to\mathbf{0}$ 时 $\mathbf{K}_k\to\mathbf{0}$（完全相信模型）；$\mathbf{H}_k=\mathbf{0}$ 表示观测与状态无关，$\mathbf{K}_k=\mathbf{0}$，观测被忽略。

**信息形式**（重要变体）：把协方差的逆定义为信息矩阵，更新可写成加法而不是求逆嵌套：

$$
\left(\mathbf{P}_k^+\right)^{-1}=\left(\mathbf{P}_k^-\right)^{-1}+\mathbf{H}_k^{\top}\mathbf{R}_k^{-1}\mathbf{H}_k
$$

这说明“信息可以相加”：先验的信息与观测提供的信息直接累加。多传感器融合时这一形式避免了频繁求逆。

## 1.5　增益 K 的独立推导：最小均方误差

上节的增益来自高斯条件分布。也可以从优化角度独立得到同一结果，这更能说明“最优”的含义。

设后验估计取线性修正形式

$$
\hat{\mathbf{x}}_k=\hat{\mathbf{x}}_k^-+\mathbf{K}\left(\mathbf{z}_k-\mathbf{H}_k\hat{\mathbf{x}}_k^-\right)
$$

估计误差为

$$
\mathbf{e}_k=\mathbf{x}_k-\hat{\mathbf{x}}_k=\left(\mathbf{I}-\mathbf{K}\mathbf{H}_k\right)\mathbf{e}_k^--\mathbf{K}\mathbf{v}_k
$$

其协方差为

$$
\mathbf{P}_k=\left(\mathbf{I}-\mathbf{K}\mathbf{H}_k\right)\mathbf{P}_k^-\left(\mathbf{I}-\mathbf{K}\mathbf{H}_k\right)^{\top}+\mathbf{K}\mathbf{R}_k\mathbf{K}^{\top}
$$

取代价函数为均方误差之和 $J=\text{tr}\left(\mathbf{P}_k\right)=\mathbb{E}\left[\mathbf{e}_k^{\top}\mathbf{e}_k\right]$，对 $\mathbf{K}$ 求导并令其为零：

$$
\frac{\partial J}{\partial\mathbf{K}}=-2\left(\mathbf{I}-\mathbf{K}\mathbf{H}_k\right)\mathbf{P}_k^-\mathbf{H}_k^{\top}+2\mathbf{K}\mathbf{R}_k=\mathbf{0}
$$

整理得

$$
\mathbf{K}\left(\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^{\top}+\mathbf{R}_k\right)=\mathbf{P}_k^-\mathbf{H}_k^{\top}
$$

$$
\mathbf{K}_k=\mathbf{P}_k^-\mathbf{H}_k^{\top}\mathbf{S}_k^{-1}
$$

与 1.4 节结果完全一致。二阶导数为正，确为极小点。因此卡尔曼增益是**使后验协方差的迹最小**的线性增益；在联合高斯的条件下，线性估计已经是最优的，所以它同时是全局的最小均方误差估计。

另一个等价解释是**正交投影**：最优估计的误差必须与新息正交，

$$
\mathbb{E}\left[\left(\mathbf{x}_k-\hat{\mathbf{x}}_k^+\right)\boldsymbol{\nu}_k^{\top}\right]=\mathbf{0}
$$

展开后同样解出 $\mathbf{K}_k=\mathbf{P}_k^-\mathbf{H}_k^{\top}\mathbf{S}_k^{-1}$。这就是“新息中不再包含可用于修正的信息”的意思。

## 1.6　五条核心公式汇总

| 步骤 | 公式 | 含义 |
|---|---|---|
| 预测（均值） | $\hat{\mathbf{x}}_k^-=\mathbf{F}_{k-1}\hat{\mathbf{x}}_{k-1}^++\mathbf{B}_{k-1}\mathbf{u}_{k-1}$ | 状态按模型外推 |
| 预测（协方差） | $\mathbf{P}_k^-=\mathbf{F}_{k-1}\mathbf{P}_{k-1}^+\mathbf{F}_{k-1}^{\top}+\mathbf{Q}_{k-1}$ | 不确定性增大 |
| 新息与协方差 | $\mathbf{S}_k=\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^{\top}+\mathbf{R}_k$ | 观测预测的方差 |
| 增益 | $\mathbf{K}_k=\mathbf{P}_k^-\mathbf{H}_k^{\top}\mathbf{S}_k^{-1}$ | 按信噪比加权 |
| 更新 | $\hat{\mathbf{x}}_k^+=\hat{\mathbf{x}}_k^-+\mathbf{K}_k\boldsymbol{\nu}_k$，$\mathbf{P}_k^+=\left(\mathbf{I}-\mathbf{K}_k\mathbf{H}_k\right)\mathbf{P}_k^-$ | 修正并降低不确定性 |

完整算法还需初始化 $\hat{\mathbf{x}}_0^+$、$\mathbf{P}_0^+$，并保证 $\mathbf{Q}_k$、$\mathbf{R}_k$ 的对称性与正定性。

## 1.7　连续时间模型与离散化

物理系统通常以连续微分方程给出：

$$
\dot{\mathbf{x}}(t)=\mathbf{A}\,\mathbf{x}(t)+\mathbf{B}\,\mathbf{u}(t)+\mathbf{w}_c(t)
$$

$$
\mathbb{E}\left[\mathbf{w}_c(t)\mathbf{w}_c(\tau)^{\top}\right]=\mathbf{Q}_c\,\delta\left(t-\tau\right)
$$

在采样周期 $\Delta t$ 上做零阶保持离散化：

$$
\mathbf{F}=e^{\mathbf{A}\Delta t}=\mathbf{I}+\mathbf{A}\Delta t+\frac{1}{2}\mathbf{A}^{2}\Delta t^{2}+\cdots
$$

$$
\mathbf{Q}=\int_{0}^{\Delta t}e^{\mathbf{A}\tau}\mathbf{Q}_c\,e^{\mathbf{A}^{\top}\tau}\,\text{d}\tau
\approx\mathbf{Q}_c\Delta t+\frac{1}{2}\left(\mathbf{A}\mathbf{Q}_c+\mathbf{Q}_c\mathbf{A}^{\top}\right)\Delta t^{2}+\cdots
$$

工程上最常用的近似是 $\mathbf{F}\approx\mathbf{I}+\mathbf{A}\Delta t$、$\mathbf{Q}\approx\mathbf{Q}_c\Delta t$；当 $\Delta t$ 较小（1 kHz）时误差可忽略。传感器手册给出的噪声密度（单位 $\text{/}\sqrt{\text{Hz}}$）换算为单步离散标准差时用

$$
\sigma=\frac{\text{ND}}{\sqrt{\Delta t}}
$$

理由：噪声密度 $\text{ND}$ 对应功率谱密度 $\text{ND}^{2}$，在带宽 $1/(2\Delta t)$ 内的方差为 $\text{ND}^{2}/(2\Delta t)$ 的两倍量级，工程上统一取 $\text{ND}^{2}/\Delta t$，即 $\sigma=\text{ND}/\sqrt{\Delta t}$。

## 1.8　扩展卡尔曼滤波（EKF）

真实系统几乎都是非线性的：

$$
\mathbf{x}_k=\mathbf{f}\left(\mathbf{x}_{k-1},\mathbf{u}_{k-1}\right)+\mathbf{w}_{k-1}
$$

$$
\mathbf{z}_k=\mathbf{h}\left(\mathbf{x}_k\right)+\mathbf{v}_k
$$

EKF 的思路是**在当前估计点做一阶线性化**，然后套用线性卡尔曼滤波的公式。雅可比矩阵为

$$
\mathbf{F}_{k-1}=\left.\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\right|_{\hat{\mathbf{x}}_{k-1}^+,\ \mathbf{u}_{k-1}},
\qquad
\mathbf{H}_k=\left.\frac{\partial\mathbf{h}}{\partial\mathbf{x}}\right|_{\hat{\mathbf{x}}_k^-}
$$

预测与更新变为

$$
\hat{\mathbf{x}}_k^-=\mathbf{f}\left(\hat{\mathbf{x}}_{k-1}^+,\mathbf{u}_{k-1}\right)
$$

$$
\mathbf{P}_k^-=\mathbf{F}_{k-1}\mathbf{P}_{k-1}^+\mathbf{F}_{k-1}^{\top}+\mathbf{Q}_{k-1}
$$

$$
\mathbf{K}_k=\mathbf{P}_k^-\mathbf{H}_k^{\top}\left(\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^{\top}+\mathbf{R}_k\right)^{-1}
$$

$$
\hat{\mathbf{x}}_k^+=\hat{\mathbf{x}}_k^-+\mathbf{K}_k\left(\mathbf{z}_k-\mathbf{h}\left(\hat{\mathbf{x}}_k^-\right)\right)
$$

$$
\mathbf{P}_k^+=\left(\mathbf{I}-\mathbf{K}_k\mathbf{H}_k\right)\mathbf{P}_k^-
$$

**EKF 的三个问题**，正是下面选择 ESKF 的原因。

1. 线性化点在估计值处，估计误差越大线性化越差，强非线性时可能发散或不一致。
2. 姿态是流形上的量（旋转群 $SO(3)$），用四元数 4 维表示做 EKF 时，协方差会沿“模长为 1”的约束方向出现奇异，必须额外归一化并处理奇异值。
3. 每次更新都直接改状态，状态被反复“拉回”，线性化点不断跳动。

## 1.9　误差状态卡尔曼滤波（ESKF）

ESKF 把状态拆成两部分：**名义状态**（大信号、非线性、不含噪声）与**误差状态**（小信号、线性、含噪声）。

$$
\mathbf{x}_t=\mathbf{x}\oplus\delta\mathbf{x},
\qquad
\delta\mathbf{x}\sim\mathcal{N}\left(\mathbf{0},\mathbf{P}\right)
$$

其中 $\oplus$ 表示在流形上的复合（对位置速度是加法，对姿态是四元数乘法）。

**预测**：名义状态用含控制输入的非线性模型直接积分（不含噪声）；误差状态用一阶线性模型传播：

$$
\mathbf{x}_k=\mathbf{f}\left(\mathbf{x}_{k-1},\mathbf{u}_{k-1}\right)
$$

$$
\delta\mathbf{x}_k^-=\boldsymbol{\Phi}_{k-1}\,\delta\mathbf{x}_{k-1}+\mathbf{w}_{k-1},
\qquad
\boldsymbol{\Phi}_{k-1}\approx\mathbf{I}+\mathbf{F}_c\,\Delta t
$$

$$
\mathbf{P}_k^-=\boldsymbol{\Phi}_{k-1}\mathbf{P}_{k-1}^+\boldsymbol{\Phi}_{k-1}^{\top}+\mathbf{Q}_{k-1}
$$

$\mathbf{F}_c$ 是误差状态的连续系统矩阵，由标称轨迹线性化得到（第 2.5 节给出导航系统的具体形式）。

**更新**：观测模型对误差状态线性化，线性化点恒为 $\delta\mathbf{x}=\mathbf{0}$：

$$
\mathbf{H}_k=\left.\frac{\partial\,\mathbf{h}\left(\mathbf{x}\oplus\delta\mathbf{x}\right)}{\partial\,\delta\mathbf{x}}\right|_{\delta\mathbf{x}=\mathbf{0}}
$$

$$
\mathbf{K}_k=\mathbf{P}_k^-\mathbf{H}_k^{\top}\left(\mathbf{H}_k\mathbf{P}_k^-\mathbf{H}_k^{\top}+\mathbf{R}_k\right)^{-1}
$$

$$
\delta\hat{\mathbf{x}}_k=\mathbf{K}_k\left(\mathbf{z}_k-\mathbf{h}\left(\mathbf{x}_k\right)\right)
$$

**注入**：把误差估计写回名义状态，得到真正的状态估计：

$$
\mathbf{x}_k\leftarrow\mathbf{x}_k\oplus\delta\hat{\mathbf{x}}_k
$$

**重置**：注入之后误差状态的估计值归零，协方差做一次一阶重置变换（常可忽略）：

$$
\delta\hat{\mathbf{x}}_k\leftarrow\mathbf{0},
\qquad
\mathbf{P}_k^{+}\leftarrow\mathbf{G}\,\mathbf{P}_k^{+}\,\mathbf{G}^{\top}
$$

**ESKF 相对 EKF 的四个优势**：

1. 线性化点固定在误差为零处，误差本身是小量，一阶近似精度高得多。
2. 姿态误差用 3 维旋转矢量表示，协方差是 $3\times3$ 正定矩阵，没有四元数模长约束带来的奇异。
3. 名义状态由非线性方程积分，永远保持归一化、永远合法，不会被观测噪声“污染”。
4. 高频 IMU 下误差始终很小，$O(\|\delta\mathbf{x}\|^2)$ 的忽略项可以放心丢掉。

代价是需要维护“名义 + 误差”两套量，并在每次更新后做注入与重置。对导航系统而言，这点代价换来的是长期稳定性，非常值得。

## 1.10　从通用 KF 到轮式里程计的对应关系

把上面的一般公式逐项对应到本文第 2 章的具体算法：

| 卡尔曼滤波中的概念 | 轮式里程计中的具体对象 |
|---|---|
| 状态 $\mathbf{x}$ | 位置 $\mathbf{p}$、速度 $\mathbf{v}$、姿态四元数 $q$、陀螺零偏 $\mathbf{b}_g$、加计零偏 $\mathbf{b}_a$ |
| 控制输入 $\mathbf{u}$ | IMU 的陀螺角速度与加计比力 |
| 状态方程 $\mathbf{f}$ | 惯性机械编排（四元数积分 + 速度位置积分） |
| 过程噪声 $\mathbf{Q}$ | IMU 噪声密度与零偏随机游走 |
| 观测 $\mathbf{z}$ | 左右轮速、轮速差、非完整约束、静止检测 |
| 观测方程 $\mathbf{h}$ | 差速运动学与轮速约束 |
| 观测噪声 $\mathbf{R}$ | 编码器量化噪声、电机转速误差、打滑 |
| 误差状态 | 15 维：$\delta\mathbf{p},\delta\mathbf{v},\delta\boldsymbol{\theta},\delta\mathbf{b}_g,\delta\mathbf{b}_a$ |

## 1.11　数值实现与一致性检验

**协方差对称化**：理论上 $\mathbf{P}$ 恒对称，浮点运算会破坏对称性并导致发散。每次更新后强制

$$
\mathbf{P}\leftarrow\frac{1}{2}\left(\mathbf{P}+\mathbf{P}^{\top}\right)
$$

**Joseph 形式**：当增益不是严格最优（数值误差）时，$\left(\mathbf{I}-\mathbf{K}\mathbf{H}\right)\mathbf{P}^-$ 可能失去正定性，用等价形式

$$
\mathbf{P}^+=\left(\mathbf{I}-\mathbf{K}\mathbf{H}\right)\mathbf{P}^-\left(\mathbf{I}-\mathbf{K}\mathbf{H}\right)^{\top}+\mathbf{K}\mathbf{R}\mathbf{K}^{\top}
$$

可以保证正定性。

**新息一致性检验**：若模型与噪声统计正确，归一化新息平方和服从卡方分布：

$$
d^{2}=\boldsymbol{\nu}^{\top}\mathbf{S}^{-1}\boldsymbol{\nu}\ \sim\ \chi^{2}\left(p\right)
$$

$p$ 为观测维数。95% 置信门限：1 维观测 $3.84$，2 维观测 $5.99$，3 维观测 $7.81$。超过门限说明观测异常（打滑、碰撞、传感器故障），应剔除该观测或将 $\mathbf{R}$ 放大。

**其他要点**：角度必须归一化到 $\left(-\pi,\pi\right]$；四元数每步归一化；$d^{2}$ 与协方差都用双精度或单精度但要检查溢出；低频观测与高频预测的比例由 $\mathbf{Q}$ 与 $\mathbf{R}$ 的比值决定，这是调参的核心。

# 第 2 章　轮式里程计 ESKF：完整公式

本章把第 1 章的通用公式落到具体系统：两轮差速（含平衡）底盘，传感器为 BMI088 IMU 与左右轮电机转速/编码器。

## 2.1　符号与坐标系

世界系 $\{W\}$ 取 $z$ 轴朝天，重力矢量

$$
\mathbf{g}_W=\begin{bmatrix}0&0&-9.80665\end{bmatrix}^{\top}
\quad\left[\text{m/s}^{2}\right]
$$

机体系 $\{B\}$ 取 $x$ 轴向前、$y$ 轴向左、$z$ 轴向上。若 BMI088 的轴系与机体系不重合，先用安装矩阵变换：

$$
\boldsymbol{\omega}_B=R_{BM}\,\boldsymbol{\omega}_{\text{raw}},
\qquad
\mathbf{a}_B=R_{BM}\,\mathbf{a}_{\text{raw}}
$$

| 符号 | 含义 |
|---|---|
| $\mathbf{p}\in\mathbb{R}^{3}$ | 位置，世界系，m |
| $\mathbf{v}\in\mathbb{R}^{3}$ | 速度，世界系，m/s |
| $q\in\mathbb{S}^{3}$ | 姿态四元数，$B\to W$ |
| $\mathbf{b}_g,\ \mathbf{b}_a\in\mathbb{R}^{3}$ | 陀螺、加计零偏，机体系 |
| $R(q)$ | 由四元数得到的旋转矩阵 |
| $\left[\mathbf{u}\right]_{\times}$ | 向量对应的反对称矩阵 |
| $r_i,\ b,\ i_{g,i}$ | 轮半径、轮距、减速比 |

**IMU 量测模型**

$$
\boldsymbol{\omega}_m=\boldsymbol{\omega}_B+\mathbf{b}_g+\mathbf{n}_g
$$

$$
\mathbf{a}_m=R^{\top}\left(\mathbf{a}_W-\mathbf{g}_W\right)+\mathbf{b}_a+\mathbf{n}_a
$$

其中 $\mathbf{a}_W$ 为载体系加速度，$\mathbf{a}_m$ 为比力（加速度计读数）。**反对称矩阵**

$$
\left[\mathbf{u}\right]_{\times}=
\begin{bmatrix}
0 & -u_3 & u_2\\
u_3 & 0 & -u_1\\
-u_2 & u_1 & 0
\end{bmatrix}
$$

**四元数运算**

$$
q\otimes p=
\begin{bmatrix}
w_qw_p-\mathbf{v}_q^{\top}\mathbf{v}_p\\
w_q\mathbf{v}_p+w_p\mathbf{v}_q+\mathbf{v}_q\times\mathbf{v}_p
\end{bmatrix},
\qquad q=\left(w_q,\mathbf{v}_q\right)
$$

$$
R(q)=\mathbf{I}_3+2w\left[\mathbf{v}\right]_{\times}+2\left[\mathbf{v}\right]_{\times}^{2}
$$

$$
\text{Exp}\left(\boldsymbol{\phi}\right)=
\begin{bmatrix}
\cos\left(\left\|\boldsymbol{\phi}\right\|/2\right)\\
\sin\left(\left\|\boldsymbol{\phi}\right\|/2\right)\dfrac{\boldsymbol{\phi}}{\left\|\boldsymbol{\phi}\right\|}
\end{bmatrix}
$$

## 2.2　编码器 / 电机转速到轮速

对每侧 $i\in\{L,R\}$，编码器每转计数为

$$
N_{\text{rev},i}=\text{PPR}_i\cdot 4\cdot i_{g,i}
$$

其中 $\text{PPR}$ 为电机轴每转脉冲数，因子 4 为四倍频正交解码。编码器差分得到轮速

$$
v_{i,k}=\frac{2\pi r_i\left(N_{i,k}-N_{i,k-1}\right)}{N_{\text{rev},i}\,\Delta t_{\text{enc}}}
\quad\left[\text{m/s}\right]
$$

由电机反馈转速 $n_i$（rpm）或电机角速度 $\omega_{\text{motor},i}$ 得到

$$
v_i=\frac{2\pi r_i\,n_i}{60\,i_{g,i}}=\frac{r_i\,\omega_{\text{motor},i}}{i_{g,i}}
$$

编码器差分的量化噪声标准差为

$$
\sigma_{\text{enc},i}=\frac{2\pi r_i}{N_{\text{rev},i}\,\Delta t_{\text{enc}}\sqrt{12}}
$$

**差速运动学**（无滑移假设）

$$
v_x^{w}=\frac{v_L+v_R}{2},
\qquad
\omega_z^{w}=\frac{v_R-v_L}{b}
$$

工程分工建议：电机转速平滑、延迟小，用作速度观测；编码器差分用于里程增量、轮径/轮距标定与打滑检测。低速时编码器差分噪声大，应改用多周期差分或电机转速。

## 2.3　滤波器状态定义

**名义状态**

$$
\mathbf{x}=\left[\mathbf{p},\ \mathbf{v},\ q,\ \mathbf{b}_g,\ \mathbf{b}_a\right]
$$

**误差状态**

$$
\delta\mathbf{x}=\left[\delta\mathbf{p},\ \delta\mathbf{v},\ \delta\boldsymbol{\theta},\ \delta\mathbf{b}_g,\ \delta\mathbf{b}_a\right]\in\mathbb{R}^{15}
$$

误差定义采用**局部（机体系）角度误差**：

$$
\mathbf{p}_t=\mathbf{p}+\delta\mathbf{p},
\quad
\mathbf{v}_t=\mathbf{v}+\delta\mathbf{v},
\quad
q_t=q\otimes\text{Exp}\left(\delta\boldsymbol{\theta}\right),
\quad
\mathbf{b}_{g,t}=\mathbf{b}_g+\delta\mathbf{b}_g,
\quad
\mathbf{b}_{a,t}=\mathbf{b}_a+\delta\mathbf{b}_a
$$

每次更新注入后误差状态清零。

## 2.4　预测：IMU 机械编排

对每个 IMU 采样周期 $\Delta t$，先做零偏补偿：

$$
\boldsymbol{\omega}=\boldsymbol{\omega}_m-\mathbf{b}_g,
\qquad
\mathbf{a}_B=\mathbf{a}_m-\mathbf{b}_a,
\qquad
R=R(q)
$$

四元数积分（精确形式）：

$$
\Delta\boldsymbol{\theta}=\boldsymbol{\omega}\,\Delta t,
\qquad
\Delta q=\text{Exp}\left(\Delta\boldsymbol{\theta}\right)
$$

$$
q^-=\text{normalize}\left(q\otimes\Delta q\right)
$$

世界系加速度、速度与位置：

$$
\mathbf{a}_W=R\,\mathbf{a}_B+\mathbf{g}_W
$$

$$
\mathbf{v}^-=\mathbf{v}+\mathbf{a}_W\,\Delta t
$$

$$
\mathbf{p}^-=\mathbf{p}+\mathbf{v}\,\Delta t+\frac{1}{2}\,\mathbf{a}_W\,\Delta t^{2}
$$

零偏在预测中保持不变：

$$
\mathbf{b}_g^-=\mathbf{b}_g,
\qquad
\mathbf{b}_a^-=\mathbf{b}_a
$$

若需更高精度，可用相邻两条 IMU 的平均值 $\frac{1}{2}\left(\boldsymbol{\omega}_k+\boldsymbol{\omega}_{k+1}\right)$、$\frac{1}{2}\left(\mathbf{a}_k+\mathbf{a}_{k+1}\right)$ 做中点积分。

## 2.5　误差状态方程与协方差传播

**连续误差动力学**（对 2.1 节的量测模型线性化）

$$
\delta\dot{\mathbf{p}}=\delta\mathbf{v}
$$

$$
\delta\dot{\mathbf{v}}=-R\left[\mathbf{a}_B\right]_{\times}\delta\boldsymbol{\theta}-R\,\delta\mathbf{b}_a-R\,\mathbf{n}_a
$$

$$
\delta\dot{\boldsymbol{\theta}}=-\left[\boldsymbol{\omega}\right]_{\times}\delta\boldsymbol{\theta}-\delta\mathbf{b}_g-\mathbf{n}_g
$$

$$
\delta\dot{\mathbf{b}}_g=\mathbf{n}_{bg},
\qquad
\delta\dot{\mathbf{b}}_a=\mathbf{n}_{ba}
$$

**连续系统矩阵**（列序 $\delta\mathbf{p},\delta\mathbf{v},\delta\boldsymbol{\theta},\delta\mathbf{b}_g,\delta\mathbf{b}_a$）

$$
\mathbf{F}_c=
\begin{bmatrix}
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3\\
\mathbf{0}_3 & \mathbf{0}_3 & -R\left[\mathbf{a}_B\right]_{\times} & \mathbf{0}_3 & -R\\
\mathbf{0}_3 & \mathbf{0}_3 & -\left[\boldsymbol{\omega}\right]_{\times} & -\mathbf{I}_3 & \mathbf{0}_3\\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3\\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3
\end{bmatrix}
$$

**离散化与协方差预测**

$$
\boldsymbol{\Phi}=\mathbf{I}_{15}+\mathbf{F}_c\,\Delta t
$$

$$
\mathbf{P}^-=\boldsymbol{\Phi}\,\mathbf{P}\,\boldsymbol{\Phi}^{\top}+\mathbf{Q}_d
$$

$$
\mathbf{P}^-\leftarrow\frac{1}{2}\left(\mathbf{P}^-+\left(\mathbf{P}^-\right)^{\top}\right)
$$

$\delta\boldsymbol{\theta}$ 对应的对角块可用精确式 $\text{Exp}\left(-\left[\boldsymbol{\omega}\right]_{\times}\Delta t\right)$ 代替一阶近似。

**离散过程噪声**（$\sigma$ 均为单步离散标准差）

$$
\mathbf{Q}_d=
\begin{bmatrix}
\frac{1}{4}\sigma_a^{2}\Delta t^{4}\mathbf{I}_3 & \frac{1}{2}\sigma_a^{2}\Delta t^{3}\mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3\\
\frac{1}{2}\sigma_a^{2}\Delta t^{3}\mathbf{I}_3 & \sigma_a^{2}\Delta t^{2}\mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3\\
\mathbf{0}_3 & \mathbf{0}_3 & \sigma_g^{2}\Delta t^{2}\mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3\\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \sigma_{bg}^{2}\Delta t\,\mathbf{I}_3 & \mathbf{0}_3\\
\mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \sigma_{ba}^{2}\Delta t\,\mathbf{I}_3
\end{bmatrix}
$$

## 2.6　观测方程

记机体系速度 $\mathbf{w}=R^{\top}\mathbf{v}$，基向量 $\mathbf{e}_1=\left[1,0,0\right]^{\top}$、$\mathbf{e}_2=\left[0,1,0\right]^{\top}$、$\mathbf{e}_3=\left[0,0,1\right]^{\top}$。以下 $\mathbf{H}$ 的列序固定为 $\left[\delta\mathbf{p}\vert\delta\mathbf{v}\vert\delta\boldsymbol{\theta}\vert\delta\mathbf{b}_g\vert\delta\mathbf{b}_a\right]$。

### 2.6.1　轮速（二维块观测，推荐形式）

$$
\mathbf{z}=\begin{bmatrix}v_L\\v_R\end{bmatrix}
$$

$$
h_L\left(\mathbf{x}\right)=\mathbf{e}_1^{\top}R^{\top}\mathbf{v}-\frac{b}{2}\,\mathbf{e}_3^{\top}\left(\boldsymbol{\omega}_m-\mathbf{b}_g\right)
$$

$$
h_R\left(\mathbf{x}\right)=\mathbf{e}_1^{\top}R^{\top}\mathbf{v}+\frac{b}{2}\,\mathbf{e}_3^{\top}\left(\boldsymbol{\omega}_m-\mathbf{b}_g\right)
$$

$$
\mathbf{H}_L=
\begin{bmatrix}
\mathbf{0}_{1\times3}\ \vert\ \mathbf{e}_1^{\top}R^{\top}\ \vert\ \left(\mathbf{e}_1\times\mathbf{w}\right)^{\top}\ \vert\ +\frac{b}{2}\mathbf{e}_3^{\top}\ \vert\ \mathbf{0}_{1\times3}
\end{bmatrix}
$$

$$
\mathbf{H}_R=
\begin{bmatrix}
\mathbf{0}_{1\times3}\ \vert\ \mathbf{e}_1^{\top}R^{\top}\ \vert\ \left(\mathbf{e}_1\times\mathbf{w}\right)^{\top}\ \vert\ -\frac{b}{2}\mathbf{e}_3^{\top}\ \vert\ \mathbf{0}_{1\times3}
\end{bmatrix}
$$

$$
\mathbf{R}_{\text{meas}}=\text{diag}\left(\sigma_L^{2},\ \sigma_R^{2}\right)
$$

这一形式同时把“前向速度”和“轮速差对应偏航角速度”纳入同一观测，并直接给出陀螺 $z$ 轴零偏的可观测性。

### 2.6.2　等价的标量拆解

前向速度观测：

$$
z_1=\frac{v_L+v_R}{2},
\qquad
h_1=\mathbf{e}_1^{\top}R^{\top}\mathbf{v}
$$

$$
\mathbf{H}_1=\begin{bmatrix}\mathbf{0}\vert\mathbf{e}_1^{\top}R^{\top}\vert\left(\mathbf{e}_1\times\mathbf{w}\right)^{\top}\vert\mathbf{0}\vert\mathbf{0}\end{bmatrix},
\qquad
R_1=\frac{\sigma_L^{2}+\sigma_R^{2}}{4}
$$

偏航角速度观测：

$$
z_3=\frac{v_R-v_L}{b},
\qquad
h_3=\mathbf{e}_3^{\top}\left(\boldsymbol{\omega}_m-\mathbf{b}_g\right)
$$

$$
\mathbf{H}_3=\begin{bmatrix}\mathbf{0}\vert\mathbf{0}\vert\mathbf{0}\vert-\mathbf{e}_3^{\top}\vert\mathbf{0}\end{bmatrix},
\qquad
R_3=\frac{\sigma_L^{2}+\sigma_R^{2}}{b^{2}}
$$

两者与 2.6.1 的块观测在数学上完全等价，可按实现方便选择。

### 2.6.3　侧向零速约束（非完整约束，必加）

$$
z_2=0,
\qquad
h_2=\mathbf{e}_2^{\top}R^{\top}\mathbf{v}
$$

$$
\mathbf{H}_2=\begin{bmatrix}\mathbf{0}\vert\mathbf{e}_2^{\top}R^{\top}\vert\left(\mathbf{e}_2\times\mathbf{w}\right)^{\top}\vert\mathbf{0}\vert\mathbf{0}\end{bmatrix},
\qquad
R_2=\sigma_{vy}^{2}\in\left[0.05,\ 0.5\right]
$$

差速底盘与平衡底盘都不能侧向平移，该约束是抑制航向与速度耦合漂移的主要手段。

### 2.6.4　垂向零速约束（平地）

$$
z_4=0,
\qquad
h_4=\mathbf{e}_3^{\top}\mathbf{v}
$$

$$
\mathbf{H}_4=\begin{bmatrix}\mathbf{0}\vert\mathbf{e}_3^{\top}\vert\mathbf{0}\vert\mathbf{0}\vert\mathbf{0}\end{bmatrix},
\qquad
R_4=\sigma_{vz}^{2}
$$

用于抑制高度通道的积分漂移。

### 2.6.5　静止门控观测

零速更新（ZUPT）：

$$
\mathbf{z}=\mathbf{0}_3,
\qquad
h=\mathbf{v},
\qquad
\mathbf{H}=\begin{bmatrix}\mathbf{0}\vert\mathbf{I}_3\vert\mathbf{0}\vert\mathbf{0}\vert\mathbf{0}\end{bmatrix},
\qquad
R=\sigma_{\text{zupt}}^{2}\mathbf{I}_3
$$

零角速度更新（ZARU，用于估计陀螺零偏）：

$$
\mathbf{z}=\mathbf{0}_3,
\qquad
h=\boldsymbol{\omega}_m-\mathbf{b}_g,
\qquad
\mathbf{H}=\begin{bmatrix}\mathbf{0}\vert\mathbf{0}\vert\mathbf{0}\vert-\mathbf{I}_3\vert\mathbf{0}\end{bmatrix},
\qquad
R=\sigma_{\omega}^{2}\mathbf{I}_3
$$

准静态重力方向观测，用于给 roll/pitch 与加计零偏提供可观测性。下面两式同时满足时启用：

$$
\left|\ \left\|\mathbf{a}_m-\mathbf{b}_a\right\|-g\ \right|<\varepsilon_1
$$

$$
\left|\frac{\Delta v_{\text{wheel}}}{\Delta t}\right|<\varepsilon_2
$$

观测方程：

$$
\mathbf{z}=\mathbf{a}_m,
\qquad
h=\left\|\mathbf{g}_W\right\|\,\mathbf{u}+\mathbf{b}_a,
\qquad
\mathbf{u}=R^{\top}\mathbf{e}_3
$$

$$
\mathbf{H}=\begin{bmatrix}\mathbf{0}\vert\mathbf{0}\vert\left\|\mathbf{g}\right\|\left[\mathbf{u}\right]_{\times}\vert\mathbf{0}\vert\mathbf{I}_3\end{bmatrix},
\qquad
R=\sigma_a^{2}\mathbf{I}_3
$$

### 2.6.6　外部定位（有场地定位、二维码、UWB 时可选）

位置观测：

$$
\mathbf{z}=\mathbf{p}^{\text{field}},
\qquad
h=T^{W\to\text{field}}\left(\mathbf{p}\right),
\qquad
\mathbf{H}=\begin{bmatrix}\partial h/\partial\mathbf{p}\vert\mathbf{0}\vert\mathbf{0}\vert\mathbf{0}\vert\mathbf{0}\end{bmatrix}
$$

航向观测：

$$
z=\psi^{\text{field}},
\qquad
\psi=\text{atan2}\left(2\left(q_wq_z+q_xq_y\right),\ 1-2\left(q_y^{2}+q_z^{2}\right)\right)
$$

## 2.7　更新：增益、注入与重置

按第 1.4、1.9 节的通用形式：

$$
\mathbf{r}=\mathbf{z}-h\left(\mathbf{x}^-\right)
$$

$$
\mathbf{S}=\mathbf{H}\,\mathbf{P}^-\mathbf{H}^{\top}+\mathbf{R}
$$

$$
\mathbf{K}=\mathbf{P}^-\mathbf{H}^{\top}\mathbf{S}^{-1}
$$

$$
\delta\hat{\mathbf{x}}=\mathbf{K}\,\mathbf{r}
$$

Joseph 形式协方差更新与对称化：

$$
\mathbf{P}^+=\left(\mathbf{I}-\mathbf{K}\mathbf{H}\right)\mathbf{P}^-\left(\mathbf{I}-\mathbf{K}\mathbf{H}\right)^{\top}+\mathbf{K}\,\mathbf{R}\,\mathbf{K}^{\top}
$$

$$
\mathbf{P}^+\leftarrow\frac{1}{2}\left(\mathbf{P}^++\left(\mathbf{P}^+\right)^{\top}\right)
$$

误差注入：

$$
\mathbf{p}\leftarrow\mathbf{p}+\delta\hat{\mathbf{p}},
\qquad
\mathbf{v}\leftarrow\mathbf{v}+\delta\hat{\mathbf{v}}
$$

$$
q\leftarrow\text{normalize}\left(q\otimes\text{Exp}\left(\delta\hat{\boldsymbol{\theta}}\right)\right)
$$

$$
\mathbf{b}_g\leftarrow\mathbf{b}_g+\delta\hat{\mathbf{b}}_g,
\qquad
\mathbf{b}_a\leftarrow\mathbf{b}_a+\delta\hat{\mathbf{b}}_a
$$

清零误差状态后，可选一阶重置雅可比（二阶小量，工程上常忽略）：

$$
\mathbf{G}=\text{diag}\left(\mathbf{I}_3,\ \mathbf{I}_3,\ \mathbf{I}_3-\left[\delta\hat{\boldsymbol{\theta}}\right]_{\times},\ \mathbf{I}_3,\ \mathbf{I}_3\right),
\qquad
\mathbf{P}^+\leftarrow\mathbf{G}\,\mathbf{P}^+\,\mathbf{G}^{\top}
$$

异常观测剔除（马氏距离门限）：

$$
d^{2}=\mathbf{r}^{\top}\mathbf{S}^{-1}\mathbf{r}
$$

95% 置信度下，1 维观测的门限为 $3.84$、2 维观测的门限为 $5.99$：超过门限即判为异常观测，剔除该次更新或将 $\mathbf{R}$ 放大。

## 2.8　初始化

静止时取

$$
\mathbf{p}_0=\mathbf{0},
\qquad
\mathbf{v}_0=\mathbf{0}
$$

零偏取标定值或零。姿态初值由静止时的比力求 roll/pitch，yaw 取零或由首段转向估计：

$$
\phi_0=\text{atan2}\left(a_y,\ a_z\right)
$$

$$
\theta_0=\text{atan2}\left(-a_x,\ \sqrt{a_y^{2}+a_z^{2}}\right)
$$

初始协方差：

$$
\mathbf{P}_0=\text{diag}\left(\sigma_p^{2}\mathbf{I}_3,\ \sigma_v^{2}\mathbf{I}_3,\ \sigma_\theta^{2}\mathbf{I}_3,\ \sigma_{bg}^{2}\mathbf{I}_3,\ \sigma_{ba}^{2}\mathbf{I}_3\right)
$$

典型取值：$\sigma_p=10^{-2}\,\text{m}$，$\sigma_v=10^{-1}\,\text{m/s}$，$\sigma_\theta=10^{-2}\,\text{rad}$，$\sigma_{bg}=10^{-2}\,\text{rad/s}$，$\sigma_{ba}=10^{-1}\,\text{m/s^{2}}$。

## 2.9　噪声换算与打滑判定

由传感器手册的噪声密度 $\text{ND}$（单位 $\text{rad/s}/\sqrt{\text{Hz}}$ 或 $\text{m/s}^{2}/\sqrt{\text{Hz}}$）与零偏随机游走 $\text{RW}$ 换算为单步离散标准差：

$$
\sigma_g=\frac{\text{ND}_g}{\sqrt{\Delta t}},
\qquad
\sigma_a=\frac{\text{ND}_a}{\sqrt{\Delta t}}
$$

$$
\sigma_{bg}=\text{RW}_g\sqrt{\Delta t},
\qquad
\sigma_{ba}=\text{RW}_a\sqrt{\Delta t}
$$

调参的本质是调 $\mathbf{R}$ 与 $\mathbf{Q}$ 的比值：比值越小越信任观测，跟随快但噪声大；越大越平滑但滞后。轮速可信（平路直行）时取较小 $\mathbf{R}$；检测到打滑、悬空、急转时把 $\mathbf{R}$ 放大 $10\sim100$ 倍。

打滑判据（可用电机电流或加速度一致性辅助判定）：

$$
\left|v_{\text{wheel}}-\mathbf{e}_1^{\top}R^{\top}\mathbf{v}\right|>\varepsilon
$$

$$
\left|\frac{\Delta v_{\text{wheel}}}{\Delta t}-\mathbf{e}_1^{\top}R^{\top}\mathbf{a}_W\right|>\varepsilon_a
$$

## 2.10　平衡底盘的杠杆臂修正

轮速对应的是**车轴**速度而非质心速度。设车轴相对质心在机体系内的臂长为 $\mathbf{r}_a$，则

$$
\mathbf{v}_{\text{axle}}^{W}=\mathbf{v}+R\left[\boldsymbol{\omega}\right]_{\times}\mathbf{r}_a,
\qquad
\boldsymbol{\omega}=\boldsymbol{\omega}_m-\mathbf{b}_g
$$

两侧轮速观测应使用车轴速度的机体系投影：

$$
\mathbf{e}_1^{\top}R^{\top}\mathbf{v}_{\text{axle}}^{W}
=\mathbf{e}_1^{\top}R^{\top}\mathbf{v}+\mathbf{e}_1^{\top}\left[\boldsymbol{\omega}\right]_{\times}\mathbf{r}_a
$$

于是

$$
h_L=\mathbf{e}_1^{\top}R^{\top}\mathbf{v}+\mathbf{e}_1^{\top}\left[\boldsymbol{\omega}_m-\mathbf{b}_g\right]_{\times}\mathbf{r}_a-\frac{b}{2}\mathbf{e}_3^{\top}\left(\boldsymbol{\omega}_m-\mathbf{b}_g\right)
$$

$$
\mathbf{H}_L=
\begin{bmatrix}
\mathbf{0}\vert\mathbf{e}_1^{\top}R^{\top}\vert\left(\mathbf{e}_1\times\mathbf{w}\right)^{\top}\vert+\frac{b}{2}\mathbf{e}_3^{\top}+\mathbf{e}_1^{\top}\left[\mathbf{r}_a\right]_{\times}\vert\mathbf{0}
\end{bmatrix}
$$

$h_R$ 与 $\mathbf{H}_R$ 同理，$\frac{b}{2}$ 项取负。

**量级说明**：轴心低于质心 $0.1\sim0.2\,\text{m}$、俯仰角速度 $1\sim2\,\text{rad/s}$ 时，忽略该项会带来 $0.2\sim0.4\,\text{m/s}$ 的速度误差，会明显污染估计结果。

## 2.11　输出与场地系变换

平面里程计输出为

$$
\left(x,y\right)=\left(p_x,p_y\right)
$$

欧拉角（ZYX 顺序）：

$$
\psi=\text{atan2}\left(2\left(q_wq_z+q_xq_y\right),\ 1-2\left(q_y^{2}+q_z^{2}\right)\right)
$$

$$
\theta=\arcsin\left(2\left(q_wq_y-q_zq_x\right)\right)
$$

$$
\phi=\text{atan2}\left(2\left(q_wq_x+q_yq_z\right),\ 1-2\left(q_x^{2}+q_y^{2}\right)\right)
$$

映射到场地坐标系：

$$
\mathbf{p}^{\text{field}}=R_z\left(\psi_{\text{off}}\right)
\begin{bmatrix}p_x\\p_y\end{bmatrix}
+\mathbf{t}_{\text{off}}
$$

## 2.12　每个控制周期的执行顺序

1. 读取 IMU，做安装矩阵变换与零偏补偿。
2. 预测：按 2.4 节做机械编排，按 2.5 节传播 $\mathbf{P}^-$。
3. 读取编码器与电机转速，计算 $v_L$、$v_R$，做打滑与量化一致性检查。
4. 构造观测：轮速块观测 → 侧向约束 → 垂向约束 → 门控的 ZUPT / ZARU / 重力方向观测。
5. 更新：计算增益与新息，注入误差，清零误差状态，做协方差对称化与四元数归一化。
6. 输出 $(x,y,\psi)$ 与车体速度，供底盘控制或导航使用。

## 2.13　可选：在线标定的状态增广

把待标定参数并入状态，各自增加随机游走噪声，即可在线辨识。常用的灵敏度关系为

$$
\frac{\partial v_i}{\partial r_i}=\frac{v_i}{r_i},
\qquad
\frac{\partial\omega_z^{w}}{\partial b}=-\frac{v_R-v_L}{b^{2}},
\qquad
\frac{\partial v_i}{\partial k_i}=\frac{v_i}{k_i}
$$

其中 $r_i$ 为轮半径，$b$ 为轮距，$k_i$ 为轮速标度因数（补偿轮径磨损与减速比误差）。陀螺轴间失准可增广一个 $3\times3$ 矩阵 $M$：

$$
\boldsymbol{\omega}_{\text{true}}=M\left(\boldsymbol{\omega}_m-\mathbf{b}_g\right)
$$

## 附：工程要点小结

1. 4 轮贴地、俯仰被机械约束的差速底盘，可退化为 9 维平面 ESKF，仅保留 $\delta p_{2},\delta v_{2},\delta\theta_z,\delta b_{gz},\delta b_{az}$，删掉对应行列即得特例。
2. 15 维 ESKF 在 1 kHz 下用 STM32F4/H7 单精度浮点完全可行，关键是 $\mathbf{P}$ 对称化、四元数归一化与角度归一化，否则长时间运行必然漂移。
3. 若只需要简单里程计，可只保留轮速观测与侧向约束，其余观测按需启用；效果已远好于直接对轮速积分。
4. 所有公式中的噪声参数都应实测标定：静止时记录陀螺与加计输出估计 $\sigma_g$、$\sigma_a$，匀速直行时比较轮速与 IMU 推断速度估计 $\mathbf{R}$，转向时估计陀螺零偏与轮距。
