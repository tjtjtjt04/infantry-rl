"""BMI088 IMU 传感器噪声配置：在已有观测上叠加随机噪声（不增加观测维度）。

噪声大小来自用户提供的 BMI088 实测标定值：

=======  ==========================  ===========================
轴        加速度计 σ (m/s²)           陀螺仪 σ (rad/s)
=======  ==========================  ===========================
X        0.031145                    0.009233
Y        0.034774                    0.005299
Z        0.041130                    0.005555
=======  ==========================  ===========================

如何接入观测（Isaac Lab 的 ``ObservationTermCfg.noise``）：

- ``projected_gravity`` 来自加速度计：观测值是机体系重力单位向量（单位 g），
  所以逐轴噪声 σ = 加速度 σ / 9.81（m/s² → g）。
- ``base_ang_vel`` 来自陀螺仪：观测的原始量就是角速度（rad/s），
  噪声 σ 直接取陀螺仪三轴 σ（**在 scale 之前**施加，因此 ``*0.25`` 的缩放对噪声同样生效，
  这与实机把陀螺仪读数乘 0.25 再喂策略一致）。
- 其它观测没有对应的传感器噪声模型（关节编码器、轮速、上一动作、状态估计线速度），保持干净。

注意：只有观测组的 ``enable_corruption=True`` 时 Isaac Lab 才会施加噪声
（见 ``stand_up_env_cfg.ObservationsCfg.PolicyCfg.__post_init__``）；
``ChassisSquatBalanceEnvCfg_PLAY`` 会把它关掉，便于干净回放。
"""

from __future__ import annotations

import torch

from isaaclab.utils import noise

# --------------------------------------------------------------------------- #
# 用户提供的 BMI088 噪声标定值（1σ）
# --------------------------------------------------------------------------- #
BMI088_ACC_NOISE_X: float = 0.031145  # m/s²
BMI088_ACC_NOISE_Y: float = 0.034774  # m/s²
BMI088_ACC_NOISE_Z: float = 0.041130  # m/s²

BMI088_GYRO_NOISE_X: float = 0.009233  # rad/s
BMI088_GYRO_NOISE_Y: float = 0.005299  # rad/s
BMI088_GYRO_NOISE_Z: float = 0.005555  # rad/s

#: 标准重力加速度，用于把加速度计噪声换算成 projected_gravity 的单位（g）
GRAVITY_MPS2: float = 9.81


def make_acc_noise_cfg(gravity: float = GRAVITY_MPS2) -> "noise.GaussianNoiseCfg":
    """加速度计噪声 → ``projected_gravity`` 观测的逐轴高斯噪声 (3,)。

    σ_g = [0.031145, 0.034774, 0.041130] / 9.81 ≈ [0.00317, 0.00355, 0.00419] g
    """
    std = torch.tensor(
        [BMI088_ACC_NOISE_X, BMI088_ACC_NOISE_Y, BMI088_ACC_NOISE_Z], dtype=torch.float32
    ) / float(gravity)
    return noise.GaussianNoiseCfg(mean=0.0, std=std, operation="add")


def make_gyro_noise_cfg() -> "noise.GaussianNoiseCfg":
    """陀螺仪噪声 → ``base_ang_vel`` 观测的逐轴高斯噪声 (3,)，单位 rad/s。"""
    std = torch.tensor(
        [BMI088_GYRO_NOISE_X, BMI088_GYRO_NOISE_Y, BMI088_GYRO_NOISE_Z], dtype=torch.float32
    )
    return noise.GaussianNoiseCfg(mean=0.0, std=std, operation="add")


#: 供 env cfg 直接复用的配置实例（首次调用时噪声函数会把 std 搬到数据所在 device）
ACC_NOISE_CFG = make_acc_noise_cfg()
GYRO_NOISE_CFG = make_gyro_noise_cfg()

__all__ = [
    "BMI088_ACC_NOISE_X",
    "BMI088_ACC_NOISE_Y",
    "BMI088_ACC_NOISE_Z",
    "BMI088_GYRO_NOISE_X",
    "BMI088_GYRO_NOISE_Y",
    "BMI088_GYRO_NOISE_Z",
    "GRAVITY_MPS2",
    "ACC_NOISE_CFG",
    "GYRO_NOISE_CFG",
    "make_acc_noise_cfg",
    "make_gyro_noise_cfg",
]
