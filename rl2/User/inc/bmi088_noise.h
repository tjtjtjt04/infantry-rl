#ifndef BMI088_NOISE_H
#define BMI088_NOISE_H

/* ==========================================================================
 * BMI088 六面标定实测噪声（2026-10-25，静止采样统计）
 * ==========================================================================
 * 单位：加速度计 m/s²，陀螺仪 rad/s；数值为各轴静态输出的标准差 σ。
 *
 * 这是滤波器里"已知噪声"的唯一来源：
 *   - BMI088driver.c 用这些宏导出 BMI088_ACC_NOISE[] / BMI088_GYRO_NOISE[]
 *     数组，方便调试查看或其它模块取用；
 *   - imu_odom.h 用它们直接推导过程噪声 Q（ODOM_Q_YAW_RATE / ODOM_Q_ACCEL），
 *     避免两处各写一份数字、改一处忘另一处。
 *
 * 注意：静态 σ 里已含一部分环境振动；实车电机振动更大，里程计里另外乘
 *       振动余量 ODOM_IMU_VIBRATION_MARGIN（见 imu_odom.h）。
 * ========================================================================== */

/* 加速度计三轴噪声 σ (m/s²) */
#define BMI088_ACC_NOISE_X    (0.031145f)
#define BMI088_ACC_NOISE_Y    (0.034774f)
#define BMI088_ACC_NOISE_Z    (0.041130f)

/* 陀螺仪三轴噪声 σ (rad/s) */
#define BMI088_GYRO_NOISE_X   (0.009233f)
#define BMI088_GYRO_NOISE_Y   (0.005299f)
#define BMI088_GYRO_NOISE_Z   (0.005555f)

#endif /* BMI088_NOISE_H */
