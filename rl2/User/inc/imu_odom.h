#ifndef IMU_ODOM_H
#define IMU_ODOM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * imu_odom —— 轮式里程计（实现参考：香港大学轮腿开源 Chassis_Task.c）
 * ==========================================================================
 * 数据流（与参考实现一致）：
 *   1) 编码器 ecd -> 机械角 -> 圈数展开 -> 轮子累计距离
 *        参考：distance = (position/2π + turns) * 轮周长 - offset
 *        本实现等价地用累计计数表示：distance = ecd_total * 2πr/(8192·i)
 *   2) speed_rpm -> 轮子线速度：speed = rpm * 2πr/(60·i)，右轮(M2)取反号
 *      （两台 3508 出轴反向安装，见下面 ODOM_*_WHEEL_RPM_SIGN 的说明）
 *   3) foot_distance = (L+R)/2，foot_speed = (L+R)/2
 *   4) 二阶卡尔曼：状态 x = [距离, 速度]，输入 a = 车体前向加速度，
 *      量测 z = [foot_distance, foot_speed]，输出 distance / speed
 *   5) 前向加速度由 IMU 比力用 roll/pitch 投影到水平面得到（pitch 倾斜补偿）：
 *        a_f = cosθ·ax + sinθ·sinφ·ay + sinθ·cosφ·az
 *
 * 注意：参考实现的 pitch 补偿用于"腿角"（leg_angle -= pitch，供 VMC 用），
 *       与里程无关；本模块里 pitch 只用在第 5 步的加速度投影上。
 *
 * 调用示例（1 kHz，已在 CHASSIS_TASK 中）：
 *   imu_odom_update(&ODOM, EKF.accel, EKF.gyro, EKF.roll, EKF.pitch,
 *                   DJI_M2_MEASURE.speed_rpm,   // 右轮 0x202
 *                   DJI_M1_MEASURE.speed_rpm,   // 左轮 0x201
 *                   DJI_M2_MEASURE.ecd,
 *                   DJI_M1_MEASURE.ecd, dt_s);
 * ========================================================================== */

/* ==================== 轮子标定（换车 / 换轮只改这里） ==================== */
#define ODOM_WHEEL_RADIUS_M       (0.068f)      /* 轮半径 68 mm —— 必须与训练一致
                                                 * （robot_model.py: WHEEL_RADIUS = 0.068，
                                                 *  由左右轮 STL 网格顶点实测；曾误写 0.070） */
#define ODOM_GEAR_RATIO           (19.203208f)  /* DJI 3508 减速比 3591/187 */
#define ODOM_RIGHT_WHEEL_RPM_SIGN (-1.0f)       /* M2(0x202) 右轮：+rpm = 后退 */
#define ODOM_LEFT_WHEEL_RPM_SIGN  (+1.0f)       /* M1(0x201) 左轮：+rpm = 前进 */

/* 为什么左右符号相反（不要再"修正"成同号）：
 *   两台 3508 出轴反向安装，所以同一个电流/同一个 rpm 符号在两个轮子上对应
 *   相反的物理方向。标定依据是 lqr.h 的 LQR_T_DIRECTION_LEFT = +1 /
 *   LQR_T_DIRECTION_RIGHT = -1 —— 同一个推力需要左右相反的电流。
 *   这里统一成"输出前进为正"，于是左轮 +1、右轮 -1。
 *   换电机/换线序只改这两个宏，不要改公式。 */

/* 里程标定系数：跑一段已知距离后填 实际/显示；1.0 = 不修正 */
#define ODOM_WHEEL_CAL_SCALE      (1.0f)
#define ODOM_WHEEL_EFF_RADIUS_M   (ODOM_WHEEL_RADIUS_M * ODOM_WHEEL_CAL_SCALE)

#define ODOM_WHEEL_PERIMETER_M    (6.283185307179586f * ODOM_WHEEL_EFF_RADIUS_M)

/* DJI 3508/C620：机械角 0~8191，即每转 8192 计数（转子侧） */
#define ODOM_ECD_PER_REV          (8192.0f)

/* 每编码器计数对应的车轮位移：2πr / (8192·i) ≈ 2.8 µm */
#define ODOM_M_PER_ECD_COUNT \
    (ODOM_WHEEL_PERIMETER_M / (ODOM_ECD_PER_REV * ODOM_GEAR_RATIO))

/* 电机轴 rpm -> 车轮线速度：2πr·n / (60·i) */
#define ODOM_RPM_TO_MPS \
    (ODOM_WHEEL_PERIMETER_M / (60.0f * ODOM_GEAR_RATIO))

/* ==================== 二阶卡尔曼参数 ==================== */
#define ODOM_KF_SIGMA_ACCEL   (1.0f)     /* 前向加速度过程噪声 (m/s²) */
#define ODOM_KF_SIGMA_DIST    (5.0e-3f)  /* 距离量测噪声 (m)，打滑严重时调大 */
#define ODOM_KF_SIGMA_SPEED   (0.05f)    /* 速度量测噪声 (m/s) */

/* ==================== 其他 ==================== */
#define ODOM_DT_MIN_S  (1.0e-4f)  /* 允许的最小采样间隔 (s) */
#define ODOM_DT_MAX_S  (5.0e-2f)  /* 允许的最大采样间隔 (s) */

/** 里程计句柄 */
typedef struct
{
    /* ---------- 对外输出 ---------- */
    float distance;     /**< 累计里程 (m)，二阶 KF 输出（有符号，前进为正） */
    float speed;        /**< 车体速度 (m/s)，二阶 KF 输出 */
    float mileage;      /**< 累计路程 (m)，单调不减（取轮子距离增量的绝对值） */

    /* ---------- 轮子原始量（调试用） ---------- */
    float wheel_distance_left;   /**< 左轮累计距离 (m)，前进为正 */
    float wheel_distance_right;  /**< 右轮累计距离 (m)，前进为正 */
    float wheel_speed_left;      /**< 左轮线速度 (m/s)，前进为正 */
    float wheel_speed_right;     /**< 右轮线速度 (m/s)，前进为正 */
    float foot_distance;         /**< (左+右)/2 (m) */
    float foot_speed;            /**< (左+右)/2 (m/s) */

    /* ---------- 编码器累计 ---------- */
    int32_t ecd_total_left;      /**< 左轮累计计数（已做 13 位回绕展开） */
    int32_t ecd_total_right;     /**< 右轮累计计数 */
    uint16_t last_ecd_left;      /**< 上一次左轮编码器值 */
    uint16_t last_ecd_right;     /**< 上一次右轮编码器值 */
    float last_foot_distance;    /**< 上一次的 foot_distance（算路程用） */
    uint8_t ecd_ready;           /**< 编码器基准已建立 */

    /* ---------- 二阶卡尔曼 ---------- */
    float kf_x[2];               /**< 状态 [距离, 速度] */
    float kf_P[2][2];            /**< 协方差 */

    uint8_t ready;               /**< 初始化完成标志 */
} imu_odom_t;

/**
 * @brief 初始化里程计：轮子量、编码器累计、二阶卡尔曼全部清零。
 * @param odom 待初始化句柄（为空时直接返回）。
 */
void imu_odom_init(imu_odom_t *odom);

/**
 * @brief 复位里程：输出距离/速度/路程清零，并重新建立编码器基准；标定参数不变。
 * @param odom 已初始化的句柄。
 */
void imu_odom_reset(imu_odom_t *odom);

/**
 * @brief 单步更新里程计（建议 1 kHz，在固定周期任务里调用）。
 *
 * @param odom           已初始化的句柄。
 * @param accel_body     车体系加速度，含重力（m/s²），即 EKF.accel。
 * @param gyro_body      车体系角速度（rad/s），即 EKF.gyro（本实现未用，保留接口）。
 * @param roll_deg       姿态 EKF 的 roll（deg），即 EKF.roll。
 * @param pitch_deg      姿态 EKF 的 pitch（deg），即 EKF.pitch（pitch 倾斜补偿用）。
 * @param wheel_right_rpm 右轮原始转速（DJI_M2_MEASURE.speed_rpm，0x202）。
 * @param wheel_left_rpm  左轮原始转速（DJI_M1_MEASURE.speed_rpm，0x201）。
 * @param wheel_right_ecd 右轮编码器机械角（DJI_M2_MEASURE.ecd，0x202，0~8191）。
 * @param wheel_left_ecd  左轮编码器机械角（DJI_M1_MEASURE.ecd，0x201，0~8191）。
 * @param dt_s           本次采样间隔（s），建议用 DWT 实测值。
 */
void imu_odom_update(imu_odom_t *odom,
                     const float accel_body[3],
                     const float gyro_body[3],
                     float roll_deg,
                     float pitch_deg,
                     int16_t wheel_right_rpm,
                     int16_t wheel_left_rpm,
                     uint16_t wheel_right_ecd,
                     uint16_t wheel_left_ecd,
                     float dt_s);

/** 全局里程计句柄（单实例；未初始化前 ready = 0，update 直接返回） */
extern imu_odom_t ODOM;

#ifdef __cplusplus
}
#endif

#endif /* IMU_ODOM_H */
