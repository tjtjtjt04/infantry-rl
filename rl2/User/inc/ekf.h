/**
 * @file    ekf.h
 * @brief   姿态 EKF 模块：输入 BMI088 数据，输出欧拉角
 *
 * 对外只暴露 2 个接口，其余全部是本模块内部的 static 实现：
 *   ekf_init()          初始化滤波句柄
 *   ekf_update()        传入 BMI088 数据，内部拷贝六轴后计算，返回欧拉角
 *
 * 典型用法：
 *   static ekf_t g_ekf;
 *
 *   ekf_init(&g_ekf);
 *
 *   // 1 kHz 任务中
 *   BMI088_read(&BMI088);
 *   ekf_euler_t euler = ekf_update(&g_ekf, &BMI088);
 *   // euler.roll / euler.pitch / euler.yaw（单位：deg）
 *
 * 说明：
 *   - ekf_update() 只把传入 BMI088_t 的六轴数据（accel[3] / gyro[3]）拷贝到句柄内部，
 *     之后的 dt 计算、滤波更新全部基于这份副本，调用者可立即复用原结构体。
 *   - 采样间隔 dt 限制在 0.2 ms ~ 20 ms。越界时本次不更新滤波状态，
 *     返回上一次的欧拉角，具体原因记录在 g_ekf.status 中。
 *   - yaw 由陀螺仪积分得到，没有绝对参考，不会自动收敛。
 *
 * 依赖：BMI088driver.h（BMI088_t）、arm_math.h（CMSIS-DSP 矩阵运算）
 */

#ifndef EKF_H
#define EKF_H

#include <stdint.h>
#include <math.h>
#include <string.h>

#include "BMI088driver.h"
#include "arm_math.h"
#include "bsp_dwt.h"
#include "common_function.h"
#include "main.h"

#ifdef __cplusplus
extern "C" {
#endif

/** 滤波状态维数：四元数 q0..q3 + 陀螺仪 x/y 轴零偏（内部常量） */
#define EKF_STATE_DIM 6U
/** 量测维数：归一化加速度（内部常量） */
#define EKF_MEAS_DIM 3U

/** 滤波/应用层返回状态，最近一次结果记录在 ekf_t.status 中 */
typedef enum {
    EKF_OK = 0,             /**< 预测 + 校正均完成 */
    EKF_PREDICT_ONLY = 1,   /**< 仅预测（加速度不可用，如自由落体） */
    EKF_REJECTED = 2,       /**< 量测被门限拒绝，仅预测 */
    EKF_NUMERIC_ERROR = 3,  /**< 矩阵运算/数值异常 */
    EKF_ERROR = -1,         /**< 应用层错误：空指针或未初始化 */
    EKF_ERROR_DT = -2       /**< 采样间隔越界 */
} ekf_status_t;

/** ekf_update() 的返回值：欧拉角，单位 deg（度） */
typedef struct {
    float roll;   /**< 横滚角（deg） */
    float pitch;  /**< 俯仰角（deg） */
    float yaw;    /**< 偏航角（deg，陀螺仪积分，无绝对参考） */
} ekf_euler_t;

/** 滤波参数（内部使用，初始化时自动载入默认值） */
typedef struct {
    float32_t q_gyro;                     /**< 四元数过程噪声 */
    float32_t q_bias;                     /**< 零偏过程噪声 */
    float32_t r_accel;                    /**< 加速度量测噪声 */
    float32_t initial_attitude_variance;  /**< 姿态初值方差 */
    float32_t initial_bias_variance;      /**< 零偏初值方差 */
    float32_t bias_fading_lambda;         /**< 零偏渐消因子（0 或 1 表示关闭） */
    float32_t chi_square_gate;            /**< 卡方拒绝门限，<=0 关闭 */
    float32_t adaptive_gate;              /**< 自适应增益门限，<=0 关闭 */
    float32_t adaptive_gain_min;          /**< 自适应增益下限 */
    float32_t max_bias_correction;        /**< 单步零偏修正上限 */
    float32_t accel_norm_min;             /**< 加速度模长下限 */
    float32_t accel_norm_max;             /**< 加速度模长上限 */
    float32_t gyro_static_limit;          /**< 强制更新时允许的陀螺模长 */
    uint16_t force_update_after_rejects;  /**< 连续拒绝次数上限，达到后强制更新 */
} ekf_config_t;

/** EKF 句柄：滤波状态 + 输入数据副本 + 输出缓存 */
typedef struct {
    /* ---------- 滤波状态 ---------- */
    float32_t x[EKF_STATE_DIM];                             /**< [q0,q1,q2,q3,bx,by] */
    float32_t P[EKF_STATE_DIM * EKF_STATE_DIM];             /**< 状态协方差 */
    float32_t Q[EKF_STATE_DIM * EKF_STATE_DIM];             /**< 过程噪声 */
    float32_t R[EKF_MEAS_DIM * EKF_MEAS_DIM];               /**< 量测噪声 */
    ekf_config_t cfg;                                       /**< 参数副本 */
    uint16_t reject_count;                                  /**< 连续拒绝计数 */
    float32_t last_chi_square;                              /**< 最近一次卡方值 */
    float32_t last_accel_norm;                              /**< 最近一次加速度模长 */
    float32_t last_gain_scale;                              /**< 最近一次增益缩放 */

    /* ---------- 预分配工作区（避免运行期大块栈分配） ---------- */
    float32_t F[EKF_STATE_DIM * EKF_STATE_DIM];
    float32_t Ft[EKF_STATE_DIM * EKF_STATE_DIM];
    float32_t H[EKF_MEAS_DIM * EKF_STATE_DIM];
    float32_t Ht[EKF_STATE_DIM * EKF_MEAS_DIM];
    float32_t S[EKF_MEAS_DIM * EKF_MEAS_DIM];
    float32_t S_inv[EKF_MEAS_DIM * EKF_MEAS_DIM];
    float32_t HP[EKF_MEAS_DIM * EKF_STATE_DIM];
    float32_t PHt[EKF_STATE_DIM * EKF_MEAS_DIM];
    float32_t K[EKF_STATE_DIM * EKF_MEAS_DIM];
    float32_t KH[EKF_STATE_DIM * EKF_STATE_DIM];
    float32_t I_KH[EKF_STATE_DIM * EKF_STATE_DIM];
    float32_t tmp66_a[EKF_STATE_DIM * EKF_STATE_DIM];
    float32_t tmp66_b[EKF_STATE_DIM * EKF_STATE_DIM];

    /* ---------- 输入数据副本（仅六轴） ---------- */
    float32_t accel[3];           /**< 加速度副本（m/s²），ekf_update() 写入 */
    float32_t gyro[3];            /**< 角速度副本（rad/s），ekf_update() 写入 */

    /* ---------- 应用层状态 ---------- */
    uint32_t last_update_us;      /**< 上一次更新时间戳 */
    float32_t gyro_bias_z;        /**< 静止学习的 yaw 轴陀螺零偏（rad/s） */
    float roll;                   /**< 最近一次输出的 roll（deg） */
    float pitch;                  /**< 最近一次输出的 pitch（deg） */
    float yaw;                    /**< 最近一次输出的 yaw（deg） */
    float dt_s;                   /**< 最近一次采样间隔（s） */
    ekf_status_t status;          /**< 最近一次更新的状态 */
    uint8_t ready;                /**< 初始化完成标志 */
} ekf_t;

/**
 * @brief 初始化 EKF 并载入默认参数。
 *
 * 时间基准由模块内部通过 DWT_GetTimeline_us() 获取，
 * 调用前需要保证 DWT_Init() 已执行（本工程在 BMI088_init() 中完成）。
 *
 * @param ekf    待初始化句柄。
 */
void ekf_init(ekf_t *ekf);

/**
 * @brief 传入一帧 BMI088 数据，单步更新滤波，返回欧拉角。
 *
 * 内部只把传入 BMI088_t 的 accel[3] / gyro[3] 六轴数据拷贝到句柄，再基于副本完成
 * dt 计算与滤波更新，因此调用者返回后可立即修改或复用传入的 BMI088 结构体。
 * dt 由模块内部 DWT 计时计算，调用周期需保持在 0.2ms~20ms（50Hz~5kHz）。
 *
 * @param ekf    已初始化的句柄。
 * @param imu    本帧 BMI088 数据（使用其中的 gyro 与 accel）。
 * @return 本帧计算后的欧拉角（deg）。失败时返回上一次的有效欧拉角，
 *         失败原因见 ekf->status。
 */
ekf_euler_t ekf_update(ekf_t *ekf, const BMI088_t *imu);

extern ekf_t EKF; 
#ifdef __cplusplus
}
#endif

#endif /* EKF_H */
