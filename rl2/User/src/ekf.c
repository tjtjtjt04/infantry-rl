/**
 * @file    ekf.c
 * @brief   姿态 EKF 模块实现（输入 BMI088 数据，输出欧拉角）
 *
 * 文件分为四段：
 *   1. 基础数学工具（全部 static）；
 *   2. 姿态 EKF 核心：预测、量测模型、校正（全部 static）；
 *   3. 对外接口：初始化、单步更新（仅此两个非 static）；
 *   4. 输入拷贝与欧拉角输出。
 */

#include "ekf.h"

#ifndef EKF_EPS
#define EKF_EPS 1.0e-9f
#endif

/* ================= IMU 安装方向修正 =================
 * 板载 BMI088 的安装方向与车体坐标不一致，滤波前先用一个 3x3 旋转矩阵把
 * 六轴数据从传感器坐标系变换到车体坐标系（X=roll, Y=pitch, Z=yaw）：
 *
 *     body = R · sensor
 *
 * 矩阵约定：行 = 车体轴（第 0/1/2 行分别是 X/Y/Z），列 = 传感器轴；
 * 即 R[i][j] 是车体 i 轴在传感器 j 轴上的投影（方向余弦）。
 *
 * 当前安装方向（与原有逐轴映射完全等价，行为不变）：
 *     车体 X = +传感器 Y
 *     车体 Y = -传感器 X
 *     车体 Z = +传感器 Z
 *
 * 修改方法：只改下面矩阵本体即可，不需要动其他代码。要求 R 必须是正交
 * 矩阵且 det(R) = +1（纯旋转，不含镜像）：EKF 的量测模型按“车体 Z 轴
 * 向上、右手系”推重力方向，R 若不是纯旋转，姿态解算会整体出错。
 * 若不需要任何映射，把矩阵改成单位阵 [[1,0,0],[0,1,0],[0,0,1]]。
 */
static const float32_t IMU_SENSOR_TO_BODY[3][3] =
{
    { 0.0f,  1.0f,  0.0f },   /* 车体 X = +传感器 Y */
    {1.0f,  0.0f,  0.0f },   /* 车体 Y = -传感器 X */
    { 0.0f,  0.0f,  -1.0f }    /* 车体 Z = +传感器 Z */
};

static void ekf_map_imu_axes(const float32_t src[3], float32_t dst[3])
{
    float32_t mapped[3];
    uint32_t row;
    uint32_t col;

    /* 先算到临时量再写回，保证 dst 与 src 指向同一块内存时也正确 */
    for (row = 0U; row < 3U; ++row)
    {
        mapped[row] = 0.0f;
        for (col = 0U; col < 3U; ++col)
        {
            mapped[row] += IMU_SENSOR_TO_BODY[row][col] * src[col];
        }
    }

    dst[0] = mapped[0];
    dst[1] = mapped[1];
    dst[2] = mapped[2];
}

ekf_t EKF;

/* ======================================================================
 * 1. 基础数学工具（内部）
 * ====================================================================== */

/* ---------- 矩阵运算使用 CMSIS-DSP（arm_math.h + libarm_cortexM7lfdp_math.a） ---------- */

static void ekf_zero(float32_t *data, uint32_t count)
{
    memset(data, 0, sizeof(float32_t) * count);
}

static void ekf_set_identity(float32_t *data, uint32_t n)
{
    ekf_zero(data, n * n);
    for (uint32_t i = 0U; i < n; ++i) {
        data[i * n + i] = 1.0f;
    }
}

static void ekf_normalize_quaternion(float32_t q[4])
{
    const float32_t norm2 = SQUARE(q[0]) + SQUARE(q[1])
                          + SQUARE(q[2]) + SQUARE(q[3]);
    if (norm2 <= EKF_EPS) {
        q[0] = 1.0f;
        q[1] = 0.0f;
        q[2] = 0.0f;
        q[3] = 0.0f;
        return;
    }

    const float32_t inv_norm = 1.0f / sqrtf(norm2);
    q[0] *= inv_norm;
    q[1] *= inv_norm;
    q[2] *= inv_norm;
    q[3] *= inv_norm;
}

static void ekf_symmetrize_covariance(float32_t *P)
{
    for (uint32_t r = 0U; r < EKF_STATE_DIM; ++r) {
        for (uint32_t c = r + 1U; c < EKF_STATE_DIM; ++c) {
            const float32_t v = 0.5f * (P[r * EKF_STATE_DIM + c]
                                      + P[c * EKF_STATE_DIM + r]);
            P[r * EKF_STATE_DIM + c] = v;
            P[c * EKF_STATE_DIM + r] = v;
        }
    }

    for (uint32_t i = 0U; i < EKF_STATE_DIM; ++i) {
        if (P[i * EKF_STATE_DIM + i] < EKF_EPS) {
            P[i * EKF_STATE_DIM + i] = EKF_EPS;
        }
    }
}

/** 四元数 -> 欧拉角（ZYX 顺序，与旧接口一致；输出单位：度 deg） */
static ekf_euler_t ekf_euler_from_quaternion(const float32_t q[4])
{
    ekf_euler_t euler;
    float32_t s;

    euler.roll = RAD_TO_DEG(atan2f(2.0f * (q[0] * q[1] + q[2] * q[3]),
                                   1.0f - 2.0f * (SQUARE(q[1]) + SQUARE(q[2]))));

    s = 2.0f * (q[0] * q[2] - q[3] * q[1]);
    s = CLAMP(s, -1.0f, 1.0f);
    euler.pitch = RAD_TO_DEG(asinf(s));

    euler.yaw = RAD_TO_DEG(atan2f(2.0f * (q[0] * q[3] + q[1] * q[2]),
                                  1.0f - 2.0f * (SQUARE(q[2]) + SQUARE(q[3]))));
    return euler;
}

static void ekf_mat_vec_6x3(const float32_t *A,
                            const float32_t v[3],
                            float32_t out[EKF_STATE_DIM])
{
    for (uint32_t r = 0U; r < EKF_STATE_DIM; ++r) {
        out[r] = A[r * EKF_MEAS_DIM + 0U] * v[0]
               + A[r * EKF_MEAS_DIM + 1U] * v[1]
               + A[r * EKF_MEAS_DIM + 2U] * v[2];
    }
}

static void ekf_mat_vec_3x3(const float32_t A[EKF_MEAS_DIM * EKF_MEAS_DIM],
                            const float32_t v[3],
                            float32_t out[3])
{
    for (uint32_t r = 0U; r < EKF_MEAS_DIM; ++r) {
        out[r] = A[r * EKF_MEAS_DIM + 0U] * v[0]
               + A[r * EKF_MEAS_DIM + 1U] * v[1]
               + A[r * EKF_MEAS_DIM + 2U] * v[2];
    }
}

/* ======================================================================
 * 2. 姿态 EKF 核心（内部）
 * ====================================================================== */

/** 载入默认参数 */
static void ekf_default_config(ekf_config_t *cfg)
{
    cfg->q_gyro = 1.0e-6f;
    cfg->q_bias = 1.0e-8f;
    /* 实测 BMI088 静止加速度噪声 std ≈ 0.04~0.14 m/s²，
       归一化后 σ≈0.004~0.014，R≈2e-5~2e-4；取 5e-4 保守抑制振动 */
    cfg->r_accel = 5.0e-4f;
    cfg->initial_attitude_variance = 1.0e-2f;
    cfg->initial_bias_variance = 1.0e-3f;
    cfg->bias_fading_lambda = 0.9995f;
    cfg->chi_square_gate = 11.345f;
    cfg->adaptive_gate = 6.251f;
    cfg->adaptive_gain_min = 0.2f;
    cfg->max_bias_correction = 2.0e-4f;
    /* 实测静止时 |a| 在 9.05~11.44 m/s² 波动，限制在 0.7g~1.3g，抑制加减速/冲击 */
    cfg->accel_norm_min = 6.9f;
    cfg->accel_norm_max = 12.7f;
    cfg->gyro_static_limit = 0.08f;
    cfg->force_update_after_rejects = 50U;
}

/** 用陀螺仪角速度积分四元数（一阶龙格库塔），随后归一化 */
static void ekf_predict_quaternion(float32_t q[4],
                                   const float32_t gyro_rad_s[3],
                                   float32_t bias_x,
                                   float32_t bias_y,
                                   float32_t dt_s)
{
    const float32_t wx = gyro_rad_s[0] - bias_x;
    const float32_t wy = gyro_rad_s[1] - bias_y;
    const float32_t wz = gyro_rad_s[2];
    const float32_t half_dt = 0.5f * dt_s;
    const float32_t q0 = q[0];
    const float32_t q1 = q[1];
    const float32_t q2 = q[2];
    const float32_t q3 = q[3];

    q[0] = q0 + half_dt * (-wx * q1 - wy * q2 - wz * q3);
    q[1] = q1 + half_dt * ( wx * q0 + wz * q2 - wy * q3);
    q[2] = q2 + half_dt * ( wy * q0 - wz * q1 + wx * q3);
    q[3] = q3 + half_dt * ( wz * q0 + wy * q1 - wx * q2);
    ekf_normalize_quaternion(q);
}

/** 构造状态转移矩阵 F */
static void ekf_build_F(ekf_t *ekf,
                        const float32_t gyro_rad_s[3],
                        float32_t dt_s)
{
    const float32_t *x = ekf->x;
    const float32_t q0 = x[0];
    const float32_t q1 = x[1];
    const float32_t q2 = x[2];
    const float32_t q3 = x[3];
    const float32_t wx = gyro_rad_s[0] - x[4];
    const float32_t wy = gyro_rad_s[1] - x[5];
    const float32_t wz = gyro_rad_s[2];
    const float32_t h = 0.5f * dt_s;
    float32_t *F = ekf->F;

    ekf_set_identity(F, EKF_STATE_DIM);

    F[0 * EKF_STATE_DIM + 1] = -wx * h;
    F[0 * EKF_STATE_DIM + 2] = -wy * h;
    F[0 * EKF_STATE_DIM + 3] = -wz * h;

    F[1 * EKF_STATE_DIM + 0] =  wx * h;
    F[1 * EKF_STATE_DIM + 2] =  wz * h;
    F[1 * EKF_STATE_DIM + 3] = -wy * h;

    F[2 * EKF_STATE_DIM + 0] =  wy * h;
    F[2 * EKF_STATE_DIM + 1] = -wz * h;
    F[2 * EKF_STATE_DIM + 3] =  wx * h;

    F[3 * EKF_STATE_DIM + 0] =  wz * h;
    F[3 * EKF_STATE_DIM + 1] =  wy * h;
    F[3 * EKF_STATE_DIM + 2] = -wx * h;

    F[0 * EKF_STATE_DIM + 4] =  q1 * h;
    F[1 * EKF_STATE_DIM + 4] = -q0 * h;
    F[2 * EKF_STATE_DIM + 4] = -q3 * h;
    F[3 * EKF_STATE_DIM + 4] =  q2 * h;

    F[0 * EKF_STATE_DIM + 5] =  q2 * h;
    F[1 * EKF_STATE_DIM + 5] =  q3 * h;
    F[2 * EKF_STATE_DIM + 5] = -q0 * h;
    F[3 * EKF_STATE_DIM + 5] = -q1 * h;
}

/** 构造量测矩阵 H（d h / d q） */
static void ekf_build_H(const float32_t q[4], float32_t *H)
{
    ekf_zero(H, EKF_MEAS_DIM * EKF_STATE_DIM);

    H[0 * EKF_STATE_DIM + 0] = -2.0f * q[2];
    H[0 * EKF_STATE_DIM + 1] =  2.0f * q[3];
    H[0 * EKF_STATE_DIM + 2] = -2.0f * q[0];
    H[0 * EKF_STATE_DIM + 3] =  2.0f * q[1];

    H[1 * EKF_STATE_DIM + 0] =  2.0f * q[1];
    H[1 * EKF_STATE_DIM + 1] =  2.0f * q[0];
    H[1 * EKF_STATE_DIM + 2] =  2.0f * q[3];
    H[1 * EKF_STATE_DIM + 3] =  2.0f * q[2];

    H[2 * EKF_STATE_DIM + 0] =  2.0f * q[0];
    H[2 * EKF_STATE_DIM + 1] = -2.0f * q[1];
    H[2 * EKF_STATE_DIM + 2] = -2.0f * q[2];
    H[2 * EKF_STATE_DIM + 3] =  2.0f * q[3];
}

/** 量测模型：由四元数预测重力方向单位向量 */
static void ekf_measurement_model(const float32_t q[4], float32_t h[3])
{
    h[0] = 2.0f * (q[1] * q[3] - q[0] * q[2]);
    h[1] = 2.0f * (q[2] * q[3] + q[0] * q[1]);
    h[2] = 1.0f - 2.0f * (SQUARE(q[1]) + SQUARE(q[2]));
}

/** 协方差预测：P = F * (渐消 P) * F^T + Q */
static arm_status ekf_propagate_covariance(ekf_t *ekf,
                                           const float32_t gyro_rad_s[3],
                                           float32_t dt_s)
{
    arm_matrix_instance_f32 F;
    arm_matrix_instance_f32 Ft;
    arm_matrix_instance_f32 P;
    arm_matrix_instance_f32 tmp_a;
    arm_matrix_instance_f32 tmp_b;

    ekf_build_F(ekf, gyro_rad_s, dt_s);

    arm_mat_init_f32(&F, EKF_STATE_DIM, EKF_STATE_DIM, ekf->F);
    arm_mat_init_f32(&Ft, EKF_STATE_DIM, EKF_STATE_DIM, ekf->Ft);
    arm_mat_init_f32(&P, EKF_STATE_DIM, EKF_STATE_DIM, ekf->P);
    arm_mat_init_f32(&tmp_a, EKF_STATE_DIM, EKF_STATE_DIM, ekf->tmp66_a);
    arm_mat_init_f32(&tmp_b, EKF_STATE_DIM, EKF_STATE_DIM, ekf->tmp66_b);

    arm_status st = arm_mat_trans_f32(&F, &Ft);
    if (st != ARM_MATH_SUCCESS) {
        return st;
    }

    memcpy(ekf->tmp66_b, ekf->P, sizeof(ekf->P));

    if ((ekf->cfg.bias_fading_lambda > 0.0f)
        && (ekf->cfg.bias_fading_lambda < 1.0f)) {
        const float32_t s = 1.0f / ekf->cfg.bias_fading_lambda;
        for (uint32_t i = 0U; i < EKF_STATE_DIM; ++i) {
            ekf->tmp66_b[4U * EKF_STATE_DIM + i] *= s;
            ekf->tmp66_b[5U * EKF_STATE_DIM + i] *= s;
            ekf->tmp66_b[i * EKF_STATE_DIM + 4U] *= s;
            ekf->tmp66_b[i * EKF_STATE_DIM + 5U] *= s;
        }
    }

    arm_matrix_instance_f32 faded_P;
    arm_mat_init_f32(&faded_P, EKF_STATE_DIM, EKF_STATE_DIM, ekf->tmp66_b);

    st = arm_mat_mult_f32(&F, &faded_P, &tmp_a);
    if (st != ARM_MATH_SUCCESS) {
        return st;
    }
    st = arm_mat_mult_f32(&tmp_a, &Ft, &tmp_b);
    if (st != ARM_MATH_SUCCESS) {
        return st;
    }
    for (uint32_t i = 0U; i < EKF_STATE_DIM * EKF_STATE_DIM; ++i) {
        ekf->P[i] = ekf->tmp66_b[i] + ekf->Q[i];
    }

    ekf_symmetrize_covariance(ekf->P);
    return ARM_MATH_SUCCESS;
}

/** 加速度量测校正：卡方门限、自适应增益、限幅后更新状态与协方差 */
static arm_status ekf_correct_with_accel(ekf_t *ekf,
                                         const float32_t z[3],
                                         const float32_t gyro_rad_s[3])
{
    float32_t h[3];
    float32_t e[3];
    float32_t tmp_v3[3];
    float32_t dx[EKF_STATE_DIM];
    arm_matrix_instance_f32 P;
    arm_matrix_instance_f32 H;
    arm_matrix_instance_f32 Ht;
    arm_matrix_instance_f32 HP;
    arm_matrix_instance_f32 PHt;
    arm_matrix_instance_f32 S;
    arm_matrix_instance_f32 S_inv;
    arm_matrix_instance_f32 K;
    arm_matrix_instance_f32 KH;
    arm_matrix_instance_f32 I_KH;
    arm_matrix_instance_f32 tmp66;

    ekf_measurement_model(ekf->x, h);
    e[0] = z[0] - h[0];
    e[1] = z[1] - h[1];
    e[2] = z[2] - h[2];
    ekf_build_H(ekf->x, ekf->H);

    arm_mat_init_f32(&P, EKF_STATE_DIM, EKF_STATE_DIM, ekf->P);
    arm_mat_init_f32(&H, EKF_MEAS_DIM, EKF_STATE_DIM, ekf->H);
    arm_mat_init_f32(&Ht, EKF_STATE_DIM, EKF_MEAS_DIM, ekf->Ht);
    arm_mat_init_f32(&HP, EKF_MEAS_DIM, EKF_STATE_DIM, ekf->HP);
    arm_mat_init_f32(&PHt, EKF_STATE_DIM, EKF_MEAS_DIM, ekf->PHt);
    arm_mat_init_f32(&S, EKF_MEAS_DIM, EKF_MEAS_DIM, ekf->S);
    arm_mat_init_f32(&S_inv, EKF_MEAS_DIM, EKF_MEAS_DIM, ekf->S_inv);
    arm_mat_init_f32(&K, EKF_STATE_DIM, EKF_MEAS_DIM, ekf->K);
    arm_mat_init_f32(&KH, EKF_STATE_DIM, EKF_STATE_DIM, ekf->KH);
    arm_mat_init_f32(&I_KH, EKF_STATE_DIM, EKF_STATE_DIM, ekf->I_KH);
    arm_mat_init_f32(&tmp66, EKF_STATE_DIM, EKF_STATE_DIM, ekf->tmp66_a);

    arm_status st = arm_mat_trans_f32(&H, &Ht);
    if (st != ARM_MATH_SUCCESS) {
        return st;
    }
    st = arm_mat_mult_f32(&H, &P, &HP);
    if (st != ARM_MATH_SUCCESS) {
        return st;
    }
    st = arm_mat_mult_f32(&HP, &Ht, &S);
    if (st != ARM_MATH_SUCCESS) {
        return st;
    }
    for (uint32_t i = 0U; i < EKF_MEAS_DIM * EKF_MEAS_DIM; ++i) {
        ekf->S[i] += ekf->R[i];
    }
    st = arm_mat_inverse_f32(&S, &S_inv);
    if (st != ARM_MATH_SUCCESS) {
        return st;
    }

    ekf_mat_vec_3x3(ekf->S_inv, e, tmp_v3);
    ekf->last_chi_square = e[0] * tmp_v3[0]
                         + e[1] * tmp_v3[1]
                         + e[2] * tmp_v3[2];

    const float32_t gyro_norm = sqrtf(SQUARE(gyro_rad_s[0])
                                   + SQUARE(gyro_rad_s[1])
                                   + SQUARE(gyro_rad_s[2]));
    const uint8_t force_update =
        (ekf->cfg.force_update_after_rejects > 0U)
        && (ekf->reject_count >= ekf->cfg.force_update_after_rejects)
        && (gyro_norm < ekf->cfg.gyro_static_limit);

    if ((ekf->cfg.chi_square_gate > 0.0f)
        && (ekf->last_chi_square > ekf->cfg.chi_square_gate)
        && (force_update == 0U)) {
        ekf->reject_count++;
        ekf->last_gain_scale = 0.0f;
        return ARM_MATH_ARGUMENT_ERROR;
    }

    ekf->reject_count = 0U;
    ekf->last_gain_scale = 1.0f;
    if ((ekf->cfg.adaptive_gate > 0.0f)
        && (ekf->last_chi_square > ekf->cfg.adaptive_gate)) {
        ekf->last_gain_scale = ekf->cfg.adaptive_gate / ekf->last_chi_square;
        ekf->last_gain_scale = CLAMP(ekf->last_gain_scale,
                                          ekf->cfg.adaptive_gain_min,
                                          1.0f);
    }

    st = arm_mat_mult_f32(&P, &Ht, &PHt);
    if (st != ARM_MATH_SUCCESS) {
        return st;
    }
    st = arm_mat_mult_f32(&PHt, &S_inv, &K);
    if (st != ARM_MATH_SUCCESS) {
        return st;
    }

    if (ekf->last_gain_scale < 1.0f) {
        for (uint32_t i = 0U; i < EKF_STATE_DIM * EKF_MEAS_DIM; ++i) {
            ekf->K[i] *= ekf->last_gain_scale;
        }
    }

    ekf_mat_vec_6x3(ekf->K, e, dx);

    dx[3] = 0.0f;
    dx[4] = CLAMP(dx[4],
                       -ekf->cfg.max_bias_correction,
                       ekf->cfg.max_bias_correction);
    dx[5] = CLAMP(dx[5],
                       -ekf->cfg.max_bias_correction,
                       ekf->cfg.max_bias_correction);

    for (uint32_t i = 0U; i < EKF_STATE_DIM; ++i) {
        ekf->x[i] += dx[i];
    }
    ekf_normalize_quaternion(ekf->x);

    st = arm_mat_mult_f32(&K, &H, &KH);
    if (st != ARM_MATH_SUCCESS) {
        return st;
    }
    ekf_set_identity(ekf->I_KH, EKF_STATE_DIM);
    for (uint32_t i = 0U; i < EKF_STATE_DIM * EKF_STATE_DIM; ++i) {
        ekf->I_KH[i] -= ekf->KH[i];
    }
    st = arm_mat_mult_f32(&I_KH, &P, &tmp66);
    if (st != ARM_MATH_SUCCESS) {
        return st;
    }

    memcpy(ekf->P, ekf->tmp66_a, sizeof(ekf->P));
    ekf_symmetrize_covariance(ekf->P);
    return ARM_MATH_SUCCESS;
}

/** 无符号时间戳差值换算为秒，天然处理计数溢出 */
static float ekf_elapsed_s(uint32_t now_us, uint32_t previous_us)
{
    return (float)(uint32_t)(now_us - previous_us) * 1.0e-6f;
}

/**
 * 滤波核心单步：先传播协方差，再积分四元数，最后用加速度校正。
 * 使用 ekf_update() 拷贝到句柄内部的六轴数据（ekf->accel / ekf->gyro）。
 */
static ekf_status_t ekf_step(ekf_t *ekf,
                             const float32_t gyro_rad_s[3],
                             const float32_t accel[3],
                             float32_t dt_s)
{
    if ((ekf == NULL) || (gyro_rad_s == NULL) || (accel == NULL)
        || (dt_s <= 0.0f)) {
        return EKF_NUMERIC_ERROR;
    }

    if (ekf_propagate_covariance(ekf, gyro_rad_s, dt_s) != ARM_MATH_SUCCESS) {
        return EKF_NUMERIC_ERROR;
    }
    ekf_predict_quaternion(ekf->x, gyro_rad_s, ekf->x[4], ekf->x[5], dt_s);

    const float32_t accel_norm = sqrtf(SQUARE(accel[0])
                                     + SQUARE(accel[1])
                                     + SQUARE(accel[2]));
    ekf->last_accel_norm = accel_norm;
    if (accel_norm <= EKF_EPS) {
        return EKF_PREDICT_ONLY;
    }

    if ((accel_norm < ekf->cfg.accel_norm_min)
        || (accel_norm > ekf->cfg.accel_norm_max)) {
        ekf->reject_count++;
        return EKF_REJECTED;
    }

    const float32_t inv_accel_norm = 1.0f / accel_norm;
    const float32_t z[3] = {
        accel[0] * inv_accel_norm,
        accel[1] * inv_accel_norm,
        accel[2] * inv_accel_norm
    };

    const arm_status st = ekf_correct_with_accel(ekf, z, gyro_rad_s);
    if (st == ARM_MATH_ARGUMENT_ERROR) {
        return EKF_REJECTED;
    }
    if (st != ARM_MATH_SUCCESS) {
        return EKF_NUMERIC_ERROR;
    }
    return EKF_OK;
}

/* ======================================================================
 * 3. 对外接口（仅此两个非 static）
 * ====================================================================== */

void ekf_init(ekf_t *ekf)
{
    ekf_config_t cfg;

    if (ekf == NULL) {
        return;
    }

    memset(ekf, 0, sizeof(*ekf));
    ekf_default_config(&cfg);
    ekf->cfg = cfg;
    ekf->x[0] = 1.0f;

    for (uint32_t i = 0U; i < EKF_STATE_DIM; ++i) {
        ekf->P[i * EKF_STATE_DIM + i] =
            (i < 4U) ? cfg.initial_attitude_variance
                     : cfg.initial_bias_variance;
        ekf->Q[i * EKF_STATE_DIM + i] =
            (i < 4U) ? cfg.q_gyro : cfg.q_bias;
    }
    for (uint32_t i = 0U; i < EKF_MEAS_DIM; ++i) {
        ekf->R[i * EKF_MEAS_DIM + i] = cfg.r_accel;
    }
    ekf->last_gain_scale = 1.0f;
    ekf->last_update_us = (uint32_t)DWT_GetTimeline_us();
    ekf->status = EKF_OK;
    ekf->ready = 1U;
}

ekf_euler_t ekf_update(ekf_t *ekf, const BMI088_t *imu)
{
    ekf_euler_t euler;
    uint32_t now_us = (uint32_t)DWT_GetTimeline_us();

    if (ekf == NULL) {
        euler.roll = 0.0f;
        euler.pitch = 0.0f;
        euler.yaw = 0.0f;
        return euler;
    }

    /* 默认返回上一次的有效欧拉角，失败路径直接返回它 */
    euler.roll = ekf->roll;
    euler.pitch = ekf->pitch;
    euler.yaw = ekf->yaw;

    if ((imu == NULL) || (ekf->ready == 0U)) {
        ekf->status = EKF_ERROR;
        return euler;
    }

    /* 只拷贝六轴数据，并在拷贝时做安装方向修正（传感器 X/Y -> 车体 roll/pitch） */
    ekf_map_imu_axes(imu->accel, ekf->accel);
    ekf_map_imu_axes(imu->gyro, ekf->gyro);

    ekf->dt_s = ekf_elapsed_s(now_us, ekf->last_update_us);
    ekf->last_update_us = now_us;
    if ((ekf->dt_s < 0.0002f) || (ekf->dt_s > 0.02f)) {
        ekf->status = EKF_ERROR_DT;
        return euler;
    }

    /* 静态检测 + yaw 轴零偏学习：yaw 无绝对参考，加速度计无法纠正 z 轴零偏，
       静止时用低通把 gyro[2] 的均值学成 gyro_bias_z 并扣除 */
    float32_t gyro_for_step[3];
    gyro_for_step[0] = ekf->gyro[0];
    gyro_for_step[1] = ekf->gyro[1];
    if (sqrtf(SQUARE(ekf->gyro[0]) + SQUARE(ekf->gyro[1]) + SQUARE(ekf->gyro[2])) < 0.02f)
    {
        ekf->gyro_bias_z += 0.001f * (ekf->gyro[2] - ekf->gyro_bias_z);
    }
    gyro_for_step[2] = ekf->gyro[2] - ekf->gyro_bias_z;

    ekf->status = ekf_step(ekf, gyro_for_step, ekf->accel, ekf->dt_s);

    /* 由更新后的四元数换算欧拉角，同时缓存到句柄里 */
    euler = ekf_euler_from_quaternion(ekf->x);
    ekf->roll = euler.roll;
    ekf->pitch = euler.pitch;
    ekf->yaw = euler.yaw;
    return euler;
}
