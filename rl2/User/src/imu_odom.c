#include "imu_odom.h"

#include "common_function.h"

#include <math.h>
#include <string.h>

imu_odom_t ODOM;

/* ======================================================================
 * 二阶卡尔曼（参考香港大学轮腿方案：状态 [距离, 速度]，输入前向加速度）
 * ====================================================================== */

static void odom_kf_init(imu_odom_t *odom)
{
    odom->kf_x[0] = 0.0f;
    odom->kf_x[1] = 0.0f;
    odom->kf_P[0][0] = 1.0e-4f;
    odom->kf_P[0][1] = 0.0f;
    odom->kf_P[1][0] = 0.0f;
    odom->kf_P[1][1] = 1.0e-4f;
}

/* 预测：x = [s, v]，输入加速度 a
 *   s += v·dt + 0.5·a·dt²
 *   v += a·dt
 *   F = [[1, dt], [0, 1]]，P = F·P·Fᵀ + Q
 *   Q = [[q·dt⁴/4, q·dt³/2], [q·dt³/2, q·dt²]]，q = σ_a²
 */
static void odom_kf_predict(imu_odom_t *odom, float accel, float dt)
{
    const float dt2 = dt * dt;
    const float q = ODOM_KF_SIGMA_ACCEL * ODOM_KF_SIGMA_ACCEL;
    const float p00 = odom->kf_P[0][0];
    const float p01 = odom->kf_P[0][1];
    const float p10 = odom->kf_P[1][0];
    const float p11 = odom->kf_P[1][1];

    odom->kf_x[0] += (odom->kf_x[1] * dt) + (0.5f * accel * dt2);
    odom->kf_x[1] += accel * dt;

    odom->kf_P[0][0] = p00 + (dt * (p01 + p10)) + (dt2 * p11)
                     + (0.25f * q * dt2 * dt2);
    odom->kf_P[0][1] = p01 + (dt * p11) + (0.5f * q * dt2 * dt);
    odom->kf_P[1][0] = odom->kf_P[0][1];
    odom->kf_P[1][1] = p11 + (q * dt2);
}

/* 量测更新：H = I，z = [距离, 速度]
 *   S = P + R，K = P·S⁻¹，x += K·(z − x)，P = (I − K)·P
 */
static void odom_kf_update(imu_odom_t *odom, float z_dist, float z_speed)
{
    const float r_d = ODOM_KF_SIGMA_DIST * ODOM_KF_SIGMA_DIST;
    const float r_v = ODOM_KF_SIGMA_SPEED * ODOM_KF_SIGMA_SPEED;
    const float s00 = odom->kf_P[0][0] + r_d;
    const float s01 = odom->kf_P[0][1];
    const float s10 = odom->kf_P[1][0];
    const float s11 = odom->kf_P[1][1] + r_v;
    const float det = (s00 * s11) - (s01 * s10);
    float k00;
    float k01;
    float k10;
    float k11;
    float p00;
    float p01;
    float p10;
    float p11;
    float sym;

    if (fabsf(det) < 1.0e-12f) {
        return;
    }
    const float inv_det = 1.0f / det;

    /* K = P·S⁻¹ */
    k00 = ((odom->kf_P[0][0] * s11) - (odom->kf_P[0][1] * s10)) * inv_det;
    k01 = ((-odom->kf_P[0][0] * s01) + (odom->kf_P[0][1] * s00)) * inv_det;
    k10 = ((odom->kf_P[1][0] * s11) - (odom->kf_P[1][1] * s10)) * inv_det;
    k11 = ((-odom->kf_P[1][0] * s01) + (odom->kf_P[1][1] * s00)) * inv_det;

    {
        const float r0 = z_dist - odom->kf_x[0];
        const float r1 = z_speed - odom->kf_x[1];
        odom->kf_x[0] += (k00 * r0) + (k01 * r1);
        odom->kf_x[1] += (k10 * r0) + (k11 * r1);
    }

    /* P = (I − K)·P */
    p00 = odom->kf_P[0][0];
    p01 = odom->kf_P[0][1];
    p10 = odom->kf_P[1][0];
    p11 = odom->kf_P[1][1];
    odom->kf_P[0][0] = ((1.0f - k00) * p00) - (k01 * p10);
    odom->kf_P[0][1] = ((1.0f - k00) * p01) - (k01 * p11);
    odom->kf_P[1][0] = (-k10 * p00) + ((1.0f - k11) * p10);
    odom->kf_P[1][1] = (-k10 * p01) + ((1.0f - k11) * p11);

    sym = 0.5f * (odom->kf_P[0][1] + odom->kf_P[1][0]);
    odom->kf_P[0][1] = sym;
    odom->kf_P[1][0] = sym;
}

/* ======================================================================
 * 对外接口
 * ====================================================================== */

void imu_odom_init(imu_odom_t *odom)
{
    if (odom == NULL) {
        return;
    }

    memset(odom, 0, sizeof(*odom));
    odom_kf_init(odom);
    odom->ready = 1U;
}

void imu_odom_reset(imu_odom_t *odom)
{
    if (odom == NULL) {
        return;
    }

    odom->distance = 0.0f;
    odom->speed = 0.0f;
    odom->mileage = 0.0f;
    odom->wheel_distance_left = 0.0f;
    odom->wheel_distance_right = 0.0f;
    odom->wheel_speed_left = 0.0f;
    odom->wheel_speed_right = 0.0f;
    odom->foot_distance = 0.0f;
    odom->foot_speed = 0.0f;
    odom->ecd_total_left = 0;
    odom->ecd_total_right = 0;
    odom->last_foot_distance = 0.0f;
    odom->ecd_ready = 0U;
    odom_kf_init(odom);
}

void imu_odom_update(imu_odom_t *odom,
                     const float accel_body[3],
                     const float gyro_body[3],
                     float roll_deg,
                     float pitch_deg,
                     int16_t wheel_right_rpm,
                     int16_t wheel_left_rpm,
                     uint16_t wheel_right_ecd,
                     uint16_t wheel_left_ecd,
                     float dt_s)
{
    int32_t d_l;
    int32_t d_r;
    float roll;
    float pitch;
    float sr;
    float cr;
    float sp;
    float cp;
    float a_f;

    (void)gyro_body;   /* 参考实现里二阶卡尔曼不使用角速度 */

    if ((odom == NULL) || (accel_body == NULL) || (gyro_body == NULL)
        || (odom->ready == 0U)) {
        return;
    }
    if ((dt_s < ODOM_DT_MIN_S) || (dt_s > ODOM_DT_MAX_S)) {
        return;
    }

    /* ---------- 1. 编码器累计（圈数展开，等价参考实现的 position + turns） ---------- */
    if (odom->ecd_ready == 0U) {
        odom->last_ecd_left = wheel_left_ecd;
        odom->last_ecd_right = wheel_right_ecd;
        odom->ecd_total_left = 0;
        odom->ecd_total_right = 0;
        odom->ecd_ready = 1U;
    } else {
        d_l = (int32_t)wheel_left_ecd - (int32_t)odom->last_ecd_left;
        d_r = (int32_t)wheel_right_ecd - (int32_t)odom->last_ecd_right;

        odom->last_ecd_left = wheel_left_ecd;
        odom->last_ecd_right = wheel_right_ecd;

        /* 13 位编码器（0~8191）回绕展开 */
        if (d_l > 4096) {
            d_l -= 8192;
        } else if (d_l < -4096) {
            d_l += 8192;
        }
        if (d_r > 4096) {
            d_r -= 8192;
        } else if (d_r < -4096) {
            d_r += 8192;
        }

        odom->ecd_total_left += d_l;
        odom->ecd_total_right += d_r;
    }

    /* ---------- 2. 轮子累计距离 / 线速度（与参考实现同源） ---------- */
    odom->wheel_distance_left = ODOM_LEFT_WHEEL_RPM_SIGN
                              * (float)odom->ecd_total_left
                              * ODOM_M_PER_ECD_COUNT;
    odom->wheel_distance_right = ODOM_RIGHT_WHEEL_RPM_SIGN
                               * (float)odom->ecd_total_right
                               * ODOM_M_PER_ECD_COUNT;

    odom->wheel_speed_left = ODOM_LEFT_WHEEL_RPM_SIGN
                           * ODOM_RPM_TO_MPS
                           * (float)wheel_left_rpm;
    odom->wheel_speed_right = ODOM_RIGHT_WHEEL_RPM_SIGN
                            * ODOM_RPM_TO_MPS
                            * (float)wheel_right_rpm;

    /* ---------- 3. 两轮平均（参考实现 foot_distance / foot_speed） ---------- */
    odom->foot_distance = 0.5f * (odom->wheel_distance_left
                                + odom->wheel_distance_right);
    odom->foot_speed = 0.5f * (odom->wheel_speed_left
                             + odom->wheel_speed_right);

    /* ---------- 4. 前向加速度：pitch 倾斜补偿（比力投影到水平面） ---------- */
    roll = DEG_TO_RAD(roll_deg);
    pitch = DEG_TO_RAD(pitch_deg);
    sr = sinf(roll);
    cr = cosf(roll);
    sp = sinf(pitch);
    cp = cosf(pitch);
    a_f = (cp * accel_body[0])
        + ((sp * sr) * accel_body[1])
        + ((sp * cr) * accel_body[2]);

    if (!isfinite(a_f) || !isfinite(odom->foot_distance)
        || !isfinite(odom->foot_speed)) {
        return;
    }

    /* ---------- 5. 二阶卡尔曼：输入加速度，量测 [距离, 速度] ---------- */
    odom_kf_predict(odom, a_f, dt_s);
    odom_kf_update(odom, odom->foot_distance, odom->foot_speed);

    /* ---------- 6. 输出 + 路程（单调） ---------- */
    odom->distance = odom->kf_x[0];
    odom->speed = odom->kf_x[1];
    odom->mileage += fabsf(odom->foot_distance - odom->last_foot_distance);
    odom->last_foot_distance = odom->foot_distance;
}
