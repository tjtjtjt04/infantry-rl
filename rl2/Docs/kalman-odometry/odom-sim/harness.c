/* ============================================================================
 * imu_odom 离线仿真测试台（PC 上用 MinGW gcc 编译，不需要硬件）
 * ----------------------------------------------------------------------------
 * 当前实现的参考对象：香港大学轮腿开源 Chassis_Task.c
 *   - 编码器累计 -> 轮子距离（圈数展开）
 *   - speed_rpm -> 轮子速度
 *   - foot_distance / foot_speed 取两轮平均
 *   - 二阶卡尔曼 [距离, 速度]，输入前向加速度（pitch 投影）
 *
 * 编译（在 Docs/kalman-odometry/odom-sim 目录下）：
 *   gcc -O2 -Wall -Wextra -I ../../../User/inc harness.c ../../../User/src/imu_odom.c -lm -o odom_sim.exe
 * 运行：./odom_sim.exe
 * ========================================================================== */

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "imu_odom.h"

#define OD_G       (9.80665f)
#define OD_DEG2RAD (0.017453292519943295f)
#define OD_DT      (0.001f)
#define OD_B       (0.44289f)   /* 轮距，仅"原地旋转不产生里程"测试用 */

static imu_odom_t od;
static int failures = 0;

/* 仿真真值：两轮累计行程（double，避免 float 累加误差） */
static double true_dist_L = 0.0;
static double true_dist_R = 0.0;

static int16_t rpm_from_speed(float v_mps, float sign)
{
    return (int16_t)lroundf(v_mps / (sign * ODOM_RPM_TO_MPS));
}

static uint16_t ecd_from_dist(double dist_m, float sign)
{
    const long c = lround(dist_m / ((double)sign * (double)ODOM_M_PER_ECD_COUNT));
    return (uint16_t)(c & 0x1FFF);
}

/* 统一提交一帧 */
static void submit(const float accel[3], const float gyro[3],
                   float roll_deg, float pitch_deg,
                   float v_left, float v_right)
{
    if (od.ecd_ready == 0U) {
        true_dist_L = 0.0;
        true_dist_R = 0.0;
    }
    true_dist_L += (double)v_left * (double)OD_DT;
    true_dist_R += (double)v_right * (double)OD_DT;

    imu_odom_update(&od, accel, gyro, roll_deg, pitch_deg,
                    rpm_from_speed(v_right, ODOM_RIGHT_WHEEL_RPM_SIGN),
                    rpm_from_speed(v_left, ODOM_LEFT_WHEEL_RPM_SIGN),
                    ecd_from_dist(true_dist_R, ODOM_RIGHT_WHEEL_RPM_SIGN),
                    ecd_from_dist(true_dist_L, ODOM_LEFT_WHEEL_RPM_SIGN),
                    OD_DT);
}

/* 水平静止/匀速：ax 为车体 X 加速度；pitch 非零时按"真实倾斜"生成比力 */
static void feed(float ax, float wz, float v_left, float v_right, float pitch_deg)
{
    float accel[3] = {ax, 0.0f, OD_G};
    float gyro[3] = {0.0f, 0.0f, wz};

    if (fabsf(pitch_deg) > 1e-6f) {
        const float p = pitch_deg * OD_DEG2RAD;
        accel[0] = -OD_G * sinf(p);
        accel[1] = 0.0f;
        accel[2] = OD_G * cosf(p);
    }
    submit(accel, gyro, 0.0f, pitch_deg, v_left, v_right);
}

static void check(const char *name, float got, float want, float tol)
{
    const float err = fabsf(got - want);
    const int ok = (err <= tol);
    printf("  %-32s got=%10.5f  want=%10.5f  err=%9.5f  %s\n",
           name, got, want, err, ok ? "PASS" : "FAIL");
    if (!ok) {
        failures++;
    }
}

/* [1] 静止 3 s */
static void test_static(void)
{
    printf("[1] 静止 3 s\n");
    imu_odom_init(&od);
    for (int i = 0; i < 3000; i++) {
        feed(0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    check("distance (m)", od.distance, 0.0f, 1.0e-5f);
    check("mileage (m)", od.mileage, 0.0f, 1.0e-5f);
    check("speed (m/s)", od.speed, 0.0f, 1.0e-3f);
    printf("\n");
}

/* [2] 1 m/s 匀速 3 s */
static void test_straight(void)
{
    printf("[2] 匀速直线 1 m/s × 3 s\n");
    imu_odom_init(&od);
    for (int i = 0; i < 3000; i++) {
        feed(0.0f, 0.0f, 1.0f, 1.0f, 0.0f);
    }
    check("distance (m)", od.distance, 3.0f, 1.0e-2f);
    check("mileage (m)", od.mileage, 3.0f, 1.0e-2f);
    check("speed (m/s)", od.speed, 1.0f, 1.0e-2f);
    printf("\n");
}

/* [3] 短推 1 m/s × 0.4 s（真值 0.4 m） */
static void test_short_push(void)
{
    printf("[3] 短距离 1 m/s × 0.4 s（真值 0.4 m）\n");
    imu_odom_init(&od);
    for (int i = 0; i < 400; i++) {
        feed(0.0f, 0.0f, 1.0f, 1.0f, 0.0f);
    }
    check("distance (m)", od.distance, 0.4f, 5.0e-3f);
    printf("\n");
}

/* [4] 1 m/s² 加速 1 s，再匀速 1 s */
static void test_accel(void)
{
    printf("[4] 匀加速 1 m/s² × 1 s，再匀速 1 s\n");
    imu_odom_init(&od);
    for (int i = 0; i < 1000; i++) {
        const float t = (float)(i + 1) * OD_DT;
        feed(1.0f, 0.0f, t, t * 1.0f, 0.0f);
    }
    check("distance @1s (m)", od.distance, 0.5f, 2.0e-2f);
    check("speed @1s (m/s)", od.speed, 1.0f, 2.0e-2f);
    for (int i = 0; i < 1000; i++) {
        feed(0.0f, 0.0f, 1.0f, 1.0f, 0.0f);
    }
    check("distance @2s (m)", od.distance, 1.5f, 2.0e-2f);
    printf("\n");
}

/* [5] 原地旋转 1 rad/s × 1 s */
static void test_turn_in_place(void)
{
    const float wz = 1.0f;
    const float v_half = wz * OD_B * 0.5f;
    printf("[5] 原地旋转 1 rad/s × 1 s\n");
    imu_odom_init(&od);
    for (int i = 0; i < 1000; i++) {
        feed(0.0f, wz, -v_half, +v_half, 0.0f);
    }
    check("distance (m)（旋转不产生里程）", od.distance, 0.0f, 1.0e-3f);
    check("mileage (m)", od.mileage, 0.0f, 1.0e-3f);
    printf("\n");
}

/* [6] 俯仰 10° 静止 2 s：pitch 投影应抵消重力 */
static void test_tilt_static(void)
{
    printf("[6] 机体俯仰 10° 静止 2 s\n");
    imu_odom_init(&od);
    for (int i = 0; i < 2000; i++) {
        feed(0.0f, 0.0f, 0.0f, 0.0f, 10.0f);
    }
    check("distance (m)", od.distance, 0.0f, 1.0e-5f);
    check("speed (m/s)", od.speed, 0.0f, 1.0e-2f);
    printf("\n");
}

/* [7] 静止 + EKF 姿态误差 1°（真实水平）：a_f 会漏重力，但编码器距离应为 0 */
static void test_static_attitude_error(void)
{
    const float pitch_err = 1.0f;
    printf("[7] 静止 60 s，EKF pitch 有 +1.0° 误差（真实水平）\n");
    imu_odom_init(&od);
    for (int i = 0; i < 60000; i++) {
        const float accel[3] = {0.0f, 0.0f, OD_G};
        const float gyro[3] = {0.0f, 0.0f, 0.0f};
        submit(accel, gyro, 0.0f, pitch_err, 0.0f, 0.0f);
    }
    printf("     60 s 后：distance=%.5f m, speed=%.5f m/s, mileage=%.5f m\n\n",
           (double)od.distance, (double)od.speed, (double)od.mileage);
}

/* [8] 打滑信息：轮子转、车不动 */
static void test_slip_info(void)
{
    printf("[8] 打滑信息：轮速/编码器报 1 m/s，IMU 加速度为 0（真实不动）\n");
    imu_odom_init(&od);
    for (int i = 0; i < 1000; i++) {
        feed(0.0f, 0.0f, 1.0f, 1.0f, 0.0f);
    }
    printf("     distance=%.4f m, speed=%.4f m/s —— 打滑时里程跟随轮子（需打滑检测）\n\n",
           (double)od.distance, (double)od.speed);
}

/* [9] 前进 1 m 再后退 1 m：distance 回到 0，mileage 约 2 m */
static void test_back_and_forth(void)
{
    printf("[9] 前进 1 m 再后退 1 m\n");
    imu_odom_init(&od);
    for (int i = 0; i < 1000; i++) {
        feed(0.0f, 0.0f, 1.0f, 1.0f, 0.0f);
    }
    for (int i = 0; i < 1000; i++) {
        feed(0.0f, 0.0f, -1.0f, -1.0f, 0.0f);
    }
    check("distance (m)（回原点）", od.distance, 0.0f, 2.0e-2f);
    check("mileage (m)（路程 2 m）", od.mileage, 2.0f, 2.0e-2f);
    printf("\n");
}

int main(void)
{
    printf("========== imu_odom（港大轮腿方案）离线仿真 ==========\n");
    printf("轮半径 %.4f m, 减速比 %.6f, rpm->m/s %.8f, 每计数 %.9f m\n\n",
           (double)ODOM_WHEEL_RADIUS_M, (double)ODOM_GEAR_RATIO,
           (double)ODOM_RPM_TO_MPS, (double)ODOM_M_PER_ECD_COUNT);

    test_static();
    test_straight();
    test_short_push();
    test_accel();
    test_turn_in_place();
    test_tilt_static();
    test_static_attitude_error();
    test_slip_info();
    test_back_and_forth();

    printf("========== 结果：%s（失败 %d 项）==========\n",
           failures == 0 ? "全部通过" : "存在失败", failures);
    return (failures == 0) ? 0 : 1;
}
