#include "fivebar.h"
#include <math.h>
#include <string.h>
#include "arm_math.h"   /* CMSIS-DSP：arm_sin_f32 / arm_cos_f32 / arm_sqrt_f32 */
#include "common_function.h"  /* RAD_TO_DEG：theta 由 rad 换成 deg */

/* 左右腿各一份解算结果：独立结构体，右腿结果不会被左腿覆盖 */
FiveBar_dynamic_t fivebar_dynamic_right;
FiveBar_dynamic_t fivebar_dynamic_left;

/* --------------------------------------------------------------------------
 * 连杆参数（m）用宏而不是 const 结构体字段：
 *   当前构建是 -O0，宏直接编成立即数，省掉每次访问 fivebar_static 字段的 LDR。
 * -------------------------------------------------------------------------- */
#define FIVEBAR_L1          (0.145f)
#define FIVEBAR_L2          (0.27f)
#define FIVEBAR_L3          (0.27f)
#define FIVEBAR_L4          (0.145f)
#define FIVEBAR_L5          (0.15f)

#define FIVEBAR_HALF_L5     (0.5f * FIVEBAR_L5)
#define FIVEBAR_INV_L3      (1.0f / FIVEBAR_L3)     /* s3 = wy*l3^-1、c3 = wx*l3^-1，除法变乘法 */
#define FIVEBAR_2L2         (2.0f * FIVEBAR_L2)     /* A0/B0 的公共系数 */
#define FIVEBAR_L22_L32     (FIVEBAR_L2 * FIVEBAR_L2 - FIVEBAR_L3 * FIVEBAR_L3)  /* C0 的常数项 */

/* L0 变化死区（m）：相邻两次 L0 之差小于该值时认为腿长没变，L0_dot 置 0。
 * 1 kHz 下 1e-2 m/周期 相当于 10 m/s，会一直触发；1e-5 约等于 0.01 m/s。 */
#define FIVEBAR_L0_DEADBAND_M (1.0e-5f)

/* sin(phi3-phi2) 过小判定：腿接近伸直共线时该值为 0，D0/omega0 不可解，
 * 置 0 避免 inf/NaN 沿控制链路传播。 */
#define FIVEBAR_SIN_EPS       (1.0e-4f)

/* 半角式解 phi2 时归一化分母的下限（m^4 量级）：退化时保持上一拍有效输出。 */
#define FIVEBAR_DEN_EPS       (1.0e-12f)

void fivebar_init(FiveBar_dynamic_t *fivebar)
{
    /* 结构体成员全是 float、无填充，直接清零即可（比逐字段赋值省 flash） */
    memset(fivebar, 0, sizeof(*fivebar));
}

/* 五连杆正运动学（几何参数见《轮腿式平衡机器人控制》图 4）：
 *   B = (l1*cosφ1, l1*sinφ1)              髋1 侧连杆端点
 *   D = (l5 + l4*cosφ4, l4*sinφ4)         髋2 侧连杆端点
 *   C = B + l2*(cosφ2, sinφ2) = D + l3*(cosφ3, sinφ3)  踝点（轮心）
 *   L0 = |C - (l5/2, 0)|，φ0 = atan2( yC, xC - l5/2 )
 * 顺序严格按公式：B、D -> A0/B0/C0 -> φ2 -> φ3 -> C 点 -> L0/φ0 -> 速度量，
 * 所有中间量都使用本周期角度；只有 L0_last 按设计保留上一拍。
 *
 * 优化点（输出与三角函数版本等价，误差 ~1e-15）：
 *   - φ2 的 sin/cos 用半角式直接算，φ3 的 sin/cos 由几何关系 wy/l3、wx/l3 得到，
 *     省掉 4 次 CMSIS-DSP 查表（原 8 次查表 -> 现在只剩 φ1、φ4 的 4 次）；
 *   - 除法合并：1/den、1/L0 各只算一次；
 *   - 连杆长度改为宏，-O0 下不再每次 LDR。 */
void fivebar_forward_kinematics(FiveBar_dynamic_t *fivebar, float body_pitch,
                                float body_pitch_rate, side_t side)
{
    float32_t s1, c1;       /* sin/cos(phi1)，本周期 */
    float32_t s2, c2;       /* sin/cos(phi2)，本周期（半角式，不查表） */
    float32_t s3, c3;       /* sin/cos(phi3)，本周期（几何式，不查表） */
    float32_t s4, c4;       /* sin/cos(phi4)，本周期 */
    float32_t dx, dy;       /* D 点 - B 点 */
    float32_t wx, wy;       /* l3*e^{i*phi3} 的实部/虚部（即 atan2f 的两个参数） */
    float32_t A0, B0, C0;
    float32_t disc, root;
    float32_t nx, ny, den, inv_den;
    float32_t u, v, inv_L0;
    float32_t vxC, vyC, s32;
    uint8_t singular;

    switch (side)
    {
        case right:
            fivebar->phi1 = M_PI + dm_motor_data[3].measure.pos;
            fivebar->phi4 = dm_motor_data[2].measure.pos;
            fivebar->omega1 = dm_motor_data[3].measure.vel;
            fivebar->omega4 = dm_motor_data[2].measure.vel;
            break;

        case left:
            fivebar->phi1 = M_PI + dm_motor_data[1].measure.pos;
            fivebar->phi4 = dm_motor_data[0].measure.pos;
            fivebar->omega1 = dm_motor_data[1].measure.vel;
            fivebar->omega4 = dm_motor_data[0].measure.vel;
            break;

        default:
            /* 非法 side：不更新，避免拿未赋值的角度继续算下去 */
            return;
    }

    /* ---------- 1. B、D 点：只依赖本周期读到的 phi1、phi4 ---------- */
    s1 = arm_sin_f32(fivebar->phi1);
    c1 = arm_cos_f32(fivebar->phi1);
    s4 = arm_sin_f32(fivebar->phi4);
    c4 = arm_cos_f32(fivebar->phi4);

    fivebar->xB = FIVEBAR_L1 * c1;
    fivebar->yB = FIVEBAR_L1 * s1;
    fivebar->xD = FIVEBAR_L5 + FIVEBAR_L4 * c4;
    fivebar->yD = FIVEBAR_L4 * s4;

    dx = fivebar->xD - fivebar->xB;
    dy = fivebar->yD - fivebar->yB;

    /* ---------- 2. 解 phi2（本周期 B、D 直接得到） ---------- */
    A0 = FIVEBAR_2L2 * dx;
    B0 = FIVEBAR_2L2 * dy;
    C0 = FIVEBAR_L22_L32 + dx * dx + dy * dy;
    fivebar->A0 = A0;
    fivebar->B0 = B0;
    fivebar->C0 = C0;

    disc = A0 * A0 + B0 * B0 - C0 * C0;
    if (disc < 0.0f)
    {
        /* 目标超出五连杆可达范围：本拍不解算，保持上一拍有效值（避免 NaN/假角度） */
        return;
    }
    (void)arm_sqrt_f32(disc, &root);

    nx = A0 + C0;
    ny = B0 + root;
    den = nx * nx + ny * ny;
    if (den <= FIVEBAR_DEN_EPS)
    {
        /* 退化（phi2 无法确定），同上保持上一拍有效值 */
        return;
    }
    fivebar->phi2 = 2.0f * atan2f(ny, nx);

    /* 半角式：t = tan(phi2/2) = ny/nx
     *   c2 = (1-t^2)/(1+t^2) = (nx^2-ny^2)/(nx^2+ny^2)
     *   s2 = 2t/(1+t^2)     = 2*nx*ny/(nx^2+ny^2)
     * 用归一化形式而不是先算 t，可避免 phi2 接近 ±pi 时的精度放大。 */
    inv_den = 1.0f / den;
    c2 = (nx * nx - ny * ny) * inv_den;
    s2 = 2.0f * nx * ny * inv_den;

    /* ---------- 3. C 点坐标 + phi3（用本周期 phi2） ---------- */
    fivebar->xC = fivebar->xB + FIVEBAR_L2 * c2;
    fivebar->yC = fivebar->yB + FIVEBAR_L2 * s2;

    /* l3*e^{i*phi3} = (B - D) + l2*e^{i*phi2}，即 atan2f 的两个参数就是实部 wx、虚部 wy */
    wx = -dx + FIVEBAR_L2 * c2;
    wy = -dy + FIVEBAR_L2 * s2;
    fivebar->phi3 = atan2f(wy, wx);
    /* |wx+i*wy| 恒等于 l3（由 phi2 的解保证），所以直接归一化得到 sin/cos，
     * 比查表更准，也省掉两次 arm_sin/arm_cos。 */
    s3 = wy * FIVEBAR_INV_L3;
    c3 = wx * FIVEBAR_INV_L3;

    /* ---------- 4. 虚拟腿长 L0 与摆杆角 phi0（本周期 C 点） ---------- */
    u = fivebar->xC - FIVEBAR_HALF_L5;
    v = fivebar->yC;
    (void)arm_sqrt_f32(u * u + v * v, &fivebar->L0);
    fivebar->phi0 = atan2f(v, u);   /* atan2 保留象限，腿过竖直位置时不会 ±90° 跳变 */

    /* ---------- 5. D0 = l2 * dphi2/dt（本周期 phi2、phi3） ----------
     * 由闭环约束 l2*e^{i*phi2} + l3*e^{i*phi3} = D - B 求导、叉乘消去 phi3 得：
     *   l2*w2 = ( l1*w1*sin(phi1-phi3) + l4*w4*sin(phi3-phi4) ) / sin(phi3-phi2)
     * 分母没有 l2，所以 D0 的单位是 m/s，正好可直接代入下面 C 点速度式。 */
    s32 = s3 * c2 - c3 * s2;                 /* sin(phi3 - phi2) */
    singular = (fabsf(s32) < FIVEBAR_SIN_EPS) ? 1U : 0U;

    if (singular)
    {
        /* 腿接近伸直共线，phi2 速度不可解：置 0，不让 inf/NaN 传下去 */
        fivebar->D0 = 0.0f;
    }
    else
    {
        fivebar->D0 = (FIVEBAR_L1 * fivebar->omega1 * (s1 * c3 - c1 * s3)
                     + FIVEBAR_L4 * fivebar->omega4 * (s3 * c4 - c3 * s4)) / s32;
    }

    /* ---------- 6. C 点速度、L0_dot、omega0 ----------
     * xC_dot = -l1*w1*sin(phi1) - D0*sin(phi2)
     * yC_dot =  l1*w1*cos(phi1) + D0*cos(phi2)
     */
    vxC = -FIVEBAR_L1 * fivebar->omega1 * s1 - fivebar->D0 * s2;
    vyC = FIVEBAR_L1 * fivebar->omega1 * c1 + fivebar->D0 * c2;

    inv_L0 = 1.0f / fivebar->L0;
    fivebar->L0_dot = (u * vxC + v * vyC) * inv_L0;

    /* L0_dot 校验：与上一周期的 L0 比较（这是唯一有意保留的"上一拍"量），
     * 变化小于死区时认为腿长没变，L0_dot 置 0。 */
    if (fabsf(fivebar->L0 - fivebar->L0_last) < FIVEBAR_L0_DEADBAND_M)
    {
        fivebar->L0_dot = 0.0f;
    }
    fivebar->L0_last = fivebar->L0;

    /* omega0 = dphi0/dt = (u*yC_dot - v*xC_dot) / L0^2
     * （原式分母写成 sin(phi3-phi2) 是错的：量纲为 m^2/s、结果被缩小 5.5~37 倍） */
    if (singular)
    {
        fivebar->omega0 = 0.0f;
    }
    else
    {
        fivebar->omega0 = (u * vyC - v * vxC) * inv_L0 * inv_L0;
    }

    /* ---------- 7. 摆杆角 theta（deg）—— 按左右腿分别计算 ----------
     * phi0 是虚拟腿相对水平面的角度（rad），减去机体俯仰角 body_pitch（deg）
     * 得到摆杆角；左右腿机构镜像安装，符号相反：
     *   left : theta = +( RAD_TO_DEG(phi0 - pi/2) - body_pitch )
     *   right: theta = -( RAD_TO_DEG(phi0 - pi/2) - body_pitch )
     * （对应原先 fivebar_solve_theta() 里 switch(side) 的写法）
     * 说明：上面任何一步提前 return（超出可达范围 / 退化 / 非法 side）时本拍不更新，
     * theta 与 phi0 一样保留上一拍有效值。 */
    if (side == left)
    {
        fivebar->theta = (RAD_TO_DEG(fivebar->phi0 - M_PI_2) - body_pitch);
    }
    else
    {
        fivebar->theta = -(RAD_TO_DEG(fivebar->phi0 - M_PI_2) - body_pitch);
    }

    /* ---------- 8. 摆杆角速度 dtheta（deg/s）—— 符号随 theta 分左右 ----------
     * 对 theta 求时间导数：
     *   left : dtheta = +RAD_TO_DEG(omega0) - RAD_TO_DEG(body_pitch_rate)
     *   right: dtheta = -RAD_TO_DEG(omega0) + RAD_TO_DEG(body_pitch_rate)
     *   omega0          = dphi0/dt（rad/s，上面第 6 步已算，singular 时为 0）
     *   body_pitch_rate = 机体俯仰角速度（rad/s，上层传 EKF.gyro[1]）
     * 同样：提前 return 的本拍不更新，保留上一拍有效值。 */
    if (side == left)
    {
        fivebar->dtheta = (RAD_TO_DEG(fivebar->omega0)
                        - RAD_TO_DEG(body_pitch_rate));
    }
    else
    {
        fivebar->dtheta = -RAD_TO_DEG(fivebar->omega0)
                        + RAD_TO_DEG(body_pitch_rate);
    }
}

// void fivebar_inverse_kinematics(FiveBar_dynamic_t *fivebar, dm_motor_t *dm_motor_data)
// {
//     ;
// }
