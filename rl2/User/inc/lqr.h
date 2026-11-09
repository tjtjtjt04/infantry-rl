#ifndef LQR_H
#define LQR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * 轮腿 LQR —— 精简接口（一个初始化 + 一个计算输出）
 * ==========================================================================
 * 数学形式：u = K(L0) · (x_ref − x)
 *   K 的每个元素是腿长 L0 的三次多项式：K_ij(L0) = c0 + c1·L0 + c2·L0² + c3·L0³
 *
 * 状态顺序（单位不可混用）：
 *   x = [theta, dtheta, xb, dxb, phi, dphi]
 *         rad    rad/s    m    m/s   rad  rad/s
 *   theta : 摆杆相对竖直方向的夹角      dtheta : 其角速度
 *   xb    : 机体水平位置                dxb    : 机体水平速度
 *   phi   : 机体倾角（相对水平面）      dphi   : 机体倾角角速度
 *   注意 theta 与 phi 是两个不同的角度，K 表按此含义整定。
 *
 * 输出：
 *   u = [T, Tp]
 *   T  : 轮子驱动力矩，Tp : 关节/腿轴力矩（具体落到哪个执行器由上层决定）
 *
 * 用法（每个控制周期，左右腿各一套）：
 *   lqr_init();                                  // 上电初始化一次（左右都清）
 *   LQR_Left.x[...]  = 左腿状态;                 // 填左腿输入段
 *   LQR_Left.x_ref[...] = 左腿目标;
 *   LQR_Left.l0_m = fivebar_dynamic_left.L0;
 *   LQR_Right.x[...] = 右腿状态;                 // 填右腿输入段
 *   LQR_Right.x_ref[...] = 右腿目标;
 *   LQR_Right.l0_m = fivebar_dynamic_right.L0;
 *   lqr_calc();                                  // 左右分别计算
 *   // 结果：LQR_Left.T / Tp / T_cmd，LQR_Right.T / Tp / T_cmd
 *   //       T    ：轮子力矩 (N·m)，LQR 原始输出
 *   //       T_cmd：T 换算出的 3508 电流指令（限幅 ±8192，对应 ±10 A / ±2.46 N·m）
 *
 * 失败语义：输入含 NaN/±Inf、腿长非法时输出清零，不会残留上一次结果。
 * ========================================================================== */

/* 状态向量维数，同时也是 K 矩阵列数 */
#define LQR_STATE_COUNT 6u

/* 控制输出维数，同时也是 K 矩阵行数 */
#define LQR_OUTPUT_COUNT 2u

/* ==================== 3508 电流指令换算 ====================
 * 电流标定：16384 对应 20 A；力矩常数 kt = 0.246 N·m/A
 *   T [N·m] -> I [A] = T / kt
 *           -> cmd = I × 16384/20 = T × 16384/(20·kt) ≈ T × 3330
 *   限幅：±8192 对应 ±10 A × 0.246 = ±2.46 N·m，超出即饱和到 ±8192。 */
#define LQR_3508_KT_NM_PER_A   (0.246f)
#define LQR_3508_CMD_PER_A     (16384.0f / 20.0f)
#define LQR_CMD_LIMIT          (8192.0f)   /* 电流指令限幅：±8192 = ±10 A */
#define LQR_T_DIRECTION_LEFT   (1.0f)
#define LQR_T_DIRECTION_RIGHT  (-1.0f)
/**
 * LQR 结构体（输入 + 输出合一；模块内全局实例 LQR）
 *
 * 输入段由上层每个控制周期填写，输出段由 lqr_calc() 写入。所有量集中在
 * 同一个结构体里，调试器一次就能看全"给了什么、算出了什么"。
 */
typedef struct
{
    /* ---------- 输入（上层填） ---------- */
    float x[LQR_STATE_COUNT];      /**< 当前状态 [theta, dtheta, xb, dxb, phi, dphi] */
    float x_ref[LQR_STATE_COUNT];  /**< 目标状态，同序同单位 */
    float l0_m;                    /**< 当前腿长 (m)，用于按腿长拟合 K */

    /* ---------- 输出（lqr_calc 写） ---------- */
    float T;                       /**< 轮子驱动力矩 (N·m)，LQR 原始输出 */
    float Tp;                      /**< 关节/腿轴力矩 (N·m) */
    int16_t T_cmd;                 /**< T 换算出的 3508 电流指令（限幅 ±8192 = ±10 A） */
} lqr_t;


/** 左右两套 LQR 实例：上层分别填输入段 -> lqr_calc() -> 读各自输出段 */
extern lqr_t LQR_Left;
extern lqr_t LQR_Right;

/**
 * @brief 初始化：左右两套实例的输入/输出全部清零，各自载入腿长缺省值。
 *
 * 上电后调用一次即可（重复调用等价于重新清零）。
 */
void lqr_init(void);

/**
 * @brief 分别计算左右两套实例，结果写回各自的 T / Tp。
 *
 * 每套：u = K(L0)·(x_ref − x)，K(L0) 由内置三次拟合表按各自 l0_m 插值；
 * l0_m 超出拟合区间时夹到端点（不外推）。
 * 某侧输入非法（NaN/±Inf、腿长非法）时该侧输出段清零，不影响另一侧。
 */
void lqr_calc(void);

#ifdef __cplusplus
}
#endif

#endif /* LQR_H */
