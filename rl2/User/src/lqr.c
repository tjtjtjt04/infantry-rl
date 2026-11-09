/*
 * ============================================================================
 * 轮腿 LQR —— 精简实现（一个初始化 + 一个计算输出）
 * ============================================================================
 * 本文件只保留：
 *   lqr_init()  —— 左右两套实例清空、载入内置增益拟合表参数
 *   lqr_calc()  —— 左右分别计算 u = K(L0)·(x_ref − x)，写回各自 T / Tp，
 *                  并把 T 换算成 3508 电流指令写入 T_cmd（16384 = 20 A，限幅 ±8192）
 *
 * K 的每个元素是腿长 L0 的三次多项式，系数按 [输出][状态][系数] 存放，
 * 每个元素的 4 个系数从左到右依次对应 L0^0、L0^1、L0^2、L0^3。
 * ============================================================================
 */

#include "lqr.h"

#include <float.h>

/* ==========================================================================
 * 一、内部常量与内部数据
 * ========================================================================== */

/* 每个增益元素的三次拟合系数个数 */
#define LQR_FIT_COEFF_COUNT 4u

/* 拟合有效腿长区间 (m)：与 FiveBar 默认几何的可达腿长对齐，区间外夹到端点 */
#define LQR_L0_MIN_M (0.1042f)
#define LQR_L0_MAX_M (0.3672f)

/* 有限值判定（不依赖 libm 的 isfinite，任意嵌入式 C 库可编译） */
#define LQR_IS_FINITE(v) \
    ((((v) == (v)) && ((v) <= FLT_MAX) && ((v) >= -FLT_MAX)) ? 1u : 0u)

/*
 * K 的三次拟合参数表 [2][6][4]：
 *   行 0 = T 的 6 个增益，行 1 = Tp 的 6 个增益；
 *   列序与状态向量一致 [theta, dtheta, xb, dxb, phi, dphi]。
 *
 * 该表为 2026 机型整定/拟合结果，属机器人配置而非算法常量：更换机型、
 * 改动结构件（连杆长度、质量分布、轮径）或重新整定后必须整体重新拟合。
 */
static const float k_fit_L0[LQR_OUTPUT_COUNT]
                            [LQR_STATE_COUNT]
                            [LQR_FIT_COEFF_COUNT] =
{
    {   /* T 行 */
        {    -72.1813438f,    118.7385334f,   -124.9707211f,     -7.5122864f}, /* T/theta */
        {     10.8015433f,    -14.4758382f,    -15.3121624f,     -0.9063853f}, /* T/dtheta */
        {    -26.6463491f,     27.0808742f,    -10.3367842f,     -3.3391748f}, /* T/x */
        {    -54.8284069f,     56.4270506f,    -24.0514218f,     -7.8257963f}, /* T/dx */
        {   -108.9387999f,    134.6294494f,    -72.7220581f,     23.1994318f}, /* T/phi */
        {    -22.7373273f,     29.2083757f,    -16.2373531f,      5.6733651f} /* T/dphi */
    },
    {   /* Tp 行 */
        {    124.9334832f,   -149.1037756f,     49.9553971f,     13.2509206f}, /* Tp/theta */
        {      1.6821408f,    -10.7305567f,      8.2268752f,      1.8037620f}, /* Tp/dtheta */
        {    -39.6436645f,     46.6358647f,    -24.3292902f,      7.1623332f}, /* Tp/x */
        {    -91.3355580f,    104.9568992f,    -53.8135171f,     16.1922171f}, /* Tp/dx */
        {    313.6102238f,   -320.4657794f,    123.6415666f,     22.2280469f}, /* Tp/phi */
        {     75.3047235f,    -77.4506711f,     30.2964724f,      4.5231072f} /* Tp/dphi */
    }
};

/* 全局实例：左右各一套（上层分别填输入段，lqr_calc 分别写输出段） */
lqr_t LQR_Left;
lqr_t LQR_Right;

/* ==========================================================================
 * 二、内部实现
 * ========================================================================== */

/*
 * 按腿长求增益矩阵 K(L0)。
 * 逐元素做三次多项式求值（Horner 法），任一元素非有限值即整体失败。
 * 返回 1 成功；0 失败（腿长非法或拟合结果非有限值）。
 */
static uint8_t lqr_get_gain(float l0_m,
                            float gain[LQR_OUTPUT_COUNT][LQR_STATE_COUNT])
{
    float l0;
    uint8_t row;
    uint8_t column;
    uint8_t coefficient;

    if (LQR_IS_FINITE(l0_m) == 0u)
    {
        return 0u;
    }

    /* 拟合区间外不外推：夹到最近端点（三次拟合在区间外不可信） */
    l0 = l0_m;
    if (l0 < LQR_L0_MIN_M)
    {
        l0 = LQR_L0_MIN_M;
    }
    else if (l0 > LQR_L0_MAX_M)
    {
        l0 = LQR_L0_MAX_M;
    }

    /* Horner：k = c3; k = k·L0 + c2; ...; k = k·L0 + c0 */
    for (row = 0u; row < LQR_OUTPUT_COUNT; row++)
    {
        for (column = 0u; column < LQR_STATE_COUNT; column++)
        {
            float k = k_fit_L0[row][column][LQR_FIT_COEFF_COUNT - 1u];

            for (coefficient = LQR_FIT_COEFF_COUNT - 1u;
                 coefficient > 0u;
                 coefficient--)
            {
                k = (k * l0) + k_fit_L0[row][column][coefficient - 1u];
            }
            if (LQR_IS_FINITE(k) == 0u)
            {
                return 0u;
            }
            gain[row][column] = k;
        }
    }
    return 1u;
}

/* ==========================================================================
 * 三、对外接口
 * ========================================================================== */

/* 单个实例初始化：输入/输出清零，腿长置拟合下限 */
static void lqr_init_one(lqr_t *lqr)
{
    uint8_t i;

    for (i = 0u; i < LQR_STATE_COUNT; i++)
    {
        lqr->x[i] = 0.0f;
        lqr->x_ref[i] = 0.0f;
    }
    lqr->l0_m = LQR_L0_MIN_M;
    lqr->T = 0.0f;
    lqr->Tp = 0.0f;
    lqr->T_cmd = 0;
}

/* 单个实例计算：u = K(L0)·(x_ref − x)，结果写回该实例的输出段。
 * 失败（输入非有限值、腿长非法、结果非有限值）时该实例输出段清零。 */
static void lqr_calc_one(lqr_t *lqr)
{
    float gain[LQR_OUTPUT_COUNT][LQR_STATE_COUNT];
    float error[LQR_STATE_COUNT];
    uint8_t i;
    uint8_t row;

    /* 先清零输出段：后面任何一处失败都直接返回，不会残留上次结果 */
    lqr->T = 0.0f;
    lqr->Tp = 0.0f;
    lqr->T_cmd = 0;

    /* 输入校验 + 误差 e = x_ref − x 一次算完 */
    for (i = 0u; i < LQR_STATE_COUNT; i++)
    {
        if ((LQR_IS_FINITE(lqr->x[i]) == 0u) ||
            (LQR_IS_FINITE(lqr->x_ref[i]) == 0u))
        {
            return;
        }
        error[i] = lqr->x_ref[i] - lqr->x[i];
    }

    /* 按腿长求 K */
    if (lqr_get_gain(lqr->l0_m, gain) == 0u)
    {
        return;
    }

    /* u = K · e */
    for (row = 0u; row < LQR_OUTPUT_COUNT; row++)
    {
        float u = 0.0f;

        for (i = 0u; i < LQR_STATE_COUNT; i++)
        {
            u += gain[row][i] * error[i];
        }
        if (LQR_IS_FINITE(u) == 0u)
        {
            lqr->T = 0.0f;
            lqr->Tp = 0.0f;
            return;
        }
        if (row == 0u)
        {
            lqr->T = u;
        }
        else
        {
            lqr->Tp = u;
        }
    }

    /* T [N·m] -> 3508 电流指令：cmd = T·16384/(20·kt)，并限幅到 LQR_CMD_LIMIT。
     * T 保持力矩原值不动，换算结果单独存进 T_cmd。 */
    {
        float cmd = (lqr->T * LQR_3508_CMD_PER_A) / LQR_3508_KT_NM_PER_A;

        if (cmd > LQR_CMD_LIMIT)
        {
            cmd = LQR_CMD_LIMIT;
        }
        else if (cmd < -LQR_CMD_LIMIT)
        {
            cmd = -LQR_CMD_LIMIT;
        }
        lqr->T_cmd = (int16_t)cmd;
    }
}

static void lqr_output_direction(lqr_t *lqr, float direction)
{
    lqr->T_cmd *= direction;
}

void lqr_init(void)
{
    lqr_init_one(&LQR_Left);
    lqr_init_one(&LQR_Right);
}

void lqr_calc(void)
{
    lqr_calc_one(&LQR_Left);
    lqr_output_direction(&LQR_Left, LQR_T_DIRECTION_LEFT);
    lqr_calc_one(&LQR_Right);
    lqr_output_direction(&LQR_Right, LQR_T_DIRECTION_RIGHT);
}
