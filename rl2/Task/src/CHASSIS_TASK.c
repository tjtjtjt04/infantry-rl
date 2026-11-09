#include "CHASSIS_TASK.h"
#include "ekf.h"              /* EKF.x[0..3] 四元数、EKF.gyro：机体角速度（rad/s）*/
#include "imu_odom.h"         /* 只用其中的轮子标定宏（减速比 / 轮半径）*/
#include "common_function.h"  /* DEG_TO_RAD */
#include "REMOTE_TASK.h"
#include "chassis.h"          /* X-CUBE-AI 生成的网络接口（当前模型 = chassis_getup_tripod） */
#include "chassis_data.h"     /* AI_CHASSIS_DATA_ACTIVATIONS_SIZE */

#include <math.h>
#include <string.h>

/* ============================================================================
 * 策略（X-CUBE-AI）：25 维单帧 × 5 帧历史 = 125 维输入
 * ----------------------------------------------------------------------------
 * 权威定义来自训练仓库：
 *   D:/IsaacLab-main/infantry_rl_s2r/docs/squat_torque_onnx_usage.md
 *   D:/IsaacLab-main/infantry_rl_s2r/exports/chassis_getup_tripod.json
 *   D:/IsaacLab-main/infantry_rl_s2r/mdp/observations.py  proprio_obs()
 *
 *   INPUT : history [1,125]   oldest -> newest，每帧 25 维
 *   OUTPUT: action  [1,6]     原始动作（确定性策略均值，未 clip、未换算）
 *   GRAPH : (history-mean)/denom -> Slice 取当前帧 25 维
 *                                 + encoder(125->128->64->3)
 *                                 -> Concat 28 -> actor -> action 6
 *
 *   归一化器（mean/denom）已经烧进 ONNX，固件送**原始（已乘 scale）**的历史。
 *
 * 单帧 25 维布局（下标是帧内偏移，scale 已包含在下面的赋值里）：
 *   [ 0: 3] base_ang_vel        陀螺 rad/s × 0.25
 *   [ 3: 6] projected_gravity   机体系重力投影（单位向量）
 *   [ 6: 9] command             [vx, vy, wz]，squat 任务恒为 0
 *   [ 9:13] hip_pos_rel_locked  hip_meas - LOCKED_HIPS
 *   [13:19] joint_vel           [4 髋 + 2 轮] 位置差分 rad/s × 0.05
 *   [19:25] last_action         上一拍 clip ±4 后的原始动作
 *
 * 动作后处理（训练侧 mdp/actions.py FiveBarJointAction）：
 *   a       = clamp(action, -4, +4)
 *   q_ref   = clamp(SQUAT_HIPS + a[0:4] * 0.5, limit ± 0.01)   [rad]
 *   tau_hip = 40*(q_ref - q) - 1.0*qdot                        [Nm]，限幅 ±40
 *   v_ref   = a[4:6] * 10.0                                    [rad/s]
 *   tau_whl = clamp(0.5*(v_ref - qdot), ±3.69)                 [Nm]
 *
 * 髋的 PD 不在网络里 —— 网络只给位置目标。固件把 q_ref 换算回电机原始帧塞进
 * MIT 帧的 p_des，kp = 40、kd = 1，由驱动器内部闭环（方案 B）：
 *   tau_motor = kp*(p_des - p_raw) + kd*(v_des - v_raw) + t_ff
 *   p_des - p_raw = sign*(q_ref - q)  ⇒  tau_joint = kp*(q_ref-q) - kd*qdot
 * 与训练完全等价，且比固件 100 Hz 算力矩平滑。chassis_tau[] 仍复算一份只作观测。
 * ========================================================================== */

/* ======================= 控制节拍 ======================= */
#define CHASSIS_CTRL_HZ         (100.0f)                   /* 训练 control_rate_hz = 100 */
#define CHASSIS_CTRL_DT_S       (1.0f / CHASSIS_CTRL_HZ)   /* 0.01 s */

/* ======================= 观测维度 ======================= */
#define CHASSIS_FRAME_DIM       (25)
#define CHASSIS_HIST_LEN        (5)
#define CHASSIS_OBS_DIM         (CHASSIS_FRAME_DIM * CHASSIS_HIST_LEN) /* 125 */

/* ======================= 观测 scale =======================
 * 注意：训练代码 mdp/observations.py 里 joint_vel 用的是 ×0.05，
 * 而 exports/chassis_squat_torque2.json 的 obs_layout 里写的是 0.1（导出脚本
 * 里的注释没同步）。这里以训练代码为准。若实机动作明显异常，先按文档 §4.1
 * 用一个真实帧对一遍 ONNX 输出再怀疑其它环节。
 */
#define CHASSIS_ANG_VEL_SCALE   (0.25f)
#define CHASSIS_JOINT_VEL_SCALE (0.05f)

/* ======================= 动作后处理参数 ======================= */
#define CHASSIS_ACTION_CLIP     (4.0f)   /* PPO clip_actions = 4.0 */
#define CHASSIS_HIP_POS_SCALE   (0.5f)   /* rad / action unit */
#define CHASSIS_WHEEL_VEL_SCALE (10.0f)  /* rad/s / action unit */

#define CHASSIS_HIP_KP          (40.0f)  /* Nm/rad */
#define CHASSIS_HIP_KD          (1.0f)   /* Nm/(rad/s) */
#define CHASSIS_WHEEL_KD        (0.5f)   /* Nm/(rad/s) */
#define CHASSIS_HIP_TAU_LIMIT   (40.0f)  /* Nm */
#define CHASSIS_WHEEL_TAU_LIMIT (3.69f)  /* Nm */

/* 髋电机的**驱动器内部**力矩上限（DM8009P 出厂 TMAX = 54 N·m）。
 * 方案 B 把 PD 交给驱动器后，训练的 clamp(±40 N·m) 只能由驱动器执行，
 * 所以上电时把 TMAX 写成 40（只写 RAM，见 chassis_can_init）。
 * 这个值同时也是 MIT 帧里力矩字段的映射范围，两边必须一致。 */
#define CHASSIS_HIP_TMAX_NM     (40.0f)

/* ======================= 姿态锁定点（squat 位）=======================
 * robot_data/chassis-rl/squat_pose.json，顺序 = HIP_ORDER = (lf, lb, rf, rb)。
 * 既是观测里 hip_pos_rel_locked 的参考点，也是动作 q_ref 的零点（SQUAT_HIPS）：
 * 训练里这两个量是同一个值，a = 0 时髋目标就停在锁定位。
 * ------------------------------------------------------------------------ */
static const float CHASSIS_LOCKED_HIPS[4] = {
    -0.011721f,   /* lf */
    +0.011800f,   /* lb */
    +0.011721f,   /* rf */
    -0.011800f,   /* rb */
};

/* 髋关节 URDF 限位（nominal_pose.json），训练侧对 q_ref 做了 clamp。
 * 训练用的不是原始限位，而是 limits ± limit_margin（mdp/actions.py），
 * 所以这里也必须带同一个 0.01 rad 余量，见下面的 clamp 处。 */
#define CHASSIS_HIP_LIMIT_MARGIN (0.01f)

static const float CHASSIS_HIP_LIM_LO[4] = { -1.488940f, -0.000100f, -0.000100f, -1.489639f };
static const float CHASSIS_HIP_LIM_HI[4] = { +0.000100f, +1.489639f, +1.488940f, +0.000100f };

/* ============================================================================
 * 编号 / 零点 / 极性 —— 全车只有这一处，怀疑方向反了就改这里
 * ----------------------------------------------------------------------------
 * HIP_ORDER（训练顺序）= (lf, lb, rf, rb)
 *
 * 实测接线（hipID 数据；DM 的 CAN ID 0x01~0x04 ↔ dm_motor_data[0..3]）：
 *
 *   CAN ID 1 = 左前 lf   原始量程 -0.17715 ~ +1.3118 rad  (-10.15° ~ +75.16°)
 *   CAN ID 2 = 左后 lb   原始量程 +1.8291  ~ +3.3187 rad  (104.80° ~ 190.15°)
 *   CAN ID 3 = 右后 rb   原始量程 -0.17715 ~ +1.3118 rad
 *   CAN ID 4 = 右前 rf   原始量程 +1.8291  ~ +3.3187 rad
 *
 * lf/rb 共用一副安装角，lb/rf 共用另一副（两组相差约 0.517 rad）。
 * 量程跨度 1.489 rad 恰好等于 URDF 的髋行程，所以每个电机都是一个
 * "零点偏移 + 符号"的线性映射：
 *
 *   q_train = HIP_SIGN[i] * (raw - HIP_RAW_ZERO[i])
 *   raw     = HIP_RAW_ZERO[i] + HIP_SIGN[i] * q_train
 *
 * HIP_RAW_ZERO 是 q_train = 0（URDF/CAD 零位）时该电机的原始读数，落在量程端点：
 *   lf / rb : q=0 在量程高端  -> +1.3118
 *   lb / rf : q=0 在量程低端  -> +1.8291
 * （和 URDF 限位一致：lf/rb 的 hi = +0.0001，lb/rf 的 lo = -0.0001）
 *
 * ★ 上机第一件事（吊起来做）：摆到 squat 位，读 dm_motor_data[0..3].measure.pos。
 *   下面 HIP_CAL_RAW_INCREASES_WITH_Q = 1 时（原始读数随 q 增大），四个读数应约为
 *       lf(DM1) +1.3001   lb(DM2) +1.8410   rb(DM3) +1.3000   rf(DM4) +1.8409
 *   且换算结果必须是 (lf,lb,rf,rb) = (-0.011721, +0.0118, +0.011721, -0.0118)。
 *   若实测是"读数随 q 减小"（读数量程落在 -0.165 / +3.307 附近），
 *   把 HIP_CAL_RAW_INCREASES_WITH_Q 改成 0 —— 符号和零点会一起换，不用改别处。
 *
 * 观测用 HIP_SIGN 把原始角度换成训练符号；下发时用**同一个映射**把关节目标
 * 反算回电机原始帧（镜像安装 ⇒ 角度取反 ⇒ 位置/力矩同样取反）。
 * ========================================================================== */
#define HIP_CAL_RAW_INCREASES_WITH_Q   (1)   /* 1: raw 随 q 增大；0: raw 随 q 减小 */

#define HIP_DM_LF               (0U)   /* CAN 0x01 */
#define HIP_DM_LB               (1U)   /* CAN 0x02 */
#define HIP_DM_RF               (3U)   /* CAN 0x04 */
#define HIP_DM_RB               (2U)   /* CAN 0x03 */

#if (HIP_CAL_RAW_INCREASES_WITH_Q != 0)
  #define HIP_SIGN_LF           (+1.0f)
  #define HIP_SIGN_LB           (+1.0f)
  #define HIP_SIGN_RF           (+1.0f)
  #define HIP_SIGN_RB           (+1.0f)
  #define HIP_RAW_ZERO_LF       (+1.3118f)
  #define HIP_RAW_ZERO_LB       (+1.8291f)
  #define HIP_RAW_ZERO_RF       (+1.8291f)
  #define HIP_RAW_ZERO_RB       (+1.3118f)
#else
  #define HIP_SIGN_LF           (-1.0f)
  #define HIP_SIGN_LB           (-1.0f)
  #define HIP_SIGN_RF           (-1.0f)
  #define HIP_SIGN_RB           (-1.0f)
  #define HIP_RAW_ZERO_LF       (-0.1771f)
  #define HIP_RAW_ZERO_LB       (+3.3188f)
  #define HIP_RAW_ZERO_RF       (+3.3188f)
  #define HIP_RAW_ZERO_RB       (-0.1771f)
#endif

/* 轮子：训练里 wheel joint 的正方向 —— 左轮 URDF 正 = 向后滚，右轮正 = 向前滚。
 *
 * 两台 3508 的出轴方向相反（实机确认），所以"同一电流符号"在物理上就是
 * 两个相反方向 —— 正好对上训练里左右轮关节正方向相反这件事，因此两轮的
 * 电流符号取同一个值。若整车上电后"给正速度目标反而后退"，把这两个一起取反。 */
#define WHEEL_JOINT_SIGN_LEFT   (-1.0f)  /* 前进 => 关节负方向 */
#define WHEEL_JOINT_SIGN_RIGHT  (+1.0f)  /* 前进 => 关节正方向 */
#define WHEEL_CURR_SIGN_LEFT    (-1.0f)  /* 关节正力矩 => 负电流指令 */
#define WHEEL_CURR_SIGN_RIGHT   (-1.0f)  /* 出轴反向安装：与左侧同号 */

/* C620 电流指令换算：16384/20 = 819.2 counts/A（±20 A 对应 ±16384） */
#define C620_COUNTS_PER_AMP     (819.2f)
#define C620_CURRENT_LIMIT      (8192.0f)  /* ±10 A */

/* 轮轴侧力矩常数 Kt_eff = Kt_motor × 减速比 × 效率。
 * 3508 的 0.246 N·m/A 是**转子侧**的，直接拿去换算电流会小 19 倍，
 * 必须乘减速比折算到轮轴。台架标定后改这几个数。 */
#define CHASSIS_KT_MOTOR        (0.246f)
#define CHASSIS_WHEEL_EFF       (0.85f)
#define CHASSIS_KT_EFF_WHEEL    (CHASSIS_KT_MOTOR * ODOM_GEAR_RATIO * CHASSIS_WHEEL_EFF)

/* ======================= 调试用全局量 ======================= */
float chassis_dt_s;

/** 当前帧 25 维观测（已乘 scale），调试器可直接看 */
float chassis_obs[CHASSIS_FRAME_DIM];

/** 125 维历史（oldest -> newest），即网络输入 */
float chassis_hist[CHASSIS_OBS_DIM];

/** 网络原始输出 / clip ±4 后的动作 / 关节力矩 Nm */
float chassis_action_raw[6];
float chassis_action[6];
float chassis_tau[6];

/** 关节反馈（训练符号）：4 髋 rad + 2 轮 rad/s */
float chassis_hip_pos[4];
float chassis_joint_vel[6];

/** PD 阻尼用的髋角速度（取 DM 内部估算值，比 100 Hz 差分平滑） */
float chassis_hip_vel_pd[4];

/** 下发量：髋 MIT 位置给定（电机原始帧，= zero + sign*q_ref）/ 轮 C620 电流给定 raw。
 *  关节力矩 chassis_tau[] 仍然每拍复算一份，只给调试器看（固件不再用它下发）。 */
float chassis_hip_pos_cmd[4];
int16_t chassis_wheel_current_cmd[2];

/* ======================= 内部状态 ======================= */
static ai_handle chassis_ai = AI_HANDLE_NULL;
static ai_buffer *chassis_ai_in = NULL;
static ai_buffer *chassis_ai_out = NULL;
static uint8_t chassis_ai_ready = 0U;

/* 网络激活量缓冲：尺寸由 X-CUBE-AI 生成（当前 1268 B），别硬编码旧值 */
static ai_u8 chassis_activations[AI_CHASSIS_DATA_ACTIVATIONS_SIZE]
    __attribute__((aligned(4)));

static float chassis_q_prev[4];           /* 上一拍髋角，做位置差分 */
static uint8_t chassis_q_prev_valid = 0U;
static uint8_t chassis_hist_primed = 0U;  /* 上电首帧用当前帧填满整段历史 */
static float chassis_ctrl_acc = 0.0f;     /* 100 Hz 分频累积 */
static float chassis_step_dt_s = CHASSIS_CTRL_DT_S;  /* 本拍实测周期（DWT） */

/* ======================= 按 HIP_ORDER 展开的常数表 =======================
 * 顺序固定为 (lf, lb, rf, rb)，和训练一致；下标只在这里出现一次。 */
static const uint32_t CHASSIS_HIP_DM[4]   = { HIP_DM_LF, HIP_DM_LB, HIP_DM_RF, HIP_DM_RB };
static const float    CHASSIS_HIP_SIGN[4] = { HIP_SIGN_LF, HIP_SIGN_LB, HIP_SIGN_RF, HIP_SIGN_RB };
static const float    CHASSIS_HIP_ZERO[4] = { HIP_RAW_ZERO_LF, HIP_RAW_ZERO_LB,
                                              HIP_RAW_ZERO_RF, HIP_RAW_ZERO_RB };

/* ======================= CAN 初始化 ======================= */
static void chassis_can_init(void)
{
    FDCAN_FilterTypeDef filter;
    uint32_t i;

    /* 1) FDCAN1：全接收（mask=0 表示所有标准帧都收） */
    can_filter_std(&filter, 0x000U, 0x000U, 0U);
    app_can1_filter_init(&filter);

    /* 2) FDCAN3：全接收 */
    can_filter_std(&filter, 0x000U, 0x000U, 0U);
    app_can3_filter_init(&filter);

    /* 3) FDCAN3 上注册 DM 电机：CAN ID / Master ID 全部来自 app_can.c 配置表 */
    app_dm_motor_init_all(&hfdcan3);

    /* 4) 底盘驱动轮电流帧（0x200）挂 FDCAN1（DJI 反馈也在这条总线上） */
    can_dev_chassis.hcan = &hfdcan1;
    can_dev_chassis.std_id = CAN_CHASSIS_ALL_ID;

    /* 5) 把 4 台髋电机的内部力矩上限写成 40 N·m（DM 参数帧 RID=23 TMAX）
     *    DM8009P 出厂 TMAX = 54，方案 B 之后 ±40 的限幅只能由驱动器执行。
     *    只发 0x55（写 RAM），不发 0xAA（写 flash）—— 每次上电写一遍即可，
     *    既保证生效又不会磨损电机参数区。
     *    同时把固件侧的 t_min/t_max 改成一致：MIT 帧的力矩字段映射和
     *    measure.tor 的反算都按这个范围。 */
    for (i = 0U; i < APP_DM_MOTOR_CNT; i++)
    {
        app_dm_write_param_float(&dm_motor_data[i], APP_DM_RID_TMAX, CHASSIS_HIP_TMAX_NM);
        dm_motor_data[i].t_max = CHASSIS_HIP_TMAX_NM;
        dm_motor_data[i].t_min = -CHASSIS_HIP_TMAX_NM;
    }

    /* 6) 等待总线稳定，避免 DM 电机上电后立即发的帧被丢弃 */
    HAL_Delay(500U);
    app_dm_enable_all(APP_DM_MODE_MIT);
}

/* ======================= 工具函数 ======================= */
static float chassis_clampf(float x, float lo, float hi)
{
    if (x > hi) { return hi; }
    if (x < lo) { return lo; }
    return x;
}

/** 由 EKF 四元数（body->world，q = (w, x, y, z)）计算机体系重力投影 */
static void chassis_projected_gravity(const float32_t q[4], float out[3])
{
    const float w = q[0];
    const float x = q[1];
    const float y = q[2];
    const float z = q[3];

    out[0] =  2.0f * ((w * y) - (x * z));
    out[1] = -2.0f * ((w * x) + (y * z));
    out[2] =  2.0f * ((x * x) + (y * y)) - 1.0f;
}

/** X-CUBE-AI 网络初始化：把 activations 交给 runtime，取输入/输出 buffer */
static void chassis_ai_init(void)
{
    const ai_handle activations[1] = { (ai_handle)chassis_activations };
    ai_error ai_err = ai_chassis_create_and_init(&chassis_ai, activations, NULL);

    if (ai_err.type != AI_ERROR_NONE)
    {
        return;   /* 失败不阻塞任务：观测照常拼装，只是不下发力矩 */
    }

    chassis_ai_in  = ai_chassis_inputs_get(chassis_ai, NULL);
    chassis_ai_out = ai_chassis_outputs_get(chassis_ai, NULL);

    if ((chassis_ai_in == NULL) || (chassis_ai_out == NULL))
    {
        return;
    }

    /* 维度必须和固件缓冲一致，不一致宁可不下发，也不要把内存写坏 */
    if ((AI_BUFFER_SIZE(chassis_ai_in) != (ai_size)(CHASSIS_OBS_DIM * sizeof(float))) ||
        (AI_BUFFER_SIZE(chassis_ai_out) != (ai_size)(6U * sizeof(float))))
    {
        return;
    }

    chassis_ai_ready = 1U;
}

/* ============================================================================
 * 一个 100 Hz 控制步：采关节 -> 拼 25 维 -> 压历史 -> 推理 -> PD -> 下发
 * ========================================================================== */
static void chassis_control_step(uint8_t enable_output)
{
    uint32_t i;
    float frame[CHASSIS_FRAME_DIM];
    float grav[3];

    /* ---------- 1. 关节反馈，转成训练符号 ---------- */
    for (i = 0U; i < 4U; i++)
    {
        const float raw = dm_motor_data[CHASSIS_HIP_DM[i]].measure.pos;

        chassis_hip_pos[i] = CHASSIS_HIP_SIGN[i] * (raw - CHASSIS_HIP_ZERO[i]);
    }

    /* 髋速度：训练用 position_diff（不是驱动器内部速度），dt 用本拍实测值。
     * 100 Hz 是从 1 kHz 用"减一个周期"分频出来的，单拍真实间隔会在 9~12 ms
     * 之间抖一下；固定除 0.01 s 会把那一拍的速度放大成 ~2 倍尖峰，所以这里
     * 必须用 DWT 实测的 chassis_step_dt_s（见 chassis_task）。 */
    for (i = 0U; i < 4U; i++)
    {
        if (chassis_q_prev_valid != 0U)
        {
            chassis_joint_vel[i] = (chassis_hip_pos[i] - chassis_q_prev[i]) / chassis_step_dt_s;
        }
        else
        {
            chassis_joint_vel[i] = 0.0f;
        }
        chassis_q_prev[i] = chassis_hip_pos[i];
    }

    /* PD 阻尼单独用 DM 内部速度：训练里 tau 的 kd 项取的是仿真关节速度（平滑），
     * 而 100 Hz 的编码器差分噪声会被 kd 放大成力矩抖动。 */
    for (i = 0U; i < 4U; i++)
    {
        chassis_hip_vel_pd[i] = CHASSIS_HIP_SIGN[i]
                              * dm_motor_data[CHASSIS_HIP_DM[i]].measure.vel;
    }

    /* 轮速度：DJI 反馈 rpm -> 前进 m/s（imu_odom 里已标定）-> 关节 rad/s */
    chassis_joint_vel[4] = WHEEL_JOINT_SIGN_LEFT  * ODOM.wheel_speed_left
                         / ODOM_WHEEL_RADIUS_M;
    chassis_joint_vel[5] = WHEEL_JOINT_SIGN_RIGHT * ODOM.wheel_speed_right
                         / ODOM_WHEEL_RADIUS_M;

    /* ---------- 2. 拼 25 维单帧 ---------- */
    chassis_projected_gravity(EKF.x, grav);

    for (i = 0U; i < 3U; i++)
    {
        frame[i]      = EKF.gyro[i] * CHASSIS_ANG_VEL_SCALE;  /* base_ang_vel */
        frame[3U + i] = grav[i];                              /* projected_gravity */
        frame[6U + i] = 0.0f;                                 /* command：squat 恒 0 */
    }

    for (i = 0U; i < 4U; i++)
    {
        frame[9U + i] = chassis_hip_pos[i] - CHASSIS_LOCKED_HIPS[i];
    }

    for (i = 0U; i < 6U; i++)
    {
        frame[13U + i] = chassis_joint_vel[i] * CHASSIS_JOINT_VEL_SCALE;
        frame[19U + i] = chassis_action[i];                   /* last_action */
    }

    memcpy(chassis_obs, frame, sizeof(frame));

    /* ---------- 3. 压入 125 维历史（oldest -> newest）---------- */
    if (chassis_hist_primed == 0U)
    {
        /* 上电/复位：用当前帧填满 5 帧，避免历史里出现一串 0 */
        for (i = 0U; i < (uint32_t)CHASSIS_HIST_LEN; i++)
        {
            memcpy(&chassis_hist[i * CHASSIS_FRAME_DIM], frame, sizeof(frame));
        }
        chassis_hist_primed = 1U;
    }
    else
    {
        memmove(chassis_hist,
                &chassis_hist[CHASSIS_FRAME_DIM],
                (CHASSIS_OBS_DIM - CHASSIS_FRAME_DIM) * sizeof(float));
        memcpy(&chassis_hist[CHASSIS_OBS_DIM - CHASSIS_FRAME_DIM], frame, sizeof(frame));
    }

    /* ---------- 4. 推理 ---------- */
    if (chassis_ai_ready != 0U)
    {
        memcpy(AI_BUFFER_DATA(chassis_ai_in, float), chassis_hist,
               CHASSIS_OBS_DIM * sizeof(float));

        if (ai_chassis_run(chassis_ai, chassis_ai_in, chassis_ai_out) > 0)
        {
            const float *action = AI_BUFFER_DATA(chassis_ai_out, float);
            memcpy(chassis_action_raw, action, 6U * sizeof(float));
        }
    }

    /* ---------- 5. 动作后处理 ---------- */
    for (i = 0U; i < 6U; i++)
    {
        chassis_action[i] = chassis_clampf(chassis_action_raw[i],
                                           -CHASSIS_ACTION_CLIP, CHASSIS_ACTION_CLIP);
    }

    /* 髋：位置目标 + PD -> 力矩（力矩只做复算观测，真正下发的是位置目标，见第 6 步） */
    for (i = 0U; i < 4U; i++)
    {
        float q_ref = CHASSIS_LOCKED_HIPS[i] + chassis_action[i] * CHASSIS_HIP_POS_SCALE;
        float tau;

        /* 训练侧 clamp 到 URDF 限位 ±limit_margin，不带这个余量会多放 10 mrad */
        q_ref = chassis_clampf(q_ref,
                               CHASSIS_HIP_LIM_LO[i] + CHASSIS_HIP_LIMIT_MARGIN,
                               CHASSIS_HIP_LIM_HI[i] - CHASSIS_HIP_LIMIT_MARGIN);

        /* 位置目标换回电机原始帧：raw = zero + sign*q。DM 内部 PD 算的是
         * kp*(p_des - p_raw)，两边同乘 sign 正好还原成训练里的
         * kp*(q_ref - q) - kd*qdot，所以 kp/kd 直接填 40 / 1，不取反。 */
        chassis_hip_pos_cmd[i] = CHASSIS_HIP_ZERO[i] + CHASSIS_HIP_SIGN[i] * q_ref;

        tau = CHASSIS_HIP_KP * (q_ref - chassis_hip_pos[i])
            - CHASSIS_HIP_KD * chassis_hip_vel_pd[i];

        chassis_tau[i] = chassis_clampf(tau, -CHASSIS_HIP_TAU_LIMIT, CHASSIS_HIP_TAU_LIMIT);
    }

    /* 轮：速度目标 + 阻尼 -> 力矩 */
    for (i = 0U; i < 2U; i++)
    {
        float v_ref = chassis_action[4U + i] * CHASSIS_WHEEL_VEL_SCALE;
        float tau;

        tau = CHASSIS_WHEEL_KD * (v_ref - chassis_joint_vel[4U + i]);

        chassis_tau[4U + i] = chassis_clampf(tau, -CHASSIS_WHEEL_TAU_LIMIT,
                                             CHASSIS_WHEEL_TAU_LIMIT);
    }

    /* ---------- 6. 下发 ---------- */
    if (enable_output != 0U)
    {
        float i_cmd;
        int16_t i_left;
        int16_t i_right;

        /* 髋：方案 B —— 位置目标 + kp/kd 直接进 MIT 帧，PD 由驱动器内部闭环
         * （1~2 kHz），比固件 100 Hz 算力矩平滑得多，也正好对应训练里
         * "position target -> PD -> torque" 的语义。
         *   p_des = chassis_hip_pos_cmd[]（电机原始帧，已 clamp）
         *   v_des = 0，t_ff = 0，kp = 40 N·m/rad，kd = 1 N·m/(rad/s)
         * app_can.c 的 app_dm_cmd_mit 负责无符号偏移映射；
         * 范围检查：p ±12.5 rad、kp 0~500、kd 0~5 —— 40/1 都在范围内。 */
        for (i = 0U; i < 4U; i++)
        {
            app_dm_cmd_mit(&dm_motor_data[CHASSIS_HIP_DM[i]],
                           chassis_hip_pos_cmd[i],   /* p_des */
                           0.0f,                     /* v_des */
                           CHASSIS_HIP_KP,           /* kp */
                           CHASSIS_HIP_KD,           /* kd */
                           0.0f);                    /* t_ff */
        }

        /* 轮：关节力矩 -> 电流（Kt_eff 为轮轴侧折算值，见文件头） */
        i_cmd = WHEEL_CURR_SIGN_LEFT * chassis_tau[4] / CHASSIS_KT_EFF_WHEEL * C620_COUNTS_PER_AMP;
        i_cmd = chassis_clampf(i_cmd, -C620_CURRENT_LIMIT, C620_CURRENT_LIMIT);
        i_left = (int16_t)i_cmd;

        i_cmd = WHEEL_CURR_SIGN_RIGHT * chassis_tau[5] / CHASSIS_KT_EFF_WHEEL * C620_COUNTS_PER_AMP;
        i_cmd = chassis_clampf(i_cmd, -C620_CURRENT_LIMIT, C620_CURRENT_LIMIT);
        i_right = (int16_t)i_cmd;

        chassis_wheel_current_cmd[0] = i_left;
        chassis_wheel_current_cmd[1] = i_right;

        /* 0x200 帧：m1 = 左轮 M1(0x201)，m2 = 右轮 M2(0x202) */
        app_can_cmd_chassis(&can_dev_chassis, i_left, i_right, 0, 0);
    }
    else
    {
        /* 关：零力矩 + 零电流。注意零力矩不是安全态（没有隐式阻尼），
         * 真机上应该切到 tau = -kd*qdot 的阻尼输出或抱闸。 */
        for (i = 0U; i < APP_DM_MOTOR_CNT; i++)
        {
            app_dm_cmd_mit(&dm_motor_data[i], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        }
        chassis_wheel_current_cmd[0] = 0;
        chassis_wheel_current_cmd[1] = 0;
        app_can_cmd_chassis(&can_dev_chassis, 0, 0, 0, 0);
    }

    chassis_q_prev_valid = 1U;
}

/**
 * @brief CHASSIS_TASK 任务函数
 * @param argument 未使用
 */
void chassis_task(void *argument)
{
    uint32_t last_cycle;
    uint32_t last_ctrl_cycle;

    (void)argument;

    /* CAN 初始化必须在读 DM 反馈之前完成 */
    osDelay(500U);
    chassis_can_init();

    /* 里程计必须先 init：imu_odom_update() 在 ready == 0 时直接返回，
     * 不初始化的话 wheel_speed_left/right 会一直是 0 —— 轮速观测全错，
     * 轮子 PD 也退化成纯前馈（没有阻尼）。 */
    imu_odom_init(&ODOM);

    /* 上电先清空历史与动作，避免把上一轮的残留喂进网络 */
    (void)memset(chassis_hist, 0, sizeof(chassis_hist));
    (void)memset(chassis_action, 0, sizeof(chassis_action));
    (void)memset(chassis_action_raw, 0, sizeof(chassis_action_raw));
    chassis_hist_primed = 0U;
    chassis_q_prev_valid = 0U;

    chassis_ai_init();

    /*
     * DWT 精确计时：
     *   - DWT_Init() 已由 INS_TASK 完成（INS 优先级更高，先运行）；
     *   - 直接读 DWT->CYCCNT，只在本任务内维护 last_cycle，避免和
     *     bsp_dwt 的共享 static 计数互相打架；
     *   - 无符号相减可自然处理 32 位 CYCCNT 回绕。
     */
    last_cycle = DWT->CYCCNT;
    last_ctrl_cycle = last_cycle;

    while (1)
    {
        uint32_t now_cycle = DWT->CYCCNT;

        /* 两次循环之间的真实时间（s）：CPU 主频计数差 / CPU 主频 */
        chassis_dt_s = (float)(now_cycle - last_cycle) / (float)SystemCoreClock;
        last_cycle = now_cycle;

        /* 100 Hz 分频：用减法而不是清零，避免累积漂移 */
        chassis_ctrl_acc += chassis_dt_s;

        if (chassis_ctrl_acc >= CHASSIS_CTRL_DT_S)
        {
            chassis_ctrl_acc -= CHASSIS_CTRL_DT_S;

            /* 本拍实测周期：由于分频是"累积到 10 ms 才走一步"，单拍真实间隔会
             * 在 1 kHz 量化后的 9~12 ms 之间抖，取两次控制步之间的 DWT 差值，
             * 位置差分和里程计都用它，避免出现 2× 速度尖峰。 */
            chassis_step_dt_s = (float)(now_cycle - last_ctrl_cycle)
                              / (float)SystemCoreClock;
            last_ctrl_cycle = now_cycle;

            /* 兜底：DWT 异常/任务被长时间抢占时不要把速度算疯 */
            if (chassis_step_dt_s < 0.5f * CHASSIS_CTRL_DT_S)
            {
                chassis_step_dt_s = 0.5f * CHASSIS_CTRL_DT_S;
            }
            else if (chassis_step_dt_s > 1.5f * CHASSIS_CTRL_DT_S)
            {
                chassis_step_dt_s = 1.5f * CHASSIS_CTRL_DT_S;
            }

            /* 轮速由 imu_odom 换算（减速比 / 轮径 / rpm 符号已在那里面标定），
             * dt 用本拍实测值，和训练的 position_diff 口径一致 */
            imu_odom_update(&ODOM,
                    EKF.accel, EKF.gyro, EKF.roll, EKF.pitch,
                    DJI_M2_MEASURE.speed_rpm,   /* 右轮 0x202 */
                    DJI_M1_MEASURE.speed_rpm,   /* 左轮 0x201 */
                    DJI_M2_MEASURE.ecd,
                    DJI_M1_MEASURE.ecd,
                    chassis_step_dt_s);

            chassis_control_step(remote_ch6_switch_on());
        }

        /* 1 kHz 循环：给 DM 反馈帧留出总线仲裁机会。
         * （全速发送时 0x01~0x04 指令 ID 优先级高于反馈帧 0x11~0x14，
         *   会一直抢占总线导致 DM 反馈收不到） */
        osDelay(1U);
    }
}
