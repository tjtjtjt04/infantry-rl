#ifndef APP_CAN_H
#define APP_CAN_H

#include "fdcan.h"
#include "main.h"

/* ==================== 文件说明 ====================
 * STM32H7 FDCAN 版电机接口：
 *   - DJI 电机（FDCAN1 / FDCAN2）：反馈帧 0x201~0x208，控制帧 0x200 / 0x1FE / 0x205...
 *   - DM（达妙）电机（可挂任意 FDCAN）：MIT / 位置速度模式控制 + 反馈解析
 *
 * 典型用法：
 *   FDCAN_FilterTypeDef filter;
 *   can_filter_std(&filter, 0x000U, 0x000U, 0U);   // 全接收
 *   app_can1_filter_init(&filter);                  // 配置滤波器 + 启动总线 + 开接收中断
 *
 *   // 反馈：直接读全局结构体（见第 3 节）
 *   DJI_M1_MEASURE.ecd / DJI_M2_MEASURE.speed_rpm
 *   DM_M1_MEASURE.pos  ...  DM_M4_MEASURE.pos
 *
 *   // 发送
 *   app_can_cmd_chassis(&can_dev_chassis, m1, m2, m3, m4);
 *   app_dm_cmd_mit(&app_dm_motor[0], p_des, v_des, kp, kd, t_ff);
 * ================================================== */

/* ==================== 1. 公共类型 ==================== */

/* CAN 总线枚举（DJI 反馈槽位按总线区分） */
typedef enum
{
    APP_CAN1 = 0,
    APP_CAN2 = 1,
} app_can_bus_e;

/* DJI 反馈缓存的总线数（FDCAN1 / FDCAN2） */
#define APP_CAN_BUS_NUM 2

/* DJI 电机反馈数据结构 */
typedef struct
{
    uint16_t ecd;            /* 编码器值 */
    int16_t  speed_rpm;      /* 转速（rpm） */
    int16_t  given_current;  /* 实际电流（原始值） */
    uint8_t  temperate;      /* 温度（℃） */
    int16_t  last_ecd;       /* 上一次编码器值 */
    uint8_t  error;          /* 错误码 */
} motor_measure_t __attribute__((aligned(4)));

/* DJI CAN 标准帧 ID */
typedef enum
{
    CAN_CHASSIS_ALL_ID = 0x200,   /* 底盘四电机控制帧 */
    CAN_3508_M1_ID = 0x201,       /* 底盘电机 1 反馈 */
    CAN_3508_M2_ID = 0x202,       /* 底盘电机 2 反馈 */
    CAN_3508_M3_ID = 0x203,       /* 底盘电机 3 反馈 */
    CAN_3508_M4_ID = 0x204,       /* 底盘电机 4 反馈 */
    CAN_YAW_MOTOR_ID = 0x205,     /* yaw 电机反馈 */
    CAN_PIT_MOTOR_ID = 0x206,     /* pitch 电机反馈 */
    CAN_TRIGGER_MOTOR_ID = 0x203, /* 拨盘电机反馈 */
    CAN_GIMBAL_ALL_ID = 0x1FE,    /* 云台控制帧 */
} can_msg_id_e;

/* 设备槽位下标 = 标准帧 ID - 0x201（0x201~0x208 -> 0~7） */
enum
{
    APP_IDX_M1   = CAN_3508_M1_ID       - CAN_3508_M1_ID, /* 0x201 -> 0 */
    APP_IDX_M2   = CAN_3508_M2_ID       - CAN_3508_M1_ID, /* 0x202 -> 1 */
    APP_IDX_M3   = CAN_3508_M3_ID       - CAN_3508_M1_ID, /* 0x203 -> 2 */
    APP_IDX_M4   = CAN_3508_M4_ID       - CAN_3508_M1_ID, /* 0x204 -> 3 */
    APP_IDX_YAW  = CAN_YAW_MOTOR_ID     - CAN_3508_M1_ID, /* 0x205 -> 4 */
    APP_IDX_PIT  = CAN_PIT_MOTOR_ID     - CAN_3508_M1_ID, /* 0x206 -> 5 */
    APP_IDX_TRIG = CAN_TRIGGER_MOTOR_ID - CAN_3508_M1_ID, /* 0x203 -> 2 */
};

/* 设备发送配置（挂载的总线句柄 + 发送 ID） */
typedef struct
{
    FDCAN_HandleTypeDef *hcan;  /* 设备挂载的 FDCAN 总线句柄 */
    uint32_t             std_id; /* 发送标准帧 ID */
} can_tx_device_t;

/* 总线原始接收帧（调试用） */
#define APP_CAN_RX_LOG_NUM 16U

typedef struct
{
    uint32_t tick_ms;   /* 收到帧时的 HAL_GetTick()（ms） */
    uint16_t id;        /* 标准帧 ID */
    uint8_t  dlc;       /* DLC 编码（8 字节数据 = 8） */
    uint8_t  data[8];   /* 原始 8 字节 */
} can_rx_frame_t;

/* ==================== 2. DM 电机（达妙）协议与类型 ====================
 * 经典 CAN、标准帧、8 字节：
 *
 *   控制帧（MIT 模式）   帧 ID = CAN ID
 *       D[0..1] = p_des（16 位，高字节在前）
 *       D[2]    = v_des[11:4]
 *       D[3]    = v_des[3:0]<<4 | Kp[11:8]
 *       D[4]    = Kp[7:0]
 *       D[5]    = Kd[11:4]
 *       D[6]    = Kd[3:0]<<4 | t_ff[11:8]
 *       D[7]    = t_ff[7:0]
 *   控制帧（位置速度模式）帧 ID = 0x100 + CAN ID
 *       D[0..3] = p_des（float，低位在前）  D[4..7] = v_des（float，低位在前）
 *   反馈帧               帧 ID = Master ID（MST_ID），8 字节
 *       D[0]    = 控制器 ID（低 4 位）| 故障码（高 4 位）
 *       D[1..2] = POS[15:0]     D[3] = VEL[11:4]
 *       D[4]    = VEL[3:0]<<4 | T[11:8]   D[5] = T[7:0]
 *       D[6]    = T_MOS（℃）    D[7] = T_Rotor（℃）
 *
 * 位置/速度/扭矩是"无符号偏移映射"（达妙官方 MIT 协议）：
 *   u = (x - x_min) / (x_max - x_min) * (2^N - 1)
 *   x_min/x_max 用该电机的 P_MIN/P_MAX、V_MIN/V_MAX、T_MIN/T_MAX（默认 ±12.5 / ±30 / ±10）；
 *   例：扭矩 T_MAX = 10 时 0 N·m -> 2047、+10 -> 4095、-10 -> 0；
 *   Kp/Kd 是 0~max 直接映射。换电机型号时改 dm_motor_data[i].p_max / v_max / t_max。
 *
 * 反馈分发规则（接收回调里自动处理）：
 *   - mst_id != 0：帧 ID == mst_id 的反馈帧交给该电机（多台电机请设置不同 mst_id）；
 *   - mst_id == 0：帧 ID == 0 时按 D[0] 低 4 位 == CAN ID 低 4 位匹配。
 *   - Master ID 不要设在 0x201~0x208（会被当作 DJI 反馈解析）。
 * ============================================================ */

/* 控制帧 ID 的模式偏移 */
#define APP_DM_MODE_MIT      ((uint16_t)0x000U)  /* MIT 模式：帧 ID = CAN ID */
#define APP_DM_MODE_POS_VEL  ((uint16_t)0x100U)  /* 位置速度模式：帧 ID = 0x100 + CAN ID */

/* ==================== 电机映射范围（只改 *_MAX，MIN 自动取负） ====================
 * 换电机型号时只改下面 5 个 MAX 值（默认按 DM4310-2EC）：
 *   P_MAX  位置范围 ±P_MAX   (rad)
 *   V_MAX  速度范围 ±V_MAX   (rad/s)
 *   T_MAX  力矩范围 ±T_MAX   (N·m)
 *   KP_MAX Kp 范围 0 ~ KP_MAX
 *   KD_MAX Kd 范围 0 ~ KD_MAX
 * ============================================================================== */
#define APP_DM_DEFAULT_P_MAX   ( 12.5f)
#define APP_DM_DEFAULT_V_MAX   ( 45.0f)
#define APP_DM_DEFAULT_T_MAX   ( 54.0f)
#define APP_DM_DEFAULT_KP_MAX  ( 500.0f)
#define APP_DM_DEFAULT_KD_MAX  ( 5.0f)

/* 下限自动推导：位置/速度/力矩对称，Kp/Kd 从 0 开始 */
#define APP_DM_DEFAULT_P_MIN   (-APP_DM_DEFAULT_P_MAX)
#define APP_DM_DEFAULT_V_MIN   (-APP_DM_DEFAULT_V_MAX)
#define APP_DM_DEFAULT_T_MIN   (-APP_DM_DEFAULT_T_MAX)
#define APP_DM_DEFAULT_KP_MIN  ( 0.0f)
#define APP_DM_DEFAULT_KD_MIN  ( 0.0f)

/* 本模块管理的 DM 电机槽位数（所有编译单元保持一致） */
#ifndef APP_DM_MOTOR_NUM
#define APP_DM_MOTOR_NUM 4
#endif

/* ==================== DM 电机配置表（CAN ID / Master ID） ====================
 * 电机 ID 统一在本模块维护，任务/上层代码不再保存 DM 电机 ID：
 *   app_can.c 里的 app_dm_motor_can_id[] / app_dm_motor_mst_id[] 就是配置表，
 *   索引 i 对应 dm_motor_data[i]，app_dm_motor_init_all() 按表注册全部电机。
 *
 *   can_id : 控制帧 ID（MIT 模式帧 ID = can_id；0 = 该槽位未使用）
 *   mst_id : 反馈帧 ID（Master ID，必须与电机上位机/DM 调试工具里设置一致）
 *
 * 改电机 ID / 电机数量时只改 app_can.c 的这两个数组和 APP_DM_MOTOR_CNT，
 * 不要改任务文件（CHASSIS_TASK.c 等）。
 * ======================================================================== */
#define APP_DM_MOTOR_CNT 4U   /* 配置表里实际使用的电机数量（不能大于 APP_DM_MOTOR_NUM） */

extern const uint16_t app_dm_motor_can_id[APP_DM_MOTOR_CNT];
extern const uint16_t app_dm_motor_mst_id[APP_DM_MOTOR_CNT];

/* DM 电机反馈数据（反馈帧解包结果） */
typedef struct
{
    float    pos;       /* 位置（rad） */
    float    vel;       /* 速度（rad/s） */
    float    tor;       /* 转矩（N·m） */
    float    t_mos;     /* 驱动板 MOS 平均温度（℃） */
    float    t_rotor;   /* 电机线圈平均温度（℃） */
    uint16_t pos_int;   /* 位置原始码值（16 位无符号偏移映射） */
    uint16_t vel_int;   /* 速度原始码值（12 位无符号偏移映射） */
    uint16_t tor_int;   /* 转矩原始码值（12 位无符号偏移映射） */
    uint8_t  id;        /* 控制器 ID（反馈帧 D[0] 低 4 位） */
    uint8_t  error;     /* 故障码（反馈帧 D[0] 高 4 位原始值） */
    volatile uint8_t  updated;       /* 收到过反馈置 1（只由 init 清零） */
    volatile uint32_t rx_cnt;        /* 收到反馈帧总计数（诊断用，持续累加） */
    volatile uint32_t last_tick_ms;  /* 最近一帧反馈的 HAL_GetTick()（ms，超时判断用） */
} dm_motor_measure_t;

/* 一台 DM 电机的配置 + 反馈 */
typedef struct
{
    FDCAN_HandleTypeDef *hfdcan;  /* 挂载的 FDCAN 总线句柄 */
    uint16_t can_id;              /* 电机 CAN ID（0 = 该槽位未使用） */
    uint16_t mst_id;              /* 反馈帧 ID（Master ID） */
    float    p_min;               /* 位置范围 */
    float    p_max;
    float    v_min;               /* 速度范围 */
    float    v_max;
    float    t_min;               /* 转矩范围 */
    float    t_max;
    float    kp_min;              /* MIT 模式 Kp 范围 */
    float    kp_max;
    float    kd_min;              /* MIT 模式 Kd 范围 */
    float    kd_max;
    dm_motor_measure_t measure;   /* 反馈数据（只读） */
} dm_motor_t;

/* ==================== 3. 全局数据（直接读，不使用 getter） ==================== */

/* ---------- 3.1 DJI 电机反馈（FDCAN1 / FDCAN2） ---------- */
extern motor_measure_t dji_motor_data[APP_CAN_BUS_NUM][8];   /* [总线][ID - 0x201] */
extern volatile uint32_t dji_motor_rx_cnt[APP_CAN_BUS_NUM][8];  /* 收到帧计数 */
extern volatile uint32_t dji_motor_last_tick[APP_CAN_BUS_NUM][8]; /* 最近一帧时间戳（ms） */

#define DJI_M1_MEASURE   (dji_motor_data[APP_CAN1][0])   /* FDCAN1, 0x201 */
#define DJI_M2_MEASURE   (dji_motor_data[APP_CAN1][1])   /* FDCAN1, 0x202 */

/* ---------- 3.2 DM 电机反馈（CAN ID / MST ID 见 app_can.c 里的配置表） ---------- */
extern dm_motor_t dm_motor_data[APP_DM_MOTOR_NUM];

#define DM_M1_MEASURE    (dm_motor_data[0].measure)   /* 配置表第 1 台 */
#define DM_M2_MEASURE    (dm_motor_data[1].measure)   /* 配置表第 2 台 */
#define DM_M3_MEASURE    (dm_motor_data[2].measure)   /* 配置表第 3 台 */
#define DM_M4_MEASURE    (dm_motor_data[3].measure)   /* 配置表第 4 台 */

/* ---------- 3.3 总线原始接收帧（调试用，保存最近 APP_CAN_RX_LOG_NUM 帧） ---------- */
extern can_rx_frame_t can1_rx_frame[APP_CAN_RX_LOG_NUM];
extern can_rx_frame_t can3_rx_frame[APP_CAN_RX_LOG_NUM];
extern volatile uint8_t  can1_rx_frame_pos;
extern volatile uint32_t can1_rx_frame_cnt;
extern volatile uint8_t  can3_rx_frame_pos;
extern volatile uint32_t can3_rx_frame_cnt;

/* ---------- 3.4 设备发送配置表（运行时可重新赋值 .hcan 换总线） ---------- */
extern can_tx_device_t can_dev_chassis;   /* 底盘 3508 四合一 */
extern can_tx_device_t can_dev_friction;  /* 摩擦轮 */
extern can_tx_device_t can_dev_yaw;       /* yaw 6020 */
extern can_tx_device_t can_dev_pitch;     /* pitch 6020 */
extern can_tx_device_t can_dev_shoot;     /* 拨盘 2006 */

/* ==================== 4. 总线初始化接口 ==================== */

/**
  * @brief          生成标准帧滤波器配置（填充外部创建的结构体，不创建）
  * @param[out]     filter: 外部创建的滤波器配置结构体指针
  * @param[in]      id:    期望接收的标准帧 ID（如 0x201）
  * @param[in]      mask:  掩码，位为 1 表示"必须匹配该位"，位为 0 表示"该位不关心"
  * @param[in]      bank:  FDCAN 实例内的滤波器序号（0 ~ StdFiltersNbr-1）
  * @retval         none
  */
extern void can_filter_std(FDCAN_FilterTypeDef *filter, uint32_t id, uint32_t mask, uint32_t bank);

/**
  * @brief          配置并启动 FDCAN1（内部会 HAL_FDCAN_Start + 打开 Rx FIFO0 中断）
  * @param[in]      filter: 外部创建并填充好的滤波器配置指针
  * @retval         none
  */
extern void app_can1_filter_init(FDCAN_FilterTypeDef *filter);

/**
  * @brief          配置并启动 FDCAN2
  * @param[in]      filter: 外部创建并填充好的滤波器配置指针
  * @retval         none
  */
extern void app_can2_filter_init(FDCAN_FilterTypeDef *filter);

/**
  * @brief          配置并启动 FDCAN3（DM 电机反馈帧）
  * @param[in]      filter: 外部创建并填充好的滤波器配置指针
  * @retval         none
  */
extern void app_can3_filter_init(FDCAN_FilterTypeDef *filter);

/* ==================== 5. DM 电机接口 ==================== */

/**
  * @brief          按 app_can.c 里的配置表批量初始化全部 DM 电机
  *                 （绑定总线 + 写入 CAN ID / MST ID，任务文件不需要维护 ID 表）
  * @param[in]      hfdcan: 这批电机挂载的 FDCAN 句柄（如 &hfdcan3）
  * @retval         none
  */
extern void app_dm_motor_init_all(FDCAN_HandleTypeDef *hfdcan);

/**
  * @brief          初始化一台 DM 电机（填默认映射范围、绑定总线与 ID）
  * @param[in,out]  motor:  电机配置指针
  * @param[in]      hfdcan: 挂载的 FDCAN 句柄（&hfdcan1 / &hfdcan2 / &hfdcan3）
  * @param[in]      can_id: 电机 CAN ID（0 = 不使用该槽位；必须 ≤ 0x7FF）
  * @param[in]      mst_id: 反馈帧 ID（Master ID）
  * @retval         none
  */
extern void app_dm_motor_init(dm_motor_t *motor, FDCAN_HandleTypeDef *hfdcan,
                              uint16_t can_id, uint16_t mst_id);

/**
  * @brief          使能 DM 电机（数据 0xFF...FF,0xFC，帧 ID = CAN ID + mode）
  * @param[in]      motor: 电机配置指针
  * @param[in]      mode:  目标模式 APP_DM_MODE_MIT / APP_DM_MODE_POS_VEL
  * @retval         none
  */
extern void app_dm_enable(dm_motor_t *motor, uint16_t mode);

/**
  * @brief          批量使能配置表里的全部 DM 电机
  * @param[in]      mode: 目标模式 APP_DM_MODE_MIT / APP_DM_MODE_POS_VEL
  * @retval         none
  */
extern void app_dm_enable_all(uint16_t mode);

/**
  * @brief          失能 DM 电机（数据 0xFF...FF,0xFD，帧 ID = CAN ID + mode）
  * @param[in]      motor: 电机配置指针
  * @param[in]      mode:  目标模式 APP_DM_MODE_MIT / APP_DM_MODE_POS_VEL
  * @retval         none
  */
extern void app_dm_disable(dm_motor_t *motor, uint16_t mode);

/**
  * @brief          批量失能配置表里的全部 DM 电机
  * @param[in]      mode: 目标模式 APP_DM_MODE_MIT / APP_DM_MODE_POS_VEL
  * @retval         none
  */
extern void app_dm_disable_all(uint16_t mode);

/**
  * @brief          MIT 模式控制（帧 ID = CAN ID）
  * @param[in]      motor: 电机配置指针
  * @param[in]      p_des: 位置给定（rad）
  * @param[in]      v_des: 速度给定（rad/s）
  * @param[in]      kp:    位置比例系数（0 ~ 500）
  * @param[in]      kd:    位置微分系数（0 ~ 5）
  * @param[in]      t_ff:  转矩前馈（N·m）
  * @retval         none
  */
extern void app_dm_cmd_mit(dm_motor_t *motor, float p_des, float v_des, float kp, float kd, float t_ff);

/**
  * @brief          MIT 模式纯力矩指令（一次给 4 台 DM 电机下发）
  *                 除 t_ff 外其余给定全为 0：p_des = 0、v_des = 0、kp = 0、kd = 0，
  *                 电机只输出参数里的力矩。
  * @param[in]      t0: dm_motor_data[0] 的力矩（N·m）
  * @param[in]      t1: dm_motor_data[1] 的力矩（N·m）
  * @param[in]      t2: dm_motor_data[2] 的力矩（N·m）
  * @param[in]      t3: dm_motor_data[3] 的力矩（N·m）
  * @retval         none
  * @note           电机顺序 = 配置表下标（app_can.c 的 app_dm_motor_can_id[]）：
  *                 DM[0]=CAN1、DM[1]=CAN2、DM[2]=CAN3、DM[3]=CAN4。
  *                 实机对应关系见 CHASSIS_TASK.h 顶部的 HIP_DM_* 表
  *                 （lf=DM[0]、lb=DM[1]、rb=DM[2]、rf=DM[3]）。
  *                 每台电机单独一帧（MIT 帧 ID = CAN ID），函数内部一次发完；
  *                 单台 t_ff 仍按该电机的 t_min/t_max 做协议映射。
  */
extern void app_dm_cmd_mit_torque_all(float t0, float t1, float t2, float t3);

/**
  * @brief          位置速度模式控制（帧 ID = 0x100 + CAN ID）
  * @param[in]      motor: 电机配置指针
  * @param[in]      p_des: 位置给定（rad，float 低位在前）
  * @param[in]      v_des: 速度给定（rad/s，float 低位在前）
  * @retval         none
  */
extern void app_dm_cmd_pos_vel(dm_motor_t *motor, float p_des, float v_des);

/* ==================== 5.x DM 参数（寄存器）读写 ====================
 * 达妙参数帧：标准帧 ID 固定 0x7FF
 *   D[0..1] = 目标电机 CAN ID（小端）
 *   D[2]    = 0x55 写 / 0x33 读 / 0xAA 存
 *   D[3]    = RID（寄存器号）
 *   D[4..7] = 数值（小端；float 或 uint32，看寄存器类型）
 *
 * ★ 0x55 只改 RAM，掉电丢失；要掉电保存得再发一帧 0xAA（写 flash，
 *   有寿命，别在每次上电时发）。所以"上电写一次 RAM"既生效又不磨损 flash。
 * ★ 读寄存器（0x33）的应答帧用的是 MST_ID（0x11~0x14），MIT 反馈也是这个 ID，
 *   本工程没有区分两种帧 —— 运行中读寄存器会让某一拍反馈被误解析，
 *   要读请停机后读，或临时把 MIT 反馈解析关掉。
 *
 * 常用 RID：1=KT  7=MST_ID  8=ESC_ID  9=TIMEOUT  10=CTRL_MODE
 *           21=PMAX  22=VMAX  23=TMAX（力矩映射范围，单位 N·m）
 * DM8009 的出厂 TMAX = 54 N·m。 */
#define APP_DM_PARAM_ID         (0x7FFU)
#define APP_DM_PARAM_WRITE      (0x55U)
#define APP_DM_PARAM_READ       (0x33U)
#define APP_DM_PARAM_STORE      (0xAAU)

#define APP_DM_RID_KT           (1U)
#define APP_DM_RID_MST_ID       (7U)
#define APP_DM_RID_ESC_ID       (8U)
#define APP_DM_RID_TIMEOUT      (9U)
#define APP_DM_RID_CTRL_MODE    (10U)
#define APP_DM_RID_PMAX         (21U)
#define APP_DM_RID_VMAX         (22U)
#define APP_DM_RID_TMAX         (23U)

/**
  * @brief          写一个 float 型寄存器（只改 RAM，掉电丢失）
  * @param[in]      motor: 电机配置指针（用它的 can_id + hfdcan）
  * @param[in]      rid:   寄存器号，见上面的 APP_DM_RID_*
  * @param[in]      value: 写入值
  * @retval         none
  */
extern void app_dm_write_param_float(dm_motor_t *motor, uint8_t rid, float value);

/**
  * @brief          写一个 uint32 型寄存器（只改 RAM，掉电丢失）
  * @param[in]      motor: 电机配置指针
  * @param[in]      rid:   寄存器号
  * @param[in]      value: 写入值
  * @retval         none
  */
extern void app_dm_write_param_u32(dm_motor_t *motor, uint8_t rid, uint32_t value);

/* ==================== 6. DJI 电机发送接口 ==================== */

/**
  * @brief          按设备配置发送底盘四电机电流
  * @param[in]      dev: 设备配置指针, 电流范围 [-16384,16384]
  * @retval         none
  */
extern void app_can_cmd_chassis(can_tx_device_t *dev, int16_t m1, int16_t m2, int16_t m3, int16_t m4);

/**
  * @brief          按设备配置发送 yaw 电流
  * @param[in]      dev: 设备配置指针, 电流范围 [-30000,30000]
  * @retval         none
  */
extern void app_can_cmd_yaw(can_tx_device_t *dev, int16_t yaw);

/**
  * @brief          按设备配置发送 pitch 电流
  * @param[in]      dev: 设备配置指针, 电流范围 [-30000,30000]
  * @retval         none
  */
extern void app_can_cmd_pitch(can_tx_device_t *dev, int16_t pitch);

/**
  * @brief          按设备配置发送摩擦轮 + 拨盘
  * @param[in]      dev:   设备配置指针
  * @param[in]      m1:    摩擦轮电机 1（3508），电流范围 [-16384,16384]
  * @param[in]      m2:    摩擦轮电机 2（3508），电流范围 [-16384,16384]
  * @param[in]      shoot: 拨盘电机；3508 为 [-16384,16384]，2006 为 [-10000,10000]
  * @retval         none
  */
extern void app_can_cmd_shoot(can_tx_device_t *dev, int16_t m1, int16_t m2, int16_t shoot);

#endif /* APP_CAN_H */
