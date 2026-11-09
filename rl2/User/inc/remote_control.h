#ifndef __UART_BSP_H__
#define __UART_BSP_H__

#include "main.h"

#define SBUS_FRAME_SIZE 25
#define BUFF_SIZE       25

/* ==================== 遥控通道归一化配置 ====================
 * SBUS 通道原始值 11 位：中值 1024，上下限各 ±783（即 241 ~ 1807）。
 *
 * 归一化用整数除法实现（不使用 float）：
 *     ch_norm = (ch - 1024) / 783
 * C 语言整数除法向 0 截断，因此结果只有 -1 / 0 / +1 三种值：
 *     ch <=  241  -> -1（下满/下限）
 *     242 ~ 1806 ->  0（中位附近）
 *     ch >= 1807  -> +1（上满/上限）
 *
 * sbus_frame_parse() 每次成功解包后自动把 10 路通道归一化到
 * remoter.rc.ch_norm[]；遥控离线（failsafe 帧）时 ch_norm 全部置 0（中位）。 */
#define RC_CHANNEL_COUNT    (10U)
#define RC_CH_RAW_CENTER    (1024)   /* SBUS 通道中值 */
#define RC_CH_RAW_RANGE     (783)    /* 上下限各 783 */

typedef struct
{
    uint16_t online;

    struct
    {
        int16_t ch[RC_CHANNEL_COUNT];       /* SBUS 原始值（0~2047，中值 1024） */
        int16_t ch_norm[RC_CHANNEL_COUNT];  /* 归一化整数：-1 / 0 / +1 */
    } rc;
} remoter_t;

extern uint8_t rx_buff[BUFF_SIZE];
extern remoter_t remoter;

/* 初始化：UART5 用 DMA + 空闲中断接收 SBUS */
void remote_control_init(void);

/* 解包函数（原位置，不加 static）：
 * 解出 10 路原始值，并同步把整数归一化结果写入 rc.ch_norm[] */
void sbus_frame_parse(remoter_t *remoter, const uint8_t *buf);

#endif /*__UART_BSP_H__ */
