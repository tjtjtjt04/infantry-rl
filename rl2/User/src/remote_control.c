#include "remote_control.h"
#include "common_function.h"
#include "usart.h"

#include <string.h>

#define SBUS_HEAD 0x0F
#define SBUS_END  0x00

uint8_t rx_buff[BUFF_SIZE] __attribute__((aligned(32)));
remoter_t remoter;

/**
 * @brief 单路原始值 -> 归一化整数（-1 / 0 / +1）
 *
 * 整数除法：norm = (raw - 1024) / 783，向 0 截断；
 * 结果只可能是 -1、0、+1，保留符号/方向信息，不使用浮点。
 * 最后再限幅一次，防止异常原始值（>2047 等）越界。
 */
static int16_t remote_rc_normalize_channel(int16_t raw)
{
    int32_t norm = ((int32_t)raw - (int32_t)RC_CH_RAW_CENTER) / (int32_t)RC_CH_RAW_RANGE;

    norm = CLAMP(norm, -1, 1);

    return (int16_t)norm;
}

/**
 * @brief 把 10 路原始通道值全部归一化成整数，写入 rc.ch_norm[]
 */
static void remote_rc_normalize(remoter_t *remote)
{
    uint8_t i;

    for (i = 0U; i < RC_CHANNEL_COUNT; i++)
    {
        remote->rc.ch_norm[i] = remote_rc_normalize_channel(remote->rc.ch[i]);
    }
}

void sbus_frame_parse(remoter_t *remoter, const uint8_t *buf)
{
    uint8_t i;

    if ((buf[0] != SBUS_HEAD) || (buf[SBUS_FRAME_SIZE - 1U] != SBUS_END)) {
        return;
    }

    if (buf[23] == 0x0CU) {
        remoter->online = 0U;
    } else {
        remoter->online = 1U;
    }

    remoter->rc.ch[0] = ((buf[1] | buf[2] << 8) & 0x07FF);
    remoter->rc.ch[1] = ((buf[2] >> 3 | buf[3] << 5) & 0x07FF);
    remoter->rc.ch[2] = ((buf[3] >> 6 | buf[4] << 2 | buf[5] << 10) & 0x07FF);
    remoter->rc.ch[3] = ((buf[5] >> 1 | buf[6] << 7) & 0x07FF);
    remoter->rc.ch[4] = ((buf[6] >> 4 | buf[7] << 4) & 0x07FF);
    remoter->rc.ch[5] = ((buf[7] >> 7 | buf[8] << 1 | buf[9] << 9) & 0x07FF);
    remoter->rc.ch[6] = ((buf[9] >> 2 | buf[10] << 6) & 0x07FF);
    remoter->rc.ch[7] = ((buf[10] >> 5 | buf[11] << 3) & 0x07FF);
    remoter->rc.ch[8] = ((buf[12] | buf[13] << 8) & 0x07FF);
    remoter->rc.ch[9] = ((buf[13] >> 3 | buf[14] << 5) & 0x07FF);

    /* 原始值 -> 归一化整数（10 路全通道） */
    remote_rc_normalize(remoter);

    /* 遥控失联/failsafe：原始值保留，归一化值置中位 0（上层按“无输入”处理） */
    if (remoter->online == 0U) {
        for (i = 0U; i < RC_CHANNEL_COUNT; i++) {
            remoter->rc.ch_norm[i] = 0;
        }
    }
}

/* UART5 DMA 接收事件回调：一帧结束（空闲）或 DMA 传输完成时进入 */
void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t Size)
{
    if (huart->Instance == UART5) {
        if (Size <= BUFF_SIZE) {
            HAL_UARTEx_ReceiveToIdle_DMA(&huart5, rx_buff, BUFF_SIZE); /* 接收完毕后重启 */
            sbus_frame_parse(&remoter, rx_buff);
        } else { /* 接收数据长度大于 BUFF_SIZE，错误处理 */
            HAL_UARTEx_ReceiveToIdle_DMA(&huart5, rx_buff, BUFF_SIZE);
            memset(rx_buff, 0, BUFF_SIZE);
        }
    }
}

/* UART5 错误回调：接收发生错误后重启 */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == UART5) {
        HAL_UARTEx_ReceiveToIdle_DMA(&huart5, rx_buff, BUFF_SIZE);
        memset(rx_buff, 0, BUFF_SIZE);
    }
}

/* 初始化：UART5 用 DMA + 空闲中断接收 SBUS */
void remote_control_init(void)
{
    memset(&remoter, 0, sizeof(remoter));
    memset(rx_buff, 0, BUFF_SIZE);

    HAL_UARTEx_ReceiveToIdle_DMA(&huart5, rx_buff, BUFF_SIZE);
}
