#include "REMOTE_TASK.h"
#include "remote_control.h"

#include "FreeRTOS.h"
#include "task.h"

/*
 * 遥控数据处理任务
 * ----------------------------------------------------------------------------
 * 10 路通道的归一化已在 remote_control 模块解包时完成，结果是整数：
 * remoter.rc.ch_norm[0..9]（取值 -1 / 0 / +1，中位 0）。
 * 本任务只负责从中解析 ch6 作为轮电机输出开关：
 *   实测本遥控器 ch6 大值为“开”，见 REMOTE_CH6_ON_IS_HIGH。
 *
 * 离线安全：remoter.online == 0 时 ch6 判为关，轮电机不会出力。
 */
static uint8_t remote_ch6_on = 0U;

void remote_task(void *argument)
{
    (void)argument;

    remote_control_init();

    while (1)
    {
        if (remoter.online != 0U)
        {
#if (REMOTE_CH6_ON_IS_HIGH != 0)
            /* 大值为开：归一化 +1 */
            remote_ch6_on =
                (remoter.rc.ch_norm[REMOTE_CH6_INDEX] > 0) ? 1U : 0U;
#else
            /* 小值为开：归一化 -1 */
            remote_ch6_on =
                (remoter.rc.ch_norm[REMOTE_CH6_INDEX] < 0) ? 1U : 0U;
#endif
        }
        else
        {
            remote_ch6_on = 0U;
        }

        osDelay(1);
    }
}

uint8_t remote_ch6_switch_on(void)
{
    return remote_ch6_on;
}
