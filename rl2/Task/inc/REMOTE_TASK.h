#ifndef __REMOTE_TASK_H__
#define __REMOTE_TASK_H__

#include "main.h"
#include "cmsis_os2.h"

/* ch6 作为轮电机输出开关
 * 归一化值来自 remote_control 模块：remoter.rc.ch_norm[REMOTE_CH6_INDEX]，
 * 取值为 -1 / 0 / +1（下满 -1、中位 0、上满 +1）。
 * 实测遥控器 ch6：**大值（原始约 1807，归一化 +1）为“开”**，
 * 所以 REMOTE_CH6_ON_IS_HIGH = 1；若换遥控器或方向相反改为 0。 */
#define REMOTE_CH6_INDEX            (6U)
#define REMOTE_CH6_ON_IS_HIGH       (1)   /* 1 = 大值为开；0 = 小值为开 */

/* FreeRTOS 任务入口：REMOTE_TASK（优先级 osPriorityRealtime7，栈 256 words）
 * 职责：根据 rc 模块已归一化的整数通道值解析 ch6 输出开关状态 */
void remote_task(void *argument);

/* ch6 开关状态：1 = 开（允许轮电机出力），0 = 关 / 遥控离线 */
uint8_t remote_ch6_switch_on(void);

#endif /* __REMOTE_TASK_H__ */
