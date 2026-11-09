#ifndef __CHASSIS_TASK_H__
#define __CHASSIS_TASK_H__

#include "main.h"
#include "cmsis_os2.h"
#include "app_can.h"

/* FreeRTOS 任务入口：CHASSIS_TASK（优先级 osPriorityHigh7，栈 512 words）
 * 1 ms 周期循环（osDelay 限制发送节拍），实际周期由 DWT 实测（见 chassis_dt_s）。 */
void chassis_task(void *argument);

/* DWT 实测的相邻两次循环间隔（s）。
 * 循环全速运行，MIT 发送速率不再受定时器限制；该值供后续控制/里程计做积分、
 * 微分时使用，调试器也可直接观察。 */
extern float chassis_dt_s;

#endif /* __CHASSIS_TASK_H__ */
