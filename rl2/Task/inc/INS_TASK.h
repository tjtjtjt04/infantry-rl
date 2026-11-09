#ifndef __INS_TASK_H__
#define __INS_TASK_H__

#include "main.h"
#include "cmsis_os2.h"

/* FreeRTOS 任务入口：INS_TASK（优先级 osPriorityRealtime7，栈 512 words） */
void ins_task(void *argument);

#endif /* __INS_TASK_H__ */
