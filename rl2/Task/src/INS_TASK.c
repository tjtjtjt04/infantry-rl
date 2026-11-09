#include "INS_TASK.h"

#include "spi.h"
#include "tim.h"
#include "BMI088driver.h"
#include "BMI088Middleware.h"
#include "ekf.h"
#include "bsp_dwt.h"

#include "FreeRTOS.h"
#include "task.h"

#define INS_NOTIFY_TRIGGER (1UL << 0) /* PA0(EXTI0) 唤醒任务 */

#define ACCEL_DMA_LEN      9U
#define GYRO_DMA_LEN       8U
#define TEMP_DMA_LEN       4U

/* BMI088 SPI 读寄存器时前面有 1~2 个 dummy 字节 */
#define ACCEL_DATA_OFFSET  2U
#define GYRO_DATA_OFFSET   1U
#define TEMP_DATA_OFFSET   2U

typedef enum
{
    INS_DMA_IDLE = 0,
    INS_DMA_ACCEL,
    INS_DMA_GYRO,
    INS_DMA_TEMP
} ins_dma_stage_t;

static TaskHandle_t ins_task_handle = NULL;
static volatile ins_dma_stage_t ins_dma_stage = INS_DMA_IDLE;
static volatile uint8_t imu_start_dma_flag = 0U;

/* 来自 BMI088 INT 引脚的读请求 */
static volatile uint8_t accel_dr_flag = 0U;
static volatile uint8_t gyro_dr_flag = 0U;
static volatile uint8_t temp_dr_flag = 0U;

/* DMA 已完成、等待任务解析的数据 */
static volatile uint8_t accel_ready = 0U;
static volatile uint8_t gyro_ready = 0U;
static volatile uint8_t temp_ready = 0U;

/* accel + gyro 都到齐，任务可以解算姿态 */
static volatile uint8_t imu_data_ready = 0U;

/* 发送/接收缓冲区；当前链接脚本把 .data/.bss 放 RAM_D1，DMA 可访问 */
static uint8_t accel_dma_tx_buf[ACCEL_DMA_LEN] = {0x92U, 0xFFU, 0xFFU, 0xFFU, 0xFFU,
                                                   0xFFU, 0xFFU, 0xFFU, 0xFFU};
static uint8_t gyro_dma_tx_buf[GYRO_DMA_LEN] = {0x82U, 0xFFU, 0xFFU, 0xFFU,
                                                0xFFU, 0xFFU, 0xFFU, 0xFFU};
static uint8_t temp_dma_tx_buf[TEMP_DMA_LEN] = {0xA2U, 0xFFU, 0xFFU, 0xFFU};

static uint8_t accel_dma_rx_buf[ACCEL_DMA_LEN] __attribute__((aligned(32)));
static uint8_t gyro_dma_rx_buf[GYRO_DMA_LEN] __attribute__((aligned(32)));
static uint8_t temp_dma_rx_buf[TEMP_DMA_LEN] __attribute__((aligned(32)));

/*
 * 根据 INT 引脚产生的请求启动下一路 SPI DMA。
 * 同一个传感器上一次的数据还没被任务取走时（ready=1），不覆盖它的接收缓冲。
 */
static void ins_imu_cmd_dma(void)
{
    if ((imu_start_dma_flag == 0U) || (ins_dma_stage != INS_DMA_IDLE))
    {
        return;
    }

    /* 陀螺仪优先 */
    if ((gyro_dr_flag != 0U) && (gyro_ready == 0U))
    {
        gyro_dr_flag = 0U;
        ins_dma_stage = INS_DMA_GYRO;
        BMI088_GYRO_NS_L();
        if (HAL_SPI_TransmitReceive_DMA(&hspi2, gyro_dma_tx_buf, gyro_dma_rx_buf,
                                        GYRO_DMA_LEN) != HAL_OK)
        {
            BMI088_GYRO_NS_H();
            ins_dma_stage = INS_DMA_IDLE;
            gyro_dr_flag = 1U;
        }
        return;
    }

    /* 加速度计 */
    if ((accel_dr_flag != 0U) && (accel_ready == 0U))
    {
        accel_dr_flag = 0U;
        ins_dma_stage = INS_DMA_ACCEL;
        BMI088_ACCEL_NS_L();
        if (HAL_SPI_TransmitReceive_DMA(&hspi2, accel_dma_tx_buf, accel_dma_rx_buf,
                                        ACCEL_DMA_LEN) != HAL_OK)
        {
            BMI088_ACCEL_NS_H();
            ins_dma_stage = INS_DMA_IDLE;
            accel_dr_flag = 1U;
        }
        return;
    }

    /* 温度接在加速度计 CS 上；优先级最低 */
    if ((temp_dr_flag != 0U) && (temp_ready == 0U))
    {
        temp_dr_flag = 0U;
        ins_dma_stage = INS_DMA_TEMP;
        BMI088_ACCEL_NS_L();
        if (HAL_SPI_TransmitReceive_DMA(&hspi2, temp_dma_tx_buf, temp_dma_rx_buf,
                                        TEMP_DMA_LEN) != HAL_OK)
        {
            BMI088_ACCEL_NS_H();
            ins_dma_stage = INS_DMA_IDLE;
            temp_dr_flag = 1U;
        }
        return;
    }
}

/* PA0 -> EXTI0：只负责唤醒 INS 任务 */
void EXTI0_IRQHandler(void)
{
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_0);
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if (GPIO_Pin == ACC_INT_Pin)
    {
        /* 加速度计数据就绪：请求 accel + temp 两路 DMA */
        accel_dr_flag = 1U;
        temp_dr_flag = 1U;
        ins_imu_cmd_dma();
    }
    else if (GPIO_Pin == GYRO_INT_Pin)
    {
        /* 陀螺仪数据就绪：请求 gyro DMA */
        gyro_dr_flag = 1U;
        ins_imu_cmd_dma();
    }
    else if (GPIO_Pin == GPIO_PIN_0)
    {
        BaseType_t xHigherPriorityTaskWoken = pdFALSE;
        if (ins_task_handle != NULL)
        {
            (void)xTaskNotifyFromISR(ins_task_handle, INS_NOTIFY_TRIGGER,
                                     eSetBits, &xHigherPriorityTaskWoken);
            portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
        }
    }
}

/* SPI2 DMA 完成：释放 CS、标记数据 ready、接着启动下一路请求 */
void HAL_SPI_TxRxCpltCallback(SPI_HandleTypeDef *hspi)
{
    if (hspi->Instance != SPI2)
    {
        return;
    }

    switch (ins_dma_stage)
    {
    case INS_DMA_ACCEL:
        BMI088_ACCEL_NS_H();
        accel_ready = 1U;
        ins_dma_stage = INS_DMA_IDLE;
        ins_imu_cmd_dma();
        break;

    case INS_DMA_GYRO:
        BMI088_GYRO_NS_H();
        gyro_ready = 1U;
        ins_dma_stage = INS_DMA_IDLE;
        ins_imu_cmd_dma();
        break;

    case INS_DMA_TEMP:
        BMI088_ACCEL_NS_H();
        temp_ready = 1U;
        ins_dma_stage = INS_DMA_IDLE;
        ins_imu_cmd_dma();
        break;

    default:
        break;
    }

    /* 加速度和陀螺仪都到齐后，用 PA0 软件 EXTI 唤醒任务 */
    if ((accel_ready != 0U) && (gyro_ready != 0U) && (imu_data_ready == 0U))
    {
        imu_data_ready = 1U;
        __HAL_GPIO_EXTI_GENERATE_SWIT(GPIO_PIN_0);
    }
}

void HAL_SPI_ErrorCallback(SPI_HandleTypeDef *hspi)
{
    if (hspi->Instance == SPI2)
    {
        if ((ins_dma_stage == INS_DMA_ACCEL) || (ins_dma_stage == INS_DMA_TEMP))
        {
            BMI088_ACCEL_NS_H();
        }
        else if (ins_dma_stage == INS_DMA_GYRO)
        {
            BMI088_GYRO_NS_H();
        }
        ins_dma_stage = INS_DMA_IDLE;
    }
}

void ins_task(void *argument)
{
    (void)argument;

    ins_task_handle = xTaskGetCurrentTaskHandle();

    /* 初始化 DWT 计时、BMI088 寄存器、加热 PWM */
    DWT_Init(SystemCoreClock / 1000000U);
    (void)BMI088_init();
    (void)HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_4);

    /*
     * 恒温预热：在达到 target_temp 之前不初始化 EKF、不启动姿态解算。
     * 用 osDelay(10) 让出 CPU，CAN/REMOTE 等低优先级任务仍然可以运行。
     */
    while (BMI088.temperature < target_temp)
    {
        BMI088_read(&BMI088);
        if (BMI088.temperature < target_temp)
        {
            IMU_Temperature_Control(BMI088.temperature);
        }
        osDelay(10U);
    }
    TIM3->CCR4 = 0U;

    /* 温度到位后再初始化 EKF 和中断触发 */
    ekf_init(&EKF);

    /* 使能 PA0 的 EXTI0：任务只被 PA0 唤醒 */
    HAL_NVIC_SetPriority(EXTI0_IRQn, 6, 0);
    HAL_NVIC_EnableIRQ(EXTI0_IRQn);

    /* 允许 INT 引脚触发 SPI DMA，并处理初始化期间可能已经挂起的请求 */
    imu_start_dma_flag = 1U;
    ins_imu_cmd_dma();

    uint32_t notify_bits = 0U;

    while (1)
    {
        (void)xTaskNotifyWait(0U, 0xFFFFFFFFU, &notify_bits, portMAX_DELAY);

        if ((notify_bits & INS_NOTIFY_TRIGGER) != 0U)
        {
            uint8_t do_accel = 0U;
            uint8_t do_gyro = 0U;
            uint8_t do_temp = 0U;

            /* 先原子地取走 ready 标志，避免解析过程中新数据覆盖 */
            taskENTER_CRITICAL();
            if (imu_data_ready != 0U)
            {
                imu_data_ready = 0U;

                do_accel = accel_ready;
                accel_ready = 0U;
                do_gyro = gyro_ready;
                gyro_ready = 0U;
                do_temp = temp_ready;
                temp_ready = 0U;
            }
            taskEXIT_CRITICAL();

            if ((do_accel != 0U) || (do_gyro != 0U) || (do_temp != 0U))
            {
                if (do_accel != 0U)
                {
                    BMI088_accel_read_over(accel_dma_rx_buf + ACCEL_DATA_OFFSET, &BMI088);
                }
                if (do_gyro != 0U)
                {
                    BMI088_gyro_read_over(gyro_dma_rx_buf + GYRO_DATA_OFFSET, &BMI088);
                }
                if (do_temp != 0U)
                {
                    BMI088_temperature_read_over(temp_dma_rx_buf + TEMP_DATA_OFFSET, &BMI088);
                }

                /* 姿态解算 + 温度控制 */
                IMU_Temperature_Control(BMI088.temperature);
                (void)ekf_update(&EKF, &BMI088);

                /* 解析完缓冲区后，把积压的 INT 请求继续发出去 */
                ins_imu_cmd_dma();
            }
        }
    }
}
