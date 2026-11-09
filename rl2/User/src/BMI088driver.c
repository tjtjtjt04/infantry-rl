#include "BMI088driver.h"
#include "BMI088reg.h"
#include "BMI088Middleware.h"
#include "bmi088_noise.h"   /* 六面标定实测噪声 σ：导出数组 + 供里程计推导 Q */

PID_t IMU_PID;

float BMI088_ACCEL_SEN = BMI088_ACCEL_3G_SEN;
float BMI088_GYRO_SEN = BMI088_GYRO_2000_SEN;
BMI088_t BMI088;

uint8_t BMI088_Init_Error = BMI088_NO_ERROR;   // 调试观察用：BMI088_init() 的返回值

#if defined(BMI088_USE_SPI)

#define BMI088_accel_write_single_reg(reg, data) \
    {                                            \
        BMI088_ACCEL_NS_L();                     \
        BMI088_write_single_reg((reg), (data));  \
        BMI088_ACCEL_NS_H();                     \
    }
#define BMI088_accel_read_single_reg(reg, data) \
    {                                           \
        BMI088_ACCEL_NS_L();                    \
        BMI088_read_write_byte((reg) | 0x80);   \
        BMI088_read_write_byte(0x55);           \
        (data) = BMI088_read_write_byte(0x55);  \
        BMI088_ACCEL_NS_H();                    \
    }
//#define BMI088_accel_write_muli_reg( reg,  data, len) { BMI088_ACCEL_NS_L(); BMI088_write_muli_reg(reg, data, len); BMI088_ACCEL_NS_H(); }
#define BMI088_accel_read_muli_reg(reg, data, len) \
    {                                              \
        BMI088_ACCEL_NS_L();                       \
        BMI088_read_write_byte((reg) | 0x80);      \
        BMI088_read_muli_reg(reg, data, len);      \
        BMI088_ACCEL_NS_H();                       \
    }

#define BMI088_gyro_write_single_reg(reg, data) \
    {                                           \
        BMI088_GYRO_NS_L();                     \
        BMI088_write_single_reg((reg), (data)); \
        BMI088_GYRO_NS_H();                     \
    }
#define BMI088_gyro_read_single_reg(reg, data)  \
    {                                           \
        BMI088_GYRO_NS_L();                     \
        BMI088_read_single_reg((reg), &(data)); \
        BMI088_GYRO_NS_H();                     \
    }
//#define BMI088_gyro_write_muli_reg( reg,  data, len) { BMI088_GYRO_NS_L(); BMI088_write_muli_reg( ( reg ), ( data ), ( len ) ); BMI088_GYRO_NS_H(); }
#define BMI088_gyro_read_muli_reg(reg, data, len)   \
    {                                               \
        BMI088_GYRO_NS_L();                         \
        BMI088_read_muli_reg((reg), (data), (len)); \
        BMI088_GYRO_NS_H();                         \
    }

static void BMI088_write_single_reg(uint8_t reg, uint8_t data);
static void BMI088_read_single_reg(uint8_t reg, uint8_t *return_data);
//static void BMI088_write_muli_reg(uint8_t reg, uint8_t* buf, uint8_t len );
static void BMI088_read_muli_reg(uint8_t reg, uint8_t *buf, uint8_t len);

#elif defined(BMI088_USE_IIC)


#endif

static uint8_t write_BMI088_accel_reg_data_error[BMI088_WRITE_ACCEL_REG_NUM][3] =
    {
        {BMI088_ACC_PWR_CTRL, BMI088_ACC_ENABLE_ACC_ON, BMI088_ACC_PWR_CTRL_ERROR},
        {BMI088_ACC_PWR_CONF, BMI088_ACC_PWR_ACTIVE_MODE, BMI088_ACC_PWR_CONF_ERROR},
        {BMI088_ACC_CONF,  BMI088_ACC_NORMAL| BMI088_ACC_800_HZ | BMI088_ACC_CONF_MUST_Set, BMI088_ACC_CONF_ERROR},
        {BMI088_ACC_RANGE, BMI088_ACC_RANGE_3G, BMI088_ACC_RANGE_ERROR},
        {BMI088_INT1_IO_CTRL, BMI088_ACC_INT1_IO_ENABLE | BMI088_ACC_INT1_GPIO_PP | BMI088_ACC_INT1_GPIO_LOW, BMI088_INT1_IO_CTRL_ERROR},
        {BMI088_INT_MAP_DATA, BMI088_ACC_INT1_DRDY_INTERRUPT, BMI088_INT_MAP_DATA_ERROR}

};

static uint8_t write_BMI088_gyro_reg_data_error[BMI088_WRITE_GYRO_REG_NUM][3] =
    {
        {BMI088_GYRO_RANGE, BMI088_GYRO_2000, BMI088_GYRO_RANGE_ERROR},
        {BMI088_GYRO_BANDWIDTH, BMI088_GYRO_1000_116_HZ | BMI088_GYRO_BANDWIDTH_MUST_Set, BMI088_GYRO_BANDWIDTH_ERROR},
        {BMI088_GYRO_LPM1, BMI088_GYRO_NORMAL_MODE, BMI088_GYRO_LPM1_ERROR},
        {BMI088_GYRO_CTRL, BMI088_DRDY_ON, BMI088_GYRO_CTRL_ERROR},
        {BMI088_GYRO_INT3_INT4_IO_CONF, BMI088_GYRO_INT3_GPIO_PP | BMI088_GYRO_INT3_GPIO_LOW, BMI088_GYRO_INT3_INT4_IO_CONF_ERROR},
        {BMI088_GYRO_INT3_INT4_IO_MAP, BMI088_GYRO_DRDY_IO_INT3, BMI088_GYRO_INT3_INT4_IO_MAP_ERROR}

};

/* ==== BMI088 六面标定结果（2026-10-25）====
 * 加速度计（单位 m/s²）：a_cal = C * (a_raw - bias)
 * 陀螺仪（单位 rad/s）：w_cal = w_raw - bias
 *
 * 噪声实测值 σ 统一放在 bmi088_noise.h（单一来源）；这里导出成数组，
 * 方便调试查看，也供其它滤波器（如 imu_odom）引用。
 */
const float BMI088_ACC_NOISE[3] = {
    BMI088_ACC_NOISE_X, BMI088_ACC_NOISE_Y, BMI088_ACC_NOISE_Z
};

const float BMI088_GYRO_NOISE[3] = {
    BMI088_GYRO_NOISE_X, BMI088_GYRO_NOISE_Y, BMI088_GYRO_NOISE_Z
};

static const float BMI088_ACC_BIAS[3] = { 0.062875f, -0.251427f, 0.031969f };

static const float BMI088_ACC_CORR[3][3] =
{
    {  1.00533128f,  0.00035697f, -0.00417821f },
    { -0.00116775f,  1.00792243f,  0.00198462f },
    { -0.01583314f,  0.01145396f,  1.00889865f },
};

static const float BMI088_GYRO_BIAS[3] = { 0.00650618f, 0.00261984f, 0.00100713f };

/* 加速度：a_cal = C * (a_raw - bias) */
static void BMI088_accel_calibrate(const float a_raw[3], float a_cal[3])
{
    float x = a_raw[0] - BMI088_ACC_BIAS[0];
    float y = a_raw[1] - BMI088_ACC_BIAS[1];
    float z = a_raw[2] - BMI088_ACC_BIAS[2];

    a_cal[0] = BMI088_ACC_CORR[0][0] * x + BMI088_ACC_CORR[0][1] * y + BMI088_ACC_CORR[0][2] * z;
    a_cal[1] = BMI088_ACC_CORR[1][0] * x + BMI088_ACC_CORR[1][1] * y + BMI088_ACC_CORR[1][2] * z;
    a_cal[2] = BMI088_ACC_CORR[2][0] * x + BMI088_ACC_CORR[2][1] * y + BMI088_ACC_CORR[2][2] * z;
}

uint8_t BMI088_init(void)
{
    uint8_t error = BMI088_NO_ERROR;
    // 初始化 DWT 计时基准（DWT_GetTimeline_s / DWT_Delay 供上层计时使用）
    DWT_Init(SystemCoreClock / 1000000);

    // GPIO and SPI  Init .
    BMI088_GPIO_init();
    BMI088_com_init();
    PID_Init(&IMU_PID, 200.0f, 10.0f, 10.0f, 0.0f, 300.0f, 400.0f, 0.0f, FALSE);

    error |= bmi088_accel_init();
    error |= bmi088_gyro_init();

    // 六面标定：加速度在 BMI088_read() 末尾统一做 bias + 修正矩阵；
    // 陀螺仪零偏直接用 GyroOffset（读取时做减法）。
    BMI088.GyroOffset[0] = BMI088_GYRO_BIAS[0];
    BMI088.GyroOffset[1] = BMI088_GYRO_BIAS[1];
    BMI088.GyroOffset[2] = BMI088_GYRO_BIAS[2];

    BMI088_Init_Error = error;
    return error;
}

uint8_t bmi088_accel_init(void)
{
    uint8_t res = 0;
    uint8_t write_reg_num = 0;

    //check commiunication
    BMI088_accel_read_single_reg(BMI088_ACC_CHIP_ID, res);
    BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);
    BMI088_accel_read_single_reg(BMI088_ACC_CHIP_ID, res);
    BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);

    //accel software reset
    BMI088_accel_write_single_reg(BMI088_ACC_SOFTRESET, BMI088_ACC_SOFTRESET_VALUE);
    BMI088_delay_ms(BMI088_LONG_DELAY_TIME);

    //check commiunication is normal after reset
    BMI088_accel_read_single_reg(BMI088_ACC_CHIP_ID, res);
    BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);
    BMI088_accel_read_single_reg(BMI088_ACC_CHIP_ID, res);
    BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);

    // check the "who am I"
    if (res != BMI088_ACC_CHIP_ID_VALUE)
    {
        return BMI088_NO_SENSOR;
    }

    //set accel sonsor config and check
    for (write_reg_num = 0; write_reg_num < BMI088_WRITE_ACCEL_REG_NUM; write_reg_num++)
    {

        BMI088_accel_write_single_reg(write_BMI088_accel_reg_data_error[write_reg_num][0], write_BMI088_accel_reg_data_error[write_reg_num][1]);
        BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);

        BMI088_accel_read_single_reg(write_BMI088_accel_reg_data_error[write_reg_num][0], res);
        BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);

        if (res != write_BMI088_accel_reg_data_error[write_reg_num][1])
        {
            return write_BMI088_accel_reg_data_error[write_reg_num][2];
        }
    }
    return BMI088_NO_ERROR;
}

uint8_t bmi088_gyro_init(void)
{
    uint8_t write_reg_num = 0;
    uint8_t res = 0;

    //check commiunication
    BMI088_gyro_read_single_reg(BMI088_GYRO_CHIP_ID, res);
    BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);
    BMI088_gyro_read_single_reg(BMI088_GYRO_CHIP_ID, res);
    BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);

    //reset the gyro sensor
    BMI088_gyro_write_single_reg(BMI088_GYRO_SOFTRESET, BMI088_GYRO_SOFTRESET_VALUE);
    BMI088_delay_ms(BMI088_LONG_DELAY_TIME);
    //check commiunication is normal after reset
    BMI088_gyro_read_single_reg(BMI088_GYRO_CHIP_ID, res);
    BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);
    BMI088_gyro_read_single_reg(BMI088_GYRO_CHIP_ID, res);
    BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);

    // check the "who am I"
    if (res != BMI088_GYRO_CHIP_ID_VALUE)
    {
        return BMI088_NO_SENSOR;
    }

    //set gyro sonsor config and check
    for (write_reg_num = 0; write_reg_num < BMI088_WRITE_GYRO_REG_NUM; write_reg_num++)
    {

        BMI088_gyro_write_single_reg(write_BMI088_gyro_reg_data_error[write_reg_num][0], write_BMI088_gyro_reg_data_error[write_reg_num][1]);
        BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);

        BMI088_gyro_read_single_reg(write_BMI088_gyro_reg_data_error[write_reg_num][0], res);
        BMI088_delay_us(BMI088_COM_WAIT_SENSOR_TIME);

        if (res != write_BMI088_gyro_reg_data_error[write_reg_num][1])
        {
            return write_BMI088_gyro_reg_data_error[write_reg_num][2];
        }
    }

    return BMI088_NO_ERROR;
}

void BMI088_read_gyro_who_am_i(void)
{
    uint8_t buf;
    BMI088_gyro_read_single_reg(BMI088_GYRO_CHIP_ID, buf);
}

void BMI088_read_accel_who_am_i(void)
{
    uint8_t buf;
    BMI088_accel_read_single_reg(BMI088_ACC_CHIP_ID, buf);
}

void BMI088_temperature_read_over(uint8_t *rx_buf, BMI088_t *BMI088)
{
    int16_t bmi088_raw_temp;
    bmi088_raw_temp = (int16_t)((rx_buf[0] << 3) | (rx_buf[1] >> 5));
    if (bmi088_raw_temp > 1023)
    {
        bmi088_raw_temp -= 2048;
    }
    BMI088->temperature = bmi088_raw_temp * BMI088_TEMP_FACTOR + BMI088_TEMP_OFFSET;

}

void BMI088_accel_read_over(uint8_t *rx_buf, BMI088_t *BMI088)
{
    int16_t bmi088_raw_temp;
    float acc_raw[3];

    bmi088_raw_temp = (int16_t)((rx_buf[1]) << 8) | rx_buf[0];
    acc_raw[0] = bmi088_raw_temp * BMI088_ACCEL_SEN;

    bmi088_raw_temp = (int16_t)((rx_buf[3]) << 8) | rx_buf[2];
    acc_raw[1] = bmi088_raw_temp * BMI088_ACCEL_SEN;

    bmi088_raw_temp = (int16_t)((rx_buf[5]) << 8) | rx_buf[4];
    acc_raw[2] = bmi088_raw_temp * BMI088_ACCEL_SEN;

    BMI088_accel_calibrate(acc_raw, BMI088->accel);
}

void BMI088_gyro_read_over(uint8_t *rx_buf, BMI088_t *BMI088)
{
    int16_t bmi088_raw_temp;
    bmi088_raw_temp = (int16_t)((rx_buf[1]) << 8) | rx_buf[0];
    BMI088->gyro[0] = bmi088_raw_temp * BMI088_GYRO_SEN - BMI088->GyroOffset[0];

    bmi088_raw_temp = (int16_t)((rx_buf[3]) << 8) | rx_buf[2];
    BMI088->gyro[1] = bmi088_raw_temp * BMI088_GYRO_SEN - BMI088->GyroOffset[1];

    bmi088_raw_temp = (int16_t)((rx_buf[5]) << 8) | rx_buf[4];
    BMI088->gyro[2] = bmi088_raw_temp * BMI088_GYRO_SEN - BMI088->GyroOffset[2];
}

void BMI088_read(BMI088_t *BMI088)
{
    uint8_t buf[8] = {0, 0, 0, 0, 0, 0};
    int16_t bmi088_raw_temp;

    //读取三轴加速度
    BMI088_accel_read_muli_reg(BMI088_ACCEL_XOUT_L, buf, 6);

    bmi088_raw_temp = (int16_t)((buf[1]) << 8) | buf[0];
    BMI088->accel[0] = bmi088_raw_temp * BMI088_ACCEL_SEN;

    bmi088_raw_temp = (int16_t)((buf[3]) << 8) | buf[2];
    BMI088->accel[1] = bmi088_raw_temp * BMI088_ACCEL_SEN;

    bmi088_raw_temp = (int16_t)((buf[5]) << 8) | buf[4];
    BMI088->accel[2] = bmi088_raw_temp * BMI088_ACCEL_SEN;

    //读取三轴角速度
    BMI088_gyro_read_muli_reg(BMI088_GYRO_CHIP_ID, buf, 8);
    if (buf[0] == BMI088_GYRO_CHIP_ID_VALUE)
    {
        bmi088_raw_temp = (int16_t)((buf[3]) << 8) | buf[2];
        BMI088->gyro[0] = bmi088_raw_temp * BMI088_GYRO_SEN - BMI088->GyroOffset[0];

        bmi088_raw_temp = (int16_t)((buf[5]) << 8) | buf[4];
        BMI088->gyro[1] = bmi088_raw_temp * BMI088_GYRO_SEN - BMI088->GyroOffset[1];

        bmi088_raw_temp = (int16_t)((buf[7]) << 8) | buf[6];
        BMI088->gyro[2] = bmi088_raw_temp * BMI088_GYRO_SEN - BMI088->GyroOffset[2];
    }

    BMI088_accel_read_muli_reg(BMI088_TEMP_M, buf, 2);

    bmi088_raw_temp = (int16_t)((buf[0] << 3) | (buf[1] >> 5));

    if (bmi088_raw_temp > 1023)
    {
        bmi088_raw_temp -= 2048;
    }

    BMI088->temperature = bmi088_raw_temp * BMI088_TEMP_FACTOR + BMI088_TEMP_OFFSET;

    /* 六面标定：加速度 a_cal = C * (a_raw - bias) */
    BMI088_accel_calibrate(BMI088->accel, BMI088->accel);
}


#if defined(BMI088_USE_SPI)

static void BMI088_write_single_reg(uint8_t reg, uint8_t data)
{
    BMI088_read_write_byte(reg);
    BMI088_read_write_byte(data);
}

static void BMI088_read_single_reg(uint8_t reg, uint8_t *return_data)
{
    BMI088_read_write_byte(reg | 0x80);
    *return_data = BMI088_read_write_byte(0x55);
}


static void BMI088_read_muli_reg(uint8_t reg, uint8_t *buf, uint8_t len)
{
    BMI088_read_write_byte(reg | 0x80);

    while (len != 0)
    {
        *buf = BMI088_read_write_byte(0x55);
        buf++;
        len--;
    }
}

void IMU_Temperature_Control(float temp)
{
    uint16_t PWM;
    float outcheck;
    PID_Set_Values(&IMU_PID, target_temp, temp);
    PID_Calculate(&IMU_PID, &outcheck);
    if (temp > target_temp)
    {
        outcheck = 0.0f;
    }
    PWM = (uint16_t)outcheck;
    TIM3->CCR4 = (PWM);
}

#elif defined(BMI088_USE_IIC)

#endif
