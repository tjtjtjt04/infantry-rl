#ifndef FIVEBAR
#define FIVEBAR

#include "app_can.h"
#include "math.h"

typedef struct {
    /* 摆杆长度 */
    float L0;
    float L0_dot;
    float L0_last;
    /* 连杆角度 */
    float phi0;
    float phi1;
    float phi2;
    float phi3;
    float phi4;
    /* 连杆角速度 */
    float omega0;
    float omega1;
    float omega2;
    float omega3;
    float omega4;
    /* 摆杆角（deg）：由 phi0 与 body_pitch 算出，见 fivebar_forward_kinematics() */
    float theta;
    /* 摆杆角速度（deg/s）：由 omega0 与 body_pitch_rate 算出，同函数内更新 */
    float dtheta;
    /*连杆坐标*/
    float xB;
    float xC;
    float xD;
    float yB;
    float yC;
    float yD;
    /* 正运动学参数 */
    float A0;
    float B0;
    float C0;
    float D0;
    /*逆运动学参数*/
    float A1;
    float B1;
    float C1;
} FiveBar_dynamic_t;

typedef enum
{
    right = 0U,
    left = 1U
}side_t;

/* 左右腿各一份解算结果：两腿独立维护，互不覆盖 */
extern FiveBar_dynamic_t fivebar_dynamic_right;   /* 右腿（DM 电机 [3] / [2]） */
extern FiveBar_dynamic_t fivebar_dynamic_left;    /* 左腿（DM 电机 [1] / [0]） */

void fivebar_init(FiveBar_dynamic_t *fivebar);

/**
 * @brief 五连杆正运动学：由 DM 关节角更新 B/D/C 点、L0、phi0、速度量，
 *        并算出摆杆角 theta（deg）及其角速度 dtheta（deg/s）。
 *
 * 左右腿镜像安装，符号相反（由 side 决定）：
 *   left : theta  = +( RAD_TO_DEG(phi0 - pi/2) - body_pitch )
 *          dtheta = +RAD_TO_DEG(omega0) - RAD_TO_DEG(body_pitch_rate)
 *   right: theta  = -( RAD_TO_DEG(phi0 - pi/2) - body_pitch )
 *          dtheta = -RAD_TO_DEG(omega0) + RAD_TO_DEG(body_pitch_rate)
 *   phi0  是虚拟腿相对水平面的角度（rad），omega0 = dphi0/dt（rad/s，函数内算出）；
 *   body_pitch / body_pitch_rate 由姿态 EKF 提供（EKF.pitch / EKF.gyro[1]）。
 *
 * @param fivebar          待更新的解算结构体（theta、dtheta 为输出）。
 * @param body_pitch       机体俯仰角（deg）。
 * @param body_pitch_rate  机体俯仰角速度（rad/s），即车体系 Y 轴陀螺。
 * @param side             右腿 / 左腿，决定读取哪两台 DM 电机的反馈。
 */
void fivebar_forward_kinematics(FiveBar_dynamic_t *fivebar, float body_pitch,
                                float body_pitch_rate, side_t side);

void fivebar_inverse_kinematics(FiveBar_dynamic_t *fivebar, dm_motor_t *dm_motor_data);

/* 摆杆角 theta 的计算已并入 fivebar_forward_kinematics()：调用时把 EKF.pitch
 * 作为 body_pitch 传入，结果从 fivebar->theta 读取（deg）。 */

#endif