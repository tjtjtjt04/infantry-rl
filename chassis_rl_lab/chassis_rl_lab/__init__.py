"""chassis_rl_lab：chassis-rl 五连杆轮腿机器人的 Isaac Lab 强化学习任务包。

本包只负责在**具备 Isaac Lab/Kit 运行时**的环境里触发任务注册
（``Chassis-Stand-Up-v0``）；纯 CPU 的 kinematics 工具
（``tools/fit_planar_model.py`` / ``tools/verify_kinematics.py`` 等）可以安全地
``import chassis_rl_lab.kinematics``，不会连锁 import isaaclab/pxr。

判定方式：用 ``importlib.util.find_spec("pxr")`` 检查 USD（pxr）是否可用。
这比捕获 ``ImportError`` 更严格——Kit 环境里若 ``tasks`` 自身 import 失败，
真实报错会正常抛出，不会被静默吞掉。
"""

import importlib.util

# 只有在 Isaac Lab/Kit 环境（pxr 可导入）里才注册任务；
# 纯 CPU 进程（无 pxr）不 import tasks，因此 import chassis_rl_lab.kinematics 安全。
if importlib.util.find_spec("pxr") is not None:
    from . import tasks  # noqa: F401  # 副作用：gym.register("Chassis-Stand-Up-v0")
