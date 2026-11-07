"""chassis_rl_lab 的 Isaac Lab 任务注册入口。

导入本包即触发所有任务的 ``gym.register``（由各任务子包的 ``__init__.py`` 完成）。
必须在 ``AppLauncher`` 启动之后再 import（Isaac Lab 的约定）。

当前只有一个任务：``Chassis-Stand-Up-v0``（起立/平衡/高度控制）。
"""

from . import stand_up  # noqa: F401  # 导入即注册 Chassis-Stand-Up-v0
