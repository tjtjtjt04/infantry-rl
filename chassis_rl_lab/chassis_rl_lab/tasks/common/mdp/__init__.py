"""chassis-rl 任务的共用 MDP 项汇总。

这里导出共用的 observations / rewards / terminations / events 项；
动作项 ``actions.py`` 在任务 env cfg 里直接 import（``ChassisWheelLegActionCfg``）。

统一约定（重要）：所有关节都以**绕世界 +y 轴右手定则旋转为正**，左右同号；
轮子正转/正力矩 = 车体 +x 方向（前进）。
"""

from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
