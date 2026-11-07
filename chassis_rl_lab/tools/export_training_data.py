"""把一次训练 run 的指标导出为 CSV + 图像 + 摘要到 ``data/<run 目录名>/``（无需启动 Isaac Sim）。

用法::

    D:\\miniconda\\envs\\RM\\python.exe tools\\export_training_data.py ^
        --log_dir logs\\rsl_rl\\chassis_rl_stand_up\\<run>

    # 指定导出根目录
    ... --log_dir <run> --data_root D:\\somewhere\\data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chassis_rl_lab.utils.training_data import export_training_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a training run's metrics/plots to data/")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="训练 run 目录（如 logs/rsl_rl/chassis_rl_stand_up/2026-11-06_12-30-00_v1）",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="导出根目录（默认 <项目根>/data）",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / "data"
    out_dir = export_training_data(log_dir, data_root=data_root)
    print(f"[OK] 训练数据已导出：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
