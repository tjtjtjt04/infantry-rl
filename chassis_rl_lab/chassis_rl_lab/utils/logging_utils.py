"""训练控制台打印精简工具。

rsl_rl 每轮会打印一大段表格：

    ########
     Learning iteration it/total
    Run name: ...
    Total steps: ...
    Steps per second: ...
    Collection time: ...
    Learning time: ...
    Mean value loss: ...
    Mean reward: ...
    Mean episode length: ...
    Mean action std: ...
    ...
    Iteration time / Time elapsed / ETA

本工具在 ``train.py`` 里替换 ``runner.logger.log``：

- **仍然调用 rsl_rl 原始 logger.log**（TensorBoard/W&B 记录完整，不受影响），
  但把它的控制台输出截取下来做过滤；
- 只**删掉 "Mean value loss" 之上的部分**（迭代标题、Total steps、速度、耗时等），
  保留 value loss 及以下的 reward/episode length/std/耗时/ETA 等信息；
- 在 "Mean action std" 之后插入高度任务的 **目标高度 / 当前高度 / 误差** 三行；
- 同时把高度三项写进 TensorBoard（``Height/target``、``Height/now``、``Height/error``）。

用法（见 ``scripts/train.py``）::

    from chassis_rl_lab.utils.logging_utils import make_compact_logger_log

    runner.logger.log = make_compact_logger_log(runner, raw_env)
"""

from __future__ import annotations

import contextlib
import io
import math
import re
import statistics

__all__ = ["make_compact_logger_log"]

#: TensorBoard 打印里可能夹带的 ANSI 颜色控制符（\x1b[1m ... \x1b[0m）
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def make_compact_logger_log(runner, raw_env, total_iterations: int | None = None):
    """返回替代 ``runner.logger.log`` 的过滤打印函数（保留原日志记录行为）。

    Args:
        runner: rsl_rl ``OnPolicyRunner``（用于读取 ``logger`` 与统计量）。
        raw_env: 底层 ``ManagerBasedRLEnv``（用于读 ``height_command`` / 机器人高度）。
        total_iterations: 本次训练计划的总轮次（用于打印 ``[iter 当前/总]``；None 则只打印当前轮次）。
    """
    logger = runner.logger
    original_log = logger.log

    def compact_log(**kwargs) -> None:
        it = int(kwargs.get("it", 0))

        # 1) 原始 log 仍然执行（写 TensorBoard/W&B、维护统计量），截获其控制台输出
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            original_log(**kwargs)
        raw_output = buffer.getvalue()

        # 2) 高度指令（target）与当前机体高度（now）
        h_target = h_now = float("nan")
        try:
            h_target = float(raw_env.height_command[:, 0].mean().item())
        except Exception:  # noqa: BLE001 - 没有高度指令的任务就打印 nan
            pass
        try:
            h_now = float(raw_env.scene["robot"].data.root_pos_w[:, 2].mean().item())
        except Exception:  # noqa: BLE001
            pass
        writer = getattr(logger, "writer", None)
        if writer is not None:
            for tag, value in (
                ("Height/target", h_target),
                ("Height/now", h_now),
                ("Height/error", h_now - h_target),
            ):
                if math.isfinite(value):
                    try:
                        writer.add_scalar(tag, value, it)
                    except Exception:  # noqa: BLE001 - 写日志失败不影响训练
                        pass

        # 3) 过滤控制台输出：只保留 "Mean value loss" 及以下的行
        if raw_output.strip():
            lines = raw_output.splitlines()
            plain = [_strip_ansi(line) for line in lines]
            keep_from = None
            for index, line in enumerate(plain):
                if "Mean value loss" in line:
                    keep_from = index
                    break
            if keep_from is None:  # 兜底：loss 键名不同时从 Mean reward 开始
                for index, line in enumerate(plain):
                    if "Mean reward" in line:
                        keep_from = index
                        break
            if keep_from is not None:
                kept = [_strip_ansi(line).rstrip() for line in lines[keep_from:]]
            else:
                kept = [_strip_ansi(line).rstrip() for line in lines]

            # 第一行补回"训练轮次"（原来的 Learning iteration 行在过滤时被删掉了）
            iter_line = (
                f"[iter {it + 1}/{int(total_iterations)}]"
                if total_iterations
                else f"[iter {it + 1}]"
            )
            kept.insert(0, iter_line)

            # 在 "Mean action std:" 之后插入 target/now/error（没有则插在末尾分隔线之前）
            insert_at = None
            for index, line in enumerate(kept):
                if "Mean action std" in line:
                    insert_at = index + 1
                    break
            if insert_at is None:
                for index, line in enumerate(kept):
                    if line.strip().startswith("---"):
                        insert_at = index
                        break
            if insert_at is None:
                insert_at = len(kept)
            kept[insert_at:insert_at] = [
                f"{'Height target:':>40} {h_target:.4f} m",
                f"{'Height now:':>40} {h_now:.4f} m",
                f"{'Height error:':>40} {h_now - h_target:+.4f} m",
            ]
            print("\n".join(kept))
        else:
            # writer 为 None 时原始 log 不打印，这里给一行兜底
            print(
                f"[iter {it + 1}] h_target={h_target:.4f}  h_now={h_now:.4f}  "
                f"h_err={h_now - h_target:+.4f}"
            )

    return compact_log


def mean_episode_length_log(logger) -> float:
    """便捷读取 rsl_rl logger 的 mean episode length（供外部脚本复用）。"""
    if getattr(logger, "lenbuffer", None):
        return float(statistics.mean(logger.lenbuffer))
    return float("nan")
