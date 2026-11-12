"""训练 chassis-rl 五连杆轮腿机器人的蹲姿保持平衡策略（RSL-RL）。

用法（Windows，`D:\\miniconda\\envs\\RM\\python.exe`）::

    # 最小冒烟（256 envs，100 iter）
    python chassis_rl_lab\\scripts\\train.py --headless --num_envs 256 --max_iterations 100

    # 正式训练
    python chassis_rl_lab\\scripts\\train.py --num_envs 2048 --max_iterations 1500 --run_name v1

    # 续训（--load_checkpoint 可传文件路径或正则，缺省 = 最新一次 run 的最新 checkpoint）
    python chassis_rl_lab\\scripts\\train.py --resume --load_run ".*v1" --max_iterations 500

脚本接口（规格 §7）：
    --task 默认 Chassis-Stand-Up-v0；--num_envs/--max_iterations/--seed/--run_name/
    --resume/--load_run/--load_checkpoint/--device；另支持 --experiment_name/--save_interval/--log_root。
    日志：``logs/rsl_rl/<experiment_name>/<时间戳>_<run_name>/``，并 dump ``env.yaml``/``agent.yaml``。
    支持 Ctrl+C：捕获 KeyboardInterrupt 后把最近一次迭代的模型保存为该 run 目录下的
    ``model_<iteration>.pt`` 再退出。

重要实现约束（见 `_design/01_reference_notes.md` §2）：
    * Isaac Sim 必须先由 ``AppLauncher`` 启动，之后才能 import ``isaaclab`` / ``isaaclab_rl`` /
      ``rsl_rl`` / ``gymnasium``（否则 `No module named 'pxr'`）。
    * rsl_rl 5.0.1 的 runner 配置是新式字段：``actor/critic=RslRlMLPModelCfg``、
      ``algorithm=RslRlPpoAlgorithmCfg``、``obs_groups``、``empirical_normalization``，
      且必须调用 ``handle_deprecated_rsl_rl_cfg`` 清理 5.0 已弃用字段。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import re
import statistics
import sys
import traceback
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1) 项目根加入 sys.path（必须在任何 chassis_rl_lab 相关 import 之前）
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# 2) 只做 argparse / path 操作；AppLauncher 之前不能 import 其它 Isaac Lab 模块
# --------------------------------------------------------------------------- #
from isaaclab.app import AppLauncher  # noqa: E402  (此 import 不依赖 pxr，安全)

DEFAULT_TASK = "Chassis-Stand-Up-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RSL-RL agent for the chassis-rl squat-pose balance task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- 规格 §7 要求的参数 ----
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="gym 注册的任务 id")
    parser.add_argument("--num_envs", type=int, default=None, help="并行环境数（None = 用 env cfg 默认）")
    parser.add_argument("--max_iterations", type=int, default=None, help="PPO 迭代次数（None = 用 agent cfg 默认）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（None = 用 agent cfg 默认）")
    parser.add_argument("--run_name", type=str, default=None, help="日志目录后缀")
    parser.add_argument("--resume", action="store_true", default=False, help="从 checkpoint 续训")
    parser.add_argument("--load_run", type=str, default=None, help="续训的 run 目录（正则/目录名）")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="checkpoint 路径或正则（如 model_.*.pt）")

    # ---- 辅助参数（不改变规格接口）----
    parser.add_argument("--experiment_name", type=str, default=None, help="覆盖 agent cfg 的 experiment_name")
    parser.add_argument("--save_interval", type=int, default=None, help="覆盖保存间隔（迭代数）")
    parser.add_argument("--log_root", type=str, default=None, help="日志根目录（默认相对当前工作目录的 logs/rsl_rl）")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="训练数据（CSV/图像/摘要）导出根目录（默认 <项目根>/data）",
    )
    parser.add_argument(
        "--no_data_export",
        action="store_true",
        default=False,
        help="不导出训练数据（默认训练结束后把指标/图像存到 data/<run 目录名>/）",
    )
    parser.add_argument(
        "--export_interval",
        type=int,
        default=100,
        help=(
            "训练中每隔多少 iter 导出一次**累计**训练数据（默认 100；0=只在结束时导出）；"
            "导出目录固定为 data/<run 目录名>/，每次覆盖为从头到当前的完整曲线"
        ),
    )
    # ---- AppLauncher 参数：--headless / --device / --enable_cameras / --livestream ... ----
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()

# --------------------------------------------------------------------------- #
# 3) 启动 Isaac Sim（拿到 simulation_app 之后其余 import 才安全）
# --------------------------------------------------------------------------- #
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
# 4) Kit 启动后的 import
# --------------------------------------------------------------------------- #
import importlib  # noqa: E402
import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

# 触发 chassis_rl_lab/tasks 的 gym.register
try:
    import chassis_rl_lab  # noqa: F401
except ImportError as exc:  # pragma: no cover - 环境未就绪时给出明确提示
    raise RuntimeError(
        "无法 import `chassis_rl_lab`：请确认已实现 chassis_rl_lab/chassis_rl_lab/__init__.py "
        "（其中应 import tasks 并完成 gym.register）。"
    ) from exc


# --------------------------------------------------------------------------- #
# 5) 工具函数
# --------------------------------------------------------------------------- #
def load_cfg_from_registry(task: str, entry_point_key: str):
    """从 gym registry 的 kwargs 里解析 ``"module:ClassName"`` 并实例化配置类。

    等价于 ``isaaclab_tasks.utils.load_cfg_from_registry``，但脚本自实现以减少依赖。
    """
    try:
        spec = gym.spec(task.split(":")[-1])
    except gym.error.Error as exc:
        raise RuntimeError(
            f"任务 {task!r} 尚未在 gym 注册。请先完成 chassis_rl_lab 的 gym.register，"
            f"并确认 `import chassis_rl_lab` 已执行。"
        ) from exc
    entry_point = spec.kwargs.get(entry_point_key)
    if entry_point is None:
        raise ValueError(
            f"任务 {task!r} 的 registry kwargs 缺少 {entry_point_key!r}；"
            f"现有 keys = {sorted(spec.kwargs.keys())}"
        )
    if callable(entry_point):
        return entry_point()
    module_name, class_name = entry_point.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


def _checkpoint_sort_key(filename: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", filename)
    return (int(match.group(1)) if match else -1, filename)


def resolve_checkpoint(log_root: str, load_run: str = ".*", load_checkpoint: str = "model_.*.pt") -> str:
    """解析续训/回放用的 checkpoint 路径。

    * ``load_checkpoint`` 是已存在文件 ⇒ 直接用；
    * 否则把它当正则；``load_run`` 当 run 目录正则（``-1``/空 = 最新 run）。
    """
    if load_checkpoint and os.path.isfile(load_checkpoint):
        return os.path.abspath(load_checkpoint)

    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"日志根目录不存在：{log_root}")

    run_pattern = ".*" if load_run in (None, "", "-1") else load_run
    runs = [
        os.path.join(log_root, name)
        for name in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, name)) and re.match(run_pattern, name)
    ]
    if not runs:
        raise FileNotFoundError(f"{log_root} 下没有匹配 {run_pattern!r} 的 run 目录")
    run_dir = sorted(runs)[-1]

    checkpoint_pattern = "model_.*.pt" if load_checkpoint in (None, "", "-1") else load_checkpoint
    files = [name for name in os.listdir(run_dir) if re.match(checkpoint_pattern, name)]
    if not files:
        raise FileNotFoundError(f"{run_dir} 下没有匹配 {checkpoint_pattern!r} 的 checkpoint")
    files.sort(key=_checkpoint_sort_key)
    return os.path.join(run_dir, files[-1])


def main() -> int:
    # ---------------- 1. 读取 cfg ----------------
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # ---------------- 2. 命令行覆盖 ----------------
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.seed is not None:
        agent_cfg.seed = int(args_cli.seed)
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = int(args_cli.max_iterations)
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = int(args_cli.save_interval)
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.load_checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.load_checkpoint
    agent_cfg.resume = bool(args_cli.resume or args_cli.load_checkpoint)

    # 设备：命令行优先，其次 env cfg（默认 cuda:0）
    device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.device = device
    agent_cfg.device = device

    # ---------------- 3. rsl_rl 版本兼容处理 ----------------
    installed_version = metadata.version("rsl-rl-lib")
    # 必须调用：>=5.0 时迁移/删除弃用字段（stochastic、init_noise_std、policy ...）
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # ---------------- 4. 日志目录 ----------------
    if args_cli.log_root is not None:
        log_root_path = os.path.abspath(args_cli.log_root)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_name += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_name)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg.seed = agent_cfg.seed
    env_cfg.log_dir = log_dir

    # ---------------- 5. 续训 checkpoint ----------------
    resume_path: str | None = None
    if agent_cfg.resume:
        resume_path = resolve_checkpoint(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        # 兼容 rsl_rl < 5.0 的旧存档布局（新存档是 actor_state_dict/critic_state_dict，无需转换）
        resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
        print(f"[INFO] 续训 checkpoint：{resume_path}")

    print("=" * 78)
    print(f"[INFO] 任务              : {args_cli.task}")
    print(f"[INFO] 环境数            : {env_cfg.scene.num_envs}")
    print(f"[INFO] 设备              : {device}")
    print(f"[INFO] rsl-rl 版本       : {installed_version}")
    print(f"[INFO] 迭代次数          : {agent_cfg.max_iterations}")
    print(f"[INFO] 日志目录          : {log_dir}")
    print("=" * 78)

    # ---------------- 6. 创建环境 / runner ----------------
    # 注意：gym.make 的 cfg 以关键字传入 ManagerBasedRLEnv；registry 里的
    # env_cfg_entry_point / rsl_rl_cfg_entry_point 会被 **kwargs 吞掉。
    env = gym.make(args_cli.task, cfg=env_cfg)
    # 保留裸环境引用（用于续训时对齐全局步数；RslRlVecEnvWrapper 不是 gym.Wrapper）
    raw_env = env.unwrapped
    # NaN 防护：过滤 obs/reward 中的 NaN/Inf（环境层另有 state_invalid 终止项）
    from chassis_rl_lab.utils.wrappers import SanitizeWrapper

    env = SanitizeWrapper(env)
    # RslRlVecEnvWrapper 必须是最后一个 wrapper（它不是 gym.Wrapper）；
    # clip_actions 在 wrapper 内对动作裁剪。
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # 注意：不要调用 runner.add_git_repo_to_log(__file__)。
    # rsl_rl 5.0.1 的 Logger.init_logging_writer() 会对所有注册仓库执行 git diff 并
    # 以 UTF-8 写入日志；本仓库含 GBK 文本（底盘数据.txt）与二进制资产，diff 解码会得到
    # surrogate，写入时直接 UnicodeEncodeError 中断训练。git 状态与训练无关，故跳过。
    print("[INFO] 跳过 rsl_rl 的 git code-state 记录（仓库含 GBK/二进制文件，会触发 UnicodeEncodeError）")

    if resume_path is not None:
        print(f"[INFO] 加载 checkpoint：{resume_path}")
        runner.load(resume_path)

        # 续训：把动作项的全局步数 / 环境的 common_step_counter 对齐到 checkpoint 迭代数，
        # 这样"辅助力退火课程"和"倒地方向课程"会接着上次进度走，而不是在新 run 里从 0 重来。
        try:
            resumed_iter = int(getattr(runner, "current_learning_iteration", 0) or 0)
            steps_per_iter = int(agent_cfg.num_steps_per_env or 0)
            total_steps = resumed_iter * steps_per_iter
            raw_env.common_step_counter = total_steps
            action_term = raw_env.action_manager.get_term("wheel_leg")
            if hasattr(action_term, "_global_steps"):
                action_term._global_steps = total_steps
            print(
                f"[INFO] 续训进度对齐    : iter {resumed_iter} × {steps_per_iter} = {total_steps} 控制步"
                f"（辅助退火/方向课程接着走）"
            )
            # 续训时让高度指令 curriculum 从"续训时刻"重新开始：
            # 否则 common_step_counter 已超过 curriculum_steps → 续训第一轮就给满高度范围。
            try:
                height_term_cfg = raw_env.event_manager.get_term_cfg("reset_height_command")
                if int(height_term_cfg.params.get("curriculum_steps", 0)) > 0:
                    height_term_cfg.params["curriculum_start_steps"] = total_steps
                    print(
                        f"[INFO] 高度课程从续训点开始放开: start={total_steps} 控制步, "
                        f"curriculum_steps={height_term_cfg.params.get('curriculum_steps')}"
                    )
            except Exception:  # noqa: BLE001 - 没有高度指令的任务
                pass
        except Exception:  # noqa: BLE001 - 对齐失败不影响训练
            print("[WARN] 续训进度对齐失败（不影响训练）：")
            traceback.print_exc()



    # ---------------- 7. 落盘配置 ----------------
    dump_yaml(os.path.join(log_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "agent.yaml"), agent_cfg)
    print(f"[INFO] 已保存配置：{os.path.join(log_dir, 'env.yaml')} / agent.yaml")

    # ---------------- 8. 训练（分块 + 周期性导出累计数据） ----------------
    # 控制台打印精简：保留 rsl_rl 每轮输出的 "Mean value loss" 及以下部分（去掉 Total steps/速度/耗时），
    # 并在 Mean action std 后插入 Height target / Height now / Height error 三行；
    # TensorBoard 记录保持完整（原 logger.log 仍执行，只是过滤其控制台输出）
    from chassis_rl_lab.utils.logging_utils import make_compact_logger_log

    planned_total_iter = int(getattr(runner, "current_learning_iteration", 0) or 0) + int(
        agent_cfg.max_iterations
    )
    runner.logger.log = make_compact_logger_log(
        runner, raw_env, total_iterations=planned_total_iter
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_root = Path(args_cli.data_root) if args_cli.data_root else PROJECT_ROOT / "data"
    data_extra = {
        "task": args_cli.task,
        "num_envs": int(env_cfg.scene.num_envs),
        "max_iterations": int(agent_cfg.max_iterations),
        "device": str(device),
    }

    def _export_data(stage: str = "") -> None:
        """导出"从训练开始到当前"的累计数据（CSV + 图像 + 摘要），可周期性调用。"""
        if args_cli.no_data_export:
            return
        try:
            from chassis_rl_lab.utils.training_data import export_training_data

            exported = export_training_data(
                log_dir,
                data_root=data_root,
                extra={**data_extra, "export_stage": stage},
                snapshot=True,  # 周期导出：在 data/<run>/snapshot_<时间戳>/ 下保存快照（不覆盖历史）
            )
            print(f"[INFO] 训练数据已导出（{stage}）：{exported}")
        except Exception:  # noqa: BLE001 - 数据导出失败不应影响训练
            print("[WARN] 训练数据导出失败（不影响训练）：")
            traceback.print_exc()

    total_iters = int(agent_cfg.max_iterations)
    interval = int(getattr(args_cli, "export_interval", 0) or 0)
    chunk = interval if interval > 0 else total_iters
    trained = 0
    try:
        while trained < total_iters:
            n = min(chunk, total_iters - trained)
            # init_at_random_ep_len 只在第一块打开（后续块延续当前 episode 长度）
            runner.learn(num_learning_iterations=n, init_at_random_ep_len=(trained == 0))
            trained += n
            if trained < total_iters:
                # rsl_rl 的 current_learning_iteration 存的是"最后一轮的下标"（不是累计轮数），
                # 分块调用时必须 +1，否则下一块会重复训练同一轮
                runner.current_learning_iteration += 1
                _export_data(f"iter {int(getattr(runner, 'current_learning_iteration', 0))}（累计）")
    except KeyboardInterrupt:
        # 规格 §7：Ctrl+C 时保存（只保存已完成的最后一次迭代；当前 rollout 丢弃）
        iteration = int(getattr(runner, "current_learning_iteration", 0))
        save_path = os.path.join(log_dir, f"model_{iteration}.pt")
        try:
            runner.save(save_path)
            print(f"[INFO] 收到 Ctrl+C，已保存模型：{save_path}")
        except Exception:  # noqa: BLE001 - 保存失败不应掩盖 Ctrl+C
            print("[ERROR] Ctrl+C 后保存模型失败：")
            traceback.print_exc()
    finally:
        env.close()

    # ---------------- 9. 结束导出（最终累计数据） ----------------
    _export_data("结束")
    print(f"[INFO] 训练结束，日志目录：{log_dir}")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:  # noqa: BLE001 - 统一打印 traceback 便于 Lead 排错
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit 退出时偶发线程 hang；与旧项目一致，强制结束进程
        os._exit(exit_code)
