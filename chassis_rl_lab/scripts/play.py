"""回放 / 评估 chassis-rl 五连杆轮腿机器人的蹲姿保持平衡策略（RSL-RL）。

用法（Windows，`D:\\miniconda\\envs\\RM\\python.exe`）::

    # GUI 回放最新一次训练的模型（默认 16 个环境、10 个 episode）
    python chassis_rl_lab\\scripts\\play.py

    # 指定 checkpoint 与 headless 批量评估
    python chassis_rl_lab\\scripts\\play.py --headless --checkpoint logs\\rsl_rl\\chassis_rl_stand_up\\<run>\\model_500.pt --episodes 20

脚本接口（规格 §7）：
    --task（默认 Chassis-Stand-Up-v0）、--checkpoint（默认 ``-1`` = 最新 run 的最新
    checkpoint；也可传文件路径或正则）、--num_envs（默认 16）、--headless（由 AppLauncher 提供）。
    加载 checkpoint 后跑若干 episode，打印平均回报 / 平均时长；GUI 模式下会打开 Isaac Sim 视口，
    实时渲染（关闭窗口或跑满 --episodes 后退出）。

注意：
    * 观测是 **TensorDict（按 obs group）**，不是扁平 tensor —— 这是 rsl_rl 5.0.1 +
      ``RslRlVecEnvWrapper`` 的接口（见 `_design/01_reference_notes.md` §2.13）。
    * 检查点布局为 rsl_rl >= 5.0 的 ``actor_state_dict/critic_state_dict``；若给的是旧布局
      存档，会先经 ``handle_deprecated_rsl_rl_checkpoint`` 自动转换。
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import traceback
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1) 项目根加入 sys.path
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# 2) argparse 阶段：只能 import AppLauncher
# --------------------------------------------------------------------------- #
from isaaclab.app import AppLauncher  # noqa: E402

DEFAULT_TASK = "Chassis-Stand-Up-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play/evaluate a trained RSL-RL policy for the chassis-rl squat-pose balance task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="gym 注册的任务 id")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="-1",
        help="checkpoint 路径 / 文件名正则；'-1' = 最新 run 的最新 checkpoint",
    )
    parser.add_argument("--num_envs", type=int, default=16, help="并行环境数（回放不必用训练时的规模）")
    parser.add_argument("--episodes", type=int, default=10, help="统计多少个 episode 后退出")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="每个 episode 的最大控制步数（默认 None = 环境 cfg：1000 步 / 20 s；500 步 = 10 s）",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        default=False,
        help="关闭观测噪声（训练配置默认开启 BMI088 IMU 噪声；评估/回放通常关掉）",
    )
    parser.add_argument(
        "--no-assist",
        action="store_true",
        default=False,
        help="关闭平衡辅助力矩（评估'无辅助'的从倒地到起立；训练配置里 balance_assist 默认开）",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="打印第一个 episode 的逐步姿态（pitch/roll/base_z；Height 任务额外打印 h_cmd/h_err）",
    )
    parser.add_argument(
        "--height_sweep",
        action="store_true",
        default=False,
        help="Height 任务：周期性在最低/最高目标高度之间切换（范围取任务 cfg），便于肉眼看到变高/变矮",
    )
    parser.add_argument(
        "--sweep_interval",
        type=int,
        default=100,
        help="--height_sweep 的切换间隔（控制步，50 Hz；默认 100 步 = 2 s）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（None = 用 agent cfg 默认）")
    parser.add_argument("--experiment_name", type=str, default=None, help="覆盖 agent cfg 的 experiment_name")
    parser.add_argument("--load_run", type=str, default=None, help="run 目录（正则/目录名）；配合 --checkpoint -1")
    parser.add_argument("--log_root", type=str, default=None, help="日志根目录（默认 <项目根>/logs/rsl_rl）")
    # AppLauncher 提供 --headless / --device / --enable_cameras 等
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()

# --------------------------------------------------------------------------- #
# 3) 启动 Isaac Sim
# --------------------------------------------------------------------------- #
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
# 4) Kit 启动后的 import
# --------------------------------------------------------------------------- #
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

try:
    import chassis_rl_lab  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "无法 import `chassis_rl_lab`：请确认 chassis_rl_lab/chassis_rl_lab/__init__.py 已实现 gym.register。"
    ) from exc


# --------------------------------------------------------------------------- #
# 5) 工具函数（与 train.py 同语义，脚本各自独立以保持单文件可读）
# --------------------------------------------------------------------------- #
def load_cfg_from_registry(task: str, entry_point_key: str):
    import importlib

    try:
        spec = gym.spec(task.split(":")[-1])
    except gym.error.Error as exc:
        raise RuntimeError(f"任务 {task!r} 尚未在 gym 注册。") from exc
    entry_point = spec.kwargs.get(entry_point_key)
    if entry_point is None:
        raise ValueError(f"任务 {task!r} 缺少 registry kwarg {entry_point_key!r}")
    if callable(entry_point):
        return entry_point()
    module_name, class_name = entry_point.split(":")
    return getattr(importlib.import_module(module_name), class_name)()


def _checkpoint_sort_key(filename: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", filename)
    return (int(match.group(1)) if match else -1, filename)


def resolve_checkpoint(log_root: str, load_run: str = ".*", load_checkpoint: str = "model_.*.pt") -> str:
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
    import importlib.metadata as metadata

    installed_version = metadata.version("rsl-rl-lib")

    # ---------------- 1. 读取 cfg ----------------
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # ---------------- 2. 命令行覆盖 ----------------
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.seed is not None:
        agent_cfg.seed = int(args_cli.seed)
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run

    device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.device = device
    agent_cfg.device = device

    # 必须调用：清理 rsl_rl 5.0 已弃用字段，否则 to_dict() 里残留 MISSING
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.log_dir = None  # 回放不写日志目录
    # --no-noise：回放/评估关闭观测噪声（默认训练配置里是开启 BMI088 IMU 噪声的）
    if args_cli.no_noise:
        env_cfg.observations.policy.enable_corruption = False
        print("[INFO] 观测噪声      : 已关闭（--no-noise）")
    else:
        print("[INFO] 观测噪声      : 按训练配置（BMI088 IMU 噪声开启）")

    # --no-assist：评估"无辅助"起立。训练配置里 balance_assist=True 且退火会让 scale 从 1.0
    # 重新开始，所以评估时必须显式关掉，否则等于带着辅助跑。
    if args_cli.no_assist:
        action_term_cfg = getattr(getattr(env_cfg, "actions", None), "wheel_leg", None)
        if action_term_cfg is not None and getattr(action_term_cfg, "balance_assist", False):
            action_term_cfg.balance_assist = False
            print("[INFO] 平衡辅助      : 已关闭（--no-assist：纯策略从倒地到起立）")
        else:
            print("[INFO] 平衡辅助      : 该任务未启用辅助，--no-assist 无效果")

    # ---------------- 3. 解析 checkpoint ----------------
    if args_cli.log_root is not None:
        log_root_path = os.path.abspath(args_cli.log_root)
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    resume_path = resolve_checkpoint(
        log_root_path,
        load_run=agent_cfg.load_run,
        load_checkpoint=args_cli.checkpoint,
    )
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)

    print("=" * 78)
    print(f"[INFO] 任务              : {args_cli.task}")
    print(f"[INFO] 环境数            : {env_cfg.scene.num_envs}")
    print(f"[INFO] 设备              : {device}")
    print(f"[INFO] rsl-rl 版本       : {installed_version}")
    print(f"[INFO] checkpoint        : {resume_path}")
    print(f"[INFO] 统计 episode 数   : {args_cli.episodes}")
    print("=" * 78)

    # ---------------- 4. 创建环境 / 加载策略 ----------------
    # --steps：覆盖每个 episode 的最大控制步数。
    # 注意 Isaac Lab 的 env.max_episode_length 是只读 property（由 cfg.episode_length_s 推导），
    # 所以要在 gym.make 之前改 episode_length_s，而不是运行后赋值。
    step_dt_cfg = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    if args_cli.steps is not None:
        if int(args_cli.steps) <= 0:
            raise ValueError(f"--steps 必须 > 0，收到 {args_cli.steps}")
        env_cfg.episode_length_s = int(args_cli.steps) * step_dt_cfg
    # 播放/评估时**关闭高度指令 curriculum**：新进程的 env.common_step_counter 从 0 开始，
    # 否则 reset_height_command 的 frac=0 → 高度指令恒为名义高度（看起来"机器人不会变高度"）。
    try:
        reset_height_term = env_cfg.events.reset_height_command
        if reset_height_term is not None and reset_height_term.params.get("curriculum_steps", 0):
            reset_height_term.params["curriculum_steps"] = 0
            print("[INFO] 高度指令 curriculum 已关闭：每个 episode 随机目标高度")
    except Exception:  # noqa: BLE001 - 没有高度指令的任务
        pass
    # 高度扫描范围（读取当前任务 cfg：Height 任务为 0.172~0.270）
    sweep_lo, sweep_hi = 0.172, 0.190
    try:
        sweep_lo, sweep_hi = (float(v) for v in env_cfg.height_command_range)
    except Exception:  # noqa: BLE001 - 非 Height 任务无该字段
        pass
    if args_cli.height_sweep:
        print(
            f"[INFO] 高度指令扫描  : 每 {int(args_cli.sweep_interval)} 步在 {sweep_lo:.3f} / {sweep_hi:.3f} 之间切换"
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    print(
        f"[INFO] 每 episode 最大步数: {int(env.unwrapped.max_episode_length)} "
        f"(episode_length_s={float(env_cfg.episode_length_s):.3f}s, step_dt={step_dt_cfg:.4f}s)"
    )
    from chassis_rl_lab.utils.wrappers import SanitizeWrapper

    env = SanitizeWrapper(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    # get_inference_policy 会切 eval 模式；返回 MLPModel（forward 输出分布均值）
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    num_envs = env.num_envs
    step_dt = float(env.unwrapped.step_dt)

    # 姿态统计（真值，来自仿真状态；不受观测噪声影响）
    # 机体系重力 g_b 与 pitch/roll 的关系：g_b = (sin(pitch), -sin(roll), -cos) ≈>
    #   pitch = asin(g_b.x)，roll = -asin(g_b.y)（小角度即 pitch≈g_b.x、roll≈-g_b.y）
    robot = env.unwrapped.scene["robot"]
    pitch_abs_hist: list[float] = []
    roll_abs_hist: list[float] = []
    pitch_abs_max = 0.0
    roll_abs_max = 0.0

    episode_rewards = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    finished_returns: list[float] = []
    finished_lengths: list[int] = []
    last_print = 0

    try:
        with torch.inference_mode():
            sweep_step = 0
            while simulation_app.is_running() and len(finished_returns) < int(args_cli.episodes):
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)
                # MLP 无内部状态（RNN 才需要），保持与官方 play.py 一致的调用
                policy.reset(dones)

                # --height_sweep：周期性切换目标高度，方便肉眼看到"变高/变矮"
                if args_cli.height_sweep:
                    sweep_step += 1
                    sweep_interval = max(1, int(args_cli.sweep_interval))
                    sweep_target = sweep_hi if (sweep_step // sweep_interval) % 2 == 0 else sweep_lo
                    try:
                        env.unwrapped.height_command[:] = sweep_target
                    except Exception:  # noqa: BLE001 - 非 Height 任务没有该属性
                        pass

                episode_rewards += rewards
                episode_lengths += 1

                # 姿态统计（每步全部环境的平均 |pitch| / |roll|）
                g_b = robot.data.projected_gravity_b
                pitch_abs = torch.asin(torch.clamp(g_b[:, 0], -1.0, 1.0)).abs()
                roll_abs = torch.asin(torch.clamp(-g_b[:, 1], -1.0, 1.0)).abs()
                pitch_abs_hist.append(float(pitch_abs.mean().item()))
                roll_abs_hist.append(float(roll_abs.mean().item()))
                pitch_abs_max = max(pitch_abs_max, float(pitch_abs.max().item()))
                roll_abs_max = max(roll_abs_max, float(roll_abs.max().item()))

                # --trace：打印第一个 episode 的逐步姿态（env 0），看"倒地→起立"过程；
                # Height 任务还会打印指令高度 h_cmd 与实测高度误差 h_err（判断"变腿高"是否生效）
                if args_cli.trace and not finished_returns:
                    step_idx = int(episode_lengths[0].item())
                    if step_idx % 10 == 0:
                        g0 = g_b[0]
                        pitch_deg = math.degrees(math.asin(float(torch.clamp(g0[0], -1.0, 1.0))))
                        roll_deg = math.degrees(math.asin(float(torch.clamp(-g0[1], -1.0, 1.0))))
                        height = float(robot.data.root_pos_w[0, 2].item())
                        height_cmd_attr = getattr(env.unwrapped, "height_command", None)
                        if height_cmd_attr is not None:
                            h_cmd = float(height_cmd_attr[0, 0].item())
                            h_str = f"  h_cmd={h_cmd:.4f}  h_err={height - h_cmd:+.4f}"
                        else:
                            h_str = ""
                        print(
                            f"[trace] step={step_idx:4d} t={step_idx * step_dt:5.2f}s  "
                            f"pitch={pitch_deg:+6.2f}°  roll={roll_deg:+6.2f}°  base_z={height:.4f}{h_str}"
                        )

                done_ids = torch.nonzero(dones, as_tuple=False).flatten().tolist()
                for env_id in done_ids:
                    ret = float(episode_rewards[env_id].item())
                    length = int(episode_lengths[env_id].item())
                    finished_returns.append(ret)
                    finished_lengths.append(length)
                    episode_rewards[env_id] = 0.0
                    episode_lengths[env_id] = 0
                    print(
                        f"[ep {len(finished_returns):03d}] return={ret:+.3f}  "
                        f"steps={length:4d}  duration={length * step_dt:.2f}s"
                    )

                if len(finished_returns) - last_print >= 50:
                    last_print = len(finished_returns)
                    print(f"[INFO] 已统计 {len(finished_returns)} 个 episode ...")
    finally:
        env.close()

    # ---------------- 5. 汇总 ----------------
    if not finished_returns:
        print("[WARN] 没有完成任何 episode（仿真被提前关闭？）")
        return 1

    import statistics

    mean_return = statistics.fmean(finished_returns)
    std_return = statistics.pstdev(finished_returns) if len(finished_returns) > 1 else 0.0
    mean_steps = statistics.fmean(finished_lengths)
    mean_duration = mean_steps * step_dt

    print("=" * 78)
    print(f"[RESULT] episode 数        : {len(finished_returns)}")
    print(f"[RESULT] 平均回报          : {mean_return:+.3f} ± {std_return:.3f}")
    print(f"[RESULT] 平均步数          : {mean_steps:.1f} (step_dt={step_dt:.4f}s)")
    print(f"[RESULT] 平均时长          : {mean_duration:.2f}s")
    print(f"[RESULT] 最大/最小时长     : {max(finished_lengths)} / {min(finished_lengths)} 步")
    if pitch_abs_hist:
        deg = 180.0 / math.pi
        print(
            f"[RESULT] 平均 |pitch|       : {statistics.fmean(pitch_abs_hist) * deg:.3f}°"
            f"  (最大 {pitch_abs_max * deg:.3f}°)"
        )
        print(
            f"[RESULT] 平均 |roll|        : {statistics.fmean(roll_abs_hist) * deg:.3f}°"
            f"  (最大 {roll_abs_max * deg:.3f}°)"
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        exit_code = 1
    finally:
        # Windows 上 Isaac Sim 的 close() 偶发挂起（异常路径尤其明显），只 flush 后强制退出；
        # 与 train.py / 工具脚本的退出方式保持一致。
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
