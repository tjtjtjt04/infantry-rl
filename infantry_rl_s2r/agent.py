"""RSL-RL runner configuration and checkpoint helpers.

The flat-layout equivalent of the ``tasks/*/agents/rsl_rl_ppo_cfg.py`` module plus
the checkpoint / network-shape / exploration-noise helpers that used to live inline
in ``scripts/train.py`` and ``scripts/play.py``.

Training still uses Isaac Lab's ``OnPolicyRunner`` and ``RslRlVecEnvWrapper``, so
existing rsl_rl checkpoints keep loading.  All hyperparameters, hidden dims,
activation, obs-normalisation flags and the PPO hyperparameters are copied from the
original workspace; the console is a purpose-built compact monitoring panel.
"""

from __future__ import annotations

import glob as _glob
import math
import os
import re as _re
import statistics
import time
import sys

import torch

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from config import EXPERIMENT_NAMES

#: initial exploration std (raw action units); reference wheel_legged uses 0.5
INIT_STD = float(os.environ.get("INFANTRY_RL_INIT_STD", "0.5"))
#: bounds applied to the learned std parameter
STD_MIN = float(os.environ.get("INFANTRY_RL_POLICY_STD_MIN", "0.1"))
STD_MAX = float(os.environ.get("INFANTRY_RL_POLICY_STD_MAX", "1.0"))

# --------------------------------------------------------------------------- #
# Actor: 5-frame history encoder + current observation (reference structure)
# --------------------------------------------------------------------------- #
@configclass
class HistoryEncoderActorCfg(RslRlMLPModelCfg):
    """25 current dims + 3 latent from a 5 x 25 history encoder."""

    class_name: str = "policy_model:HistoryEncoderActor"
    encoder_hidden_dims: list[int] = [128, 64]
    latent_dim: int = 3
    current_dim: int = 25
    history_length: int = 5
    frame_dim: int = 25


# --------------------------------------------------------------------------- #
# PPO for the reference joint-space squat (4 hip targets + 2 wheel velocities)
# --------------------------------------------------------------------------- #
@configclass
class InfantryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO settings matching the reference Wheel-Legged-Gym task.

    Exploration std 0.5 is important: 0.1 * 10 rad/s = 1 rad/s of wheel-speed
    exploration is too small to discover the balance; the reference learns with
    0.5.  Entropy 0.01 / gamma 0.99 / desired_kl 0.005 are also copied.
    """

    seed = 42
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = EXPERIMENT_NAMES["squat_torque"]
    run_name = ""
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    #: raw action units (hip position offsets + wheel velocity targets)
    clip_actions = 4.0

    actor = HistoryEncoderActorCfg(
        hidden_dims=[128, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=INIT_STD),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )


@configclass
class InfantryPPORunnerCfg_PLAY(InfantryPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 0
        self.num_steps_per_env = 24


def make_agent_cfg(task: str, play: bool = False) -> RslRlOnPolicyRunnerCfg:
    """Return a fresh runner config for a task (``squat_torque`` / ``getup_torque``).

    ``getup_torque`` uses the same network and PPO settings on purpose: the
    observation and action layouts are identical, so a ``squat_torque`` checkpoint
    loads into it unchanged and keeps its balance behaviour while it learns recovery.
    """
    if task not in EXPERIMENT_NAMES:
        raise ValueError(f"unknown task {task!r}; expected one of {tuple(EXPERIMENT_NAMES)}")
    cfg = InfantryPPORunnerCfg_PLAY() if play else InfantryPPORunnerCfg()
    cfg.experiment_name = EXPERIMENT_NAMES[task]
    return cfg


def apply_cli_overrides(agent_cfg: RslRlOnPolicyRunnerCfg, args) -> RslRlOnPolicyRunnerCfg:
    """Apply the CLI fields shared by train.py and play.py (same semantics as cli_args.py)."""
    if getattr(args, "seed", None) is not None:
        agent_cfg.seed = args.seed
    if getattr(args, "resume", None) is not None:
        agent_cfg.resume = args.resume
    if getattr(args, "load_run", None) is not None:
        agent_cfg.load_run = args.load_run
    if getattr(args, "checkpoint", None) is not None:
        agent_cfg.load_checkpoint = args.checkpoint
    if getattr(args, "experiment_name", None) is not None:
        agent_cfg.experiment_name = args.experiment_name
    if getattr(args, "run_name", None) is not None:
        agent_cfg.run_name = args.run_name
    if getattr(args, "logger", None) is not None:
        agent_cfg.logger = args.logger
    if getattr(args, "log_project_name", None) and agent_cfg.logger in {"wandb", "neptune"}:
        agent_cfg.wandb_project = args.log_project_name
        agent_cfg.neptune_project = args.log_project_name
    return agent_cfg


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def hidden_dims_from_state_dict(state_dict: dict) -> list[int]:
    """Recover ``[h1, h2, ...]`` from the ``mlp.<i>.weight`` entries of a model state dict.

    rsl_rl stores the MLP as ``Linear -> ELU -> Linear -> ... -> Linear``, so the
    hidden widths are the ``out_features`` of every layer except the output one.
    """
    widths: dict[int, int] = {}
    for key, value in state_dict.items():
        if not key.startswith("mlp.") or not key.endswith(".weight"):
            continue
        parts = key.split(".")
        if len(parts) != 3:
            continue
        try:
            index = int(parts[1])
        except ValueError:
            continue
        widths[index] = int(value.shape[0])
    ordered = [widths[i] for i in sorted(widths)]
    return ordered[:-1] if len(ordered) > 1 else []


def encoder_dims_from_state_dict(state_dict: dict) -> tuple[list[int], int]:
    """Recover ``(encoder_hidden_dims, latent_dim)`` from an actor state dict."""
    widths = []
    for key, value in state_dict.items():
        if key.startswith("encoder.") and key.endswith(".weight"):
            widths.append((int(key.split(".")[1]), value))
    if not widths:
        return [], 0
    widths.sort(key=lambda item: item[0])
    shapes = [tuple(w.shape) for _, w in widths]
    return [shape[0] for shape in shapes[:-1]], int(shapes[-1][0])


def match_network_to_checkpoint(checkpoint_path: str, agent_cfg) -> None:
    """Patch ``agent_cfg`` so a resumed model matches the checkpoint it loads.

    Changing ``hidden_dims`` in the config makes every existing checkpoint unloadable.
    When resuming, the checkpoint's architecture is the correct one, so read the
    widths back from the state dict and adopt them.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - never let inspection break training
        print(f"[INFO] could not inspect the checkpoint architecture ({exc}); using config")
        return

    changed = False
    for name in ("actor", "critic"):
        state_dict = ckpt.get(f"{name}_state_dict")
        if not isinstance(state_dict, dict):
            continue
        dims = hidden_dims_from_state_dict(state_dict)
        if not dims:
            continue
        current = list(getattr(agent_cfg, name).hidden_dims or [])
        if current != dims:
            print(f"[INFO] {name} hidden_dims: config {current} -> checkpoint {dims} (resuming with the checkpoint shape)")
            getattr(agent_cfg, name).hidden_dims = dims
            changed = True

    actor_sd = ckpt.get("actor_state_dict") or {}
    if actor_sd:
        enc_hidden, latent_dim = encoder_dims_from_state_dict(actor_sd)
        first_weight = actor_sd.get("mlp.0.weight")
        if enc_hidden and latent_dim and first_weight is not None:
            current_dim = int(first_weight.shape[1]) - latent_dim
            cfg_actor = agent_cfg.actor
            old = (
                list(getattr(cfg_actor, "encoder_hidden_dims", []) or []),
                int(getattr(cfg_actor, "latent_dim", 0)),
                int(getattr(cfg_actor, "current_dim", 0)),
            )
            new = (enc_hidden, latent_dim, current_dim)
            if old != new:
                print(
                    f"[INFO] encoder: config {old} -> checkpoint {new} "
                    f"(encoder_hidden_dims, latent_dim, current_dim)"
                )
                cfg_actor.encoder_hidden_dims = enc_hidden
                cfg_actor.latent_dim = latent_dim
                cfg_actor.current_dim = current_dim
                changed = True
            obs_dim = actor_sd.get("obs_normalizer._mean")
            if obs_dim is not None:
                print(
                    f"[INFO] checkpoint history dim = {int(obs_dim.shape[-1])}; "
                    f"actor input = {int(first_weight.shape[1])} "
                    f"(env history must match)"
                )
    if not changed:
        print("[INFO] checkpoint architecture matches the configured network")


#: smallest normalizer count that is safe to reset to on resume (see the docstring).
OBS_NORM_MIN_RESET_COUNT = 5_000_000


def reset_obs_normalizer_count(runner, count: int = 10_000_000) -> None:
    """Re-open the running observation statistics when resuming into a new task envelope.

    WHY THIS IS NEEDED WHEN WIDENING THE COMMAND RANGE
    -------------------------------------------------
    ``EmpiricalNormalization.update`` moves the mean/var by ``rate = batch / count``.
    The 1 m/s checkpoint carries ``count = 5.4e7``, so each update moves the statistics
    by ~2e-5 -- they are effectively frozen over one stage.  At 3 m/s the
    ``velocity_commands`` input is ~7 sigma outside those saved statistics, so a slow
    adaptation is desirable, but not one that is fast relative to the rollout.

    WHY A SMALL COUNT IS DANGEROUS
    ------------------------------
    rsl_rl updates the normalizer EVERY env step (``PPO.process_env_step``), while
    PPO stores the *raw* observations and the log-probs computed at sample time.  If
    the mean/std move appreciably inside one rollout, the update re-normalises the
    stored observations with DIFFERENT statistics than the ones that produced the
    stored log-probs -- the importance ratio is then off by a large factor and the
    policy update is garbage.  The symptom is unmistakable: the value loss spikes
    10x on the first update and the fall rate goes 0 -> 90% in a handful of
    iterations (measured with count=200k, which drifts ~25% in one 48-step rollout).

    Keeping the count at >= 5e6 bounds the within-rollout drift to <1%.  If in doubt,
    do not reset at all: even the untouched 5.4e7 adapts by ~50% over an 800-iteration
    stage.
    """
    count = int(count)
    if count < OBS_NORM_MIN_RESET_COUNT:
        print(
            f"[WARN] --reset_obs_norm_count {count} is too small: the observation "
            f"statistics would drift >1% WITHIN one rollout and corrupt PPO's "
            f"importance ratios.  Using {OBS_NORM_MIN_RESET_COUNT} instead "
            f"(pass 0 / omit the flag to leave the count untouched)."
        )
        count = OBS_NORM_MIN_RESET_COUNT
    reset = 0
    for name in ("actor", "critic"):
        model = getattr(runner.alg, name, None)
        norm = getattr(model, "obs_normalizer", None)
        counter = getattr(norm, "count", None)
        if counter is None:
            continue
        old = int(counter.item())
        with torch.no_grad():
            counter.fill_(count)
        reset += 1
        print(f"[INFO] {name}.obs_normalizer.count: {old} -> {count}")
    if reset == 0:
        print("[INFO] --reset_obs_norm_count: no empirical normalizer found; ignored")


def newest_checkpoint(log_root_path: str, load_run: str | None = None) -> str:
    """The most-trained ``model_<N>.pt`` of an experiment, across ALL run directories.

    Two traps this helper avoids:

    1. ``get_checkpoint_path`` sorts file names as *strings*, so ``model_0.pt`` sorts
       AFTER ``model_900.pt`` -- a plain ``--resume`` would restart from an untrained
       policy.
    2. Continuing a run creates a **new timestamped directory** that starts out
       containing only the checkpoint it just loaded.  Picking "the newest directory"
       is therefore wrong right after a resume.  Instead, this picks the newest run
       directory that has any checkpoints, then the highest iteration inside it
       (iteration numbers are only comparable inside one lineage).
    """
    if load_run in (None, "", ".*"):
        run_dirs = _glob.glob(os.path.join(log_root_path, "*"))
    else:
        pattern = load_run if any(c in load_run for c in "*?[]") else f"*{load_run}*"
        run_dirs = _glob.glob(os.path.join(log_root_path, pattern))

    candidates: list[tuple[int, str, str]] = []
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        for path in _glob.glob(os.path.join(run_dir, "model_*.pt")):
            m = _re.fullmatch(r"model_(\d+)\.pt", os.path.basename(path))
            if m is not None:
                candidates.append((int(m.group(1)), path, os.path.basename(run_dir)))

    if not candidates:
        raise FileNotFoundError(
            f"--resume_latest found no model_*.pt under '{log_root_path}'"
            + (f" matching '{load_run}'" if load_run else "")
        )

    by_run: dict[str, list[tuple[int, str]]] = {}
    for iteration, path, run_name in candidates:
        by_run.setdefault(run_name, []).append((iteration, path))

    def _mtime(run_name: str) -> float:
        try:
            return os.path.getmtime(os.path.join(log_root_path, run_name))
        except OSError:
            return 0.0

    newest_run = max(by_run, key=_mtime)
    iteration, path = max(by_run[newest_run], key=lambda c: c[0])

    global_iteration, _, global_run = max(candidates, key=lambda c: c[0])
    print(f"[INFO] --resume_latest: newest run '{newest_run}' -> iteration {iteration}")
    if global_run != newest_run:
        print(
            f"[INFO] NOTE: another run ('{global_run}') has a higher iteration count "
            f"({global_iteration}), but it is OLDER.  It is ignored on purpose; pass "
            f"--load_run {global_run} --checkpoint <file> if you really want it."
        )
    return os.path.abspath(path)


def resolve_play_checkpoint(
    log_root_path: str,
    load_run: str | None,
    requested: str | None,
    use_last_checkpoint: bool = False,
) -> str:
    """Resolve ``--checkpoint`` the same way the original play.py did.

    Accepts, in order: an existing file path (absolute or relative to CWD), a path
    relative to the log root / run dir, or a bare filename/glob resolved inside the
    newest matching run directory.
    """
    from isaaclab_tasks.utils import get_checkpoint_path

    pattern = "model_.*.pt" if (use_last_checkpoint or not requested) else str(requested)
    direct = os.path.abspath(pattern)
    if os.path.isfile(direct):
        return direct
    if os.path.sep in pattern or "/" in pattern:
        candidates = [
            os.path.join(log_root_path, pattern),
            os.path.join(log_root_path, load_run or "", pattern),
            os.path.join(os.getcwd(), pattern),
        ]
        hit = next((c for c in candidates if os.path.isfile(c)), None)
        if hit is None:
            raise FileNotFoundError(
                f"no checkpoint at '{pattern}'; tried:\n  " + "\n  ".join(os.path.abspath(c) for c in candidates)
            )
        return hit
    return get_checkpoint_path(
        log_root_path,
        load_run,
        f"^{_re.escape(pattern)}$" if ("*" not in pattern and "?" not in pattern) else pattern,
    )


# --------------------------------------------------------------------------- #
# Policy exploration-noise clamp
# --------------------------------------------------------------------------- #
def clamp_policy_std(runner, low: float | None = None, high: float | None = None) -> None:
    """Bound the Gaussian exploration noise of the policy in place.

    rsl_rl's default ``GaussianDistribution`` keeps the standard deviation as a raw
    unconstrained ``nn.Parameter``.  With ``std_type="scalar"`` the entropy term alone
    gives it a gradient that is always positive; if the policy-gradient side does not
    push back hard enough, sigma grows without bound (measured: 0.50 -> 33.7 -> 168.9).
    At that point every action is uniform noise and the robot thrashes.
    The opposite failure is real too: with a small entropy coefficient the noise
    collapsed to 0.067 and the policy could not discover walking.  BOTH ends need a
    bound.  The clamp is applied to the parameter BEFORE the distribution is built
    each step, so the sample itself already uses the bounded value.
    """
    low = STD_MIN if low is None else float(low)
    high = STD_MAX if high is None else float(high)
    # rsl_rl keeps the policy on ``PPO.actor`` (``get_policy()`` returns it); fall back
    # to the older ``alg.policy`` / ``alg.actor`` spelling in case the naming changes.
    policy = None
    get_policy = getattr(runner.alg, "get_policy", None)
    if callable(get_policy):
        policy = get_policy()
    if policy is None:
        policy = getattr(runner.alg, "policy", None) or getattr(runner.alg, "actor", None)
    dist = getattr(policy, "distribution", None)
    std_param = getattr(dist, "std_param", None)
    if std_param is None:
        # heteroscedastic / log-parameterised policies have no single std tensor
        print("[INFO] policy std is not a single raw tensor -- no clamp applied")
        return

    original_update = dist.update

    def clamped_update(mlp_output):
        with torch.no_grad():
            dist.std_param.clamp_(low, high)
        original_update(mlp_output)

    dist.update = clamped_update
    with torch.no_grad():
        std_param.clamp_(low, high)
    print(
        f"[INFO] policy std clamped to [{low}, {high}] "
        f"(current {float(std_param.mean()):.4f})"
    )


# --------------------------------------------------------------------------- #
# Console panel
# --------------------------------------------------------------------------- #
#: Deprecated: the console now always prints the curated panel below.  Every
#: reward / termination / metric scalar is still written to TensorBoard.
HIDDEN_CONSOLE_TERMS: tuple[str, ...] = ()


def _snapshot_mean(snapshots, key: str) -> float | None:
    """Mean over the rollout of one episode-extra entry (``Episode_Reward/...``)."""
    values = []
    for snap in snapshots:
        if key in snap:
            v = snap[key]
            v = v.detach().cpu().double() if hasattr(v, "detach") else float(v)
            values.append(float(v.mean()) if hasattr(v, "mean") else float(v))
    return sum(values) / len(values) if values else None


def _fmt(v, spec: str = "6.3f") -> str:
    return f"{v:{spec}}" if v is not None and math.isfinite(v) else "  n/a"


def _reward_weights(runner) -> dict[str, float]:
    """Read the ACTIVE weights of the terms the panel inverts from the live env cfg.

    The weights are mutated by ``--yaw_scale`` / ``--heading_l1`` in the reward cfg's
    ``__post_init__``, so they must be taken from the environment, not hard-coded.
    """
    env = getattr(runner, "env", None)
    unwrapped = getattr(env, "unwrapped", None)
    cfg = getattr(unwrapped, "cfg", None)
    rewards = getattr(cfg, "rewards", None)
    names = (
        "alive",
        "flat_orientation",
        "base_height",
        "pitch",
        "roll",
        "heading",
        "yaw_rate",
        "position",
        "lin_vel",
        "action_rate",
        "joint_torques",
        "closure_residual",
        "hip_lock",
        "fall_penalty",
    )
    out: dict[str, float] = {}
    for name in names:
        term = getattr(rewards, name, None)
        w = getattr(term, "weight", None)
        if w is not None:
            out[name] = float(w)
    return out


def _command_range(runner) -> tuple[float | None, float | None]:
    """Sampled vx range of the active command term (for the console histogram labels)."""
    env = getattr(runner, "env", None)
    unwrapped = getattr(env, "unwrapped", None)
    cmd_mgr = getattr(unwrapped, "command_manager", None)
    if cmd_mgr is None:
        return None, None
    for name in getattr(cmd_mgr, "active_terms", []):
        term = cmd_mgr.get_term(name)
        rng = getattr(term.cfg, "ranges", None)
        lin = getattr(rng, "lin_vel_x", None)
        if lin is not None:
            return float(lin[0]), float(lin[1])
    return None, None


def _is_heading_task(runner) -> bool:
    """Whether the active command derives ``wz`` from a heading target.

    The squat task does (so a heading-error line is meaningful);
    samples ``wz`` directly (so that line would show a misleading 0 deg).  Unknown
    setups default to True to keep the original console output.
    """
    env = getattr(runner, "env", None)
    unwrapped = getattr(env, "unwrapped", None)
    cmd_mgr = getattr(unwrapped, "command_manager", None)
    if cmd_mgr is None:
        return True
    return any(
        getattr(getattr(cmd_mgr.get_term(name), "cfg", None), "heading_command", False)
        for name in getattr(cmd_mgr, "active_terms", [])
    )


def _print_console_panel(
    logger,
    kwargs,
    snapshots,
    weights: dict[str, float],
    step_dt: float,
    max_episode_steps: int | None,
    cmd_low: float | None = None,
    cmd_high: float | None = None,
    heading_task: bool = True,
) -> None:
    """Compact per-iteration panel: only the values that tell you if training works.

    Everything else stays in TensorBoard.  The reward-based conversions use the ACTUAL
    weights from the live config, so ``--yaw_scale`` / ``--heading_l1`` do not distort
    the printed numbers.
    """
    weights = weights or {}

    def snap(key):
        return _snapshot_mean(snapshots, key)

    def w(name: str, fallback: float) -> float:
        value = weights.get(name)
        return abs(value) if value not in (None, 0.0) else fallback

    def l1(key: str, weight_name: str, fallback: float):
        v = snap(f"Episode_Reward/{key}")
        return -v / w(weight_name, fallback) if v is not None else None

    def l2(key: str, weight_name: str, fallback: float):
        v = snap(f"Episode_Reward/{key}")
        return math.sqrt(max(-v / w(weight_name, fallback), 0.0)) if v is not None else None

    vx_mean = l1("track_lin_vel_x_l1", "track_lin_vel_x_l1", 2.0)
    vx_rms = l2("track_lin_vel_x_l2", "track_lin_vel_x_l2", 6.0)
    wz_mean = l1("track_ang_vel_z_l1", "track_ang_vel_z_l1", 1.0)
    wz_rms = l2("track_ang_vel_z_l2", "track_ang_vel_z_l2", 2.0)
    hd_rms_rad = l2("heading_error", "heading_error", 0.6)
    hd_direct = snap("Metrics/heading_error_mean_deg")
    hd_l1_mean = l1("heading_error_l1", "heading_error_l1", 1.5)
    hd_mean_deg = (
        hd_direct
        if hd_direct is not None
        else (hd_l1_mean * 57.29577951308232 if hd_l1_mean is not None else None)
    )
    hd_rms_deg = hd_rms_rad * 57.29577951308232 if hd_rms_rad is not None else None

    vx_cmd = snap("Metrics/vx_cmd_abs_mean")
    vx_act = snap("Metrics/vx_actual_abs_mean")
    wz_cmd = snap("Metrics/wz_cmd_abs_mean")
    wz_act = snap("Metrics/wz_actual_abs_mean")
    wheel_t = snap("Metrics/wheel_target_abs_mean")
    wheel_a = snap("Metrics/wheel_actual_abs_mean")
    wheel_e = snap("Metrics/wheel_error_abs_mean")
    roll_deg = snap("Metrics/roll_abs_mean_deg")
    pitch_deg = snap("Metrics/pitch_abs_mean_deg")

    r_vx = snap("Episode_Reward/track_lin_vel_x")
    r_wz = snap("Episode_Reward/track_ang_vel_z")
    r_alive = snap("Episode_Reward/alive")
    r_fall = snap("Episode_Reward/fall_penalty")
    vx_max = w("track_lin_vel_x", 8.0)
    wz_max = w("track_ang_vel_z", 3.0)

    timeout = snap("Episode_Termination/time_out")
    fall = snap("Episode_Termination/bad_orientation")
    lenbuffer = getattr(logger, "lenbuffer", None)
    ep_len = statistics.mean(lenbuffer) if lenbuffer else None
    rewbuf = getattr(logger, "rewbuffer", None)
    mean_reward = statistics.mean(rewbuf) if rewbuf else None
    std = kwargs.get("action_std")
    std = float(std.mean()) if std is not None else None

    it = int(kwargs.get("it", 0))
    total_it = int(kwargs.get("total_it", 0))
    start_it = int(kwargs.get("start_it", 0))
    iteration_time = float(kwargs.get("collect_time") or 0.0) + float(kwargs.get("learn_time") or 0.0)
    total_time = float(getattr(logger, "tot_time", 0.0) or 0.0)
    done = max(1, it + 1 - start_it)
    eta = total_time / done * max(0, total_it - start_it - done)

    def hms(seconds: float) -> str:
        return time.strftime("%H:%M:%S", time.gmtime(max(0.0, seconds)))

    def pct(v):
        return f"{100.0 * v:5.1f}%" if v is not None else "  n/a"

    width = 78
    lines = [
        "",
        "#" * width,
        f"{'Learning iteration ' + str(it) + '/' + str(total_it):^{width}}",
        "",
    ]
    for key, value in (kwargs.get("loss_dict") or {}).items():
        lines.append(f"{('Mean ' + str(key) + ' loss:'):>34} {float(value):.4f}")
    lines.append(f"{'Mean reward:':>34} {_fmt(mean_reward, '7.2f')}")
    lines.append(f"{'Mean episode length:':>34} {_fmt(ep_len, '7.1f')}")
    lines.append(f"{'Mean action std:':>34} {_fmt(std, '7.3f')}")
    lines.append("-" * width)
    lines.append(
        f" vx        cmd {_fmt(vx_cmd)}   actual {_fmt(vx_act)} m/s   "
        f"|err| {_fmt(vx_mean)} (rms {_fmt(vx_rms)})   "
        f"exp {_fmt(r_vx, '5.2f')}/{vx_max:.2f}"
    )
    lines.append(
        f" wheel     target {_fmt(wheel_t, '7.2f')}  actual {_fmt(wheel_a, '7.2f')} rad/s   "
        f"|err| {_fmt(wheel_e, '6.2f')} rad/s"
    )
    lines.append(
        f" yaw rate  cmd {_fmt(wz_cmd)}   actual {_fmt(wz_act)} rad/s   "
        f"|err| {_fmt(wz_mean)} (rms {_fmt(wz_rms)})"
    )
    spin_cmd = snap("Metrics/wz_inplace_cmd_abs_mean")
    spin_act = snap("Metrics/wz_inplace_actual_abs_mean")
    inplace_pct = snap("Metrics/inplace_pct")
    if spin_cmd is not None or spin_act is not None:
        lines.append(
            f" spin      in-place cmd {_fmt(spin_cmd)}  actual {_fmt(spin_act)} rad/s   "
            f"share {pct(inplace_pct / 100.0) if inplace_pct is not None else '  n/a'}"
        )
    spin_pitch = snap("Metrics/spin_term_pitch_limit")
    drive_pitch = snap("Metrics/drive_term_pitch_limit")
    spin_low = snap("Metrics/spin_term_base_too_low")
    drive_low = snap("Metrics/drive_term_base_too_low")
    spin_resets = snap("Metrics/spin_reset_pct")
    if spin_pitch is not None or drive_pitch is not None:
        lines.append(
            f" term mode spin: pitch {pct(spin_pitch)}  base_low {pct(spin_low)}   "
            f"| drive: pitch {pct(drive_pitch)}  base_low {pct(drive_low)}   "
            f"spin resets {pct(spin_resets / 100.0) if spin_resets is not None else '  n/a'}"
        )
    if heading_task:
        lines.append(
            f" heading   mean {_fmt(hd_mean_deg, '5.2f')} deg   "
            f"rms {_fmt(hd_rms_deg, '5.2f')} deg"
        )
    else:
        lines.append(" heading   n/a (direct yaw-rate command)")

    # braking task metrics (logged by the command manager); absent for other tasks
    brake_speed = snap("Metrics/base_velocity/brake_speed")
    brake_distance = snap("Metrics/base_velocity/brake_distance")
    brake_time = snap("Metrics/base_velocity/time_to_stop")
    brake_stopped = snap("Metrics/base_velocity/stopped")
    brake_events = snap("Metrics/base_velocity/brake_events")
    if brake_distance is not None:
        lines.append(
            f" braking   onset {_fmt(brake_speed, '5.2f')} m/s   "
            f"stop {_fmt(brake_distance, '5.2f')} m / {_fmt(brake_time, '5.2f')} s   "
            f"stopped {pct(brake_stopped)}   events {_fmt(brake_events, '5.2f')}/env"
        )
    lines.append(
        f" attitude  roll {_fmt(roll_deg, '5.2f')} deg   pitch {_fmt(pitch_deg, '5.2f')} deg"
    )
    # fall-recovery task (absent for the driving tasks)
    success = snap("Metrics/recovery/success")
    r_upright = snap("Episode_Reward/upright")
    r_still = snap("Episode_Reward/upright_still")
    is_recovery = success is not None or r_upright is not None
    if is_recovery:
        term_roll = snap("Episode_Termination/roll_limit")
        r_tuck = snap("Episode_Reward/tuck")
        lines.append(
            f" recovery  success {pct(success)}   upright {_fmt(r_upright, '5.2f')}   "
            f"still {_fmt(r_still, '5.2f')}   tuck {_fmt(r_tuck, '4.2f')}   "
            f"roll_limit {pct(term_roll)}"
        )
    if ep_len is not None:
        total = f"{max_episode_steps}" if max_episode_steps else "max"
        if is_recovery:
            lines.append(
                f" episode   {ep_len:6.1f}/{total} steps = {ep_len * step_dt:5.2f} s   "
                f"timeout {pct(timeout)}   success {pct(success)}"
            )
        else:
            # `fall` used to count ONLY bad_orientation.  A task with a stricter pitch
            # termination (braking / teleop) can therefore lose most episodes to
            # pitch_limit while the panel reports "fall 0.0%".  Sum every non-timeout
            # termination and print the breakdown on the next line.
            term_bad = snap("Episode_Termination/bad_orientation")
            term_low = snap("Episode_Termination/base_too_low")
            term_pitch = snap("Episode_Termination/pitch_limit")
            term_roll = snap("Episode_Termination/roll_limit")
            term_parts = [v for v in (term_bad, term_low, term_pitch, term_roll) if v is not None]
            fall_all = sum(term_parts) if term_parts else fall
            lines.append(
                f" episode   {ep_len:6.1f}/{total} steps = {ep_len * step_dt:5.2f} s   "
                f"timeout {pct(timeout)}   fall {pct(fall_all)}"
            )
            lines.append(
                f" terms     bad_orient {pct(term_bad)}   base_low {pct(term_low)}   "
                f"pitch_limit {pct(term_pitch)}   roll_limit {pct(term_roll)}"
            )
    if is_recovery:
        lines.append(
            f" reward    alive {_fmt(r_alive, '4.2f')}   "
            f"closure {_fmt(snap('Episode_Reward/closure_residual'), '7.2f')}   "
            f"(upright/still above)"
        )
    else:
        lines.append(
            f" reward    track_vx {_fmt(r_vx, '5.2f')}/{vx_max:.2f}   "
            f"track_wz {_fmt(r_wz, '4.2f')}/{wz_max:.2f}   "
            f"alive {_fmt(r_alive, '4.2f')}   fall_penalty {_fmt(r_fall, '6.2f')}"
        )
    # Command-distribution histogram: proves the sampler is actually uniform over the
    # range.  The "vx cmd" above is the episode MEAN, so for [0.5, 3.0] it sits near
    # 1.5-1.6; the bin percentages are the distribution.
    pcts = [snap(f"Metrics/vx_cmd_hist{b}_pct") for b in range(5)]
    stand_pct = snap("Metrics/vx_cmd_standing_pct")
    if (
        any(p is not None for p in pcts)
        and cmd_low is not None
        and cmd_high is not None
        and cmd_high - cmd_low > 1e-6
    ):
        edges = [cmd_low + (cmd_high - cmd_low) * i / 5.0 for i in range(6)]
        bins = "  ".join(f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(5))
        vals = "  ".join(f"{p:5.1f}%" if p is not None else "   n/a" for p in pcts)
        stand_txt = f"{stand_pct:5.1f}%" if stand_pct is not None else "   n/a"
        lines.append(f" cmd dist  bins {bins}   stand")
        lines.append(f"           pct  {vals}   {stand_txt}")
    lines.append("-" * width)
    lines.append(
        f"{'Iteration time:':>34} {iteration_time:5.2f}s   "
        f"Time elapsed: {hms(total_time)}   ETA: {hms(eta)}"
    )
    print("\n".join(lines), flush=True)


def filter_iteration_console(
    runner,
    hidden: tuple[str, ...] = HIDDEN_CONSOLE_TERMS,
    step_dt: float = 0.02,
    max_episode_steps: int | None = None,
) -> None:
    """Replace rsl_rl's verbose console block with the compact tracking panel.

    TensorBoard is untouched: the original logger still receives every scalar.  Only
    stdout is suppressed while it runs, and the curated panel is printed instead.
    """
    logger = getattr(runner, "logger", None)
    original = getattr(logger, "log", None)
    if not callable(original):
        return
    weights = _reward_weights(runner)
    cmd_low, cmd_high = _command_range(runner)
    heading_task = _is_heading_task(runner)
    if not weights:
        print("[WARN] could not read the reward weights; the console panel uses the defaults")

    def filtered(**kwargs):
        snapshots = [dict(e) for e in getattr(logger, "ep_extras", [])]
        saved = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            original(**kwargs)  # TensorBoard keeps every reward/termination/metric
        finally:
            sys.stdout.close()
            sys.stdout = saved
        _print_console_panel(
            logger,
            kwargs,
            snapshots,
            weights,
            step_dt,
            max_episode_steps,
            cmd_low,
            cmd_high,
            heading_task,
        )

    logger.log = filtered
    if hidden:
        print(f"[INFO] console: compact panel (ignoring hidden={list(hidden)})")
    else:
        print(
            "[INFO] console: compact panel -- vx, wheel speed, yaw rate, heading, "
            "episode, key rewards; every per-term value stays in TensorBoard"
        )
__all__ = [
    "HIDDEN_CONSOLE_TERMS",
    "HistoryEncoderActorCfg",
    "InfantryPPORunnerCfg",
    "InfantryPPORunnerCfg_PLAY",
    "apply_cli_overrides",
    "clamp_policy_std",
    "filter_iteration_console",
    "hidden_dims_from_state_dict",
    "make_agent_cfg",
    "match_network_to_checkpoint",
    "newest_checkpoint",
    "reset_obs_normalizer_count",
    "resolve_play_checkpoint",
]
