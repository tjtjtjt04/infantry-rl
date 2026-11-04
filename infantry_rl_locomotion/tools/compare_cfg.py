"""Deep-compare the original and refactored environment / PPO configs.

This is the core evidence that the refactor is behaviour-preserving.  It boots Kit
(imports of ``isaaclab`` need it), imports both packages side by side -- the old
``isaac_ws_infantry`` from the untouched source workspace and the new flat modules
-- and recursively compares their ``to_dict()`` output.

Ignored fields (runtime, not behaviour): ``seed``, ``device``, ``log_dir``,
``env_name``.  Absolute paths are compared by file name only, because the new
project stores its assets locally.

Run from this folder:
    python tools/compare_cfg.py
    python tools/compare_cfg.py --dump-json compare_cfg_actual.json

Environment-variable variants are compared by running this script again with the
variables set, e.g.::

    set INFANTRY_RL_VX_MIN=0.3 && set INFANTRY_RL_VX_MAX=1.0
    set INFANTRY_RL_ENABLE_TURNING=1
    set INFANTRY_RL_HEADING_CONTROL=1 && set INFANTRY_RL_POSITION_CONTROL=1
    set INFANTRY_RL_NO_HEADING_OBS=1 && set INFANTRY_RL_NO_POSITION_OBS=1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
OLD_WS = Path(r"D:\IsaacLab-main\infantry-rl-main\infantry-rl-main\isaac_ws")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare the old and new infantry-rl configs.")
parser.add_argument("--dump-json", type=str, default=None, help="write the normalized comparison trees here")
parser.add_argument("--max-diffs", type=int, default=40, help="stop printing after this many differences")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows (imports require Kit to be running)."""

# --- new flat layout (imported first: it owns the top-level module names) -----
from agent import (  # noqa: E402
    InfantryBalancePPORunnerCfg as NewBalPPO,
    InfantryBalancePPORunnerCfg_PLAY as NewBalPPOPlay,
    InfantryLocomotionPPORunnerCfg as NewLocoPPO,
    InfantryLocomotionPPORunnerCfg_PLAY as NewLocoPPOPlay,
)
from env_balance import BalanceEnvCfg as NewBalCfg, BalanceEnvCfg_PLAY as NewBalPlayCfg  # noqa: E402
from env_locomotion import (  # noqa: E402
    LocomotionEnvCfg as NewLocoCfg,
    LocomotionEnvCfg_PLAY as NewLocoPlayCfg,
)

# --- old package from the untouched source workspace --------------------------
if str(OLD_WS) not in sys.path:
    sys.path.append(str(OLD_WS))
import isaac_ws_infantry  # noqa: E402,F401  (registers the gym tasks)
from isaac_ws_infantry.tasks.balance.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    InfantryBalancePPORunnerCfg as OldBalPPO,
    InfantryBalancePPORunnerCfg_PLAY as OldBalPPOPlay,
)
from isaac_ws_infantry.tasks.balance.balance_env_cfg import (  # noqa: E402
    BalanceEnvCfg as OldBalCfg,
    BalanceEnvCfg_PLAY as OldBalPlayCfg,
)
from isaac_ws_infantry.tasks.locomotion.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    InfantryLocomotionPPORunnerCfg as OldLocoPPO,
    InfantryLocomotionPPORunnerCfg_PLAY as OldLocoPPOPlay,
)
from isaac_ws_infantry.tasks.locomotion.locomotion_env_cfg import (  # noqa: E402
    LocomotionEnvCfg as OldLocoCfg,
    LocomotionEnvCfg_PLAY as OldLocoPlayCfg,
)

#: fields that are runtime inputs, not task behaviour
SKIP_KEYS = {"seed", "device", "log_dir", "env_name"}
#: dict paths whose KEY ORDER is part of the behaviour (reward/observation order)
ORDERED_PATHS = (
    "observations.policy",
    "actions",
    "rewards",
    "terminations",
    "commands",
    "events",
)


def normalize(value):
    """Make a cfg tree comparable across the two package layouts."""
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if isinstance(value, type):
        return f"<class:{value.__name__}>"
    if callable(value):
        return f"<callable:{getattr(value, '__name__', type(value).__name__)}>"
    if isinstance(value, Path):
        return f"<path:{value.name}>"
    if isinstance(value, str):
        # class_to_dict renders callables/classes as "module:name"; the package
        # prefix differs between the two layouts, the symbol does not
        head, sep, tail = value.partition(":")
        if sep and re.fullmatch(r"[\w.]+", head) and re.fullmatch(r"[\w.]+", tail):
            return f"{'.'.join(head.split('.')[-2:])}:{tail}"
        # absolute asset paths (old D:\Robomaster\..., new project-local) -> file name
        if (os.sep in value or "/" in value) and value.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]:
            return f"<path:{Path(value).name}>"
        return value
    return value


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if a == 0.0 and b == 0.0:  # signed zero is a documented behaviour here
            return math.copysign(1.0, a) == math.copysign(1.0, b)
        return a == b
    return a == b


def diff_tree(old, new, path: str, diffs: list[tuple[str, object, object]]) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key in SKIP_KEYS:
                continue
            child = f"{path}.{key}" if path else key
            if key not in old:
                diffs.append((child, "<missing>", new[key]))
            elif key not in new:
                diffs.append((child, old[key], "<missing>"))
            else:
                diff_tree(old[key], new[key], child, diffs)
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            diffs.append((f"{path}[len]", len(old), len(new)))
        for index, (a, b) in enumerate(zip(old, new)):
            diff_tree(a, b, f"{path}[{index}]", diffs)
        return
    if not same(old, new):
        diffs.append((path, old, new))


def key_order_mismatches(old, new, path: str) -> list[str]:
    """Check the order of term dicts at the known ordered paths."""
    problems = []
    for wanted in ORDERED_PATHS:
        old_node, new_node = old, new
        for part in wanted.split("."):
            old_node = old_node.get(part) if isinstance(old_node, dict) else None
            new_node = new_node.get(part) if isinstance(new_node, dict) else None
        if isinstance(old_node, dict) and isinstance(new_node, dict):
            if list(old_node) != list(new_node):
                problems.append(
                    f"{path}.{wanted}: key order differs\n  old={list(old_node)}\n  new={list(new_node)}"
                )
    return problems


def compare(name: str, old_cfg, new_cfg) -> tuple[list, list[str], dict, dict]:
    old_tree = normalize(old_cfg.to_dict())
    new_tree = normalize(new_cfg.to_dict())
    diffs: list[tuple[str, object, object]] = []
    diff_tree(old_tree, new_tree, "", diffs)
    order_problems = key_order_mismatches(old_tree, new_tree, name)
    return diffs, order_problems, old_tree, new_tree


def main() -> int:
    pairs = [
        ("locomotion", OldLocoCfg, NewLocoCfg),
        ("locomotion_play", OldLocoPlayCfg, NewLocoPlayCfg),
        ("balance", OldBalCfg, NewBalCfg),
        ("balance_play", OldBalPlayCfg, NewBalPlayCfg),
        ("agent_locomotion", OldLocoPPO, NewLocoPPO),
        ("agent_locomotion_play", OldLocoPPOPlay, NewLocoPPOPlay),
        ("agent_balance", OldBalPPO, NewBalPPO),
        ("agent_balance_play", OldBalPPOPlay, NewBalPPOPlay),
    ]

    all_ok = True
    dumped = {}
    for name, old_cls, new_cls in pairs:
        diffs, order_problems, old_tree, new_tree = compare(name, old_cls(), new_cls())
        dumped[name] = {"old": old_tree, "new": new_tree}
        status = "PASS" if not diffs and not order_problems else "FAIL"
        all_ok &= status == "PASS"
        print(f"[compare] {name:26s} {status}  ({len(diffs)} value diffs, {len(order_problems)} order diffs)")
        for path, old_value, new_value in diffs[: args_cli.max_diffs]:
            print(f"    - {path or '<root>'}: old={old_value!r}  new={new_value!r}")
        if len(diffs) > args_cli.max_diffs:
            print(f"    ... {len(diffs) - args_cli.max_diffs} more")
        for problem in order_problems:
            print(f"    - {problem}")

    # explicit documented checks (independent of the generic tree walk)
    loco = NewLocoCfg()
    rewards = loco.to_dict()["rewards"]
    checks = {
        "locomotion reward count == 24": (len(rewards), 24),
        "locomotion pitch weight is -0.0": (math.copysign(1.0, rewards["pitch"]["weight"]), -1.0),
        "locomotion action_jerk weight == -0.3": (rewards["action_jerk"]["weight"], -0.3),
    }
    print("[compare] explicit checks:")
    for label, (actual, expected) in checks.items():
        ok = actual == expected
        all_ok &= ok
        print(f"    - {label}: {actual!r} (expected {expected!r}) {'OK' if ok else 'MISMATCH'}")

    if args_cli.dump_json:
        Path(args_cli.dump_json).write_text(json.dumps(dumped, indent=2, default=str), encoding="utf-8")
        print(f"[compare] trees written to {args_cli.dump_json}")

    print(f"[compare] RESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
