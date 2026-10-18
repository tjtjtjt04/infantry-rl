"""infantry-rl: Isaac Lab tasks for the closed-chain five-bar wheel-leg robot.

Importing this package registers the gym environments, e.g.::

    import isaac_ws_infantry  # noqa: F401
    env = gym.make("Infantry-RL-Balance-v0", cfg=parse_env_cfg("Infantry-RL-Balance-v0"))

See ``isaac_ws/README`` / the task brief for how to run the scripts.
"""

__version__ = "0.1.0"

from . import tasks  # noqa: E402,F401  (registers the gym environments)

__all__ = ["tasks"]
