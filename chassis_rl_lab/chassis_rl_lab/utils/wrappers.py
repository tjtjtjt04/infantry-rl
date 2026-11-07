"""训练/回放用的 NaN 防护 wrapper：把观测与奖励里的 NaN/Inf 过滤掉。

背景：Isaac Sim/PhysX 在极端状态下（接触爆炸、数值发散）可能产生 NaN/Inf。一旦 NaN
进入 PPO 的 rollout，loss/梯度会变成 NaN，整个训练报废。环境层已有 ``state_invalid``
终止项把"坏掉"的 env 立即重置；本 wrapper 作为最后一道保险，对每步的
``obs``（TensorDict/dict/tensor）与 ``reward`` 做 ``nan_to_num`` + 限幅。

用法（见 ``scripts/train.py`` / ``scripts/play.py``）::

    env = gym.make(task, cfg=env_cfg)
    env = SanitizeWrapper(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
"""

from __future__ import annotations

import gymnasium as gym
import torch

__all__ = ["SanitizeWrapper"]


class SanitizeWrapper(gym.Wrapper):
    """过滤 obs/reward 中的 NaN/Inf，并把数值夹到安全范围。"""

    def __init__(self, env: gym.Env, obs_clip: float = 100.0, reward_clip: float = 1.0e4):
        super().__init__(env)
        self._obs_clip = float(obs_clip)
        self._reward_clip = float(reward_clip)
        #: 累计过滤次数（调试用；训练结束可打印）
        self.num_obs_sanitized = 0
        self.num_reward_sanitized = 0

    def _sanitize_tensor(self, value: torch.Tensor, clip: float) -> torch.Tensor:
        if not torch.is_tensor(value):
            return value
        fixed = torch.nan_to_num(value, nan=0.0, posinf=clip, neginf=-clip)
        return fixed.clamp(-clip, clip)

    def _sanitize_obs(self, obs):
        def _fix(tensor: torch.Tensor) -> torch.Tensor:
            if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
                self.num_obs_sanitized += 1
            return self._sanitize_tensor(tensor, self._obs_clip)

        # Isaac Lab 的 obs 可能是 TensorDict / dict / 纯 tensor
        if hasattr(obs, "items"):
            for key, value in list(obs.items()):
                obs[key] = _fix(value)
            return obs
        return _fix(obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._sanitize_obs(obs), info

    def step(self, action):
        if torch.is_tensor(action):
            action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._sanitize_obs(obs)
        if torch.is_tensor(reward) and not torch.isfinite(reward).all():
            self.num_reward_sanitized += 1
        reward = self._sanitize_tensor(reward, self._reward_clip)
        return obs, reward, terminated, truncated, info
