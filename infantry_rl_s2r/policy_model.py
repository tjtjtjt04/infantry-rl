"""RSL-RL actor with a 5-frame history encoder (reference Wheel-Legged-Gym).

The environment emits a single flattened history observation::

    history = [o_{t-4}, o_{t-3}, o_{t-2}, o_{t-1}, o_t]        (5 * 25 = 125)

This model splits it into

* ``o_t``           -> the last 25 entries, fed directly to the actor
* ``history``       -> the full 125 entries, encoded to 3 latent values
* actor input       -> ``[o_t (25), latent (3)] = 28``

which is exactly the structure in the reference implementation / diagram.  The
critic remains a normal MLP over the same 125-dim group (it is training-only).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP


class HistoryEncoderActor(MLPModel):
    """MLP actor whose input is ``current_obs + encoder(history)``."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        activation: str,
        obs_normalization: bool,
        distribution_cfg: dict | None,
        encoder_hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        latent_dim: int = 3,
        current_dim: int = 25,
        history_length: int = 5,
        frame_dim: int = 25,
        **kwargs,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        # ``super()`` built an MLP with input == the full flattened history.  Replace
        # it with the two-stage encoder + actor.
        self.current_dim = int(current_dim)
        self.latent_dim = int(latent_dim)
        self.history_length = int(history_length)
        self.frame_dim = int(frame_dim)
        expected = self.history_length * self.frame_dim
        if self.obs_dim != expected:
            raise ValueError(
                f"observation group has {self.obs_dim} dims but history_length * frame_dim = {expected}"
            )

        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.encoder = MLP(
            self.obs_dim, self.latent_dim, list(encoder_hidden_dims), activation
        )
        self.mlp = MLP(
            self.current_dim + self.latent_dim,
            mlp_output_dim,
            list(hidden_dims),
            activation,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

        print(
            f"[model] history encoder   : {self.obs_dim} -> {list(encoder_hidden_dims)} "
            f"-> {self.latent_dim}"
        )
        print(
            f"[model] actor             : {self.current_dim} current + {self.latent_dim} latent "
            f"= {self.current_dim + self.latent_dim} -> {list(hidden_dims)} -> {output_dim}"
        )

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
    ) -> torch.Tensor:
        """Normalize the 125-dim history, split current obs, encode the history."""
        x = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        x = self.obs_normalizer(x)
        current = x[..., -self.current_dim :]
        latent = self.encoder(x)
        return torch.cat((current, latent), dim=-1)

    def as_onnx(self, verbose: bool = False):
        """Not used: ONNX export is handled by ``tools/export_policy_onnx.py``."""
        raise NotImplementedError("use tools/export_policy_onnx.py for the history-encoder policy")


__all__ = ["HistoryEncoderActor"]
