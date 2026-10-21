"""RSL-RL agent configuration for the infantry-rl locomotion task.

Same network shape as the balance task, but a much longer schedule: velocity-command
tracking is a bigger skill than standing still, and it only pays off later in training.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class InfantryLocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    # 48 steps x 1024 envs = 49k samples per iteration (was 24 -> 24.5k).  A longer
    # rollout gives much lower-variance advantage estimates, and the extra samples cost
    # no additional CPU-side work per environment step (that per-step overhead is what
    # dominates this machine), so every gradient step becomes better informed.
    num_steps_per_env = 48
    max_iterations = 8000
    # 50 (not 100): the PhysX-GPU crashes land every few minutes at this speed range,
    # so the checkpoint interval is what sets how much work a crash can throw away.
    # 50 iterations x ~2.4 s = about 2 minutes at risk instead of 4.
    save_interval = 50
    experiment_name = "infantry_rl_locomotion"
    run_name = ""
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    # the raw Gaussian policy is unbounded; clip so a wild early policy cannot command
    # absurd wheel speeds (the action term multiplies by wheel_scale = 5 rad/s)
    clip_actions = 4.0
    actor = RslRlMLPModelCfg(
        # Back to the original 128-128-64.  It was widened to 256-256-128 for extra
        # capacity, but on this 8 GB laptop that also raised the per-iteration GPU work
        # (2.4 s/iteration instead of ~1.9 s) and the GPU memory pressure, at a time when
        # the runs were already dying every few minutes inside the PhysX GPU allocator.
        # Stability first: 128-128-64 is enough for a 35-dim observation / 6-dim action.
        hidden_dims=[128, 128, 64],
        activation="elu",
        # ENABLED for the from-scratch run.  The observation dimensions span roughly
        # 0.1 .. 4 (projected gravity ~1, commands 0.3..2.5, last_action up to the
        # clip_actions bound of 4), a 40x spread that makes the raw MLP badly
        # conditioned.  rsl_rl then inserts an ``EmpiricalNormalization`` module, and
        # its running mean/std become part of the state dict -- so a checkpoint trained
        # with this False cannot be resumed strictly.  That is fine here because this
        # run starts from scratch.
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
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
        # 0.001 (v1.0.5).  History of this number, all measured:
        #   0.015 -> std runaway to 168 (pure noise policy)
        #   0.005 -> std climbed to the clamp ceiling 1.0, i.e. the equilibrium sat ABOVE
        #            the allowed window; trained under +/-14 deg of hip noise the learned
        #            mean comes out jittery and the value loss stays high (~1.07)
        #   0.001 -> the entropy push is 5x weaker, so sigma settles around 0.2, which is
        #            the useful exploration band for this task
        # The hard clamp in train.py::_clamp_policy_std ([0.03, 0.4] since v1.0.5) is the
        # safety net; the coefficient is chosen so the equilibrium sits strictly inside it.
        entropy_coef=0.003,
        # 8 epochs instead of 5: with the adaptive KL schedule the extra passes are
        # guarded (if an update moves the policy too far the learning rate is reduced
        # automatically), and it squeezes more learning out of the same 49k-sample
        # rollout -- which is the expensive part on this machine.
        num_learning_epochs=8,
        # 4 minibatches x 5 epochs = 20 gradient steps per rollout.
        #
        # Measured on this machine (1024 envs, 48-step rollouts):
        #   num_mini_batches=8 -> 40 gradient steps -> 2.14 s/iteration -> 22,993 samples/s
        #   num_mini_batches=4 -> 20 gradient steps -> ~1.83 s/iteration -> ~26,800 samples/s
        # Doubling the number of gradient steps cost ~10% throughput because the rollout
        # (the environment stepping, which dominates) was unchanged while the update
        # phase doubled.  With 4 minibatches each batch is 12,288 samples instead of
        # 6,144, which is also more efficient per sample on a small GPU.
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        # 0.99 at 50 Hz gives a 100-step (2 s) effective horizon, but an episode is 20 s
        # and the failures we care about (a slow sideways lean building up, a leg angle
        # drifting) develop over several seconds.  0.995 doubles the horizon to ~4 s.
        # Raising gamma instead of lengthening the rollout is the cheap way to get
        # longer-horizon credit assignment: it adds ZERO wall-clock cost.
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class InfantryLocomotionPPORunnerCfg_PLAY(InfantryLocomotionPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 0
        self.num_steps_per_env = 24


__all__ = ["InfantryLocomotionPPORunnerCfg", "InfantryLocomotionPPORunnerCfg_PLAY"]
