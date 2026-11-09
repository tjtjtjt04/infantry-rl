# squat_torque: reference wheel_legged joint-space squat

This mirrors the 25-dim Wheel-Legged-Gym ``wheel_legged`` task: the policy outputs
**hip position targets + wheel velocity targets**, and an explicit low-level PD
sends torques to the motors.  The 4 knees stay position-controlled five-bar
closed-chain followers.

## Action / actuator

```text
action = [hip_offset_lf, hip_offset_lb, hip_offset_rf, hip_offset_rb,
          wheel_vel_l, wheel_vel_r]
```

Low-level PD (reference gains; env-overridable):

```text
q_ref = squat_hips + clip(action[:4], ±4) * 0.5       [rad]
v_ref = clip(action[4:6], ±4) * 10.0                  [rad/s]

tau_hip   = 40.0*(q_ref - q) - 1.0*qdot               [Nm], clamp ±40
tau_wheel = 0.5*(v_ref - v)                           [Nm], clamp ±3.69
```

- the motors are torque driven; the PD is the low-level controller (in firmware
  on the real robot)
- PPO `clip_actions = 4.0`
- knees: `stiffness=400, damping=10`, position targets from the loop solver every step
- `hip_bias = SQUAT_HIP_OFFSETS` is the reference squat pose at zero action

## Observation: 25 per frame, 5-frame history

Per frame:

| slice | term | dim | scale | hardware |
|---|---|---:|---:|---|
| 0:3 | base_ang_vel | 3 | 0.25 | gyro × 0.25 |
| 3:6 | projected_gravity | 3 | 1 | IMU gravity projection |
| 6:9 | command | 3 | 1 | `[vx, vy, wz]`, zero for the squat task |
| 9:13 | hip_pos_rel_locked | 4 | 1 | `hip_meas - LOCKED_HIPS` |
| 13:19 | joint_vel | 6 | 0.05 | 4 hip + 2 wheel velocities × 0.05 |
| 19:25 | last_action | 6 | 1 | previous clip(±4) raw action |

Exactly `3+3+3+4+6+6 = 25`. The knees are not observed.

The environment exposes the flattened 5-frame history (oldest → newest):

```text
history = [o_{t-4}, o_{t-3}, o_{t-2}, o_{t-1}, o_t]       # 125 dims
```

and the custom actor (`policy_model.py`) computes:

```text
encoder : 125 -> 128 -> 64 -> 3          # latent (estimates the unobserved state)
actor   : [o_t(25), latent(3)] = 28 -> 128 -> 128 -> 64 -> action(6)
```

The critic is a normal MLP over the same 125-dim history.

### Sensor noise (training only)

Training applies uniform noise, with amplitudes copied from the reference
Wheel-Legged-Gym `noise_scales` (applied before the obs scale):

| term | amplitude (physical) | in obs |
|---|---:|---:|
| base_ang_vel | ±0.2 rad/s | ±0.05 |
| projected_gravity | ±0.05 | ±0.05 |
| command | 0 | 0 |
| hip_pos | ±0.01 rad | ±0.01 |
| joint_vel | ±1.5 rad/s | ±0.075 |
| last_action | 0 | 0 |

Scale/toggle with env vars: `INFANTRY_RL_OBS_NOISE_LEVEL` (default 1.0),
`INFANTRY_RL_OBS_NOISE_ANG_VEL`, `..._GRAVITY`, `..._DOF_POS`, `..._DOF_VEL`.
`SquatEnvCfg_PLAY` disables the corruption.

### Joint velocity

By default `joint_vel` is an encoder-style finite difference of the measured hip
and wheel positions over one control step (`vel_source=position_diff`), which is
what the reference repository and the firmware do.  An optional first-order
low-pass `y = alpha*y_prev + (1-alpha)*x` can be enabled for noisy encoders:

```bat
set INFANTRY_RL_VEL_SOURCE=position_diff
set INFANTRY_RL_VEL_FILTER_ALPHA=0.5
```

`alpha=0` (default) is the unfiltered reference behaviour.  Use
`INFANTRY_RL_VEL_SOURCE=sim` for PhysX's exact joint velocity.

## Commands

```bat
C:/Users/32560/.conda/envs/isaaclab/python.exe train.py --headless --check --num_envs 4
C:/Users/32560/.conda/envs/isaaclab/python.exe train.py --headless --num_envs 1024 --iterations 600 --run_name chassis_squat_torque_s1
```

Check output should show `obs shape=(4, 25)`, `action_dim=6`, `base_z` around
0.173 m, closure < 5 mm, 0.5 action units -> 5 Nm hip torque, and 4.0 units
saturating at 40 Nm.

## Domain randomization (reference Wheel-Legged-Gym)

Sampled per environment at startup / by interval; all of it is training-only and
does not change the 25-D observation:

| item | default | env var |
|---|---|---|
| robot shape friction | 0.4–1.2 | `INFANTRY_RL_DR_FRICTION_MIN/MAX` |
| restitution | 0.0–0.2 | `INFANTRY_RL_DR_RESTITUTION_MIN/MAX` |
| added base mass | −2…+3 kg | `INFANTRY_RL_DR_MASS_MIN/MAX` |
| base CoM offset | ±5 cm (x,y), ±3 cm (z) | `INFANTRY_RL_DR_COM_XY`, `..._COM_Z` |
| knee servo gains | ×0.9–1.1 | `INFANTRY_RL_DR_GAIN_MIN/MAX` |
| per-motor torque | ×0.9–1.1 | `INFANTRY_RL_DR_MOTOR_MIN/MAX` |
| random push | 0.3 m/s every 5–8 s | `INFANTRY_RL_DR_PUSH_VEL`, `..._PUSH_INTERVAL_MIN/MAX` |
| action delay | 0–1 control step (100 Hz) | `INFANTRY_RL_ACTION_DELAY_MAX` |
| reset hip jitter | ±0.01 rad per hip | `INFANTRY_RL_RESET_HIP_JITTER` |

Disable everything with `set INFANTRY_RL_DR=0` (also sets action delay and reset
jitter to zero).  The reference uses a 2.0 m/s push for locomotion; a squat is
much less stable, so the default here is 0.3 m/s.  Raise it to 0.5–1.0 once the
policy reliably holds the pose; lower it if early episodes collapse.

The random push is a root-velocity impulse (`push_by_setting_velocity`), the same
effect as the reference's `_push_robots`.  The torque probe in `--check` forces
delay 0 and motor scale 1 so its assertions stay exact.

## Training (from scratch, reference joint-space, 100 Hz)

Policy update: 100 Hz (sim 200 Hz / decimation 2); 5-frame history = 50 ms.

The low-level PD is always on, so there is **no teacher and no decay curriculum**.
The policy learns hip position targets and wheel velocity targets directly.

```bat
set INFANTRY_RL_DR=0
set INFANTRY_RL_OBS_NOISE_LEVEL=0.5
C:/Users/32560/.conda/envs/isaaclab/python.exe train.py --task squat_torque --headless --num_envs 1024 --iterations 1000 --run_name chassis_squat_joint_s1
```

Start with DR off to confirm the action space learns; once stable, enable the
domain randomization gradually.  The reference training uses the same 5-frame
history encoder; our critic currently sees the same 125-dim history (add
privileged observations later if needed).

**Measured result** (64 envs, 250 iterations, DR off, noise 0.5):

```text
episode length : 490.3 / 500
time_out       : 98.4%
pitch fall     : 1.6%
pitch mean     : 0.46 deg
mean action std: 0.35
```

Key point: the reference PPO exploration std matters.  With `init_std=0.1`
(the old torque setting) the same action space stayed at ~31/500; with
`init_std=0.5` it learns.  The reference uses `init_noise_std=0.5`, entropy
0.01, gamma 0.99, desired_kl 0.005; our `agent.py` now matches those.

A working seed checkpoint is kept at
`logs/rsl_rl/infantry_rl_squat_torque/2026-11-06_joint250b_seed/`.

## Playback

```bat
C:/Users/32560/.conda/envs/isaaclab/python.exe play.py --headless --steps 2000 --load_run <run> --checkpoint model_<N>.pt
```

## ONNX

```bat
C:/Users/32560/.conda/envs/isaaclab/python.exe tools/export_policy_onnx.py --checkpoint <model.pt> --output exports/chassis_squat_torque.onnx
```

The sidecar JSON writes the observation layout, the hip/wheel targets and the
low-level PD that converts them to torques:

```text
tau = clamp(clip(action, -4, 4) * [10,10,10,10,1,1],
            [-40,-40,-40,-40,-3.69,-3.69],
            [ 40, 40, 40, 40, 3.69, 3.69])
```
