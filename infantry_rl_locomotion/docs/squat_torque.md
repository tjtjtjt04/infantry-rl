# squat_torque: pure-torque chassis-rl squat

Task scope: squat only. The 4 hips and 2 wheels are **pure torque**; the 4 knees stay
position-controlled five-bar closed-chain followers. The robot asset is the only asset
left in the project: `chassis-rl/`.

## Action / actuator

```text
action = [tau_lf_hip, tau_lb_hip, tau_rf_hip, tau_rb_hip, tau_left_wheel, tau_right_wheel]  [Nm]
```

- hip scale: 10 Nm/action-unit; wheel scale: 1 Nm/action-unit
- PPO `clip_actions = 4.0`
- action-term hard clamp: hips ±40 Nm, wheels ±3.69 Nm
- hips/wheels: `ImplicitActuatorCfg(stiffness=0, damping=0)` + `set_joint_effort_target`
- knees: `stiffness=400, damping=10`, position targets from the loop solver every step
- `lock_hips=True` is kept as the *reference* pose (`SQUAT_HIP_OFFSETS`) for reset,
  `hip_lock` reward, and observations; it is no longer written as a position target.

## Observation (42 dims)

| slice | term | dim | scale |
|---|---|---:|---|
| 0:3 | projected_gravity | 3 | 1 |
| 3:6 | base_ang_vel | 3 | 0.25 |
| 6:9 | base_lin_vel | 3 | 1 |
| 9:11 | heading sin/cos | 2 | 1 |
| 11:13 | position error | 2 | 1 |
| 13:17 | hip_pos_rel_locked | 4 | 1 |
| 17:21 | hip_vel | 4 | 0.1 |
| 21:25 | knee_pos_rel_solved | 4 | 1 |
| 25:29 | measured_leg_state | 4 | 1 |
| 29:33 | leg_rates | 4 | 0.5 |
| 33:35 | wheel_vel | 2 | 0.1 |
| 35:41 | last_action | 6 | 1 |
| 41:42 | base_height | 1 | 1 |

## Commands

```bat
cd /d D:\IsaacLab-main\infantry_rl_locomotion
set INFANTRY_RL_SQUAT_TEACHER_CKPT=logs\rsl_rl\infantry_rl_squat\2026-11-03_23-02-16_chassis_squat_s1_cont\model_598.pt
C:/Users/32560/.conda/envs/isaaclab/python.exe train.py --task squat_torque --headless --check --num_envs 4
C:/Users/32560/.conda/envs/isaaclab/python.exe train.py --task squat_torque --headless --num_envs 1024 --iterations 600 --run_name chassis_squat_torque_s1
```

Check output should show `action_dim=6`, `base_z` around 0.173 m, closure < 5 mm,
0.5 action units -> 5 Nm hip torque, and 4.0 units saturating at 40 Nm.

### Why the teacher is needed

Pure torque starts with **zero action = zero torque**, so the hips free-fall and the
wheels free-roll; a from-scratch run has ~0.4 s episodes and PPO never sees enough
balance.  The action term therefore adds three **training-only** assists:

1. hip PD holding the reference squat pose (`kp=200, kd=5`);
2. wheel damping / pitch PD (when no teacher is loaded);
3. an optional teacher: the existing 2-action squat checkpoint is run as a wheel
   velocity policy and its target is converted with `tau = 2.0 * (v* - v)`.

The teacher is the most reliable bootstrap (100 iterations already reach ~9 s
episodes).  All assists **linearly decay to zero over 10,000 control steps**
(~417 iterations at 24 steps/iteration); the last ~180 iterations of a 600-iteration
run are pure torque.  `SquatTorqueEnvCfg_PLAY` disables every assist, so playback and
ONNX deployment are pure torque.

If you train without a teacher (empty `INFANTRY_RL_SQUAT_TEACHER_CKPT`), expect much
slower learning; raise `INFANTRY_RL_SQUAT_HIP_TORQUE_SCALE` to 15-20, raise actor
`init_std`, or set `max_hip_torque_delta` to 2-4 Nm/step.

**Decay tuning:** the assists must reach exactly zero for a pure-torque policy.  If
the policy collapses when the assist disappears, increase
`INFANTRY_RL_SQUAT_HIP_ASSIST_DECAY_STEPS` (e.g. 20,000-30,000) and train longer, or
start a second run from the last checkpoint with the assists already at zero.  Do not
deploy a checkpoint while any assist is still active.

## Playback

```bat
C:/Users/32560/.conda/envs/isaaclab/python.exe play.py --task squat_torque --headless --steps 2000 --load_run <run> --checkpoint model_<N>.pt
```

## ONNX

```bat
C:/Users/32560/.conda/envs/isaaclab/python.exe tools/export_policy_onnx.py --checkpoint <model.pt> --deploy-kind squat_torque --output exports/chassis_squat_torque.onnx
```

The sidecar JSON writes the torque post-processing:

```text
tau = clamp(clip(action, -4, 4) * [10,10,10,10,1,1],
            [-40,-40,-40,-40,-3.69,-3.69],
            [ 40, 40, 40, 40, 3.69, 3.69])
```
