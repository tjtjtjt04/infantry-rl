# infantry_rl_s2r — reference-style wheel_legged squat

This mirrors the 25-dim Wheel-Legged-Gym `wheel_legged` task: the policy outputs
**4 hip position targets + 2 wheel velocity targets**, and an explicit low-level
PD sends torques to the motors. The four knees are passive closed-chain followers
and are position-controlled in simulation only.

```bat
cd /d D:\IsaacLab-main\infantry_rl_s2r
C:/Users/32560/.conda/envs/isaaclab/python.exe train.py --headless --check --num_envs 4
C:/Users/32560/.conda/envs/isaaclab/python.exe train.py --headless --num_envs 1024 --iterations 1000 --run_name chassis_squat_joint_s1
C:/Users/32560/.conda/envs/isaaclab/python.exe play.py --headless --steps 2000 --load_run <run> --checkpoint model_<N>.pt
C:/Users/32560/.conda/envs/isaaclab/python.exe tools/export_policy_onnx.py --checkpoint <model.pt> --output exports/chassis_squat_torque.onnx
```

Policy I/O:

```text
frame (25) = base_ang_vel(3)*0.25
           + projected_gravity(3)
           + command(3)
           + hip_pos_rel_locked(4)
           + joint_vel(4 hips + 2 wheels)(6)*0.05
           + last_action(6)

history (125) = [o_{t-4}, ..., o_t]         (5 frames, oldest -> newest)
encoder       = 125 -> 128 -> 64 -> 3
actor         = [o_t(25), latent(3)] = 28 -> 128 -> 128 -> 64 -> action(6)

action (6) = [hip_offset_lf, hip_offset_lb, hip_offset_rf, hip_offset_rb,
              wheel_vel_l, wheel_vel_r]

q_ref = squat_hips + clip(a[:4], ±4) * 0.5      [rad]
v_ref = clip(a[4:6], ±4) * 10.0                 [rad/s]
tau_hip   = clamp(40*(q_ref - q) - 1*qdot, ±40) [Nm]
tau_wheel = clamp(0.5*(v_ref - v), ±3.69)       [Nm]
```

Training adds observation noise, encoder-style joint-velocity estimation,
domain randomization (friction, base mass/CoM, motor strength, knee gains,
random pushes) and per-environment action delay. The actor stays at 25 dims;
the history encoder has 3 latent dims. See `docs/squat_torque.md` and
`docs/squat_torque_onnx_usage.md`.
