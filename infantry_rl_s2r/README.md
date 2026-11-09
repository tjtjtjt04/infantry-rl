# infantry_rl_s2r — reference-style wheel_legged squat + fall recovery

Two tasks share one 25-dim observation / 6-dim action interface:

| task | start | goal |
|---|---|---|
| `squat_torque` | upright, +-1.5 deg pitch | hold the locked squat stance |
| `getup_torque` | **on the ground**, 30-40 deg pitch | drive back into the balance envelope, then hand off |

Because the interface is identical, a `squat_torque` checkpoint loads into
`getup_torque` with `--resume` and the ONNX export is the same for both.

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

REM fall recovery (same observation / action; starts on the ground)
C:/Users/32560/.conda/envs/isaaclab/python.exe train.py --task getup_torque --headless --check --num_envs 8
C:/Users/32560/.conda/envs/isaaclab/python.exe train.py --task getup_torque --resume --resume_latest --num_envs 1024 --iterations 400 --reset_obs_norm_count 5000000 --run_name getup_from_balance
C:/Users/32560/.conda/envs/isaaclab/python.exe play.py --task getup_torque --headless --steps 2000 --load_run <run> --checkpoint model_<N>.pt
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
the history encoder has 3 latent dims. See `docs/squat_torque.md`,
`docs/squat_torque_onnx_usage.md` and `docs/getup_torque.md`.

## Fall recovery: why the reset looks the way it does

Measured with `tools/probe_fallen_geometry.py` (2 mm lowering test + contact sensor
on every link) and `tools/probe_recovery.py` (closed-loop):

* 0-17 deg pitch: the **wheels** are the contact;
* 17-20 deg: wheels + one chassis corner = a stable **tripod**;
* >= 20 deg: only the chassis corner touches, the wheels are in the air (4.8 cm at
  35 deg), and the CoM is behind the corner, so gravity rotates the robot back down.

Every start between 20 and 60 deg settles at |pitch| = 17 deg (the tripod) within
~1.2 s of zero action: **30-40 deg is a transient, not a resting state**, and tipping
back from it makes the robot bounce.  The task therefore starts from the tripod only
(`INFANTRY_RL_GETUP_REST_PROB = 1.0`).  The balance checkpoint already recovers from
30-40 deg when the task terminations are disabled, but it cannot leave the 17 deg
tripod; open-loop wheel thrust and rocking cannot either.  A `getup_torque` fine-tune
can, and the domain randomization matters a lot: with DR **off** 200 iterations reach
100 % in the DR-off env but only 59 % once DR is switched on, while a DR **on** run of
300 iterations from the tripod reaches 100 % at every start from 10 to 40 deg.
`docs/getup_torque.md` has the tables.
