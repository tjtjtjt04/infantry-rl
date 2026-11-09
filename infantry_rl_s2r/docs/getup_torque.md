python tools/export_policy_onnx.py ^
  --checkpoint checkpoints\getup_tripod_model_2895.pt ^
  --output exports\chassis_getup_tripod.onnx# getup_torque: self-righting from the ground

`getup_torque` is the fall-recovery counterpart of `squat_torque`.  The robot is put
**on the ground** at a fore/aft pitch and has to drive itself back into the balance
envelope; from there the balance policy takes over.

The observation and the action are **identical to `squat_torque`** -- 25 dims per
frame, 5-frame history (125), `[4 hip position offsets, 2 wheel velocity targets]`.
That is deliberate and non-negotiable:

* the ONNX export / firmware interface stays exactly the same;
* a `squat_torque` checkpoint loads into this task unchanged (`--resume`), so the
  recovery is a fine-tune of the proven balance policy instead of a new skill from
  zero.

## What the real "fallen" state is (measured, not assumed)

`tools/probe_fallen_geometry.py` lowers the robot 2 mm at a time with zero velocity,
using a contact sensor on **every** link, and records the height and the body that
touches first.  `robot_model.fallen_geometry()` ships that table.

| root pitch | contact height | first contact |
|---:|---:|---|
| 0 deg | 0.1720 m | both wheels |
| 10 deg | 0.1700 m | wheels |
| 15 deg | 0.1680 m | wheels |
| 17 deg | 0.1660 m | wheels (+ chassis corner joins at ~0.170) |
| 20 deg | 0.1720 m | **chassis corner** |
| 25 deg | 0.1900 m | chassis corner |
| 30 deg | 0.2060 m | chassis corner |
| 35 deg | 0.2200 m | chassis corner (wheels are 4.8 cm in the air) |
| 40 deg | 0.2360 m | chassis corner |
| 45 deg | 0.2480 m | chassis corner |
| 60 deg | 0.2700 m | chassis corner |

Three regimes, and they decide the whole task design:

1. **0-17 deg**: the wheels carry the robot.  The chassis bottom is 7.4 cm above the
   ground when level, so a 17 deg lean is where the corner just reaches it.
2. **17-20 deg**: wheels + one chassis corner = the **tripod**.  This is a stable
   local minimum.
3. **>= 20 deg**: only the corner touches.  The CoM projects *behind* the corner, so
   this pose is **unstable**: gravity rotates the robot back down.

Settling the robot at every pitch from 0 to 60 deg with zero action (1.2 s each)
gives the same answer every time: **it comes to rest at |pitch| = 17 deg**, on the
tripod, within ~0.5 s.  So

> 30-40 deg is not a resting state -- it is the transient on the way back down.
> The stable "fallen" pose of this robot, with the legs retracted, is the 17 deg tripod.

## What a policy can and cannot do from there

`tools/probe_recovery.py` places the robot at a chosen pitch on the measured contact
height and runs a *closed-loop* controller with the task terminations disabled, so it
measures the controller rather than the episode limits.

Balance checkpoint (`model_2596.pt`), 3 s per start:

| start pitch | end pitch | result |
|---:|---:|---|
| -40 ... +10 deg | ~0 deg | **upright** |
| +15, +17, +20, +25 deg | +17.1 deg | **stuck in the tripod** |
| +30, +35, +40 deg | ~0 deg | **upright** |

Two conclusions:

* the balance policy already *has* the catch-the-fall skill from 30-40 deg -- in
  `squat_torque` the episode simply ends at 0.24 rad before it can show it;
* the **positive** tripod is a wheel-only dead end *for open-loop control*.
  Constant wheel thrust (+-2 raw action = +-20 rad/s) and 0.5-5 Hz rocking (+-3.5)
  all leave the pitch at 17.2 deg: the velocity loop rattles against the corner
  contact instead of delivering thrust.  It is **not** a dead end for a learned
  policy -- see the measured result below.

### Why the task starts at 17 deg and not at 30-40 deg

A 30-40 deg start is an unstable pose: only the chassis corner touches, the wheels
are 2-5 cm in the air, and the CoM is behind the corner, so gravity tips the robot
back down.  When the wheels come back down they **bounce** -- visible in the
simulation and not a state the hardware ever rests in.  Training on that transient
mostly teaches the policy to survive a bounce, while the state it actually has to
leave is the 17 deg tripod.  The reset therefore starts at the tripod only
(`INFANTRY_RL_GETUP_REST_PROB = 1.0`); the 30-40 deg population is still available
via `INFANTRY_RL_GETUP_REST_PROB < 1` plus `INFANTRY_RL_GETUP_FALL_DEG_*` for a
broader fine-tune.

## Measured result: 200 iterations of fine-tuning

`train.py --task getup_torque --resume ... --num_envs 256 --iterations 200` from
`model_2596.pt` (domain randomization off for this verification run, see below):

| iteration | `Metrics/recovery/success` | mean pitch | mean episode |
|---:|---:|---:|---:|
| 2596 | 0 % | 28.5 deg | 13.7 steps |
| 2598 | 84 % | 14.1 deg | 66 steps |
| 2605 | 96 % | 14.4 deg | 72 steps |
| 2777-2795 | **100 %** | ~12 deg | ~45 steps (0.45 s) |

Closed-loop probe of the fine-tuned checkpoint (`tools/probe_recovery.py`,
terminations disabled, 2.5 s per start, 17 starts from -40 to +40 deg): **100 %
upright from every pitch**, including the +15/+17/+20/+25 deg tripod that the
balance policy could not leave, with all of them inside +-2.6 deg after 0.4 s and
the legs back at the squat height (0.170 m).  So the policy found a tripod escape
that the hand-written open-loop attempts did not -- which is the point of training it.

That run used the *mixed* reset (70 % 30-40 deg, 30 % tripod).  The current default
is tripod-only; the numbers are below.

### Tripod-only reset (current default)

`INFANTRY_RL_GETUP_REST_PROB = 1.0` (the default), DR on, 300 iterations from
`model_2596.pt`, 256 envs, 361 s:

| iteration | `Metrics/recovery/success` | mean pitch | mean episode |
|---:|---:|---:|---:|
| 2691 | 92 % | 7.7 deg | 91 steps |
| 2790 | 97 % | 6.9 deg | 62 steps |
| 2870-2895 | **94-100 % (mean ~97 %)** | ~6.6 deg | 50-65 steps (0.5-0.65 s) |

Closed-loop probe under DR (11 starts, one env each): **11/11 upright** -- 10, 15,
16, 17, 18, 19, 20, 25, 30, 35 and 40 deg, every one ending at |pitch| < 1.5 deg and
base_z ~0.170 m.  Training only on the tripod does **not** lose the 30-40 deg case:
it generalises across the whole range, and it avoids the bounce of the 30-40 deg
start.  Checkpoint: `checkpoints/getup_tripod_model_2895.pt`.

### Domain randomization: on vs off (measured)

Both runs start from `model_2596.pt` (the balance checkpoint), 256 envs, and are
then evaluated with `tools/probe_recovery.py --task getup_torque` (DR **on**, 17
starts from -40 to +40 deg, terminations disabled).  These two used the mixed reset:

| training | iterations | probe result under DR |
|---|---:|---|
| DR **off** | 200 | 10/17 upright (59 %) |
| DR **on** | 300 | **16/17 upright (94 %)**, only +40 deg fails |

So a DR-off policy looks perfect in its own environment (100 %) and loses about
40 points the moment friction 0.4-1.2, base mass -2..+3 kg, CoM +-5 cm, motor gain
+-10 % and the 0-1 step action delay are switched on.  Train the deployment model
with DR on -- it is the default, just do not set `INFANTRY_RL_DR=0`.

Interval pushes stay **off** for this task (`INFANTRY_RL_GETUP_PUSH_VEL=0`): the
balance task's 0.3 m/s push fires mid-manoeuvre here and a 0.5 s recovery has no
time to recover from it.  The `[cfg] domain rand` line printed by the inherited
`squat_torque` post-init shows the balance task's push value, not this one.

Verification checkpoints: `checkpoints/getup_demo_model_2795.pt` (DR-off mix,
behaviour demo only), `checkpoints/getup_dr_model_2895.pt` (DR-on mix) and
`checkpoints/getup_tripod_model_2895.pt` (DR-on, tripod-only -- the current
default):

```bat
C:/Users/32560/.conda/envs/isaaclab/python.exe play.py --task getup_torque --headless ^
    --steps 400 --checkpoint checkpoints/getup_tripod_model_2895.pt
```

## Reset

`mdp.reset_root_state_fallen` writes the root pose at the **measured** contact height
for the sampled pitch (plus 1.5 mm), never an offset from the standing height, so the
robot never starts inside the ground.  The hips are written to the squat pose and the
knees to the closed-chain solution by the usual `reset_squat_pose` term.

| population | default | env var |
|---|---|---|
| **tripod** (the resting state) | **100%** | `INFANTRY_RL_GETUP_REST_PROB` (1.0) |
| fallen, the 30-40 deg transient (opt-in) | 0% | `INFANTRY_RL_GETUP_REST_PROB` < 1 + `INFANTRY_RL_GETUP_FALL_DEG_MIN/MAX` |
| tripod band | +-2 deg around 17 deg (15-19) | `INFANTRY_RL_GETUP_REST_DEG_MIN/MAX` |
| roll | +-1.5 deg | `INFANTRY_RL_GETUP_ROLL_DEG` |

Yaw is random; the root velocity is +-0.03 m/s and +-0.05 rad/s.  Set both
`INFANTRY_RL_GETUP_REST_DEG_MIN` and `..._MAX` to `17` for a single exact angle.

## Rewards

| term | weight | why |
|---|---:|---|
| `alive` | +0.5 | do not end the episode |
| `upright` (`sum(g_xy^2)`) | -3.0 | 0 upright, 1 on its side, 2 upside down |
| `pitch` (`|pitch|`) | -6.0 | -1.8/step at the tripod, beats `alive` |
| `roll` | -1.0 | |
| `base_height` | -4.0 | weak sanity term; the root is near the wheel axle, so height barely moves with pitch |
| `lin_vel_z` / `ang_vel_y` / `ang_vel_x` / `yaw_rate` | -1.0 / -0.2 / -0.05 / -0.25 | damping |
| `success` | **+40** | one-shot, paid on the step `upright_stable` terminates |
| `action_rate` / `joint_torques` / `joint_acc` | -1e-2 / -2e-4 / -1e-7 | smoothness |
| `wheel_spin` | -3e-4 | a swing-up needs spin; only endless spin is charged |
| `wheel_axle` | -2.0 | do not skew the two legs |
| `action_jerk` / `hip_vel` | -5e-3 / -1e-3 | |
| `closure_residual` | -2e3 | keep the five-bar closed |
| `hip_lock` | -0.1 | weak: the hips must stay usable for the recovery |
| `base_contact` | -4.0 | the corner is the *start* state, so this is a nudge, not the balance task's -20 |

## Termination

| term | value |
|---|---|
| `upright_enough` | `|pitch| < 6 deg`, `|roll| < 0.15`, base_z in `[0.155, 0.200] m`, `|body rate| < 0.7 rad/s`, held 10 control steps (0.1 s) |
| `fell_flat` | `|pitch| > 65 deg` |
| `roll_limit` | `|roll| > 35 deg` |
| `time_out` | 4 s |

The height **upper** bound is a guard, not the mechanism: extending the legs raises
the base to ~0.287 m and the balance policy cannot take over from there, so a policy
that "stands up tall" must come back down.  The measured recovery never does that --
see below -- so the band simply never binds.

### What actually does the work: the wheels, not the hips

Traced with `tools/probe_recovery.py --trace_every 5` on the fine-tuned policy,
starting in the 17 deg tripod:

| t | pitch | base_z | wheel speed | hip target error | hip torque |
|---:|---:|---:|---:|---:|---:|
| 0.00 s | 16.9 deg | 0.168 m | -0.5 / -0.6 rad/s | 0.002 rad | 0.08 Nm |
| 0.08 s | 12.8 deg | 0.167 m | -6.6 / +7.3 rad/s | 0.002 rad | 0.63 Nm |
| 0.20 s | 5.0 deg | 0.171 m | -7.2 / +4.9 rad/s | 0.002 rad | 0.42 Nm |
| 0.36 s | 1.7 deg | 0.171 m | -3.7 / +4.0 rad/s | 0.002 rad | 0.39 Nm |

The hip **action** channels are saturated (raw +5.3 / -5.4 / -4.1 / +1.9), but the
action term's `limit_margin = 0.01` clamps the hip *targets* to 1.8 mrad from the
squat pose, so the hip torque stays under 1 Nm out of the 40 Nm available, and the
base height never leaves 0.167-0.172 m: the legs stay in the retracted stance and the
whole manoeuvre is the wheels.  In other words the vehicle really is the inverted
pendulum the task description assumes -- the hips only *hold* the squat pose.

(An earlier hand-written open-loop test showed that driving the hips to full
extension *can* lift the body to 0.22 m, but the learned policy does not do that and
does not need to.)

## Training

Interface check first (it verifies the same 125/6 layout, a real on-the-ground reset,
and that 0.6 s of zero action does not terminate):

```bat
C:/Users/32560/.conda/envs/isaaclab/python.exe train.py --task getup_torque --headless --check --num_envs 8
```

Fine-tune from the balance checkpoint:

```bat
C:/Users/32560/.conda/envs/isaaclab/python.exe train.py --task getup_torque --resume --resume_latest ^
    --num_envs 1024 --iterations 400 --reset_obs_norm_count 5000000 --run_name getup_from_balance
```

`--reset_obs_norm_count 5000000` matters: the balance checkpoint's observation
normalizer only ever saw +-1.5 deg of pitch, so the fallen gravity vector is tens of
sigma outside its statistics.  Re-opening the count lets it adapt within ~50
iterations while staying slow enough (< 1 % drift per rollout) not to corrupt PPO's
importance ratios.  From scratch, drop `--resume` and train ~800 iterations.

Playback and the closed-loop probe:

```bat
C:/Users/32560/.conda/envs/isaaclab/python.exe play.py --task getup_torque --headless --steps 2000 ^
    --load_run <run> --checkpoint model_<N>.pt
C:/Users/32560/.conda/envs/isaaclab/python.exe tools/probe_recovery.py --headless ^
    --checkpoint <model.pt> --pitches -40 -35 -30 -20 -15 0 15 17 20 30 35 40
```

`play.py --task getup_torque` starts every episode on the ground, so the printed
`pitch` trace shows the recovery itself.  Measured with the checkpoint above
(4 envs, 400 steps): the first printed state is `base_z=0.218 m, pitch=+33.2 deg`
(a real on-the-ground start) and every reset is `upright_enough` -- 22 successful
recoveries in 400 steps, i.e. one every ~0.45 s per environment.

## Deployment hand-off (recommended: two policies)

Fine-tuning on the recovery task **does** cost some balance robustness, so keep the
two policies separate.  Measured on `play.py --task squat_torque --steps 600`
(16 envs, domain randomization on, seed 42), pitch-limit resets over 600 steps:

| policy | resets | held pitch | base_z |
|---|---:|---:|---:|
| balance `model_2596.pt` | 2 | 0.26 deg | 0.1704 m |
| getup fine-tune `model_2795.pt` | 15 | 0.55 deg | 0.1704 m |

The equilibrium is unchanged (same pitch and height); what degrades is disturbance
rejection, because the recovery task deliberately trains from states where falling
is normal and has no `fall_penalty` (the balance task pays -50 for it).  That is
fine when the recovery policy is only used for the first ~0.5 s.

The firmware rule is one line on state the robot already has (IMU pitch + gyro):

```text
if |pitch| > 10 deg  or  |gyro| > 1.5 rad/s:  run the recovery policy
else:                                         run the balance policy
```

Both policies share the same 25-dim input and 6-dim output, so the switch is a
pointer swap in the inference loop; no re-normalisation or extra state is needed.
The recovery policy's success envelope (6 deg, 0.7 rad/s) sits inside the balance
policy's training envelope (13.75 deg), so the hand-off happens with margin.

A single-policy variant is possible (one network for both), but it needs the
balance reward mixed into the recovery task -- in particular a `fall_penalty`
after the first second -- and enough near-upright resets to keep the balance
behaviour.  That is a separate experiment, not what this task does.

## Env vars

| env var | default | meaning |
|---|---:|---|
| `INFANTRY_RL_GETUP_FALL_DEG_MIN/MAX` | 30 / 40 | the fallen transient (only used when `REST_PROB` < 1) |
| `INFANTRY_RL_GETUP_REST_DEG_MIN/MAX` | 15 / 19 | the tripod band |
| `INFANTRY_RL_GETUP_REST_PROB` | **1.0** | tripod share of the resets (1.0 = tripod only) |
| `INFANTRY_RL_GETUP_ROLL_DEG` | 1.5 | roll noise |
| `INFANTRY_RL_GETUP_CONTACT_MARGIN` | 0.0015 | height above the measured contact |
| `INFANTRY_RL_GETUP_EPISODE` | 4.0 | episode seconds |
| `INFANTRY_RL_GETUP_SUCCESS_PITCH_DEG` | 6.0 | success pitch |
| `INFANTRY_RL_GETUP_SUCCESS_HEIGHT` / `_MAX` | 0.155 / 0.200 | success height band |
| `INFANTRY_RL_GETUP_SUCCESS_RATE` | 0.7 | success body-rate limit |
| `INFANTRY_RL_GETUP_HOLD_STEPS` | 10 | consecutive upright steps |
| `INFANTRY_RL_GETUP_SUCCESS_BONUS` | 40 | terminal reward |
| `INFANTRY_RL_GETUP_FELL_FLAT_DEG` | 65 | failed-recovery pitch |
| `INFANTRY_RL_GETUP_ROLL_LIMIT_DEG` | 35 | roll termination |
| `INFANTRY_RL_GETUP_PUSH_VEL` | 0.0 | interval pushes (off: they hit mid-manoeuvre) |

The observation noise, joint-velocity estimate, friction/mass/CoM/motor/gain domain
randomization and action delay are the **same** as `squat_torque`
(`INFANTRY_RL_OBS_*`, `INFANTRY_RL_DR*`, `INFANTRY_RL_ACTION_DELAY_MAX`).
