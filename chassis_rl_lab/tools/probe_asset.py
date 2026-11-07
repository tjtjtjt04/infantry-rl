"""资产探针：spawn chassis-rl，验证关节顺序/限位/蹲姿高度/五连杆闭链残差。

用法（本机 conda 环境 RM，headless）::

    D:\\miniconda\\envs\\RM\\python.exe tools\\probe_asset.py --num_envs 2 --steps 300

检查项（与 ``_design/00_design_spec.md`` §8.4 对应）：
1. articulation 关节名/顺序、按名字解析后的 10 个关节；
2. 关节限位表（USD 实测 vs 期望）；
3. squat 姿态下 PD 保持后的 base 高度（应 ≈ squat_pose.json 的 base_height）；
4. 五连杆闭链残差 |轮轴点 - 后小腿轴孔锚点|（应 < 5 mm）；
5. knee 目标跟踪误差；
6. hip 正转方向语义：给前 hip +0.1 / 后 hip -0.1（伸腿方向）后 base 高度变化。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="chassis-rl asset probe")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from chassis_rl_lab.kinematics import (  # noqa: E402
    HIP_ORDER,
    KNEE_ORDER,
    WHEEL_ORDER,
    FiveBarSolver,
    LegPlanarModel,
    load_planar_model,
    load_squat_pose,
)
from chassis_rl_lab.robots.chassis_rl_cfg import (  # noqa: E402
    CHASSIS_RL_CFG,
    SQUAT_BASE_HEIGHT,
    SQUAT_HIPS,
    SQUAT_KNEES,
    WHEEL_RADIUS,
)

# 后小腿轴孔锚点在 back_calf link 系下的位置（来自 CAD URDF loop_joint link2 xyz）
BACK_CALF_ANCHOR = {
    "left": (0.21773000, 0.03494500, -0.15966200),
    "right": (-0.21772998, 0.03494390, -0.15966200),
}


@configclass
class ProbeSceneCfg(InteractiveSceneCfg):
    """最小场景：地面 + 机器人 + 灯。"""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = CHASSIS_RL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0)
    )


def _fmt(v: float) -> str:
    return "nan" if v != v else f"{v:.5f}"


def main() -> int:
    num_envs = args_cli.num_envs
    device = args_cli.device if getattr(args_cli, "device", None) is not None else "cuda:0"

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=device))
    scene = InteractiveScene(ProbeSceneCfg(num_envs=num_envs, env_spacing=4.0))
    sim.reset()
    robot = scene["robot"]

    print("\n==================== [probe] 1. 关节顺序 ====================")
    print("articulation joint_names (内部顺序):")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i:2d}] {name}")
    hip_ids, hip_names = robot.find_joints(list(HIP_ORDER), preserve_order=True)
    knee_ids, knee_names = robot.find_joints(list(KNEE_ORDER), preserve_order=True)
    wheel_ids, wheel_names = robot.find_joints(list(WHEEL_ORDER), preserve_order=True)
    print("HIP_ORDER   ->", hip_names)
    print("KNEE_ORDER  ->", knee_names)
    print("WHEEL_ORDER ->", wheel_names)
    body_names = robot.body_names
    body_ids = {name: body_names.index(name) for name in body_names}
    for need in ("base_link", "left_wheel", "right_wheel", "left_back_calf", "right_back_calf"):
        if need not in body_ids:
            raise RuntimeError(f"缺少 body: {need}; 现有: {body_names}")

    print("\n==================== [probe] 2. 关节限位 ====================")
    limits = robot.data.joint_pos_limits[0].cpu()  # (J, 2)
    planar = load_planar_model()
    expected = planar.get("joint_limits")
    for i, name in enumerate(robot.joint_names):
        lo, hi = float(limits[i, 0]), float(limits[i, 1])
        extra = ""
        if expected and name in expected and expected[name][0] is not None:
            elo, ehi = expected[name]
            ok = abs(lo - elo) < 1e-3 and abs(hi - ehi) < 1e-3
            extra = f"  expected=[{elo:.6f},{ehi:.6f}] {'OK' if ok else 'MISMATCH!'}"
        print(f"  {name:<18} limits=[{lo:+.6f}, {hi:+.6f}]{extra}")

    # ---- 初始状态：squat 姿态 ----
    joint_pos = robot.data.default_joint_pos.clone()  # (N, J)，来自 ArticulationCfg.init_state
    joint_pos[:, hip_ids] = torch.tensor(SQUAT_HIPS, device=robot.device, dtype=torch.float32)
    joint_pos[:, knee_ids] = torch.tensor(SQUAT_KNEES, device=robot.device, dtype=torch.float32)
    joint_pos[:, wheel_ids] = 0.0
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = SQUAT_BASE_HEIGHT
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=robot.device))

    solver = FiveBarSolver(device=device)
    squat_hips_t = torch.tensor(SQUAT_HIPS, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    squat_knees_t = torch.tensor(SQUAT_KNEES, device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    prev_knees = squat_knees_t.clone()
    anchor_local = torch.tensor(
        [BACK_CALF_ANCHOR["left"], BACK_CALF_ANCHOR["right"]], device=device, dtype=torch.float32
    )  # (2,3)
    back_calf_ids = [body_ids["left_back_calf"], body_ids["right_back_calf"]]
    wheel_ids_b = [body_ids["left_wheel"], body_ids["right_wheel"]]

    target = torch.zeros(num_envs, len(robot.joint_names), device=device)
    target[:, hip_ids] = squat_hips_t
    target[:, knee_ids] = squat_knees_t
    zero_vel = torch.zeros(num_envs, len(wheel_ids), device=device)

    base_z_hist = []
    closure_hist = []
    track_err_hist = []
    knee_err_hist = []
    for step in range(args_cli.steps):
        measured_hips = robot.data.joint_pos[:, hip_ids]
        knees, _res = solver.solve(measured_hips, prev_knees)
        prev_knees = knees.detach()
        target[:, knee_ids] = knees.detach()
        robot.set_joint_position_target(target)
        robot.set_joint_velocity_target(zero_vel, joint_ids=wheel_ids)
        robot.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        axle = robot.data.body_pos_w[:, wheel_ids_b, :]  # (N,2,3)
        anchor = robot.data.body_pos_w[:, back_calf_ids, :] + quat_apply(
            robot.data.body_quat_w[:, back_calf_ids, :].reshape(-1, 4), anchor_local.repeat(num_envs, 1)
        ).reshape(num_envs, 2, 3)
        closure = torch.linalg.vector_norm(axle - anchor, dim=-1)  # (N,2)
        base_z_hist.append(robot.data.root_pos_w[:, 2].mean().item())
        closure_hist.append(closure.mean().item())
        track_err_hist.append((robot.data.joint_pos[:, hip_ids] - target[:, hip_ids]).abs().mean().item())
        knee_err_hist.append((robot.data.joint_pos[:, knee_ids] - target[:, knee_ids]).abs().mean().item())

    import statistics

    def _tail(hist: list[float], n: int = 50) -> float:
        return statistics.mean(hist[-n:])

    base_z = _tail(base_z_hist)
    closure = _tail(closure_hist)
    print("\n==================== [probe] 3. squat 姿态保持 ====================")
    print(f"  squat_base_height(JSON)      = {SQUAT_BASE_HEIGHT:.5f} m")
    print(f"  measured base height (末 50步) = {base_z:.5f} m   err={base_z - SQUAT_BASE_HEIGHT:+.5f} m")
    print(f"  hip target tracking err      = {_tail(track_err_hist):.5f} rad")
    print(f"  knee target tracking err     = {_tail(knee_err_hist):.5f} rad")
    print(f"  五连杆闭链残差 |axle-anchor|  = {closure * 1000.0:.3f} mm  (期望 < 5 mm)")

    # ---------------- 3b. 详细诊断：为什么 base 高度会掉？ ---------------- #
    print("\n==================== [probe] 3b. squat 姿态诊断 ====================")
    from isaaclab.utils.math import euler_xyz_from_quat  # noqa: PLC0415

    left_model = LegPlanarModel.from_dict(planar, "left")
    jmap = {name: float(robot.data.joint_pos[0, i].item()) for i, name in enumerate(robot.joint_names)}
    print("  关节实测 vs 目标:")
    for name, target_val in zip(("left_front_hip", "left_back_hip", "right_front_hip", "right_back_hip"), SQUAT_HIPS):
        print(f"    {name:<16} meas={jmap[name]:+.5f}  target={float(target_val):+.5f}  err={jmap[name] - float(target_val):+.5f}")
    knee_target = target[0, knee_ids].cpu().tolist()
    for idx, name in enumerate(("left_front_knee", "left_back_knee", "right_front_knee", "right_back_knee")):
        print(f"    {name:<16} meas={jmap[name]:+.5f}  target={knee_target[idx]:+.5f}  err={jmap[name] - knee_target[idx]:+.5f}")
    roll, pitch, _yaw = euler_xyz_from_quat(robot.data.root_quat_w[:1])
    print(f"  机身姿态: roll={float(roll[0]):+.4f} rad  pitch={float(pitch[0]):+.4f} rad")
    root_z = float(robot.data.root_pos_w[0, 2].item())
    axle_world = robot.data.body_pos_w[0, wheel_ids_b, 2].cpu().tolist()
    print(f"  root_z={root_z:.5f} m")
    print(f"  轮轴 world z（实测 body）: left={axle_world[0]:.5f}  right={axle_world[1]:.5f}  (若接触地面应≈轮半径 {WHEEL_RADIUS:.5f})")
    _kfpt, axle_f = left_model.fk_front(jmap["left_front_hip"], jmap["left_front_knee"])
    _kbpt, axle_b = left_model.fk_back(jmap["left_back_hip"], jmap["left_back_knee"])
    axle_base = torch.tensor([axle_f[0], 0.0, axle_f[1]], device=device, dtype=torch.float32)
    axle_fk_world = robot.data.root_pos_w[0] + quat_apply(robot.data.root_quat_w[:1], axle_base.unsqueeze(0))[0]
    print(f"  由实测关节 FK 的轮轴 world z = {float(axle_fk_world[2]):.5f} m")
    print(f"  前链轮轴(base系)={axle_f}  后链锚点(base系) xz={axle_b}  z差={(axle_f[1] - axle_b[1]) * 1000:.3f} mm")
    measured_hips = robot.data.joint_pos[:, hip_ids]
    measured_knees = robot.data.joint_pos[:, knee_ids]
    kin_closure = solver.closure_residual(measured_hips, measured_knees).mean().item()
    print(f"  由实测关节角前向计算的闭链残差 = {kin_closure * 1000.0:.3f} mm")

    print("\n==================== [probe] 4. hip 正转方向语义 ====================")
    # 前 hip 正转 +0.1 / 后 hip 负转 -0.1 = 伸腿方向，base 应升高
    delta = 0.1
    delta_vec = torch.tensor([+delta, -delta, +delta, -delta], device=device, dtype=torch.float32)
    target2 = target.clone()
    target2[:, hip_ids] = squat_hips_t + delta_vec.unsqueeze(0)
    prev_knees = squat_knees_t.clone()
    z_before = robot.data.root_pos_w[:, 2].mean().item()
    for _ in range(120):
        measured_hips = robot.data.joint_pos[:, hip_ids]
        knees, _res = solver.solve(measured_hips, prev_knees)
        prev_knees = knees.detach()
        target2[:, knee_ids] = knees.detach()
        robot.set_joint_position_target(target2)
        robot.set_joint_velocity_target(zero_vel, joint_ids=wheel_ids)
        robot.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    z_after = robot.data.root_pos_w[:, 2].mean().item()
    print(f"  hip delta (front +{delta}, back -{delta}) 后 base 高度: {z_before:.5f} -> {z_after:.5f} m")
    print(f"  变化 = {z_after - z_before:+.5f} m  (伸腿应 > 0)  {'OK' if z_after > z_before else 'UNEXPECTED'}")
    print(f"  wheel_radius = {WHEEL_RADIUS:.4f} m")

    print("\n==================== [probe] 5. 轮子正转方向（实验） ====================")
    # 回到 squat 静止姿态，再给两个轮 +5 rad/s 的速度目标，观察车体 x 方向速度符号。
    from isaaclab.utils.math import quat_apply as _quat_apply  # noqa: PLC0415,F811

    joint_pos = robot.data.default_joint_pos.clone()
    joint_pos[:, hip_ids] = torch.tensor(SQUAT_HIPS, device=robot.device, dtype=torch.float32)
    joint_pos[:, knee_ids] = torch.tensor(SQUAT_KNEES, device=robot.device, dtype=torch.float32)
    joint_pos[:, wheel_ids] = 0.0
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    root_state = robot.data.default_root_state.clone()
    root_state[:, 2] = SQUAT_BASE_HEIGHT
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=robot.device))
    prev_knees = squat_knees_t.clone()
    hold = target.clone()
    hold[:, hip_ids] = squat_hips_t
    wheel_cmd = torch.full((num_envs, len(wheel_ids)), 5.0, device=device)
    for _ in range(60):  # 0.3 s
        measured_hips = robot.data.joint_pos[:, hip_ids]
        knees, _res = solver.solve(measured_hips, prev_knees)
        prev_knees = knees.detach()
        hold[:, knee_ids] = knees.detach()
        robot.set_joint_position_target(hold)
        robot.set_joint_velocity_target(wheel_cmd, joint_ids=wheel_ids)
        robot.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    # base 线速度（世界系 x 分量）：+ 表示车体向 +x（前进）
    v_world = robot.data.root_lin_vel_w.mean(dim=0)
    # base 线速度（机体系 x 分量，倾角不大时符号一致）
    v_body = robot.data.root_lin_vel_b.mean(dim=0)
    print(f"  轮速目标 = +5 rad/s（绕世界 +y 右手正转）")
    print(f"  base 线速度 world = ({float(v_world[0]):+.4f}, {float(v_world[1]):+.4f}, {float(v_world[2]):+.4f}) m/s")
    print(f"  base 线速度 body  = ({float(v_body[0]):+.4f}, {float(v_body[1]):+.4f}, {float(v_body[2]):+.4f}) m/s")
    sign = "前进(+x)" if float(v_world[0]) > 0 else "后退(-x)"
    print(f"  结论：轮子正转（q_wheel>0 绕 +y）= 车体 {sign}；量级应≈|v|/{5.0:.1f}≈{WHEEL_RADIUS:.3f} m/(rad/s)")
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        # 注意：Windows 上 Isaac Sim 的 close() 偶发挂起（会卡住退出），
        # 所以这里只 flush 后直接 os._exit；Kit 资源由进程退出回收。
        import sys as _sys

        _sys.stdout.flush()
        _sys.stderr.flush()
        os._exit(exit_code)

