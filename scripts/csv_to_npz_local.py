"""Replay a motion from a CSV file (or a folder of CSVs) and save as NPZ (no wandb).

Example (single file):
    python csv_to_npz_local.py \
        --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
        --output_fps 50

Example (folder, 自动遍历所有 csv):
    python csv_to_npz_local.py \
        --input_file LAFAN/g1 --input_fps 30 --output_fps 50
"""

import argparse
import os
import glob
import subprocess
import shutil
import sys
from pathlib import Path
import numpy as np

REPO_SOURCE = Path(__file__).resolve().parents[1] / "source" / "whole_body_tracking"
if REPO_SOURCE.is_dir():
    sys.path.insert(0, str(REPO_SOURCE))

from isaaclab.app import AppLauncher

# ========== 1. CLI Arguments ==========
parser = argparse.ArgumentParser(description="Replay motion from CSV and save to NPZ (no wandb).")
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Path to input motion CSV file OR a directory containing CSV files.",
)
parser.add_argument("--input_fps", type=int, default=30, help="FPS of input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Frame range START END (inclusive, 1-based).",
)
parser.add_argument(
    "--output_name",
    type=str,
    help="Name of the output NPZ file (single-file mode only; default: same as input file).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./motions",
    help="Directory to save the output NPZ files (and mp4 if --record).",
)
parser.add_argument("--output_fps", type=int, default=50, help="FPS of output motion.")
parser.add_argument(
    "--render",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to render every frame. Disable for faster offline conversion (default: false).",
)
parser.add_argument(
    "--yup_to_zup",
    action="store_true",
    help="Rotate root motion from Y-up to Z-up before replay/logging.",
)
parser.add_argument(
    "--target_min_z",
    type=float,
    default=0.02,
    help="Target global min body z after optional Y-up->Z-up conversion.",
)

# 录制开关：开了就导出 mp4（和 npz 同名同目录）
parser.add_argument(
    "--record",
    action="store_true",
    help="Enable recording to mp4 (same dir/name as npz, with .mp4).",
)

# 是否保留中间 PNG 帧
parser.add_argument(
    "--keep_frames",
    action="store_true",
    help="Keep intermediate PNG frames (do not delete after mp4 generation).",
)

# 由 AppLauncher 自动添加 --headless、--renderer 等
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 判断是文件还是目录
INPUT_IS_DIR = os.path.isdir(args_cli.input_file)
if INPUT_IS_DIR:
    print(f"[INFO] Input is a directory, will traverse all CSVs in: {args_cli.input_file}")
else:
    print(f"[INFO] Input is a single CSV file: {args_cli.input_file}")

# 创建输出目录（npz 和 mp4 都在这里）
os.makedirs(args_cli.output_dir, exist_ok=True)


def get_base_name(csv_path: str) -> str:
    """所有输出的基名（npz/mp4 都用这个）。"""
    if INPUT_IS_DIR:
        # 目录模式：总是用各自文件名
        base = os.path.splitext(os.path.basename(csv_path))[0]
    else:
        # 单文件模式：优先使用 --output_name，否则用文件名
        if args_cli.output_name is not None:
            base = args_cli.output_name
        else:
            base = os.path.splitext(os.path.basename(csv_path))[0]
    return base


def make_output_path(csv_path: str) -> str:
    """根据当前 csv 路径生成输出 npz 路径。"""
    base = get_base_name(csv_path)
    return os.path.join(args_cli.output_dir, f"{base}.npz")


if INPUT_IS_DIR and args_cli.output_name is not None:
    print("[WARN] input_file 是目录模式，--output_name 将被忽略，自动使用各 CSV 文件名。")


# ========== 2. Launch Isaac App ==========

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

# ======== 录制相关依赖：omni.kit.renderer.capture ========
_capture_available = False
_capture_iface = None
try:
    import omni.kit.renderer.capture as renderer_capture  # 官方文档模块名

    _capture_iface = renderer_capture.acquire_renderer_capture_interface()
    if _capture_iface is not None:
        _capture_iface.startup()
        _capture_iface.start_frame_updates()
        _capture_available = True
        print("[Capture] omni.kit.renderer.capture 初始化成功")
    else:
        print("[Capture] acquire_renderer_capture_interface() 返回 None")
except Exception as e:
    print(f"[Capture] 无法导入或初始化 omni.kit.renderer.capture: {e}")
    _capture_available = False


# ========== 3. Scene Configuration ==========
@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ========== 4. Motion Loader ==========
class MotionLoader:
    def __init__(self, motion_file, input_fps, output_fps, device, frame_range, yup_to_zup=False):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.device = device
        self.frame_range = frame_range
        self.yup_to_zup = yup_to_zup
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    @staticmethod
    def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        aw, ax, ay, az = a.unbind(dim=-1)
        bw, bx, by, bz = b.unbind(dim=-1)
        w = aw * bw - ax * bx - ay * by - az * bz
        x = aw * bx + ax * bw + ay * bz - az * by
        y = aw * by - ax * bz + ay * bw + az * bx
        z = aw * bz + ax * by - ay * bx + az * bw
        return torch.stack((w, x, y, z), dim=-1)

    def _load_motion(self):
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            start, end = self.frame_range
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=start - 1,
                    max_rows=end - start + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]  # to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        if self.yup_to_zup:
            # Rx(+90deg): [x, y, z] -> [x, -z, y]
            rot_mat = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
                device=self.device,
            )
            qfix = torch.tensor(
                [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            self.motion_base_poss_input = self.motion_base_poss_input @ rot_mat.T
            qfix = qfix.expand(self.motion_base_rots_input.shape[0], -1)
            self.motion_base_rots_input = self._quat_mul_wxyz(qfix, self.motion_base_rots_input)
            quat_norm = torch.linalg.norm(self.motion_base_rots_input, dim=-1, keepdim=True).clamp_min(1e-8)
            self.motion_base_rots_input = self.motion_base_rots_input / quat_norm
            print("[INFO] Applied root transform: Y-up -> Z-up")

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"[INFO] Loaded motion: {self.motion_file}")
        print(f"       frames={self.input_frames}, duration={self.duration:.3f}s")

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        idx0 = (phase * (self.input_frames - 1)).floor().long()
        idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - idx0
        return idx0, idx1, blend

    @staticmethod
    def _lerp(a, b, blend):
        return a * (1.0 - blend) + b * blend

    def _slerp(self, a, b, blend):
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device)
        self.output_frames = times.shape[0]
        i0, i1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[i0], self.motion_base_poss_input[i1], blend.unsqueeze(1)
        )
        self.motion_base_rots = self._slerp(self.motion_base_rots_input[i0], self.motion_base_rots_input[i1], blend)
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[i0], self.motion_dof_poss_input[i1], blend.unsqueeze(1)
        )
        print(
            f"[INFO] Interpolated: {self.input_frames}@{self.input_fps}fps "
            f"→ {self.output_frames}@{self.output_fps}fps"
        )

    def _so3_derivative(self, rotations, dt):
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx:self.current_idx + 1],
            self.motion_base_rots[self.current_idx:self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx:self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx:self.current_idx + 1],
            self.motion_dof_poss[self.current_idx:self.current_idx + 1],
            self.motion_dof_vels[self.current_idx:self.current_idx + 1],
        )
        self.current_idx += 1
        reset = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset = True
        return state, reset

    def reset(self):
        self.current_idx = 0


# ========== 4.1 Capture Helpers (frames + ffmpeg) ==========
def _get_frame_dir(base_name: str) -> str:
    """为当前 motion 建一个帧缓存目录，如 ./motions/frames_walk1/"""
    frame_dir = os.path.join(args_cli.output_dir, f"frames_{base_name}")
    os.makedirs(frame_dir, exist_ok=True)
    return frame_dir


def _frame_file(frame_dir: str, base_name: str, frame_idx: int) -> str:
    """生成某一帧的 png 路径，如 frames_walk1/walk1_00001.png"""
    return os.path.join(frame_dir, f"{base_name}_{frame_idx:05d}.png")


def _capture_frame(frame_dir: str, base_name: str, frame_idx: int):
    """请求在下一次 render 后保存一帧 png。"""
    if not args_cli.record or not _capture_available:
        return
    if _capture_iface is None:
        return

    file_path = _frame_file(frame_dir, base_name, frame_idx)

    try:
        # 捕获当前 app window 的 swapchain 到文件
        _capture_iface.capture_next_frame_swapchain_to_file(file_path)
    except Exception as e:
        print(f"[Capture] 捕获帧失败: {e}")


def _frames_to_mp4(frame_dir: str, base_name: str, fps: int):
    """调用 ffmpeg，把缓存帧拼成 base_name.mp4，并自动把分辨率调整为偶数。"""
    if not args_cli.record:
        return

    mp4_path = os.path.join(args_cli.output_dir, f"{base_name}.mp4")
    pattern = os.path.join(frame_dir, f"{base_name}_%05d.png")

    if shutil.which("ffmpeg") is None:
        print("[Capture] 未检测到 ffmpeg，可用 `sudo apt install ffmpeg` 安装。只保留 png 帧，不生成 mp4。")
        return

    # ✅ 先等所有异步 capture 完成，确保不会在删目录后又写新帧出来
    try:
        if _capture_available and _capture_iface is not None:
            _capture_iface.wait_async_capture()
    except Exception as e:
        print(f"[Capture] wait_async_capture 失败: {e}")

    # 用 scale 滤镜把宽高截成最近的偶数，避免 x264 报 width not divisible by 2
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        mp4_path,
    ]
    print("[Capture] Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)

    # 是否保留中间帧
    if args_cli.keep_frames:
        print(f"[Capture] 保留帧目录: {frame_dir}")
    else:
        try:
            shutil.rmtree(frame_dir)
            print(f"[Capture] 删除临时帧目录: {frame_dir}")
        except Exception as e:
            print(f"[Capture] 删除帧目录失败: {e}")


# ========== 5. Simulation Runner ==========
def _write_motion_state_to_sim(robot, scene, joint_idx, state, root_z_offset: float):
    (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
    ) = state

    root = robot.data.default_root_state.clone()
    root[:, :3] = motion_base_pos
    root[:, 2] += root_z_offset
    root[:, :2] += scene.env_origins[:, :2]
    root[:, 3:7] = motion_base_rot
    root[:, 7:10] = motion_base_lin_vel
    root[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root)

    jp, jv = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    jp[:, joint_idx], jv[:, joint_idx] = motion_dof_pos, motion_dof_vel
    robot.write_joint_state_to_sim(jp, jv)


def _estimate_root_z_offset(sim, scene, robot, joint_idx, motion) -> float:
    if not args_cli.yup_to_zup:
        return 0.0

    print("[INFO] Estimating z offset for floor alignment...")
    min_z = float("inf")
    motion.reset()

    while simulation_app.is_running():
        state, reset_flag = motion.get_next_state()
        _write_motion_state_to_sim(robot, scene, joint_idx, state, root_z_offset=0.0)
        sim.forward()
        scene.update(sim.get_physics_dt())
        cur_min_z = float(robot.data.body_pos_w[0, :, 2].min().item())
        min_z = min(min_z, cur_min_z)
        if reset_flag:
            break

    motion.reset()
    if not np.isfinite(min_z):
        print("[WARN] Failed to estimate min body z, using offset=0.0")
        return 0.0

    z_offset = args_cli.target_min_z - min_z
    print(f"[INFO] body_min_z={min_z:.5f}, applying root_z_offset={z_offset:.5f}")
    return z_offset


def run_simulator_for_file(sim, scene, joint_names, csv_path: str):
    """对单个 csv 跑一遍模拟并保存 npz，并可选录制 mp4（通过帧 + ffmpeg）。"""
    print(f"\n[INFO] ===== Processing file: {csv_path} =====")
    output_path = make_output_path(csv_path)
    base_name = get_base_name(csv_path)
    print(f"[INFO] Output NPZ will be saved to: {output_path}")
    if args_cli.record:
        print(f"[INFO] Output MP4 will be saved to: {os.path.join(args_cli.output_dir, base_name + '.mp4')}")

    motion = MotionLoader(
        csv_path,
        args_cli.input_fps,
        args_cli.output_fps,
        sim.device,
        args_cli.frame_range,
        yup_to_zup=args_cli.yup_to_zup,
    )
    motion.reset()

    frame_dir = None
    if args_cli.record and _capture_available:
        frame_dir = _get_frame_dir(base_name)

    robot = scene["robot"]
    joint_idx = robot.find_joints(joint_names, preserve_order=True)[0]
    root_z_offset = _estimate_root_z_offset(sim, scene, robot, joint_idx, motion)

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    frame_idx = 0

    while simulation_app.is_running():
        state, reset_flag = motion.get_next_state()
        _write_motion_state_to_sim(robot, scene, joint_idx, state, root_z_offset=root_z_offset)

        if args_cli.record and frame_dir is not None:
            frame_idx += 1
            _capture_frame(frame_dir, base_name, frame_idx)

        # For offline conversion we only need up-to-date kinematics, not viewport rendering.
        # Rendering each frame is significantly slower for large batch conversion.
        if args_cli.record or args_cli.render:
            sim.render()
        else:
            sim.forward()
        scene.update(sim.get_physics_dt())

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0].cpu().numpy().copy())
        
        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)
            np.savez(output_path, **log)
            print(f"[INFO] Motion saved to {output_path}")

            if args_cli.record and frame_dir is not None:
                _frames_to_mp4(frame_dir, base_name, args_cli.output_fps)
            break
 

# ========== 6. Main ==========
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    # dt = 1/output_fps，让录制 fps 和仿真一致
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO] Isaac Sim setup complete.")

    joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    if INPUT_IS_DIR:
        csv_pattern = os.path.join(args_cli.input_file, "*.csv")
        csv_files = sorted(glob.glob(csv_pattern))
        if not csv_files:
            print(f"[WARN] No CSV files found in directory: {args_cli.input_file}")
            return
        print(f"[INFO] Found {len(csv_files)} CSV files.")
        for csv_path in csv_files:
            run_simulator_for_file(sim, scene, joint_names, csv_path)
    else:
        run_simulator_for_file(sim, scene, joint_names, args_cli.input_file)


if __name__ == "__main__":
    main()
    # SimulationApp teardown occasionally hangs after long recording runs.
    # We hard-exit after all outputs are written to keep batch pipelines progressing.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
