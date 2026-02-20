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
import sys
import time
import shutil
import subprocess
import math
from importlib import metadata
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
    "--speed_scale",
    type=float,
    default=1.0,
    help="Playback speed scale. 1.0=original, >1.0 faster, <1.0 slower.",
)
parser.add_argument(
    "--render",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to render every frame. Disable for faster offline conversion (default: false).",
)

# 录制开关：开了就导出 mp4（和 npz 同名同目录）
parser.add_argument(
    "--record",
    action="store_true",
    help="Enable recording to mp4 (same dir/name as npz, with .mp4).",
)
parser.add_argument(
    "--record_backend",
    type=str,
    default="auto",
    choices=("auto", "viewport", "renderer"),
    help="Recording backend: auto | viewport | renderer (default: auto).",
)

# 是否保留中间 PNG 帧
parser.add_argument(
    "--keep_frames",
    action="store_true",
    help="Keep intermediate PNG frames in renderer fallback mode.",
)
parser.add_argument(
    "--disable_5090_noise_fix",
    action="store_true",
    help="Disable automatic render stability fix for Isaac Sim 4.5 on RTX 5090.",
)
parser.add_argument(
    "--camera_follow_front",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Keep camera in front of the character and look at the character (default: true).",
)
parser.add_argument(
    "--camera_distance",
    type=float,
    default=8,
    help="Camera distance in front of character root when --camera_follow_front is enabled.",
)
parser.add_argument(
    "--camera_height",
    type=float,
    default=0.92,
    help="Camera height offset from root when --camera_follow_front is enabled.",
)
parser.add_argument(
    "--camera_lookat_height",
    type=float,
    default=0.62,
    help="Look-at height offset from root when --camera_follow_front is enabled.",
)
parser.add_argument(
    "--camera_yaw_offset_deg",
    type=float,
    default=-60,
    help="Extra yaw offset (degrees) for a 3/4 front view when --camera_follow_front is enabled.",
)

# 由 AppLauncher 自动添加 --headless、--renderer 等
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.keep_frames:
    print("[INFO] --keep_frames 仅在 renderer.capture 回退模式下生效。")

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

# ======== 录制相关依赖（优先 viewport，失败则回退 renderer.capture） ========
_capture_available = False
_capture_backend = "none"  # "viewport" | "renderer" | "none"
_capture_backend_requested = args_cli.record_backend
_capture = None
_capture_options_cls = None
_capture_range_type = None
_vp_utils = None
_renderer_capture = None
_renderer_iface = None
try:
    import omni.kit.app as kit_app

    ext_mgr = kit_app.get_app().get_extension_manager()
    for ext_name in ("omni.kit.capture.viewport", "omni.kit.viewport.utility", "omni.videoencoding"):
        if not ext_mgr.is_extension_enabled(ext_name):
            ext_mgr.set_extension_enabled_immediate(ext_name, True)

    import omni.kit.capture.viewport as viewport_capture
    import omni.kit.viewport.utility as vp_utils

    _capture = viewport_capture.CaptureExtension.get_instance()
    _capture_options_cls = viewport_capture.CaptureOptions
    _capture_range_type = viewport_capture.CaptureRangeType
    _vp_utils = vp_utils

    if _capture is None:
        raise RuntimeError("CaptureExtension.get_instance() 返回 None")

    _capture_available = True
    _capture_backend = "viewport"
    print("[Capture] omni.kit.capture.viewport 初始化成功")
except Exception as e:
    print(f"[Capture] viewport 录制不可用，将尝试 renderer.capture 回退: {e}")
    try:
        import omni.kit.renderer.capture as renderer_capture

        _renderer_capture = renderer_capture
        _renderer_iface = renderer_capture.acquire_renderer_capture_interface()
        if _renderer_iface is None:
            raise RuntimeError("acquire_renderer_capture_interface() 返回 None")
        _renderer_iface.startup()
        _renderer_iface.start_frame_updates()
        _capture_available = True
        _capture_backend = "renderer"
        print("[Capture] omni.kit.renderer.capture 初始化成功（回退模式）")
    except Exception as e2:
        print(f"[Capture] renderer.capture 回退也失败: {e2}")
        _capture_available = False
        _capture_backend = "none"

# 无论 viewport 是否可用，都尽量初始化 renderer 接口，便于 4.5 自动切换后端。
if _renderer_iface is None:
    try:
        import omni.kit.renderer.capture as renderer_capture

        _renderer_capture = renderer_capture
        _renderer_iface = renderer_capture.acquire_renderer_capture_interface()
        if _renderer_iface is not None:
            _renderer_iface.startup()
            _renderer_iface.start_frame_updates()
    except Exception:
        _renderer_iface = None

# 录制后端策略：
# - 用户显式指定优先
# - auto 模式下，Isaac Sim 4.5 默认使用 renderer，避免 viewport 在 4.5 上卡 CAPTURING/时长异常
try:
    _isaacsim_ver_for_record = metadata.version("isaacsim")
except Exception:
    _isaacsim_ver_for_record = "unknown"

if _capture_backend_requested == "renderer":
    if _renderer_iface is not None:
        _capture_backend = "renderer"
        _capture_available = True
        print("[Capture] 使用用户指定后端: renderer")
    else:
        print("[Capture] 用户指定 renderer，但不可用；回退到自动选择。")
elif _capture_backend_requested == "viewport":
    if _capture is not None:
        _capture_backend = "viewport"
        _capture_available = True
        print("[Capture] 使用用户指定后端: viewport")
    else:
        print("[Capture] 用户指定 viewport，但不可用；回退到自动选择。")
else:
    if _isaacsim_ver_for_record.startswith("4.5") and _renderer_iface is not None:
        _capture_backend = "renderer"
        _capture_available = True
        print("[Capture] auto 模式检测到 Isaac Sim 4.5，默认改用 renderer 后端以保证时长/退出稳定。")


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
    def __init__(self, motion_file, input_fps, output_fps, speed_scale, device, frame_range):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.speed_scale = float(speed_scale)
        if self.speed_scale <= 0:
            raise ValueError(f"speed_scale must be > 0, got {self.speed_scale}")
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

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

        self.input_frames = motion.shape[0]
        # Clip duration follows frame-count convention: N frames at fps => N/fps seconds.
        self.duration = max(self.input_frames * self.input_dt, 1e-6)
        print(f"[INFO] Loaded motion: {self.motion_file}")
        print(f"       frames={self.input_frames}, duration={self.duration:.3f}s, speed_scale={self.speed_scale:.3f}")

    def _compute_frame_blend(self, frame_pos):
        idx0 = frame_pos.floor().long()
        idx0 = torch.clamp(idx0, min=0, max=max(self.input_frames - 1, 0))
        idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = frame_pos - idx0
        blend = torch.clamp(blend, min=0.0, max=1.0)
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
        # Preserve clip duration under resampling, then apply speed scaling.
        # speed_scale > 1.0 shortens duration (faster), < 1.0 lengthens duration (slower).
        self.output_frames = max(1, int(round(self.input_frames * self.output_fps / (self.input_fps * self.speed_scale))))
        out_idx = torch.arange(self.output_frames, device=self.device, dtype=torch.float32)
        frame_pos = out_idx * (self.input_fps * self.speed_scale / self.output_fps)
        i0, i1, blend = self._compute_frame_blend(frame_pos)
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
        if rotations.shape[0] < 3:
            return torch.zeros((rotations.shape[0], 3), device=rotations.device, dtype=rotations.dtype)
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def _compute_velocities(self):
        if self.output_frames < 2:
            z3 = torch.zeros((self.output_frames, 3), device=self.device, dtype=self.motion_base_poss.dtype)
            zd = torch.zeros_like(self.motion_dof_poss)
            self.motion_base_lin_vels = z3
            self.motion_dof_vels = zd
            self.motion_base_ang_vels = z3
            return
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


# ========== 4.1 Capture Helpers ==========
def _get_capture_camera_path() -> str | None:
    if _vp_utils is None:
        return None
    viewport = _vp_utils.get_active_viewport()
    if viewport is None:
        return None
    camera_path = getattr(viewport, "camera_path", None)
    if camera_path is None:
        return None
    return getattr(camera_path, "pathString", str(camera_path))


def _capture_status_name() -> str:
    status_map = {
        0: "NONE",
        1: "CAPTURING",
        2: "PAUSED",
        3: "FINISHING",
        4: "TO_START_ENCODING",
        5: "ENCODING",
        6: "CANCELLED",
    }

    def _normalize_status(value) -> str:
        if value is None:
            return "UNKNOWN"
        # enum-like with .name
        name = getattr(value, "name", None)
        if isinstance(name, str) and name:
            return name.upper()
        # int-like enum value
        if isinstance(value, int):
            return status_map.get(int(value), f"UNKNOWN_{value}")
        # string-like fallback
        raw = str(value).strip()
        if raw.isdigit():
            return status_map.get(int(raw), f"UNKNOWN_{raw}")
        if "." in raw:
            raw = raw.split(".")[-1]
        return raw.upper()

    if _capture is None:
        return "NONE"

    progress = getattr(_capture, "progress", None)
    if progress is not None:
        status = getattr(progress, "capture_status", None)
        if status is not None:
            return _normalize_status(status)

    # 兼容新旧 API：旧路径无 progress 时退回 done。
    if hasattr(_capture, "done"):
        try:
            return "NONE" if bool(_capture.done) else "CAPTURING"
        except Exception:
            pass
    return "UNKNOWN"


def _capture_is_active() -> bool:
    status = _capture_status_name().upper()
    return status in {"CAPTURING", "PAUSED", "FINISHING", "TO_START_ENCODING", "ENCODING"}


def _get_frame_dir(base_name: str) -> str:
    frame_dir = os.path.join(args_cli.output_dir, f"frames_{base_name}")
    os.makedirs(frame_dir, exist_ok=True)
    return frame_dir


def _frame_file(frame_dir: str, base_name: str, frame_idx: int) -> str:
    return os.path.join(frame_dir, f"{base_name}_{frame_idx:05d}.png")


def _capture_frame_renderer(frame_dir: str, base_name: str, frame_idx: int):
    if _capture_backend != "renderer" or _renderer_iface is None:
        return
    try:
        _renderer_iface.capture_next_frame_swapchain_to_file(_frame_file(frame_dir, base_name, frame_idx))
    except Exception as e:
        print(f"[Capture] renderer.capture 抓帧失败: {e}")


def _frames_to_mp4_renderer(frame_dir: str, base_name: str, fps: int):
    mp4_path = os.path.join(args_cli.output_dir, f"{base_name}.mp4")
    pattern = os.path.join(frame_dir, f"{base_name}_%05d.png")

    if shutil.which("ffmpeg") is None:
        print("[Capture] 未检测到 ffmpeg，跳过 mp4 合成。")
        return

    try:
        if _renderer_iface is not None:
            _renderer_iface.wait_async_capture()
    except Exception as e:
        print(f"[Capture] wait_async_capture 失败: {e}")

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

    if args_cli.keep_frames:
        print(f"[Capture] 保留帧目录: {frame_dir}")
    else:
        try:
            shutil.rmtree(frame_dir)
        except Exception as e:
            print(f"[Capture] 删除帧目录失败: {e}")


def _frames_to_mp4_viewport(frame_dir: str, base_name: str, fps: int, expected_frames: int | None = None):
    mp4_path = os.path.join(args_cli.output_dir, f"{base_name}.mp4")
    pattern = os.path.join(frame_dir, "*.png")

    if shutil.which("ffmpeg") is None:
        print("[Capture] 未检测到 ffmpeg，无法把 viewport 帧目录合成 mp4。")
        return

    frame_files = sorted(Path(frame_dir).glob("*.png"))
    captured_frames = len(frame_files)
    if captured_frames <= 0:
        print(f"[Capture] viewport 帧目录为空，无法合成 mp4: {frame_dir}")
        return

    # Keep target playback fps fixed. If viewport produced extra frames, trim tail.
    # If viewport dropped frames, stretch timeline to preserve the target duration.
    vf_filters = ["scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    output_frame_limit = None
    if expected_frames is not None and expected_frames > 0 and captured_frames != expected_frames:
        if captured_frames > expected_frames:
            output_frame_limit = int(expected_frames)
            print(
                f"[Capture] viewport 帧数偏多: captured={captured_frames}, expected={expected_frames}。"
                " 将截断尾部多余帧以对齐时长。"
            )
        else:
            stretch = float(expected_frames) / float(max(captured_frames, 1))
            vf_filters.insert(0, f"setpts={stretch:.8f}*PTS")
            print(
                f"[Capture] viewport 帧数偏少: captured={captured_frames}, expected={expected_frames}。"
                f" 将按 setpts={stretch:.4f} 拉伸时间轴以对齐时长。"
            )

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        pattern,
    ]
    if output_frame_limit is not None:
        cmd += [
            "-frames:v",
            str(output_frame_limit),
        ]
    cmd += [
        "-vf",
        ",".join(vf_filters),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        mp4_path,
    ]
    print("[Capture] Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)

    if args_cli.keep_frames:
        print(f"[Capture] 保留帧目录: {frame_dir}")
    else:
        try:
            shutil.rmtree(frame_dir)
        except Exception as e:
            print(f"[Capture] 删除帧目录失败: {e}")


def _start_video_capture(base_name: str, total_frames: int):
    if not args_cli.record:
        return None
    if not _capture_available:
        print("[Capture] 录制不可用，跳过 mp4 导出。")
        return None

    if _capture_backend == "renderer":
        frame_dir = _get_frame_dir(base_name)
        print(f"[Capture] 使用 renderer.capture 回退模式录制（先导出帧再合成 mp4）: {frame_dir}")
        return {"backend": "renderer", "frame_dir": frame_dir, "frame_idx": 0}

    if _capture_backend != "viewport" or _capture is None:
        print("[Capture] 未识别的录制后端，跳过 mp4 导出。")
        return None

    camera_path = _get_capture_camera_path()
    if camera_path is None:
        print("[Capture] 未找到 active viewport/camera，跳过 mp4 导出（通常需要非 headless 模式）。")
        return None

    try:
        if _capture_is_active():
            _capture.cancel()

        options = _capture_options_cls()
        options.camera = camera_path
        options.output_folder = args_cli.output_dir
        options.file_name = base_name
        # Isaac Sim 4.5 下 viewport 直接 mp4 容易卡在 CAPTURING。
        # 改为先导出 viewport 帧序列，再由 ffmpeg 合成为 mp4。
        use_viewport_png_sequence = _isaacsim_ver_for_record.startswith("4.5")
        options.file_type = ".png" if use_viewport_png_sequence else ".mp4"
        options.range_type = _capture_range_type.FRAMES
        options.start_frame = 1
        options.end_frame = max(1, int(total_frames))
        options.capture_every_Nth_frames = 1
        options.fps = args_cli.output_fps
        options.overwrite_existing_frames = True
        _capture.options = options

        started = bool(_capture.start())
        if started:
            if use_viewport_png_sequence:
                print(
                    f"[Capture] 录制已开始(viewport->png序列): "
                    f"{os.path.join(args_cli.output_dir, base_name + '_frames')} "
                    f"(frames={options.start_frame}-{options.end_frame}, fps={options.fps})"
                )
            else:
                print(
                    f"[Capture] 录制已开始: {os.path.join(args_cli.output_dir, base_name + '.mp4')} "
                    f"(frames={options.start_frame}-{options.end_frame}, fps={options.fps})"
                )
        else:
            print("[Capture] start() 返回 False，未开始录制。")
        if started:
            return {
                "backend": "viewport",
                "viewport_png_sequence": use_viewport_png_sequence,
                "expected_frames": int(total_frames),
            }
        return None
    except Exception as e:
        print(f"[Capture] 启动录制失败: {e}")
        return None


def _finalize_video_capture(sim, base_name: str, capture_ctx, timeout_s: float = 120.0):
    if not args_cli.record or capture_ctx is None:
        return
    backend = capture_ctx.get("backend")

    if backend == "renderer":
        _frames_to_mp4_renderer(capture_ctx["frame_dir"], base_name, args_cli.output_fps)
        return
    if backend != "viewport" or _capture is None:
        return

    viewport_png_sequence = bool(capture_ctx.get("viewport_png_sequence", False))
    expected_frames = int(capture_ctx.get("expected_frames", 0)) or None
    if viewport_png_sequence:
        # Important: do NOT keep calling sim.render() here. Otherwise capture may keep
        # appending static tail frames after the motion itself has ended.
        settle_timeout_s = min(max(1.0, timeout_s), 5.0)
        deadline = time.time() + settle_timeout_s
        last_log_time = 0.0
        while _capture_is_active() and time.time() < deadline:
            time.sleep(0.05)
            now = time.time()
            if now - last_log_time > 1.0:
                print(f"[Capture] 等待 viewport 收尾... status={_capture_status_name()}")
                last_log_time = now
        if _capture_is_active():
            try:
                _capture.cancel()
            except Exception:
                pass
            print(
                f"[Capture] viewport 收尾超时（>{settle_timeout_s:.0f}s），已取消录制以避免继续追加帧。"
            )
        # Give extension a brief moment to flush file handles.
        time.sleep(0.15)
    else:
        deadline = time.time() + max(1.0, timeout_s)
        last_log_time = 0.0
        while simulation_app.is_running() and _capture_is_active() and time.time() < deadline:
            sim.render()
            now = time.time()
            if now - last_log_time > 5.0:
                print(f"[Capture] 等待编码完成... status={_capture_status_name()}")
                last_log_time = now

        if _capture_is_active():
            try:
                _capture.cancel()
                # 让 capture 扩展再跑几帧，确保收尾完成
                for _ in range(10):
                    sim.render()
            except Exception:
                pass
            print(
                f"[Capture] 等待录制完成超时（>{timeout_s:.0f}s），"
                f"当前状态={_capture_status_name()}，文件可能未完整写入。"
            )

    outputs = []
    if hasattr(_capture, "get_outputs"):
        try:
            outputs = _capture.get_outputs(validate=False)
        except TypeError:
            outputs = _capture.get_outputs()
        except Exception as e:
            print(f"[Capture] 获取输出文件列表失败: {e}")
            outputs = []

    if outputs:
        print(f"[Capture] 视频导出完成: {outputs}")
        return

    expected_mp4 = os.path.join(args_cli.output_dir, base_name + ".mp4")
    frame_dir = os.path.join(args_cli.output_dir, f"{base_name}_frames")

    if viewport_png_sequence:
        if os.path.isdir(frame_dir):
            print("[Capture] viewport 帧序列录制结束，开始 ffmpeg 合成 mp4...")
            _frames_to_mp4_viewport(frame_dir, base_name, args_cli.output_fps, expected_frames=expected_frames)
            if os.path.isfile(expected_mp4):
                print(f"[Capture] 视频导出完成（viewport+ffmpeg）: {expected_mp4}")
                return
        print(f"[Capture] 未找到 viewport 帧目录或 mp4 输出: {expected_mp4}")
        return

    if os.path.isfile(expected_mp4):
        print(f"[Capture] 视频导出完成: {expected_mp4}")
    elif os.path.isdir(frame_dir):
        print("[Capture] 未找到 mp4，但检测到 viewport 帧目录，尝试 ffmpeg 合成...")
        _frames_to_mp4_viewport(frame_dir, base_name, args_cli.output_fps, expected_frames=expected_frames)
        if os.path.isfile(expected_mp4):
            print(f"[Capture] 视频导出完成（ffmpeg 兼容合成）: {expected_mp4}")
            return
        print(f"[Capture] 录制结束，但未找到预期输出: {expected_mp4}")
    else:
        print(f"[Capture] 录制结束，但未找到预期输出: {expected_mp4}")


# ========== 5. Simulation Runner ==========
def _as_torch_clone(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=torch.float32).clone()
    return torch.as_tensor(data, dtype=torch.float32, device=device).clone()


def _slice_to_numpy(data_slice):
    if isinstance(data_slice, torch.Tensor):
        return data_slice.detach().cpu().numpy().copy()
    return np.asarray(data_slice).copy()


def _get_isaacsim_version() -> str:
    try:
        return metadata.version("isaacsim")
    except Exception:
        return "unknown"


def _get_primary_gpu_name() -> str:
    try:
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return "unknown"


def _maybe_apply_5090_noise_fix(sim_cfg):
    if args_cli.disable_5090_noise_fix:
        print("[RenderFix] 用户禁用 --disable_5090_noise_fix，跳过自动修复。")
        return

    sim_ver = _get_isaacsim_version()
    gpu_name = _get_primary_gpu_name()
    device_name = str(args_cli.device).lower()

    is_45 = sim_ver.startswith("4.5")
    is_cuda = device_name.startswith("cuda")
    is_5090 = "5090" in gpu_name
    if not (is_45 and is_cuda and is_5090):
        print(f"[RenderFix] 未触发: isaacsim={sim_ver}, device={args_cli.device}, gpu={gpu_name}")
        return

    # Isaac Sim 4.5 + RTX 5090: avoid DLSS-related sparkling/snow noise artifacts.
    sim_cfg.render.antialiasing_mode = "TAA"
    sim_cfg.render.enable_dlssg = False
    sim_cfg.render.enable_dl_denoiser = False
    sim_cfg.render.rendering_mode = "balanced"
    print(
        "[RenderFix] 已启用 5090 雪花噪点修复: "
        "AA=TAA, DLSS-G=off, DL denoiser=off, rendering_mode=balanced"
    )


def _write_motion_state_to_sim(robot, scene, joint_idx, state):
    (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
    ) = state

    root = _as_torch_clone(robot.data.default_root_state, motion_base_pos.device)
    root[:, :3] = motion_base_pos
    root[:, :2] += scene.env_origins[:, :2]
    root[:, 3:7] = motion_base_rot
    root[:, 7:10] = motion_base_lin_vel
    root[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root)

    jp = _as_torch_clone(robot.data.default_joint_pos, motion_dof_pos.device)
    jv = _as_torch_clone(robot.data.default_joint_vel, motion_dof_vel.device)
    jp[:, joint_idx], jv[:, joint_idx] = motion_dof_pos, motion_dof_vel
    robot.write_joint_state_to_sim(jp, jv)


def _quat_wxyz_to_yaw(quat_wxyz: torch.Tensor) -> float:
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _update_front_camera(sim, state, cam_ctx=None):
    if not args_cli.camera_follow_front:
        return
    if cam_ctx is not None and cam_ctx.get("locked", False):
        return
    motion_base_pos = state[0]
    motion_base_rot = state[1]
    if motion_base_pos.numel() < 3 or motion_base_rot.numel() < 4:
        return

    root_pos = motion_base_pos[0].detach().cpu()
    root_quat = motion_base_rot[0].detach().cpu()  # wxyz
    yaw = _quat_wxyz_to_yaw(root_quat) + math.radians(float(args_cli.camera_yaw_offset_deg))
    forward_xy = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)

    cam_pos = np.array(
        [
            float(root_pos[0]) + args_cli.camera_distance * forward_xy[0],
            float(root_pos[1]) + args_cli.camera_distance * forward_xy[1],
            float(root_pos[2]) + args_cli.camera_height,
        ],
        dtype=np.float64,
    )
    target = np.array(
        [
            float(root_pos[0]),
            float(root_pos[1]),
            float(root_pos[2]) + args_cli.camera_lookat_height,
        ],
        dtype=np.float64,
    )

    try:
        sim.set_camera_view(eye=cam_pos.tolist(), target=target.tolist())
    except TypeError:
        sim.set_camera_view(cam_pos.tolist(), target.tolist())
    except Exception as e:
        print(f"[Camera] set_camera_view failed: {e}")
        return

    # Lock camera after first frame: fixed viewpoint facing the first frame.
    if cam_ctx is not None:
        cam_ctx["locked"] = True


def run_simulator_for_file(sim, scene, joint_names, csv_path: str):
    """对单个 csv 跑一遍模拟并保存 npz，并可选录制 mp4。"""
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
        args_cli.speed_scale,
        sim.device,
        args_cli.frame_range,
    )
    motion.reset()

    robot = scene["robot"]
    joint_idx = robot.find_joints(joint_names, preserve_order=True)[0]
    capture_ctx = _start_video_capture(base_name, motion.output_frames)
    cam_ctx = {"locked": False}

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

    while simulation_app.is_running():
        state, reset_flag = motion.get_next_state()
        _write_motion_state_to_sim(robot, scene, joint_idx, state)
        _update_front_camera(sim, state, cam_ctx)

        if capture_ctx is not None and capture_ctx.get("backend") == "renderer":
            capture_ctx["frame_idx"] += 1
            _capture_frame_renderer(capture_ctx["frame_dir"], base_name, capture_ctx["frame_idx"])

        # For offline conversion we only need up-to-date kinematics, not viewport rendering.
        # Rendering each frame is significantly slower for large batch conversion.
        if args_cli.record or args_cli.render:
            sim.render()
        else:
            sim.forward()
        scene.update(sim.get_physics_dt())

        if not file_saved:
            log["joint_pos"].append(_slice_to_numpy(robot.data.joint_pos[0]))
            log["joint_vel"].append(_slice_to_numpy(robot.data.joint_vel[0]))
            log["body_pos_w"].append(_slice_to_numpy(robot.data.body_pos_w[0]))
            log["body_quat_w"].append(_slice_to_numpy(robot.data.body_quat_w[0]))
            log["body_lin_vel_w"].append(_slice_to_numpy(robot.data.body_lin_vel_w[0]))
            log["body_ang_vel_w"].append(_slice_to_numpy(robot.data.body_ang_vel_w[0]))
        
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

            if capture_ctx is not None:
                _finalize_video_capture(sim, base_name, capture_ctx)
            break
 

# ========== 6. Main ==========
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    # dt = 1/output_fps，让录制 fps 和仿真一致
    sim_cfg.dt = 1.0 / args_cli.output_fps
    _maybe_apply_5090_noise_fix(sim_cfg)
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
