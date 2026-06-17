"""Script to play a checkpoint if an RL agent from RSL-RL (no wandb)."""
# python scripts/rsl_rl/play_local.py --task=Tracking-Flat-G1-v0 --num_envs=2
# python scripts/rsl_rl/play_local.py --task=Tracking-Flat-G1-Wo-State-Estimation-v0 --num_envs=2

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import pathlib
import glob
from importlib import metadata

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL (load from logs).")
#TODO no video 
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.") 
parser.add_argument("--video_length", type=int, default=6000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# NEW: allow overriding logs location and checkpoint name
parser.add_argument("--log_dir", type=str, default=None,
                    help="Root dir of the run to load from. Defaults to logs/rsl_rl/<experiment_name>.")
parser.add_argument("--checkpoint_name", type=str, default=None,
                    help="Checkpoint filename to load (e.g., model_0300.pt). If omitted, load the latest.")

# Keep motion_file but also auto-discover below if not provided
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion npz file.")
parser.add_argument(
    "--motion_files",
    type=str,
    nargs="+",
    default=None,
    help="List of motion npz files for mixed playback.",
)
parser.add_argument(
    "--motion_dir",
    type=str,
    default=None,
    help="Directory containing motion npz files (all *.npz will be used).",
)
parser.add_argument(
    "--disable_5090_noise_fix",
    action="store_true",
    default=False,
    help="Disable the automatic Isaac Sim 4.5 + RTX 5090 render noise fix.",
)
parser.add_argument(
    "--hide_motion_debug_vis",
    action="store_true",
    default=False,
    help="Hide the original motion current/goal arrows and helper lines.",
)
parser.add_argument(
    "--hide_contact_debug_vis",
    action="store_true",
    default=False,
    help="Hide contact-force debug markers.",
)
parser.add_argument(
    "--disable_gui_low_vram_fix",
    action="store_true",
    default=False,
    help="Disable the automatic low-VRAM GUI launch settings for local playback.",
)
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override viewer eye position for playback/video.",
)
parser.add_argument(
    "--camera_lookat",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override viewer lookat position for playback/video.",
)
parser.add_argument(
    "--camera_origin_type",
    type=str,
    default=None,
    choices=["world", "asset_root", "env"],
    help="Override viewer origin type for playback/video.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True


def _maybe_apply_gui_low_vram_launch_fix(args: argparse.Namespace) -> None:
    if args.headless:
        return
    if args.disable_gui_low_vram_fix:
        print("[GuiFix] Disabled by --disable_gui_low_vram_fix.")
        return

    # Isaac Sim 4.5 GUI defaults are expensive for local playback on crowded GPUs:
    # 1280x720 internal render resolution, 64 spp, denoiser on, multi-GPU on.
    # For this script we bias toward stable interactive playback over image quality.
    args.width = 960
    args.height = 540
    args.window_width = 1280
    args.window_height = 720
    args.samples_per_pixel_per_frame = 1
    args.denoiser = False
    args.multi_gpu = False
    args.max_bounces = 2
    args.max_specular_transmission_bounces = 2
    args.max_volume_bounces = 0
    args.renderer = "RaytracedLighting"
    print(
        "[GuiFix] Enabled low-VRAM GUI launch fix: "
        "render=960x540, window=1280x720, spp=1, denoiser=off, multi_gpu=off"
    )


_maybe_apply_gui_low_vram_launch_fix(args_cli)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from rsl_rl.runners import OnPolicyRunner

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.utils.rsl_rl_noise_guard import sanitize_scalar_policy_std


REFERENCE_GHOST_NAME = "reference_robot"
REFERENCE_GHOST_PRIM_PATH = "{ENV_REGEX_NS}/ReferenceRobot"
REFERENCE_GHOST_SCALE_FACTOR = 1.035
REFERENCE_GHOST_DIFFUSE_COLOR = (0.10, 1.00, 0.18)
REFERENCE_GHOST_EMISSIVE_COLOR = (0.08, 0.42, 0.10)
REFERENCE_GHOST_OPACITY = 0.55
REFERENCE_MARKERS_ROOT = "/Visuals/ReferenceMarkers"
REFERENCE_MARKER_ANCHOR_RADIUS = 0.065
REFERENCE_MARKER_BODY_RADIUS = 0.032


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


def _maybe_apply_5090_noise_fix(env_cfg) -> None:
    if args_cli.disable_5090_noise_fix:
        print("[RenderFix] Disabled by --disable_5090_noise_fix.")
        return

    sim_ver = _get_isaacsim_version()
    gpu_name = _get_primary_gpu_name()
    device_name = str(getattr(env_cfg.sim, "device", args_cli.device)).lower()

    is_45 = sim_ver.startswith("4.5")
    is_cuda = device_name.startswith("cuda")
    is_5090 = "5090" in gpu_name
    if not (is_45 and is_cuda and is_5090):
        print(f"[RenderFix] Skipped: isaacsim={sim_ver}, device={device_name}, gpu={gpu_name}")
        return

    # Isaac Sim 4.5 + RTX 5090: avoid DLSS-related sparkling/snow noise artifacts in GUI playback.
    env_cfg.sim.render.antialiasing_mode = "TAA"
    env_cfg.sim.render.enable_dlssg = False
    env_cfg.sim.render.enable_dl_denoiser = False
    env_cfg.sim.render.rendering_mode = "balanced"
    print(
        "[RenderFix] Enabled 5090 noise fix: "
        "AA=TAA, DLSS-G=off, DL denoiser=off, rendering_mode=balanced"
    )


def _expected_joint_dims_for_task(task_name: str) -> tuple[tuple[int, ...] | None, str | None]:
    if "MagicBot-Z1" in task_name:
        return (24,), "MagicBot-Z1"
    if "Tracking-Flat-G1" in task_name:
        return (29,), "G1"
    return None, None


def _validate_motion_files_for_task(task_name: str, motion_files: list[str]) -> None:
    expected_joint_dims, robot_label = _expected_joint_dims_for_task(task_name)
    if expected_joint_dims is None:
        return

    mismatches: list[str] = []
    for motion_path in motion_files:
        with np.load(motion_path) as motion_data:
            if "joint_pos" not in motion_data:
                raise KeyError(f"Motion file is missing joint_pos: {motion_path}")
            actual_joint_dim = int(motion_data["joint_pos"].shape[1])
        if actual_joint_dim not in expected_joint_dims:
            mismatches.append(f"{os.path.basename(motion_path)}={actual_joint_dim}")

    if mismatches:
        expected_str = ", ".join(str(dim) for dim in expected_joint_dims)
        raise ValueError(
            f"Motion DOF mismatch for task={task_name} ({robot_label}). "
            f"Expected joint_pos dim in {{{expected_str}}}, but got {', '.join(mismatches)}. "
            "Regenerate the motion with the updated robot model."
        )


def _find_latest_checkpoint(run_dir: str) -> str | None:
    """Return path to latest model_XXXX.pt in run_dir/checkpoints or run_dir."""
    # common patterns
    candidates = []
    # search in 'checkpoints' subdir first
    for pat in ["checkpoints/model_*.pt", "model_*.pt"]:
        candidates.extend(glob.glob(os.path.join(run_dir, pat)))
    if not candidates:
        return None
    # pick the largest XXXX number
    def _step_num(p):
        name = os.path.basename(p)
        # model_XXXX.pt
        try:
            core = name.split(".")[0]
            step = core.split("_")[1]
            return int(step)
        except Exception:
            return -1
    candidates.sort(key=_step_num, reverse=True)
    return candidates[0]


def _auto_find_motion(log_dir: str) -> str | None:
    """Try to find a motion.npz within the run directory."""
    # typical locations: same dir, subdir 'motions', or anywhere under run dir
    for rel in ["motion.npz", "motions/motion.npz"]:
        p = os.path.join(log_dir, rel)
        if os.path.isfile(p):
            return p
    # fallback: first match anywhere under run dir
    for p in glob.glob(os.path.join(log_dir, "**", "motion.npz"), recursive=True):
        return p
    return None


def _auto_find_motions(log_dir: str) -> list[str] | None:
    """Try to find multiple motions under log_dir (motion_list.txt or motions/*.npz)."""
    motion_list = os.path.join(log_dir, "motion_list.txt")
    if os.path.isfile(motion_list):
        motions_dir = os.path.join(log_dir, "motions")
        resolved: list[str] = []
        with open(motion_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                name = parts[0]
                original = parts[1] if len(parts) > 1 else None
                local_path = os.path.join(motions_dir, name)
                if os.path.isfile(local_path):
                    resolved.append(local_path)
                elif original and os.path.isfile(original):
                    resolved.append(original)
        return resolved if resolved else None

    motions_dir = os.path.join(log_dir, "motions")
    if os.path.isdir(motions_dir):
        files = sorted(glob.glob(os.path.join(motions_dir, "*.npz")))
        return files if files else None

    return None


def _resolve_motion_files(args_cli: argparse.Namespace) -> list[str] | None:
    files: list[str] = []
    if args_cli.motion_files:
        files.extend(args_cli.motion_files)
    if args_cli.motion_file:
        if "," in args_cli.motion_file:
            files.extend([p for p in args_cli.motion_file.split(",") if p])
        else:
            files.append(args_cli.motion_file)
    if args_cli.motion_dir:
        if not os.path.isdir(args_cli.motion_dir):
            raise FileNotFoundError(f"Motion dir not found: {args_cli.motion_dir}")
        files.extend(sorted(glob.glob(os.path.join(args_cli.motion_dir, "*.npz"))))
        if not files:
            files.extend(sorted(glob.glob(os.path.join(args_cli.motion_dir, "**", "*.npz"), recursive=True)))

    # expand any glob patterns
    expanded: list[str] = []
    for path in files:
        if any(ch in path for ch in ["*", "?", "["]):
            expanded.extend(sorted(glob.glob(path)))
        else:
            expanded.append(path)

    # de-duplicate while preserving order
    seen = set()
    deduped: list[str] = []
    for path in expanded:
        abspath = os.path.abspath(path)
        if abspath in seen:
            continue
        seen.add(abspath)
        deduped.append(abspath)

    if not deduped:
        return None

    missing = [p for p in deduped if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Motion file(s) not found: {missing}")

    return deduped


def _attach_reference_ghost(scene_cfg) -> bool:
    """Add a green semi-transparent ghost robot for motion playback visualization."""
    robot_cfg = getattr(scene_cfg, "robot", None)
    if robot_cfg is None:
        return False
    if getattr(scene_cfg, REFERENCE_GHOST_NAME, None) is not None:
        return True

    reference_robot_cfg: ArticulationCfg = robot_cfg.copy()
    reference_robot_cfg.prim_path = REFERENCE_GHOST_PRIM_PATH

    spawn_cfg = reference_robot_cfg.spawn
    if spawn_cfg is None:
        return False

    rigid_props = spawn_cfg.rigid_props.copy() if spawn_cfg.rigid_props is not None else sim_utils.RigidBodyPropertiesCfg()
    rigid_props.disable_gravity = True
    rigid_props.kinematic_enabled = False
    spawn_cfg.rigid_props = rigid_props
    current_scale = spawn_cfg.scale if spawn_cfg.scale is not None else (1.0, 1.0, 1.0)
    spawn_cfg.scale = tuple(float(s) * REFERENCE_GHOST_SCALE_FACTOR for s in current_scale)
    if hasattr(spawn_cfg, "make_instanceable"):
        spawn_cfg.make_instanceable = False
    spawn_cfg.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
    spawn_cfg.activate_contact_sensors = False
    spawn_cfg.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=REFERENCE_GHOST_DIFFUSE_COLOR,
        emissive_color=REFERENCE_GHOST_EMISSIVE_COLOR,
        roughness=0.20,
        metallic=0.0,
        opacity=REFERENCE_GHOST_OPACITY,
    )
    spawn_cfg.visual_material_path = "Looks/reference_ghost"
    if spawn_cfg.articulation_props is not None:
        spawn_cfg.articulation_props = spawn_cfg.articulation_props.copy()
        spawn_cfg.articulation_props.enabled_self_collisions = False

    setattr(scene_cfg, REFERENCE_GHOST_NAME, reference_robot_cfg)
    return True


def _get_reference_ghost_handles(env) -> tuple[object, object, object, int] | None:
    base_env = env.unwrapped
    if REFERENCE_GHOST_NAME not in base_env.scene.keys():
        return None
    try:
        motion_command = base_env.command_manager.get_term("motion")
    except KeyError:
        return None

    reference_robot = base_env.scene[REFERENCE_GHOST_NAME]
    root_body_name = reference_robot.body_names[0] if reference_robot.body_names else None
    tracked_body_names = list(getattr(motion_command.cfg, "body_names", []) or [])
    root_body_index = tracked_body_names.index(root_body_name) if root_body_name in tracked_body_names else 0
    return base_env, reference_robot, motion_command, root_body_index


def _finalize_reference_ghost_visuals(base_env) -> None:
    """Re-bind the ghost material on spawned envs so the green tint is visually obvious."""
    if REFERENCE_GHOST_NAME not in base_env.scene.keys():
        return

    stage = getattr(base_env.scene, "stage", None)
    env_prim_paths = getattr(base_env.scene, "env_prim_paths", None)
    if stage is None or not env_prim_paths:
        return

    bound_paths: set[str] = set()
    recolored_mesh_count = 0
    for env_prim_path in env_prim_paths:
        reference_root = f"{env_prim_path}/ReferenceRobot"
        material_path = f"{reference_root}/Looks/reference_ghost"
        if not stage.GetPrimAtPath(reference_root).IsValid() or not stage.GetPrimAtPath(material_path).IsValid():
            continue

        # Root binding gives a global fallback, and per-link visuals make the tint obvious even when
        # the original USD has descendant materials.
        candidate_paths = [reference_root]
        reference_root_prim = stage.GetPrimAtPath(reference_root)
        for child_prim in reference_root_prim.GetChildren():
            visuals_path = f"{child_prim.GetPath()}/visuals"
            if stage.GetPrimAtPath(visuals_path).IsValid():
                candidate_paths.append(visuals_path)

        for prim_path in candidate_paths:
            prim_path = str(prim_path)
            if prim_path in bound_paths:
                continue
            try:
                sim_utils.bind_visual_material(
                    prim_path,
                    material_path,
                    stage=stage,
                    stronger_than_descendants=True,
                )
                bound_paths.add(prim_path)
            except Exception:
                continue

        # Force mesh-level display color/opacity so the ghost stays visibly green even when descendant
        # materials from the original asset win over the root material binding.
        for prim in Usd.PrimRange(stage.GetPrimAtPath(reference_root)):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            gprim = UsdGeom.Gprim(prim)
            try:
                display_color_attr = gprim.GetDisplayColorAttr()
                if not display_color_attr:
                    display_color_attr = gprim.CreateDisplayColorAttr()
                display_color_attr.Set([Gf.Vec3f(*REFERENCE_GHOST_DIFFUSE_COLOR)])

                display_opacity_attr = gprim.GetDisplayOpacityAttr()
                if not display_opacity_attr:
                    display_opacity_attr = gprim.CreateDisplayOpacityAttr()
                display_opacity_attr.Set([REFERENCE_GHOST_OPACITY])
                recolored_mesh_count += 1
            except Exception:
                continue

    if bound_paths or recolored_mesh_count:
        print(
            "[Ghost] Strengthened green appearance: "
            f"material_bindings={len(bound_paths)}, recolored_meshes={recolored_mesh_count}"
        )


def _sync_reference_ghost(reference_robot, motion_command, root_body_index: int) -> None:
    with torch.inference_mode():
        root_state = torch.cat(
            [
                motion_command.body_pos_w[:, root_body_index],
                motion_command.body_quat_w[:, root_body_index],
                motion_command.body_lin_vel_w[:, root_body_index],
                motion_command.body_ang_vel_w[:, root_body_index],
            ],
            dim=-1,
        ).clone()
        joint_pos = motion_command.joint_pos.clone()
        joint_vel = motion_command.joint_vel.clone()
        # Isaac Sim 4.5 can be flaky when writing a cloned debug articulation for multiple envs in one
        # batched call. Updating each env explicitly is slower but stable for local playback.
        env_ids = torch.arange(root_state.shape[0], device=root_state.device, dtype=torch.long)
        for env_id in env_ids:
            env_id_batch = env_id.unsqueeze(0)
            reference_robot.write_root_state_to_sim(root_state[env_id_batch], env_ids=env_id_batch)
            reference_robot.write_joint_state_to_sim(
                joint_pos[env_id_batch], joint_vel[env_id_batch], env_ids=env_id_batch
            )
            # Keep articulation targets aligned so scene.write_data_to_sim() doesn't pull the ghost back to defaults.
            reference_robot.set_joint_position_target(joint_pos[env_id_batch], env_ids=env_id_batch)
            reference_robot.set_joint_velocity_target(joint_vel[env_id_batch], env_ids=env_id_batch)


def _make_reference_marker_cfg(prim_path: str, radius: float) -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "reference": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=REFERENCE_GHOST_DIFFUSE_COLOR,
                    emissive_color=REFERENCE_GHOST_EMISSIVE_COLOR,
                    roughness=0.15,
                    metallic=0.0,
                    opacity=0.95,
                ),
            )
        },
    )


def _get_reference_marker_handles(env) -> tuple[object, object, list[VisualizationMarkers], list[VisualizationMarkers]] | None:
    base_env = env.unwrapped
    try:
        motion_command = base_env.command_manager.get_term("motion")
    except KeyError:
        return None

    anchor_markers: list[VisualizationMarkers] = []
    body_markers: list[VisualizationMarkers] = []
    for env_id in range(base_env.scene.num_envs):
        anchor_marker = VisualizationMarkers(
            _make_reference_marker_cfg(
                f"{REFERENCE_MARKERS_ROOT}/env_{env_id}/anchor",
                REFERENCE_MARKER_ANCHOR_RADIUS,
            )
        )
        body_marker = VisualizationMarkers(
            _make_reference_marker_cfg(
                f"{REFERENCE_MARKERS_ROOT}/env_{env_id}/bodies",
                REFERENCE_MARKER_BODY_RADIUS,
            )
        )
        anchor_marker.set_visibility(True)
        body_marker.set_visibility(True)
        anchor_markers.append(anchor_marker)
        body_markers.append(body_marker)

    print(
        "[Markers] Created green reference markers: "
        f"envs={base_env.scene.num_envs}, bodies_per_env={len(getattr(motion_command.cfg, 'body_names', []) or [])}"
    )
    return base_env, motion_command, anchor_markers, body_markers


def _sync_reference_markers(
    motion_command,
    anchor_markers: list[VisualizationMarkers],
    body_markers: list[VisualizationMarkers],
) -> None:
    with torch.inference_mode():
        anchor_pos_w = motion_command.anchor_pos_w.detach().clone()
        body_pos_w = motion_command.body_pos_w.detach().clone()

    for env_id, (anchor_marker, body_marker) in enumerate(zip(anchor_markers, body_markers)):
        anchor_marker.visualize(translations=anchor_pos_w[env_id : env_id + 1])
        body_marker.visualize(translations=body_pos_w[env_id])


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent loaded from local logs."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    _maybe_apply_5090_noise_fix(env_cfg)

    if args_cli.camera_eye is not None:
        env_cfg.viewer.eye = tuple(float(v) for v in args_cli.camera_eye)
    if args_cli.camera_lookat is not None:
        env_cfg.viewer.lookat = tuple(float(v) for v in args_cli.camera_lookat)
    if args_cli.camera_origin_type is not None:
        env_cfg.viewer.origin_type = args_cli.camera_origin_type
    if (
        args_cli.camera_eye is not None
        or args_cli.camera_lookat is not None
        or args_cli.camera_origin_type is not None
    ):
        print(
            "[ViewerOverride] "
            f"eye={getattr(env_cfg.viewer, 'eye', None)}, "
            f"lookat={getattr(env_cfg.viewer, 'lookat', None)}, "
            f"origin_type={getattr(env_cfg.viewer, 'origin_type', None)}, "
            f"asset_name={getattr(env_cfg.viewer, 'asset_name', None)}"
        )

    # resolve run directory
    # default: logs/rsl_rl/<experiment_name>
    default_run_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_dir = os.path.abspath(args_cli.log_dir) if args_cli.log_dir else default_run_dir
    print(f"[INFO] Loading experiment from directory: {run_dir}")

    # resolve checkpoint path
    resume_path = None
    if args_cli.checkpoint_name:
        # explicit file name within run_dir or absolute path passed
        if os.path.isabs(args_cli.checkpoint_name):
            resume_path = args_cli.checkpoint_name
        else:
            # try checkpoints/ subdir first, then run_dir root
            cand1 = os.path.join(run_dir, "checkpoints", args_cli.checkpoint_name)
            cand2 = os.path.join(run_dir, args_cli.checkpoint_name)
            resume_path = cand1 if os.path.isfile(cand1) else cand2
    else:
        # leverage existing helper if available, otherwise fall back to scan
        try:
            resume_path = get_checkpoint_path(run_dir, agent_cfg.load_run, agent_cfg.load_checkpoint)
        except Exception as exc:
            print(f"[WARN] get_checkpoint_path failed for run_dir={run_dir}: {exc}")
            resume_path = None
        if resume_path is None or not os.path.isfile(resume_path):
            resume_path = _find_latest_checkpoint(run_dir)

    if resume_path is None or not os.path.isfile(resume_path):
        raise FileNotFoundError(
            f"Could not locate a checkpoint. Tried run_dir={run_dir}, "
            f"checkpoint_name={args_cli.checkpoint_name or agent_cfg.load_checkpoint}."
        )
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # resolve motion file
    motion_files = _resolve_motion_files(args_cli)
    if motion_files:
        _validate_motion_files_for_task(args_cli.task, motion_files)
        env_cfg.commands.motion.motion_files = motion_files
        env_cfg.commands.motion.motion_file = motion_files[0]
        print(f"[INFO]: Using motion files from CLI: {len(motion_files)} files")
    else:
        auto_motion_files = _auto_find_motions(os.path.dirname(resume_path))
        if auto_motion_files is None:
            auto_motion_files = _auto_find_motions(run_dir)
        if auto_motion_files:
            _validate_motion_files_for_task(args_cli.task, auto_motion_files)
            env_cfg.commands.motion.motion_files = auto_motion_files
            env_cfg.commands.motion.motion_file = auto_motion_files[0]
            print(f"[INFO]: Auto-discovered motion files: {len(auto_motion_files)} files")
        else:
            # fallback to single motion.npz
            auto_motion = _auto_find_motion(os.path.dirname(resume_path))
            if auto_motion is None:
                auto_motion = _auto_find_motion(run_dir)
            if auto_motion:
                _validate_motion_files_for_task(args_cli.task, [auto_motion])
                env_cfg.commands.motion.motion_file = auto_motion
                print(f"[INFO]: Auto-discovered motion file: {auto_motion}")
            else:
                print("[WARN]: motion.npz not found under logs. "
                      "If this task needs motion data, please specify --motion_file <path>.")

    if getattr(getattr(env_cfg, "commands", None), "motion", None) is not None:
        if args_cli.hide_motion_debug_vis:
            env_cfg.commands.motion.debug_vis = False
        if args_cli.hide_contact_debug_vis and getattr(env_cfg.scene, "contact_forces", None) is not None:
            env_cfg.scene.contact_forces.debug_vis = False
    if getattr(env_cfg, "events", None) is not None:
        for event_name in (
            "physics_material",
            "collider_offsets",
            "add_joint_default_pos",
            "base_com",
            "mass_scale",
            "actuator_gains",
            "joint_params",
            "gravity",
            "push_robot",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    if sanitize_scalar_policy_std(ppo_runner.alg.policy):
        print("[WARN] Loaded checkpoint had invalid scalar std values; clamped them for playback.")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=getattr(ppo_runner, "obs_normalizer", None),
        path=export_model_dir,
        filename="policy.onnx",
    )
    # Since wandb removed, tag with run_dir for traceability
    attach_onnx_metadata(env.unwrapped, run_dir, export_model_dir)

    # reset environment
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs, _ = obs
    reference_marker_handles = None
    if not args_cli.headless and not args_cli.hide_motion_debug_vis:
        reference_marker_handles = _get_reference_marker_handles(env)
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            if reference_marker_handles is not None:
                _, motion_command, anchor_markers, body_markers = reference_marker_handles
                _sync_reference_markers(motion_command, anchor_markers, body_markers)
            actions = policy(obs)
            step_result = env.step(actions)
            obs = step_result[0]
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
