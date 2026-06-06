#!/usr/bin/env python3
"""Batch export LAFAN1 BVHs to GMR PKLs and MP4s via the MuJoCo viewport path."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GMR_ROOT = Path("/home/hiyio/HoloMotion/thirdparties/GMR")
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting import KinematicsModel, RobotMotionViewer  # noqa: E402
from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402


CSV_TO_NPZ_SCRIPT = REPO_ROOT / "scripts" / "csv_to_npz_local.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch export LAFAN1 BVHs to PKL + MP4.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/home/hiyio/GMR_hxl/Motion_data/lafan1"),
        help="Folder containing input .bvh files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/hiyio/.openclaw/workspace-video-pipeline/lafan1_z1_batch_full"),
        help="Root folder to write PKL/CSV/NPZ/MP4 outputs.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="magicbot_z1",
        help="Target robot understood by GMR.",
    )
    parser.add_argument(
        "--format",
        choices=("lafan1", "nokov"),
        default="lafan1",
        help="BVH loader format.",
    )
    parser.add_argument("--video_width", type=int, default=960, help="Viewport video width.")
    parser.add_argument("--video_height", type=int, default=540, help="Viewport video height.")
    parser.add_argument("--input_fps", type=int, default=30, help="Input FPS for CSV->NPZ.")
    parser.add_argument("--output_fps", type=int, default=50, help="Output FPS for CSV->NPZ.")
    parser.add_argument(
        "--sim_env",
        type=str,
        default=os.environ.get("SIM_ENV", "BeyondMimic"),
        help="Conda env used for csv_to_npz_local.py.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before exporting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-export files even if PKL/MP4 already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many BVHs to process.",
    )
    return parser.parse_args()


def _write_progress(progress_path: Path, payload: dict) -> None:
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _motion_base_name(motion_name: str, robot: str) -> str:
    return f"{motion_name}_{robot}"


def _motion_output_paths(output_dir: Path, motion_name: str, robot: str) -> tuple[Path, Path, Path, Path]:
    stem = _motion_base_name(motion_name, robot)
    return (
        output_dir / "pkl" / f"{stem}.pkl",
        output_dir / "pkl" / f"{stem}.stabilization.json",
        output_dir / "pkl" / f"{stem}.mp4",
        output_dir / "csv" / f"{stem}_qpos.csv",
    )


def _export_single_motion(
    bvh_path: Path,
    robot: str,
    bvh_format: str,
    video_width: int,
    video_height: int,
    output_dir: Path,
    progress_path: Path,
) -> dict:
    motion_name = bvh_path.stem
    pkl_path, json_path, mp4_path, csv_path = _motion_output_paths(output_dir, motion_name, robot)
    started_at = time.time()
    _write_progress(
        progress_path,
        {
            "motion": motion_name,
            "status": "started",
            "bvh_path": str(bvh_path),
            "timestamp": started_at,
        },
    )

    frames, actual_human_height = load_bvh_file(str(bvh_path), format=bvh_format)
    retargeter = GMR(
        src_human=f"bvh_{bvh_format}",
        tgt_robot=robot,
        actual_human_height=actual_human_height,
        verbose=False,
    )

    qpos_list = []
    for frame in frames:
        qpos = retargeter.retarget(frame)
        qpos_list.append(qpos.copy())
    qpos_list = np.asarray(qpos_list, dtype=np.float32)

    root_pos = qpos_list[:, :3]
    root_rot = qpos_list[:, 3:7][:, [1, 2, 3, 0]]
    dof_pos = qpos_list[:, 7:]

    kinematics_model = KinematicsModel(retargeter.xml_file, device="cpu")
    num_frames = root_pos.shape[0]
    fk_root_pos = torch.zeros((num_frames, 3), device="cpu")
    fk_root_rot = torch.zeros((num_frames, 4), device="cpu")
    fk_root_rot[:, -1] = 1.0
    local_body_pos, _ = kinematics_model.forward_kinematics(
        fk_root_pos,
        fk_root_rot,
        torch.from_numpy(dof_pos).to(device="cpu", dtype=torch.float),
    )
    body_names = kinematics_model.body_names

    body_pos, _ = kinematics_model.forward_kinematics(
        torch.from_numpy(root_pos).to(device="cpu", dtype=torch.float),
        torch.from_numpy(root_rot).to(device="cpu", dtype=torch.float),
        torch.from_numpy(dof_pos).to(device="cpu", dtype=torch.float),
    )
    lowest_height = torch.min(body_pos[..., 2]).item()
    root_pos[:, 2] = root_pos[:, 2] - lowest_height
    root_pos[:, :2] -= root_pos[0, :2]

    metadata = {
        "pipeline": "raw_ik_mesh_collision_only",
        "stabilizer_applied": False,
        "input_frames": int(len(frames)),
        "output_frames": int(num_frames),
        "fps": 30,
        "ik_limits": [type(limit).__name__ for limit in retargeter.ik_limits],
        "ik_mesh_collision_enabled": bool(getattr(retargeter, "ik_mesh_self_collision_enabled", False)),
        "ik_mesh_geom_pairs": int(getattr(retargeter, "ik_mesh_self_collision_geom_pair_count", 0)),
        "ik_solve_dt": float(getattr(retargeter, "solve_dt", 0.0)),
        "ik_max_iter": int(getattr(retargeter, "max_iter", 0)),
    }

    motion_data = {
        "fps": 30,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": body_names,
        "retarget_stabilization": metadata,
    }
    with pkl_path.open("wb") as f:
        pickle.dump(motion_data, f)
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    qpos = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
    np.savetxt(csv_path, qpos, delimiter=",")

    viewer = RobotMotionViewer(
        robot_type=robot,
        motion_fps=30,
        camera_follow=False,
        record_video=True,
        video_path=str(mp4_path),
        video_width=video_width,
        video_height=video_height,
    )
    try:
        for frame_idx in range(num_frames):
            viewer.step(
                root_pos[frame_idx],
                root_rot[frame_idx, [3, 0, 1, 2]],
                dof_pos[frame_idx],
                rate_limit=False,
                follow_camera=True,
            )
    finally:
        try:
            viewer.close()
        except Exception:
            # Video is already flushed before this occasional GLFW shutdown warning.
            pass

    finished_at = time.time()
    summary = {
        "motion": motion_name,
        "status": "ok",
        "bvh_path": str(bvh_path),
        "pkl_path": str(pkl_path),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "mp4_path": str(mp4_path),
        "frames": int(num_frames),
        "seconds": round(finished_at - started_at, 3),
        "timestamp": finished_at,
    }
    _write_progress(progress_path, summary)
    return summary


def _run_csv_to_npz_batch(args: argparse.Namespace, output_dir: Path, progress_path: Path) -> None:
    csv_dir = output_dir / "csv"
    npz_dir = output_dir / "npz"
    log_dir = output_dir / "logs"
    npz_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda",
        "run",
        "-n",
        args.sim_env,
        "python",
        str(CSV_TO_NPZ_SCRIPT),
        "--input_file",
        str(csv_dir),
        "--robot",
        args.robot,
        "--input_fps",
        str(args.input_fps),
        "--output_fps",
        str(args.output_fps),
        "--output_dir",
        str(npz_dir),
        "--record",
        "--record_backend",
        "viewport",
        "--render",
    ]
    _write_progress(
        progress_path,
        {
            "phase": "csv_to_npz",
            "status": "started",
            "timestamp": time.time(),
            "cmd": cmd,
        },
    )
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=os.environ.copy())
    _write_progress(
        progress_path,
        {
            "phase": "csv_to_npz",
            "status": "ok",
            "timestamp": time.time(),
            "npz_dir": str(npz_dir),
        },
    )


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    progress_path = output_dir / "progress.jsonl"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("pkl", "csv", "npz", "logs"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    bvh_files = sorted(input_dir.glob("*.bvh"))
    if args.limit is not None:
        bvh_files = bvh_files[: args.limit]
    if not bvh_files:
        raise FileNotFoundError(f"No .bvh files found under {input_dir}")

    print(f"[BATCH] Input dir: {input_dir}")
    print(f"[BATCH] Output dir: {output_dir}")
    print(f"[BATCH] Robot: {args.robot}")
    print(f"[BATCH] Files: {len(bvh_files)}")
    print(f"[BATCH] DISPLAY: {os.environ.get('DISPLAY', '')}")
    print(f"[BATCH] SIM_ENV: {args.sim_env}")

    success = 0
    failed = 0
    for index, bvh_path in enumerate(bvh_files, start=1):
        motion_name = bvh_path.stem
        pkl_path, json_path, mp4_path, csv_path = _motion_output_paths(output_dir, motion_name, args.robot)
        if (not args.overwrite) and pkl_path.is_file() and json_path.is_file() and mp4_path.is_file() and csv_path.is_file():
            print(f"[SKIP] {index}/{len(bvh_files)} {motion_name}")
            _write_progress(
                progress_path,
                {
                    "motion": motion_name,
                    "status": "skipped",
                    "bvh_path": str(bvh_path),
                    "timestamp": time.time(),
                },
            )
            success += 1
            continue

        print(f"[RUN] {index}/{len(bvh_files)} {motion_name}")
        try:
            summary = _export_single_motion(
                bvh_path=bvh_path,
                robot=args.robot,
                bvh_format=args.format,
                video_width=args.video_width,
                video_height=args.video_height,
                output_dir=output_dir,
                progress_path=progress_path,
            )
            success += 1
            print(
                f"[OK] {motion_name}: frames={summary['frames']} "
                f"seconds={summary['seconds']} mp4={Path(summary['mp4_path']).name}"
            )
        except Exception as exc:
            failed += 1
            error_payload = {
                "motion": motion_name,
                "status": "error",
                "bvh_path": str(bvh_path),
                "timestamp": time.time(),
                "error": repr(exc),
            }
            _write_progress(progress_path, error_payload)
            print(f"[ERR] {motion_name}: {exc!r}", file=sys.stderr)

    if failed == 0:
        _run_csv_to_npz_batch(args, output_dir, progress_path)

    print(f"[DONE] success={success} failed={failed} total={len(bvh_files)}")


if __name__ == "__main__":
    main()
