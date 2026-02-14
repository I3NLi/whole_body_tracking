#!/usr/bin/env python3
"""Sequentially convert all AIST raw motion PKLs to per-motion NPZ + MP4.

Output layout (same style as converted/<motion_id>/npz/<motion_id>.npz):
  <output_root>/<motion_id>/
    smpl/<motion_id>.npz
    smplx/<motion_id>.npz
    gmr/<motion_id>.pkl
    csv/<motion_id>_qpos.csv
    npz/<motion_id>.npz
    npz/<motion_id>.mp4

Notes:
  - No parallel processing is used in this script.
  - GMR stage is forced to run with --num_cpus 1.
  - NPZ export uses csv_to_npz_local.py with --record --yup_to_zup to produce MP4.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data)}")
    return data


def save_smpl_from_aist_raw(src_pkl: Path, dst_npz: Path, raw_fps: float) -> None:
    data = load_pickle(src_pkl)
    required = ("smpl_poses", "smpl_trans", "smpl_scaling")
    for key in required:
        if key not in data:
            raise KeyError(f"{src_pkl} missing key '{key}'. keys={list(data.keys())}")

    poses = np.asarray(data["smpl_poses"], dtype=np.float32)
    trans_raw = np.asarray(data["smpl_trans"], dtype=np.float32)
    scaling = float(np.asarray(data["smpl_scaling"]).reshape(-1)[0])

    if poses.ndim != 2:
        raise ValueError(f"{src_pkl}: smpl_poses must be 2D, got {poses.shape}")
    if poses.shape[1] > 72:
        poses = poses[:, :72]
    if poses.shape[1] != 72:
        raise ValueError(f"{src_pkl}: smpl_poses must have 72 dims, got {poses.shape}")
    if trans_raw.ndim != 2 or trans_raw.shape[1] != 3:
        raise ValueError(f"{src_pkl}: smpl_trans must be [T,3], got {trans_raw.shape}")

    # Match existing conversion style used for gBR_sBM... sample:
    # swap Y/Z then scale to meters.
    trans = trans_raw[:, [0, 2, 1]] / scaling

    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dst_npz,
        poses=poses.astype(np.float32),
        trans=trans.astype(np.float32),
        mocap_framerate=np.array(raw_fps, dtype=np.float64),
        betas=np.zeros((10,), dtype=np.float32),
        gender=np.array("neutral"),
    )


def convert_smpl_to_smplx(src_npz: Path, dst_npz: Path) -> None:
    smpl_data = np.load(src_npz, allow_pickle=True)
    data_dict = dict(smpl_data)

    if "betas" in data_dict:
        betas = data_dict["betas"]
        if betas.shape == (10,):
            data_dict["betas"] = np.concatenate([betas, np.zeros(6, dtype=betas.dtype)])
        elif betas.shape not in ((16,), (1, 16)):
            raise ValueError(f"{src_npz}: unexpected betas shape {betas.shape}")

    if "mocap_framerate" in data_dict:
        data_dict["mocap_frame_rate"] = data_dict.pop("mocap_framerate")

    if "poses" not in data_dict:
        raise ValueError(f"{src_npz}: missing poses")
    poses = data_dict["poses"]
    if poses.shape[1] > 72:
        poses = poses[:, :72]
    data_dict["root_orient"] = poses[:, :3]
    data_dict["pose_body"] = poses[:, 3:66]
    if "gender" not in data_dict:
        data_dict["gender"] = np.array("neutral")
    del data_dict["poses"]

    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst_npz, **data_dict)


def convert_gmr_pkl_to_csv(src_pkl: Path, dst_csv: Path) -> None:
    data = load_pickle(src_pkl)
    for key in ("root_pos", "root_rot", "dof_pos"):
        if key not in data:
            raise KeyError(f"{src_pkl}: missing key {key}")

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
    if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
        raise ValueError(f"{src_pkl}: frame mismatch")

    qpos = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst_csv, qpos, delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential AIST raw PKL -> NPZ+MP4 converter.")
    parser.add_argument(
        "--motions_dir",
        type=str,
        default="/home/hiyio/whole_body_tracking/datasets/aistplusplus/fullset/aist_plusplus_final/motions",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/hiyio/whole_body_tracking/datasets/aistplusplus/converted",
    )
    parser.add_argument(
        "--gmr_script",
        type=str,
        default="/home/hiyio/GMR/scripts/smplx_to_robot_dataset.py",
    )
    parser.add_argument(
        "--csv_to_npz_script",
        type=str,
        default="/home/hiyio/whole_body_tracking/scripts/csv_to_npz_local.py",
    )
    parser.add_argument("--gmr_env", type=str, default="gmr")
    parser.add_argument("--isaac_env", type=str, default="env_isaaclab")
    parser.add_argument("--raw_fps", type=float, default=60.0)
    parser.add_argument("--input_fps", type=int, default=30)
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Optional frame range passed to csv_to_npz_local.py (inclusive, 1-based).",
    )
    parser.add_argument("--target_min_z", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    motions_dir = Path(args.motions_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    gmr_script = Path(args.gmr_script).expanduser().resolve()
    csv_to_npz_script = Path(args.csv_to_npz_script).expanduser().resolve()
    if not motions_dir.is_dir():
        raise FileNotFoundError(f"motions_dir not found: {motions_dir}")
    if not gmr_script.is_file():
        raise FileNotFoundError(f"gmr_script not found: {gmr_script}")
    if not csv_to_npz_script.is_file():
        raise FileNotFoundError(f"csv_to_npz_script not found: {csv_to_npz_script}")

    motion_ids = sorted(p.stem for p in motions_dir.glob("*.pkl"))
    if args.limit is not None:
        motion_ids = motion_ids[: args.limit]
    if not motion_ids:
        print("[INFO] No motion files found.")
        return

    print(f"[INFO] selected motions: {len(motion_ids)}", flush=True)

    cache_root = output_root / "_batch_cache"
    smplx_flat_dir = cache_root / "smplx"
    gmr_flat_dir = cache_root / "gmr"
    csv_pending_dir = cache_root / "csv_pending"
    npz_video_flat_dir = cache_root / "npz_video"
    smplx_flat_dir.mkdir(parents=True, exist_ok=True)
    gmr_flat_dir.mkdir(parents=True, exist_ok=True)
    npz_video_flat_dir.mkdir(parents=True, exist_ok=True)
    if csv_pending_dir.exists():
        shutil.rmtree(csv_pending_dir)
    csv_pending_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: raw pkl -> smpl/smplx (per-motion), plus flat smplx links for GMR.
    smpl_written = 0
    smplx_written = 0
    for motion_id in motion_ids:
        raw_pkl = motions_dir / f"{motion_id}.pkl"
        motion_dir = output_root / motion_id
        smpl_path = motion_dir / "smpl" / f"{motion_id}.npz"
        smplx_path = motion_dir / "smplx" / f"{motion_id}.npz"

        if args.overwrite or not smpl_path.is_file():
            save_smpl_from_aist_raw(raw_pkl, smpl_path, raw_fps=args.raw_fps)
            smpl_written += 1
        if args.overwrite or not smplx_path.is_file():
            convert_smpl_to_smplx(smpl_path, smplx_path)
            smplx_written += 1

        flat_link = smplx_flat_dir / f"{motion_id}.npz"
        if flat_link.exists() or flat_link.is_symlink():
            flat_link.unlink()
        flat_link.symlink_to(smplx_path.resolve())

    print(f"[SMPL] wrote={smpl_written}", flush=True)
    print(f"[SMPLX] wrote={smplx_written}", flush=True)

    # Stage 2: GMR (forced sequential: num_cpus=1).
    gmr_cmd = [
        "conda",
        "run",
        "-n",
        args.gmr_env,
        "python",
        str(gmr_script),
        "--robot",
        "unitree_g1",
        "--src_folder",
        str(smplx_flat_dir),
        "--tgt_folder",
        str(gmr_flat_dir),
        "--num_cpus",
        "1",
    ]
    if args.overwrite:
        gmr_cmd.append("--override")
    run_cmd(gmr_cmd)

    # Stage 3: gmr pkl -> per-motion gmr/csv, and build pending csv list for npz+video export.
    csv_written = 0
    pending_motion_ids: list[str] = []
    for motion_id in motion_ids:
        motion_dir = output_root / motion_id
        gmr_src = gmr_flat_dir / f"{motion_id}.pkl"
        gmr_dst = motion_dir / "gmr" / f"{motion_id}.pkl"
        csv_path = motion_dir / "csv" / f"{motion_id}_qpos.csv"
        npz_path = motion_dir / "npz" / f"{motion_id}.npz"
        mp4_path = motion_dir / "npz" / f"{motion_id}.mp4"

        if not gmr_src.is_file():
            print(f"[WARN] missing gmr pkl: {gmr_src}", flush=True)
            continue

        gmr_dst.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not gmr_dst.is_file():
            shutil.copy2(gmr_src, gmr_dst)

        if args.overwrite or not csv_path.is_file():
            convert_gmr_pkl_to_csv(gmr_dst, csv_path)
            csv_written += 1

        if args.overwrite or (not npz_path.is_file()) or (not mp4_path.is_file()):
            pending_motion_ids.append(motion_id)
            link_csv = csv_pending_dir / f"{motion_id}.csv"
            if link_csv.exists() or link_csv.is_symlink():
                link_csv.unlink()
            link_csv.symlink_to(csv_path.resolve())

    print(f"[CSV] wrote={csv_written}", flush=True)
    print(f"[NPZ+MP4] pending={len(pending_motion_ids)}", flush=True)

    # Stage 4: csv -> npz+mp4 (single Isaac run, sequential over csv files).
    if pending_motion_ids:
        env = os.environ.copy()
        repo_source = "/home/hiyio/whole_body_tracking/source/whole_body_tracking"
        env["PYTHONPATH"] = f"{repo_source}:{env.get('PYTHONPATH', '')}"
        npz_cmd = [
            "conda",
            "run",
            "-n",
            args.isaac_env,
            "--no-capture-output",
            "python",
            str(csv_to_npz_script),
            "--input_file",
            str(csv_pending_dir),
            "--input_fps",
            str(args.input_fps),
            "--output_fps",
            str(args.output_fps),
            "--output_dir",
            str(npz_video_flat_dir),
            "--record",
            "--yup_to_zup",
            "--target_min_z",
            str(args.target_min_z),
        ]
        if args.frame_range is not None:
            npz_cmd += ["--frame_range", str(args.frame_range[0]), str(args.frame_range[1])]
        run_cmd(npz_cmd, env=env)

        moved = 0
        for motion_id in pending_motion_ids:
            src_npz = npz_video_flat_dir / f"{motion_id}.npz"
            src_mp4 = npz_video_flat_dir / f"{motion_id}.mp4"
            dst_dir = output_root / motion_id / "npz"
            dst_npz = dst_dir / f"{motion_id}.npz"
            dst_mp4 = dst_dir / f"{motion_id}.mp4"
            dst_dir.mkdir(parents=True, exist_ok=True)

            if src_npz.is_file():
                if dst_npz.exists():
                    dst_npz.unlink()
                shutil.move(str(src_npz), str(dst_npz))
            else:
                print(f"[WARN] missing generated npz: {src_npz}", flush=True)

            if src_mp4.is_file():
                if dst_mp4.exists():
                    dst_mp4.unlink()
                shutil.move(str(src_mp4), str(dst_mp4))
            else:
                print(f"[WARN] missing generated mp4: {src_mp4}", flush=True)

            if dst_npz.is_file() and dst_mp4.is_file():
                moved += 1
        print(f"[NPZ+MP4] finalized={moved}", flush=True)

    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
