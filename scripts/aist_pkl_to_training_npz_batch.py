#!/usr/bin/env python3
"""Batch convert AIST++ raw motion PKL files to training-ready NPZ files.

Pipeline:
1) AIST raw motion PKL -> SMPL NPZ
2) SMPL NPZ -> SMPL-X NPZ
3) SMPL-X NPZ -> GMR robot PKL
4) GMR robot PKL -> CSV(qpos)
5) CSV -> Isaac replay NPZ (raw world frame)
6) Raw NPZ Y-up -> Z-up NPZ (training-ready)
7) Optional symlink export directory

The script is resumable by default (skip existing outputs unless --overwrite).
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np


STAGES = ("smpl", "smplx", "gmr", "csv", "npz", "zup", "link")


def _stage_index(name: str) -> int:
    if name not in STAGES:
        raise ValueError(f"Invalid stage: {name}. Choices: {STAGES}")
    return STAGES.index(name)


def _should_run(stage: str, from_stage: str, to_stage: str) -> bool:
    s = _stage_index(stage)
    return _stage_index(from_stage) <= s <= _stage_index(to_stage)


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def motion_ids_from_dir(motions_dir: Path) -> list[str]:
    return sorted(p.stem for p in motions_dir.glob("*.pkl"))


def maybe_limit(ids: list[str], limit: int | None) -> list[str]:
    return ids if limit is None else ids[:limit]


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data)}")
    return data


def save_smpl_from_aist_raw(src_pkl: Path, dst_npz: Path, raw_fps: float, gender: str) -> None:
    data = load_pickle(src_pkl)
    required = ("smpl_poses", "smpl_trans", "smpl_scaling")
    for key in required:
        if key not in data:
            raise KeyError(f"{src_pkl} missing key '{key}'. keys={list(data.keys())}")

    poses = np.asarray(data["smpl_poses"], dtype=np.float32)
    trans_raw = np.asarray(data["smpl_trans"], dtype=np.float32)
    scaling = float(np.asarray(data["smpl_scaling"]).reshape(-1)[0])
    if poses.ndim != 2:
        raise ValueError(f"{src_pkl} smpl_poses must be 2D, got {poses.shape}")
    if poses.shape[1] > 72:
        poses = poses[:, :72]
    if poses.shape[1] != 72:
        raise ValueError(f"{src_pkl} smpl_poses must have 72 dims, got {poses.shape}")
    if trans_raw.ndim != 2 or trans_raw.shape[1] != 3:
        raise ValueError(f"{src_pkl} smpl_trans must be [T,3], got {trans_raw.shape}")

    # AIST raw trans uses xyz order with Y/Z swapped for this downstream toolchain.
    trans = trans_raw[:, [0, 2, 1]] / scaling

    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dst_npz,
        poses=poses,
        trans=trans.astype(np.float32),
        mocap_framerate=np.array(raw_fps, dtype=np.float64),
        betas=np.zeros((10,), dtype=np.float32),
        gender=np.array(gender),
    )


def convert_smpl_to_smplx(src_npz: Path, dst_npz: Path, gender: str) -> None:
    smpl_data = np.load(src_npz, allow_pickle=True)
    data_dict = dict(smpl_data)

    if "betas" in data_dict:
        betas = data_dict["betas"]
        if betas.shape == (10,):
            data_dict["betas"] = np.concatenate([betas, np.zeros(6, dtype=betas.dtype)])
        elif betas.shape not in ((16,), (1, 16)):
            raise ValueError(f"{src_npz}: unexpected betas shape: {betas.shape}")

    if "mocap_framerate" in data_dict:
        data_dict["mocap_frame_rate"] = data_dict.pop("mocap_framerate")

    if "poses" not in data_dict:
        raise ValueError(f"{src_npz}: missing 'poses' key")
    poses = data_dict["poses"]
    if poses.shape[1] > 72:
        poses = poses[:, :72]

    data_dict["root_orient"] = poses[:, :3]
    data_dict["pose_body"] = poses[:, 3:66]
    if "gender" not in data_dict:
        data_dict["gender"] = np.array(gender)
    del data_dict["poses"]

    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst_npz, **data_dict)


def convert_gmr_pkl_to_csv(src_pkl: Path, dst_csv: Path) -> None:
    data = load_pickle(src_pkl)
    for key in ("root_pos", "root_rot", "dof_pos"):
        if key not in data:
            raise KeyError(f"{src_pkl}: missing key '{key}'")

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{src_pkl}: root_pos shape invalid: {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"{src_pkl}: root_rot shape invalid: {root_rot.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"{src_pkl}: dof_pos shape invalid: {dof_pos.shape}")
    if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
        raise ValueError(
            f"{src_pkl}: frame mismatch root_pos={root_pos.shape[0]}, root_rot={root_rot.shape[0]}, dof_pos={dof_pos.shape[0]}"
        )

    qpos = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst_csv, qpos, delimiter=",")


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.split(a, 4, axis=-1)
    bw, bx, by, bz = np.split(b, 4, axis=-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.concatenate([w, x, y, z], axis=-1)


def convert_npz_yup_to_zup(src_npz: Path, dst_npz: Path, target_min_z: float) -> None:
    rot_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rot_quat = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float32)  # wxyz

    with np.load(src_npz, allow_pickle=False) as data:
        out = {k: data[k].copy() for k in data.files}

    required = {"body_pos_w", "body_quat_w"}
    missing = [k for k in required if k not in out]
    if missing:
        raise KeyError(f"{src_npz}: missing keys {missing}")

    for key in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        if key in out:
            arr = out[key].astype(np.float32, copy=False)
            out[key] = np.einsum("ij,tbj->tbi", rot_mat, arr)

    quat = out["body_quat_w"].astype(np.float32, copy=False)
    qfix = np.broadcast_to(rot_quat, quat.shape)
    quat = quat_mul_wxyz(qfix, quat)
    quat /= np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    out["body_quat_w"] = quat

    min_z = float(out["body_pos_w"][..., 2].min())
    out["body_pos_w"][..., 2] += target_min_z - min_z

    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst_npz, **out)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_selected_csv(csv_dir: Path, pending_dir: Path, ids: Iterable[str]) -> int:
    ensure_clean_dir(pending_dir)
    count = 0
    for motion_id in ids:
        src = csv_dir / f"{motion_id}_qpos.csv"
        if not src.is_file():
            continue
        dst = pending_dir / f"{motion_id}.csv"
        dst.symlink_to(src.resolve())
        count += 1
    return count


def symlink_selected_smplx(smplx_dir: Path, gmr_input_dir: Path, ids: Iterable[str]) -> int:
    ensure_clean_dir(gmr_input_dir)
    count = 0
    for motion_id in ids:
        src = smplx_dir / f"{motion_id}.npz"
        if not src.is_file():
            continue
        dst = gmr_input_dir / f"{motion_id}.npz"
        dst.symlink_to(src.resolve())
        count += 1
    return count


def write_manifest(npz_dir: Path, manifest_path: Path, ids: Iterable[str]) -> int:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    for motion_id in ids:
        p = npz_dir / f"{motion_id}.npz"
        if p.is_file():
            kept.append(str(p.resolve()))
    with manifest_path.open("w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert AIST raw PKL to training NPZ.")
    parser.add_argument(
        "--motions_dir",
        type=str,
        default="/home/hiyio/whole_body_tracking/datasets/aistplusplus/fullset/aist_plusplus_final/motions",
        help="Directory containing AIST raw motion *.pkl files.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/hiyio/whole_body_tracking/datasets/aistplusplus/converted_full",
        help="Output root directory for all intermediate/final artifacts.",
    )
    parser.add_argument(
        "--gmr_script",
        type=str,
        default="/home/hiyio/GMR/scripts/smplx_to_robot_dataset.py",
        help="Path to GMR SMPL-X->robot conversion script.",
    )
    parser.add_argument(
        "--csv_to_npz_script",
        type=str,
        default="/home/hiyio/whole_body_tracking/scripts/csv_to_npz_local.py",
        help="Path to CSV->NPZ conversion script.",
    )
    parser.add_argument("--gmr_env", type=str, default="gmr", help="Conda env used for GMR stage.")
    parser.add_argument("--isaac_env", type=str, default="env_isaaclab", help="Conda env used for CSV->NPZ stage.")
    parser.add_argument("--num_cpus", type=int, default=8, help="CPU workers for GMR script.")
    parser.add_argument("--raw_fps", type=float, default=60.0, help="FPS for AIST raw motions.")
    parser.add_argument("--input_fps", type=int, default=30, help="Input FPS for CSV->NPZ stage.")
    parser.add_argument("--output_fps", type=int, default=50, help="Output FPS for CSV->NPZ stage.")
    parser.add_argument("--target_min_z", type=float, default=0.02, help="Target minimum body height after Z-up fix.")
    parser.add_argument(
        "--from_stage",
        type=str,
        choices=STAGES,
        default="smpl",
        help="Start stage (inclusive).",
    )
    parser.add_argument(
        "--to_stage",
        type=str,
        choices=STAGES,
        default="link",
        help="End stage (inclusive).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N motions.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--link_dir",
        type=str,
        default="/home/hiyio/whole_body_tracking/motions/aist_full",
        help="Directory to create final NPZ symlinks. Empty string disables linking.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Manifest file for final NPZ absolute paths. Default: <output_root>/npz_manifest.txt",
    )
    args = parser.parse_args()

    if _stage_index(args.from_stage) > _stage_index(args.to_stage):
        raise ValueError(f"from_stage({args.from_stage}) must be <= to_stage({args.to_stage})")

    motions_dir = Path(args.motions_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    gmr_script = Path(args.gmr_script).expanduser().resolve()
    csv_to_npz_script = Path(args.csv_to_npz_script).expanduser().resolve()
    link_dir = Path(args.link_dir).expanduser().resolve() if args.link_dir else None
    manifest_path = (
        Path(args.manifest).expanduser().resolve() if args.manifest else (output_root / "npz_manifest.txt")
    )

    if not motions_dir.is_dir():
        raise FileNotFoundError(f"motions_dir not found: {motions_dir}")
    if not gmr_script.is_file():
        raise FileNotFoundError(f"gmr_script not found: {gmr_script}")
    if not csv_to_npz_script.is_file():
        raise FileNotFoundError(f"csv_to_npz_script not found: {csv_to_npz_script}")

    smpl_dir = output_root / "smpl"
    smplx_dir = output_root / "smplx"
    gmr_dir = output_root / "gmr"
    csv_dir = output_root / "csv"
    raw_npz_dir = output_root / "npz_raw"
    final_npz_dir = output_root / "npz"
    gmr_input_dir = output_root / "_gmr_input"
    pending_csv_dir = output_root / "_pending_csv"

    all_ids = motion_ids_from_dir(motions_dir)
    ids = maybe_limit(all_ids, args.limit)
    if not ids:
        print("[INFO] No motion files found.")
        return
    print(f"[INFO] motions total={len(all_ids)}, selected={len(ids)}")

    # Stage 1: raw pkl -> smpl npz
    if _should_run("smpl", args.from_stage, args.to_stage):
        done = 0
        for motion_id in ids:
            src = motions_dir / f"{motion_id}.pkl"
            dst = smpl_dir / f"{motion_id}.npz"
            if dst.is_file() and not args.overwrite:
                continue
            save_smpl_from_aist_raw(src, dst, raw_fps=args.raw_fps, gender="neutral")
            done += 1
        print(f"[SMPL] wrote={done}")

    # Stage 2: smpl npz -> smplx npz
    if _should_run("smplx", args.from_stage, args.to_stage):
        done = 0
        for motion_id in ids:
            src = smpl_dir / f"{motion_id}.npz"
            dst = smplx_dir / f"{motion_id}.npz"
            if not src.is_file():
                print(f"[SMPLX][WARN] missing source: {src}")
                continue
            if dst.is_file() and not args.overwrite:
                continue
            convert_smpl_to_smplx(src, dst, gender="neutral")
            done += 1
        print(f"[SMPLX] wrote={done}")

    # Stage 3: smplx npz -> gmr pkl
    if _should_run("gmr", args.from_stage, args.to_stage):
        linked = symlink_selected_smplx(smplx_dir, gmr_input_dir, ids)
        print(f"[GMR] linked inputs={linked}")
        cmd = [
            "conda",
            "run",
            "-n",
            args.gmr_env,
            "python",
            str(gmr_script),
            "--robot",
            "unitree_g1",
            "--src_folder",
            str(gmr_input_dir),
            "--tgt_folder",
            str(gmr_dir),
            "--num_cpus",
            str(args.num_cpus),
        ]
        if args.overwrite:
            cmd.append("--override")
        run_cmd(cmd)

    # Stage 4: gmr pkl -> csv
    if _should_run("csv", args.from_stage, args.to_stage):
        done = 0
        for motion_id in ids:
            src = gmr_dir / f"{motion_id}.pkl"
            dst = csv_dir / f"{motion_id}_qpos.csv"
            if not src.is_file():
                print(f"[CSV][WARN] missing source: {src}")
                continue
            if dst.is_file() and not args.overwrite:
                continue
            convert_gmr_pkl_to_csv(src, dst)
            done += 1
        print(f"[CSV] wrote={done}")

    # Stage 5: csv -> raw npz (Isaac replay)
    if _should_run("npz", args.from_stage, args.to_stage):
        pending_ids: list[str] = []
        for motion_id in ids:
            src_csv = csv_dir / f"{motion_id}_qpos.csv"
            dst_npz = raw_npz_dir / f"{motion_id}.npz"
            if not src_csv.is_file():
                continue
            if dst_npz.is_file() and not args.overwrite:
                continue
            pending_ids.append(motion_id)
        pending_count = symlink_selected_csv(csv_dir, pending_csv_dir, pending_ids)
        print(f"[NPZ] pending csv={pending_count}")
        if pending_count > 0:
            env = os.environ.copy()
            repo_source = "/home/hiyio/whole_body_tracking/source/whole_body_tracking"
            env["PYTHONPATH"] = f"{repo_source}:{env.get('PYTHONPATH', '')}"
            cmd = [
                "conda",
                "run",
                "-n",
                args.isaac_env,
                "--no-capture-output",
                "python",
                str(csv_to_npz_script),
                "--input_file",
                str(pending_csv_dir),
                "--input_fps",
                str(args.input_fps),
                "--output_fps",
                str(args.output_fps),
                "--output_dir",
                str(raw_npz_dir),
                "--headless",
            ]
            run_cmd(cmd, env=env)

    # Stage 6: raw npz -> final z-up npz
    if _should_run("zup", args.from_stage, args.to_stage):
        done = 0
        for motion_id in ids:
            src = raw_npz_dir / f"{motion_id}.npz"
            dst = final_npz_dir / f"{motion_id}.npz"
            if not src.is_file():
                print(f"[ZUP][WARN] missing source: {src}")
                continue
            if dst.is_file() and not args.overwrite:
                continue
            convert_npz_yup_to_zup(src, dst, target_min_z=args.target_min_z)
            done += 1
        print(f"[ZUP] wrote={done}")

    # Stage 7: optional symlink export + manifest
    if _should_run("link", args.from_stage, args.to_stage):
        if link_dir is not None:
            link_dir.mkdir(parents=True, exist_ok=True)
            link_count = 0
            for motion_id in ids:
                src = final_npz_dir / f"{motion_id}.npz"
                if not src.is_file():
                    continue
                dst = link_dir / f"{motion_id}.npz"
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
                link_count += 1
            print(f"[LINK] linked={link_count} -> {link_dir}")
        manifest_count = write_manifest(final_npz_dir, manifest_path, ids)
        print(f"[MANIFEST] wrote={manifest_count} -> {manifest_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()
