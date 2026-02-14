#!/usr/bin/env python3
"""Batch-fix AIST motion npz files from Y-up to Z-up.

This script applies the same fix used for single-file recovery:
1) Rotate world-frame motion tensors by +90 deg around X (Y-up -> Z-up).
2) Lift the motion so the lowest body point is at target min-z (default: 0.02).

Supported tensors (if present):
  - body_pos_w
  - body_quat_w (wxyz)
  - body_lin_vel_w
  - body_ang_vel_w

Default input search path:
  datasets/aistplusplus/converted/**/npz/*.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


# +90 deg around X
ROT_MAT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
# quaternion (wxyz) for +90 deg around X
ROT_QUAT_WXYZ = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float32)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion multiplication for (..., 4) arrays in wxyz order."""
    aw, ax, ay, az = np.split(a, 4, axis=-1)
    bw, bx, by, bz = np.split(b, 4, axis=-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.concatenate([w, x, y, z], axis=-1)


def find_motion_npz_files(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/npz/*.npz" if recursive else "*/npz/*.npz"
    return sorted(root.glob(pattern))


def convert_one(
    src: Path,
    dst: Path,
    target_min_z: float,
    overwrite: bool,
    dry_run: bool,
) -> tuple[str, str]:
    if dst.exists() and not overwrite:
        return ("skip", f"{src} -> {dst} (exists)")

    with np.load(src, allow_pickle=False) as data:
        keys = set(data.files)
        required = {"body_pos_w", "body_quat_w"}
        if not required.issubset(keys):
            return ("skip", f"{src} (missing required keys: {sorted(required - keys)})")

        out = {k: data[k].copy() for k in data.files}

    # Rotate vector-valued world tensors
    for key in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        if key in out:
            arr = out[key].astype(np.float32, copy=False)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                return ("error", f"{src} ({key} shape invalid: {arr.shape})")
            out[key] = np.einsum("ij,tbj->tbi", ROT_MAT, arr)

    # Rotate orientation tensors
    quat = out["body_quat_w"].astype(np.float32, copy=False)
    if quat.ndim != 3 or quat.shape[-1] != 4:
        return ("error", f"{src} (body_quat_w shape invalid: {quat.shape})")
    qfix = np.broadcast_to(ROT_QUAT_WXYZ, quat.shape)
    quat = quat_mul_wxyz(qfix, quat)
    quat_norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    out["body_quat_w"] = quat / np.clip(quat_norm, 1e-8, None)

    # Lift whole motion above ground
    min_z_before = float(out["body_pos_w"][..., 2].min())
    dz = target_min_z - min_z_before
    out["body_pos_w"][..., 2] += dz
    min_z_after = float(out["body_pos_w"][..., 2].min())

    if dry_run:
        return ("ok", f"{src} -> {dst} (dry-run, dz={dz:.6f}, min_z_after={min_z_after:.6f})")

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    return ("ok", f"{src} -> {dst} (dz={dz:.6f}, min_z_after={min_z_after:.6f})")


def iter_sources(cli_paths: Iterable[str], root: Path, recursive: bool) -> list[Path]:
    if cli_paths:
        return sorted(Path(p).expanduser().resolve() for p in cli_paths)
    return [p.resolve() for p in find_motion_npz_files(root, recursive)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-fix AIST motion npz from Y-up to Z-up.")
    parser.add_argument(
        "--input_root",
        type=str,
        default="datasets/aistplusplus/converted",
        help="Root directory to search in (used when --inputs is not provided).",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Explicit input npz file list. If set, --input_root scan is ignored.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_zup",
        help="Output suffix appended before '.npz'.",
    )
    parser.add_argument(
        "--target_min_z",
        type=float,
        default=0.02,
        help="Shift motion so global minimum body z equals this value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan input_root (default: true).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not write files; only print what would be done.",
    )
    args = parser.parse_args()

    root = Path(args.input_root).expanduser().resolve()
    sources = iter_sources(args.inputs, root, args.recursive)
    if not sources:
        print("[INFO] No input files found.")
        return

    ok = 0
    skipped = 0
    errors = 0

    for src in sources:
        if not src.name.endswith(".npz"):
            skipped += 1
            print(f"[SKIP] {src} (not .npz)")
            continue
        if src.stem.endswith(args.suffix):
            skipped += 1
            print(f"[SKIP] {src} (already suffixed)")
            continue

        dst = src.with_name(f"{src.stem}{args.suffix}.npz")
        status, msg = convert_one(
            src=src,
            dst=dst,
            target_min_z=args.target_min_z,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if status == "ok":
            ok += 1
            print(f"[OK] {msg}")
        elif status == "skip":
            skipped += 1
            print(f"[SKIP] {msg}")
        else:
            errors += 1
            print(f"[ERROR] {msg}")

    print(f"\n[SUMMARY] ok={ok}, skipped={skipped}, errors={errors}, total={len(sources)}")


if __name__ == "__main__":
    main()
