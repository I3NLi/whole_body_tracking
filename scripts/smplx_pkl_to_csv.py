#!/usr/bin/env python3
"""
Convert a GMR/HoloMotion SMPL-X pkl motion to CSV for csv_to_npz_local.py.

Example:
  python /home/hiyio/whole_body_tracking/scripts/smplx_pkl_to_csv.py \\
    --input_pkl /home/hiyio/HoloMotion/data/gmr_retargeted/GVHMR_telegram_0039/smplx.pkl

Default output:
  <basename>_qpos.csv
Columns (no header by default):
  root_pos(3) + root_rot_xyzw(4) + dof_pos(29)
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def _load_pkl(path: Path) -> dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in pkl, got {type(data)}")
    return data


def _ensure_2d(name: str, arr: np.ndarray, dim2: int | None = None) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if dim2 is not None and arr.shape[1] != dim2:
        raise ValueError(f"{name} must have {dim2} columns, got {arr.shape[1]}")
    return arr


def _write_csv(path: Path, data: np.ndarray, header: str | None) -> None:
    if header:
        np.savetxt(path, data, delimiter=",", header=header, comments="")
    else:
        np.savetxt(path, data, delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SMPL-X pkl motion to CSV.")
    parser.add_argument("--input_pkl", required=True, help="Path to smplx.pkl")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: same directory as input.",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help="Base name for output files (without extension). Default: input stem.",
    )
    parser.add_argument(
        "--root_rot_format",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order in pkl root_rot. Default: xyzw.",
    )
    parser.add_argument(
        "--with_header",
        action="store_true",
        help="Write a header row to CSV.",
    )
    parser.add_argument(
        "--write_local_body_pos",
        action="store_true",
        help="Also write local_body_pos CSV (flattened).",
    )
    args = parser.parse_args()

    input_pkl = Path(args.input_pkl).expanduser().resolve()
    if not input_pkl.is_file():
        raise FileNotFoundError(f"Input pkl not found: {input_pkl}")

    data = _load_pkl(input_pkl)
    for k in ("root_pos", "root_rot", "dof_pos"):
        if k not in data:
            raise KeyError(f"Missing key '{k}' in pkl. Available keys: {list(data.keys())}")

    root_pos = _ensure_2d("root_pos", np.asarray(data["root_pos"], dtype=np.float32), dim2=3)
    root_rot = _ensure_2d("root_rot", np.asarray(data["root_rot"], dtype=np.float32), dim2=4)
    dof_pos = _ensure_2d("dof_pos", np.asarray(data["dof_pos"], dtype=np.float32))

    if args.root_rot_format == "wxyz":
        root_rot = root_rot[:, [1, 2, 3, 0]]  # -> xyzw

    if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
        raise ValueError(
            f"Frame mismatch: root_pos={root_pos.shape[0]}, root_rot={root_rot.shape[0]}, "
            f"dof_pos={dof_pos.shape[0]}"
        )

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_pkl.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base = args.output_name if args.output_name else input_pkl.stem

    qpos = np.concatenate([root_pos, root_rot, dof_pos], axis=1)

    header = None
    if args.with_header:
        dof_cols = [f"dof_{i:02d}" for i in range(dof_pos.shape[1])]
        header = ",".join(
            ["root_pos_x", "root_pos_y", "root_pos_z", "root_rot_x", "root_rot_y", "root_rot_z", "root_rot_w"]
            + dof_cols
        )

    qpos_path = out_dir / f"{base}_qpos.csv"
    _write_csv(qpos_path, qpos, header)
    print(f"[OK] Wrote qpos CSV: {qpos_path}")

    if args.write_local_body_pos:
        if "local_body_pos" not in data:
            raise KeyError("Missing key 'local_body_pos' in pkl for --write_local_body_pos.")
        local_body_pos = np.asarray(data["local_body_pos"], dtype=np.float32)
        if local_body_pos.ndim != 3 or local_body_pos.shape[2] != 3:
            raise ValueError(f"local_body_pos must be [T, B, 3], got {local_body_pos.shape}")
        local_flat = local_body_pos.reshape(local_body_pos.shape[0], -1)

        local_header = None
        if args.with_header:
            names = data.get("link_body_list", [])
            cols = []
            if isinstance(names, (list, tuple)) and len(names) == local_body_pos.shape[1]:
                for name in names:
                    cols.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
            else:
                for i in range(local_body_pos.shape[1]):
                    cols.extend([f"body_{i:02d}_x", f"body_{i:02d}_y", f"body_{i:02d}_z"])
            local_header = ",".join(cols)

        local_path = out_dir / f"{base}_local_body_pos.csv"
        _write_csv(local_path, local_flat, local_header)
        print(f"[OK] Wrote local_body_pos CSV: {local_path}")


if __name__ == "__main__":
    main()
