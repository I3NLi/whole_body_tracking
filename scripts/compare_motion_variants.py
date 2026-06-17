#!/usr/bin/env python3
"""Generate side-by-side comparison variants from a retargetable SMPL-X motion.

Typical usage:
  python scripts/compare_motion_variants.py \
    --bundle_dir /home/hiyio/whole_body_tracking/motions/20260413-113843-magicbot \
    --robot magicbot_z1
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GMR_SCRIPT = Path("/home/hiyio/HoloMotion/thirdparties/GMR/scripts/smplx_to_robot_dataset.py")
RENDER_GMR_SCRIPT = REPO_ROOT / "scripts" / "render_gmr_pkl_video.py"
PKL_TO_CSV_SCRIPT = REPO_ROOT / "scripts" / "smplx_pkl_to_csv.py"
CSV_TO_NPZ_SCRIPT = REPO_ROOT / "scripts" / "csv_to_npz_local.py"

DEFAULT_VARIANTS = ("base", "support", "support_ankle", "support_ankle_spike")
KNOWN_VARIANTS = set(DEFAULT_VARIANTS) | {"default"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison variants for a motion bundle.")
    parser.add_argument(
        "--bundle_dir",
        type=Path,
        default=None,
        help="Existing bundle directory. If set, the script auto-discovers gvhmr/*/smplx.npz.",
    )
    parser.add_argument(
        "--smplx_file",
        type=Path,
        default=None,
        help="Direct path to an input smplx.npz. Overrides --bundle_dir auto-discovery.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Where to write the comparison outputs. Default: <bundle>/comparisons.",
    )
    parser.add_argument(
        "--robot",
        choices=("unitree_g1", "magicbot_z1"),
        default="unitree_g1",
        help="Target robot for GMR and CSV->NPZ replay.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help=f"Variant profiles to run. Known: {sorted(KNOWN_VARIANTS)}",
    )
    parser.add_argument("--input_fps", type=int, default=30, help="Input motion FPS for CSV->NPZ.")
    parser.add_argument("--output_fps", type=int, default=50, help="Output NPZ FPS.")
    parser.add_argument("--num_cpus", type=int, default=1, help="CPU workers for GMR.")
    parser.add_argument(
        "--gmr_device",
        type=str,
        default=os.environ.get("GMR_DEVICE", "cpu"),
        help="Device passed to GMR via env, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--holomotion_env",
        type=str,
        default=os.environ.get("HOLOMOTION_ENV", "holomotion_train"),
        help="Conda env for GMR scripts.",
    )
    parser.add_argument(
        "--sim_env",
        type=str,
        default=os.environ.get("SIM_ENV", "BeyondMimic"),
        help="Conda env for csv_to_npz_local.py.",
    )
    parser.add_argument(
        "--preview_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render a GMR preview mp4 for each variant.",
    )
    parser.add_argument("--preview_width", type=int, default=640, help="Preview mp4 width.")
    parser.add_argument("--preview_height", type=int, default=360, help="Preview mp4 height.")
    parser.add_argument(
        "--export_npz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert each variant from pkl -> csv -> npz.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete each variant output directory before regenerating it.",
    )
    return parser.parse_args()


def discover_smplx_file(bundle_dir: Path) -> Path:
    matches = sorted(bundle_dir.glob("gvhmr/*/smplx.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one gvhmr/*/smplx.npz under {bundle_dir}, found {len(matches)}: {matches}"
        )
    return matches[0].resolve()


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict pickle at {path}, got {type(data)}")
    return data


def load_npz_shapes(path: Path) -> dict[str, list[int] | None]:
    if not path.is_file():
        return {}
    with np.load(path) as data:
        return {
            "joint_pos": list(data["joint_pos"].shape) if "joint_pos" in data else None,
            "body_pos_w": list(data["body_pos_w"].shape) if "body_pos_w" in data else None,
            "fps": data["fps"].tolist() if "fps" in data else None,
        }


def collect_summary(variant: str, artifact_root: Path, pkl_path: Path, npz_path: Path | None) -> dict:
    motion_data = load_pickle(pkl_path)
    stabilization = motion_data.get("retarget_stabilization", {})

    summary = {
        "variant": variant,
        "artifacts": {
            "root": str(artifact_root),
            "pkl": str(pkl_path),
            "preview_mp4": str(pkl_path.with_suffix(".mp4")),
            "csv": str(artifact_root / "csv" / f"{pkl_path.stem}_qpos.csv"),
            "npz": str(npz_path) if npz_path is not None else None,
        },
        "frames": {
            "pkl_root_pos": int(np.asarray(motion_data["root_pos"]).shape[0]),
            "pkl_dof_dim": int(np.asarray(motion_data["dof_pos"]).shape[1]),
        },
        "retarget_stabilization": stabilization,
    }
    if npz_path is not None and npz_path.is_file():
        summary["npz"] = load_npz_shapes(npz_path)
    return summary


def write_markdown_summary(path: Path, summaries: list[dict]) -> None:
    header = (
        "| variant | bad frames | interp | smoothed | support mean after | flatness mean final | pkl frames | dof |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    rows = []
    for summary in summaries:
        stats = summary["retarget_stabilization"]
        rows.append(
            "| {variant} | {final_bad_frames} | {interpolated_frames} | {smoothed_frames} | "
            "{support_excess_mean_after} | {foot_flatness_mean_final} | {pkl_root_pos} | {pkl_dof_dim} |".format(
                variant=summary["variant"],
                final_bad_frames=stats.get("final_bad_frames"),
                interpolated_frames=stats.get("interpolated_frames"),
                smoothed_frames=stats.get("smoothed_frames"),
                support_excess_mean_after=_fmt_float(stats.get("support_excess_mean_after")),
                foot_flatness_mean_final=_fmt_float(stats.get("foot_flatness_mean_final")),
                pkl_root_pos=summary["frames"]["pkl_root_pos"],
                pkl_dof_dim=summary["frames"]["pkl_dof_dim"],
            )
        )
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def _fmt_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def main() -> None:
    args = parse_args()

    if args.smplx_file is None and args.bundle_dir is None:
        raise ValueError("Provide either --smplx_file or --bundle_dir.")

    variants = []
    for variant in args.variants:
        normalized = variant.strip().lower()
        if normalized not in KNOWN_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Known variants: {sorted(KNOWN_VARIANTS)}")
        variants.append(normalized)

    smplx_file = args.smplx_file.resolve() if args.smplx_file else discover_smplx_file(args.bundle_dir.resolve())
    if not smplx_file.is_file():
        raise FileNotFoundError(f"SMPL-X file not found: {smplx_file}")

    if args.output_root is not None:
        output_root = args.output_root.resolve()
    elif args.bundle_dir is not None:
        output_root = args.bundle_dir.resolve() / "comparisons"
    else:
        output_root = smplx_file.parent / "comparisons"
    output_root.mkdir(parents=True, exist_ok=True)

    motion_name = smplx_file.parent.name
    all_summaries: list[dict] = []

    for variant in variants:
        variant_root = output_root / variant
        if args.force and variant_root.exists():
            shutil.rmtree(variant_root)
        (variant_root / "gmr").mkdir(parents=True, exist_ok=True)
        (variant_root / "csv").mkdir(parents=True, exist_ok=True)
        if args.export_npz:
            (variant_root / "npz").mkdir(parents=True, exist_ok=True)

        output_name = f"{motion_name}_{variant}"
        src_dir = variant_root / "_src"
        src_link = src_dir / f"{output_name}.npz"
        ensure_symlink(smplx_file, src_link)

        gmr_env = os.environ.copy()
        gmr_env["PYTHONNOUSERSITE"] = "1"
        gmr_env["GMR_DEVICE"] = args.gmr_device
        gmr_env["GMR_STABILIZER_PROFILE"] = variant

        pkl_path = variant_root / "gmr" / f"{output_name}.pkl"
        run_command(
            [
                "conda",
                "run",
                "-n",
                args.holomotion_env,
                "python",
                str(GMR_SCRIPT),
                "--robot",
                args.robot,
                "--src_folder",
                str(src_dir),
                "--tgt_folder",
                str(variant_root / "gmr"),
                "--num_cpus",
                str(args.num_cpus),
                "--override",
            ],
            env=gmr_env,
        )
        if not pkl_path.is_file():
            raise FileNotFoundError(f"Expected GMR output not found: {pkl_path}")

        if args.preview_video:
            run_command(
                [
                    "conda",
                    "run",
                    "-n",
                    args.holomotion_env,
                    "python",
                    str(RENDER_GMR_SCRIPT),
                    "--robot",
                    args.robot,
                    "--robot_motion_path",
                    str(pkl_path),
                    "--video_path",
                    str(pkl_path.with_suffix(".mp4")),
                    "--video_width",
                    str(args.preview_width),
                    "--video_height",
                    str(args.preview_height),
                ],
                env={"PYTHONNOUSERSITE": "1", **os.environ},
            )

        run_command(
            [
                "python",
                str(PKL_TO_CSV_SCRIPT),
                "--input_pkl",
                str(pkl_path),
                "--output_dir",
                str(variant_root / "csv"),
                "--output_name",
                output_name,
            ]
        )

        npz_path: Path | None = None
        if args.export_npz:
            npz_path = variant_root / "npz" / f"{output_name}.npz"
            sim_env = os.environ.copy()
            sim_env["PYTHONNOUSERSITE"] = "1"
            run_command(
                [
                    "conda",
                    "run",
                    "-n",
                    args.sim_env,
                    "python",
                    str(CSV_TO_NPZ_SCRIPT),
                    "--robot",
                    args.robot,
                    "--input_file",
                    str(variant_root / "csv" / f"{output_name}_qpos.csv"),
                    "--input_fps",
                    str(args.input_fps),
                    "--output_fps",
                    str(args.output_fps),
                    "--output_dir",
                    str(variant_root / "npz"),
                    "--output_name",
                    output_name,
                    "--headless",
                ],
                env=sim_env,
            )
            if not npz_path.is_file():
                raise FileNotFoundError(f"Expected NPZ output not found: {npz_path}")

        summary = collect_summary(variant, variant_root, pkl_path, npz_path)
        summary_path = variant_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        all_summaries.append(summary)

    aggregate = {
        "source_smplx": str(smplx_file),
        "robot": args.robot,
        "variants": variants,
        "summaries": all_summaries,
    }
    (output_root / "comparison_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(output_root / "comparison_summary.md", all_summaries)
    print(f"[OK] Comparison outputs written to: {output_root}")


if __name__ == "__main__":
    main()
