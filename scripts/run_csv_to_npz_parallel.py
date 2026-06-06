#!/usr/bin/env python3
"""Run csv_to_npz_local.py over a folder of CSV files with bounded parallelism."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_TO_NPZ_SCRIPT = REPO_ROOT / "scripts" / "csv_to_npz_local.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel csv_to_npz_local.py runner.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing *_qpos.csv files.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to write NPZ/MP4 files.")
    parser.add_argument("--logs_dir", type=Path, required=True, help="Directory for per-file logs.")
    parser.add_argument(
        "--progress_path",
        type=Path,
        default=None,
        help="Optional jsonl progress file. Default: <output_dir>/csv_to_npz_parallel_progress.jsonl",
    )
    parser.add_argument("--robot", type=str, default="magicbot_z1")
    parser.add_argument("--input_fps", type=int, default=30)
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=2, help="Max concurrent csv_to_npz workers.")
    parser.add_argument("--sim_env", type=str, default=os.environ.get("SIM_ENV", "BeyondMimic"))
    parser.add_argument("--overwrite", action="store_true", help="Rebuild outputs even if npz/mp4 already exist.")
    parser.add_argument("--record", action="store_true", help="Enable mp4 recording.")
    parser.add_argument("--render", action="store_true", help="Render simulation frames during replay.")
    parser.add_argument("--record_backend", type=str, default="viewport")
    return parser.parse_args()


def _write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _motion_base_name(csv_path: Path) -> str:
    stem = csv_path.stem
    return stem[:-5] if stem.endswith("_qpos") else stem


def _build_command(args: argparse.Namespace, csv_path: Path) -> list[str]:
    output_name = _motion_base_name(csv_path)
    cmd = [
        "conda",
        "run",
        "-n",
        args.sim_env,
        "python",
        str(CSV_TO_NPZ_SCRIPT),
        "--input_file",
        str(csv_path),
        "--robot",
        args.robot,
        "--input_fps",
        str(args.input_fps),
        "--output_fps",
        str(args.output_fps),
        "--output_dir",
        str(args.output_dir),
        "--output_name",
        output_name,
    ]
    if args.record:
        cmd.extend(["--record", "--record_backend", args.record_backend])
    if args.render:
        cmd.append("--render")
    return cmd


def _launch_job(args: argparse.Namespace, csv_path: Path, progress_path: Path) -> tuple[subprocess.Popen, object, dict]:
    motion_name = _motion_base_name(csv_path)
    log_path = args.logs_dir / f"{motion_name}.csv_to_npz.log"
    log_file = log_path.open("w", encoding="utf-8")
    cmd = _build_command(args, csv_path)
    _write_progress(
        progress_path,
        {
            "motion": motion_name,
            "status": "started",
            "csv_path": str(csv_path),
            "log_path": str(log_path),
            "cmd": cmd,
            "timestamp": time.time(),
        },
    )
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    meta = {
        "motion": motion_name,
        "csv_path": str(csv_path),
        "log_path": str(log_path),
        "started_at": time.time(),
        "output_name": motion_name,
    }
    return proc, log_file, meta


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.logs_dir = args.logs_dir.expanduser().resolve()
    progress_path = (
        args.progress_path.expanduser().resolve()
        if args.progress_path is not None
        else (args.output_dir / "csv_to_npz_parallel_progress.jsonl")
    )

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    # Viewport recording is a single-step csv->npz+mp4 workflow and must be serialized.
    # Running multiple Isaac Sim viewport captures concurrently on this machine exhausts GPU memory.
    if args.record and args.record_backend == "viewport" and args.jobs != 1:
        print(
            f"[CSV2NPZ] viewport recording requested; forcing jobs=1 (requested {args.jobs}).",
            file=sys.stderr,
        )
        args.jobs = 1

    csv_files = sorted(args.input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {args.input_dir}")

    pending = []
    for csv_path in csv_files:
        base = _motion_base_name(csv_path)
        npz_path = args.output_dir / f"{base}.npz"
        mp4_path = args.output_dir / f"{base}.mp4"
        output_ready = npz_path.is_file() and ((not args.record) or mp4_path.is_file())
        if (not args.overwrite) and output_ready:
            _write_progress(
                progress_path,
                {
                    "motion": base,
                    "status": "skipped",
                    "csv_path": str(csv_path),
                    "timestamp": time.time(),
                },
            )
            continue
        pending.append(csv_path)

    print(f"[CSV2NPZ] input_dir={args.input_dir}")
    print(f"[CSV2NPZ] output_dir={args.output_dir}")
    print(f"[CSV2NPZ] logs_dir={args.logs_dir}")
    print(f"[CSV2NPZ] jobs={args.jobs}")
    print(f"[CSV2NPZ] files={len(pending)}")
    print(f"[CSV2NPZ] DISPLAY={os.environ.get('DISPLAY', '')}")

    running: dict[int, tuple[subprocess.Popen, object, dict]] = {}
    completed = 0
    failed = 0
    queue = list(pending)

    while queue or running:
        while queue and len(running) < args.jobs:
            csv_path = queue.pop(0)
            proc, log_file, meta = _launch_job(args, csv_path, progress_path)
            running[proc.pid] = (proc, log_file, meta)
            print(f"[START] pid={proc.pid} motion={meta['motion']}")

        finished_pids = []
        for pid, (proc, log_file, meta) in running.items():
            returncode = proc.poll()
            if returncode is None:
                continue
            log_file.close()
            finished_pids.append(pid)
            payload = {
                "motion": meta["motion"],
                "csv_path": meta["csv_path"],
                "log_path": meta["log_path"],
                "returncode": int(returncode),
                "seconds": round(time.time() - meta["started_at"], 3),
                "timestamp": time.time(),
            }
            if returncode == 0:
                completed += 1
                payload["status"] = "ok"
                print(f"[OK] pid={pid} motion={meta['motion']} completed={completed}")
            else:
                failed += 1
                payload["status"] = "error"
                print(f"[ERR] pid={pid} motion={meta['motion']} returncode={returncode}", file=sys.stderr)
            _write_progress(progress_path, payload)

        for pid in finished_pids:
            running.pop(pid, None)

        if running:
            time.sleep(5)

    print(f"[DONE] completed={completed} failed={failed} requested={len(pending)}")


if __name__ == "__main__":
    main()
