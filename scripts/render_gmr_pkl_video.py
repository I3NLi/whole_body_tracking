#!/usr/bin/env python3
"""Render a saved GMR robot motion pickle to an mp4 once and exit."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm


GMR_ROOT = Path("/home/hiyio/HoloMotion/thirdparties/GMR")
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import RobotMotionViewer, load_robot_motion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a GMR robot motion pickle to mp4.")
    parser.add_argument("--robot", type=str, required=True, help="Robot name understood by GMR.")
    parser.add_argument("--robot_motion_path", type=str, required=True, help="Path to the GMR pickle motion.")
    parser.add_argument("--video_path", type=str, required=True, help="Output mp4 path.")
    parser.add_argument("--video_width", type=int, default=960, help="Output video width.")
    parser.add_argument("--video_height", type=int, default=540, help="Output video height.")
    parser.add_argument(
        "--transparent_robot",
        type=int,
        default=0,
        help="Whether to enable MuJoCo transparent robot rendering (0 or 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    motion_path = Path(args.robot_motion_path).expanduser().resolve()
    video_path = Path(args.video_path).expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"Robot motion file not found: {motion_path}")

    video_path.parent.mkdir(parents=True, exist_ok=True)

    (
        _motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        _motion_local_body_pos,
        _motion_link_body_list,
    ) = load_robot_motion(str(motion_path))

    fps = int(round(float(motion_fps))) if float(motion_fps) > 0 else 30
    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=fps,
        camera_follow=True,
        transparent_robot=args.transparent_robot,
        record_video=True,
        video_path=str(video_path),
        video_width=args.video_width,
        video_height=args.video_height,
    )

    try:
        for frame_idx in tqdm(range(len(motion_root_pos)), desc="Render GMR Robot"):
            viewer.step(
                motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=False,
                follow_camera=True,
            )
    finally:
        viewer.close()

    print(f"[OK] Saved GMR robot video: {video_path}")


if __name__ == "__main__":
    main()
