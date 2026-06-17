#!/usr/bin/env python3

import argparse
import contextlib
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


GVHMR_ROOT = Path("/home/hiyio/HoloMotion/thirdparties/GVHMR")
HOLOMOTION_SRC = Path("/home/hiyio/HoloMotion/holomotion/src")
GVHMR_ENTRY = HOLOMOTION_SRC / "data_curation" / "video_to_smpl_gvhmr.py"

COCO17_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

POSE_COLORS = [
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 200, 255),
    (0, 200, 255),
    (0, 200, 255),
    (0, 200, 255),
    (0, 200, 255),
    (0, 200, 255),
]


def patch_numpy_aliases() -> None:
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        setattr(np, name, value)


def load_gvhmr_module():
    sys.path.insert(0, str(HOLOMOTION_SRC))
    sys.path.insert(0, str(GVHMR_ROOT))
    spec = importlib.util.spec_from_file_location("video_to_smpl_gvhmr", GVHMR_ENTRY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load GVHMR entry: {GVHMR_ENTRY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def should_skip(path: Path, force: bool) -> bool:
    return path.exists() and path.stat().st_size > 0 and not force


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def open_video_writer(path: Path, fps: float, width: int, height: int):
    ensure_parent(path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


class Cv2FrameWriter:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = fps
        self.writer = None

    def write_frame(self, frame) -> None:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 frame, got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if self.writer is None:
            height, width = frame.shape[:2]
            self.writer = open_video_writer(self.path, self.fps, width, height)
        self.writer.write(frame[:, :, ::-1])

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


def draw_bbox_overlay(video_path: Path, bbx_path: Path, out_path: Path, force: bool) -> None:
    if should_skip(out_path, force):
        print(f"[SKIP] bbox overlay exists: {out_path}")
        return

    bbx_xyxy = torch.load(bbx_path)["bbx_xyxy"].cpu().numpy()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_video_writer(out_path, fps, width, height)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= len(bbx_xyxy):
            break
        x1, y1, x2, y2 = np.round(bbx_xyxy[frame_idx]).astype(int)
        x1 = int(np.clip(x1, 0, width - 1))
        x2 = int(np.clip(x2, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        y2 = int(np.clip(y2, 0, height - 1))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 178, 255), 2)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[OK] bbox overlay: {out_path}")


def draw_pose_overlay(video_path: Path, pose_path: Path, out_path: Path, force: bool, score_thr: float) -> None:
    if should_skip(out_path, force):
        print(f"[SKIP] pose overlay exists: {out_path}")
        return

    keypoints = torch.load(pose_path).cpu().numpy()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_video_writer(out_path, fps, width, height)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= len(keypoints):
            break
        pose = keypoints[frame_idx]
        for i, j in COCO17_EDGES:
            if pose[i, 2] < score_thr or pose[j, 2] < score_thr:
                continue
            p1 = tuple(np.round(pose[i, :2]).astype(int))
            p2 = tuple(np.round(pose[j, :2]).astype(int))
            cv2.line(frame, p1, p2, (255, 255, 0), 2, lineType=cv2.LINE_AA)
        for idx, (x, y, score) in enumerate(pose):
            if score < score_thr:
                continue
            color = POSE_COLORS[idx]
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, color, -1, lineType=cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[OK] pose overlay: {out_path}")


def maybe_remove(path: Path, force: bool) -> None:
    if force and path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("-s", "--static_cam", action="store_true")
    parser.add_argument("--use_dpvo", action="store_true")
    parser.add_argument("--f_mm", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--score_thr", type=float, default=0.3)
    args = parser.parse_args()

    patch_numpy_aliases()
    module = load_gvhmr_module()

    gvhmr_args = argparse.Namespace(
        video=args.video,
        output_root=args.output_root,
        static_cam=args.static_cam,
        use_dpvo=args.use_dpvo,
        f_mm=args.f_mm,
        verbose=False,
    )

    with pushd(GVHMR_ROOT):
        cfg = module.parse_args_to_cfg(gvhmr_args)
        module.run_preprocess(cfg)

        draw_bbox_overlay(
            Path(cfg.video_path),
            Path(cfg.paths.bbx),
            Path(cfg.paths.bbx_xyxy_video_overlay),
            args.force,
        )
        draw_pose_overlay(
            Path(cfg.video_path),
            Path(cfg.paths.vitpose),
            Path(cfg.paths.vitpose_video_overlay),
            args.force,
            args.score_thr,
        )

        maybe_remove(Path(cfg.paths.incam_video), args.force)
        maybe_remove(Path(cfg.paths.global_video), args.force)
        maybe_remove(Path(cfg.paths.incam_global_horiz_video), args.force)

        module.get_writer = lambda video_path, fps=30, crf=17: Cv2FrameWriter(Path(video_path), fps)
        module.render_incam(cfg)
        module.render_global(cfg)
        if not Path(cfg.paths.incam_global_horiz_video).exists():
            module.merge_videos_horizontal(
                [cfg.paths.incam_video, cfg.paths.global_video],
                cfg.paths.incam_global_horiz_video,
            )
        print(f"[DONE] GVHMR stage videos: {cfg.output_dir}")


if __name__ == "__main__":
    main()
