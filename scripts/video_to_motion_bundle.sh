#!/usr/bin/env bash
set -euo pipefail

# Convert a video into a motion NPZ bundle under:
#   /home/hiyio/whole_body_tracking/motions/<timestamp>-<video_name>/
#
# Usage:
#   bash /home/hiyio/whole_body_tracking/scripts/video_to_motion_bundle.sh \
#     /home/hiyio/whole_body_tracking/Video/webster.mp4
#
# Optional:
#   VIDEO_NAME and TIMESTAMP can be overridden:
#     VIDEO_NAME=custom_name TIMESTAMP=20260212-101010 \
#     bash .../video_to_motion_bundle.sh /path/to/video.mp4
#
#   Robot can be selected with ROBOT_KEY (default: g1):
#     ROBOT_KEY=t1 bash .../video_to_motion_bundle.sh /path/to/video.mp4

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/video.mp4"
  exit 1
fi

VIDEO="$1"
if [[ ! -f "$VIDEO" ]]; then
  echo "Video not found: $VIDEO"
  exit 1
fi

VIDEO_BASENAME="$(basename "$VIDEO")"
VIDEO_NAME="${VIDEO_NAME:-${VIDEO_BASENAME%.*}}"
TIMESTAMP="${TIMESTAMP:-$(date +"%Y%m%d-%H%M%S")}"
ROBOT_KEY="${ROBOT_KEY:-g1}"
GMR_ROOT="${GMR_ROOT:-/home/hiyio/HoloMotion/thirdparties/GMR}"

case "$ROBOT_KEY" in
  g1)
    GMR_ROBOT="unitree_g1"
    NPZ_ROBOT="g1"
    ;;
  t1)
    # Use booster_t1 (23-DoF) so it stays consistent with T1_serial.urdf.
    GMR_ROBOT="booster_t1"
    NPZ_ROBOT="t1"
    ;;
  *)
    echo "Unsupported ROBOT_KEY: $ROBOT_KEY (expected: g1 or t1)"
    exit 1
    ;;
esac

RUN_DIR="/home/hiyio/whole_body_tracking/motions/${TIMESTAMP}-${VIDEO_NAME}"
GVHMR_OUT="${RUN_DIR}/gvhmr"
SMPL_DIR="${GVHMR_OUT}/${VIDEO_NAME}"
SMPL_FILE="${SMPL_DIR}/smpl.npz"
SMPLX_FILE="${SMPL_DIR}/smplx.npz"

SMPLX_ONLY="${RUN_DIR}/smplx_only"
GMR_OUT="${RUN_DIR}/gmr"
CSV_OUT="${RUN_DIR}/csv"
NPZ_OUT="${RUN_DIR}/npz"

mkdir -p "$GVHMR_OUT" "$SMPLX_ONLY" "$GMR_OUT" "$CSV_OUT" "$NPZ_OUT"

echo "[1/5] GVHMR: video -> SMPL"
GVHMR_DEVICE=cpu conda run -n gvhmr python \
  /home/hiyio/HoloMotion/holomotion/src/data_curation/video_to_smpl_gvhmr.py \
  --video "$VIDEO" \
  --output_root "$GVHMR_OUT" \
  -s --no_render

if [[ ! -f "$SMPL_FILE" ]]; then
  echo "SMPL output not found: $SMPL_FILE"
  exit 1
fi

echo "[2/5] SMPL -> SMPLX"
python "$GMR_ROOT/scripts/smpl_to_smplx.py" \
  --input_file "$SMPL_FILE" \
  --output_file "$SMPLX_FILE" \
  --gender neutral

if [[ ! -f "$SMPLX_FILE" ]]; then
  echo "SMPLX output not found: $SMPLX_FILE"
  exit 1
fi

echo "[3/5] SMPLX -> GMR PKL (robot=${GMR_ROBOT})"
ln -sf "$SMPLX_FILE" "$SMPLX_ONLY/${VIDEO_NAME}.npz"
conda run -n gmr python "$GMR_ROOT/scripts/smplx_to_robot_dataset.py" \
  --robot "$GMR_ROBOT" \
  --src_folder "$SMPLX_ONLY" \
  --tgt_folder "$GMR_OUT" \
  --num_cpus 1 --override

PKL_FILE="${GMR_OUT}/${VIDEO_NAME}.pkl"
if [[ ! -f "$PKL_FILE" ]]; then
  echo "GMR output not found: $PKL_FILE"
  exit 1
fi

echo "[4/5] PKL -> CSV"
python /home/hiyio/whole_body_tracking/scripts/smplx_pkl_to_csv.py \
  --input_pkl "$PKL_FILE" \
  --output_dir "$CSV_OUT" \
  --output_name "$VIDEO_NAME"

CSV_FILE="${CSV_OUT}/${VIDEO_NAME}_qpos.csv"
if [[ ! -f "$CSV_FILE" ]]; then
  echo "CSV output not found: $CSV_FILE"
  exit 1
fi

echo "[5/5] CSV -> NPZ"
conda run -n env_isaaclab python /home/hiyio/whole_body_tracking/scripts/csv_to_npz_local.py \
  --input_file "$CSV_FILE" \
  --robot "$NPZ_ROBOT" \
  --input_fps 30 --output_fps 50 \
  --output_dir "$NPZ_OUT" \
  --output_name "$VIDEO_NAME" \
  --headless

NPZ_FILE="${NPZ_OUT}/${VIDEO_NAME}.npz"
if [[ ! -f "$NPZ_FILE" ]]; then
  echo "NPZ output not found: $NPZ_FILE"
  exit 1
fi

# Convenience link at bundle root
ln -sf "$NPZ_FILE" "${RUN_DIR}/${VIDEO_NAME}.npz"

echo "[DONE] Bundle created: ${RUN_DIR}"
echo "NPZ: ${NPZ_FILE}"
