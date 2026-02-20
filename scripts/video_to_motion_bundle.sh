#!/usr/bin/env bash
set -euo pipefail

# Convert a video into a motion NPZ bundle under:
#   /home/hiyio/whole_body_tracking/motions/<timestamp>-<video_name>/
#
# Usage:
#   bash /home/hiyio/whole_body_tracking/scripts/video_to_motion_bundle.sh \
#     /home/hiyio/whole_body_tracking/Video/webster.mp4
#   bash /home/hiyio/whole_body_tracking/scripts/video_to_motion_bundle.sh \
#     /home/hiyio/whole_body_tracking/Video/webster.mp4 --headless
#
# Optional:
#   VIDEO_NAME and TIMESTAMP can be overridden:
#     VIDEO_NAME=custom_name TIMESTAMP=20260212-101010 \
#     bash .../video_to_motion_bundle.sh /path/to/video.mp4
#   MP4 is exported by default. To disable:
#     EXPORT_MP4=0 bash .../video_to_motion_bundle.sh /path/to/video.mp4

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/video.mp4 [--headless]"
  exit 1
fi

VIDEO="$1"
USER_HEADLESS=0
if [[ $# -ge 2 ]]; then
  case "$2" in
    --headless) USER_HEADLESS=1 ;;
    *)
      echo "Unknown option: $2"
      echo "Usage: $0 /path/to/video.mp4 [--headless]"
      exit 1
      ;;
  esac
fi

if [[ ! -f "$VIDEO" ]]; then
  echo "Video not found: $VIDEO"
  exit 1
fi

VIDEO_BASENAME="$(basename "$VIDEO")"
VIDEO_NAME="${VIDEO_NAME:-${VIDEO_BASENAME%.*}}"
TIMESTAMP="${TIMESTAMP:-$(date +"%Y%m%d-%H%M%S")}"

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
mkdir -p "$SMPL_DIR/preprocess"
mkdir -p /tmp/Ultralytics

# Ensure GVHMR relative checkpoint paths resolve when running from this repo.
CKPT_SRC="/home/hiyio/HoloMotion/thirdparties/GVHMR/inputs/checkpoints"
CKPT_DST="/home/hiyio/whole_body_tracking/inputs/checkpoints"
mkdir -p "$CKPT_DST"
for d in vitpose gvhmr hmr2 yolo dpvo body_models; do
  if [[ -e "$CKPT_SRC/$d" && ! -e "$CKPT_DST/$d" ]]; then
    ln -s "$CKPT_SRC/$d" "$CKPT_DST/$d"
  fi
done

echo "[1/5] GVHMR: video -> SMPL"
HOLOMOTION_ENV="${HOLOMOTION_ENV:-holomotion_train}"
GVHMR_ENV="${GVHMR_ENV:-BeyondMimic}"
SIM_ENV="${SIM_ENV:-BeyondMimic}"
GVHMR_DEVICE="${GVHMR_DEVICE:-cuda}"
RECORD_BACKEND="${RECORD_BACKEND:-viewport}"
EXPORT_MP4="${EXPORT_MP4:-1}"
if ! PYTHONPATH="/home/hiyio/HoloMotion/thirdparties/GVHMR:/home/hiyio/HoloMotion/holomotion/src:${PYTHONPATH:-}" \
  conda run -n "$GVHMR_ENV" python -c "import pytorch_lightning, ultralytics, colorlog, pytorch3d, hydra_zen, ffmpeg, wis3d, yacs, timm, pycolmap, smplx; from hmr4d.configs import register_store_gvhmr" >/dev/null 2>&1; then
  echo "[WARN] Env '$GVHMR_ENV' missing GVHMR deps, fallback to 'gvhmr' for step 1."
  GVHMR_ENV="gvhmr"
fi
run_gvhmr() {
  local dev="$1"
  echo "[GVHMR] device=$dev"
  GVHMR_DEVICE="$dev" \
  PYTHONPATH="/home/hiyio/HoloMotion/thirdparties/GVHMR:/home/hiyio/HoloMotion/holomotion/src:${PYTHONPATH:-}" \
  YOLO_CONFIG_DIR="/tmp/Ultralytics" \
  conda run -n "$GVHMR_ENV" python \
    /home/hiyio/HoloMotion/holomotion/src/data_curation/video_to_smpl_gvhmr.py \
    --video "$VIDEO" \
    --output_root "$GVHMR_OUT" \
    -s --no_render
}

if [[ "$GVHMR_DEVICE" == "cuda" ]]; then
  if ! run_gvhmr cuda; then
    echo "[GVHMR] cuda failed, fallback to cpu"
    run_gvhmr cpu
  fi
else
  run_gvhmr "$GVHMR_DEVICE"
fi

if [[ ! -f "$SMPL_FILE" ]]; then
  echo "SMPL output not found: $SMPL_FILE"
  exit 1
fi

echo "[2/5] SMPL -> SMPLX"
conda run -n "$HOLOMOTION_ENV" python /home/hiyio/HoloMotion/thirdparties/GMR/scripts/smpl_to_smplx.py \
  --input_file "$SMPL_FILE" \
  --output_file "$SMPLX_FILE" \
  --gender neutral

if [[ ! -f "$SMPLX_FILE" ]]; then
  echo "SMPLX output not found: $SMPLX_FILE"
  exit 1
fi

echo "[3/5] SMPLX -> GMR PKL"
ln -sf "$SMPLX_FILE" "$SMPLX_ONLY/${VIDEO_NAME}.npz"
conda run -n "$HOLOMOTION_ENV" python /home/hiyio/HoloMotion/thirdparties/GMR/scripts/smplx_to_robot_dataset.py \
  --robot unitree_g1 \
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
run_csv_to_npz() {
  local backend="$1"
  local headless_flag=()
  local record_flag=()
  if [[ "$EXPORT_MP4" == "1" ]]; then
    record_flag=(--record --record_backend "$backend")
  fi
  # Default behavior:
  # - exporting mp4 => non-headless by default
  # - unless user explicitly passes --headless to this wrapper script
  # - when not exporting mp4 => keep headless
  if [[ "$USER_HEADLESS" == "1" ]]; then
    headless_flag=(--headless)
  elif [[ "$EXPORT_MP4" != "1" ]]; then
    headless_flag=(--headless)
  fi
  conda run -n "$SIM_ENV" python /home/hiyio/whole_body_tracking/scripts/csv_to_npz_local.py \
    --input_file "$CSV_FILE" \
    --input_fps 30 --output_fps 50 \
    --output_dir "$NPZ_OUT" \
    --output_name "$VIDEO_NAME" \
    "${record_flag[@]}" \
    "${headless_flag[@]}"
}

run_csv_to_npz "$RECORD_BACKEND"

NPZ_FILE="${NPZ_OUT}/${VIDEO_NAME}.npz"
if [[ ! -f "$NPZ_FILE" ]]; then
  echo "NPZ output not found: $NPZ_FILE"
  exit 1
fi

MP4_FILE="${NPZ_OUT}/${VIDEO_NAME}.mp4"
if [[ "$EXPORT_MP4" == "1" && ! -f "$MP4_FILE" ]]; then
  echo "[WARN] MP4 not found after headless run. Retrying with viewport backend (non-headless)."
  run_csv_to_npz "viewport"
fi

# Convenience link at bundle root
ln -sf "$NPZ_FILE" "${RUN_DIR}/${VIDEO_NAME}.npz"

echo "[DONE] Bundle created: ${RUN_DIR}"
echo "NPZ: ${NPZ_FILE}"
if [[ "$EXPORT_MP4" != "1" ]]; then
  echo "MP4: disabled (EXPORT_MP4=${EXPORT_MP4})"
elif [[ -f "$MP4_FILE" ]]; then
  echo "MP4: ${MP4_FILE}"
else
  echo "[WARN] MP4 still not found: ${MP4_FILE}"
fi
