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
SOURCE_VIDEO_STEM="${VIDEO_BASENAME%.*}"
VIDEO_NAME="${VIDEO_NAME:-${SOURCE_VIDEO_STEM}}"
TIMESTAMP="${TIMESTAMP:-$(date +"%Y%m%d-%H%M%S")}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/home/hiyio/whole_body_tracking/motions}"

RUN_DIR="${OUTPUT_ROOT_BASE%/}/${TIMESTAMP}-${VIDEO_NAME}"
GVHMR_OUT="${RUN_DIR}/gvhmr"
SMPL_DIR="${GVHMR_OUT}/${SOURCE_VIDEO_STEM}"
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
GMR_ROOT="${GMR_ROOT:-/home/hiyio/HoloMotion/thirdparties/GMR}"
GVHMR_DEVICE="${GVHMR_DEVICE:-cuda}"
RECORD_BACKEND="${RECORD_BACKEND:-auto}"
EXPORT_MP4="${EXPORT_MP4:-1}"
CSV2NPZ_TIMEOUT="${CSV2NPZ_TIMEOUT:-1200}"

case "$RECORD_BACKEND" in
  auto|viewport|renderer) ;;
  *)
    echo "[WARN] Unsupported RECORD_BACKEND=$RECORD_BACKEND, falling back to auto"
    RECORD_BACKEND="auto"
    ;;
esac
if [[ "$EXPORT_MP4" != "1" ]]; then
  echo "[WARN] Forcing EXPORT_MP4=1 because this workflow requires mp4 output."
  EXPORT_MP4="1"
fi

if ! PYTHONNOUSERSITE=1 PYTHONPATH="/home/hiyio/HoloMotion/thirdparties/GVHMR:/home/hiyio/HoloMotion/holomotion/src:${PYTHONPATH:-}" \
  conda run -n "$GVHMR_ENV" python -c "import torch, torchvision, ultralytics, smplx; from hmr4d.configs import register_store_gvhmr" >/dev/null 2>&1; then
  echo "[ERROR] Env '$GVHMR_ENV' failed the GVHMR preflight check."
  echo "[ERROR] Automatic fallback to old 'gvhmr' env is disabled."
  exit 1
fi
run_gvhmr() {
  local dev="$1"
  local gvhmr_root="/home/hiyio/HoloMotion/thirdparties/GVHMR"
  echo "[GVHMR] env=$GVHMR_ENV device=$dev cwd=$gvhmr_root"
  (
    cd "$gvhmr_root"
    GVHMR_DEVICE="$dev" \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="/home/hiyio/HoloMotion/thirdparties/GVHMR:/home/hiyio/HoloMotion/holomotion/src:${PYTHONPATH:-}" \
    YOLO_CONFIG_DIR="/tmp/Ultralytics" \
    conda run -n "$GVHMR_ENV" python \
      /home/hiyio/HoloMotion/holomotion/src/data_curation/video_to_smpl_gvhmr.py \
      --video "$VIDEO" \
      --output_root "$GVHMR_OUT" \
      -s --no_render
  )
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
env PYTHONPATH="$GMR_ROOT:${PYTHONPATH:-}" \
  conda run -n "$HOLOMOTION_ENV" python "$GMR_ROOT/scripts/smpl_to_smplx.py" \
  --input_file "$SMPL_FILE" \
  --output_file "$SMPLX_FILE" \
  --gender neutral

if [[ ! -f "$SMPLX_FILE" ]]; then
  echo "SMPLX output not found: $SMPLX_FILE"
  exit 1
fi

echo "[3/5] SMPLX -> GMR PKL"
ln -sf "$SMPLX_FILE" "$SMPLX_ONLY/${VIDEO_NAME}.npz"
env PYTHONPATH="$GMR_ROOT:${PYTHONPATH:-}" \
  conda run -n "$HOLOMOTION_ENV" python "$GMR_ROOT/scripts/smplx_to_robot_dataset.py" \
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
  local cmd=()

  if [[ "$backend" == "viewport" && -z "${DISPLAY:-}" ]]; then
    echo "[CSV2NPZ] viewport requires a real DISPLAY, but DISPLAY is unset."
    return 1
  fi

  if [[ "$EXPORT_MP4" == "1" ]]; then
    record_flag=(--record --record_backend "$backend")
  fi

  if [[ "$USER_HEADLESS" == "1" ]]; then
    headless_flag=(--headless)
  elif [[ "$EXPORT_MP4" != "1" ]]; then
    headless_flag=(--headless)
  fi

  cmd=(env PYTHONNOUSERSITE=1 conda run -n "$SIM_ENV" python /home/hiyio/whole_body_tracking/scripts/csv_to_npz_local.py
    --input_file "$CSV_FILE"
    --input_fps 30 --output_fps 50
    --output_dir "$NPZ_OUT"
    --output_name "$VIDEO_NAME"
    "${record_flag[@]}"
    "${headless_flag[@]}")

  echo "[CSV2NPZ] DISPLAY=${DISPLAY:-<unset>}"
  echo "[CSV2NPZ] Command: ${cmd[*]}"

  set +e
  timeout --foreground --signal=TERM --kill-after=20s "$CSV2NPZ_TIMEOUT" "${cmd[@]}"
  local rc=$?
  set -e
  return $rc
}

NPZ_FILE="${NPZ_OUT}/${VIDEO_NAME}.npz"
MP4_FILE="${NPZ_OUT}/${VIDEO_NAME}.mp4"
CSV2NPZ_RC=0
if run_csv_to_npz "$RECORD_BACKEND"; then
  CSV2NPZ_RC=0
else
  CSV2NPZ_RC=$?
fi

if [[ $CSV2NPZ_RC -ne 0 && -f "$NPZ_FILE" ]]; then
  echo "[WARN] CSV->NPZ exited rc=${CSV2NPZ_RC}, but NPZ exists, continue."
fi

if [[ ! -f "$NPZ_FILE" ]]; then
  echo "CSV->NPZ failed rc=${CSV2NPZ_RC} and NPZ not found: $NPZ_FILE"
  exit 1
fi

if [[ "$EXPORT_MP4" == "1" && ! -f "$MP4_FILE" && "$RECORD_BACKEND" != "renderer" ]]; then
  echo "[WARN] MP4 not found after first pass. Retrying with renderer backend."
  run_csv_to_npz "renderer" || true
fi

if [[ "$EXPORT_MP4" == "1" && ! -f "$MP4_FILE" && -n "${DISPLAY:-}" && "$RECORD_BACKEND" != "viewport" ]]; then
  echo "[WARN] MP4 still not found. Retrying with viewport backend (non-headless)."
  run_csv_to_npz "viewport" || true
fi

if [[ ! -f "$MP4_FILE" ]]; then
  echo "CSV->NPZ finished but MP4 not found: $MP4_FILE"
  exit 1
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
