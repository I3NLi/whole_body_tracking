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
RECORD_BACKEND="${RECORD_BACKEND:-auto}"
EXPORT_MP4="${EXPORT_MP4:-1}"
CSV2NPZ_TIMEOUT="${CSV2NPZ_TIMEOUT:-1200}"
CSV2NPZ_LOCK_DIR="${CSV2NPZ_LOCK_DIR:-/tmp/whole_body_tracking_csv2npz_viewport.lock}"
VIRTUAL_DISPLAY="${VIRTUAL_DISPLAY:-1}"
VIRTUAL_DISPLAY_REQUIRED="${VIRTUAL_DISPLAY_REQUIRED:-0}"
VIRTUAL_DISPLAY_NUM="${VIRTUAL_DISPLAY_NUM:-99}"
VIRTUAL_DISPLAY_WHD="${VIRTUAL_DISPLAY_WHD:-1920x1080x24}"
HEADLESS_FALLBACK_BACKEND="${HEADLESS_FALLBACK_BACKEND:-renderer}"

# Force GVHMR stage to use whole_body_tracking's BeyondMimic env by default.
if [[ -z "${GVHMR_ENV_FORCE_ALLOW:-}" ]]; then
  GVHMR_ENV="BeyondMimic"
fi

case "$RECORD_BACKEND" in
  auto|viewport|renderer) ;;
  *)
    echo "[WARN] Unsupported RECORD_BACKEND=$RECORD_BACKEND, falling back to auto"
    RECORD_BACKEND="auto"
    ;;
esac
if [[ "$HEADLESS_FALLBACK_BACKEND" != "renderer" ]]; then
  echo "[WARN] Unsupported HEADLESS_FALLBACK_BACKEND=$HEADLESS_FALLBACK_BACKEND, forcing renderer"
  HEADLESS_FALLBACK_BACKEND="renderer"
fi
if [[ "$EXPORT_MP4" != "1" ]]; then
  echo "[WARN] Forcing EXPORT_MP4=1 because this workflow requires mp4 output."
  EXPORT_MP4="1"
fi
if [[ "$USER_HEADLESS" == "1" ]]; then
  echo "[INFO] User requested --headless; csv_to_npz stage will honor it when renderer/headless mode is selected."
fi

acquire_csv2npz_lock() {
  local waited=0
  while ! mkdir "$CSV2NPZ_LOCK_DIR" 2>/dev/null; do
    if [[ -f "$CSV2NPZ_LOCK_DIR/pid" ]]; then
      local holder_pid
      holder_pid="$(cat "$CSV2NPZ_LOCK_DIR/pid" 2>/dev/null || true)"
      if [[ -n "$holder_pid" && ! "$holder_pid" =~ ^[0-9]+$ ]]; then
        holder_pid=""
      fi
      if [[ -n "$holder_pid" && ! -e "/proc/$holder_pid" ]]; then
        echo "[LOCK] Removing stale csv_to_npz lock held by dead pid=$holder_pid"
        rm -rf "$CSV2NPZ_LOCK_DIR"
        continue
      fi
    fi
    echo "[LOCK] Waiting for viewport csv_to_npz lock: $CSV2NPZ_LOCK_DIR (${waited}s)"
    sleep 5
    waited=$((waited + 5))
  done
  echo "$$" > "$CSV2NPZ_LOCK_DIR/pid"
  date -Is > "$CSV2NPZ_LOCK_DIR/acquired_at"
}

cleanup_csv2npz_children() {
  if [[ -n "${CSV2NPZ_PID:-}" ]]; then
    kill -TERM "$CSV2NPZ_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$CSV2NPZ_PID" 2>/dev/null || true
    pkill -TERM -P "$CSV2NPZ_PID" 2>/dev/null || true
    sleep 1
    pkill -KILL -P "$CSV2NPZ_PID" 2>/dev/null || true
  fi
}

CSV2NPZ_DISPLAY_PID=""
CSV2NPZ_DISPLAY_LOG=""

csv2npz_display_ready() {
  local display="${1:-${DISPLAY:-}}"
  if [[ -z "$display" ]]; then
    return 1
  fi
  DISPLAY="$display" xdpyinfo >/dev/null 2>&1
}

start_csv2npz_virtual_display() {
  if csv2npz_display_ready "${DISPLAY:-}"; then
    echo "[DISPLAY] Reusing existing display: ${DISPLAY}"
    return 0
  fi

  if [[ "$VIRTUAL_DISPLAY" != "1" ]]; then
    echo "[DISPLAY] Virtual display disabled (VIRTUAL_DISPLAY=$VIRTUAL_DISPLAY)"
    return 1
  fi

  if ! command -v Xvfb >/dev/null 2>&1; then
    echo "[DISPLAY] Xvfb not found; cannot start virtual display."
    return 1
  fi

  local display=":${VIRTUAL_DISPLAY_NUM}"
  local log="/tmp/whole_body_tracking_xvfb_${VIRTUAL_DISPLAY_NUM}.log"
  rm -f "$log"

  echo "[DISPLAY] Starting Xvfb on $display (screen=$VIRTUAL_DISPLAY_WHD)"
  Xvfb "$display" -screen 0 "$VIRTUAL_DISPLAY_WHD" -nolisten tcp -ac +extension GLX +render -noreset >"$log" 2>&1 &
  CSV2NPZ_DISPLAY_PID=$!
  CSV2NPZ_DISPLAY_LOG="$log"

  for _ in $(seq 1 40); do
    if csv2npz_display_ready "$display"; then
      export DISPLAY="$display"
      echo "[DISPLAY] Xvfb ready on ${DISPLAY} (pid=${CSV2NPZ_DISPLAY_PID})"
      return 0
    fi
    if ! kill -0 "$CSV2NPZ_DISPLAY_PID" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done

  echo "[DISPLAY] Xvfb failed to become ready on $display"
  if [[ -f "$log" ]]; then
    tail -n 80 "$log" || true
  fi
  stop_csv2npz_virtual_display
  return 1
}

stop_csv2npz_virtual_display() {
  if [[ -n "$CSV2NPZ_DISPLAY_PID" ]]; then
    kill -TERM "$CSV2NPZ_DISPLAY_PID" 2>/dev/null || true
    sleep 1
    kill -KILL "$CSV2NPZ_DISPLAY_PID" 2>/dev/null || true
    wait "$CSV2NPZ_DISPLAY_PID" 2>/dev/null || true
    CSV2NPZ_DISPLAY_PID=""
  fi
}

release_csv2npz_lock() {
  if [[ -d "$CSV2NPZ_LOCK_DIR" ]] && [[ "$(cat "$CSV2NPZ_LOCK_DIR/pid" 2>/dev/null || true)" == "$$" ]]; then
    rm -rf "$CSV2NPZ_LOCK_DIR"
  fi
}

trap 'cleanup_csv2npz_children; stop_csv2npz_virtual_display; release_csv2npz_lock' EXIT

if ! PYTHONNOUSERSITE=1 PYTHONPATH="/home/hiyio/HoloMotion/thirdparties/GVHMR:/home/hiyio/HoloMotion/holomotion/src:${PYTHONPATH:-}" \
  conda run -n "$GVHMR_ENV" python -c "import pytorch_lightning, ultralytics, colorlog, pytorch3d, hydra_zen, ffmpeg, wis3d, yacs, timm, pycolmap, smplx; from hmr4d.configs import register_store_gvhmr" >/dev/null 2>&1; then
  echo "[WARN] Env '$GVHMR_ENV' import check failed."
  if [[ "$GVHMR_ENV" != "gvhmr" ]]; then
    echo "[WARN] Falling back to 'gvhmr' only for step 1."
    GVHMR_ENV="gvhmr"
  fi
fi
run_gvhmr() {
  local dev="$1"
  echo "[GVHMR] device=$dev"
  env \
    GVHMR_DEVICE="$dev" \
    PYTHONNOUSERSITE=1 \
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
  local record_flag=()
  local app_flags=()
  local cmd=()
  local mode="viewport"
  local prev_display="${DISPLAY:-}"

  acquire_csv2npz_lock

  echo "[CSV2NPZ] Cleaning stale child processes from previous attempts"
  pkill -f "python .*/csv_to_npz_local.py" 2>/dev/null || true
  sleep 2

  case "$backend" in
    viewport)
      if csv2npz_display_ready "${DISPLAY:-}" || start_csv2npz_virtual_display; then
        mode="viewport"
        record_flag=(--record --record_backend viewport)
      else
        if [[ "$VIRTUAL_DISPLAY_REQUIRED" == "1" ]]; then
          echo "[CSV2NPZ] viewport requested but no real/virtual display is available."
          stop_csv2npz_virtual_display
          release_csv2npz_lock
          return 1
        fi
        echo "[CSV2NPZ] viewport requested but no usable display is available; falling back to headless renderer."
        mode="headless-renderer"
        record_flag=(--record --record_backend "$HEADLESS_FALLBACK_BACKEND")
        app_flags=(--headless --enable_cameras)
      fi
      ;;
    renderer)
      mode="headless-renderer"
      record_flag=(--record --record_backend renderer)
      app_flags=(--headless --enable_cameras)
      ;;
    auto|*)
      if [[ "$USER_HEADLESS" == "1" ]]; then
        echo "[CSV2NPZ] auto mode: --headless requested by user; using headless renderer."
        mode="headless-renderer"
        record_flag=(--record --record_backend "$HEADLESS_FALLBACK_BACKEND")
        app_flags=(--headless --enable_cameras)
      elif csv2npz_display_ready "${DISPLAY:-}" || start_csv2npz_virtual_display; then
        mode="viewport"
        record_flag=(--record --record_backend viewport)
      else
        echo "[CSV2NPZ] auto mode: no usable display found; using headless renderer fallback."
        mode="headless-renderer"
        record_flag=(--record --record_backend "$HEADLESS_FALLBACK_BACKEND")
        app_flags=(--headless --enable_cameras)
      fi
      ;;
  esac

  echo "[CSV2NPZ] Launch mode: $mode"
  echo "[CSV2NPZ] DISPLAY=${DISPLAY:-<unset>}"

  cmd=(env PYTHONNOUSERSITE=1 conda run -n "$SIM_ENV" python /home/hiyio/whole_body_tracking/scripts/csv_to_npz_local.py
    --input_file "$CSV_FILE"
    --input_fps 30 --output_fps 50
    --output_dir "$NPZ_OUT"
    --output_name "$VIDEO_NAME"
    "${record_flag[@]}"
    "${app_flags[@]}")

  echo "[CSV2NPZ] Command: ${cmd[*]}"
  set +e
  timeout --foreground --signal=TERM --kill-after=20s "$CSV2NPZ_TIMEOUT" "${cmd[@]}" &
  CSV2NPZ_PID=$!
  wait "$CSV2NPZ_PID"
  local rc=$?
  CSV2NPZ_PID=""
  set -e

  if [[ $rc -eq 124 || $rc -eq 137 ]]; then
    echo "[CSV2NPZ] Timed out / killed, cleaning child processes"
    cleanup_csv2npz_children
    pkill -f "python .*/csv_to_npz_local.py" 2>/dev/null || true
    sleep 2
  fi

  stop_csv2npz_virtual_display
  if [[ -n "$prev_display" ]]; then
    export DISPLAY="$prev_display"
  else
    unset DISPLAY || true
  fi

  release_csv2npz_lock
  return $rc
}

NPZ_FILE="${NPZ_OUT}/${VIDEO_NAME}.npz"
MP4_FILE="${NPZ_OUT}/${VIDEO_NAME}.mp4"
CSV2NPZ_RC=0
set +e
run_csv_to_npz "$RECORD_BACKEND"
CSV2NPZ_RC=$?
set -e

if [[ $CSV2NPZ_RC -ne 0 ]]; then
  echo "[WARN] CSV->NPZ first attempt failed rc=${CSV2NPZ_RC}."
fi
if [[ $CSV2NPZ_RC -ne 0 && -f "$NPZ_FILE" ]]; then
  echo "[WARN] CSV->NPZ exited rc=${CSV2NPZ_RC}, but NPZ exists, continue."
fi

if [[ ! -f "$NPZ_FILE" || ! -f "$MP4_FILE" ]]; then
  echo "[WARN] Missing output after first pass; retrying exactly once in auto mode (display if available, otherwise headless renderer fallback)."
  set +e
  run_csv_to_npz "auto"
  CSV2NPZ_RC=$?
  set -e
fi

if [[ $CSV2NPZ_RC -ne 0 && -f "$NPZ_FILE" ]]; then
  echo "[WARN] CSV->NPZ retry exited rc=${CSV2NPZ_RC}, but NPZ exists."
fi

if [[ ! -f "$NPZ_FILE" ]]; then
  echo "CSV->NPZ failed rc=${CSV2NPZ_RC} and NPZ not found: $NPZ_FILE"
  exit 1
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
