#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-/home/hiyio/LAFAN1_Retargeting_Dataset/g1}"
OUT_DIR="${2:-/home/hiyio/whole_body_tracking/datasets/lafan}"
ENV_NAME="${ENV_NAME:-BeyondMimic}"
BACKEND="${BACKEND:-viewport}"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source dir not found: $SRC_DIR"
  exit 1
fi

mkdir -p "$OUT_DIR"

shopt -s nullglob
csv_files=("$SRC_DIR"/*.csv)
total="${#csv_files[@]}"
if [[ "$total" -eq 0 ]]; then
  echo "No CSV files found in: $SRC_DIR"
  exit 1
fi

echo "[INFO] Serial processing started."
echo "[INFO] Source: $SRC_DIR"
echo "[INFO] Output: $OUT_DIR"
echo "[INFO] Files:  $total"

i=0
for csv in "${csv_files[@]}"; do
  i=$((i + 1))
  base="$(basename "$csv" .csv)"
  echo "[${i}/${total}] ${base}"

  # Strictly serial: process one file at a time.
  conda run -n "$ENV_NAME" python /home/hiyio/whole_body_tracking/scripts/csv_to_npz_local.py \
    --input_file "$csv" \
    --input_fps 30 \
    --output_fps 50 \
    --output_dir "$OUT_DIR" \
    --output_name "$base" \
    --record \
    --record_backend "$BACKEND"
done

echo "[DONE] All g1 CSV files processed serially."
