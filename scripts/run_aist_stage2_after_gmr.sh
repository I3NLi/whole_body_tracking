#!/usr/bin/env bash
set -euo pipefail

PATTERN="smplx_to_robot_dataset.py --robot unitree_g1 --src_folder /home/hiyio/whole_body_tracking/datasets/aistplusplus/converted_full/_gmr_input"

echo "[INFO] stage2 watcher started at $(date '+%F %T')"
while true; do
  if pgrep -f "$PATTERN" >/dev/null; then
    echo "[WAIT] gmr still running: $(date '+%F %T')"
    sleep 30
  else
    break
  fi
done

echo "[RUN] stage csv->link: $(date '+%F %T')"
env PYTHONUNBUFFERED=1 python /home/hiyio/whole_body_tracking/scripts/aist_pkl_to_training_npz_batch.py \
  --output_root /home/hiyio/whole_body_tracking/datasets/aistplusplus/converted_full \
  --from_stage csv \
  --to_stage link \
  --num_cpus 8 \
  --link_dir /home/hiyio/whole_body_tracking/motions/aist_full

echo "[DONE] stage2 finished at $(date '+%F %T')"
