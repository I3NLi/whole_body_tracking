# whole_body_tracking Local Guide

This guide covers local training and playback using the provided `scripts/rsl_rl/*_local.py` utilities.


## Setup

```bash
cd /home/hiyio/whole_body_tracking
```

Make sure you are using the Isaac Lab Python environment that can import `isaaclab`.

# export mp4 needs  ffmpeg
```
sudo apt update
sudo apt install ffmpeg -y
```
## Local Training (Single Motion)

```bash
python scripts/rsl_rl/train_local.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --num_envs=4096 \
  --motion_file /home/hiyio/whole_body_tracking/motions/dance1_subject2.npz \
  --headless
```

Resume from a checkpoint:

```bash
python scripts/rsl_rl/train_local.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --num_envs=4096 \
  --resume=True \
  --resume_path /home/hiyio/whole_body_tracking/logs/rsl_rl/g1_flat/<run_name>/model_XXXXX.pt \
  --motion_file /home/hiyio/whole_body_tracking/motions/dance1_subject2.npz \
  --headless
```

## Local Training (Multiple Motions)

Explicit list:

```bash
python scripts/rsl_rl/train_local.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --num_envs=4096 \
  --motion_files /home/hiyio/whole_body_tracking/motions/dance1_subject2.npz \
                 /home/hiyio/whole_body_tracking/motions/dance2_subject4.npz \
  --headless
```

Directory (auto-loads `*.npz`, falls back to recursive search if empty):

```bash
python scripts/rsl_rl/train_local.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --num_envs=4096 \
  --motion_dir /home/hiyio/whole_body_tracking/motions \
  --headless
```

Sequential schedule (train N iterations per motion):

```bash
python scripts/rsl_rl/train_local.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --num_envs=4096 \
  --motion_dir /home/hiyio/whole_body_tracking/motions \
  --motion_rounds 500 \
  --headless
```

## Local Playback

```bash
python scripts/rsl_rl/play_local.py \
  --task=Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --motion_file /home/hiyio/whole_body_tracking/motions/dance1_subject2.npz \
  --headless
```

## Logs

Training logs are written to:

```
/home/hiyio/whole_body_tracking/logs/rsl_rl/
```

## Tips

- If GPU memory is tight, reduce `--num_envs` to 2048 or 1024.
- Keep `--headless` for stable performance.
