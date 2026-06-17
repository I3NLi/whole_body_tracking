# whole_body_tracking Local Guide

This guide covers local training and playback using the provided `scripts/rsl_rl/*_local.py` utilities.


## Setup

```bash
cd /home/hiyio/whole_body_tracking
```

Recommended local environment:

```bash
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python
```

Use the same environment that can import Isaac Sim 5.1 / IsaacLab 3.3, and keep `PYTHONPATH` on this extension:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/hiyio/whole_body_tracking/source/whole_body_tracking
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Install/update the extension:

```bash
cd /home/hiyio/whole_body_tracking
PYTHONNOUSERSITE=1 /home/hiyio/anaconda3/envs/env_isaacsim51/bin/python -m pip install -e source/whole_body_tracking
```

# export mp4 needs  ffmpeg
```
sudo apt update
sudo apt install ffmpeg -y
```

## MagicBot Z1 Chain

Z1 task:

```text
Tracking-Flat-MagicBot-Z1-Wo-State-Estimation-v0
```

Z1 robot/motion data lives mainly under:

```text
/home/hiyio/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/magicbot_z1.py
/home/hiyio/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/magicbot_z1/
/home/hiyio/whole_body_tracking/motions/magicbot_z1/
/home/hiyio/whole_body_tracking/logs/rsl_rl/magicbot_z1_flat/
```

Z1 local training with one motion:

```bash
cd /home/hiyio/whole_body_tracking

OMNI_KIT_ACCEPT_EULA=YES \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/hiyio/whole_body_tracking/source/whole_body_tracking \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python scripts/rsl_rl/train_local.py \
  --task=Tracking-Flat-MagicBot-Z1-Wo-State-Estimation-v0 \
  --num_envs=4096 \
  --motion_file=/home/hiyio/whole_body_tracking/motions/magicbot_z1/collected/dance1_subject1_magicbot_z1.npz \
  --headless \
  --device=cuda:0 \
  --kit_args=--portable
```

Z1 local training with all collected motions:

```bash
cd /home/hiyio/whole_body_tracking

OMNI_KIT_ACCEPT_EULA=YES \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/hiyio/whole_body_tracking/source/whole_body_tracking \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/hiyio/anaconda3/envs/env_isaacsim51/bin/python scripts/rsl_rl/train_local.py \
  --task=Tracking-Flat-MagicBot-Z1-Wo-State-Estimation-v0 \
  --num_envs=4096 \
  --motion_dir=/home/hiyio/whole_body_tracking/motions/magicbot_z1/collected \
  --headless \
  --device=cuda:0 \
  --kit_args=--portable
```

Z1 playback smoke test (verified locally). `timeout` returning `124` means the sim entered the continuous playback loop:

```bash
cd /home/hiyio/whole_body_tracking

OMNI_KIT_ACCEPT_EULA=YES \
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/hiyio/whole_body_tracking/source/whole_body_tracking \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
timeout 90s /home/hiyio/anaconda3/envs/env_isaacsim51/bin/python scripts/rsl_rl/play_local.py \
  --task=Tracking-Flat-MagicBot-Z1-Wo-State-Estimation-v0 \
  --headless \
  --num_envs=1 \
  --log_dir=/home/hiyio/whole_body_tracking/logs/rsl_rl/magicbot_z1_flat/2026-05-04_18-04-55_dance1_height+Tracking-Flat-MagicBot-Z1-Wo-State-Estimation-v0_resume-model_124000 \
  --checkpoint_name=model_183999.pt \
  --motion_file=/home/hiyio/whole_body_tracking/motions/magicbot_z1/collected/dance1_subject1_magicbot_z1.npz \
  --hide_contact_debug_vis \
  --hide_motion_debug_vis \
  --device=cuda:0 \
  --kit_args=--portable
```

For GUI playback, remove `--headless`, keep `--hide_contact_debug_vis --hide_motion_debug_vis` first, and use `--num_envs=1`.

Notes for Isaac Sim 5.1:

- `play_local.py` disables training-time domain randomization events during playback. This keeps playback deterministic and avoids Warp dtype issues in IsaacLab 3.3.
- Headless playback does not create debug/reference markers; Isaac Sim 5.1 currently raises a material API error for those marker spheres.
- ONNX export may print a `CastLike` version-conversion warning. The exporter continues and leaves the ONNX at the newer opset when conversion to opset 11 is not supported.

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

MagicBot-Z1 (24 DOF):

```bash
python scripts/rsl_rl/train_local.py \
  --task=Tracking-Flat-MagicBot-Z1-Wo-State-Estimation-v0 \
  --num_envs=4096 \
  --motion_file /home/hiyio/whole_body_tracking/motions/20260413-005208-magicbot/npz/magicbot_z1.npz \
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
