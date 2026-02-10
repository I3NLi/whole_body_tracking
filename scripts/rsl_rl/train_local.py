# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
python scripts/rsl_rl/train_local.py \
--task=Tracking-Flat-G1-v0 --num_envs=4096 \
--resume=True --resume_path=/home/hiyio/whole_body_tracking/logs/rsl_rl/g1_flat/2025-11-09_09-42-31_dance1_subject1+Tracking-Flat-G1-v0/model_3500.pt \
--motion_file=/home/hiyio/whole_body_tracking/logs/rsl_rl/g1_flat/2025-11-09_09-42-31_dance1_subject1+Tracking-Flat-G1-v0/motion.npz \
"""


"""
python scripts/rsl_rl/train_local.py \
--task=Tracking-Flat-G1-Wo-State-Estimation-v0 --num_envs=4096 \
--motion_file=/home/hiyio/whole_body_tracking/motions/dance1_subject2.npz \
--headless
"""

"""
python scripts/rsl_rl/train_local.py \
--task=Tracking-Flat-G1-Wo-State-Estimation-v0 --num_envs=4096 \
--motion_files /home/hiyio/whole_body_tracking/motions/dance1_subject2.npz \
                /home/hiyio/whole_body_tracking/motions/dance2_subject4.npz \
--headless
"""

"""
# 使用一个文件夹里多个 npz（自动收集 *.npz，若空则递归搜索子目录）
python scripts/rsl_rl/train_local.py \
--task=Tracking-Flat-G1-Wo-State-Estimation-v0 --num_envs=4096 \
--motion_dir /home/hiyio/whole_body_tracking/motions \
--headless
"""

"""
python scripts/rsl_rl/train_local.py \
--task=Tracking-Flat-G1-Wo-State-Estimation-v0 --num_envs=4096 \
--resume=True --resume_path=/home/hiyio/whole_body_tracking/logs/rsl_rl/g1_flat/2025-12-26_11-12-52_dance1_subject2+Tracking-Flat-G1-Wo-State-Estimation-v0/model_63500.pt \
--motion_file=/home/hiyio/whole_body_tracking/motions/dance1_subject2.npz \
--headless
"""

"""Script to train RL agent with RSL-RL (local-only: load motion locally, save locally)."""

# ---------- Launch Isaac Sim Simulator first ----------
import argparse
import sys
import shutil
import glob

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# -------------------- CLI --------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL (local only).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")  # required to satisfy Hydra decorator
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Path to local motion npz file (e.g. /path/to/motion.npz).",
)
parser.add_argument(
    "--motion_files",
    type=str,
    nargs="+",
    default=None,
    help="List of motion npz files for mixed training.",
)
parser.add_argument(
    "--motion_dir",
    type=str,
    default=None,
    help="Directory containing motion npz files (all *.npz will be used).",
)
parser.add_argument(
    "--motion_rounds",
    type=int,
    default=None,
    help="If set and multiple motions are provided, train sequentially for N iterations per motion.",
)
parser.add_argument("--resume_path", type=str, default=None, help="Name of the task.") 
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------- Rest everything follows ----------
import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
# 使用你本地仅保存版本的 Runner
from whole_body_tracking.utils.my_on_policy_runner_local import MotionOnPolicyRunner as OnPolicyRunner

# cuDNN / TF32 设置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# 多 npz 逻辑：
# 1) --motion_files 按给定顺序加入；2) --motion_file 支持单个或逗号分隔追加；
# 3) --motion_dir 收集目录下 *.npz（若为空则递归），按文件名排序；
# 4) 展开 glob 并去重（保留首次出现的顺序）。
# 解析得到的列表传给 env_cfg.commands.motion.motion_files；
# 环境侧会为每个 env 采样一个 motion_id（可设权重，否则均匀），
# 在 episode 结束或 motion 播完时重新采样；motion_files[0] 同时用于兼容单文件接口与 run_name。
def _resolve_motion_files(args_cli: argparse.Namespace) -> list[str]:
    files: list[str] = []
    if args_cli.motion_files:
        files.extend(args_cli.motion_files)
    if args_cli.motion_file:
        if "," in args_cli.motion_file:
            files.extend([p for p in args_cli.motion_file.split(",") if p])
        else:
            files.append(args_cli.motion_file)
    if args_cli.motion_dir:
        if not os.path.isdir(args_cli.motion_dir):
            raise FileNotFoundError(f"Motion dir not found: {args_cli.motion_dir}")
        files.extend(sorted(glob.glob(os.path.join(args_cli.motion_dir, "*.npz"))))
        if not files:
            files.extend(sorted(glob.glob(os.path.join(args_cli.motion_dir, "**", "*.npz"), recursive=True)))

    # expand any glob patterns
    expanded: list[str] = []
    for path in files:
        if any(ch in path for ch in ["*", "?", "["]):
            expanded.extend(sorted(glob.glob(path)))
        else:
            expanded.append(path)

    # de-duplicate while preserving order
    seen = set()
    deduped: list[str] = []
    for path in expanded:
        abspath = os.path.abspath(path)
        if abspath in seen:
            continue
        seen.add(abspath)
        deduped.append(abspath)

    if not deduped:
        raise ValueError("No motion files provided. Use --motion_file, --motion_files, or --motion_dir.")

    missing = [p for p in deduped if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Motion file(s) not found: {missing}")

    return deduped


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent (local-only)."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # env count / iterations
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # seed / device
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # --------- Local motion file only ---------
    motion_files = _resolve_motion_files(args_cli)
    env_cfg.commands.motion.motion_files = motion_files
    env_cfg.commands.motion.motion_file = motion_files[0]
    if args_cli.motion_rounds is not None and args_cli.motion_rounds <= 0:
        raise ValueError("motion_rounds must be a positive integer.")
    motion_total_rounds = None
    use_motion_schedule = args_cli.motion_rounds is not None and len(motion_files) > 1
    if args_cli.motion_rounds is not None:
        if len(motion_files) == 1:
            print("[INFO] motion_rounds set but only one motion; using it as max_iterations.")
            agent_cfg.max_iterations = args_cli.motion_rounds
        else:
            motion_total_rounds = args_cli.motion_rounds * len(motion_files)
            agent_cfg.max_iterations = motion_total_rounds
            print(
                f"[INFO] Motion schedule enabled: {len(motion_files)} motions x {args_cli.motion_rounds} iterations "
                f"= {motion_total_rounds} total iterations."
            )

    # --------- Default run_name: motion_name + task_name + timestamp ---------
    motion_stem = os.path.splitext(os.path.basename(motion_files[0]))[0]
    task_name = args_cli.task
    timestamp_compact = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not getattr(agent_cfg, "run_name", None):
        agent_cfg.run_name = f"{motion_stem}+{task_name}"

    # --------- Logging directories ---------
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # log_dir = {YYYY-mm-dd_HH-MM-SS}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create log directory before copy
    os.makedirs(log_dir, exist_ok=True)

    # copy motion npz into log directory
    if len(motion_files) == 1:
        dst_motion = os.path.join(log_dir, "motion.npz")
        shutil.copy2(motion_files[0], dst_motion)
        print(f"[INFO] Motion file copied to: {dst_motion}")

        motion_mp4 = os.path.splitext(motion_files[0])[0] + ".mp4"
        if os.path.isfile(motion_mp4):
            dst_mp4 = os.path.join(log_dir, "motion.mp4")
            shutil.copy2(motion_mp4, dst_mp4)
            print(f"[INFO] Motion video copied to: {dst_mp4}")
    else:
        motions_dir = os.path.join(log_dir, "motions")
        os.makedirs(motions_dir, exist_ok=True)
        used_names: set[str] = set()
        motion_list_path = os.path.join(log_dir, "motion_list.txt")
        with open(motion_list_path, "w", encoding="utf-8") as f:
            for idx, motion_path in enumerate(motion_files):
                base = os.path.basename(motion_path)
                name = base if base not in used_names else f"{idx:03d}_{base}"
                used_names.add(name)
                dst_motion = os.path.join(motions_dir, name)
                shutil.copy2(motion_path, dst_motion)
                f.write(f"{name}\t{motion_path}\n")

                motion_mp4 = os.path.splitext(motion_path)[0] + ".mp4"
                if os.path.isfile(motion_mp4):
                    dst_mp4 = os.path.join(motions_dir, os.path.splitext(name)[0] + ".mp4")
                    shutil.copy2(motion_mp4, dst_mp4)
        print(f"[INFO] Motion files copied to: {motions_dir}")
    # --------- Create env ---------
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # video wrapper (optional)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # MARL -> single agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # RSL-RL VecEnv wrapper
    env = RslRlVecEnvWrapper(env)

    # --------- Runner (local-only saver) ---------
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if use_motion_schedule:
        runner.motion_schedule_enabled = True
        runner.motion_schedule_total = motion_total_rounds
        runner.motion_schedule_rounds = args_cli.motion_rounds
        runner.motion_schedule_files = motion_files
        runner.motion_schedule_start_iter = runner.current_learning_iteration

    # log git state
    runner.add_git_repo_to_log(__file__)

    # resume if requested
    if agent_cfg.resume:
        # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        resume_path =args_cli.resume_path
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(args_cli.resume_path)

    # persist configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # --------- Train ---------
    if use_motion_schedule:
        motion_cmd = runner.env.unwrapped.command_manager.get_term("motion")
        env_ids = torch.arange(motion_cmd.num_envs, device=motion_cmd.device)
        original_weights = getattr(motion_cmd.cfg, "motion_sampling_weights", None)
        for idx, motion_path in enumerate(motion_files):
            runner.motion_schedule_index = idx
            runner.motion_schedule_segment_start_iter = runner.current_learning_iteration
            completed_rounds = idx * args_cli.motion_rounds
            weights = [0.0] * len(motion_files)
            weights[idx] = 1.0
            motion_cmd.cfg.motion_sampling_weights = weights
            with torch.inference_mode():
                motion_cmd._resample_command(env_ids)
            print(
                f"[INFO] Motion schedule {idx + 1}/{len(motion_files)}: {os.path.basename(motion_path)} "
                f"| completed {completed_rounds}/{motion_total_rounds} iterations"
            )
            runner.learn(num_learning_iterations=args_cli.motion_rounds, init_at_random_ep_len=(idx == 0))
            completed_rounds = (idx + 1) * args_cli.motion_rounds
            print(
                f"[INFO] Motion schedule progress: {completed_rounds}/{motion_total_rounds} iterations done "
                f"(last: {os.path.basename(motion_path)})"
            )
        motion_cmd.cfg.motion_sampling_weights = original_weights
    else:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
