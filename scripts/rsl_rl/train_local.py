# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL (local-only: load motion locally, save locally)."""

# ---------- Launch Isaac Sim Simulator first ----------
import argparse
import sys
import shutil  

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
    required=True,
    help="Path to local motion npz file (e.g. /path/to/motion.npz).",
)

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
    motion_file = os.path.abspath(args_cli.motion_file)
    if not os.path.isfile(motion_file):
        raise FileNotFoundError(f"Motion file not found: {motion_file}")
    env_cfg.commands.motion.motion_file = motion_file

    # --------- Default run_name: motion_name + task_name + timestamp ---------
    motion_stem = os.path.splitext(os.path.basename(args_cli.motion_file))[0]
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
    dst_motion = os.path.join(log_dir, "motion.npz")
    shutil.copy2(motion_file, dst_motion)
    print(f"[INFO] Motion file copied to: {dst_motion}")
    
    motion_mp4 = os.path.splitext(motion_file)[0] + ".mp4"
    if os.path.isfile(motion_mp4):
        dst_mp4 = os.path.join(log_dir, "motion.mp4")
        shutil.copy2(motion_mp4, dst_mp4)
        print(f"[INFO] Motion video copied to: {dst_mp4}")
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

    # log git state
    runner.add_git_repo_to_log(__file__)

    # resume if requested
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # persist configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # --------- Train ---------
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
