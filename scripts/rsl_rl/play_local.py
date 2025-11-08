"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# -----------------------
# 1. CLI 参数（和原版几乎一样，只多一个 --checkpoint）
# -----------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")

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

"""Rest everything follows."""

# ------------------------------------------------
# 下面这块 import 顺序保持和原版一致（此时 omni 已经 ready）
# ------------------------------------------------
import gymnasium as gym
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


# -----------------------
# 小工具：从目录中自动选最新 model_*.pt
# -----------------------
def _find_latest_model_in_dir(dir_path: str) -> str:
    """在给定目录中查找所有 model_*.pt 文件，按数字后缀排序，返回迭代数最大的那个。"""
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"'{dir_path}' is not a directory.")

    candidates = [
        f for f in os.listdir(dir_path)
        if f.startswith("model_") and f.endswith(".pt") and os.path.isfile(os.path.join(dir_path, f))
    ]
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt files found in directory: {dir_path}")

    def _extract_iter(filename: str) -> int:
        # model_200000.pt -> 200000
        name = os.path.splitext(filename)[0]
        parts = name.split("_")
        try:
            return int(parts[-1])
        except ValueError:
            return -1

    latest_file = max(candidates, key=_extract_iter)
    return os.path.join(dir_path, latest_file)


def _resolve_checkpoint_path(args_cli, log_root_path: str, agent_cfg: RslRlOnPolicyRunnerCfg) -> str:
    """
    统一处理 checkpoint 逻辑：
    1) 如果传了 --checkpoint：
        - 是目录：自动选最新 model_*.pt
        - 是文件：直接用
    2) 否则退回到原来的 get_checkpoint_path(log_root_path, load_run, load_checkpoint)
    """
    if args_cli.checkpoint is not None:
        cp_input = os.path.abspath(args_cli.checkpoint)
        if os.path.isdir(cp_input):
            resume_path = _find_latest_model_in_dir(cp_input)
            real_path = os.path.realpath(resume_path)
            print(f"[INFO] Checkpoint directory provided: {cp_input}")
            print(f"[INFO] Using latest model symlink:   {resume_path}")
            if real_path != resume_path:
                print(f"[INFO] Real checkpoint path:        {real_path}")
            return resume_path
        elif os.path.isfile(cp_input):
            real_path = os.path.realpath(cp_input)
            print(f"[INFO] Using checkpoint file: {cp_input}")
            if real_path != cp_input:
                print(f"[INFO] Real checkpoint path: {real_path}")
            return cp_input
        else:
            raise FileNotFoundError(f"Checkpoint path is neither a file nor a directory: {cp_input}")

    # ⭐ 没传 --checkpoint 时，沿用原版逻辑（logs/rsl_rl + get_checkpoint_path）
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    return resume_path


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # ⭐ 和原版一样：再用 cli_args 覆盖 agent_cfg
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments（原版逻辑保留）
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    # ✅ 完全不走 wandb，统一用 _resolve_checkpoint_path
    resume_path = _resolve_checkpoint_path(args_cli, log_root_path, agent_cfg)

    # motion_file：如果 CLI 指定了，就覆盖 env_cfg
    if args_cli.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(os.path.realpath(resume_path))

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(os.path.realpath(resume_path)), "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    # 这里不再写 wandb run id，简单记录为 local_checkpoint
    attach_onnx_metadata(env.unwrapped, "local_checkpoint", export_model_dir)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
