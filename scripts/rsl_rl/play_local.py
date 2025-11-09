"""Script to play a checkpoint if an RL agent from RSL-RL (no wandb)."""
#python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2
"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import pathlib
import glob

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL (load from logs).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# NEW: allow overriding logs location and checkpoint name
parser.add_argument("--log_dir", type=str, default=None,
                    help="Root dir of the run to load from. Defaults to logs/rsl_rl/<experiment_name>.")
parser.add_argument("--checkpoint_name", type=str, default=None,
                    help="Checkpoint filename to load (e.g., model_0300.pt). If omitted, load the latest.")

# Keep motion_file but also auto-discover below if not provided
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion npz file.")

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

import gymnasium as gym
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


def _find_latest_checkpoint(run_dir: str) -> str | None:
    """Return path to latest model_XXXX.pt in run_dir/checkpoints or run_dir."""
    # common patterns
    candidates = []
    # search in 'checkpoints' subdir first
    for pat in ["checkpoints/model_*.pt", "model_*.pt"]:
        candidates.extend(glob.glob(os.path.join(run_dir, pat)))
    if not candidates:
        return None
    # pick the largest XXXX number
    def _step_num(p):
        name = os.path.basename(p)
        # model_XXXX.pt
        try:
            core = name.split(".")[0]
            step = core.split("_")[1]
            return int(step)
        except Exception:
            return -1
    candidates.sort(key=_step_num, reverse=True)
    return candidates[0]


def _auto_find_motion(log_dir: str) -> str | None:
    """Try to find a motion.npz within the run directory."""
    # typical locations: same dir, subdir 'motions', or anywhere under run dir
    for rel in ["motion.npz", "motions/motion.npz"]:
        p = os.path.join(log_dir, rel)
        if os.path.isfile(p):
            return p
    # fallback: first match anywhere under run dir
    for p in glob.glob(os.path.join(log_dir, "**", "motion.npz"), recursive=True):
        return p
    return None


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent loaded from local logs."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # resolve run directory
    # default: logs/rsl_rl/<experiment_name>
    default_run_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_dir = os.path.abspath(args_cli.log_dir) if args_cli.log_dir else default_run_dir
    print(f"[INFO] Loading experiment from directory: {run_dir}")

    # resolve checkpoint path
    resume_path = None
    if args_cli.checkpoint_name:
        # explicit file name within run_dir or absolute path passed
        if os.path.isabs(args_cli.checkpoint_name):
            resume_path = args_cli.checkpoint_name
        else:
            # try checkpoints/ subdir first, then run_dir root
            cand1 = os.path.join(run_dir, "checkpoints", args_cli.checkpoint_name)
            cand2 = os.path.join(run_dir, args_cli.checkpoint_name)
            resume_path = cand1 if os.path.isfile(cand1) else cand2
    else:
        # leverage existing helper if available, otherwise fall back to scan
        resume_path = get_checkpoint_path(run_dir, agent_cfg.load_run, agent_cfg.load_checkpoint)
        if resume_path is None or not os.path.isfile(resume_path):
            resume_path = _find_latest_checkpoint(run_dir)

    if resume_path is None or not os.path.isfile(resume_path):
        raise FileNotFoundError(
            f"Could not locate a checkpoint. Tried run_dir={run_dir}, "
            f"checkpoint_name={args_cli.checkpoint_name or agent_cfg.load_checkpoint}."
        )
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # resolve motion file
    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
    else:
        # attempt to auto-discover motion.npz in the run directory
        auto_motion = _auto_find_motion(os.path.dirname(resume_path))
        if auto_motion is None:
            auto_motion = _auto_find_motion(run_dir)
        if auto_motion:
            env_cfg.commands.motion.motion_file = auto_motion
            print(f"[INFO]: Auto-discovered motion file: {auto_motion}")
        else:
            print("[WARN]: motion.npz not found under logs. "
                  "If this task needs motion data, please specify --motion_file <path>.")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
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
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    # Since wandb removed, tag with run_dir for traceability
    attach_onnx_metadata(env.unwrapped, run_dir, export_model_dir)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
