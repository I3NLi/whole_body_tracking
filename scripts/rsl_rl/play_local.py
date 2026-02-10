"""Script to play a checkpoint if an RL agent from RSL-RL (no wandb)."""
# python scripts/rsl_rl/play_local.py --task=Tracking-Flat-G1-v0 --num_envs=2
# python scripts/rsl_rl/play_local.py --task=Tracking-Flat-G1-Wo-State-Estimation-v0 --num_envs=2

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
#TODO no video 
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.") 
parser.add_argument("--video_length", type=int, default=6000, help="Length of the recorded video (in steps).")
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
parser.add_argument(
    "--motion_files",
    type=str,
    nargs="+",
    default=None,
    help="List of motion npz files for mixed playback.",
)
parser.add_argument(
    "--motion_dir",
    type=str,
    default=None,
    help="Directory containing motion npz files (all *.npz will be used).",
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


def _auto_find_motions(log_dir: str) -> list[str] | None:
    """Try to find multiple motions under log_dir (motion_list.txt or motions/*.npz)."""
    motion_list = os.path.join(log_dir, "motion_list.txt")
    if os.path.isfile(motion_list):
        motions_dir = os.path.join(log_dir, "motions")
        resolved: list[str] = []
        with open(motion_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                name = parts[0]
                original = parts[1] if len(parts) > 1 else None
                local_path = os.path.join(motions_dir, name)
                if os.path.isfile(local_path):
                    resolved.append(local_path)
                elif original and os.path.isfile(original):
                    resolved.append(original)
        return resolved if resolved else None

    motions_dir = os.path.join(log_dir, "motions")
    if os.path.isdir(motions_dir):
        files = sorted(glob.glob(os.path.join(motions_dir, "*.npz")))
        return files if files else None

    return None


def _resolve_motion_files(args_cli: argparse.Namespace) -> list[str] | None:
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
        return None

    missing = [p for p in deduped if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Motion file(s) not found: {missing}")

    return deduped


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
    motion_files = _resolve_motion_files(args_cli)
    if motion_files:
        env_cfg.commands.motion.motion_files = motion_files
        env_cfg.commands.motion.motion_file = motion_files[0]
        print(f"[INFO]: Using motion files from CLI: {len(motion_files)} files")
    else:
        auto_motion_files = _auto_find_motions(os.path.dirname(resume_path))
        if auto_motion_files is None:
            auto_motion_files = _auto_find_motions(run_dir)
        if auto_motion_files:
            env_cfg.commands.motion.motion_files = auto_motion_files
            env_cfg.commands.motion.motion_file = auto_motion_files[0]
            print(f"[INFO]: Auto-discovered motion files: {len(auto_motion_files)} files")
        else:
            # fallback to single motion.npz
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
