"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch
import os
import sys
import time
from pathlib import Path

REPO_SOURCE = Path(__file__).resolve().parents[1] / "source" / "whole_body_tracking"
if REPO_SOURCE.is_dir():
    sys.path.insert(0, str(REPO_SOURCE))

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument(
    "--motion_file",
    type=str,
    required=True,
    help="Path to local motion .npz file.",
)
parser.add_argument(
    "--robot",
    type=str,
    default="auto",
    choices=("auto", "unitree_g1", "magicbot_z1"),
    help="Robot articulation used to replay the motion. Defaults to inferring from joint_pos width.",
)
parser.add_argument(
    "--max_cycles",
    type=int,
    default=0,
    help="How many full motion cycles to replay before exiting. 0 means loop forever.",
)
parser.add_argument(
    "--pre_roll_seconds",
    type=float,
    default=0.0,
    help="Hold the first motion frame for a short time before playback starts.",
)
parser.add_argument(
    "--ready_file",
    type=str,
    default="",
    help="Optional file path to touch once the first viewport frame is ready for capture.",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
args_cli.motion_file = os.path.abspath(os.path.expanduser(args_cli.motion_file))
if not os.path.isfile(args_cli.motion_file):
    raise FileNotFoundError(f"Motion file not found: {args_cli.motion_file}")


def _infer_robot_from_motion_file(motion_file: str) -> str:
    with np.load(motion_file, allow_pickle=True) as data:
        if "joint_pos" not in data:
            raise KeyError(f"Missing 'joint_pos' in motion file: {motion_file}")
        joint_count = int(data["joint_pos"].shape[1])
    if joint_count == 29:
        return "unitree_g1"
    if joint_count == 24:
        return "magicbot_z1"
    raise ValueError(f"Unsupported joint_pos width={joint_count} in motion file: {motion_file}")


if args_cli.robot == "auto":
    args_cli.robot = _infer_robot_from_motion_file(args_cli.motion_file)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.robots.magicbot_z1 import MAGICBOT_Z1_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader

ROBOT_CFGS = {
    "unitree_g1": G1_CYLINDER_CFG,
    "magicbot_z1": MAGICBOT_Z1_CFG,
}


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()


    motion = MotionLoader(
        args_cli.motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    def _render_motion_frame(step_index: torch.Tensor):
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[step_index][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[step_index][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[step_index][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[step_index][:, 0]
        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[step_index], motion.joint_vel[step_index])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

    _render_motion_frame(time_steps)

    if args_cli.ready_file:
        ready_path = Path(os.path.abspath(os.path.expanduser(args_cli.ready_file)))
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.write_text("ready\n", encoding="utf-8")

    if args_cli.pre_roll_seconds > 0.0:
        hold_start = time.perf_counter()
        while simulation_app.is_running() and (time.perf_counter() - hold_start) < float(args_cli.pre_roll_seconds):
            _render_motion_frame(time_steps)

    # Simulation loop
    cycles_completed = 0
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        if reset_ids.any():
            cycles_completed += 1
            if args_cli.max_cycles > 0 and cycles_completed >= args_cli.max_cycles:
                break
            time_steps[reset_ids] = 0

        _render_motion_frame(time_steps)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = ROBOT_CFGS[args_cli.robot].replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print(f"[INFO] Replaying with robot={args_cli.robot}")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
