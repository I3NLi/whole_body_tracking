import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.robots.actuator import DelayedImplicitActuatorCfg
from whole_body_tracking.robots.g1 import (
    DAMPING_RATIO,
    DELAYED_ACTUATOR_DEFAULTS,
    NATURAL_FREQ,
)

MAGICBOT_Z1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "head_joint",
]

HOLOMOTION_ROOT = Path(os.environ.get("HOLOMOTION_ROOT", "/home/hiyio/HoloMotion"))
MAGICBOT_Z1_URDF = HOLOMOTION_ROOT / "thirdparties" / "GMR" / "assets" / "magicbot_z1" / "urdf" / "SR-URDF_0410-02_isaaclab.urdf"

# 120 Nm and 50 Nm motor family values come from magiclab_rl_lab's MagicBotZ1 locomotion assets.
MAGICBOT_Z1_BIG_ARMATURE = 0.02863
MAGICBOT_Z1_SMALL_ARMATURE = 0.01503

MAGICBOT_Z1_BIG_STIFFNESS = MAGICBOT_Z1_BIG_ARMATURE * NATURAL_FREQ**2
MAGICBOT_Z1_BIG_DAMPING = 2.0 * DAMPING_RATIO * MAGICBOT_Z1_BIG_ARMATURE * NATURAL_FREQ
MAGICBOT_Z1_SMALL_STIFFNESS = MAGICBOT_Z1_SMALL_ARMATURE * NATURAL_FREQ**2
MAGICBOT_Z1_SMALL_DAMPING = 2.0 * DAMPING_RATIO * MAGICBOT_Z1_SMALL_ARMATURE * NATURAL_FREQ

MAGICBOT_Z1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=str(MAGICBOT_Z1_URDF),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.69),
        joint_pos={
            ".*_knee_joint": 0.35,
            ".*_ankle_pitch_joint": -0.18,
            ".*_elbow_joint": 0.5,
            "left_shoulder_pitch_joint": 0.15,
            "left_shoulder_roll_joint": 0.15,
            "right_shoulder_pitch_joint": 0.15,
            "right_shoulder_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            **DELAYED_ACTUATOR_DEFAULTS,
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=120.0,
            velocity_limit_sim=25.0,
            stiffness=MAGICBOT_Z1_BIG_STIFFNESS,
            damping=MAGICBOT_Z1_BIG_DAMPING,
            armature=MAGICBOT_Z1_BIG_ARMATURE,
        ),
        "feet": DelayedImplicitActuatorCfg(
            **DELAYED_ACTUATOR_DEFAULTS,
            effort_limit_sim=50.0,
            velocity_limit_sim=20.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=MAGICBOT_Z1_SMALL_STIFFNESS,
            damping=MAGICBOT_Z1_SMALL_DAMPING,
            armature=MAGICBOT_Z1_SMALL_ARMATURE,
        ),
        "waist_yaw": DelayedImplicitActuatorCfg(
            **DELAYED_ACTUATOR_DEFAULTS,
            effort_limit_sim=120.0,
            velocity_limit_sim=25.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=MAGICBOT_Z1_BIG_STIFFNESS,
            damping=MAGICBOT_Z1_BIG_DAMPING,
            armature=MAGICBOT_Z1_BIG_ARMATURE,
        ),
        "arms": DelayedImplicitActuatorCfg(
            **DELAYED_ACTUATOR_DEFAULTS,
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=50.0,
            velocity_limit_sim=18.0,
            stiffness=MAGICBOT_Z1_SMALL_STIFFNESS,
            damping=MAGICBOT_Z1_SMALL_DAMPING,
            armature=MAGICBOT_Z1_SMALL_ARMATURE,
        ),
        "head": DelayedImplicitActuatorCfg(
            **DELAYED_ACTUATOR_DEFAULTS,
            effort_limit_sim=50.0,
            velocity_limit_sim=25.0,
            joint_names_expr=["head_joint"],
            stiffness=MAGICBOT_Z1_SMALL_STIFFNESS,
            damping=MAGICBOT_Z1_SMALL_DAMPING,
            armature=MAGICBOT_Z1_SMALL_ARMATURE,
        ),
    },
)

MAGICBOT_Z1_ACTION_SCALE = {}
for actuator in MAGICBOT_Z1_CFG.actuators.values():
    effort = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    joint_exprs = actuator.joint_names_expr
    if not isinstance(effort, dict):
        effort = {name: effort for name in joint_exprs}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in joint_exprs}
    for name in joint_exprs:
        if name in effort and name in stiffness and stiffness[name]:
            MAGICBOT_Z1_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
