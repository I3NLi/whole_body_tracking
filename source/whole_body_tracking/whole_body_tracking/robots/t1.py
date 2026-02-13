import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.robots.actuator import DelayedImplicitActuatorCfg

# Prefer an explicit env var, and keep a stable local default for this workspace.
GMR_ROOT = Path(os.environ.get("HOLOMOTION_GMR_ROOT", "/home/hiyio/HoloMotion/thirdparties/GMR"))
T1_URDF_PATH = GMR_ROOT / "assets" / "booster_t1" / "T1_serial.urdf"

T1_JOINT_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

# Keep actuator gains conservative for first-pass motion replay / conversion stability.
STIFFNESS_HEAD = 20.0
STIFFNESS_ARMS = 40.0
STIFFNESS_WAIST = 35.0
STIFFNESS_LEGS = 80.0
STIFFNESS_FEET = 60.0

DAMPING_HEAD = 1.0
DAMPING_ARMS = 2.0
DAMPING_WAIST = 1.5
DAMPING_LEGS = 4.0
DAMPING_FEET = 3.0

DELAYED_ACTUATOR_DEFAULTS = dict(enable_delay=False, min_delay=0, max_delay=5)

T1_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=str(T1_URDF_PATH),
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            "Left_Hip_Pitch": -0.20,
            "Right_Hip_Pitch": -0.20,
            "Left_Knee_Pitch": 0.42,
            "Right_Knee_Pitch": 0.42,
            "Left_Ankle_Pitch": -0.22,
            "Right_Ankle_Pitch": -0.22,
            "Left_Shoulder_Pitch": 0.25,
            "Right_Shoulder_Pitch": 0.25,
            "Left_Shoulder_Roll": 0.10,
            "Right_Shoulder_Roll": -0.10,
            "Left_Elbow_Pitch": 0.50,
            "Right_Elbow_Pitch": 0.50,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "head": DelayedImplicitActuatorCfg(
            **DELAYED_ACTUATOR_DEFAULTS,
            joint_names_expr=["AAHead_yaw", "Head_pitch"],
            effort_limit_sim=7.0,
            velocity_limit_sim=12.56,
            stiffness=STIFFNESS_HEAD,
            damping=DAMPING_HEAD,
            armature=0.01,
        ),
        "arms": DelayedImplicitActuatorCfg(
            **DELAYED_ACTUATOR_DEFAULTS,
            joint_names_expr=[
                "Left_Shoulder_Pitch",
                "Left_Shoulder_Roll",
                "Left_Elbow_Pitch",
                "Left_Elbow_Yaw",
                "Right_Shoulder_Pitch",
                "Right_Shoulder_Roll",
                "Right_Elbow_Pitch",
                "Right_Elbow_Yaw",
            ],
            effort_limit_sim=18.0,
            velocity_limit_sim=18.84,
            stiffness=STIFFNESS_ARMS,
            damping=DAMPING_ARMS,
            armature=0.01,
        ),
        "waist": DelayedImplicitActuatorCfg(
            **DELAYED_ACTUATOR_DEFAULTS,
            joint_names_expr=["Waist"],
            effort_limit_sim=30.0,
            velocity_limit_sim=18.84,
            stiffness=STIFFNESS_WAIST,
            damping=DAMPING_WAIST,
            armature=0.01,
        ),
        "legs": DelayedImplicitActuatorCfg(
            **DELAYED_ACTUATOR_DEFAULTS,
            joint_names_expr=[
                "Left_Hip_Pitch",
                "Left_Hip_Roll",
                "Left_Hip_Yaw",
                "Left_Knee_Pitch",
                "Right_Hip_Pitch",
                "Right_Hip_Roll",
                "Right_Hip_Yaw",
                "Right_Knee_Pitch",
            ],
            effort_limit_sim={
                "Left_Hip_Pitch": 45.0,
                "Left_Hip_Roll": 30.0,
                "Left_Hip_Yaw": 30.0,
                "Left_Knee_Pitch": 60.0,
                "Right_Hip_Pitch": 45.0,
                "Right_Hip_Roll": 30.0,
                "Right_Hip_Yaw": 30.0,
                "Right_Knee_Pitch": 60.0,
            },
            velocity_limit_sim=18.84,
            stiffness=STIFFNESS_LEGS,
            damping=DAMPING_LEGS,
            armature=0.01,
        ),
        "feet": DelayedImplicitActuatorCfg(
            **DELAYED_ACTUATOR_DEFAULTS,
            joint_names_expr=[
                "Left_Ankle_Pitch",
                "Left_Ankle_Roll",
                "Right_Ankle_Pitch",
                "Right_Ankle_Roll",
            ],
            effort_limit_sim={
                "Left_Ankle_Pitch": 20.0,
                "Left_Ankle_Roll": 15.0,
                "Right_Ankle_Pitch": 20.0,
                "Right_Ankle_Roll": 15.0,
            },
            velocity_limit_sim=18.84,
            stiffness=STIFFNESS_FEET,
            damping=DAMPING_FEET,
            armature=0.01,
        ),
    },
)

T1_ACTION_SCALE = {}
for actuator_cfg in T1_CYLINDER_CFG.actuators.values():
    effort = actuator_cfg.effort_limit_sim
    stiffness = actuator_cfg.stiffness
    names = actuator_cfg.joint_names_expr
    if not isinstance(effort, dict):
        effort = {name: effort for name in names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in names}
    for name in names:
        if name in effort and name in stiffness and stiffness[name]:
            T1_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
