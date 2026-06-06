from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.magicbot_z1 import MAGICBOT_Z1_ACTION_SCALE, MAGICBOT_Z1_CFG
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import (
    G1FlatBackflipEnvCfg,
    G1FlatEnvCfg,
    G1FlatLowFreqEnvCfg,
    G1FlatWoStateCurriculumEnvCfg,
    G1FlatWoStateEstimationEnvCfg,
    WOSTATE_CURRICULUM_TOTAL_STEPS,
)


MAGICBOT_Z1_TRACKING_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

MAGICBOT_Z1_REWARD_BODY_NAMES = [
    body_name
    for body_name in MAGICBOT_Z1_TRACKING_BODY_NAMES
    if body_name not in {"left_ankle_roll_link", "right_ankle_roll_link"}
]

MAGICBOT_Z1_EXCLUDED_COMMAND_JOINT_NAMES = [
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

MAGICBOT_Z1_UNDESIRED_CONTACT_REGEX = [
    (
        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
        r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
    )
]


def _apply_magicbot_z1_overrides(env_cfg) -> None:
    env_cfg.scene.robot = MAGICBOT_Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.actions.joint_pos.scale = MAGICBOT_Z1_ACTION_SCALE
    env_cfg.commands.motion.anchor_body_name = "torso_link"
    env_cfg.commands.motion.body_names = list(MAGICBOT_Z1_TRACKING_BODY_NAMES)
    env_cfg.commands.motion.reset_exclude_joint_names = list(MAGICBOT_Z1_EXCLUDED_COMMAND_JOINT_NAMES)
    env_cfg.observations.policy.command.func = mdp.generated_commands_filtered
    env_cfg.observations.policy.command.params["exclude_joint_names"] = list(MAGICBOT_Z1_EXCLUDED_COMMAND_JOINT_NAMES)
    env_cfg.observations.critic.command.func = mdp.generated_commands_filtered
    env_cfg.observations.critic.command.params["exclude_joint_names"] = list(MAGICBOT_Z1_EXCLUDED_COMMAND_JOINT_NAMES)
    # Motion rewards ignore ankle links because the foot pose data is noisy and
    # otherwise turns into an implicit ankle-angle imitation target.
    env_cfg.rewards.motion_body_pos.params["body_names"] = list(MAGICBOT_Z1_REWARD_BODY_NAMES)
    env_cfg.rewards.motion_body_ori.params["body_names"] = list(MAGICBOT_Z1_REWARD_BODY_NAMES)
    env_cfg.rewards.motion_body_lin_vel.params["body_names"] = list(MAGICBOT_Z1_REWARD_BODY_NAMES)
    env_cfg.rewards.motion_body_ang_vel.params["body_names"] = list(MAGICBOT_Z1_REWARD_BODY_NAMES)
    env_cfg.commands.motion.ground_reference_on_reset = True
    env_cfg.commands.motion.ground_reference_clearance = 0.02
    env_cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces",
        body_names=MAGICBOT_Z1_UNDESIRED_CONTACT_REGEX,
    )


@configclass
class MagicBotZ1FlatEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_magicbot_z1_overrides(self)


@configclass
class MagicBotZ1FlatWoStateEstimationEnvCfg(G1FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_magicbot_z1_overrides(self)


@configclass
class MagicBotZ1FlatBackflipEnvCfg(G1FlatBackflipEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_magicbot_z1_overrides(self)


@configclass
class MagicBotZ1FlatLowFreqEnvCfg(G1FlatLowFreqEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_magicbot_z1_overrides(self)


@configclass
class MagicBotZ1FlatWoStateCurriculumEnvCfg(G1FlatWoStateCurriculumEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_magicbot_z1_overrides(self)
        self.curriculum.wostate_progressive = CurrTerm(
            func=mdp.wostate_progressive_curriculum,
            params={
                "total_steps": WOSTATE_CURRICULUM_TOTAL_STEPS,
                "stage_ends": (0.35, 0.60, 0.85),
                "max_stop_ratio": 0.20,
            },
        )
