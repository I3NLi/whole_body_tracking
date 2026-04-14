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
                "total_steps": 6_000_000,
                "stage_ends": (0.35, 0.60, 0.85),
                "max_stop_ratio": 0.20,
            },
        )
