from isaaclab.utils import configclass

from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatBackflipPPORunnerCfg,
    G1FlatLowFreqPPORunnerCfg,
    G1FlatPPORunnerCfg,
    G1FlatWoStateCurriculumPPORunnerCfg,
)


@configclass
class MagicBotZ1FlatPPORunnerCfg(G1FlatPPORunnerCfg):
    experiment_name = "magicbot_z1_flat"


@configclass
class MagicBotZ1FlatBackflipPPORunnerCfg(G1FlatBackflipPPORunnerCfg):
    experiment_name = "magicbot_z1_backflip"


@configclass
class MagicBotZ1FlatLowFreqPPORunnerCfg(G1FlatLowFreqPPORunnerCfg):
    experiment_name = "magicbot_z1_flat_low_freq"


@configclass
class MagicBotZ1FlatWoStateCurriculumPPORunnerCfg(G1FlatWoStateCurriculumPPORunnerCfg):
    experiment_name = "magicbot_z1_wostate_curriculum"
