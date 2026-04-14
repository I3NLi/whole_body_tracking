import gymnasium as gym

from . import agents, flat_env_cfg


gym.register(
    id="Tracking-Flat-MagicBot-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MagicBotZ1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MagicBotZ1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-MagicBot-Z1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MagicBotZ1FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MagicBotZ1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-MagicBot-Z1-Backflip-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MagicBotZ1FlatBackflipEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MagicBotZ1FlatBackflipPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-MagicBot-Z1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MagicBotZ1FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MagicBotZ1FlatLowFreqPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-MagicBot-Z1-Wo-State-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.MagicBotZ1FlatWoStateCurriculumEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MagicBotZ1FlatWoStateCurriculumPPORunnerCfg",
    },
)
