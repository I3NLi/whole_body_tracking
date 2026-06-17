from __future__ import annotations

import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def _sanitize_tensor(name: str, value: torch.Tensor, replacement: float = 0.0) -> torch.Tensor:
    if torch.isfinite(value).all():
        return value
    bad_count = int((~torch.isfinite(value)).sum().item())
    print(f"[WARN] Non-finite {name} detected ({bad_count} values); replacing with {replacement}.")
    return torch.nan_to_num(value, nan=replacement, posinf=replacement, neginf=replacement)


def _sanitize_obs_dict(obs_dict: dict) -> dict:
    for key, value in list(obs_dict.items()):
        if torch.is_tensor(value):
            obs_dict[key] = _sanitize_tensor(f"observation[{key!r}]", value)
        elif isinstance(value, dict):
            obs_dict[key] = _sanitize_obs_dict(value)
    return obs_dict


class FiniteRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL wrapper that prevents non-finite env outputs from entering PPO storage."""

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs, extras = super().get_observations()
        extras = _sanitize_obs_dict(extras)
        obs = _sanitize_tensor("policy observation", obs)
        return obs, extras

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs, extras = super().reset()
        extras = _sanitize_obs_dict(extras)
        obs = _sanitize_tensor("policy observation", obs)
        return obs, extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if not torch.isfinite(actions).all():
            bad_count = int((~torch.isfinite(actions)).sum().item())
            raise RuntimeError(f"Policy produced non-finite actions ({bad_count} values).")

        obs, rewards, dones, extras = super().step(actions)
        extras = _sanitize_obs_dict(extras)
        obs = _sanitize_tensor("policy observation", obs)
        rewards = _sanitize_tensor("reward", rewards)
        return obs, rewards, dones, extras
