from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ----------------------------
# Utils
# ----------------------------

def _safe_std(std: float | torch.Tensor) -> torch.Tensor:
    """Clamp std to avoid division-by-zero / extremely sharp exponentials."""
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(float(std), dtype=torch.float32)
    return torch.clamp(std, min=1e-6)

def _select_indexes(all_names: Iterable[str], wanted: Optional[Iterable[str]]) -> list[int]:
    if wanted is None:
        return list(range(len(all_names)))
    wanted_set = set(wanted)
    return [i for i, name in enumerate(all_names) if name in wanted_set]

def _reduce(x: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "sum":
        return torch.sum(x, dim=-1)
    # default "mean"
    return torch.mean(x, dim=-1)

def _weighted_mean_sq_error(a: torch.Tensor, b: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
    """
    a, b: (..., B, D)  -> 返回 (...,) 上对(B,D)维度的加权均方差
    w:    (..., B) 或 (B,) 或 None
    """
    diff = a - b
    mse_per_body = torch.sum(diff * diff, dim=-1)  # (..., B)
    if w is not None:
        # 广播到 (..., B)
        if w.ndim == 1:
            w = w.unsqueeze(0).expand_as(mse_per_body)
        elif w.ndim == 0:
            w = w.reshape(1).expand_as(mse_per_body)
        weighted = (mse_per_body * w)
        denom = torch.clamp(torch.sum(w, dim=-1), min=1e-12)
        return torch.sum(weighted, dim=-1) / denom  # (...,)
    else:
        return torch.mean(mse_per_body, dim=-1)  # (...,)


# ----------------------------
# Anchor errors (global)
# ----------------------------

def motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    *,
    beta: float = 1.0,  # 温度（越大越“宽”）
) -> torch.Tensor:
    """
    exp(-beta * ||pos_err||^2 / std^2)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    s = _safe_std(std)
    err_sq = torch.sum((command.anchor_pos_w - command.robot_anchor_pos_w) ** 2, dim=-1)  # (N,)
    return torch.exp(-beta * err_sq / (s * s))


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    exp(-beta * quat_err^2 / std^2)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    s = _safe_std(std)
    err_sq = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2  # (N,)
    return torch.exp(-beta * err_sq / (s * s))


# ----------------------------
# Body errors (relative/global, pos/ori/vel)
# ----------------------------

def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: Optional[list[str]] = None,
    *,
    body_weights: Optional[torch.Tensor] = None,  # (B,) or (N,B)
    reduction: str = "mean",
    beta: float = 1.0,
) -> torch.Tensor:
    """
    exp(-beta * <weighted mean body mse> / std^2)
    reduction: 'mean' | 'sum'  —— 先在 D 维做平方和，再对 B 维做 mean/sum（默认 mean）。
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    idx = _select_indexes(command.cfg.body_names, body_names)
    if len(idx) == 0:
        return torch.zeros(env.num_envs, device=command.body_pos_relative_w.device)

    s = _safe_std(std)
    # (..., B, 3)
    a = command.body_pos_relative_w[:, idx]
    b = command.robot_body_pos_w[:, idx]
    mse = _weighted_mean_sq_error(a, b, body_weights)  # (N,)
    return torch.exp(-beta * mse / (s * s))


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: Optional[list[str]] = None,
    *,
    body_weights: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> torch.Tensor:
    """
    exp(-beta * <weighted mean quat_err^2> / std^2)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    idx = _select_indexes(command.cfg.body_names, body_names)
    if len(idx) == 0:
        return torch.zeros(env.num_envs, device=command.body_quat_relative_w.device)

    s = _safe_std(std)
    a = command.body_quat_relative_w[:, idx]
    b = command.robot_body_quat_w[:, idx]
    err_sq = quat_error_magnitude(a, b) ** 2  # (N,B)
    # 加权均值到 (N,)
    if body_weights is not None:
        bw = body_weights
        if bw.ndim == 1:
            bw = bw.unsqueeze(0).expand_as(err_sq)
        denom = torch.clamp(torch.sum(bw, dim=-1), min=1e-12)
        fused = torch.sum(err_sq * bw, dim=-1) / denom
    else:
        fused = torch.mean(err_sq, dim=-1)
    return torch.exp(-beta * fused / (s * s))


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: Optional[list[str]] = None,
    *,
    body_weights: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    idx = _select_indexes(command.cfg.body_names, body_names)
    if len(idx) == 0:
        return torch.zeros(env.num_envs, device=command.body_lin_vel_w.device)

    s = _safe_std(std)
    a = command.body_lin_vel_w[:, idx]
    b = command.robot_body_lin_vel_w[:, idx]
    mse = _weighted_mean_sq_error(a, b, body_weights)
    return torch.exp(-beta * mse / (s * s))


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: Optional[list[str]] = None,
    *,
    body_weights: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    idx = _select_indexes(command.cfg.body_names, body_names)
    if len(idx) == 0:
        return torch.zeros(env.num_envs, device=command.body_ang_vel_w.device)

    s = _safe_std(std)
    a = command.body_ang_vel_w[:, idx]
    b = command.robot_body_ang_vel_w[:, idx]
    mse = _weighted_mean_sq_error(a, b, body_weights)
    return torch.exp(-beta * mse / (s * s))


# ----------------------------
# Contacts
# ----------------------------

def feet_contact_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    *,
    smooth: bool = False,
    grace_steps: int = 0,
    smooth_k: float = 20.0,
) -> torch.Tensor:
    """
    默认与你原逻辑一致：最近接触时间 last_contact_time < threshold 且 first_air 为 True 计分。
    - smooth=True 时，用 sigmoid(threshold - last_contact_time) 做平滑，避免离散跳变。
    - grace_steps>0 时，允许短时超过阈值但在宽限步内恢复不扣分（更稳）。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # shape: (N, B)
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]

    if smooth:
        # s = sigmoid(k * (threshold - t))，t < threshold 时趋近 1，> 时趋近 0
        s = torch.sigmoid(smooth_k * (threshold - last_contact_time))
        score = s * first_air
        reward = torch.sum(score, dim=-1)
    else:
        mask = (last_contact_time < threshold)

        if grace_steps > 0:
            # “宽限”：若刚刚离地（first_air==1）且离地步数 <= grace_steps，也算通过
            # 需要接触传感器里有 step_since_contact / step_since_air 之类的计数；若没有，可删掉本段或改成你现有字段。
            step_since_contact = getattr(contact_sensor.data, "step_since_contact", None)
            if step_since_contact is not None:
                step_since_contact = step_since_contact[:, sensor_cfg.body_ids]
                grace_mask = (step_since_contact <= grace_steps) * first_air
                mask = mask | grace_mask.bool()

        reward = torch.sum(mask * first_air, dim=-1)

    return reward
# --- shim for terminations.py ---
def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    if body_names is None:
        return list(range(len(command.cfg.body_names)))
    wanted = set(body_names)
    return [i for i, name in enumerate(command.cfg.body_names) if name in wanted]
