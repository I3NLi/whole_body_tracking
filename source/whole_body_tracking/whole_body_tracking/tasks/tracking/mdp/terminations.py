from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def non_finite_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate envs whose articulation state has NaN/Inf values.

    PhysX can occasionally produce non-finite body or joint state under aggressive
    resumed policies. Normal threshold-based terminations do not catch NaN because
    comparisons against NaN are false, so the bad env can otherwise survive until
    timeout and poison PPO returns.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    checks = (
        asset.data.root_state_w,
        asset.data.body_state_w,
        asset.data.joint_pos,
        asset.data.joint_vel,
    )
    bad = torch.zeros(asset.data.root_state_w.shape[0], dtype=torch.bool, device=asset.data.root_state_w.device)
    for value in checks:
        bad |= ~torch.isfinite(value.flatten(start_dim=1)).all(dim=1)
    return bad


def _apply_hold_seconds(
    env: ManagerBasedRLEnv,
    key: str,
    violation: torch.Tensor,
    hold_seconds: float = 0.0,
) -> torch.Tensor:
    if hold_seconds <= 0.0:
        return violation

    step_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
    if step_dt <= 0.0:
        return violation

    hold_steps = max(1, int(round(hold_seconds / step_dt)))
    state_name = "_whole_body_tracking_termination_hold_counts"
    hold_state = getattr(env, state_name, None)
    if hold_state is None:
        hold_state = {}
        setattr(env, state_name, hold_state)

    counter = hold_state.get(key)
    if counter is None or counter.shape != violation.shape or counter.device != violation.device:
        counter = torch.zeros_like(violation, dtype=torch.long)

    counter = torch.where(violation, counter + 1, torch.zeros_like(counter))
    hold_state[key] = counter
    return counter >= hold_steps


def bad_anchor_pos_z_only(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    hold_seconds: float = 0.0,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    violation = torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold
    return _apply_hold_seconds(env, "anchor_pos_z_only", violation, hold_seconds)


def bad_anchor_ori(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
    hold_seconds: float = 0.0,
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    quat_apply_inverse = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    violation = (motion_projected_gravity_b[:, 2] - quat_apply_inverse[:, 2]).abs() > threshold
    return _apply_hold_seconds(env, "anchor_ori", violation, hold_seconds)


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
