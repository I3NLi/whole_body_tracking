from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import (
    _randomize_prop_by_op,
    randomize_actuator_gains as _BaseRandomizeActuatorGains,
    randomize_fixed_tendon_parameters as _BaseRandomizeFixedTendonParameters,
    randomize_joint_parameters as _BaseRandomizeJointParameters,
    randomize_physics_scene_gravity as _base_randomize_physics_scene_gravity,
    randomize_rigid_body_collider_offsets as _base_randomize_rigid_body_collider_offsets,
    randomize_rigid_body_material as _BaseRandomizeRigidBodyMaterial,
    randomize_rigid_body_mass as _BaseRandomizeRigidBodyMass,
    randomize_rigid_body_scale as _base_randomize_rigid_body_scale,
    randomize_visual_color as _BaseRandomizeVisualColor,
    randomize_visual_texture_material as _BaseRandomizeVisualTextureMaterial,
)
from isaaclab.managers import EventTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


__all__ = (
    "randomize_joint_default_pos",
    "randomize_rigid_body_com",
    "randomize_rigid_body_scale",
    "randomize_rigid_body_collider_offsets",
    "randomize_rigid_body_material",
    "randomize_rigid_body_mass",
    "randomize_actuator_gains",
    "randomize_joint_parameters",
    "randomize_fixed_tendon_parameters",
    "randomize_physics_scene_gravity",
    "randomize_visual_texture_material",
    "randomize_visual_color",
)


_RANDOMIZATION_LOGGER: logging.Logger | None = None
_RANDOMIZATION_LOG_PATH: Path | None = None
_RANDOMIZATION_LOG_ENABLED = (
    os.getenv("WBT_RANDOMIZATION_DEBUG", os.getenv("ISAACLAB_RANDOMIZATION_DEBUG", "1")).lower()
    not in {"0", "false", "off"}
)


def _ensure_randomization_logger_initialized() -> None:
    """Prepare the file logger used for domain-randomization debug output."""
    global _RANDOMIZATION_LOGGER, _RANDOMIZATION_LOG_PATH, _RANDOMIZATION_LOG_ENABLED

    if not _RANDOMIZATION_LOG_ENABLED or _RANDOMIZATION_LOGGER is not None:
        return

    log_path_env = os.getenv("WBT_RANDOMIZATION_LOG_PATH", os.getenv("ISAACLAB_RANDOMIZATION_LOG_PATH"))
    log_path = (
        Path(log_path_env).expanduser()
        if log_path_env
        else Path.home() / "whole_body_tracking_domain_randomization_log.txt"
    )

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive
        logging.getLogger(__name__).warning(
            "Failed to create directory for domain-randomization log '%s': %s. Disabling logging.", log_path, exc
        )
        _RANDOMIZATION_LOG_ENABLED = False
        return

    logger = logging.getLogger("whole_body_tracking.domain_randomization")
    if not logger.handlers:
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    _RANDOMIZATION_LOGGER = logger
    _RANDOMIZATION_LOG_PATH = log_path


if _RANDOMIZATION_LOG_ENABLED:
    _ensure_randomization_logger_initialized()


def _summarize_env_ids(env_ids: torch.Tensor | slice | Sequence[int] | None) -> str:
    """Return a concise description of which environments are affected."""
    if env_ids is None:
        return "all_envs"
    if isinstance(env_ids, slice):
        return f"slice(start={env_ids.start}, stop={env_ids.stop}, step={env_ids.step})"
    if isinstance(env_ids, torch.Tensor):
        tensor = env_ids.detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
        flat = tensor.reshape(-1)
        count = flat.numel()
        if count == 0:
            return "0 envs []"
        preview = flat[: min(count, 8)].tolist()
        if count > len(preview):
            return f"{count} envs (first={preview})"
        return f"{count} envs {preview}"
    if isinstance(env_ids, Sequence) and not isinstance(env_ids, (str, bytes)):
        env_list = list(env_ids)
        count = len(env_list)
        if count == 0:
            return "0 envs []"
        preview = env_list[: min(count, 8)]
        if count > len(preview):
            return f"{count} envs (first={preview})"
        return f"{count} envs {preview}"
    return str(env_ids)


def _stringify_value(value: Any) -> str:
    """Format values for logging without dumping huge tensors."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
        shape = tuple(tensor.shape)
        flat = tensor.reshape(-1)
        count = flat.numel()
        if count == 0:
            return f"Tensor(shape={shape}, values=[])"
        preview = flat[: min(count, 8)].tolist()
        if count > len(preview):
            return f"Tensor(shape={shape}, sample={preview})"
        return f"Tensor(shape={shape}, values={preview})"
    text = repr(value)
    return text if len(text) <= 256 else text[:253] + "..."


def record_randomization_event(
    event_name: str,
    *,
    phase: Literal["register", "execute"] = "execute",
    env_ids: torch.Tensor | slice | Sequence[int] | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Write a record of a domain-randomization call to the log file."""
    if not _RANDOMIZATION_LOG_ENABLED:
        return

    _ensure_randomization_logger_initialized()
    if _RANDOMIZATION_LOGGER is None:
        return

    parts = [f"{event_name}[{phase}]"]
    if env_ids is not None:
        parts.append(f"envs={_summarize_env_ids(env_ids)}")
    if details:
        detail_parts = []
        for key, value in details.items():
            if value is None:
                continue
            detail_parts.append(f"{key}={_stringify_value(value)}")
        if detail_parts:
            parts.append(", ".join(detail_parts))

    message = " | ".join(parts)
    _RANDOMIZATION_LOGGER.info(message)
    if phase == "register":
        logging.getLogger(__name__).info("[DomainRandomization] %s", message)


def randomize_rigid_body_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
):
    record_randomization_event(
        "randomize_rigid_body_scale",
        env_ids=env_ids,
        details={
            "asset": asset_cfg.name,
            "scale_range": scale_range,
            "relative_child_path": relative_child_path,
        },
    )
    return _base_randomize_rigid_body_scale(env, env_ids, scale_range, asset_cfg, relative_child_path)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    record_randomization_event(
        "randomize_joint_default_pos",
        env_ids=env_ids,
        details={
            "asset": asset_cfg.name,
            "joint_names": asset_cfg.joint_names,
            "operation": operation,
            "distribution": distribution,
            "pos_range": pos_distribution_params,
        },
    )

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    record_randomization_event(
        "randomize_rigid_body_com",
        env_ids=env_ids,
        details={
            "asset": asset_cfg.name,
            "body_names": asset_cfg.body_names,
            "com_range": com_range,
        },
    )

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range for selected environments
    coms[env_ids[:, None], body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_rigid_body_collider_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    rest_offset_distribution_params: tuple[float, float] | None = None,
    contact_offset_distribution_params: tuple[float, float] | None = None,
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    record_randomization_event(
        "randomize_rigid_body_collider_offsets",
        env_ids=env_ids,
        details={
            "asset": asset_cfg.name,
            "body_names": getattr(asset_cfg, "body_names", None),
            "rest_offset_range": rest_offset_distribution_params,
            "contact_offset_range": contact_offset_distribution_params,
            "distribution": distribution,
        },
    )
    return _base_randomize_rigid_body_collider_offsets(
        env,
        env_ids,
        asset_cfg,
        rest_offset_distribution_params=rest_offset_distribution_params,
        contact_offset_distribution_params=contact_offset_distribution_params,
        distribution=distribution,
    )


def randomize_physics_scene_gravity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    gravity_distribution_params: tuple[list[float], list[float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    record_randomization_event(
        "randomize_physics_scene_gravity",
        env_ids=None,
        details={
            "operation": operation,
            "distribution": distribution,
            "range": gravity_distribution_params,
        },
    )
    return _base_randomize_physics_scene_gravity(
        env,
        env_ids,
        gravity_distribution_params,
        operation,
        distribution=distribution,
    )


class randomize_rigid_body_material(_BaseRandomizeRigidBodyMaterial):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        asset_cfg: SceneEntityCfg = params.get("asset_cfg")
        record_randomization_event(
            "randomize_rigid_body_material",
            phase="register",
            details={
                "asset": asset_cfg.name if asset_cfg else None,
                "body_names": getattr(asset_cfg, "body_names", None) if asset_cfg else None,
                "num_buckets": params.get("num_buckets"),
                "static_range": params.get("static_friction_range"),
                "dynamic_range": params.get("dynamic_friction_range"),
                "restitution_range": params.get("restitution_range"),
                "make_consistent": params.get("make_consistent", False),
            },
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        record_randomization_event(
            "randomize_rigid_body_material",
            env_ids=env_ids,
            details={
                "asset": asset_cfg.name,
                "num_buckets": num_buckets,
                "make_consistent": make_consistent,
            },
        )
        return super().__call__(
            env,
            env_ids,
            static_friction_range,
            dynamic_friction_range,
            restitution_range,
            num_buckets,
            asset_cfg,
            make_consistent=make_consistent,
        )


class randomize_rigid_body_mass(_BaseRandomizeRigidBodyMass):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        asset_cfg: SceneEntityCfg = params.get("asset_cfg")
        record_randomization_event(
            "randomize_rigid_body_mass",
            phase="register",
            details={
                "asset": asset_cfg.name if asset_cfg else None,
                "body_names": getattr(asset_cfg, "body_names", None) if asset_cfg else None,
                "operation": params.get("operation"),
                "mass_range": params.get("mass_distribution_params"),
            },
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
    ):
        record_randomization_event(
            "randomize_rigid_body_mass",
            env_ids=env_ids,
            details={
                "asset": asset_cfg.name,
                "operation": operation,
                "distribution": distribution,
                "mass_range": mass_distribution_params,
                "recompute_inertia": recompute_inertia,
            },
        )
        return super().__call__(
            env,
            env_ids,
            asset_cfg,
            mass_distribution_params,
            operation,
            distribution=distribution,
            recompute_inertia=recompute_inertia,
        )


class randomize_actuator_gains(_BaseRandomizeActuatorGains):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        asset_cfg: SceneEntityCfg = params.get("asset_cfg")
        record_randomization_event(
            "randomize_actuator_gains",
            phase="register",
            details={
                "asset": asset_cfg.name if asset_cfg else None,
                "joint_names": getattr(asset_cfg, "joint_names", None) if asset_cfg else None,
                "operation": params.get("operation"),
                "stiffness_range": params.get("stiffness_distribution_params"),
                "damping_range": params.get("damping_distribution_params"),
            },
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        record_randomization_event(
            "randomize_actuator_gains",
            env_ids=env_ids,
            details={
                "asset": asset_cfg.name,
                "operation": operation,
                "distribution": distribution,
                "stiffness_range": stiffness_distribution_params,
                "damping_range": damping_distribution_params,
            },
        )
        return super().__call__(
            env,
            env_ids,
            asset_cfg,
            stiffness_distribution_params=stiffness_distribution_params,
            damping_distribution_params=damping_distribution_params,
            operation=operation,
            distribution=distribution,
        )


class randomize_joint_parameters(_BaseRandomizeJointParameters):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        asset_cfg: SceneEntityCfg = params.get("asset_cfg")
        record_randomization_event(
            "randomize_joint_parameters",
            phase="register",
            details={
                "asset": asset_cfg.name if asset_cfg else None,
                "joint_names": getattr(asset_cfg, "joint_names", None) if asset_cfg else None,
                "operation": params.get("operation"),
                "friction_range": params.get("friction_distribution_params"),
                "armature_range": params.get("armature_distribution_params"),
                "lower_limit_range": params.get("lower_limit_distribution_params"),
                "upper_limit_range": params.get("upper_limit_distribution_params"),
            },
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        logger = logging.getLogger(__name__)
        record_randomization_event(
            "randomize_joint_parameters",
            env_ids=env_ids,
            details={
                "asset": asset_cfg.name,
                "operation": operation,
                "distribution": distribution,
                "friction_range": friction_distribution_params,
                "armature_range": armature_distribution_params,
                "lower_limit_range": lower_limit_distribution_params,
                "upper_limit_range": upper_limit_distribution_params,
            },
        )
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device, dtype=torch.long)[env_ids]
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.to(device=self.asset.device, dtype=torch.long)
        else:
            env_ids = torch.tensor(env_ids, device=self.asset.device, dtype=torch.long)

        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        if friction_distribution_params is not None:
            friction_coeff = _randomize_prop_by_op(
                self.asset.data.default_joint_friction_coeff.clone(),
                friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            friction_coeff = torch.clamp(friction_coeff, min=0.0)
            static_friction_coeff = friction_coeff[env_ids[:, None], joint_ids]

            major_version = int(env.sim.get_version()[0])
            if major_version >= 5:
                dynamic_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_dynamic_friction_coeff.clone(),
                    friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                viscous_friction_coeff = _randomize_prop_by_op(
                    self.asset.data.default_joint_viscous_friction_coeff.clone(),
                    friction_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
                dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
                viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)
                dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)
                dynamic_friction_coeff = dynamic_friction_coeff[env_ids[:, None], joint_ids]
                viscous_friction_coeff = viscous_friction_coeff[env_ids[:, None], joint_ids]
            else:
                dynamic_friction_coeff = None
                viscous_friction_coeff = None

            self.asset.write_joint_friction_coefficient_to_sim(
                joint_friction_coeff=static_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
            if dynamic_friction_coeff is not None:
                dynamic_writer = getattr(self.asset, "write_joint_dynamic_friction_coefficient_to_sim", None)
                if dynamic_writer is not None:
                    dynamic_writer(dynamic_friction_coeff, joint_ids=joint_ids, env_ids=env_ids)
                else:
                    logger.warning("采样到了动态摩擦系数，但当前关节实现不支持写入该参数。")
            if viscous_friction_coeff is not None:
                viscous_writer = getattr(self.asset, "write_joint_viscous_friction_coefficient_to_sim", None)
                if viscous_writer is not None:
                    viscous_writer(viscous_friction_coeff, joint_ids=joint_ids, env_ids=env_ids)
                else:
                    logger.warning("采样到了粘滞摩擦系数，但当前关节实现不支持写入该参数。")

        if armature_distribution_params is not None:
            armature = _randomize_prop_by_op(
                self.asset.data.default_joint_armature.clone(),
                armature_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_armature_to_sim(
                armature[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
            )

        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            joint_pos_limits = self.asset.data.default_joint_pos_limits.clone()
            if lower_limit_distribution_params is not None:
                joint_pos_limits[..., 0] = _randomize_prop_by_op(
                    joint_pos_limits[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
            if upper_limit_distribution_params is not None:
                joint_pos_limits[..., 1] = _randomize_prop_by_op(
                    joint_pos_limits[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

            joint_pos_limits = joint_pos_limits[env_ids[:, None], joint_ids]
            if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
                raise ValueError(
                    "随机化项 'randomize_joint_parameters' 设置的关节下限大于上限，请检查关节位置范围的采样参数。"
                )
            self.asset.write_joint_position_limit_to_sim(
                joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
            )


class randomize_fixed_tendon_parameters(_BaseRandomizeFixedTendonParameters):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        asset_cfg: SceneEntityCfg = params.get("asset_cfg")
        record_randomization_event(
            "randomize_fixed_tendon_parameters",
            phase="register",
            details={
                "asset": asset_cfg.name if asset_cfg else None,
                "tendon_names": getattr(asset_cfg, "fixed_tendon_names", None) if asset_cfg else None,
                "operation": params.get("operation"),
                "stiffness_range": params.get("stiffness_distribution_params"),
                "damping_range": params.get("damping_distribution_params"),
                "limit_stiffness_range": params.get("limit_stiffness_distribution_params"),
                "lower_limit_range": params.get("lower_limit_distribution_params"),
                "upper_limit_range": params.get("upper_limit_distribution_params"),
                "rest_length_range": params.get("rest_length_distribution_params"),
                "offset_range": params.get("offset_distribution_params"),
            },
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        limit_stiffness_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        rest_length_distribution_params: tuple[float, float] | None = None,
        offset_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        record_randomization_event(
            "randomize_fixed_tendon_parameters",
            env_ids=env_ids,
            details={
                "asset": asset_cfg.name,
                "operation": operation,
                "distribution": distribution,
                "stiffness_range": stiffness_distribution_params,
                "damping_range": damping_distribution_params,
                "limit_stiffness_range": limit_stiffness_distribution_params,
                "lower_limit_range": lower_limit_distribution_params,
                "upper_limit_range": upper_limit_distribution_params,
                "rest_length_range": rest_length_distribution_params,
                "offset_range": offset_distribution_params,
            },
        )
        return super().__call__(
            env,
            env_ids,
            asset_cfg,
            stiffness_distribution_params=stiffness_distribution_params,
            damping_distribution_params=damping_distribution_params,
            limit_stiffness_distribution_params=limit_stiffness_distribution_params,
            lower_limit_distribution_params=lower_limit_distribution_params,
            upper_limit_distribution_params=upper_limit_distribution_params,
            rest_length_distribution_params=rest_length_distribution_params,
            offset_distribution_params=offset_distribution_params,
            operation=operation,
            distribution=distribution,
        )


class randomize_visual_texture_material(_BaseRandomizeVisualTextureMaterial):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        asset_cfg: SceneEntityCfg = params.get("asset_cfg")
        record_randomization_event(
            "randomize_visual_texture_material",
            phase="register",
            details={
                "asset": asset_cfg.name if asset_cfg else None,
                "body_names": getattr(asset_cfg, "body_names", None) if asset_cfg else None,
                "event_name": params.get("event_name"),
                "texture_paths": params.get("texture_paths"),
                "texture_rotation": params.get("texture_rotation"),
            },
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str],
        texture_rotation: tuple[float, float] = (0.0, 0.0),
    ):
        record_randomization_event(
            "randomize_visual_texture_material",
            env_ids=env_ids,
            details={
                "asset": asset_cfg.name,
                "event_name": event_name,
                "texture_paths": texture_paths,
                "texture_rotation": texture_rotation,
            },
        )
        return super().__call__(
            env,
            env_ids,
            event_name,
            asset_cfg,
            texture_paths,
            texture_rotation=texture_rotation,
        )


class randomize_visual_color(_BaseRandomizeVisualColor):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        asset_cfg: SceneEntityCfg = params.get("asset_cfg")
        record_randomization_event(
            "randomize_visual_color",
            phase="register",
            details={
                "asset": asset_cfg.name if asset_cfg else None,
                "mesh_name": params.get("mesh_name"),
                "event_name": params.get("event_name"),
                "colors": params.get("colors"),
            },
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
        mesh_name: str = "",
    ):
        record_randomization_event(
            "randomize_visual_color",
            env_ids=env_ids,
            details={
                "asset": asset_cfg.name,
                "event_name": event_name,
                "mesh_name": mesh_name,
                "colors": colors,
            },
        )
        return super().__call__(
            env,
            env_ids,
            event_name,
            asset_cfg,
            colors,
            mesh_name=mesh_name,
        )
