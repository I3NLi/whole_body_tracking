from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand


def _lerp(start: float, end: float, alpha: float) -> float:
    return start + (end - start) * alpha


class wostate_progressive_curriculum(ManagerTermBase):
    """Four-stage curriculum for Wo-State tracking with smooth schedules."""

    _IMITATION_TERMS = (
        "motion_global_anchor_pos",
        "motion_global_anchor_ori",
        "motion_body_pos",
        "motion_body_ori",
        "motion_body_lin_vel",
        "motion_body_ang_vel",
    )

    _PUSH_RANGES = {
        "s1": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "s2": {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (-0.05, 0.05),
            "roll": (-0.15, 0.15),
            "pitch": (-0.15, 0.15),
            "yaw": (-0.2, 0.2),
        },
        "s3": {
            "x": (-0.6, 0.6),
            "y": (-0.6, 0.6),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        },
        "s4": {
            "x": (-0.75, 0.75),
            "y": (-0.75, 0.75),
            "z": (-0.25, 0.25),
            "roll": (-0.62, 0.62),
            "pitch": (-0.62, 0.62),
            "yaw": (-0.9, 0.9),
        },
    }

    def __init__(self, cfg: CurriculumTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._imitation_base_weights = {
            name: float(env.reward_manager.get_term_cfg(name).weight) for name in self._IMITATION_TERMS
        }
        self._motion_cmd: MotionCommand = env.command_manager.get_term("motion")
        anchor_pos_cfg = env.termination_manager.get_term_cfg("anchor_pos")
        anchor_ori_cfg = env.termination_manager.get_term_cfg("anchor_ori")
        self._anchor_pos_base_threshold = float(anchor_pos_cfg.params["threshold"])
        self._anchor_ori_base_threshold = float(anchor_ori_cfg.params["threshold"])
        self._anchor_pos_base_hold = float(anchor_pos_cfg.params.get("hold_seconds", 0.0) or 0.0)
        self._anchor_ori_base_hold = float(anchor_ori_cfg.params.get("hold_seconds", 0.0) or 0.0)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: Sequence[int],
        total_steps: int = 1_920_000,
        stage_ends: tuple[float, float, float] = (0.35, 0.60, 0.85),
        max_stop_ratio: float = 0.20,
        anchor_pos_threshold_range: tuple[float, float] | None = None,
        anchor_ori_threshold_range: tuple[float, float] | None = None,
        anchor_hold_seconds_range: tuple[float, float] | None = None,
    ) -> dict[str, float]:
        del env_ids
        step = float(env.common_step_counter)
        total_steps = max(float(total_steps), 1.0)
        progress = min(step / total_steps, 1.0)
        s1_end, s2_end, s3_end = stage_ends

        if progress < s1_end:
            stage = 1
            imitation_scale = 1.0
            upright_weight = 0.10
            balance_scale = 0.0
            action_rate_weight = -0.08
            push_range = self._PUSH_RANGES["s1"]
            push_interval = (6.0, 10.0)
            rand_level = 0.20
            stop_prob = 0.0
            stop_dur = (0.0, 0.0)
        elif progress < s2_end:
            stage = 2
            alpha = (progress - s1_end) / max(s2_end - s1_end, 1e-6)
            imitation_scale = _lerp(1.0, 0.75, alpha)
            upright_weight = _lerp(0.10, 0.40, alpha)
            balance_scale = _lerp(0.30, 0.65, alpha)
            action_rate_weight = _lerp(-0.08, -0.10, alpha)
            push_range = {
                key: (_lerp(self._PUSH_RANGES["s2"][key][0], self._PUSH_RANGES["s3"][key][0], 0.3 * alpha),
                      _lerp(self._PUSH_RANGES["s2"][key][1], self._PUSH_RANGES["s3"][key][1], 0.3 * alpha))
                for key in self._PUSH_RANGES["s2"]
            }
            push_interval = (_lerp(3.0, 2.0, alpha), _lerp(5.0, 3.2, alpha))
            rand_level = _lerp(0.30, 0.45, alpha)
            stop_prob = 0.0
            stop_dur = (0.0, 0.0)
        elif progress < s3_end:
            stage = 3
            alpha = (progress - s2_end) / max(s3_end - s2_end, 1e-6)
            imitation_scale = _lerp(0.75, 0.60, alpha)
            upright_weight = _lerp(0.40, 0.55, alpha)
            balance_scale = _lerp(0.65, 1.0, alpha)
            action_rate_weight = _lerp(-0.10, -0.15, alpha)
            push_range = {
                key: (_lerp(self._PUSH_RANGES["s3"][key][0], self._PUSH_RANGES["s4"][key][0], 0.25 * alpha),
                      _lerp(self._PUSH_RANGES["s3"][key][1], self._PUSH_RANGES["s4"][key][1], 0.25 * alpha))
                for key in self._PUSH_RANGES["s3"]
            }
            push_interval = (_lerp(1.8, 1.1, alpha), _lerp(2.8, 2.0, alpha))
            rand_level = _lerp(0.50, 0.80, alpha)
            stop_prob = 0.0
            stop_dur = (0.0, 0.0)
        else:
            stage = 4
            alpha = (progress - s3_end) / max(1.0 - s3_end, 1e-6)
            imitation_scale = _lerp(0.60, 0.55, alpha)
            upright_weight = 0.55
            balance_scale = 1.0
            action_rate_weight = -0.15
            push_range = self._PUSH_RANGES["s4"]
            push_interval = (_lerp(1.1, 0.8, alpha), _lerp(2.0, 1.6, alpha))
            rand_level = 0.80
            stop_prob = min(max_stop_ratio, _lerp(0.05, 0.18, alpha))
            stop_dur = (_lerp(0.2, 0.8, alpha), _lerp(0.5, 1.5, alpha))

        if anchor_pos_threshold_range is None:
            anchor_pos_threshold_range = (
                max(self._anchor_pos_base_threshold * 2.0, self._anchor_pos_base_threshold + 0.20),
                max(self._anchor_pos_base_threshold, 0.35),
            )
        if anchor_ori_threshold_range is None:
            anchor_ori_threshold_range = (
                max(self._anchor_ori_base_threshold * 1.5, self._anchor_ori_base_threshold + 0.30),
                max(self._anchor_ori_base_threshold, 0.95),
            )
        if anchor_hold_seconds_range is None:
            anchor_hold_seconds_range = (
                max(self._anchor_pos_base_hold, self._anchor_ori_base_hold, 0.60),
                max(self._anchor_pos_base_hold, self._anchor_ori_base_hold, 0.15),
            )

        anchor_pos_threshold = _lerp(anchor_pos_threshold_range[0], anchor_pos_threshold_range[1], progress)
        anchor_ori_threshold = _lerp(anchor_ori_threshold_range[0], anchor_ori_threshold_range[1], progress)
        anchor_hold_seconds = _lerp(anchor_hold_seconds_range[0], anchor_hold_seconds_range[1], progress)

        self._update_rewards(env, imitation_scale, upright_weight, balance_scale, action_rate_weight)
        self._update_push(env, push_range=push_range, interval_s=push_interval)
        self._update_randomization(env, level=rand_level)
        self._update_terminations(
            env,
            anchor_pos_threshold=anchor_pos_threshold,
            anchor_ori_threshold=anchor_ori_threshold,
            anchor_hold_seconds=anchor_hold_seconds,
        )
        self._motion_cmd.set_random_pause(pause_prob=stop_prob, duration_s=stop_dur)

        tracking_error = (
            self._motion_cmd.metrics["error_body_pos"].mean() + self._motion_cmd.metrics["error_anchor_pos"].mean()
        ).item()
        fall_rate = (
            env.termination_manager.get_term("anchor_pos").float().mean()
            + env.termination_manager.get_term("anchor_ori").float().mean()
        ).item()
        return {
            "stage": float(stage),
            "progress": float(progress),
            "imitation_scale": float(imitation_scale),
            "rand_level": float(rand_level),
            "stop_prob": float(stop_prob),
            "tracking_err_proxy": float(tracking_error),
            "fall_rate_proxy": float(fall_rate),
            "anchor_pos_threshold": float(anchor_pos_threshold),
            "anchor_ori_threshold": float(anchor_ori_threshold),
            "anchor_hold_seconds": float(anchor_hold_seconds),
        }

    def _update_rewards(
        self,
        env: "ManagerBasedRLEnv",
        imitation_scale: float,
        upright_weight: float,
        balance_scale: float,
        action_rate_weight: float,
    ) -> None:
        for term_name, base_weight in self._imitation_base_weights.items():
            term_cfg = env.reward_manager.get_term_cfg(term_name)
            term_cfg.weight = float(base_weight * imitation_scale)
            env.reward_manager.set_term_cfg(term_name, term_cfg)

        base_height_cfg = env.reward_manager.get_term_cfg("base_height_above")
        base_height_cfg.weight = upright_weight
        env.reward_manager.set_term_cfg("base_height_above", base_height_cfg)

        lin_vel_cfg = env.reward_manager.get_term_cfg("lin_vel_z_l2")
        lin_vel_cfg.weight = -0.5 * balance_scale
        env.reward_manager.set_term_cfg("lin_vel_z_l2", lin_vel_cfg)

        ang_vel_cfg = env.reward_manager.get_term_cfg("ang_vel_xy_l2")
        ang_vel_cfg.weight = -0.08 * balance_scale
        env.reward_manager.set_term_cfg("ang_vel_xy_l2", ang_vel_cfg)

        flat_cfg = env.reward_manager.get_term_cfg("flat_orientation_l2")
        flat_cfg.weight = -0.5 * balance_scale
        env.reward_manager.set_term_cfg("flat_orientation_l2", flat_cfg)

        action_cfg = env.reward_manager.get_term_cfg("action_rate_l2")
        action_cfg.weight = action_rate_weight
        env.reward_manager.set_term_cfg("action_rate_l2", action_cfg)

    def _update_push(
        self,
        env: "ManagerBasedRLEnv",
        push_range: dict[str, tuple[float, float]],
        interval_s: tuple[float, float],
    ) -> None:
        push_cfg = env.event_manager.get_term_cfg("push_robot")
        push_cfg.interval_range_s = interval_s
        push_cfg.params["velocity_range"] = push_range
        env.event_manager.set_term_cfg("push_robot", push_cfg)

    def _update_terminations(
        self,
        env: "ManagerBasedRLEnv",
        anchor_pos_threshold: float,
        anchor_ori_threshold: float,
        anchor_hold_seconds: float,
    ) -> None:
        anchor_pos_cfg = env.termination_manager.get_term_cfg("anchor_pos")
        anchor_pos_cfg.params["threshold"] = float(anchor_pos_threshold)
        anchor_pos_cfg.params["hold_seconds"] = float(anchor_hold_seconds)
        env.termination_manager.set_term_cfg("anchor_pos", anchor_pos_cfg)

        anchor_ori_cfg = env.termination_manager.get_term_cfg("anchor_ori")
        anchor_ori_cfg.params["threshold"] = float(anchor_ori_threshold)
        anchor_ori_cfg.params["hold_seconds"] = float(anchor_hold_seconds)
        env.termination_manager.set_term_cfg("anchor_ori", anchor_ori_cfg)

    def _update_randomization(self, env: "ManagerBasedRLEnv", level: float) -> None:
        mass_cfg = env.event_manager.get_term_cfg("mass_scale")
        mass_span = _lerp(0.02, 0.20, level)
        mass_cfg.params["mass_distribution_params"] = (1.0 - mass_span, 1.0 + mass_span)
        env.event_manager.set_term_cfg("mass_scale", mass_cfg)

        actuator_cfg = env.event_manager.get_term_cfg("actuator_gains")
        stiff_span = _lerp(0.10, 0.40, level)
        damp_span = _lerp(0.10, 0.50, level)
        actuator_cfg.params["stiffness_distribution_params"] = (1.0 - stiff_span, 1.0 + stiff_span)
        actuator_cfg.params["damping_distribution_params"] = (1.0 - damp_span, 1.0 + damp_span)
        env.event_manager.set_term_cfg("actuator_gains", actuator_cfg)

        joint_cfg = env.event_manager.get_term_cfg("joint_params")
        fric_span = _lerp(0.20, 0.70, level)
        arm_span = _lerp(0.20, 0.70, level)
        joint_cfg.params["friction_distribution_params"] = (1.0 - fric_span, 1.0 + fric_span)
        joint_cfg.params["armature_distribution_params"] = (1.0 - arm_span, 1.0 + arm_span)
        env.event_manager.set_term_cfg("joint_params", joint_cfg)

        gravity_cfg = env.event_manager.get_term_cfg("gravity")
        gravity_jitter = _lerp(0.05, 1.0, level)
        gravity_cfg.params["gravity_distribution_params"] = (
            [0.0, 0.0, -gravity_jitter],
            [0.0, 0.0, gravity_jitter],
        )
        env.event_manager.set_term_cfg("gravity", gravity_cfg)
