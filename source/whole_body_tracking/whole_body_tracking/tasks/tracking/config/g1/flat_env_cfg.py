from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE

        # Hold the final motion frame for stability before resampling.
        # self.commands.motion.motion_hold_seconds = 2.0
        # Randomly pause (freeze current frame) to mimic deploy pause behavior.
        # self.commands.motion.random_pause_prob = 0.01
        # Each pause lasts 1-2 seconds (in motion fps), then resumes.
        # self.commands.motion.random_pause_duration_s = (1.0, 2.0)
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
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
        # High-CoM guidance term (more aggressive for flips).
        self.rewards.base_height_above.weight = 0.4
        self.rewards.base_height_above.params["min_height"] = 0.78
        self.rewards.base_height_above.params["max_height"] = 0.95
        # Base stability rewards (controlled here for G1; keep disabled for now).
        # Original weights: lin_vel_z_l2=-0.5, ang_vel_xy_l2=-0.05, flat_orientation_l2=-0.5
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        # Keep the last motion frame for 3s before resampling.
        self.commands.motion.motion_hold_seconds = 3.0
        # Enable progressive curriculum for Wo-State estimation by default.
        self.curriculum.wostate_progressive = CurrTerm(
            func=mdp.wostate_progressive_curriculum,
            params={
                "total_steps": 6_000_000,
                "stage_ends": (0.35, 0.60, 0.85),
                "max_stop_ratio": 0.20,
            },
        )


@configclass
class G1FlatBackflipEnvCfg(G1FlatWoStateEstimationEnvCfg):
    """专门针对空翻动作的训练配置。

    继承自 G1FlatWoStateEstimationEnvCfg 以兼容 WoState checkpoint（无 base_lin_vel / motion_anchor_pos_b）。

    主要调整：
    - 禁用 anchor_ori 终止条件（空翻过程中机器人会暂时大幅偏离参考朝向）
    - 放宽 anchor_pos_z 终止阈值（起跳初期允许更大高度误差）
    - 移除 base_height_above 上限（允许高跳跃获得奖励）
    - 放宽速度追踪 std（空翻时线速度/角速度远大于正常行走）
    """

    def __post_init__(self):
        super().__post_init__()

        # ── 终止条件 ──────────────────────────────────────────────
        # 空翻过程中躯干朝向会大幅偏离参考（翻转中途差值可达 2.0），禁用该终止
        self.terminations.anchor_ori = None
        # 起跳初期高度误差允许更大
        self.terminations.anchor_pos.params["threshold"] = 0.4

        # ── 奖励 ─────────────────────────────────────────────────
        # 移除高度上限，空翻腾空高度通常在 1.5~2m，当前 0.95 会压制跳跃动机
        self.rewards.base_height_above.params["max_height"] = 3.0
        self.rewards.base_height_above.params["min_height"] = 0.5

        # 空翻时身体线速度可达 3~5 m/s，放宽 std 避免奖励接近 0
        self.rewards.motion_body_lin_vel.params["std"] = 2.5
        # 空翻时角速度可达 10+ rad/s
        self.rewards.motion_body_ang_vel.params["std"] = 6.0

        # 空翻过程中位置追踪允许更大误差
        self.rewards.motion_body_pos.params["std"] = 0.5
        self.rewards.motion_global_anchor_pos.params["std"] = 0.5


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE


@configclass
class G1FlatFallRecoveryEnvCfg(G1FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Fall-recovery task: aggressively randomize reset pose/velocity so policy learns to stand up.
        self.commands.motion.pose_range = {
            "x": (-0.40, 0.40),
            "y": (-0.40, 0.40),
            "z": (-0.28, 0.12),
            "roll": (-2.2, 2.2),
            "pitch": (-2.2, 2.2),
            "yaw": (-3.14, 3.14),
        }
        self.commands.motion.velocity_range = {
            "x": (-1.4, 1.4),
            "y": (-1.4, 1.4),
            "z": (-0.9, 0.9),
            "roll": (-2.8, 2.8),
            "pitch": (-2.8, 2.8),
            "yaw": (-3.0, 3.0),
        }
        self.commands.motion.joint_position_range = (-0.40, 0.40)

        # Stronger perturbations during rollout to enforce recovery robustness.
        self.events.push_robot.interval_range_s = (6, 20)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-4.5, 4.5),
            "y": (-4.5, 4.5),
            "z": (-2.0, 2.0),
            "roll": (-3.8, 3.8),
            "pitch": (-3.8, 3.8),
            "yaw": (-4.5, 4.5),
        }

        # Relax early termination so policy has time to recover from severe falls.
        self.terminations.anchor_pos.params["threshold"] = 0.55
        self.terminations.anchor_pos.params["hold_seconds"] = 1.2
        self.terminations.anchor_ori.params["threshold"] = 1.6
        self.terminations.anchor_ori.params["hold_seconds"] = 1.2

        # Strengthen motion-tracking targets while keeping recovery bias.
        self.rewards.motion_global_anchor_pos.weight = 0.8
        self.rewards.motion_global_anchor_pos.params["std"] = 0.25
        self.rewards.motion_global_anchor_ori.weight = 0.8
        self.rewards.motion_global_anchor_ori.params["std"] = 0.35
        self.rewards.motion_body_pos.weight = 1.6
        self.rewards.motion_body_pos.params["std"] = 0.25
        self.rewards.motion_body_ori.weight = 1.6
        self.rewards.motion_body_ori.params["std"] = 0.35
        self.rewards.motion_body_lin_vel.weight = 1.3
        self.rewards.motion_body_lin_vel.params["std"] = 0.8
        self.rewards.motion_body_ang_vel.weight = 1.3
        self.rewards.motion_body_ang_vel.params["std"] = 2.5


@configclass
class G1FlatWoStateCurriculumEnvCfg(G1FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.wostate_progressive = CurrTerm(
            func=mdp.wostate_progressive_curriculum,
            params={
                "total_steps": 6_000_000,
                "stage_ends": (0.35, 0.60, 0.85),
                "max_stop_ratio": 0.20,
            },
        )
