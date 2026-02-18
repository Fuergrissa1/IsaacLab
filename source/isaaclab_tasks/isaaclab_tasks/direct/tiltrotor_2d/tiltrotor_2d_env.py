from __future__ import annotations
import math
import torch
from typing import Optional, Sequence, cast

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ManagerBasedEnv
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, sample_uniform

# Import robot configuration
from isaaclab_assets.robots.tiltrotor_2d import TILTROTOR_CFG

# Ensure that TILTROTOR_CFG and CUBOID_MARKER_CFG are correctly defined and imported
from isaaclab.markers import CUBOID_MARKER_CFG


class TiltrotorEnvWindow(BaseEnvWindow):
    """Window manager for the Tiltrotor environment."""

    def __init__(self, env: ManagerBasedEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # Initialize base window
        super().__init__(cast(ManagerBasedEnv, env), window_name)
        # Add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # Add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class TiltrotorEnvCfg(DirectRLEnvCfg):
    # Environment Configuration
    episode_length_s: float = 15.0
    decimation: int = 2
    action_space: int = 2
    observation_space: int = 12
    state_space: int = 0
    debug_vis: bool = True

    ui_window_class_type = TiltrotorEnvWindow

    # Simulation Configuration
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Scene Configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)

    # Robot Configuration
    robot: ArticulationCfg = TILTROTOR_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_scale: float = 10.0
    angle_limit: float = math.pi / 4

    # Reward Scales - Redesigned for goal-seeking behavior
    position_reward_scale: float = 50.0  # Much larger positive reward for being close
    progress_reward_scale: float = 20.0  # New: reward for making progress toward goal
    velocity_penalty_scale: float = -15.0  # Increased penalty for smoother flight (was -4.0, increased by ~4x)
    angular_velocity_penalty_scale: float = -4.0  # Increased penalty for smoother flight (was -1.0, increased by 4x)
    orientation_penalty_scale: float = 0.0  # DISABLED - let aircraft find natural hover attitude
    joint_limit_penalty_scale: float = -5.0  # Increased penalty (was -2.0)
    action_penalty_scale: float = -2.0  # Increased penalty for smoother actions (was -0.5, increased by 4x)
    acceleration_penalty_scale: float = -0.05  # Increased penalty for bobbing/lunging (was -0.01, increased by 5x)
    
    # Goal reaching incentives
    goal_transition_bonus: float = 50.0  # Reduced from 100.0 to balance with penalties
    hover_time_requirement: float = 1.5  # Shorter requirement (was 2.0)

    # Goal tracking
    max_consecutive_goals: int = 10  # Increased from 5

    # Central anchor penalty (prevent resting on origin)
    origin_penalty_scale: float = -50.0  # Penalty for being too close to local origin
    origin_penalty_radius: float = 0.3  # Distance from origin where penalty applies
    
    # Hover completion rewards
    hover_bonus_scale: float = 200.0  # Increased bonus for successful hover
    hover_position_tolerance: float = 0.15  # Slightly more forgiving (was 0.1)
    hover_velocity_tolerance: float = 0.15  # Slightly more forgiving (was 0.1)
    hover_angular_velocity_tolerance: float = 0.15  # Slightly more forgiving (was 0.1)


class TiltrotorEnv(DirectRLEnv):
    cfg: TiltrotorEnvCfg

    def __init__(self, cfg: TiltrotorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize action and force tensors
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)  # Force vector
        self._torque = torch.zeros(self.num_envs, 1, 3, device=self.device)  # Torque vector
        self._desired_angle = torch.zeros(self.num_envs, device=self.device)  # Target joint angle

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Progress tracking for goal-seeking rewards
        self._previous_distance_to_goal = torch.full((self.num_envs,), float('inf'), device=self.device)
        
        # Acceleration tracking for bobbing/lunging penalties
        self._previous_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Hover tracking
        self._hover_time = torch.zeros(self.num_envs, device=self.device)
        self._is_hovering = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "position_reward",
                "progress_reward",
                "close_to_goal_bonus",
                "origin_penalty",
                "velocity_penalty", 
                "angular_velocity_penalty",
                "orientation_penalty",
                "joint_limit_penalty",
                "action_penalty",
                "acceleration_penalty",
                "hover_bonus",
            ]
        }

        # Add handle for debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # Initialize extras dictionary
        self.extras = {}
        
        # Note: Cannot cache body/joint indices here - robot doesn't exist until _setup_scene()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()  # No collisions - even w/ ground plane
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.env_origins[:, 2] += 2.5  # Raise the environments by 1.0 units

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # Note: Body/joint indices will be cached lazily when first needed

    def _cache_body_joint_indices(self):
        """Cache body and joint indices for performance optimization (lazy initialization)."""
        # Check if already cached or if robot data is not ready
        if hasattr(self, '_fuse_body_idx') or not hasattr(self._robot, '_data') or self._robot._data is None:
            return
            
        try:
            # Cache body indices
            body_names = self._robot.data.body_names
            self._fuse_body_idx = body_names.index("fuse")
            self._rotor_body_idx = body_names.index("rotor")
            
            # Cache joint indices - all joints needed for observations and control
            joint_names = self._robot.data.joint_names
            self._revolute0_idx = joint_names.index("revolute0")
            self._revolute1_idx = joint_names.index("revolute1")
            self._revolute2_idx = joint_names.index("revolute2")
            self._revolute3_idx = joint_names.index("revolute3")
            
            # Cache joint ID arrays for find_joints calls
            self._rotor_id = self._robot.find_bodies("rotor")[0]        
            self._revolute0_id = self._robot.find_joints("revolute0")[0]
            self._revolute1_id = self._robot.find_joints("revolute1")[0]
            self._revolute2_id = self._robot.find_joints("revolute2")[0]
            self._revolute3_id = self._robot.find_joints("revolute3")[0]

            # Cache joint limit arrays - only for joints that still have limits
            # revolute0 and revolute1 limits were removed from USD
            # revolute2 and revolute3 still have limits and should be monitored for penalties
            self._revolute_joint_indices = [self._revolute2_idx, self._revolute3_idx]
            joint_limits = self._robot.data.joint_pos_limits  # Shape: (num_envs, num_joints, 2)
            self._joint_lower_limits = joint_limits[:, self._revolute_joint_indices, 0]  # Shape: (num_envs, num_joints)
            self._joint_upper_limits = joint_limits[:, self._revolute_joint_indices, 1]  # Shape: (num_envs, num_joints)
        except (AttributeError, ValueError, IndexError):
            # Robot data not ready yet, will try again later
            pass

    def _pre_physics_step(self, actions: torch.Tensor):
        # Clamp actions between -1 and 1
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # Action 0: Thrust force in Y direction
        self._thrust[:, 0, 1] = ((self.actions[:, 0] + 1.0) / 2.0 * self.cfg.thrust_scale)  # Scale from [-1,1] to [0, thrust_scale]

        # Action 1: Desired joint angle between -angle_limit and angle_limit
        self.desired_angle = (self.actions[:, 1] * self.cfg.angle_limit).unsqueeze(1)

    def _apply_action(self):
        """Apply the processed actions to the simulation."""
        # Apply external force to the rotor body per environment
        self._robot.set_external_force_and_torque(self._thrust, self._torque, body_ids=self._rotor_id)

        # Set the target angle for the revolute3 joint per environment
        self._robot.set_joint_position_target(self.desired_angle, joint_ids=self._revolute3_id)

    def _get_observations(self) -> dict:
        """Generate observations for the current state of the environment."""
        # Ensure indices are cached
        self._cache_body_joint_indices()
        
        # Extract joint angles and angular velocities using cached indices
        theta1 = self._robot.data.joint_pos[:, self._revolute0_idx].unsqueeze(1)
        theta2 = self._robot.data.joint_pos[:, self._revolute1_idx].unsqueeze(1)
        theta3 = self._robot.data.joint_pos[:, self._revolute3_idx].unsqueeze(1)
        omega1 = self._robot.data.joint_vel[:, self._revolute0_idx].unsqueeze(1)
        omega2 = self._robot.data.joint_vel[:, self._revolute1_idx].unsqueeze(1)
        omega3 = self._robot.data.joint_vel[:, self._revolute3_idx].unsqueeze(1)

        # Use consistent body position from physics simulation
        fuse_pos_w = self._robot.data.body_pos_w[:, self._fuse_body_idx, :]  # Shape: (num_envs, 3)
        
        # Alternative observation using forward kinematics (commented out for physical robot compatibility)
        # Link length
        # L = 1.03
        # x1 = L * torch.cos(theta1.squeeze())
        # z1 = L * torch.sin(theta1.squeeze())
        # x2 = x1 + L * torch.cos(theta1.squeeze() + theta2.squeeze())
        # z2 = z1 + L * torch.sin(theta1.squeeze() + theta2.squeeze())
        # y2 = torch.zeros_like(x2)
        # fuse_pos_kinematic = torch.stack([x2, y2, z2], dim=1) + self.scene.env_origins
        # Use fuse_pos_kinematic instead of fuse_pos_w for forward kinematics based observation
        
        # Goal vector relative to robot position
        goal_vector_relative = self._desired_pos_w - fuse_pos_w

        # Concatenate observations
        obs = torch.cat(
            [
                theta1,
                theta2,
                theta3,
                omega1,
                omega2,
                omega3,
                fuse_pos_w,
                goal_vector_relative,
            ],
            dim=-1,
        )  # Shape: (num_envs, 12)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute the reward for each environment with strong goal-seeking incentives."""
        # Ensure indices are cached
        self._cache_body_joint_indices()
        
        # Get current robot position using cached index
        current_pos = self._robot.data.body_pos_w[:, self._fuse_body_idx, :]
        current_vel = self._robot.data.body_lin_vel_w[:, self._fuse_body_idx, :]
        current_ang_vel = self._robot.data.body_ang_vel_w[:, self._fuse_body_idx, :]
        
        # Safety checks for NaN values in robot data
        if torch.any(torch.isnan(current_pos)):
            current_pos = torch.nan_to_num(current_pos, nan=0.0)
        if torch.any(torch.isnan(current_vel)):
            current_vel = torch.nan_to_num(current_vel, nan=0.0)
        if torch.any(torch.isnan(current_ang_vel)):
            current_ang_vel = torch.nan_to_num(current_ang_vel, nan=0.0)
        
        # Calculate distance to goal
        distance_to_goal = torch.norm(self._desired_pos_w - current_pos, dim=1)
        
        # 1. Position reward - gentler exponential decay + linear component
        # Exponential component for close distances, linear for far distances
        exp_position_reward = torch.exp(-2.0 * distance_to_goal) * self.cfg.position_reward_scale * 0.7
        linear_position_reward = torch.clamp(2.0 - distance_to_goal, min=0.0) * self.cfg.position_reward_scale * 0.3
        position_reward = exp_position_reward + linear_position_reward
        
        # 2. Progress reward - reward for getting closer to goal
        distance_improvement = self._previous_distance_to_goal - distance_to_goal
        # Handle first step where previous distance is inf
        valid_improvement = torch.where(
            torch.isinf(self._previous_distance_to_goal),
            torch.zeros_like(distance_improvement),  # No reward on first step
            distance_improvement
        )
        progress_reward = torch.clamp(valid_improvement, min=0.0) * self.cfg.progress_reward_scale / self.step_dt
        
        # Update previous distance
        self._previous_distance_to_goal = distance_to_goal.clone()
        
        # 3. Close to goal bonus - exponentially increasing reward for precision
        close_to_goal_bonus = torch.where(
            distance_to_goal < 0.5,  # Within 50cm
            self.cfg.goal_transition_bonus * torch.exp(-3.0 * distance_to_goal),  # Very sharp increase - little reward until extremely close
            torch.zeros_like(distance_to_goal)
        )
        
        # 4. Origin penalty - discourage resting on central anchor
        # Calculate distance to local origin (relative to environment)
        env_origins = self.scene.env_origins  # Shape: (num_envs, 3)
        distance_to_origin = torch.norm(current_pos - env_origins, dim=1)
        origin_penalty = torch.where(
            distance_to_origin < self.cfg.origin_penalty_radius,
            self.cfg.origin_penalty_scale * torch.exp(-2.0 * distance_to_origin),
            torch.zeros_like(distance_to_origin)
        )
        
        # 5. Velocity penalty - stronger when close to goal to encourage hovering
        velocity_magnitude = torch.norm(current_vel, dim=1)
        # Increase velocity penalty when close to goal to encourage stable hovering
        velocity_penalty_scale = torch.where(
            distance_to_goal < 0.5,  # When close to goal
            self.cfg.velocity_penalty_scale * 2.0,  # Double penalty for movement near goal
            self.cfg.velocity_penalty_scale * 0.5   # Reduced penalty when far from goal
        )
        velocity_penalty = velocity_magnitude ** 2 * velocity_penalty_scale
        
        # 6. Angular velocity penalty - distance dependent for precise hovering
        angular_velocity_magnitude = torch.norm(current_ang_vel, dim=1)
        # Increase angular velocity penalty when close to goal
        angular_velocity_penalty_scale = torch.where(
            distance_to_goal < 0.5,  # When close to goal
            self.cfg.angular_velocity_penalty_scale * 3.0,  # Triple penalty for rotation near goal
            self.cfg.angular_velocity_penalty_scale * 0.5   # Reduced penalty when far from goal
        )
        angular_velocity_penalty = angular_velocity_magnitude ** 2 * angular_velocity_penalty_scale
        
        # 7. Orientation penalty - keep minimal for stability
        fuse_quat = self._robot.data.body_quat_w[:, self._fuse_body_idx, :]
        # Safety check for valid quaternions
        quat_norm = fuse_quat.norm(dim=-1, keepdim=True)
        quat_norm = torch.clamp(quat_norm, min=1e-8)  # Prevent division by zero
        fuse_quat = fuse_quat / quat_norm
        
        up_vector_body = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32)
        up_vector_body = up_vector_body.unsqueeze(0).expand(self.num_envs, 3)
        up_vector_world = self._quaternion_apply(fuse_quat, up_vector_body)
        world_up_vector = up_vector_body
        dot_product = (up_vector_world * world_up_vector).sum(dim=1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        orientation_penalty = (1.0 - dot_product) * self.cfg.orientation_penalty_scale
        
        # 8. Joint limit penalty - minimal to allow full range of motion
        if hasattr(self, '_revolute_joint_indices') and hasattr(self, '_joint_lower_limits'):
            joint_positions = self._robot.data.joint_pos[:, self._revolute_joint_indices]
            lower_limit_dist = joint_positions - self._joint_lower_limits
            upper_limit_dist = self._joint_upper_limits - joint_positions
            min_limit_dist = torch.min(lower_limit_dist, upper_limit_dist)
            threshold = 0.05  # Tighter threshold (was 0.1)
            limit_penalty = torch.clamp(threshold - min_limit_dist, min=0.0) / threshold
            joint_limit_penalty = limit_penalty.sum(dim=1) * self.cfg.joint_limit_penalty_scale
        else:
            # Fallback if joint indices not cached yet
            joint_limit_penalty = torch.zeros_like(distance_to_goal)
        
        # 9. Action penalty - minimal to allow active control
        # Safety check for actions
        if hasattr(self, 'actions') and self.actions is not None:
            safe_actions = torch.nan_to_num(self.actions, nan=0.0)
            action_magnitude = torch.norm(safe_actions, dim=1)
        else:
            action_magnitude = torch.zeros(self.num_envs, device=self.device)
        action_penalty = action_magnitude * self.cfg.action_penalty_scale
        
        # 10. Acceleration penalty - distance dependent to reduce bobbing near goals
        acceleration = (current_vel - self._previous_velocity) / self.step_dt
        acceleration_magnitude = torch.norm(acceleration, dim=1)
        # Increase acceleration penalty when close to goal to prevent bobbing
        acceleration_penalty_scale = torch.where(
            distance_to_goal < 0.5,  # When close to goal
            self.cfg.acceleration_penalty_scale * 4.0,  # Quadruple penalty for acceleration near goal
            self.cfg.acceleration_penalty_scale * 0.5   # Reduced penalty when far from goal
        )
        acceleration_penalty = acceleration_magnitude ** 2 * acceleration_penalty_scale
        
        # 11. Hover detection and bonus - only when close to goal
        in_position = distance_to_goal < self.cfg.hover_position_tolerance
        low_velocity = velocity_magnitude < self.cfg.hover_velocity_tolerance
        low_angular_velocity = angular_velocity_magnitude < self.cfg.hover_angular_velocity_tolerance
        is_hovering = in_position & low_velocity & low_angular_velocity
        
        # Update hover time tracking
        self._hover_time = torch.where(is_hovering, self._hover_time + self.step_dt, torch.zeros_like(self._hover_time))
        hover_completed = self._hover_time >= self.cfg.hover_time_requirement
        
        # Hover bonus (large one-time reward for achieving hover AT THE GOAL)
        hover_bonus = torch.where(hover_completed & ~self._is_hovering, 
                                  torch.full_like(hover_completed, self.cfg.hover_bonus_scale, dtype=torch.float32),
                                  torch.zeros_like(hover_completed, dtype=torch.float32))
        
        # Update hover state
        self._is_hovering = hover_completed
        
        # Update previous velocity
        self._previous_velocity = current_vel.clone()
        
        # Safety check: ensure no NaN values in reward components
        position_reward = torch.nan_to_num(position_reward, nan=0.0)
        progress_reward = torch.nan_to_num(progress_reward, nan=0.0)
        close_to_goal_bonus = torch.nan_to_num(close_to_goal_bonus, nan=0.0)
        origin_penalty = torch.nan_to_num(origin_penalty, nan=0.0)
        velocity_penalty = torch.nan_to_num(velocity_penalty, nan=0.0)
        angular_velocity_penalty = torch.nan_to_num(angular_velocity_penalty, nan=0.0)
        orientation_penalty = torch.nan_to_num(orientation_penalty, nan=0.0)
        joint_limit_penalty = torch.nan_to_num(joint_limit_penalty, nan=0.0)
        action_penalty = torch.nan_to_num(action_penalty, nan=0.0)
        acceleration_penalty = torch.nan_to_num(acceleration_penalty, nan=0.0)
        hover_bonus = torch.nan_to_num(hover_bonus, nan=0.0)
        
        # Total reward - goal seeking strongly encouraged
        reward = (position_reward + progress_reward + close_to_goal_bonus +
                  origin_penalty + velocity_penalty + angular_velocity_penalty + orientation_penalty + 
                  joint_limit_penalty + action_penalty + acceleration_penalty + hover_bonus)

        # Logging
        self._episode_sums["position_reward"] += position_reward
        self._episode_sums["progress_reward"] += progress_reward
        self._episode_sums["close_to_goal_bonus"] += close_to_goal_bonus
        self._episode_sums["origin_penalty"] += origin_penalty
        self._episode_sums["velocity_penalty"] += velocity_penalty
        self._episode_sums["angular_velocity_penalty"] += angular_velocity_penalty
        self._episode_sums["orientation_penalty"] += orientation_penalty
        self._episode_sums["joint_limit_penalty"] += joint_limit_penalty
        self._episode_sums["action_penalty"] += action_penalty
        self._episode_sums["acceleration_penalty"] += acceleration_penalty
        self._episode_sums["hover_bonus"] += hover_bonus

        return reward
    
    def _quaternion_apply(self, quat, vec):
        """Rotate vector(s) vec by quaternion(s) quat."""
        # quat: (..., 4)
        # vec: (..., 3)
        # Normalize quaternion
        quat = quat / quat.norm(dim=-1, keepdim=True)
        qvec = quat[..., 1:]
        uv = torch.cross(qvec, vec, dim=-1)
        uuv = torch.cross(qvec, uv, dim=-1)
        return vec + 2 * (quat[..., :1] * uv + uuv)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine which environments are done."""
        # Ensure indices are cached
        self._cache_body_joint_indices()
        
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Add hover completion as a success termination
        hover_success = self._is_hovering.clone()
        
        # Out of bounds check - only if indices are available
        if hasattr(self, '_fuse_body_idx'):
            current_pos = self._robot.data.body_pos_w[:, self._fuse_body_idx, :]
            out_of_bounds = torch.logical_or(
                current_pos[:, 2] < 1.0,
                current_pos[:, 2] > 5.0,
            )
        else:
            # Fallback if indices not cached yet - no out of bounds termination
            out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Combined termination condition
        terminated = out_of_bounds | hover_success
        
        return terminated, time_out

    def _sample_goals_in_annulus(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample goal positions in an annulus pattern.
        
        Args:
            env_ids: Environment indices to sample goals for
            
        Returns:
            Goal positions in world coordinates [num_envs, 3]
        """
        num_envs_to_reset = len(env_ids)
        
        # Sample radius uniformly in annulus (0.4 to 1.75 meters)
        r_min, r_max = 0.4, 1.75
        # For uniform distribution in annulus, sample r^2 uniformly then take sqrt
        r_squared = torch.rand(num_envs_to_reset, device=self.device) * (r_max**2 - r_min**2) + r_min**2
        r = torch.sqrt(r_squared)
        
        # Sample theta uniformly (0 to 180 degrees = 0 to pi radians)
        theta = torch.rand(num_envs_to_reset, device=self.device) * math.pi
        
        # Convert polar to cartesian coordinates
        x_goals = r * torch.cos(theta)  # X coordinate
        y_goals = torch.zeros(num_envs_to_reset, device=self.device)  # Y = 0 (2D plane)
        z_goals = r * torch.sin(theta)  # Z coordinate (height)
        
        # Create goal positions relative to environment origins
        goals_relative = torch.stack([x_goals, y_goals, z_goals], dim=1)
        
        # Add environment origins to get world coordinates
        goals_world = goals_relative + self.scene.env_origins[env_ids]
        
        return goals_world

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the specified environments and generate new target positions."""
        # Ensure indices are cached
        self._cache_body_joint_indices()
        
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        elif isinstance(env_ids, (list, tuple)):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Logging - only if indices are available
        if hasattr(self, '_fuse_body_idx'):
            final_distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[env_ids] - self._robot.data.body_pos_w[env_ids, self._fuse_body_idx, :], dim=1
            ).mean()
        else:
            # Fallback if indices not cached yet
            final_distance_to_goal = torch.tensor(0.0, device=self.device)
        
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Episode_Termination/hover_success"] = torch.count_nonzero(self._is_hovering[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        
        # Debug: Check position variance and goal diversity
        if len(env_ids) >= 4 and hasattr(self, '_fuse_body_idx'):  # Only if we have enough environments and cached indices
            current_positions = self._robot.data.body_pos_w[env_ids, self._fuse_body_idx, :]
            goal_positions = self._desired_pos_w[env_ids]
            env_origins = self.scene.env_origins[env_ids]
            
            # Calculate positions relative to each robot's local origin
            relative_robot_positions = current_positions - env_origins
            relative_goal_positions = goal_positions - env_origins
            
            # Check for invalid positions
            pos_has_nan = torch.any(torch.isnan(relative_robot_positions))
            pos_has_inf = torch.any(torch.isinf(relative_robot_positions))
            
            # Only calculate variance if positions are valid
            if not pos_has_nan and not pos_has_inf:
                # Variance in relative positions (what we actually care about)
                pos_variance_x = torch.var(relative_robot_positions[:, 0]).item()
                pos_variance_z = torch.var(relative_robot_positions[:, 2]).item()
                goal_variance_x = torch.var(relative_goal_positions[:, 0]).item()
                goal_variance_z = torch.var(relative_goal_positions[:, 2]).item()
                
                extras["Debug/robot_relative_variance_x"] = pos_variance_x
                extras["Debug/robot_relative_variance_z"] = pos_variance_z
                extras["Debug/goal_relative_variance_x"] = goal_variance_x
                extras["Debug/goal_relative_variance_z"] = goal_variance_z
            else:
                extras["Debug/robot_relative_variance_x"] = float('nan')
                extras["Debug/robot_relative_variance_z"] = float('nan')
        
        self.extras["log"].update(extras)

        # Reset robot state using tensor format
        self._robot.reset(env_ids)  # type: ignore
        super()._reset_idx(env_ids)  # type: ignore
        
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset actions and hover tracking
        self._actions[env_ids] = 0.0
        self._hover_time[env_ids] = 0.0
        self._is_hovering[env_ids] = False
        
        # Reset progress tracking for goal seeking
        self._previous_distance_to_goal[env_ids] = float('inf')
        
        # Reset acceleration tracking
        self._previous_velocity[env_ids] = 0.0

        # Sample new desired positions using annulus pattern
        self._desired_pos_w[env_ids] = self._sample_goals_in_annulus(env_ids)
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Write to simulation using tensor indices
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)  # type: ignore
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)  # type: ignore
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)  # type: ignore

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up or disable debug visualization."""
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/goal_position",
                    markers={
                        "cuboid": sim_utils.CuboidCfg(
                            size=(0.05, 0.05, 0.05),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        ),
                    }
                )
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualizations."""
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

