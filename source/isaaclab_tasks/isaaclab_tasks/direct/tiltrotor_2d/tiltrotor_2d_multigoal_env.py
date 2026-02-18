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

# Import base environment and configuration
from .tiltrotor_2d_env import TiltrotorEnv, TiltrotorEnvWindow
from .tiltrotor_2d_multigoal_cfg import TiltrotorMultiGoalEnvCfg


class TiltrotorMultiGoalEnv(TiltrotorEnv):
    """Multi-goal tiltrotor environment for consecutive goal seeking behavior.
    
    Inherits joint limit modifications from base TiltrotorEnv class.
    revolute2 joint limits are automatically set to -15° to +150°.
    """
    
    cfg: TiltrotorMultiGoalEnvCfg

    def __init__(self, cfg: TiltrotorMultiGoalEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Multi-goal tracking tensors
        self._current_goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._goals_completed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_goals = torch.zeros(
            self.num_envs, self.cfg.num_goals_per_episode, 3, device=self.device
        )
        
        # Goal completion tracking
        self._goal_hover_time = torch.zeros(self.num_envs, device=self.device)
        self._goal_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Goal timeout tracking
        self._goal_start_time = torch.zeros(self.num_envs, device=self.device)
        
        # Add goal transition tracking to episode sums
        self._episode_sums["goal_transition_bonus"] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def _get_observations(self) -> dict:
        """Generate observations using current goal from sequence."""
        # Get base observations
        obs_dict = super()._get_observations()
        base_obs = obs_dict["policy"]
        
        # Override goal vector with current goal from sequence
        self._cache_body_joint_indices()
        if hasattr(self, '_fuse_body_idx'):
            current_goals = self._episode_goals[torch.arange(self.num_envs), self._current_goal_idx]
            fuse_pos_w = self._robot.data.body_pos_w[:, self._fuse_body_idx, :]
            goal_vector_relative = current_goals - fuse_pos_w
            
            # Replace the last 3 elements (goal vector) in base observations
            obs_dict["policy"] = torch.cat([
                base_obs[:, :-3],  # Everything except last 3 (old goal vector)
                goal_vector_relative  # New goal vector to current goal
            ], dim=-1)
        
        return obs_dict

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards with multi-goal transition bonuses."""
        # Temporarily override desired position to use current goal
        original_desired_pos = self._desired_pos_w.clone()
        current_goals = self._episode_goals[torch.arange(self.num_envs), self._current_goal_idx]
        self._desired_pos_w = current_goals
        
        # Get base reward using current goal
        base_reward = super()._get_rewards()
        
        # Restore original desired position
        self._desired_pos_w = original_desired_pos
        
        # Add goal transition detection and bonus
        goal_transition_bonus = torch.zeros_like(base_reward)
        
        # Check for goal completion
        self._cache_body_joint_indices()
        if hasattr(self, '_fuse_body_idx'):
            current_pos = self._robot.data.body_pos_w[:, self._fuse_body_idx, :]
            current_vel = self._robot.data.body_lin_vel_w[:, self._fuse_body_idx, :]
            current_ang_vel = self._robot.data.body_ang_vel_w[:, self._fuse_body_idx, :]
            
            # Distance to current goal
            distance_to_current_goal = torch.norm(current_goals - current_pos, dim=1)
            velocity_magnitude = torch.norm(current_vel, dim=1)
            angular_velocity_magnitude = torch.norm(current_ang_vel, dim=1)
            
            # Check if hovering at current goal
            at_goal = distance_to_current_goal < self.cfg.hover_position_tolerance
            low_velocity = velocity_magnitude < self.cfg.hover_velocity_tolerance
            low_angular_velocity = angular_velocity_magnitude < self.cfg.hover_angular_velocity_tolerance
            is_hovering_at_goal = at_goal & low_velocity & low_angular_velocity
            
            # Update goal hover time
            self._goal_hover_time = torch.where(
                is_hovering_at_goal,
                self._goal_hover_time + self.step_dt,
                torch.zeros_like(self._goal_hover_time)
            )
            
            # Update goal start time (time since current goal became active)
            self._goal_start_time += self.step_dt
            
            # Check for goal completion (hover condition met)
            goal_completed_by_hover = (
                (self._goal_hover_time >= self.cfg.goal_completion_time) & 
                ~self._goal_completed
            )
            
            # Check for goal timeout (5 seconds elapsed without completion)
            goal_timed_out = (
                (self._goal_start_time >= self.cfg.goal_timeout) &
                ~self._goal_completed
            )
            
            # Award transition bonus only for hover completion (not timeout)
            goal_transition_bonus = torch.where(
                goal_completed_by_hover,
                torch.full_like(goal_completed_by_hover, self.cfg.goal_transition_bonus, dtype=torch.float32),
                torch.zeros_like(goal_completed_by_hover, dtype=torch.float32)
            )
            
            # Mark goals as completed and transition to next goal (either by hover or timeout)
            goal_just_completed = goal_completed_by_hover | goal_timed_out
            self._goal_completed |= goal_just_completed
            self._goals_completed += goal_just_completed.long()
            
            # Transition to next goal for completed environments
            next_goal_available = self._current_goal_idx < (self.cfg.num_goals_per_episode - 1)
            should_transition = goal_just_completed & next_goal_available
            
            self._current_goal_idx = torch.where(
                should_transition,
                self._current_goal_idx + 1,
                self._current_goal_idx
            )
            
            # Reset completion tracking for new goal
            self._goal_completed = torch.where(
                should_transition,
                torch.zeros_like(self._goal_completed),
                self._goal_completed
            )
            self._goal_hover_time = torch.where(
                should_transition,
                torch.zeros_like(self._goal_hover_time),
                self._goal_hover_time
            )
            # Reset goal start time for new goal
            self._goal_start_time = torch.where(
                should_transition,
                torch.zeros_like(self._goal_start_time),
                self._goal_start_time
            )
        
        # Add to episode sums
        self._episode_sums["goal_transition_bonus"] += goal_transition_bonus
        
        return base_reward + goal_transition_bonus

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Modified termination - only end on crashes or all goals completed."""
        # Get base termination conditions (time out)
        _, time_out = super()._get_dones()
        
        # Check for crashes/out of bounds only (remove hover success termination)
        self._cache_body_joint_indices()
        if hasattr(self, '_fuse_body_idx'):
            current_pos = self._robot.data.body_pos_w[:, self._fuse_body_idx, :]
            crashed = torch.logical_or(
                current_pos[:, 2] < 1.0,  # Below minimum height
                current_pos[:, 2] > 5.0,  # Above maximum height
            )
        else:
            crashed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Check if all goals completed
        all_goals_completed = self._goals_completed >= self.cfg.num_goals_per_episode
        
        # Terminate on crash or all goals completed
        terminated = crashed | all_goals_completed
        
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments and generate new goal sequences."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        elif isinstance(env_ids, (list, tuple)):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Additional logging for multi-goal performance
        if len(env_ids) > 0:
            avg_goals_completed = self._goals_completed[env_ids].float().mean()
            completion_rate = (self._goals_completed[env_ids] >= self.cfg.num_goals_per_episode).float().mean()
            
            # Ensure extras structure exists
            if not hasattr(self, 'extras'):
                self.extras = {}
            if "log" not in self.extras:
                self.extras["log"] = {}
                
            self.extras["log"]["MultiGoal/avg_goals_completed"] = avg_goals_completed.item()
            self.extras["log"]["MultiGoal/completion_rate"] = completion_rate.item()
        
        # Reset multi-goal tracking
        self._current_goal_idx[env_ids] = 0
        self._goals_completed[env_ids] = 0
        self._goal_hover_time[env_ids] = 0.0
        self._goal_completed[env_ids] = False
        self._goal_start_time[env_ids] = 0.0
        
        # Generate new goal sequences for reset environments
        self._generate_goal_sequence(env_ids)
        
        # Call base reset 
        super()._reset_idx(env_ids)
        
        # Override the base's single goal with first goal from sequence
        self._desired_pos_w[env_ids] = self._episode_goals[env_ids, 0]

    def _generate_goal_sequence(self, env_ids: torch.Tensor):
        """Generate a sequence of goals in an annulus pattern (0.4-1.75m radius, 0-180 degrees)."""
        num_envs_to_reset = len(env_ids)
        
        for goal_idx in range(self.cfg.num_goals_per_episode):
            # Generate goals with proper spacing constraints
            for attempt in range(50):  # Max attempts to avoid infinite loops
                # Sample goals in annulus pattern
                # Sample radius uniformly in annulus (0.4 to 1.5 meters)
                r_min, r_max = 0.5, 1.5
                # For uniform distribution in annulus, sample r^2 uniformly then take sqrt
                r_squared = torch.rand(num_envs_to_reset, device=self.device) * (r_max**2 - r_min**2) + r_min**2
                r = torch.sqrt(r_squared)
                
                # Sample theta uniformly (0 to 180 degrees = 0 to pi radians)
                theta = torch.rand(num_envs_to_reset, device=self.device) * math.pi
                
                # Convert polar to cartesian coordinates
                x_goals = r * torch.cos(theta)  # X coordinate
                y_goals = torch.zeros(num_envs_to_reset, device=self.device)  # Y = 0 (2D plane)
                z_goals = r * torch.sin(theta)  # Z coordinate (height)
                
                new_goals = torch.stack([x_goals, y_goals, z_goals], dim=1)
                
                # Check minimum distance from previous goal
                if goal_idx > 0:
                    prev_goals = self._episode_goals[env_ids, goal_idx - 1] - self.scene.env_origins[env_ids]
                    distances_from_prev = torch.norm(new_goals - prev_goals, dim=1)
                    valid_separation = distances_from_prev >= self.cfg.min_goal_separation
                    
                    if valid_separation.all():
                        break
                    else:
                        # Regenerate only invalid goals
                        invalid_mask = ~valid_separation
                        num_invalid = invalid_mask.sum().item()
                        if num_invalid > 0:
                            # Regenerate invalid goals with annulus pattern
                            r_squared_new = torch.rand(num_invalid, device=self.device) * (r_max**2 - r_min**2) + r_min**2
                            r_new = torch.sqrt(r_squared_new)
                            theta_new = torch.rand(num_invalid, device=self.device) * math.pi
                            
                            new_goals[invalid_mask, 0] = r_new * torch.cos(theta_new)  # X
                            new_goals[invalid_mask, 2] = r_new * torch.sin(theta_new)  # Z
                else:
                    break  # First goal, no previous goal to check
            
            # Add environment origins and store
            self._episode_goals[env_ids, goal_idx] = new_goals + self.scene.env_origins[env_ids]

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Enhanced debug visualization for multi-goal environment."""
        super()._set_debug_vis_impl(debug_vis)
        
        if debug_vis:
            # Disable cyan markers for cleaner visualization
            # if not hasattr(self, "all_goals_visualizer"):
            #     marker_cfg = VisualizationMarkersCfg(
            #         prim_path="/Visuals/Command/all_goals",
            #         markers={
            #             "sphere": sim_utils.SphereCfg(
            #                 radius=0.03,
            #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            #             ),
            #         }
            #     )
            #     self.all_goals_visualizer = VisualizationMarkers(marker_cfg)
            # self.all_goals_visualizer.set_visibility(True)
            
            # Current goal marker (yellow) - keep this one
            if hasattr(self, "goal_pos_visualizer"):
                # Override with yellow marker for current goal
                current_goal_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/current_goal",
                    markers={
                        "cuboid": sim_utils.CuboidCfg(
                            size=(0.07, 0.07, 0.07),  # Slightly larger for current goal
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                        ),
                    }
                )
                self.goal_pos_visualizer = VisualizationMarkers(current_goal_cfg)
        # else:
        #     if hasattr(self, "all_goals_visualizer"):
        #         self.all_goals_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualizations for multi-goal."""
        if hasattr(self, "goal_pos_visualizer"):
            # Show current goals (yellow cubes)
            current_goals = self._episode_goals[torch.arange(self.num_envs), self._current_goal_idx]
            self.goal_pos_visualizer.visualize(current_goals)
        
        # Disabled cyan spheres for cleaner visualization
        # if hasattr(self, "all_goals_visualizer"):
        #     # Show all goals as small cyan spheres
        #     all_goals_flat = self._episode_goals.reshape(-1, 3)
        #     self.all_goals_visualizer.visualize(all_goals_flat) 