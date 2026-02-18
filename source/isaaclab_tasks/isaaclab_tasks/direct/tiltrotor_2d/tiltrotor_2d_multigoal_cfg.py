from __future__ import annotations

from isaaclab.utils import configclass

from .tiltrotor_2d_env import TiltrotorEnvCfg


@configclass
class TiltrotorMultiGoalEnvCfg(TiltrotorEnvCfg):
    """Configuration for multi-goal tiltrotor environment."""
    
    # Override episode length
    episode_length_s: float = 30.0  # 30 seconds for multiple goals
    
    # Multi-goal specific settings
    num_goals_per_episode: int = 10  # Number of goals to complete per episode
    goal_transition_bonus: float = 500.0  # Large bonus for completing a goal
    goal_completion_time: float = 0.5  # Very fast transitions - only 0.5 seconds on target
    goal_timeout: float = 5.0  # Maximum time allowed per goal before auto-transition
    
    # Greatly relaxed tolerances for much easier goal completion
    hover_position_tolerance: float = 0.30  # Very relaxed - 30cm radius
    hover_velocity_tolerance: float = 1.5  # Greatly relaxed - allow substantial movement
    hover_angular_velocity_tolerance: float = 1.5  # Greatly relaxed - allow significant rotation
    
    # Goal spacing constraints (for sequential goal separation)
    min_goal_separation: float = 0.6  # Minimum distance between consecutive goals
    
    # Goal bounds: Uses annulus pattern for reachable area coverage
    # Radius: 0.4 to 1.75 meters from environment origin
    # Theta: 0 to 180 degrees (upper half circle, staying above base link)
    # Y: 0.0 (2D plane constraint)
    # Results in natural flight envelope matching tiltrotor capabilities
    
    # Success tracking
    episode_success_threshold: int = 10  # Complete all goals to succeed


# For now, use the base environment class with the multi-goal config
TiltrotorMultiGoalEnv = TiltrotorEnvCfg  # Will be extended later 