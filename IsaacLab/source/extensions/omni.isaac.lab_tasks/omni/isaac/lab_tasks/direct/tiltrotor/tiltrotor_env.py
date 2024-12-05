from __future__ import annotations
import math
import torch

import gymnasium as gym

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms, sample_uniform

# Ensure that TILTROTOR_CFG and CUBOID_MARKER_CFG are correctly defined and imported
from omni.isaac.lab_assets import TILTROTOR_CFG
from omni.isaac.lab.markers import CUBOID_MARKER_CFG


class TiltrotorEnvWindow(BaseEnvWindow):
    """Window manager for the Tiltrotor environment."""

    def __init__(self, env: TiltrotorEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # Initialize base window
        super().__init__(env, window_name)
        # Add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # Add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class TiltrotorEnvCfg(DirectRLEnvCfg):
    # Environment Configuration
    episode_length_s: float = 10.0
    decimation: int = 2
    action_space: int = 2
    observation_space: int = 10
    state_space: int = 0  # Corrected to 0 to avoid negative dimensions
    debug_vis: bool = True

    ui_window_class_type = TiltrotorEnvWindow

    # Simulation Configuration
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # Robot Configuration
    robot: ArticulationCfg = TILTROTOR_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_scale: float = 10.0  
    angle_limit: float = math.pi / 4  

    # Reward Scales
    ang_vel_reward_scale: float = -0.05
    distance_to_goal_reward_scale: float = -0.1
    orientation_penalty_scale: float = -0.1
    joint_limit_penalty_scale: float = -1


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

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "ang_vel",
                "distance_to_goal",
                "orientation_penalty",
                "limit_contact_penalty",
            ]
        }

        # Retrieve body and joint indices
        self._rotor_id = self._robot.find_bodies("rotor")[0]        
        self._revolute0_id = self._robot.find_joints("revolute0")[0]
        self._revolute1_id = self._robot.find_joints("revolute1")[0]
        self._revolute2_id = self._robot.find_joints("revolute2")[0]
        self._revolute3_id = self._robot.find_joints("revolute3")[0]

        joint_names = self._robot.data.joint_names
        self._revolute_joint_indices = [joint_names.index(name) for name in ["revolute0", "revolute1", "revolute3"]]
        joint_limits = self._robot.data.joint_limits  # Shape: (num_envs, num_joints, 2)
        self._joint_lower_limits = joint_limits[:, self._revolute_joint_indices, 0]  # Shape: (num_envs, num_joints)
        self._joint_upper_limits = joint_limits[:, self._revolute_joint_indices, 1]  # Shape: (num_envs, num_joints)

        # Add handle for debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # Initialize extras dictionary
        self.extras = {}

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions() # No collisions - even w/ ground plane
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.env_origins[:, 2] += 2.5  # Raise the environments by 1.0 units

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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
        # Extract joint angles and angular velocities using joint indices
        theta1 = self._robot.data.joint_pos[:, self._revolute0_id]
        theta2 = self._robot.data.joint_pos[:, self._revolute1_id]
        omega1 = self._robot.data.joint_vel[:, self._revolute0_id]
        omega2 = self._robot.data.joint_vel[:, self._revolute1_id]


        # Link length
        L = 1.03

        # Calculate positions based on joint angles
        x1 = L * torch.cos(theta1)
        z1 = L * torch.sin(theta1)
        x2 = x1 + L * torch.cos(theta1 + theta2)
        z2 = z1 + L * torch.sin(theta1 + theta2)
        y2 = torch.zeros_like(x2)  # Assuming movement in XZ plane

        # Fuse position relative to environment origin
        fuse_pos = torch.cat([x2, y2, z2], dim=1)  # Shape: (num_envs, 3)
        env_origins = self.scene.env_origins  # Shape: (num_envs, 3)
        fuse_pos_relative = fuse_pos + env_origins

        # Desired position relative to robot
        body_names = self._robot.data.body_names
        body_idx = body_names.index("fuse")
        goal_vector_relative = self._desired_pos_w - self._robot.data.body_pos_w[:, body_idx, :]

        # Concatenate observations
        obs = torch.cat(
            [
                theta1,
                theta2,
                omega1,
                omega2,
                fuse_pos_relative,
                goal_vector_relative,
            ],
            dim=-1,
        )  # Shape: (num_envs, 10)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute the reward for each environment."""
        # Distance to goal
        body_names = self._robot.data.body_names
        body_idx = body_names.index("fuse")
        distance_to_goal = torch.norm(self._desired_pos_w - self._robot.data.body_pos_w[:, body_idx, :])
        distance_reward = distance_to_goal * self.cfg.distance_to_goal_reward_scale * self.step_dt

        # Orientation penalty
        fuse_quat = self._robot.data.body_quat_w[:, body_idx, :]  # Shape: (num_envs, 4)
        fuse_quat = fuse_quat / fuse_quat.norm(dim=-1, keepdim=True)  # Normalize quaternion
        # Up vector in body frame
        up_vector_body = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32)
        up_vector_body = up_vector_body.unsqueeze(0).expand(self.num_envs, 3)
        up_vector_world = self._quaternion_apply(fuse_quat, up_vector_body)
        world_up_vector = up_vector_body
        dot_product = (up_vector_world * world_up_vector).sum(dim=1)
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        orientation_penalty = (1.0 - dot_product) * self.cfg.orientation_penalty_scale * self.step_dt

        # Joint Limit Penalty
        joint_positions = self._robot.data.joint_pos[:, self._revolute_joint_indices]  # Shape: (num_envs, num_joints)
        lower_limit_dist = joint_positions - self._joint_lower_limits  # Distance from lower limit
        upper_limit_dist = self._joint_upper_limits - joint_positions  # Distance from upper limit
        min_limit_dist = torch.min(lower_limit_dist, upper_limit_dist)  # Shape: (num_envs, num_joints)
        threshold = 0.1  # Adjust as needed (in radians)
        limit_penalty = torch.clamp(threshold - min_limit_dist, min=0.0) / threshold  # Normalize to [0,1]
        joint_limit_penalty = limit_penalty.sum(dim=1) * self.cfg.joint_limit_penalty_scale * self.step_dt


        # Angular velocity penalty
        joint_names = self._robot.data.joint_names
        desired_joints = ["revolute0", "revolute1"]
        joint_idx = [joint_names.index(joint) for joint in desired_joints]
        ang_vel = torch.sum(self._robot.data.joint_vel[:, joint_idx] ** 2, dim=1)
        ang_vel_penalty = ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt

        # Total reward
        reward = distance_reward + ang_vel_penalty + joint_limit_penalty

        # Logging
        self._episode_sums["distance_to_goal"] += distance_reward
        self._episode_sums["ang_vel"] += ang_vel_penalty
        self._episode_sums["orientation_penalty"] += orientation_penalty
        self._episode_sums["limit_contact_penalty"] += joint_limit_penalty
        

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
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        body_names = self._robot.data.body_names
        body_idx = body_names.index("fuse")
        out_of_bounds = torch.logical_or(
            self._robot.data.body_pos_w[:, body_idx, 2] < 1.0,
            self._robot.data.body_pos_w[:, body_idx, 2] > 5.0,
        )
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the specified environments and generate new target positions."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        body_names = self._robot.data.body_names
        body_idx = body_names.index("fuse")
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.body_pos_w[env_ids, body_idx, :], dim=1
        ).mean()
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
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # Sample new desired positions
        self._desired_pos_w[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0  # Uniform between -1.0 and 1.0
        self._desired_pos_w[env_ids, 1] = 0.0  # y-coordinate
        self._desired_pos_w[env_ids, 2] = torch.rand(len(env_ids), device=self.device) + 0.5  # Uniform between 0.5 and 1.5
        self._desired_pos_w[env_ids] += self.scene.env_origins[env_ids]

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Write to simulation
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)      

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up or disable debug visualization."""
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualizations."""
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

