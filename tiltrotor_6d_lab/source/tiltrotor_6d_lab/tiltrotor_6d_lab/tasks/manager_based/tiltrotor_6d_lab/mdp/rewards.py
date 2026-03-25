# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _xy_distance_to_command(env, command_name: str) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    asset: RigidObject = env.scene["robot"]
    return torch.norm(command_term.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)


def _tilt_angle(asset: RigidObject) -> torch.Tensor:
    gravity_b = asset.data.projected_gravity_b
    lateral_gravity = torch.norm(gravity_b[:, [0, 2]], dim=1)
    vertical_gravity = torch.clamp(gravity_b[:, 1].abs(), min=1e-6)
    return torch.atan2(lateral_gravity, vertical_gravity)


def flat_orientation_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from level using the imported asset's local vertical axis."""
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b
    return torch.sum(torch.square(gravity_b[:, [0, 2]]), dim=1)


def takeoff_height_tanh(
    env,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward climbing away from the ground during takeoff."""
    asset: RigidObject = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return 1.0 - torch.tanh(torch.clamp(target_height - height, min=0.0) / std)


def xy_command_error_tanh(env, std: float, command_name: str = "landing_target") -> torch.Tensor:
    """Reward reducing horizontal error to the landing target."""
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :2], dim=1)
    return 1.0 - torch.tanh(distance / std)


def heading_command_error_abs(env, command_name: str = "landing_target") -> torch.Tensor:
    """Penalize heading error relative to the landing target."""
    command = env.command_manager.get_command(command_name)
    return command[:, 3].abs()


def airborne_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward being clear of the ground with no fuselage contact."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_force = torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1)
    has_contact = torch.any(torch.max(contact_force, dim=1)[0] > threshold, dim=1)
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return ((height >= min_height) & (~has_contact)).float()


def liftoff_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward clearly lifting off the pad with the fuselage fully clear."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_force = torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1)
    has_contact = torch.any(torch.max(contact_force, dim=1)[0] > threshold, dim=1)
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return ((height >= min_height) & (~has_contact)).float()


class goal_progress_reward(ManagerTermBase):
    """Reward reduction in horizontal distance to the landing target."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_distance = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids):
        if env_ids is None:
            env_ids = slice(None)
        self._prev_distance[env_ids] = _xy_distance_to_command(
            self._env, self.cfg.params["command_name"]
        )[env_ids]

    def __call__(self, env: ManagerBasedRLEnv, command_name: str = "landing_target") -> torch.Tensor:
        distance = _xy_distance_to_command(env, command_name)
        progress = self._prev_distance - distance
        self._prev_distance[:] = distance
        return progress


def ground_slide_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal sliding while the fuselage is in ground contact."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_force = torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1)
    has_contact = torch.any(torch.max(contact_force, dim=1)[0] > threshold, dim=1)
    horizontal_speed = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    return has_contact.float() * horizontal_speed


def stationary_penalty(
    env,
    min_speed: float,
    activation_distance: float,
    command_name: str = "landing_target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize sitting nearly motionless while still far from the goal."""
    asset: RigidObject = env.scene[asset_cfg.name]
    distance = _xy_distance_to_command(env, command_name)
    horizontal_speed = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    far_from_goal = distance > activation_distance
    speed_deficit = torch.clamp(min_speed - horizontal_speed, min=0.0) / min_speed
    return far_from_goal.float() * speed_deficit


def final_approach_reward(
    env,
    activation_distance: float,
    max_lin_speed: float,
    max_ang_speed: float,
    command_name: str = "landing_target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward controlled slow flight once the aircraft is near the landing target."""
    asset: RigidObject = env.scene[asset_cfg.name]
    distance = _xy_distance_to_command(env, command_name)
    lin_speed = torch.norm(asset.data.root_lin_vel_w, dim=1)
    ang_speed = torch.norm(asset.data.root_ang_vel_b, dim=1)

    near_goal = distance <= activation_distance
    speed_score = torch.clamp(1.0 - lin_speed / max_lin_speed, min=0.0)
    ang_score = torch.clamp(1.0 - ang_speed / max_ang_speed, min=0.0)
    return near_goal.float() * speed_score * ang_score


def nacelle_asymmetry_l1(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize large left-right nacelle angle differences."""
    asset: Articulation = env.scene[asset_cfg.name]
    nacelle_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.abs(nacelle_pos[:, 0] - nacelle_pos[:, 1])


def valid_fuselage_touchdown_bonus(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    landing_radius: float,
    max_tilt: float,
    max_lin_speed: float,
    max_ang_speed: float,
    touchdown_height: float,
    command_name: str = "landing_target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward high-quality touchdown, scaled by speed and heading precision."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_force = torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1)
    has_contact = torch.any(torch.max(contact_force, dim=1)[0] > threshold, dim=1)

    command_term = env.command_manager.get_term(command_name)
    landing_distance = torch.norm(command_term.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)
    lin_speed = torch.norm(asset.data.root_lin_vel_w, dim=1)
    ang_speed = torch.norm(asset.data.root_ang_vel_b, dim=1)
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    stable_touchdown = _tilt_angle(asset) <= max_tilt
    heading_error = heading_command_error_abs(env, command_name)

    valid = has_contact
    valid &= landing_distance <= landing_radius
    valid &= stable_touchdown
    valid &= lin_speed <= max_lin_speed
    valid &= ang_speed <= max_ang_speed
    valid &= height <= touchdown_height

    distance_score = torch.clamp(1.0 - landing_distance / landing_radius, min=0.0)
    lin_speed_score = torch.clamp(1.0 - lin_speed / max_lin_speed, min=0.0)
    ang_speed_score = torch.clamp(1.0 - ang_speed / max_ang_speed, min=0.0)
    heading_score = torch.clamp(1.0 - heading_error / torch.pi, min=0.0)
    quality = distance_score * lin_speed_score * ang_speed_score * heading_score
    return valid.float() * quality


def landing_zone_violation(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    takeoff_radius: float,
    landing_radius: float,
    max_tilt: float,
    command_name: str = "landing_target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize fuselage contacts that occur outside valid touchdown zones."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_force = torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1)
    has_contact = torch.any(torch.max(contact_force, dim=1)[0] > threshold, dim=1)

    takeoff_distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    command_term = env.command_manager.get_term(command_name)
    landing_distance = torch.norm(command_term.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)

    valid_zone = (takeoff_distance <= takeoff_radius) | (landing_distance <= landing_radius)
    stable_touchdown = _tilt_angle(asset) <= max_tilt
    return (has_contact & (~valid_zone | ~stable_touchdown)).float()


def body_linear_acceleration_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize linear acceleration magnitude for the selected bodies."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1)), dim=1)


def agl_height_shaped(
    env,
    scale_m: float = 3.0,
    cap_m: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Nonnegative reward increasing with AGL; always rewards being higher (within cap)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    h = torch.clamp(asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2], min=0.0, max=cap_m)
    return h / scale_m


def upright_projection_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sharpness: float = 5.0,
) -> torch.Tensor:
    """In (0, 1]: 1 when level (matches flat_orientation basis: small gx, gz)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b
    lat = torch.sum(torch.square(gravity_b[:, [0, 2]]), dim=1)
    return torch.exp(-sharpness * lat)


def flight_rotor_rpm_differential_squared(
    env,
    action_term_name: str = "flight",
    rpm_scale: float = 1600.0,
) -> torch.Tensor:
    """Penalize differential thrust command (left vs right RPM targets), normalized."""
    term = env.action_manager.get_term(action_term_name)
    pa = term.processed_actions
    rpm = pa[:, -2:]
    d = (rpm[:, 0] - rpm[:, 1]) / rpm_scale
    return d * d


def flight_nacelle_command_differential_squared(
    env,
    action_term_name: str = "flight",
) -> torch.Tensor:
    """Penalize left/right nacelle angle command mismatch (processed joint targets, rad)."""
    term = env.action_manager.get_term(action_term_name)
    pa = term.processed_actions
    n_nac = pa.shape[1] - 2
    q = pa[:, :n_nac]
    d = q[:, 0] - q[:, 1]
    return d * d


def ang_vel_l2_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared angular velocity magnitude (body frame) — stricter low-rate flight."""
    asset: RigidObject = env.scene[asset_cfg.name]
    w = asset.data.root_ang_vel_b
    return torch.sum(w * w, dim=1)
