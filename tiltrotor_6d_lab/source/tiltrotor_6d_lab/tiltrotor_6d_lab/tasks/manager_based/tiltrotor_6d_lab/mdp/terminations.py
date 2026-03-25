from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _body_has_contact(contact_sensor: ContactSensor, body_ids, threshold: float) -> torch.Tensor:
    contact_force = torch.norm(contact_sensor.data.net_forces_w_history[:, :, body_ids], dim=-1)
    return torch.any(torch.max(contact_force, dim=1)[0] > threshold, dim=1)


def _tilt_angle(asset: RigidObject) -> torch.Tensor:
    gravity_b = asset.data.projected_gravity_b
    lateral_gravity = torch.norm(gravity_b[:, [0, 2]], dim=1)
    vertical_gravity = torch.clamp(gravity_b[:, 1].abs(), min=1e-6)
    return torch.atan2(lateral_gravity, vertical_gravity)


def invalid_fuselage_contact(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    takeoff_radius: float,
    landing_radius: float,
    max_tilt: float,
    command_name: str = "landing_target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate on fuselage contact outside the takeoff/landing zones or with too much tilt."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    has_contact = _body_has_contact(contact_sensor, sensor_cfg.body_ids, threshold)
    takeoff_distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    command_term = env.command_manager.get_term(command_name)
    landing_distance = torch.norm(command_term.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)
    stable_touchdown = _tilt_angle(asset) <= max_tilt
    valid_zone = (takeoff_distance <= takeoff_radius) | (landing_distance <= landing_radius)
    return has_contact & (~valid_zone | ~stable_touchdown)


def successful_landing(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    landing_radius: float,
    touchdown_height: float,
    max_tilt: float,
    max_lin_speed: float,
    max_ang_speed: float,
    command_name: str = "landing_target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate successfully after a stable touchdown in the landing zone."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    has_contact = _body_has_contact(contact_sensor, sensor_cfg.body_ids, threshold)
    command_term = env.command_manager.get_term(command_name)
    landing_distance = torch.norm(command_term.pos_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    lin_speed = torch.norm(asset.data.root_lin_vel_w, dim=1)
    ang_speed = torch.norm(asset.data.root_ang_vel_b, dim=1)
    stable_touchdown = _tilt_angle(asset) <= max_tilt

    success = has_contact
    success &= landing_distance <= landing_radius
    success &= stable_touchdown
    success &= lin_speed <= max_lin_speed
    success &= ang_speed <= max_ang_speed
    success &= height <= touchdown_height
    return success


def excessive_body_linear_acceleration(
    env,
    limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when selected bodies exceed the allowed acceleration magnitude."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_acc = torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1)
    return torch.any(body_acc > limit, dim=1)


def excessive_root_motion(
    env,
    max_lin_speed: float,
    max_ang_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when root motion exceeds safety limits."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_speed = torch.norm(asset.data.root_lin_vel_w, dim=1)
    ang_speed = torch.norm(asset.data.root_ang_vel_b, dim=1)
    return (lin_speed > max_lin_speed) | (ang_speed > max_ang_speed)
