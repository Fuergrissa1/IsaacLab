from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def root_height(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return root height above the environment origin."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(1)


def landing_target_xy_heading(env, command_name: str = "landing_target") -> torch.Tensor:
    """Return the relative XY landing target and heading error."""
    command = env.command_manager.get_command(command_name)
    return torch.cat((command[:, :2], command[:, 3:4]), dim=1)


def feasible_nacelle_joint_clearance(
    env,
    asset_cfg: SceneEntityCfg,
    lever_arm_m: float | None = None,
    hub_to_rotor_plane_m: float = 0.245,
    rotor_radius_m: float = 0.75,
    height_offset_m: float = 0.0,
    q_min_mech: float | None = None,
    q_max_mech: float | None = None,
    low_altitude_m: float = 1.0,
    low_altitude_max_forward_q: float = -math.pi / 6.0,
) -> torch.Tensor:
    """Conservative joint bounds (rad) vs altitude so rotor tips stay clear of the ground.

    Lever arm defaults to hub offset (245 mm) + rotor radius (750 mm) for worst-case tip reach.

    Uses a 2D sagittal approximation: from vertical (q = -pi/2), bound max forward q with
    q <= -pi/2 + atan2(h, L); over-vertical symmetrically. Clamped to mechanical limits.

    Below ``low_altitude_m`` AGL, max forward tilt is further limited to ``low_altitude_max_forward_q``
    (default -pi/6 rad: user angle at least 30 deg from forward toward vertical, with 0 = forward).

    Returns shape (num_envs, 4): q_min_feas, q_max_feas, q_left, q_right.
    """
    if q_min_mech is None:
        q_min_mech = -7.0 * math.pi / 12.0
    if q_max_mech is None:
        q_max_mech = math.pi / 12.0

    l_m = float(lever_arm_m) if lever_arm_m is not None else (hub_to_rotor_plane_m + rotor_radius_m)

    asset: Articulation = env.scene[asset_cfg.name]
    h_raw = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] + height_offset_m
    h = torch.clamp(h_raw, min=1e-4)
    l = torch.full_like(h, l_m)
    span = torch.atan2(h, l)
    q_vert = -math.pi / 2.0
    q_max_geo = q_vert + span
    q_min_geo = q_vert - span
    q_max_feas = torch.minimum(torch.full_like(h, q_max_mech), q_max_geo)
    q_min_feas = torch.maximum(torch.full_like(h, q_min_mech), q_min_geo)

    low = h_raw < low_altitude_m
    cap = torch.full_like(q_max_feas, low_altitude_max_forward_q)
    q_max_feas = torch.where(low, torch.minimum(q_max_feas, cap), q_max_feas)

    names = asset_cfg.joint_names
    if names is None or len(names) < 2:
        names = ["Nacelle_Left_Hinge", "Nacelle_Right_Hinge"]
    jids, _ = asset.find_joints(names[:2], preserve_order=True)

    qj = asset.data.joint_pos[:, jids]

    return torch.cat(
        (
            q_min_feas.unsqueeze(1),
            q_max_feas.unsqueeze(1),
            qj[:, 0:1],
            qj[:, 1:2],
        ),
        dim=1,
    )
