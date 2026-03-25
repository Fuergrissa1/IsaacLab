# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Hover / takeoff curriculum: short horizons, goal at spawn."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .tiltrotor_6d_lab_env_cfg import (
    ActionsCfg,
    EventCfg,
    ObservationsCfg,
    Tiltrotor6dLabSceneCfg,
)


@configclass
class HoverCommandsCfg:
    """Landing target fixed at environment origin (hover over pad)."""

    landing_target = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(30.0, 30.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class HoverRewardsCfg:
    """Hover curriculum: always reward AGL + survival signals; penalize asymmetry in commands."""

    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    agl_height = RewTerm(
        func=mdp.agl_height_shaped,
        weight=1.5,
        params={"scale_m": 3.0, "cap_m": 5.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    upright = RewTerm(
        func=mdp.upright_projection_reward,
        weight=2.0,
        params={"sharpness": 5.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    airborne = RewTerm(
        func=mdp.airborne_reward,
        weight=5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Fusilage"]),
            "threshold": 1.0,
            "min_height": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    liftoff = RewTerm(
        func=mdp.liftoff_reward,
        weight=10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Fusilage"]),
            "threshold": 1.0,
            "min_height": 0.8,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    fail = RewTerm(
        func=mdp.is_terminated_term,
        weight=-4.0,
        params={
            "term_keys": [
                "illegal_contact",
                "invalid_fuselage_contact",
                "excessive_motion",
                "excessive_acceleration",
                "crash",
            ]
        },
    )
    takeoff_height = RewTerm(
        func=mdp.takeoff_height_tanh,
        weight=6.0,
        params={"target_height": 1.5, "std": 0.5},
    )
    vertical_speed = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    angular_rate = RewTerm(
        func=mdp.ang_vel_l2_penalty,
        weight=-0.06,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.12,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    fuselage_acceleration = RewTerm(
        func=mdp.body_linear_acceleration_l2,
        weight=-0.008,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Fusilage"])},
    )
    illegal_contact = RewTerm(
        func=mdp.illegal_contact,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "Nacelle_Left",
                    "Nacelle_Right",
                    "Rotor_Left",
                    "Rotor_Right",
                    "Aileron_Left",
                    "Aileron_Right",
                    "Ruddervator_Left",
                    "Ruddervator_Right",
                ],
            ),
            "threshold": 1.0,
        },
    )
    ground_slide = RewTerm(
        func=mdp.ground_slide_penalty,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Fusilage"]),
            "threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    rotor_differential = RewTerm(
        func=mdp.flight_rotor_rpm_differential_squared,
        weight=-0.02,
        params={"action_term_name": "flight", "rpm_scale": 1600.0},
    )
    nacelle_differential = RewTerm(
        func=mdp.flight_nacelle_command_differential_squared,
        weight=-0.12,
        params={"action_term_name": "flight"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.001)


@configclass
class HoverTerminationsCfg:
    """Same safety terminations; success is unlikely while goal is at origin."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "Nacelle_Left",
                    "Nacelle_Right",
                    "Rotor_Left",
                    "Rotor_Right",
                    "Aileron_Left",
                    "Aileron_Right",
                    "Ruddervator_Left",
                    "Ruddervator_Right",
                ],
            ),
            "threshold": 1.0,
        },
    )
    invalid_fuselage_contact = DoneTerm(
        func=mdp.invalid_fuselage_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Fusilage"]),
            "threshold": 1.0,
            "takeoff_radius": 1.0,
            "landing_radius": 2.0,
            "max_tilt": 0.35,
            "command_name": "landing_target",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    excessive_motion = DoneTerm(
        func=mdp.excessive_root_motion,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_lin_speed": 18.0,
            "max_ang_speed": 12.0,
        },
    )
    excessive_acceleration = DoneTerm(
        func=mdp.excessive_body_linear_acceleration,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Fusilage"]),
            "limit": 75.0,
        },
    )
    crash = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.1},
    )
    success = DoneTerm(
        func=mdp.successful_landing,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Fusilage"]),
            "threshold": 1.0,
            "landing_radius": 2.0,
            "touchdown_height": 1.0,
            "max_tilt": 0.2,
            "max_lin_speed": 1.0,
            "max_ang_speed": 1.0,
            "command_name": "landing_target",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class Tiltrotor6dHoverEnvCfg(ManagerBasedRLEnvCfg):
    """Short episodes; command at origin; no navigation shaping."""

    scene: Tiltrotor6dLabSceneCfg = Tiltrotor6dLabSceneCfg(
        num_envs=64,
        env_spacing=100.0,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: HoverCommandsCfg = HoverCommandsCfg()
    rewards: HoverRewardsCfg = HoverRewardsCfg()
    terminations: HoverTerminationsCfg = HoverTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.scene.contact_forces.update_period = self.sim.dt
