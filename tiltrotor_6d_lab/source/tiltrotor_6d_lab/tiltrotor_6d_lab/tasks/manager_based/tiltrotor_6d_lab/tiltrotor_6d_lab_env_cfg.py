# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTermCfg as CommandTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from tiltrotor_6d_lab.assets import TILTROTOR_6D_CFG  # isort:skip

from . import mdp


##
# Scene definition
##


@configclass
class Tiltrotor6dLabSceneCfg(InteractiveSceneCfg):
    """Configuration for the Tiltrotor 6D scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(5000.0, 5000.0)),
    )

    # robot
    robot: ArticulationCfg = TILTROTOR_6D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        force_threshold=0.5,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    flight = mdp.TiltrotorThrustActionCfg(
        asset_name="robot",
        nacelle_joint_names=["Nacelle_Left_Hinge", "Nacelle_Right_Hinge"],
        rotor_joint_names=["Rotor_Left_Hinge", "Rotor_Right_Hinge"],
        rotor_body_names=["Rotor_Left", "Rotor_Right"],
        force_directions={
            "Rotor_Left": (0.0, 0.0, 1.0),
            "Rotor_Right": (0.0, 0.0, -1.0),
        },
        rotor_velocity_signs={
            "Rotor_Left_Hinge": -1.0,
            "Rotor_Right_Hinge": 1.0,
        },
        nacelle_scale=math.pi / 3.0,
        nacelle_offset=-math.pi / 4.0,
        nacelle_clip=(-7.0 * math.pi / 12.0, math.pi / 12.0),
        rpm_range=(0.0, 1600.0),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        # observation terms (order preserved)
        root_pos_w = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.01, n_max=0.01))
        root_quat_w = ObsTerm(
            func=mdp.root_quat_w,
            params={"make_quat_unique": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        root_height = ObsTerm(
            func=mdp.root_height,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        landing_target = ObsTerm(func=mdp.landing_target_xy_heading, params={"command_name": "landing_target"})
        feasible_nacelle = ObsTerm(
            func=mdp.feasible_nacelle_joint_clearance,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["Nacelle_Left_Hinge", "Nacelle_Right_Hinge"],
                ),
                "hub_to_rotor_plane_m": 0.245,
                "rotor_radius_m": 0.75,
                "height_offset_m": 0.0,
                "low_altitude_m": 1.0,
                "low_altitude_max_forward_q": -math.pi / 6.0,
            },
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    landing_target = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(60.0, 60.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(55.0, 80.0),
            pos_y=(-20.0, 20.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    airborne = RewTerm(
        func=mdp.airborne_reward,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Fusilage"]),
            "threshold": 1.0,
            "min_height": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    liftoff = RewTerm(
        func=mdp.liftoff_reward,
        weight=6.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Fusilage"]),
            "threshold": 1.0,
            "min_height": 0.8,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    fail = RewTerm(
        func=mdp.is_terminated_term,
        weight=-5.0,
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
    takeoff_height = RewTerm(func=mdp.takeoff_height_tanh, weight=5.0, params={"target_height": 1.8, "std": 0.6})
    goal_progress = RewTerm(
        func=mdp.goal_progress_reward,
        weight=10.0,
        params={"command_name": "landing_target"},
    )
    target_xy = RewTerm(
        func=mdp.xy_command_error_tanh,
        weight=6.0,
        params={"std": 40.0, "command_name": "landing_target"},
    )
    stationary = RewTerm(
        func=mdp.stationary_penalty,
        weight=-3.0,
        params={
            "min_speed": 2.0,
            "activation_distance": 10.0,
            "command_name": "landing_target",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    final_approach = RewTerm(
        func=mdp.final_approach_reward,
        weight=4.0,
        params={
            "activation_distance": 10.0,
            "max_lin_speed": 3.0,
            "max_ang_speed": 2.0,
            "command_name": "landing_target",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    target_heading = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.1,
        params={"command_name": "landing_target"},
    )
    vertical_speed = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot")})
    angular_rate = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot")})
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.35, params={"asset_cfg": SceneEntityCfg("robot")})
    fuselage_acceleration = RewTerm(
        func=mdp.body_linear_acceleration_l2,
        weight=-0.01,
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
    nacelle_asymmetry = RewTerm(
        func=mdp.nacelle_asymmetry_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Nacelle_Left_Hinge", "Nacelle_Right_Hinge"],
            )
        },
    )
    bad_touchdown = RewTerm(
        func=mdp.landing_zone_violation,
        weight=-2.0,
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
    touchdown_bonus = RewTerm(
        func=mdp.valid_fuselage_touchdown_bonus,
        weight=10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Fusilage"]),
            "threshold": 1.0,
            "landing_radius": 2.0,
            "max_tilt": 0.2,
            "max_lin_speed": 1.0,
            "max_ang_speed": 1.0,
            "touchdown_height": 1.0,
            "command_name": "landing_target",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.002)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

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
        params={"asset_cfg": SceneEntityCfg("robot"), "max_lin_speed": 18.0, "max_ang_speed": 12.0},
    )
    excessive_acceleration = DoneTerm(
        func=mdp.excessive_body_linear_acceleration,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Fusilage"]), "limit": 75.0},
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


##
# Environment configuration
##


@configclass
class Tiltrotor6dLabEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Tiltrotor6dLabSceneCfg = Tiltrotor6dLabSceneCfg(num_envs=64, env_spacing=100.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = self.commands.landing_target.resampling_time_range[1]
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
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