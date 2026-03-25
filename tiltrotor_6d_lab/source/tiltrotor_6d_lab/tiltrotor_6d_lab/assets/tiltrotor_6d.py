from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_ROBOT_USD = _PROJECT_ROOT / "urdf" / "Tiltrotor_6D" / "Tiltrotor_6D" / "Tiltrotor_6D.usd"

TILTROTOR_6D_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ROBOT_USD.as_posix(),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.27235889587036244),
        rot=(0.5, 0.5, 0.5, 0.5),
        joint_pos={
            "Nacelle_Left_Hinge": -math.pi / 2.0,
            "Nacelle_Right_Hinge": -math.pi / 2.0,
            "Rotor_Left_Hinge": 0.0,
            "Rotor_Right_Hinge": 0.0,
            "Aileron_Left_Hinge": 0.0,
            "Aileron_Right_Hinge": 0.0,
            "Ruddervator_Left_Hinge": 0.0,
            "Ruddervator_Right_Hinge": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    # Implicit PD: drive torque ~ Kp*(q_cmd - q) - Kd*q_dot (see Isaac Lab actuator docs).
    # Rough torque scale ~ Kp * delta_q; angular accel ~ tau / I_axis. Tune Kp/Kd with
    # nacelle+rotor inertia about the hinge; rotors use velocity targets (stiffness 0).
    actuators={
        "nacelles": ImplicitActuatorCfg(
            joint_names_expr=["Nacelle_.*_Hinge"],
            stiffness=40.0,
            damping=4.0,
        ),
        "control_surfaces": ImplicitActuatorCfg(
            joint_names_expr=["Aileron_.*_Hinge", "Ruddervator_.*_Hinge"],
            stiffness=20.0,
            damping=2.0,
        ),
        "rotors": ImplicitActuatorCfg(
            joint_names_expr=["Rotor_.*_Hinge"],
            stiffness=0.0,
            damping=0.1,
        ),
    },
)
