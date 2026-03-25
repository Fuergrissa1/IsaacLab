from __future__ import annotations

import argparse

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect initial reset state.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import tiltrotor_6d_lab.tasks  # noqa: E402,F401


def _print_state(tag: str, env) -> None:
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    joints = robot.data.joint_pos[0].detach().cpu()
    joint_names = robot.joint_names
    nacelle_names = ["Nacelle_Left_Hinge", "Nacelle_Right_Hinge"]
    nacelle_ids = [joint_names.index(name) for name in nacelle_names]
    nacelle_deg = [float(torch.rad2deg(joints[idx]).item()) for idx in nacelle_ids]

    quat = robot.data.root_quat_w[0:1]
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    roll_deg = float(torch.rad2deg(roll[0]).item())
    pitch_deg = float(torch.rad2deg(pitch[0]).item())
    yaw_deg = float(torch.rad2deg(yaw[0]).item())
    gravity_b = robot.data.projected_gravity_b[0]
    tilt_deg = float(
        torch.rad2deg(
            torch.atan2(
                torch.norm(gravity_b[:2]),
                torch.clamp(gravity_b[2].abs(), min=1e-6),
            )
        ).item()
    )

    fuselage_id = robot.body_names.index("Fusilage")
    fuselage_force = torch.norm(
        contact_sensor.data.net_forces_w_history[0, :, fuselage_id],
        dim=-1,
    )
    fuselage_contact = bool(torch.max(fuselage_force).item() > 1.0)

    env_origin_xy = env.unwrapped.scene.env_origins[0, :2]
    root_xy = robot.data.root_pos_w[0, :2]
    takeoff_distance = float(torch.norm(root_xy - env_origin_xy).item())
    command_term = env.unwrapped.command_manager.get_term("landing_target")
    landing_distance = float(
        torch.norm(command_term.pos_command_w[0, :2] - root_xy).item()
    )

    tm = env.unwrapped.termination_manager
    term_values = {name: bool(tm.get_term(name)[0].item()) for name in tm.active_terms}

    print(f"\n[{tag}]", flush=True)
    print(f"nacelles_deg={nacelle_deg}", flush=True)
    print(
        "root_rpy_deg="
        f"({roll_deg:.2f}, {pitch_deg:.2f}, {yaw_deg:.2f})"
        f", tilt_deg={tilt_deg:.2f}",
        flush=True,
    )
    print(
        f"fuselage_contact={fuselage_contact}, "
        f"takeoff_distance={takeoff_distance:.3f}, "
        f"landing_distance={landing_distance:.3f}",
        flush=True,
    )
    print(f"projected_gravity_b={gravity_b.tolist()}", flush=True)
    print(f"termination_terms={term_values}", flush=True)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    _print_state("after_reset", env)

    zero_actions = torch.zeros(
        (
            env.unwrapped.num_envs,
            env.unwrapped.action_manager.total_action_dim,
        ),
        device=env.unwrapped.device,
    )
    for i in range(3):
        env.step(zero_actions)
        _print_state(f"after_step_{i + 1}", env)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
