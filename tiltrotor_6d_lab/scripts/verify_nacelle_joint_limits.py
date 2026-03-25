# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Headless: load env, print nacelle joint position limits (rad) and names."""

from __future__ import annotations

import argparse
import math
import traceback

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Template-Tiltrotor-6d-Lab-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print("[verify_nacelle_joint_limits] starting AppLauncher...", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[verify_nacelle_joint_limits] app ready, importing env...", flush=True)

import tiltrotor_6d_lab.tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if args_cli.device is not None else "cuda:0",
        num_envs=args_cli.num_envs,
    )
    import gymnasium as gym

    print("[verify_nacelle_joint_limits] gym.make...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    names = robot.joint_names
    want = ["Nacelle_Left_Hinge", "Nacelle_Right_Hinge"]
    ids = [names.index(n) for n in want]

    # After reset, articulation data is populated
    env.reset()
    lim = robot.data.joint_pos_limits[0, ids].cpu()
    exp_lo = -7.0 * math.pi / 12.0
    exp_hi = math.pi / 12.0

    print("nacelle joint position limits [lower, upper] (rad):")
    for j, n in enumerate(want):
        lo, hi = float(lim[j, 0]), float(lim[j, 1])
        print(f"  {n}: [{lo:.6f}, {hi:.6f}]  ({math.degrees(lo):.2f} deg, {math.degrees(hi):.2f} deg)")

    print(f"expected (code/URDF): [{exp_lo:.6f}, {exp_hi:.6f}] rad")
    ok = all(
        abs(float(lim[j, 0]) - exp_lo) < 0.02 and abs(float(lim[j, 1]) - exp_hi) < 0.02
        for j in range(len(ids))
    )
    print("MATCH_EXPECTED" if ok else "MISMATCH — check USD vs articulation parsing")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[verify_nacelle_joint_limits] FAILED", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
