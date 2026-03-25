# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Set nacelle revolute limits in Tiltrotor_6D.usd to match URDF/code (-105 deg .. +15 deg)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Usd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--usd",
        type=Path,
        default=None,
        help="Path to Tiltrotor_6D.usd (default: next to this repo layout).",
    )
    parser.add_argument("--lower-deg", type=float, default=-105.0)
    parser.add_argument("--upper-deg", type=float, default=15.0)
    args = parser.parse_args()

    if args.usd is None:
        root = Path(__file__).resolve().parents[1]
        args.usd = root / "urdf" / "Tiltrotor_6D" / "Tiltrotor_6D" / "Tiltrotor_6D.usd"

    path = str(args.usd)
    stage = Usd.Stage.Open(path)
    if not stage:
        raise RuntimeError(f"Could not open: {path}")

    joint_paths = [
        "/Tiltrotor_6D/joints/Nacelle_Left_Hinge",
        "/Tiltrotor_6D/joints/Nacelle_Right_Hinge",
    ]
    for jp in joint_paths:
        prim = stage.GetPrimAtPath(jp)
        if not prim.IsValid():
            raise RuntimeError(f"Missing prim: {jp}")
        lo = prim.GetAttribute("physics:lowerLimit")
        hi = prim.GetAttribute("physics:upperLimit")
        if not lo or not hi:
            raise RuntimeError(f"{jp} missing physics:lowerLimit/upperLimit")
        lo.Set(args.lower_deg)
        hi.Set(args.upper_deg)
        print(f"{jp}: physics:lowerLimit={args.lower_deg} physics:upperLimit={args.upper_deg}")

    stage.GetRootLayer().Save()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
