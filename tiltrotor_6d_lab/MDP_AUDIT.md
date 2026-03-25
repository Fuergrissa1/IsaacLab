# MDP audit: Tiltrotor 6D Lab

## Goals (what we ask the policy to do)

- Take off, fly to a **distant** landing pose (XY + heading), land in zone with bounded speed/tilt.
- Avoid **non-fuselage contact** (rotor/nacelle strikes) and bad fuselage contact outside zones.
- Intermediate curriculum: **hover** task (`Template-Tiltrotor-6d-Hover-v0`) with goal at origin and short horizon.

## Actions

| Dim | Meaning |
|-----|--------|
| 2 | Per-nacelle tilt command, mapped linearly in joint rad with clip (see `TiltrotorThrustActionCfg`). |
| 2 | Per-rotor RPM → thrust via linear curve + velocity command on rotor joints. |

**Notes:** Nacelle PD is implicit (`stiffness`/`damping`); thrust is applied via `permanent_wrench_composer`. Reset puts nacelles at **vertical** (`nacelle_reset_pos = -pi/2`), not at the mechanical minimum angle. **Below 1 m AGL**, commanded nacelle angle is **hard-clamped** so max forward joint q is **-π/6** (same 30° rule), matching the feasible observation.

## Observations (policy group)

| Term | Role |
|------|------|
| `root_pos_w`, `root_quat_w`, `root_lin_vel_w` | World pose + velocity. |
| `base_ang_vel`, `projected_gravity` | Attitude / rate. |
| `joint_pos_rel`, `joint_vel_rel` | All actuator DOF. |
| `landing_target` | **Yaw-aligned body frame:** XY offset from aircraft to goal (updated each step) plus **heading error** (`wrap_to_pi`). This is the relative **direction and distance** in the horizontal plane the policy needs for navigation (not world-fixed coordinates). See Isaac Lab `pose_2d_command.py` (`pos_command_b`, `heading_command_b`). |
| `feasible_nacelle` | **4** values: `q_min_feas`, `q_max_feas` from geometry (lever = **245 mm hub-to-disk + 750 mm rotor radius**), plus current left/right nacelle `q`. Below **1 m** AGL, `q_max_feas` is also capped to **-π/6** rad (≥ **30°** user angle from forward toward vertical: 0° = forward, 90° = vertical). |
| `last_action` | Previous action (4). |

**Observation size:** 43 (was 39 before `feasible_nacelle`). **Old checkpoints are incompatible** with the new observation vector.

## Goal → information matrix

| Goal | Needs | Observation / signal | Gap addressed |
|------|--------|-------------------------|---------------|
| Navigate to landing XY | Relative horizontal error + heading | `landing_target` (body XY + heading error) | None (yaw-aligned frame) |
| Land in zone | Proximity, speed, tilt, contact | Rewards + `successful_landing` / terminations | — |
| Avoid rotor strike | Height vs tilt | `feasible_nacelle` + `illegal_contact` | Feasible bounds added |
| Hover curriculum | Altitude, stability | Hover env rewards + shorter horizon | `Template-Tiltrotor-6d-Hover-v0` |

## Gap analysis (achievability)

- **Navigation:** Command is already in **body / yaw** frame; no change required for basic “fly toward goal” observability.
- **Rotor–ground:** Policy lacked explicit **feasible tilt vs height**; added `feasible_nacelle_joint_clearance` (conservative geometry, sim still authoritative via `illegal_contact`).
- **Temporal:** Single-step `last_action` only; if needed later, add short stacks or recurrent policy.

## Curriculum promotion (suggested thresholds)

Use TensorBoard + termination logs from `Tiltrotor6dLabEnv`. These are **heuristic**—adjust to your runs.

1. **Hover → translate (Phase 2):** Mean episode length **>** ~40% of `episode_length_s` (hover) **and** `illegal_contact` share of terminations **<** ~50% over a window.
2. **Translate → full mission (Phase 3):** Restore `Template-Tiltrotor-6d-Lab-v0` or widen hover command ranges; require stable **time aloft** and acceptable **goal_progress** slope before long-range landing.

## USD

- URDF nacelle limits match **~[-7π/12, π/12] rad**. **`Tiltrotor_6D.usd`** was patched so `physics:lowerLimit` / `physics:upperLimit` on both nacelle revolute joints are **-105°** and **+15°** (same band). Script: [`scripts/patch_tiltrotor_usd_limits.py`](scripts/patch_tiltrotor_usd_limits.py). Verify in sim: [`scripts/verify_nacelle_joint_limits.py`](scripts/verify_nacelle_joint_limits.py) (prints `robot.data.joint_pos_limits`; expect **MATCH_EXPECTED**).
