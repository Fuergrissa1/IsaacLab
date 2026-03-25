from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor


class TiltrotorThrustAction(ActionTerm):
    """Control per-nacelle tilt and per-rotor RPM-derived thrust."""

    cfg: "TiltrotorThrustActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "TiltrotorThrustActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._nacelle_joint_ids, self._nacelle_joint_names = self._asset.find_joints(
            cfg.nacelle_joint_names, preserve_order=True
        )
        self._rotor_joint_ids, self._rotor_joint_names = self._asset.find_joints(
            cfg.rotor_joint_names, preserve_order=True
        )
        self._rotor_body_ids, self._rotor_body_names = self._asset.find_bodies(
            cfg.rotor_body_names, preserve_order=True
        )

        if len(self._rotor_joint_ids) != len(self._rotor_body_ids):
            raise ValueError("Rotor joint and body counts must match.")

        self._num_rotors = len(self._rotor_joint_ids)
        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._nacelle_targets = torch.zeros((self.num_envs, len(self._nacelle_joint_ids)), device=self.device)
        self._rpm_targets = torch.zeros((self.num_envs, self._num_rotors), device=self.device)

        self._force_directions = torch.tensor(
            [cfg.force_directions[name] for name in self._rotor_body_names], dtype=torch.float32, device=self.device
        )
        self._rotor_velocity_signs = torch.tensor(
            [cfg.rotor_velocity_signs[name] for name in self._rotor_joint_names],
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def action_dim(self) -> int:
        return len(self._nacelle_joint_ids) + self._num_rotors

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> "GenericActionIODescriptor":
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "TiltrotorThrustAction"
        self._IO_descriptor.joint_names = [*self._nacelle_joint_names, *self._rotor_joint_names]
        return self._IO_descriptor

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        q0 = self.cfg.nacelle_reset_pos
        self._processed_actions[env_ids, : len(self._nacelle_joint_ids)] = q0
        self._nacelle_targets[env_ids] = q0
        self._rpm_targets[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        num_nacelles = len(self._nacelle_joint_ids)
        nacelle_actions = torch.clamp(actions[:, :num_nacelles], min=-1.0, max=1.0)
        nacelle_command = self.cfg.nacelle_offset + self.cfg.nacelle_scale * nacelle_actions
        nacelle_command = torch.clamp(
            nacelle_command,
            min=self.cfg.nacelle_clip[0],
            max=self.cfg.nacelle_clip[1],
        )
        if self.cfg.low_altitude_tilt_cap_m is not None:
            h = self._asset.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
            q_cap = self.cfg.low_altitude_max_forward_q
            eff_max = torch.full(
                (self.num_envs,),
                self.cfg.nacelle_clip[1],
                device=nacelle_command.device,
                dtype=nacelle_command.dtype,
            )
            low = h < self.cfg.low_altitude_tilt_cap_m
            eff_max = torch.where(
                low,
                torch.minimum(
                    eff_max,
                    torch.full_like(eff_max, q_cap),
                ),
                eff_max,
            )
            nacelle_command = torch.clamp(
                nacelle_command,
                min=torch.full_like(nacelle_command, self.cfg.nacelle_clip[0]),
                max=eff_max.unsqueeze(1),
            )

        rpm_actions = torch.clamp(actions[:, num_nacelles:], min=-1.0, max=1.0)
        rpm_min, rpm_max = self.cfg.rpm_range
        rpm_targets = 0.5 * (rpm_actions + 1.0) * (rpm_max - rpm_min) + rpm_min

        self._processed_actions[:, :num_nacelles] = nacelle_command
        self._processed_actions[:, num_nacelles:] = rpm_targets
        self._nacelle_targets[:] = nacelle_command
        self._rpm_targets[:] = rpm_targets

    def apply_actions(self):
        self._asset.set_joint_position_target(self._nacelle_targets, joint_ids=self._nacelle_joint_ids)

        rotor_velocity_targets = self._rpm_targets * (2.0 * math.pi / 60.0)
        rotor_velocity_targets = rotor_velocity_targets * self._rotor_velocity_signs.unsqueeze(0)
        self._asset.set_joint_velocity_target(rotor_velocity_targets, joint_ids=self._rotor_joint_ids)

        thrust_newtons = torch.clamp(
            self.cfg.thrust_slope * self._rpm_targets + self.cfg.thrust_intercept,
            min=0.0,
        )
        forces = thrust_newtons.unsqueeze(-1) * self._force_directions.unsqueeze(0)
        torques = torch.zeros_like(forces)
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=self._rotor_body_ids,
        )


@configclass
class TiltrotorThrustActionCfg(ActionTermCfg):
    """Configuration for the tiltrotor flight action term."""

    class_type: type[ActionTerm] = TiltrotorThrustAction

    asset_name: str = MISSING
    nacelle_joint_names: list[str] = MISSING
    rotor_joint_names: list[str] = MISSING
    rotor_body_names: list[str] = MISSING
    force_directions: dict[str, tuple[float, float, float]] = MISSING
    rotor_velocity_signs: dict[str, float] = MISSING
    # Maps a in [-1, 1] to q = offset + scale * a, then clamps to nacelle_clip.
    # Default clip ~ -105 deg to +15 deg (0 = forward, -pi/2 = vertical in joint space).
    nacelle_scale: float = math.pi / 3.0
    nacelle_offset: float = -math.pi / 4.0
    nacelle_clip: tuple[float, float] = (-7.0 * math.pi / 12.0, math.pi / 12.0)
    nacelle_reset_pos: float = -math.pi / 2.0
    # Below this AGL (m), cap max forward nacelle q to low_altitude_max_forward_q (rad).
    low_altitude_tilt_cap_m: float | None = 1.0
    low_altitude_max_forward_q: float = -math.pi / 6.0
    rpm_range: tuple[float, float] = (0.0, 1600.0)
    # Approximate linear fit of the provided 0-degree thrust curve, converted to Newtons.
    thrust_slope: float = 0.1687
    thrust_intercept: float = -72.0
