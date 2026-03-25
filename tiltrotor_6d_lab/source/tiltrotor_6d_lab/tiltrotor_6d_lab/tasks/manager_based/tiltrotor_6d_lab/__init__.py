# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Tiltrotor-6d-Lab-v0",
    entry_point=f"{__name__}.tiltrotor_6d_lab_env:Tiltrotor6dLabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tiltrotor_6d_lab_env_cfg:Tiltrotor6dLabEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_PPO_cfg.yaml",
        "skrl_AMP_cfg_entry_point": f"{agents.__name__}:skrl_AMP_cfg.yaml",
        "skrl_PPO_cfg_entry_point": f"{agents.__name__}:skrl_PPO_cfg.yaml",
    },
)

gym.register(
    id="Template-Tiltrotor-6d-Hover-v0",
    entry_point=f"{__name__}.tiltrotor_6d_lab_env:Tiltrotor6dLabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tiltrotor_6d_lab_hover_env_cfg:Tiltrotor6dHoverEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_hover_cfg.yaml",
        "skrl_AMP_cfg_entry_point": f"{agents.__name__}:skrl_AMP_cfg.yaml",
        "skrl_PPO_cfg_entry_point": f"{agents.__name__}:skrl_ppo_hover_cfg.yaml",
    },
)