# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tiltrotor 2D environment."""

from . import agents
from .tiltrotor_2d_env import TiltrotorEnv, TiltrotorEnvCfg
from .tiltrotor_2d_multigoal_env import TiltrotorMultiGoalEnv
from .tiltrotor_2d_multigoal_cfg import TiltrotorMultiGoalEnvCfg

__all__ = ["TiltrotorEnv", "TiltrotorEnvCfg", "TiltrotorMultiGoalEnv", "TiltrotorMultiGoalEnvCfg"]

##
# Register Gym environments.
##

import gymnasium as gym

gym.register(
    id="Isaac-Tiltrotor-2d-Direct-v0",
    entry_point=f"{__name__}.tiltrotor_2d_env:TiltrotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tiltrotor_2d_env:TiltrotorEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Tiltrotor-2d-MultiGoal-v0",
    entry_point=f"{__name__}.tiltrotor_2d_multigoal_env:TiltrotorMultiGoalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tiltrotor_2d_multigoal_cfg:TiltrotorMultiGoalEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)