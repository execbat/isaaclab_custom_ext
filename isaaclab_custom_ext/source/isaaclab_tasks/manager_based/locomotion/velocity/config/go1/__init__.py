# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
#from . import env_scene

##
# Register Gym environments.
##



gym.register(
    id="Isaac-Velocity-Sber-Unitree-Go1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaac_hydra_ext.source.isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg:UnitreeGo1RoughEnvCfg",
        "appo_cfg_entry_point": f"{agents.__name__}:appo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Sber-Unitree-Go1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaac_hydra_ext.source.isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg:UnitreeGo1RoughEnvCfg_PLAY",
        "appo_cfg_entry_point": f"{agents.__name__}:appo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

  
