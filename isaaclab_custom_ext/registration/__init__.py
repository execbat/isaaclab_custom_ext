#isaaclab_custom_ext/registration/__init__.py
import gymnasium as gym

from isaaclab_custom_ext import agents
#from isaaclab_custom_ext import custom_env_1

##
# Register Gym environments.
##

# STANDARD ENVS. JUST MOVED TO ISAAC EXTENSION - Version 0

gym.register(
    id="Ext-Isaac-Velocity-Rough-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",	
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_custom_ext.custom_env_0.rough_env_cfg:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"isaaclab_custom_ext.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Ext-Isaac-Velocity-Rough-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_custom_ext.custom_env_0.rough_env_cfg:G1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"isaaclab_custom_ext.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Ext-Isaac-Velocity-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_custom_ext.custom_env_0.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"isaaclab_custom_ext.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Ext-Isaac-Velocity-Flat-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_custom_ext.custom_env_0.flat_env_cfg:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"isaaclab_custom_ext.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


# LESSON SENSORN AND OBSERVATIONS - Version 1

gym.register(
    id="Ext-Isaac-Velocity-Rough-G1-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_custom_ext.custom_env_1.rough_env_1_cfg:G1RoughEnv1Cfg",
        "rsl_rl_cfg_entry_point": f"isaaclab_custom_ext.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Ext-Isaac-Velocity-Rough-G1-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_custom_ext.custom_env_1.rough_env_1_cfg:G1RoughEnv1Cfg_PLAY",
        "rsl_rl_cfg_entry_point": f"isaaclab_custom_ext.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Ext-Isaac-Velocity-Flat-G1-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_custom_ext.custom_env_1.flat_env_1_cfg:G1FlatEnv1Cfg",
        "rsl_rl_cfg_entry_point": f"isaaclab_custom_ext.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Ext-Isaac-Velocity-Flat-G1-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_custom_ext.custom_env_1.flat_env_1_cfg:G1FlatEnv1Cfg_PLAY",
        "rsl_rl_cfg_entry_point": f"isaaclab_custom_ext.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

# LESSON COMMANDS AND ACTIONS - Version 2

gym.register(
    id="Ext-Isaac-Velocity-Rough-G1-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_custom_ext.custom_env_2.rough_env_2_cfg:G1RoughEnv2Cfg",
        "rsl_rl_cfg_entry_point": f"isaaclab_custom_ext.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Ext-Isaac-Velocity-Rough-G1-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_custom_ext.custom_env_2.rough_env_2_cfg:G1RoughEnv2Cfg_PLAY",
        "rsl_rl_cfg_entry_point": f"isaaclab_custom_ext.agents.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Ext-Isaac-Velocity-Flat-G1-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_custom_ext.custom_env_2.flat_env_2_cfg:G1FlatEnv2Cfg",
        "rsl_rl_cfg_entry_point": f"isaaclab_custom_ext.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Ext-Isaac-Velocity-Flat-G1-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"isaaclab_custom_ext.custom_env_2.flat_env_2_cfg:G1FlatEnv2Cfg_PLAY",
        "rsl_rl_cfg_entry_point": f"isaaclab_custom_ext.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
