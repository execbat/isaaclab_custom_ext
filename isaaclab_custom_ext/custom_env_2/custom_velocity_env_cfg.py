from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from .custom_observations_cfg import ObservationsCfg
from .custom_rewards_cfg import G1Rewards
from .custom_commands_cfg import CommandsCfg
from .custom_event_cfg import EventCfg
from .custom_scene_cfg import SceneCfg

class CustomLocomotionVelocityRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    scene : SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    rewards: G1Rewards = G1Rewards()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()


