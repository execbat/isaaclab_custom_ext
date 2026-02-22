from isaaclab.envs import ManagerBasedRLEnvCfg
from .custom_observations_cfg import ObservationsCfg
from .custom_rewards_cfg import RewardsCfg
from .custom_commands_cfg import CommandsCfg
from .custom_event_cfg import EventCfg
from .custom_scene_cfg import SceneCfg
from .custom_terminations_cfg import TerminationsCfg
from .custom_curriculum_cfg import CurriculumCfg
from .custom_actions_cfg import ActionsCfg

from isaaclab.utils import configclass

@configclass
class CustomLocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    scene : SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    rewards:  RewardsCfg = RewardsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    terminations : TerminationsCfg  = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    actions: ActionsCfg = ActionsCfg()
    
    
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5  # 20 Hz (because sim.dt = 0.01)
        
        self.episode_length_s = 20.0
        self.is_finite_horizon = False # Send DONE/Truncated siglan to the agent when episode Timeout

        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15   
             



