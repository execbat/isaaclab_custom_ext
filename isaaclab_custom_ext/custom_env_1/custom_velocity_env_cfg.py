from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from .custom_observations_cfg import ObservationsCfg

class CustomLocomotionVelocityRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()


