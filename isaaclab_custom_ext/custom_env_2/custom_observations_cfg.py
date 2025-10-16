import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch

from .rtx_lidar_lazy_hook import obs_rtx_lidar_points
from .observations import depth_avgpool, compressed_image_features

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

# help function for LIDAR obs compression
def lidar_height_channels_min(env, sensor_cfg: SceneEntityCfg, offset: float = 0.0) -> torch.Tensor:
    """
    Converts RayCaster output into a compact 1D vector: for each vertical channel
    Take the minimum azimuth (i.e., the closest height/obstacle). Size: (num_envs, channels).

    Works with the pattern (channels x azimuth). Based on mdp.height_scan.
    """
    # height_scan returns (N, num_rays) in the same order as the lidar template
    hs = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)          # (N, num_rays)
    sensor = env.scene.sensors[sensor_cfg.name]
    # extract the number of channels from the pattern
    channels = int(sensor.cfg.pattern_cfg.channels)
    # expand into (N, channels, azimuth_bins) and take the minimum in azimuth
    hs = hs.view(env.num_envs, channels, -1).amin(dim=-1)                    # (N, channels)
    # (optional) normalize/clip to stabilize the scale
    return torch.clamp(hs, -2.0, 2.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )         
              
        # CUSTOM ADDED OBSERVATIONS
        # front camera Intel RealSense D435i
        # Used Visual Transformer "theia-tiny-patch16-224-cdiv " input: RGB output: Features vector 
        cam_rgb_feat = ObsTerm(
            func=compressed_image_features,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "model_name": "theia-tiny-patch16-224-cdiv",  # or cddsv
                #"model_device": env.device,                  
            },
        )
        # preprocessed depth map from the front camera
        cam_depth_vec = ObsTerm(
            func=depth_avgpool,
            params={"sensor_cfg": SceneEntityCfg("front_camera"), "pool": 4},
        )
        

#       # RTX LIDAR
        rtx_lidar_points = ObsTerm(
            func=obs_rtx_lidar_points,
            params={"debug" : False},            
        )
        
        
        # imu data
        imu_projected_gravity = ObsTerm(func=mdp.imu_projected_gravity)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel)
        imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc)
        

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


