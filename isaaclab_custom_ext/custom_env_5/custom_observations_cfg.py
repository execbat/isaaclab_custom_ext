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
from isaaclab.sensors import patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch


from .observations import depth_avgpool, compressed_image_features

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


    
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
#        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
#        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
#        projected_gravity = ObsTerm(
#            func=mdp.projected_gravity,
#            noise=Unoise(n_min=-0.05, n_max=0.05),
#        )

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
       
              
        # CUSTOM ADDED OBSERVATIONS
        
        # front camera Intel RealSense D435i
        cam_rgb_feat = ObsTerm(
            func=compressed_image_features,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "model_name": "theia-tiny-patch16-224-cdiv",  # or cddsv
                #"model_device": env.device,                  
            },
        )
        '''
        # preprocessed depth map from the front camera
        cam_depth_vec = ObsTerm(
            func=depth_avgpool,
            params={"sensor_cfg": SceneEntityCfg("front_camera"), "pool": 4},
        )
        
        # lidar observations RayCaster
        lidar_scan_full = ObsTerm(
            func=regex_lidar_distance_channels_all,
            params={
                "sensor_cfg": SceneEntityCfg("lidar_top"),
                "normalize": True,
                "clip_to_unit": False,
                #"data_type": "distance",   # "distance" | "height" | "points"                
                "fill_no_hit": 0.0,         # if detected nothing
                "flatten": True,           # shape (N, C*A)
            },
        )
        '''
       
        
        # imu data
        
        imu_projected_gravity = ObsTerm(func=mdp.imu_projected_gravity)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel)
        imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc)
        
        

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = 5
            
    # observation groups
    policy: PolicyCfg = PolicyCfg()
