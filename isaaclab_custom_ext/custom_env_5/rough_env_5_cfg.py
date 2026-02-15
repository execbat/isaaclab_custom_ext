# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import configclass



from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

##
# Pre-defined configs
##
#from isaaclab_assets import G1_MINIMAL_CFG, MATH_G1_23DF_CFG  # isort: skip
from isaaclab_custom_ext.unitree_g1_23dof.asset_unitree_g1_23dof import MATH_G1_23DF_CFG
from .custom_velocity_env_cfg import CustomLocomotionVelocityRoughEnvCfg
from .objects import TARGET_MARKER, OBSTACLE_CYL

###


# import of sensors
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, patterns


from isaaclab.sensors.imu import ImuCfg

from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import  UsdFileCfg
from isaaclab.sensors import TiledCameraCfg, TiledCamera


import numpy as np
import omni
# from isaacsim.sensors.rtx import LidarRtx

#from .regex_ray_caster import RegexRayCaster


MAX_OBS = 30
        
@configclass
class G1RoughEnv5Cfg(CustomLocomotionVelocityRoughEnvCfg):
    
    

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        
        
        # columns
        objs = {}
        for i in range(MAX_OBS):
            name = f"obst_{i:02d}"
            objs[name] = OBSTACLE_CYL.replace(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=OBSTACLE_CYL.spawn.replace(copy_from_source=False),
            )
        self.scene.obstacles = RigidObjectCollectionCfg(rigid_objects=objs)     
        
        # target
        self.scene.target = TARGET_MARKER.replace(
            prim_path="{ENV_REGEX_NS}/Target",
            spawn=TARGET_MARKER.spawn.replace(copy_from_source=False), 
        )  
        
        self.scene.robot = MATH_G1_23DF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # additional items
        self.episode_length_s = 15.0

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None


        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["torso_link", "pelvis", ".*_hip_.*", ".*knee_.*", ".*elbow_.*", ".*_wrist_.*"]
        

        # SENSORS
        
        # ====== paths to mount places ======
        #cam_mount   = "{ENV_REGEX_NS}/Robot/torso_link/d435_link"
        #lidar_mount = "{ENV_REGEX_NS}/Robot/torso_link/mid360_link"
        #imu_mount   = "{ENV_REGEX_NS}/Robot/torso_link/imu_in_torso"

        # 1 === FRONT RGB-D CAMERA  ===
        
        cam_spawn = sim_utils.PinholeCameraCfg(  # USD Camera spawner
            focal_length=0.88,                   
            horizontal_aperture=2.0,              
            clipping_range=(0.1, 15.0),
        )

        self.scene.front_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/camera",  
            offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), convention="world"), # Offset is d435_link frame in reference to torso_link from urdf
            spawn=cam_spawn,            
            width=224,   # 640
            height=224,  # 480
            data_types=["rgb"], # "distance_to_image_plane"],  # RGB + "depth" ["rgb", "distance_to_image_plane"]
            update_period= 0.1,                    # every step of env env (sync)
            update_latest_camera_pose=True,
            depth_clipping_behavior="max",
        )
        
        # === IMU inside of torso ===
        
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",   # SHOULD BE A RIGID BODY!
            update_period= 0.1,                   # every step (sync)
            history_length=5,
            offset=ImuCfg.OffsetCfg(               # Offset is imu link frame in reference to torso_link from urdf
                pos=(-0.03959, -0.00224, 0.13792),                
                rot=(1.0, 0.0, 0.0, 0.0),            
            ),
            debug_vis=False
            
        )  
           



        
        
    def get_metrics(self) -> dict:
        metrics = {}
        metrics["Metrics/command_range/lin_vel_x_max"] = self.command_manager.commands["base_velocity"].ranges.lin_vel_x[1]
        metrics["Metrics/command_range/lin_vel_x_min"] = self.command_manager.commands["base_velocity"].ranges.lin_vel_x[0]
        return metrics     




@configclass
class G1RoughEnv5Cfg_PLAY(G1RoughEnv5Cfg):


    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        
        self.episode_length_s = 15.0

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["torso_link", "pelvis"]	

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        
        # switch ON debug vis
        #self.observations.policy.rtx_lidar_points.params["debug"] = True
        # self.scene.lidar_top.debug_vis = True
        #self.scene.imu.debug_vis = True
        self.scene.height_scanner.debug_vis=True

        
    def get_metrics(self) -> dict:
        metrics = {}
        metrics["Metrics/command_range/lin_vel_x_max"] = self.command_manager.commands["base_velocity"].ranges.lin_vel_x[1]
        metrics["Metrics/command_range/lin_vel_x_min"] = self.command_manager.commands["base_velocity"].ranges.lin_vel_x[0]
        return metrics   
