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
from isaaclab_custom_ext.custom_env_2.custom_velocity_env_cfg import CustomLocomotionVelocityRoughEnvCfg
from isaaclab_custom_ext.custom_env_2.objects import TARGET_MARKER, OBSTACLE_CYL
###


# import of sensors
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.imu import ImuCfg



MAX_OBS = 40
        
@configclass
class G1RoughEnv2Cfg(CustomLocomotionVelocityRoughEnvCfg):
    
    

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
        
        self.scene.robot = MATH_G1_23DF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        
        # additional items

        self.scene.obstacles = RigidObjectCollectionCfg(rigid_objects=objs)       
        # target
        self.scene.target = TARGET_MARKER.replace(
            prim_path="{ENV_REGEX_NS}/Target",
            spawn=TARGET_MARKER.spawn.replace(copy_from_source=False), 
        )  
        
         
        self.episode_length_s = 40.0

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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

        # Rewards
        # self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        # self.rewards.flat_orientation_l2.weight = -1.0
        # self.rewards.action_rate_l2.weight = -0.005
        # self.rewards.dof_acc_l2.weight = -1.25e-7
        # self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        # )
        # self.rewards.dof_torques_l2.weight = -1.5e-7
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        # )	

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["torso_link", "pelvis", ".*_hip_.*", ".*_wrist_.*", ".*shoulder_.*", ".*knee_.*", ".*elbow_.*"]
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

        self.scene.front_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/camera",  
            offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), convention="world"), # Offset is d435_link frame in reference to torso_link from urdf
            spawn=cam_spawn,            
            width=160,   # 640
            height=120,  # 480
            data_types=["distance_to_image_plane"],  # RGB + "depth" ["rgb", "distance_to_image_plane"]
            update_period= 0.1,                    # every step of env env (sync)
            update_latest_camera_pose=True,
            depth_clipping_behavior="max",
        )

        # === 360° LiDAR via RayCaster  ===
        lidar_pattern = LidarPatternCfg(
            channels=8 ,                           # number of vertical rays
            vertical_fov_range=(-90.0, 90.0),      # degrees
            horizontal_fov_range=(-180, 180.0),     
            horizontal_res=0.2,                    # grad/step (0.2° -> 1800 datapoints for 360°)
        )
        self.scene.lidar_top = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",   # SHOULD BE A RIGID BODY!
            update_period=0.02,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0002835, 0.00003, 0.40618), rot=(0.999799, 0.0, 0.020070, 0.0)), # Offset is mid360 link frame in reference to torso_link from urdf
            mesh_prim_paths=["/World/ground",
                            # /{ENV_REGEX_NS}/obst_.*",   
                            #"{ENV_REGEX_NS}/Target",
                            ],     
            ray_alignment="base",                  # Specify in what frame the rays are projected onto the ground. Default is "base" ["base", "yaw", "world"]
            pattern_cfg= lidar_pattern,
            debug_vis=False,  
            max_distance=100,         
        )
        

        # === IMU inside of torso ===
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",   # SHOULD BE A RIGID BODY!
            update_period= 0.02,                   # every step (sync)
            history_length=1,
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
class G1RoughEnv2Cfg_PLAY(G1RoughEnv2Cfg):


    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.episode_length_s = 40.0

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        
        # switch ON debug vis
        #self.scene.lazy_sensor_update = False
        self.scene.lidar_top.debug_vis = True
        self.scene.imu.debug_vis = True

        
    def get_metrics(self) -> dict:
        metrics = {}
        metrics["Metrics/command_range/lin_vel_x_max"] = self.command_manager.commands["base_velocity"].ranges.lin_vel_x[1]
        metrics["Metrics/command_range/lin_vel_x_min"] = self.command_manager.commands["base_velocity"].ranges.lin_vel_x[0]
        return metrics   
