# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause



from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg  

from isaaclab.assets.articulation import ArticulationCfg

##
# Pre-defined configs
##
#from isaaclab_assets import G1_MINIMAL_CFG, MATH_G1_23DF_CFG  # isort: skip
from isaaclab_custom_ext.unitree_g1_23dof.asset_unitree_g1_23dof import MATH_G1_23DF_CFG
from isaaclab_custom_ext.custom_env_1.custom_velocity_env_cfg import CustomLocomotionVelocityRoughEnvCfg
###


# import of sensors
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.imu import ImuCfg


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )

        
@configclass
class G1RoughEnv1Cfg(CustomLocomotionVelocityRoughEnvCfg):
    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = MATH_G1_23DF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        
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
            spawn=cam_spawn,
            width=160,   # 640
            height=120,  # 480
            data_types=["distance_to_image_plane"],  # RGB + "depth" ["rgb", "distance_to_image_plane"]
            update_period= 1/ 60,           # every step of env env (sync)
            update_latest_camera_pose=True,
            depth_clipping_behavior="max",
        )

        # === 360° LiDAR via RayCaster  ===
        lidar_pattern = LidarPatternCfg(
            channels=8 ,                           # number of vertical rays
            vertical_fov_range=(-15.0, 15.0),      # degrees
            horizontal_fov_range=(0.0, 360.0),     # full circle
            horizontal_res=0.2,                    # grad/step (0.2° -> 1800 datapoints for 360°)
        )
        self.scene.lidar_top = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/mid360_link",
            update_period=1 / 60,
            offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.5)),
            mesh_prim_paths=["/World/ground"],     # The list of mesh primitive paths to ray cast against
            ray_alignment="yaw",                   # Specify in what frame the rays are projected onto the ground. Default is "base" ["base", "yaw", "world"]
            pattern_cfg= lidar_pattern,
            debug_vis= False,           
        )
        

        # === IMU inside of torso ===
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",   # This prom should have rigid body
            update_period= 1 / 60,     # every step (sync)
            history_length=1,
            offset=ImuCfg.OffsetCfg(
                pos=(-0.03959, -0.00224, 0.13792),                
                rot=(1.0, 0.0, 0.0, 0.0),            
            )
        )       
        
        
        
        
    def get_metrics(self) -> dict:
        metrics = {}
        metrics["Metrics/command_range/lin_vel_x_max"] = self.command_manager.commands["base_velocity"].ranges.lin_vel_x[1]
        metrics["Metrics/command_range/lin_vel_x_min"] = self.command_manager.commands["base_velocity"].ranges.lin_vel_x[0]
        return metrics     

@configclass
class G1RoughEnv1Cfg_PLAY(G1RoughEnv1Cfg):


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
        self.scene.lidar_top.debug_vis = True

        
    def get_metrics(self) -> dict:
        metrics = {}
        metrics["Metrics/command_range/lin_vel_x_max"] = self.command_manager.commands["base_velocity"].ranges.lin_vel_x[1]
        metrics["Metrics/command_range/lin_vel_x_min"] = self.command_manager.commands["base_velocity"].ranges.lin_vel_x[0]
        return metrics   
