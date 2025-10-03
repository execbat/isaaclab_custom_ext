# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaac_hydra_ext.source.isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, TestLocomotionVelocityRoughEnvCfg
from isaac_hydra_ext.source.isaaclab_tasks.manager_based.locomotion.velocity.config.go1.env_scene import TARGET_MARKER, OBSTACLE_CYL, ChaseTestCommandsCfg, ChaseTestEventCfg
##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort: skip
from dataclasses import MISSING
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
#['trunk', 'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip', 'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh', 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf', 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']

from isaaclab.sensors.camera import CameraCfg
import isaaclab.sim as sim_utils

MAX_OBS = 40

@configclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.episode_length_s = 20.0
        
        self.decimation = 4 
        self.sim.dt =  0.005
        self.sim.render_interval = self.decimation
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        
        
        # scene
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
         
        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01


        # additional items
        self.scene.target = TARGET_MARKER.replace(
            prim_path="{ENV_REGEX_NS}/Target",
            spawn=TARGET_MARKER.spawn.replace(copy_from_source=False), 
        )

        objs = {}
        for i in range(MAX_OBS):
            name = f"obst_{i:02d}"
            objs[name] = OBSTACLE_CYL.replace(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",               
                spawn=OBSTACLE_CYL.spawn.replace(copy_from_source=False),
            )

        self.scene.obstacles = RigidObjectCollectionCfg(rigid_objects=objs)




        # reduce action scale
        # self.actions.joint_pos.scale = 0.5

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
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

        # REWARDS
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.params["threshold"] = 0.5 
         #-0.08     
        

        #self.rewards.undesired_contacts = None
        
        
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
#        self.rewards.track_vel_exp_product.weight = 10
        self.rewards.com_over_support_h.weight = 0.5
        self.rewards.upright.weight = 0.5
        self.rewards.heading_align.weight = 0.25
        #self.rewards.trot_rew.weight = 10
#        self.rewards.progress_to_target.weight = 6.0 
        
        
        
        # PENALTIES
        
        
        #self.rewards.flat_orientation_l2.weight = 0.0    
          
        
        self.rewards.lin_vel_z_l2.weight = 0.0 
        self.rewards.ang_vel_xy_l2.weight = 0.0 
        self.rewards.dof_torques_l2.weight = 0.0  
        self.rewards.joint_vel_l2.weight = 0.0  
        self.rewards.action_rate_l2.weight = 0.0 
        self.rewards.dof_acc_l2.weight = 0.0 
        self.rewards.dof_pos_limits.weight = 0.0 
        self.rewards.feet_air_time.weight =  0.0
        self.rewards.feet_slide.weight =  0.0
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.alive.weight = 0.05
        self.rewards.feet_impact_vel.weight = 0.0
        self.rewards.idle_penalty.weight = 0.0
        self.rewards.trot_rew.weight = 0.0
        
         
        
        #self.rewards.track_lin_vel_xy_mse.weight = -3.0 # penalty for not following desired direction
        #self.rewards.track_ang_vel_z_mse.weight = -1.5 # penalty for not following desired direction

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["trunk"] #, 'trunk', '.*_thigh', '.*_hip']
        self.rewards.termination_penalty.weight = -200.0
        
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        
   

@configclass
class UnitreeGo1RoughEnvCfg_PLAY(TestLocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.episode_length_s = 99999.0
        
        self.decimation = 4 
        self.sim.dt =  0.005
        self.sim.render_interval = self.decimation
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        
        
        # scene
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
         
        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.02)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.001, 0.002)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01


        # additional items
        self.scene.target = TARGET_MARKER.replace(
            prim_path="{ENV_REGEX_NS}/Target",
            spawn=TARGET_MARKER.spawn.replace(copy_from_source=False), 
        )

        objs = {}
        for i in range(MAX_OBS):
            name = f"obst_{i:02d}"
            objs[name] = OBSTACLE_CYL.replace(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",               
                spawn=OBSTACLE_CYL.spawn.replace(copy_from_source=False),
            )

        self.scene.obstacles = RigidObjectCollectionCfg(rigid_objects=objs)




        # reduce action scale
        # self.actions.joint_pos.scale = 0.5

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
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

        # REWARDS
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.params["threshold"] = 0.5 
         #-0.08     
        

        #self.rewards.undesired_contacts = None
        
        
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
#        self.rewards.track_vel_exp_product.weight = 10
        #self.rewards.com_over_support_h.weight = 10.0
        self.rewards.upright.weight = 0.5
        self.rewards.heading_align.weight = 0.25
        #self.rewards.trot_rew.weight = 10
#        self.rewards.progress_to_target.weight = 6.0 
        
        
        
        # PENALTIES
        
        
        #self.rewards.flat_orientation_l2.weight = 0.0    
          
        
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -1.0e-5 
        self.rewards.joint_vel_l2.weight = -1.0e-4 
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_pos_limits.weight = -0.05 
        self.rewards.feet_air_time.weight =  0.0
        self.rewards.feet_slide.weight =  0.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.alive.weight = 0.05
        self.rewards.feet_impact_vel.weight = 0.0
        
         
        
        #self.rewards.track_lin_vel_xy_mse.weight = -3.0 # penalty for not following desired direction
        #self.rewards.track_ang_vel_z_mse.weight = -1.5 # penalty for not following desired direction

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["trunk"] #, 'trunk', '.*_thigh', '.*_hip']
        self.rewards.termination_penalty.weight = -200.0
        
        # Commands
        #self.commands.base_velocity.ranges.lin_vel_x = (0.0, 2.0)
        #self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        #self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        


        # ---- Follow camera on trunk (no USD edits) ----
        cam_prim = "{ENV_REGEX_NS}/Robot/trunk/FollowCam"

        self.scene.follow_cam = CameraCfg(
            prim_path=cam_prim,
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-4.0, 0.0, 2.5),                    
                rot=(0.9762960, 0.0, 0.2164396, 0.0),    
                convention="world",
            ),
        )
        
    
