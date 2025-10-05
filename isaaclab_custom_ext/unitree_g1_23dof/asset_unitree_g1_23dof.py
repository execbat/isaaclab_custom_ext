# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`UNITREE_A1_CFG`: Unitree A1 robot with DC motor model for the legs
* :obj:`UNITREE_GO1_CFG`: Unitree Go1 robot with actuator net model for the legs
* :obj:`UNITREE_GO2_CFG`: Unitree Go2 robot with DC motor model for the legs
* :obj:`H1_CFG`: H1 humanoid robot
* :obj:`H1_MINIMAL_CFG`: H1 humanoid robot with minimal collision bodies
* :obj:`G1_CFG`: G1 humanoid robot
* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


import importlib.util
from pathlib import Path
import os

PKG = "isaaclab_custom_ext"
spec = importlib.util.find_spec(PKG)
pkg_dir = os.path.dirname(spec.origin)
##
# Configuration - Actuators.
##

usd_path = os.path.join(
    pkg_dir,
    "unitree_g1_23dof",
    "unitree_23dof_usd",
    "g1_23dof",
    "g1_23dof.usd",
)


"""Configuration for the Unitree G1 23DF AnimalMath Humanoid robot."""
MATH_G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4  # was switched off
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            #".*_hip_roll_joint": 0, #!
            #".*_hip_yaw_joint": 0,  #!
             
            ".*_knee_joint": 0.175, # 0.42, (10 degrees)
            ".*_ankle_pitch_joint": -0.23,
            #".*_ankle_roll_joint": 0,  #!
            
            #"waist_yaw_joint": 0,  #!
            
            ".*_elbow_joint": 1.39, #0.87,              
            "left_shoulder_roll_joint": 0.176,
            "right_shoulder_roll_joint": -0.176,
            ".*_shoulder_pitch_joint": 0.35,
            #".*_shoulder_yaw_joint": 0,  #!
            #".*_wrist_roll_joint": 0.87, #!
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "waist_yaw_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "waist_yaw_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint", 
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint"
                
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01

            },
        ),
    },
)




MATH_G1_23DF_CFG = MATH_G1_CFG.copy()
#MATH_G1_23DF_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd"
"""Configuration for the Unitree G1 Humanoid robot with fewer collision meshes"""


