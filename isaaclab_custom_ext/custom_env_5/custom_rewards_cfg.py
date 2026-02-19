import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg  
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import math

from .rewards import feet_impact_vel, pelvis_height_target_reward, leg_pelvis_torso_coalignment_reward, step_width_penalty, target_distance_exp_reward, feet_air_time_positive_biped

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    
    
    target_distance_exp = RewTerm(
        func=target_distance_exp_reward,
        weight=5.0,
        params={
            "alpha": 0.5,
            "use_xy": True,
            "max_dist": 10.0,
        },
    )   
 
    action_rate_l2 =      RewTerm(func=mdp.action_rate_l2,   weight=-0.001)
    dof_torques_l2 =      RewTerm(func=mdp.joint_torques_l2, weight=-1e-6)
    joint_vel_l2 =        RewTerm(func=mdp.joint_vel_l2,     weight= -1.0e-5)
    dof_acc_l2 =          RewTerm(func=mdp.joint_acc_l2,     weight=-2e-07)
    
    

    feet_air_time = None 
     
    feet_air_time_positive_biped = RewTerm(
        func=feet_air_time_positive_biped,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
            "threshold": 0.5,
        },
    )  

    

    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )   
    
    feet_impact_vel = RewTerm( 
        func=feet_impact_vel,
        weight=-0.0001,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg":  SceneEntityCfg("robot",          body_names=".*_ankle_roll_link"),
            "clip": 0.6,
            "contact_force_threshold": 5.0,
            "use_history": True,
        # "store_key": "_feet_prev_contact__foot"
        }
    )
    
    pelvis_height_target_reward = RewTerm( 
        func=pelvis_height_target_reward, weight=0.5)    

    
    termination_penalty = RewTerm(func=mdp.is_terminated,    weight=-50.0)  # -200.0
    lin_vel_z_l2 =        RewTerm(func=mdp.lin_vel_z_l2,     weight=-2.0)
    ang_vel_xy_l2 =       RewTerm(func=mdp.ang_vel_xy_l2,    weight=-0.05)
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,	
        weight=-0.5,
        #params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ "torso_link", "pelvis", ".*_hip_.*", ".*_wrist_.*", ".*shoulder_.*", ".*knee_.*", ".*elbow_.*"]),
        "threshold": 8.0}
    )
    
    
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    

    coalignment_chain = RewTerm( 
        func=leg_pelvis_torso_coalignment_reward,
        weight=2.0,  # 2
        params={
            "asset_cfg": SceneEntityCfg("robot"),

            
            "pelvis_body": "pelvis",
            "torso_body": "torso_link",
            "left_thigh_body":  "left_hip_pitch_link",   
            "left_shank_body":  "left_knee_link",        
            "right_thigh_body": "right_hip_pitch_link",
            "right_shank_body": "right_knee_link",

           
            "forward_local": (1.0, 0.0, 0.0),#

            # internal weights
            "w_yaw": 1.0,    
            "w_chain": 0.7,  
            "w_upright": 0.3, 
        },
    )
    
    body_lin_acc_l2 = RewTerm(func=mdp.body_lin_acc_l2, weight=-2.5e-6)   
    
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    
 
    step_width_penalty = RewTerm(
        func=step_width_penalty,
        weight=-1.0,  
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                         body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
            "asset_cfg":  SceneEntityCfg("robot",
                         body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
            "nominal_width": 0.25,
            #"beta": 20.0,
            "contact_force_threshold": 5.0,
            "use_history": True,
            "gate_by_support": True,
        },
    )
    
 
    
