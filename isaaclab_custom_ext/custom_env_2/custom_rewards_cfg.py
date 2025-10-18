import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg  
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import math

from .rewards import feet_impact_vel, pelvis_height_target_reward, no_command_motion_penalty, lateral_slip_penalty, heading_alignment_reward, leg_pelvis_torso_coalignment_reward

@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=4.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=4.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )    

    action_rate_l2 =      RewTerm(func=mdp.action_rate_l2,   weight=-0.015)
    dof_torques_l2 =      RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    joint_vel_l2 =        RewTerm(func=mdp.joint_vel_l2,     weight= -1.0e-4)
    dof_acc_l2 =          RewTerm(func=mdp.joint_acc_l2,     weight=-1e-07)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )   
    feet_impact_vel = RewTerm( 
        func=feet_impact_vel,
        weight=-0.001,
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
        func=pelvis_height_target_reward, weight=0.3)    


    termination_penalty = RewTerm(func=mdp.is_terminated,    weight=-200.0) 
    lin_vel_z_l2 =        RewTerm(func=mdp.lin_vel_z_l2,     weight=-0.02)
    ang_vel_xy_l2 =       RewTerm(func=mdp.ang_vel_xy_l2,    weight=-0.0005)
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,	
        weight=-0.005,
        #params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ "torso_link", "pelvis", ".*_hip_.*", ".*_wrist_.*", ".*shoulder_.*", ".*knee_.*", ".*elbow_.*"]),
        "threshold": 8.0}
    )
    
    no_cmd_motion = RewTerm( 
        func=no_command_motion_penalty,
        weight=-0.1,   
        params={
            "command_name": "base_velocity",
            "lin_deadband": 0.03,   # sensitivity to "zero" linear command (m/s)
            "ang_deadband": 0.03,   # sensitivity to "zero" angular command (rad/s)
            "lin_scale": 0.6,       # expected operating Vmax ~0.6 m/s
            "ang_scale": 1.0,       # expected working Wmax ~1 rad/s
        },
    )
    
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.005)
    
    lateral_slip = RewTerm( 
        func=lateral_slip_penalty,
        weight=-0.001,
        params={"command_name": "base_velocity"}
    )    
 
    heading_align = RewTerm( 
        func=heading_alignment_reward,
        weight=0.1,
        params={"command_name": "base_velocity", "lin_cmd_threshold": 0.05, "beta": 4.0},
    )    
    
    coalignment_chain = RewTerm( 
        func=leg_pelvis_torso_coalignment_reward,
        weight=0.5,  # 2
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
       
    body_lin_acc_l2 = RewTerm(func=mdp.body_lin_acc_l2, weight=-2.5e-7)   
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5) 
    
                
        
