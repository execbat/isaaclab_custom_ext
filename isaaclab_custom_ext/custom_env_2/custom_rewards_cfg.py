import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg  
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import math

from .rewards import feet_impact_vel, pelvis_height_target_reward, no_command_motion_penalty, lateral_slip_penalty, heading_alignment_reward, leg_pelvis_torso_coalignment_reward, idle_penalty, angvel_flat_l2_product, alternating_airtime_reward, step_phase_reward, com_projection_reward, step_width_penalty

@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
#    track_vel_exp_product = RewTerm(
#        func=angvel_flat_l2_product,
#        weight=50.0,  
#        params=dict(
#            command_name="base_velocity",
#            std=math.sqrt(0.25),
#        ),
#    )
    '''
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    ) 
     
    feet_air_time_positive_biped = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )  
    '''  
    
    action_rate_l2 =      RewTerm(func=mdp.action_rate_l2,   weight=-0.01)
    dof_torques_l2 =      RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    joint_vel_l2 =        RewTerm(func=mdp.joint_vel_l2,     weight= -1.0e-3)
    dof_acc_l2 =          RewTerm(func=mdp.joint_acc_l2,     weight=-2e-07)
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )   
    '''
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

    '''
    termination_penalty = RewTerm(func=mdp.is_terminated,    weight=-20.0)  # -200.0
    lin_vel_z_l2 =        RewTerm(func=mdp.lin_vel_z_l2,     weight=-2.0)
    ang_vel_xy_l2 =       RewTerm(func=mdp.ang_vel_xy_l2,    weight=-0.05)
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,	
        weight=-0.0005,
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
    
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    
    lateral_slip = RewTerm( 
        func=lateral_slip_penalty,
        weight=-0.1,
        params={"command_name": "base_velocity"}
    )    
    '''
    heading_align = RewTerm( 
        func=heading_alignment_reward,
        weight=0.5,
        params={"command_name": "base_velocity", "lin_cmd_threshold": 0.05, "beta": 4.0},
    )    
    '''
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
    
    body_lin_acc_l2 = RewTerm(func=mdp.body_lin_acc_l2, weight=-2.5e-6)   
    
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.6)
    
    idle_penalty = RewTerm(
        func = idle_penalty,
        weight = - 0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_cmd_speed": 0.1,
            "lin_speed_threshold": 0.05,
            "scale": 1.0,
        }
    )                

    '''
    alt_airtime_term = RewTerm(
        func=alternating_airtime_reward,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg":  SceneEntityCfg("robot",          body_names=".*_ankle_roll_link"),

            # --- commands and gates ---
            "lin_deadband": 0.03,                 # m/s: "command ≈ 0"
            "ang_deadband": 0.03,                 # rad/s

            # --- contacts ---
            "contact_force_threshold": 5.0,       # H: contact is considered if |F| > thr
            "use_history": True,                  # take Max based on the sensor history (resistant to noise)

            # --- target by leg flight time ---
            "target_swing_time": 0.8,             # sec - target time "leg in the air"
            "swing_sigma": 0.10,                  # sec is the bell width for exp(−(t−T)^2/σ^2)

            # --- weights and fines ---
            "idle_double_support_bonus_val": 1.0, # bonus at rest for bipedalism
            "touchdown_bonus": 1.0,               # bonus at the moment of touch, if swing≈target
            "shaping_weight": 0.3,                # soft bonus during the flight (every step)
            "same_lead_penalty": 0.5,             # penalty if the same leg "leads" in a row
            "flight_penalty": 1.0,                # penalty if both legs are in the air while moving
                },
    )   
    '''
            
    step_phase_reward = RewTerm(
        func=step_phase_reward,
        weight=4.0,   
        params={
        "command_name": "base_velocity",
        "sensor_cfg": SceneEntityCfg("contact_forces",
                                     body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
        "lin_deadband": 0.03,
        "amp_ref": 800.0,             
        "freq_gain_hz_per_mps": 2.0,  # at |v|=0.5 -> 1 Hz, at 1.0 -> 2 Hz
        "std_vel": 0.25,
        "use_history": True,
        },
    )  
    
    com_reward = RewTerm(
        func=com_projection_reward,
        weight=1.0,  
        params={
        "command_name": "base_velocity",
        "sensor_cfg": SceneEntityCfg("contact_forces",
                                     body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
        "asset_cfg":  SceneEntityCfg("robot",
                                     body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
        "lin_deadband": 0.03,
        "contact_force_threshold": 5.0,
        "use_history": True,
        "com_offset_gain": 0.15,   # at |v|=1 m/s we aim for a displacement of ≈0.15 m
        "max_offset": 0.25,
        "beta": 10.0,
        "no_support_penalty": 0.0, # if desired, you can enter a small penalty, for example 0.1
        },
    )

    step_width_penalty = RewTerm(
        func=step_width_penalty,
        weight=-0.5,  
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                         body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
            "asset_cfg":  SceneEntityCfg("robot",
                         body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
            "nominal_width": 0.20,
            #"beta": 20.0,
            "contact_force_threshold": 5.0,
            "use_history": True,
            "gate_by_support": True,
        },
    ) 
