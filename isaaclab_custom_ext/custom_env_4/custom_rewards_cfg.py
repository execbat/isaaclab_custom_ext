import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg  
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import math

from .rewards import feet_impact_vel, pelvis_height_target_reward, no_command_motion_penalty, lateral_slip_penalty, heading_alignment_reward, leg_pelvis_torso_coalignment_reward, idle_penalty, angvel_flat_l2_product, alternating_airtime_reward, step_phase_reward, com_projection_reward, step_width_penalty, track_lin_vel_xy_exp_custom, track_ang_vel_z_exp_custom, foot_symmetry_step_reward_cmddir, target_distance_exp_reward

@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
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
    
    action_rate_l2 =      RewTerm(func=mdp.action_rate_l2,   weight=-0.001)
    dof_torques_l2 =      RewTerm(func=mdp.joint_torques_l2, weight=-1e-6)
    joint_vel_l2 =        RewTerm(func=mdp.joint_vel_l2,     weight= -1.0e-5)
    dof_acc_l2 =          RewTerm(func=mdp.joint_acc_l2,     weight=-2e-07)
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5, 
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
        weight = - 10.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),

            # --- linear component ---
            "lin_speed_threshold": 0.03,  # m/s: actually "almost standing"
            "lin_scale": 1.0,             # additional scale for linear penalty

            # --- corner component around z ---
            "ang_speed_threshold": 0.03,  # rad/s: essentially "almost no rotation"
            "ang_scale": 1.0,             # additional scale for corner penalty

            # --- deadbands for readability (duplicate the meanings of min_cmd_* above) ---
            "lin_deadband": 0.03,         # m/s: "command ≈ 0"
            "ang_deadband": 0.03,         # rad/s: "command ≈ 0"
        }
    )                

    
    alt_airtime_term = RewTerm(
        func=alternating_airtime_reward,
        weight=4.0, # 4.0
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg":  SceneEntityCfg("robot",          body_names=".*_ankle_roll_link"),
            # --- command gating ---
            "lin_deadband": 0.03,     # m/s: below -> linear command considered "near zero"
            "ang_deadband": 0.03,     # rad/s: below -> angular command considered "near zero"

            # --- contacts ---
            "contact_force_threshold": 5.0,  # N: contact if |F| > threshold
            "use_history": True,             # robust to noise (use max over history window)

            # --- target swing timing (per-step shaping while airborne) ---
            "target_swing_time": 1.0,       # s: desired airborne duration per leg
            "swing_sigma": 0.10,             # s: Gaussian width; smaller = stricter to target

            # --- swing time cap ---
            "max_swing_time": 1.5,           # s: hard cap (excess is penalized every step)
            "excess_penalty_scale": 1.0,     # penalty per 1s of excess per step

            # --- helpers ---
            "same_lead_penalty": 0.4,        # penalty if the same leg "leads" twice in a row
            "flight_penalty": 1.0,           # penalty when both feet are airborne while moving
            "idle_double_support_bonus_val": 1.0,  # bonus for double support at rest
        },
    )   
    '''
    foot_symmetry_step_reward = RewTerm(
        func=foot_symmetry_step_reward_cmddir,
        weight=6.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg":  SceneEntityCfg("robot",          body_names=".*_ankle_roll_link"),
            # --- command gating ---
            "lin_deadband": 0.03,     # m/s
            "ang_deadband": 0.03,     # rad/s

            # --- contacts ---
            "contact_force_threshold": 5.0,  # N: contact if |F| > threshold
            "use_history": True,              # robust to noise (max over history)

            # --- touchdown kernel (direction-aware, nonnegative) ---
            "sym_lambda": 0.06,       # m: exp(-|x_td + dir*x_lo| / sym_lambda)
            "sign_margin": 0.0,       # m: tolerance around 0 for sign checks
            
            # --- penalty scale when sign is wrong ---
            "wrong_sign_penalty_scale":  0.0, #0.10,  # multiplies core and flips sign: reward -> -scale*reward

            # --- standing preference (nonnegative) ---
            "stand_sigma": 0.08,      # m
            "stand_bonus": 1.0,       # scale
        },
    )   
    


    
            
    step_phase_reward = RewTerm(
        func=step_phase_reward,
        weight=4.0,   
        params={
        "command_name": "base_velocity",
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
        # --- commands and gates ---
        "lin_deadband": 0.03,      # m/s: "almost stopped" (linear)
        "ang_deadband": 0.03,      # rad/s: "almost stopped" (angular)
        "use_history": True,       # take max over history for robustness

        # --- contact force ---
        "contact_force_threshold": 5.0,   # H: optional threshold (0 - no threshold)
        "amp_ref": 400.0,                 # H: desired max force for normalization and reference A

        # --- phase generator ---
        "freq_gain_hz_per_mps": 0.5, # 2.0,     # f = k_f * |v|; at |v|=0.5 => 1 Hz; at |v|=1.0 => 2 Hz
        "clamp_freq": (0.0, 4.0),        

        # --- exponent from MAE (gaussian kernel) ---
        "std_vel": 0.25,                 # controls sharpness of exp decay
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
    '''
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
    
    target_distance_exp = RewTerm(
        func=target_distance_exp_reward,
        weight=20.0,
        params={
            "alpha": 0.5,
            "use_xy": True,
            "max_dist": 10.0,
        },
    )    
     
