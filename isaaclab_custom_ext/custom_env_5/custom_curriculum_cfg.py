from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .param_scheduler import lerp_scalar, lerp_tuple


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    
    # CONDITION BASED CURRICULUM CHANGER
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    
    '''
    # WEIGHT SCALAR RAPID CHANGERS  
    ar_50k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"action_rate_l2","weight":-0.02,"num_steps":50_000})
    tq_50k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"dof_torques_l2","weight":-3e-5,"num_steps":50_000})
    jv_50k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"joint_vel_l2","weight":-1e-4,"num_steps":50_000})
    ja_50k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"dof_acc_l2","weight":-1e-6,"num_steps":50_000})    
        

    ar_100k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"action_rate_l2","weight":-0.2,"num_steps":100_000})
    tq_100k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"dof_torques_l2","weight":-3e-4,"num_steps":100_000})
    jv_100k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"joint_vel_l2","weight":-1e-3,"num_steps":100_000})
    ja_100k = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name":"dof_acc_l2","weight":-1e-5,"num_steps":100_000})         
        
    
    # WEIGHT SCALAR SMOOTH CHANGERS    (bad practice to increase the reward weights smoothly) 
    tr_lin_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.track_lin_vel_xy_exp.weight", 
            "modify_fn": lerp_scalar,          
            "modify_params": {
                            "start": 0.0, 
                            "end": 2.0, 
                            "num_steps": 10_000, 
                            "start_after": 1_000,
                            "log_name": "track_lin_reward_weight"
                            },
        },
    )
    tr_yaw_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.track_ang_vel_z_exp.weight", 
            "modify_fn": lerp_scalar,          
            "modify_params": {
                            "start": 0.0, 
                            "end": 0.5, 
                            "num_steps": 10_000, 
                            "start_after": 1_000,
                            "log_name": "track_ang_reward_weight"
                            },
        },
    )
    

    
    # COMMAND RANGE SMOOTH CHANGERS 
    cmd_lin_x_range = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": lerp_tuple,
            "modify_params": {
                            "start": (0.0, 0.1), 
                            "end": (0.0, 1.0), 
                            "num_steps": 100_000, 
                            "start_after": 1_000,
                            "log_name": "cmd_lin_x_range"
                            },
            
        },
    )
    
    cmd_yaw_range_sched = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.ang_vel_z",
            "modify_fn": lerp_tuple,
            "modify_params": {
                            "start": (-0.1, 0.1),
                            "end":   (-1.0, 1.0),
                            "num_steps": 100_000,
                            "start_after": 1_000,
                            "log_name": "cmd_ang_yaw_range"
                            },
        },
    ) 
    '''
