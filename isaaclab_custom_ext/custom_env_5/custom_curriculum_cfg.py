from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .curriculums import DifficultyScheduler, initial_final_interpolate_fn
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    

    adr = CurrTerm(
        func=DifficultyScheduler,
        params={
            "init_difficulty": 0,
            "min_difficulty": 0,
            "max_difficulty": 10,
            "object_cfg": SceneEntityCfg("target"), 

            "term_name": "target_distance_exp",
            "promote_threshold": 0.7,
            "demote_threshold": 0.3,
            "ema_alpha": 0.05,
            "warmup_steps": 1000,
        },
    )
    

    joint_pos_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.joint_pos.noise.n_min",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.1, "difficulty_term_str": "adr"},
        },
    )
    joint_pos_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.joint_pos.noise.n_max",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.1, "difficulty_term_str": "adr"},
        },
    )    
    
    joint_vel_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.joint_vel.noise.n_min",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.1, "difficulty_term_str": "adr"},
        },
    )
    joint_vel_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.joint_vel.noise.n_max",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.1, "difficulty_term_str": "adr"},
        },
    )  
    
    cam_rgb_feat_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.cam_rgb_feat.noise.n_min",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.1, "difficulty_term_str": "adr"},
        },
    )
    cam_rgb_feat_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.cam_rgb_feat.noise.n_max",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.1, "difficulty_term_str": "adr"},
        },
    )      
    
