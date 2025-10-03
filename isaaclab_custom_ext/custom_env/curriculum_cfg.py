import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
import .param_scheduler as ps



@configclass
class ChaseCurriculumCfg:
    # --- Phase A: Acceleration and balance (0 → 200k steps) ---
    # We immediately provide strong tracking, soft orientation, without penalties.
    # (track_lin_vel_xy_exp уже = 2.0 at start already)
    upr_0 = CurrTerm(func=mdp.modify_reward_weight,
                     params={"term_name": "upright", "weight": 0.5, "num_steps": 0})
    head_0 = CurrTerm(func=mdp.modify_reward_weight,
                      params={"term_name": "heading_align", "weight": 0.25, "num_steps": 0})
    idle_0 = CurrTerm(func=mdp.modify_reward_weight,
                      params={"term_name": "idle_penalty", "weight": 0.0, "num_steps": 0})  	
    track_lin_vel_xy_exp_0 = CurrTerm(func=mdp.modify_reward_weight,
                      params={"term_name": "track_lin_vel_xy_exp", "weight": 0.0, "num_steps": 0})      
    track_ang_vel_z_exp_0 = CurrTerm(func=mdp.modify_reward_weight,
                      params={"term_name": "track_ang_vel_z_exp", "weight": 0.0, "num_steps": 0})                                  
                      
                      
    '''                
    idle_pen_20k = CurrTerm(func=mdp.modify_reward_weight,
                      params={"term_name": "idle_penalty", "weight": -15.0, "num_steps": 20_000})
    bad_contacts_20k = CurrTerm(func=mdp.modify_reward_weight,
                      params={"term_name": "undesired_contacts", "weight": -2.0, "num_steps": 20_000})
    tr_lin_20k = CurrTerm(func=mdp.modify_reward_weight,
                           params={"term_name": "track_lin_vel_xy_exp", "weight": 2.0, "num_steps": 20_000})
    tr_yaw_20k = CurrTerm(func=mdp.modify_reward_weight,
                           params={"term_name": "track_ang_vel_z_exp", "weight": 1.0, "num_steps": 20_000})                       
    air_on_20k = CurrTerm(func=mdp.modify_reward_weight,
                          params={"term_name": "feet_air_time", "weight": 0.05, "num_steps": 20_000})
    trot_on_20k = CurrTerm(func=mdp.modify_reward_weight,
                          params={"term_name": "trot_rew", "weight": 0.5, "num_steps": 20_000})
    com_on_20k = CurrTerm(func=mdp.modify_reward_weight,
                          params={"term_name": "com_over_support_h", "weight": 1.0, "num_steps": 20_000})                      
                          
 
 
    imp_30k = CurrTerm(func=mdp.modify_reward_weight,
                        params={"term_name": "feet_impact_vel", "weight": -0.05, "num_steps": 30_000})

    slip_30k = CurrTerm(func=mdp.modify_reward_weight,
                         params={"term_name": "feet_slide", "weight": -0.05, "num_steps": 30_000}) 
    act_30k = CurrTerm(func=mdp.modify_reward_weight,
                        params={"term_name": "action_rate_l2", "weight": -0.005, "num_steps": 30_000})   
    tq_30k = CurrTerm(func=mdp.modify_reward_weight,
                       params={"term_name": "dof_torques_l2", "weight": -1e-5, "num_steps": 30_000}) 
    slip_30k = CurrTerm(func=mdp.modify_reward_weight,
                         params={"term_name": "feet_slide", "weight": -0.05, "num_steps": 30_000})                       

   '''
   #
   
   
    # REWARD WEIGHTS SMOOTH CHANGERS
    
    tr_lin_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.track_lin_vel_xy_exp.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": 20.0, "num_steps": 1_000, "start_after": 10_000},
        },
    )
    tr_yaw_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.track_ang_vel_z_exp.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": 20.0, "num_steps": 1_000, "start_after": 10_000},
        },
    )
    idle_for_cmd = CurrTerm(func=mdp.modify_reward_weight,
                      params={"term_name": "idle_penalty", "weight": -200.0, "num_steps": 10_000})    
    
    
    
#    air_warmup = CurrTerm(
#        func=mdp.modify_term_cfg,
#        params={
#            "address": "rewards.feet_air_time.weight", 
#            "modify_fn": ps.lerp_scalar,          
#            "modify_params": {"start": 0.5, 
#            "end": 0.5, 
#            "num_steps": 100_000,
#            "start_after": 20_000
#            },
#        },
#    )    
#    trot_warmup = CurrTerm(
#        func=mdp.modify_term_cfg,
#        params={
#            "address": "rewards.trot_rew.weight", 
#            "modify_fn": ps.lerp_scalar,          
#            "modify_params": {"start": 1.0, 
#            "end": 1.0, 
#            "num_steps": 100_000,s
#            "start_after": 20_000
#            },
#        },
#    )    
    com_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.com_over_support_h.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.5, "end": 0.5, "num_steps": 100_000},
        },
    )  
         
    # PENALTY WEIGHTS SMOOTH CHANGERS
    imp_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.feet_impact_vel.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.1, "num_steps": 10_000, "start_after": 40_000},
        },
    )       
    slip_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.feet_slide.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": -0.0, "end": -0.1, "num_steps": 10_000, "start_after": 40_000},
        },
    )    
    act_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.action_rate_l2.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.015, "num_steps": 10_000, "start_after": 40_000},
        },
    )     
    tq_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.dof_torques_l2.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -1e-4, "num_steps": 10_000, "start_after": 40_000},
        },
    )     
    lin_vel_z_l2_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.lin_vel_z_l2.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -2.0, "num_steps": 10_000, "start_after": 40_000},
        },
    )  
    ang_vel_xy_l2_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.ang_vel_xy_l2.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.05, "num_steps": 10_000, "start_after": 40_000},
        },
    ) 
    joint_vel_l2_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.joint_vel_l2.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -1.0e-4 , "num_steps": 10_000, "start_after": 40_000},
        },
    )
    dof_acc_l2_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.dof_acc_l2.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -1e-07 , "num_steps": 10_000, "start_after": 40_000},
        },
    )
    dof_pos_limits_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.dof_pos_limits.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": 0.0, "end": -0.05 , "num_steps": 10_000, "start_after": 40_000},
        },
    )
    undesired_contacts_warmup = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.undesired_contacts.weight", 
            "modify_fn": ps.lerp_scalar,          
            "modify_params": {"start": -5.0, "end": -5.0 , "num_steps": 100_000, "start_after": 40_000},
        },
    )
    
    
    
    # COMMANDS SMOOTH CHANGERS 
    cmd_lin_x_range_1 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": ps.lerp_tuple,
            "modify_params": {"start": (0.0, 0.0), 
            "end": (0.0, 1.0), 
            "num_steps": 1_000,
            "start_after": 10
            },
        },
    )
    cmd_lin_x_range_2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": ps.lerp_tuple,
            "modify_params": {"start": (0.0, 1.0), 
            "end": (0.0, 2.0), 
            "num_steps": 10_000,
            "start_after": 60_000
            },
        },
    )   
     
    cmd_yaw_range_sched_1 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.ang_vel_z",
            "modify_fn": ps.lerp_tuple,
            "modify_params": {
                "start": (0.0, 0.0),
                "end":   (-1.0, 1.0),
                "num_steps": 1_000,
                "start_after": 10
            },
        },
    )
    cmd_yaw_range_sched_2 = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.ang_vel_z",
            "modify_fn": ps.lerp_tuple,
            "modify_params": {
                "start": (-1.0, 1.0),
                "end":   (-2.0, 2.0),
                "num_steps": 10_000,
                "start_after": 60_000
            },
        },
    )







    '''
    cmd_easy_z_10   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.ang_vel_z", "value": (-0.6, 0.6), "num_steps": 10_000}) 
    cmd_easy_10   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.lin_vel_x", "value": (0.0, 0.6), "num_steps": 10_000})  
        
    cmd_easy_z_20   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.ang_vel_z", "value": (-0.7, 0.7), "num_steps": 20_000})         
    cmd_easy_20   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.lin_vel_x", "value": (0.0, 0.7), "num_steps": 20_000})          

    cmd_easy_z_30   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.ang_vel_z", "value": (-0.8, 0.8), "num_steps": 30_000}) 
    cmd_easy_30   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.lin_vel_x", "value": (0.0, 0.8), "num_steps": 30_000})     

    cmd_easy_z_40   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.ang_vel_z", "value": (-0.9, 0.9), "num_steps": 40_000})         
    cmd_easy_40   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.lin_vel_x", "value": (0.0, 0.9), "num_steps": 40_000})          

    cmd_easy_z_50   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.ang_vel_z", "value": (-1.0, 1.0), "num_steps": 50_000})         
    cmd_easy_50   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.lin_vel_x", "value": (0.0, 1.0), "num_steps": 50_000})  

    cmd_easy_z_60   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.ang_vel_z", "value": (-1.1, 1.1), "num_steps": 60_000})  
    cmd_easy_60   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.lin_vel_x", "value": (0.0, 1.2), "num_steps": 60_000})  
        
    cmd_easy_z_70   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.ang_vel_z", "value": (-1.2, 1.2), "num_steps": 70_000})          
    cmd_easy_70   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.lin_vel_x", "value": (0.0, 1.4), "num_steps": 70_000})          

    cmd_easy_z_80   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.ang_vel_z", "value": (-1.3, 1.3), "num_steps": 80_000})  
    cmd_easy_80   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.lin_vel_x", "value": (0.0, 1.6), "num_steps": 80_000})     

    cmd_easy_z_90   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.ang_vel_z", "value": (-1.4, 1.4), "num_steps": 90_000})          
    cmd_easy_90   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.lin_vel_x", "value": (0.0, 1.8), "num_steps": 90_000})          
 
    cmd_easy_z_100   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.ang_vel_z", "value": (-1.5, 1.5), "num_steps": 100_000})         
    cmd_easy_100   = CurrTerm(func=mdp.modify_env_param,
        params={"address": "commands.base_velocity.ranges.lin_vel_x", "value": (0.0, 2.0), "num_steps": 100_000})   
    '''                                                                             
    '''
    idle_pen_40k = CurrTerm(func=mdp.modify_reward_weight,
                      params={"term_name": "idle_penalty", "weight": -20.0, "num_steps": 40_000})
    bad_contacts_40k = CurrTerm(func=mdp.modify_reward_weight,
                      params={"term_name": "undesired_contacts", "weight": -3.0, "num_steps": 40_000})
    tr_lin_40k = CurrTerm(func=mdp.modify_reward_weight,
                           params={"term_name": "track_lin_vel_xy_exp", "weight": 4.0, "num_steps": 40_000})
    tr_yaw_40k = CurrTerm(func=mdp.modify_reward_weight,
                           params={"term_name": "track_ang_vel_z_exp", "weight": 4.0, "num_steps": 40_000})                       
    air_on_40k = CurrTerm(func=mdp.modify_reward_weight,
                          params={"term_name": "feet_air_time", "weight": 0.15, "num_steps": 40_000})
    trot_on_40k = CurrTerm(func=mdp.modify_reward_weight,
                          params={"term_name": "trot_rew", "weight": 2.0, "num_steps": 40_000})
                          
                          

    idle_pen_50k = CurrTerm(func=mdp.modify_reward_weight,
                      params={"term_name": "idle_penalty", "weight": -25.0, "num_steps": 50_000})
    bad_contacts_50k = CurrTerm(func=mdp.modify_reward_weight,
                      params={"term_name": "undesired_contacts", "weight": -5.0, "num_steps": 50_000})
    tr_lin_50k = CurrTerm(func=mdp.modify_reward_weight,
                           params={"term_name": "track_lin_vel_xy_exp", "weight": 5.0, "num_steps": 50_000})
    tr_yaw_50k = CurrTerm(func=mdp.modify_reward_weight,
                           params={"term_name": "track_ang_vel_z_exp", "weight": 5.0, "num_steps": 50_000})                       
    air_on_50k = CurrTerm(func=mdp.modify_reward_weight,
                          params={"term_name": "feet_air_time", "weight": 0.5, "num_steps": 50_000})
    trot_on_50k = CurrTerm(func=mdp.modify_reward_weight,
                          params={"term_name": "trot_rew", "weight": 5.0, "num_steps": 50_000})
    
    # --- Phase C: Tightening up the accuracy ---
    # We gradually increase the smoothing of actions and the penalty for the moment


    # ---------- ПРИМЕРЫ ДЛЯ РЕВАРДОВ ----------

    # 1) Ступенью (built-in) увеличить вес трекинга после 150k шагов: 2.0 -> 4.0
    tr_lin_up = CurriculumTermCfg(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "track_lin_vel_xy_exp",  # имя реварда как в RewardManager
            "weight": 4.0,
            "num_steps": 150_000,                 # после этого шага вес станет 4.0
        },
    )

    # 2) Плавный прогрев upright (0.2 -> 0.8 за 120k) через modify_env_param + планировщик
    upr_warmup = CurriculumTermCfg(
        func=mdp.modify_env_param,
        params={
            "address": "rewards.upright.weight",  # точечный путь в env.cfg
            "modify_fn": cosine_warmup,           # наша маленькая функция из schedules.py
            "modify_params": {"start": 0.2, "end": 0.8, "num_steps": 120_000},
        },
    )

    # 3) Постепенно усиливаем penalty на action_rate_l2 (-0.002 -> -0.01)
    act_rate_tighten = CurriculumTermCfg(
        func=mdp.modify_env_param,
        params={
            "address": "rewards.action_rate_l2.weight",
            "modify_fn": lerp_scalar,
            "modify_params": {"start": -0.002, "end": -0.01, "num_steps": 100_000},
        },
    )

    # ---------- ПРИМЕРЫ ДЛЯ КОМАНД ----------

    # 4) Расширяем диапазон лин. скорости по X: (-0.3..0.3) -> (-1.0..1.0)
    cmd_lin_x_range = CurriculumTermCfg(
        func=mdp.modify_env_param,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": lerp_tuple,
            "modify_params": {"start": (-0.3, 0.3), "end": (-1.0, 1.0), "num_steps": 100_000},
        },
    )

    # 5) Расширяем разрешённую угловую скорость yaw: (-0.3..0.3) -> (-1.0..1.0)
    cmd_yaw_range = CurriculumTermCfg(
        func=mdp.modify_env_param,
        params={
            "address": "commands.base_velocity.ranges.ang_vel_z",
            "modify_fn": lerp_tuple,
            "modify_params": {"start": (-0.3, 0.3), "end": (-1.0, 1.0), "num_steps": 120_000},
        },
    )

    # 6) Пример: вероятность «стоячих» эпизодов (если у команды есть такое поле)
    idle_prob = CurriculumTermCfg(
        func=mdp.modify_env_param,
        params={
            "address": "commands.base_velocity.rel_standing_envs",
            "modify_fn": lerp_scalar,
            "modify_params": {"start": 0.00, "end": 0.15, "num_steps": 80_000},
        },
    )

    # ---------- ПРИМЕР ИЗМЕНЕНИЯ ПАРАМЕТРА ТЕРМА (modify_term_cfg) ----------

    # 7) Допустим, у терма rewards.undesired_contacts есть параметр threshold
    #    (пример — как меняли бы вложенное поле конфигурации самого терма):
    #    address для modify_term_cfg — это (manager_name, term_name)
    #    а внутри modify_fn — работаешь с самим term_cfg (dict-like / dataclass)
    #    Подробнее см. секцию "Modify Term Configuration" в доках.
    #    Если не нужно — этот пример можно удалить.
    # undesired_contacts_threshold = CurriculumTermCfg(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "manager_name": "rewards",
    #         "term_name": "undesired_contacts",
    #         "modify_fn": your_fn,           # см. доку — modify_fn получает term_cfg
    #         "modify_params": {...},
    #     },
    # )
    '''
