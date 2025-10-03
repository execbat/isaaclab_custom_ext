# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, MathManagerBasedRLEnvCfg, MathManagerBasedRLEnv, MathTeleopManagerBasedRLEnvCfg, MathTeleopManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
#import isaac_hydra_ext.source.isaaclab_tasks.manager_based.locomotion.velocity.config.go1.env_scene as mdp

from isaaclab.envs.mdp.curriculums import modify_env_param

from isaac_hydra_ext.source.isaaclab_tasks.manager_based.locomotion.velocity.config.go1.env_scene import ObstacklesSceneCfg, ChaseTrainCommandsCfg, ChaseTestCommandsCfg, ChaseObservationsCfg, ChaseTerminationsCfg, ChaseTrainEventCfg, ChaseTestEventCfg, ChaseCurriculumCfg


##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )
    
@configclass
class MathCommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

    target_joint_pose = mdp.UniformVectorCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 15.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        dim = 23,
        ranges=((-1.0, 1.0),) * 23,
        
    )
    
    dof_mask = mdp.BernoulliMaskCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 15.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        dim = 23,
        ranges=((-1.0, 1.0),) * 23,
        
    )        
    
    
@configclass
class MathTeleopCommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

    target_joint_pose = mdp.UniformVectorCommandCfg(
        asset_name="robot",
        resampling_time_range=(9999999999999999.0, 9999999999999999.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        dim = 23,
        ranges=((-1.0, 1.0),) * 23,
        
    )
    
    dof_mask = mdp.BernoulliMaskCommandCfg(
        asset_name="robot",
        resampling_time_range=(9999999999999999.0, 9999999999999999.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        dim = 23,
        ranges=((-1.0, 1.0),) * 23,
        
    )    
#    # ── 1. Команда «нормированная поза суставов» длиной 23 ────────────────────
#    target_joint_pose = mdp.UniformVectorCommandCfg(
#        asset_name="robot",
#        dim=23,
#        resampling_time_range=(10.0, 10.0),       # менять один раз в 10 сек
#        debug_vis=True,
#        ranges=((-1.0, 1.0),) * 23,               # одинаковый интервал для всех
#    )

#    # ── 2. Команда «маска активных DOF» длиной 23 (0.0 / 1.0) ─────────────────
#    dof_mask = mdp.BernoulliMaskCommandCfg(
#        asset_name="robot",
#        dim=23,
#        resampling_time_range=(10.0, 10.0),
#        debug_vis=True,
#        p_one=0.5,                               # 50 % шанса, что элемент == 1.0
#    )   


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True) # was True


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
    
@configclass
class MathObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)) # data of the robot object from virtual env
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # data of the robot object from virtual env
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) # lin_vel_x (0.0, 2.0), lin_vel_y (-0.0, 0.0), ang_vel_z (-1.0, 1.0)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        
        ############################
        # new values for AnimalMath
        axis_act_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        
        # generated from commands from Command manager.
        # arget_axis_cmd_norm = ObsTerm(func=mdp.target_axis_cmd_norm)
        # target_axis_swtchr_mask = ObsTerm(func=mdp.target_axis_swtchr_mask)
        
        # exctract observation from commands
        target_axis_cmd_norm = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_joint_pose"})
        target_axis_swtchr_mask = ObsTerm(func=mdp.generated_commands, params={"command_name": "dof_mask"})
        ############################
        
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()    
    
    
@configclass
class MathTeleopObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)) # data of the robot object from virtual env
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)) # data of the robot object from virtual env
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}) # lin_vel_x (0.0, 2.0), lin_vel_y (-0.0, 0.0), ang_vel_z (-1.0, 1.0)
        velocity_commands = ObsTerm(func=mdp.commands_udp_read) # udp_read_commands
        
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        
        ############################
        # new values for AnimalMath
        axis_act_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        
        # generated from commands from Command manager.
        # target_axis_cmd_norm = ObsTerm(func=mdp.target_axis_cmd_norm)
        # target_axis_swtchr_mask = ObsTerm(func=mdp.target_axis_swtchr_mask)
        
        # target_axis_cmd_norm = ObsTerm(func=mdp.target_axis_cmd_norm_udp_read)
        # target_axis_swtchr_mask = ObsTerm(func=mdp.target_axis_swtchr_mask_udp_read)
        
        # exctract observation from commands
        target_axis_cmd_norm = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_joint_pose"})
        target_axis_swtchr_mask = ObsTerm(func=mdp.generated_commands, params={"command_name": "dof_mask"})
        ############################
        
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()     


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp_custom, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.35), "DEBUG": False}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp_custom, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.35), "DEBUG": False}
    )
    #track_vel_exp_product = RewTerm(
    #    func=mdp.track_lin_ang_vel_exp_product,
    #    weight=5.0,  
    #    params=dict(
    #        command_name="base_velocity",
    #        std=math.sqrt(0.5),
    #    ),
    #)
    
    com_over_support_h = RewTerm(
        func=mdp.com_over_support_height_reward_fast,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg":  SceneEntityCfg("robot",           body_names=".*_foot"),
            "contact_force_threshold": 5.0,     
            "target_height": 0.33,
            "height_tolerance": 0.05,
            "slope_aware": True,
            "weighted": True,
            "inside_margin": 0.08,
            "beta_inside": 4.0,
            "weight_height": 1.0,
            "weight_inside": 1.0,
            "DEBUG": False,                     
            "DEBUG_MAX_ENVS": 1,
        },
	    )
    


    
#    progress_to_target = RewTerm(
#        func=mdp.progress_towards_target,
#        weight=6.0, 
#        params={"scale": 1.0, "clamp": 0.5, "stop_radius": 0.35, "near_bonus": 2.0},
#    )
    
#    heading_to_target = RewTerm(
#        func=mdp.heading_alignment_to_target,
#        weight=1.5,
#        params={"stop_radius": 0.35},
#    )    
    upright = RewTerm(
        func=mdp.trunk_upright_alignment,
        weight=0.5,
        params={"body_up_axis": "z", "power": 1.0, "DEBUG" : False},
    )
    
    heading_align = RewTerm(
        func=mdp.heading_alignment_reward,
        weight=0.25,
        params={"command_name": "base_velocity", "lin_cmd_threshold": 0.05, "beta": 4.0},
    )     
    
    trot_rew = RewTerm(
        func=mdp.trot_rhythm_reward,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg":  SceneEntityCfg("robot",           body_names=".*_foot"),
            "target_hz": 2.0,
            "duty": 0.5,
            "contact_force_threshold": 5.0,
            "leg_order": [0,1,2,3],
            "phase_offsets": [0.0, math.pi, math.pi, 0.0],
            "k_match": 10.0,
            "w_phase": 1.0,
            "w_balance": 0.25,
            # new (optional):
            "tau_leak": 0.20,
            "alpha_phase": 1.0,
            "beta_balance": 1.0,
            "gamma_gate": 2.0,
            "zeta_stance": 1.0,
            "k_contact_sigmoid": 5.0,
            "eps_reward": 0.02,
            "DEBUG": False,
        },
    )

    
    # -- penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0.0)   
    joint_vel_l2  = RewTerm(func=mdp.joint_vel_l2,  weight=-1.0e-4)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.00)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.05)
    
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), 
            "command_name": "base_velocity",
            "threshold": 6.0,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg":  SceneEntityCfg("robot",          body_names=".*_foot"),
        },
    )    
    
    
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["trunk", ".*calf.*", ".*hip.*", ".*thigh", ]), "threshold": 10.0}, # question about calf  , ".*calf.*"
    )
    
    alive = RewTerm(func=mdp.is_alive, weight=0.05)
    feet_impact_vel = RewTerm(
        func=mdp.feet_impact_vel,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg":  SceneEntityCfg("robot",          body_names=".*_foot"),
            "clip": 0.6,
            "contact_force_threshold": 5.0,
            "use_history": True,
            # "store_key": "_feet_prev_contact__foot"
        }
    )
    idle_penalty = RewTerm(
        func = mdp.idle_penalty,
        weight = -10.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_cmd_speed": 0.1,
            "lin_speed_threshold": 0.05,
            "scale": 1.0,
        }
    )
    # -- optional penalties
#    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    
    
#    track_lin_vel_xy_mse = RewTerm(
#        func=mdp.track_lin_vel_xy_mse, weight=-3.0, params={"command_name": "base_velocity"}
#    )
#    track_ang_vel_z_mse = RewTerm(
#        func=mdp.track_ang_vel_z_mse, weight=-0.8, params={"command_name": "base_velocity"}
#    )
    
@configclass
class MathRewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # reward for following target axis values with axis_mask == 1 # AnumalMath
    # miander_tracking_reward = RewTerm(func=mdp.miander_tracking_reward, weight=4.0) # 4
    #miander_tracking_reward = RewTerm(
    #    func=mdp.miander_tracking_reward_exp,
    #    weight=4.0,
    #    params={"eps_half": 0.25}  
    #)
    
    # reward for following init axis values with axis_mask == 0 # AnumalMath
   #  miander_untracking_reward = RewTerm(func=mdp.miander_untracking_reward, weight=2.0) # 2
    
#    miander_tracking_reward = RewTerm(
#        func=mdp.miander_tracking_reward_exp,
#        weight=10.0,                 
#        params={"std": 0.25},        # 0.25 норм. ошибки ≈ e^{-1} ~ 0.37
#    )

#    miander_untracking_reward = RewTerm(
#        func=mdp.miander_untracking_reward_exp,
#        weight=10.0,                   
#        params={
#            "std": 0.25,               # ~норм. ошибка 0.25 даёт e^{-1} ≈ 0.37
#            "command_name": "base_velocity",
#            "lin_cmd_threshold": 0.05,
#            "leg_bits": [0,1,3,4,7,8,11,12,15,16,19,20],
#            "mask_name": "dof_mask",
#            "init_attr": "JOINT_INIT_POS_NORM",
#        },
#    )
    
    # 1) прогресс к цели по маскированным суставам
    #masked_progress = RewTerm(
    #    func=mdp.masked_progress_reward,
    #    weight=5.0,                      #  2.0–4.0
    #    params={}                        # опционально: {"eps": 0.02}
    #)
#    masked_progress_potential = RewTerm(
#        func=mdp.masked_progress_reward,
#        weight=3.0,
#        params={"eps": 0.002, "neg_scale": 0.1, "mask_name": "dof_mask"},
#    )
    
#    unmasked_progress_init = RewTerm(
#        func=mdp.unmasked_progress_to_init_reward,
#        weight=3.0,   # стартово мягко; подстрой после первых запусков
#        params=dict(
#            mask_name="dof_mask",
#            eps=0.002,           # мёртвая зона против дрожи
#            neg_scale=0.1,       # ослабление регресса
#            command_name="base_velocity",  # терм, по которому детектим движение
#            lin_deadband=0.03,
#            ang_deadband=0.03,
#            leg_bits=[0,1,3,4,7,8,11,12,15,16,19,20],  # индексы осей ног
#            init_attr="JOINT_INIT_POS_NORM",           # где кэшируем инит в [-1,1]
#        ),
#    )    
    
    
    
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=4.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
        
    pelvis_height_target_reward = RewTerm(
        func=mdp.pelvis_height_target_reward, weight=2.0)
        
        
    # -- penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.015)
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        #params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ "torso_link", "pelvis", ".*_hip_.*", ".*_wrist_.*", ".*shoulder_.*", ".*knee_.*", ".*elbow_.*"]),
        "threshold": 2.0}
    )
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-5.0e-2, # NEGATIVE ONLY
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )   

    
    feet_sep_align = RewTerm(
        func=mdp.feet_separation_and_alignment_penalty,
        weight=-5.0e-1,   # штраф → отрицательный вес
        params={
            # ВАЖНО: порядок тел — [левая, правая]
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),

            "ankle_pitch_joint_names": ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
            "w_ankle_neutral": 2.0,           # при необходимости можно временно поднять до 3–4

            "command_name": "base_velocity",
            "contact_force_threshold": 1.0,

            # Доп. термы (можно ослабить, т.к. основной — ankle_neutral)
            "w_tilt": 1.0,
            "foot_normal_local":  (0.0, 0.0, 1.0),
            "foot_forward_local": (1.0, 0.0, 0.0),
            "w_align": 0.5,
            "w_stride": 0.7,
            "step_gain": 0.40,
            "beta_stride": 3.0,
            "v_near_zero": 0.05,
            "w_opposite_at_zero": 0.5,

            "w_cross": 0.7,
            "beta_cross": 6.0,

            "w_sep": 2.0, # penalty weight for big dist btw legs
            "shoulder_width": 0.2,
            "beta_sep": 2.0,
        },
    )

    no_cmd_motion = RewTerm(
        func=mdp.no_command_motion_penalty,
        weight=-5.0e-1,   
        params={
            "command_name": "base_velocity",
            "lin_deadband": 0.03,   # чувствительность к «нулевой» линейной команде (м/с)
            "ang_deadband": 0.03,   # чувствительность к «нулевой» угловой команде (рад/с)
            "lin_scale": 0.6,       # ожидаемая рабочая Vmax ~0.6 м/с
            "ang_scale": 1.0,       # ожидаемая рабочая Wmax ~1 рад/с
        },
    )


    
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0) # MAKE WEIGHT MORE THAN 0 ELSE INGORED!
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.05)
    

    
    
    # === новые, «чувствительность к таргетам и маске» ===



#    # 2) тишина на НЕмаскированных (скорость/момент)
#    unmasked_stillness = RewTerm(
#        func=mdp.unmasked_stillness_penalty,
#        weight=-1.0,                     # при надобности усилить до -2.0
#        params={"w_vel": 1.0, "w_tau": 0.1}
#    )

#    # 3) бонус «все активные в допуске»
#    masked_success = RewTerm(
#        func=mdp.masked_success_bonus,
#        weight=50.0,                      # можно поднять до 1.0
#        params={"eps": 0.03, "bonus": 1.0}
#    )

    # 4) штраф за боковой слип относительно направления команды
#    lateral_slip = RewTerm(
#        func=mdp.lateral_slip_penalty,
#        weight=-1.0,
#        params={"command_name": "base_velocity"}
#    )
    
#    com_over_support = RewTerm(
#        func=mdp.com_over_support_reward_fast,   # твоя экспо-версия
#        weight=3.0,
#        params={
#            "sensor_cfg": SceneEntityCfg(
#                "contact_forces",
#                # выбери один вариант:
#                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],  # точные имена
#                # или — если хочешь по маске:
#                # body_names=".*_ankle_roll_link",
#            ),
#            "asset_cfg": SceneEntityCfg("robot"),
#            "contact_force_threshold": 5.0,
#            # вместо старого `sigma` теперь используем eps_half
#            # если раньше стояло sigma=0.06 → eps_half ≈ 1.177*0.06 = 0.0706
#            "sigma"   : 0.06,
#            "weighted": True,
#            
#            # new:
#            "require_both_when_idle": True,
#            "mask_name": "dof_mask",
#            "lin_deadband": 0.03,
#            "ang_deadband": 0.03,
#            "leg_bits": (0,1,3,4,7,8,11,12,15,16,19,20),
#        },
#    )
    
    limit_saturation = RewTerm(
        func=mdp.joint_limit_saturation_penalty,
        weight=-1.0,   
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "margin": 0.05,      # можно 0.08–0.15
            "beta": 8.0,
            "use_mask": False,   # или True, если хочешь штрафовать только активные DOF
            "mask_name": "dof_mask",
        },
    ) 
    
    single_foot_stationary = RewTerm(
        func=mdp.single_foot_stationary_penalty,
        weight=-5.0e-2,   
        params={
            # сенсор и ссылки на стопы
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "asset_cfg":  SceneEntityCfg("robot"),

            # пороги
            "contact_force_threshold": 2.0,   # Н/стопу для «есть контакт»
            "lin_deadband": 0.03,             # м/с — «ноль» по лин. скорости
            "ang_deadband": 0.03,             # рад/с — «ноль» по рысканью

            # команды движения
            "command_name": "base_velocity",  # как в track_* термах

            # маска с ЮДП (оси ног)
            "use_mask": True, #True, !!!!!!!!!!!!!!!!!!!!
            "mask_name": "dof_mask",
            # индексы осей ног в порядке сверху-вниз из твоей таблицы:
            # 0: L hip pitch, 1: R hip pitch, 3: L hip roll, 4: R hip roll,
            # 7: L hip yaw, 8: R hip yaw, 11: L knee, 12: R knee,
            # 15: L ankle pitch, 16: R ankle pitch, 19: L ankle roll, 20: R ankle roll
            "leg_bits": [0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20],

            # опционально: масштабировать штраф по асимметрии опор
            "scale_by_asymmetry": True,
            "eps": 1e-6,
        },
    )
    
    leg_symmetry_idle = RewTerm(
        func=mdp.leg_symmetry_idle_reward_norm,
        weight=1.0,   # диапазон 2.0–5.0
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "dof_mask_name": "dof_mask",
            "obs_key_norm": "dof_pos_norm",  # ключ в obs с норм-позами [-1..1]; если нет — посчитаем сами

            # «почти нулевые» команды
            "lin_deadband": 0.03,
            "ang_deadband": 0.03,

            # индексы ног из твоего списка (сверху-вниз)
            "left_dofs":  [0, 3, 7, 11, 15, 19],   # L hip pitch/roll/yaw, knee, ankle pitch/roll
            "right_dofs": [1, 4, 8, 12, 16, 20],

            # веса и «резкость» колпаков
            "w_sym": 0.6, "w_init": 0.4,
            "beta_sym": 6.0, "beta_init": 4.0,
            "eps": 1e-6,
        },
    )
    
    alternating_steps_reward = RewTerm(
        func=mdp.alternating_step_reward,
        weight=2.0,   # 2.0–4.0
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "asset_cfg":  SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "lin_cmd_threshold": 0.05,
            "use_mask": True, #True, !!!!!!!!!!!!!!!!!
            "mask_name": "dof_mask",
            "leg_bits": [0,1,3,4,7,8,11,12,15,16,19,20],
            "contact_force_threshold": 5.0,
            "initial_lead": "right",
            "step_gain": 0.40,
            "beta_stride": 3.0,
        },
    )    
    
    same_lead_penalty = RewTerm(
        func=mdp.alternating_same_lead_penalty,
        weight=-5.0e-2,   # стартово -1.5…-3.0
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "asset_cfg":  SceneEntityCfg("robot"),
            "command_name": "base_velocity",

            "lin_cmd_threshold": 0.05,
            "use_mask": True,
            "mask_name": "dof_mask",
            "leg_bits": [0,1,3,4,7,8,11,12,15,16,19,20],
            "contact_force_threshold": 5.0,
            "step_gain": 0.40,
            "beta_stride": 3.0,
        },
    )   
    
    
    heading_align = RewTerm(
        func=mdp.heading_alignment_reward,
        weight=1.0,
        params={"command_name": "base_velocity", "lin_cmd_threshold": 0.05, "beta": 4.0},
    )       
    
    swing_clearance = RewTerm(
        func=mdp.swing_foot_clearance_reward,
        weight=0.5,   # 1.0–2.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link","right_ankle_roll_link"]),
            "asset_cfg":  SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "contact_force_threshold": 5.0,
            "h_des": 0.06,
            "beta": 120.0,
        },
    )
    
#    masked_action_rate = RewTerm(func=mdp.masked_action_rate_l2, weight=-0.05, params={"mask_name": "dof_mask"})
    
#    masked_success_stable = RewTerm(
#        func=mdp.masked_success_stable_bonus,
#        weight=100.0,
#        params={
#            "eps": 0.08,      # начни мягко (0.06–0.10), потом ужесточай куррикулумом до 0.02–0.03
#            "vel_eps": 0.08,
#            "bonus": 1.0,
#            "mask_name": "dof_mask",
#            
#        },
#    )

    coalignment_chain = RewTerm(
        func=mdp.leg_pelvis_torso_coalignment_reward,
        weight=2.0,  # 2
        params={
            "asset_cfg": SceneEntityCfg("robot"),

            # имена звеньев из твоего скрина:
            "pelvis_body": "pelvis",
            "torso_body": "torso_link",
            "left_thigh_body":  "left_hip_pitch_link",   # thigh ~ бедро
            "left_shank_body":  "left_knee_link",        # shank ~ голень
            "right_thigh_body": "right_hip_pitch_link",
            "right_shank_body": "right_knee_link",

            # у этих линков «вперёд» обычно локальный +X
            "forward_local": (1.0, 0.0, 0.0),

            # внутренние веса
            "w_yaw": 1.0,     # сонаправленность c тазом
            "w_chain": 0.7,   # согласованность бедро↔голень
            "w_upright": 0.3, # чуть поощряем горизонтальность таза/торса
        },
    )

    
    idle_double_support = RewTerm(
        func=mdp.idle_double_support_bonus,
        weight=1.0,   # 1
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "command_name": "base_velocity",
            "contact_force_threshold": 5.0,
            "lin_deadband": 0.03,
            "ang_deadband": 0.03,
            "bonus": 1.0,
            "leg_bits": (0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20),
        },
    )
    
    body_lin_acc_l2 = RewTerm(func=mdp.body_lin_acc_l2, weight=-2.5e-7) # НОВОЕ (при дрожи/подпрыг.) 

    # --- (опционально) анти-залипание в одноопоре ---
    prolonged_single_support = RewTerm(
        func=mdp.prolonged_single_support_penalty,
        weight=-5.0e-1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "command_name": "base_velocity",
            "contact_force_threshold": 3.0,
            "lin_deadband": 0.03,
            "ang_deadband": 0.03,
            "use_mask": True, #True, !!!!!!!!!!!!!!!!!!!!!!!!
            "mask_name": "dof_mask",
            "left_dofs":  (0, 3, 7, 11, 15, 19),
            "right_dofs": (1, 4, 8, 12, 16, 20),
        },
    )

#    # mask=1 → к таргету
#    target_proximity_exp_vel = RewTerm(
#        func=mdp.masked_target_proximity_reward_exp_vel,
#        weight= 50, #50.0,
#        params=dict(
#            mask_name="dof_mask",
#            std_pos=1.5, #0.7,  
#            std_vel=0.8,
#            w_pos=1.0,  w_vel=0.4,
#            gate=0.7,     # скорость «включается» ближе к цели
#            init_attr="JOINT_INIT_POS_NORM",
#        ),
#    )

#    # mask=0 → к иниту (ноги выкидываем при движении)
#    unmasked_init_proximity_exp_vel = RewTerm(
#        func=mdp.unmasked_init_proximity_reward_exp_vel,
#        weight=10, #10.0,
#        params=dict(
#            mask_name="dof_mask",
#            exclude_legs_when_moving = True, # was True
#            std_pos=1.5, #0.7, 
#            std_vel=0.8,
#            w_pos=1.0,  w_vel=0.4,
#            gate=0.7,
#            command_name="base_velocity",
#            lin_deadband=0.03, ang_deadband=0.03,
#            leg_bits=[0,1,3,4,7,8,11,12,15,16,19,20],
#            init_attr="JOINT_INIT_POS_NORM",
#        ),
#    )

#    masked_far_from_target_penalty = RewTerm(
#        func=mdp.masked_near_target_penalty_linear,
#        weight=-1.0,
#        params=dict(mask_name="dof_mask"),
#    )

#    unmasked_far_from_init_penalty = RewTerm(
#        func=mdp.unmasked_near_init_penalty_linear,
#        weight=-0.1,
#        params=dict(
#            mask_name="dof_mask",
#            exclude_legs_when_moving= False, # was True
#            command_name="base_velocity",
#            lin_deadband=0.03, ang_deadband=0.03,
#            leg_bits=[0,1,3,4,7,8,11,12,15,16,19,20],
#            init_attr="JOINT_INIT_POS_NORM",
#        ),
#    )

    # mask=1 → to target pose with zero-velocity bonus near target
    target_proximity_exp_pos_and_zero_vel = RewTerm(
        func=mdp.masked_target_proximity_reward_exp_pos_and_zero_vel,
        weight=100.0,  # базовый вес; при необходимости подстройте 30–80
        params=dict(
            mask_name="dof_mask",     # какие DOF гоним к таргету (bool mask)
            std_pos=0.25,             # “ширина” допусков по позиции (в норме [-1..1]); 0.25–0.5
            std_vel=0.20,             # “ширина” допусков по скорости; 0.2–0.3
            w_pos=1.0,                # вес позиции внутри экспоненты
            w_vel=1.5,                # вес нулевой скорости внутри экспоненты (0.5–1.5)
            gate=0.20,                # зона (в норме позы), где включается проверка на ~нулевую скорость
            init_attr="JOINT_INIT_POS_NORM",  # где хранится нормированная init-поза (если нужно)
        ),
    )

    # mask=0 → to init pose with zero-velocity bonus; legs optionally excluded while moving
    unmasked_init_proximity_exp_pos_and_zero_vel = RewTerm(
        func=mdp.unmasked_init_proximity_reward_exp_pos_and_zero_vel,
        weight=1.0,   # как в вашем примере; диапазон 5–20
        params=dict(
            mask_name="dof_mask",           # оцениваем инверт-маску (mask==0)
            exclude_legs_when_moving=True,  # исключать ноги, когда есть команда на движение
            std_pos=0.30,
            std_vel=0.25,
            w_pos=1.0,
            w_vel=1.2,
            gate=0.25,
            command_name="base_velocity",   # терм с командой [vx, vy, wz] или [vx, wz]
            lin_deadband=0.03,              # пороги, когда считаем «движется»
            ang_deadband=0.03,
            leg_bits=[0,1,3,4,7,8,11,12,15,16,19,20],  # какие индексы DOF считать «ногами»
            init_attr="JOINT_INIT_POS_NORM",
        ),
    )

    # === DWELL: бонус за удержание и штраф за выход из окна ===
    dwell_bonus = RewTerm(
        func=mdp.masked_dwell_bonus,
        weight=100.0,   # даёт до ~4*max_bonus; подстрой по кривым
        params=dict(
            mask_name="dof_mask",
            eps=0.04, vel_eps=0.03,
            hold_steps=4,
            bonus=0.5, growth=0.1, max_bonus=3.0,
        ),
    )

    # === DWELL (UNMASKED): удержание около INIT для DOF из инверт-маски (mask==0) ===
    dwell_unmasked = RewTerm(
        func=mdp.unmasked_dwell_bonus,
        weight=1.0,    # подстрой по месту; обычно 500..1500
        params=dict(
            mask_name="dof_mask",
            # окно допуска (в норме [-1,1] поз и скоростей):
            eps=0.04,
            vel_eps=0.03,
            # сколько подряд шагов «ОК» нужно для старта бонуса:
            hold_steps=4,
            # форма бонуса (как у masked-варианта):
            bonus=0.5, growth=0.1, max_bonus=3.0,
            # вокруг какой позы держимся (INIT) и как обращаться с «ногами» при движении:
            init_attr="JOINT_INIT_POS_NORM",
            exclude_legs_when_moving=True,
            command_name="base_velocity",
            lin_deadband=0.03,
            ang_deadband=0.03,
            leg_bits=(0,1,3,4,7,8,11,12,15,16,19,20),
        ),
    )

#    leaving_ok = RewTerm(
#        func=mdp.leaving_target_penalty,
#        weight=-2.0,   # ощутимо больно «сваливать» из окна
#        params=dict(
#            mask_name="dof_mask",
#            eps=0.05, vel_eps=0.04,
#            cooldown=1,   # штрафуем, если 1 шаг подряд вне окна после успеха
#        ),
#    )

#    # === СГЛАДИТЬ У ЦЕЛИ (но не мешать подходу) ===
#    near_target_smooth = RewTerm(
#        func=mdp.masked_action_rate_near_target,
#        weight=-0.6,   # сильнее глобального action_rate_l2,
#        params=dict(mask_name="dof_mask", gate=0.22),
#    )



@configclass
class MathTeleopRewardsCfg:
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    
#    root_too_low = DoneTerm(
#        func=mdp.root_height_below_minimum,
#        params={
#            "minimum_height": 0.40,                 # порог по высоте, м
#            "asset_cfg": SceneEntityCfg("robot"), 
#        },
#    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    
@configclass
class MathCurriculumCfg:
    miander_scale = CurrTerm(
        func=modify_env_param,
        params={"address": "MIANDER_SCALE", "modify_fn": mdp.modify_miander_scale},
    )
    max_period = CurrTerm(
        func=modify_env_param,
        params={"address": "MAX_PERIOD", "modify_fn": mdp.modify_max_period},
    )
    mask_prob = CurrTerm(
        func=modify_env_param,
        params={"address": "MASK_PROB_LEVEL", "modify_fn": mdp.modify_mask_prob},
    )
    
@configclass    
class MathAdaptiveCurriculumCfg:
    def __init__(self):
        
        self.miander_scale = CurrTerm(
            func=modify_env_param,
            params={
                "address": "MIANDER_SCALE",
                "modify_fn": mdp.modify_miander_scale_adaptive
            }
        )
        self.max_period = CurrTerm(
            func=modify_env_param,
            params={
                "address": "MAX_PERIOD",
                "modify_fn": mdp.modify_max_period_adaptive
            }
        )
        self.mask_prob = CurrTerm(
            func=modify_env_param,
            params={
                "address": "MASK_PROB_LEVEL",
                "modify_fn": mdp.modify_mask_prob_adaptive
            }
        )

class MathAdaptiveCurriculum:
    """
    Куррикулум с возможностью начать сразу с произвольной стадии.

    Поведение:
    - Если start_stage != 0 → куррикулум активируется на первой же update(),
      стадия = start_stage, далее увеличивается каждые stage_interval шагов.
    - Если start_stage == 0 → куррикулум не активен до тех пор, пока
      mean_tracking_reward не превысит reward_threshold; после этого стартует
      со стадии 0 и дальше растёт каждые stage_interval шагов.

    Интерфейс совместим: update(...) -> bool, get_* геттеры.
    """

    def __init__(
        self,
        initial_lr: float = 1e-3,
        lr_decay_factor: float = 10,
        delay_multiplier: float = 10,
        stage_interval: int = 200, #1000
        reward_threshold: float = 7.0,
        start_stage: int = 1, # 0	,              # ← желаемая стартовая стадия
    ):
        # (факультативные тренировочные параметры, оставлены как были)
        self.lr = float(initial_lr)
        self.lr_decay_factor = float(lr_decay_factor)
        self.delay_multiplier = float(delay_multiplier)

        # логика куррикулума
        self.reward_threshold = float(reward_threshold)
        self.stage_interval   = int(stage_interval)

        # состояние
        self.stage: int   = 0
        self.started: bool = False
        self.start_step: int = 0

        # желаемая стартовая стадия: применяется при первом update()
        self._pending_start_stage: int = int(start_stage)

    # --- геттеры (оставь как используешь в коде) ---
    def get_learning_rate(self) -> float:
        return self.lr

    def get_mask_prob(self) -> float:
        # пример зависимости от стадии; подстрой при желании
        return   0.15 # min(0.5, 0.05 + self.stage / 5000.0)

    def get_max_period(self) -> float:
        return 10.0

    def get_miander_scale(self) -> float:
        return 1.0 # min(1.0, 0.1 + self.stage / 5000.0)
        
    def get_exponent_multiplicator(self) -> float:
        # пример зависимости от стадии; подстрой при желании
        return 0.5 # max(0.001, 1.0 - self.stage / 5000.0)        

    # --- внутренняя утилита: выставить start_step так, чтобы сейчас была ровно target_stage ---
    def _align_start_to_now(self, current_step: int, target_stage: int) -> None:
        self.start_step = int(current_step) - int(target_stage) * int(self.stage_interval)
        self.stage      = int(target_stage)
        self.started    = True

    def update(self, env, mean_tracking_reward: float) -> bool:
        """
        Обновляет состояние куррикулума.
        Возвращает True, если стадия изменилась (или куррикулум только что активировался).
        """
        step = int(env.common_step_counter)
        changed = False

        # 0) перехват старта с нестандартной стадии
        if not self.started and self._pending_start_stage != 0:
            self._align_start_to_now(step, self._pending_start_stage)
            self._pending_start_stage = 0
            return True

        # 1) обычный старт по порогу (работает только если start_stage==0)
        if not self.started:
            if mean_tracking_reward > self.reward_threshold:
                self._align_start_to_now(step, 0)   # стартуем со стадии 0
                return True
            return False

        # 2) уже активны — ступенчатый рост раз в stage_interval
        new_stage = max(0, (step - self.start_step) // self.stage_interval)
        if new_stage > self.stage:
            self.stage = int(new_stage)
            print(f"Curriculum stage: {self.stage}")
            changed = True

        return changed

        

##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = ObstacklesSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ChaseObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = ChaseTrainCommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = ChaseTerminationsCfg()
    events: EventCfg = ChaseTrainEventCfg()
    curriculum: CurriculumCfg = ChaseCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 40.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

@configclass
class TestLocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = ObstacklesSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ChaseObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = ChaseTestCommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = ChaseTerminationsCfg()
    events: EventCfg = ChaseTestEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 40.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
