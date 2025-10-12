import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass
import math
from .commands import TargetChaseVelocityCommand
from isaaclab.managers import CommandTermCfg
from isaaclab.markers.config import (
    VisualizationMarkersCfg,
    GREEN_ARROW_X_MARKER_CFG,
    BLUE_ARROW_X_MARKER_CFG,
)


@configclass
class TargetChaseVelocityCommandCfg(CommandTermCfg):
    """Configuration for target-chasing velocity commands (base-frame)."""

    class_type: type = TargetChaseVelocityCommand

    # who to command / who to chase
    asset_name: str = "robot"
    target_asset_name: str = "target"

    # heading P-behavior
    heading_command: bool = True
    heading_control_stiffness: float = 0.5

    # motion shaping
    max_speed: float = 1.0
    k_lin: float = 1.0
    stop_radius: float = 0.15
    allow_strafe: bool = False

    # standing probability (like Uniform)
    rel_standing_envs: float = 0.0

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)
        heading:   tuple[float, float] | None = (-math.pi, math.pi)

    ranges: Ranges = Ranges()

    # visualization (same defaults & scales as Uniform)
    debug_vis: bool = True
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # when to trigger _resample_command (we only update standing flags)
    resampling_time_range: tuple[float, float] = (1.0, 1.0) 
 
 
 
 
 
 
 
 
 
 
 
 
 
    
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    
    base_velocity = TargetChaseVelocityCommandCfg(
        asset_name="robot",
        target_asset_name="target",
        heading_command=True,
        heading_control_stiffness=0.5,
        max_speed=1.0,
        k_lin=1.0,
        stop_radius=0.15,
        allow_strafe=False,
        ranges=TargetChaseVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
        debug_vis=True,
        resampling_time_range=(0.0, 0.0),  
    )
       
