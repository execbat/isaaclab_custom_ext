from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import TargetChaseVelocityCommandCfg


class TargetChaseVelocityCommand(CommandTerm):
    r"""Drives the robot towards a per-env target on the XY plane.

    Output command is in the robot **base frame** (b): ``[vx_b, vy_b, yaw_rate]``.

    - Linear part points to the target (optionally without strafe).
    - Yaw rate is a P-control on the heading error to the target (Uniform-like semantics).

    If :attr:`cfg.allow_strafe` is False, vy is forced to zero (non-holonomic style).
    """

    cfg: "TargetChaseVelocityCommandCfg"

    def __init__(self, cfg: "TargetChaseVelocityCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # checks like in UniformVelocityCommand
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "TargetChaseVelocityCommand: heading_command=True but `ranges.heading` is None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            # not critical, just a warning like in Uniform
            import omni.log
            omni.log.warn(
                f"TargetChaseVelocityCommand: 'ranges.heading'={self.cfg.ranges.heading} "
                f"but heading_command=False. Consider enabling heading command."
            )

        # assets
        self.robot: Articulation = env.scene[cfg.asset_name]
        try:
            self.target = env.scene[cfg.target_asset_name]
        except KeyError:
            raise KeyError(
                f"[TargetChaseVelocityCommand] target_asset_name '{cfg.target_asset_name}' not found in scene"
            )

        # buffers 
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)   # [vx_b, vy_b, yaw_rate]
        self.heading_target = torch.zeros(self.num_envs, device=self.device)     # world heading to target
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "TargetChaseVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    # ------------------------
    # Properties
    # ------------------------

    @property
    def command(self) -> torch.Tensor:
        """Desired base velocity command in base frame. Shape: (num_envs, 3)."""
        return self.vel_command_b

    # ------------------------
    # Impl specifics
    # ------------------------

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt if max_command_time > 0.0 else 1.0
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Generating a chase team:
           vx_b — by distance to target (with soft saturation),
           vy_b — 0,
           yaw_rate — P by angular heading error,
           + slowing down the linear speed until the course is turned,
           + optional "turn on the spot" in case of a large error.
        """
        # --- state (world) ---
        base_pos_w   = self.robot.data.root_pos_w          # (N,3)
        base_yaw_w   = self.robot.data.heading_w           # (N,)
        target_pos_w = self.target.data.root_pos_w         # (N,3)

        # --- vector to target (world, XY) ---
        delta_xy_w = (target_pos_w - base_pos_w)[:, :2]    # (N,2)
        dist_xy = torch.linalg.norm(delta_xy_w, dim=1)     # (N,)

        # unit vector to the target (world, XY)
        eps = 1e-6
        dir_tgt_w = torch.where(
            (dist_xy > eps).unsqueeze(-1),
            delta_xy_w / (dist_xy.unsqueeze(-1) + eps),
            torch.zeros_like(delta_xy_w),
        )  # (N,2)

        # --- unit vector X of the base in the world (where the robot is "looking") ---
        cx, sx = torch.cos(base_yaw_w), torch.sin(base_yaw_w)
        x_base_w = torch.stack([cx, sx], dim=1)            # (N,2)

        # --- signed angle between x_base_w and dir_tgt_w (in the world) ---
        # angle = atan2( cross_z(a,b), dot(a,b) )
        dot = (x_base_w * dir_tgt_w).sum(dim=1)            # (N,)
        cross_z = x_base_w[:, 0]*dir_tgt_w[:, 1] - x_base_w[:, 1]*dir_tgt_w[:, 0]
        heading_err = torch.atan2(cross_z, dot)            # ∈ (-pi, pi)

        # --- angular velocity (P by angular error)---
        k_yaw = getattr(self.cfg, "heading_control_stiffness", 1.5)
        yaw_min, yaw_max = self.cfg.ranges.ang_vel_z
        yaw_rate = torch.clamp(k_yaw * heading_err, yaw_min, yaw_max)

        # --- longitudinal speed: smooth over distance + "stop radius" ---
        stop_r   = getattr(self.cfg, "stop_radius", 0.3)
        k_lin    = getattr(self.cfg, "k_lin", 1.5)                 # «steepness» tanh
        vmax_lin = getattr(self.cfg, "max_speed", 1.2)

        dist_eff = torch.clamp(dist_xy - stop_r, min=0.0)          # (N,)
        vx_b = vmax_lin * torch.tanh(k_lin * dist_eff)             # (N,)
        vy_b = torch.zeros_like(vx_b)

        # --- Clamps by command ranges---
        vx_min, vx_max = self.cfg.ranges.lin_vel_x
        vy_min, vy_max = self.cfg.ranges.lin_vel_y
        vx_b = torch.clamp(vx_b, vx_min, vx_max)
        vy_b = torch.clamp(vy_b, vy_min, vy_max)                   # here 0 will remain in the acceptable range

        # --- angular error deceleration (one mechanism!) ---
        # with |heading_err| = 0 -> scale=1,
        # with |heading_err| >= theta_slow -> scale=min_factor,
        # between - linear interpolation.
        theta_slow = getattr(self.cfg, "heading_slowdown_angle", 1.0)  # rad (~57°)
        min_factor = getattr(self.cfg, "heading_slowdown_min", 0.25)   # min 25% from vx
        abs_err = torch.abs(heading_err)

        if theta_slow > 0.0:
            scale = min_factor + (1.0 - min_factor) * torch.clamp(1.0 - abs_err / (theta_slow + 1e-6), 0.0, 1.0)
            scale = torch.where(abs_err >= theta_slow, torch.full_like(scale, min_factor), scale)
            vx_b = vx_b * scale

        # ---option: "turn on the spot" in case of a very large heading error ---
        turn_in_place_angle = getattr(self.cfg, "turn_in_place_angle", None)  # for example 1.2 рад (~69°)
        if turn_in_place_angle is not None:
            big_err = abs_err > turn_in_place_angle
            if big_err.any():
                vx_b = torch.where(big_err, torch.zeros_like(vx_b), vx_b)

        # ---standing envy (if enabled) ---
        if hasattr(self, "is_standing_env"):
            standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
            if standing_ids.numel() > 0:
                vx_b[standing_ids] = 0.0
                vy_b[standing_ids] = 0.0
                yaw_rate[standing_ids] = 0.0

        # --- writing a command to the buffer (base frame) ---
        self.vel_command_b[:, 0] = vx_b
        self.vel_command_b[:, 1] = vy_b
        self.vel_command_b[:, 2] = yaw_rate

        # for debugging: "target" orientation in the world (not required, but useful)
        self.heading_target = torch.atan2(dir_tgt_w[:, 1], dir_tgt_w[:, 0])

    # ------------------------
    # Debug visualization
    # ------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
            
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_des_scale, vel_des_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_cur_scale, vel_cur_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_quat, vel_des_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_cur_quat, vel_cur_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert base-frame XY velocity to world-frame arrow (Uniform-style)."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle_b = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle_b)
        arrow_quat_b = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle_b)

        base_quat_w = self.robot.data.root_quat_w
        arrow_quat_w = math_utils.quat_mul(base_quat_w, arrow_quat_b)
        return arrow_scale, arrow_quat_w

