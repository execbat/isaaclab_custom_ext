from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
    
import math
import isaaclab.utils.math as math_utils


def feet_impact_vel(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float = 0.6,
    contact_force_threshold: float = 5.0,
    use_history: bool = True,
    store_key: str | None = None,
) -> torch.Tensor:
    """
    Penalty for the vertical foot impact velocity at the moment of initial contact.
    Returns a tensor [num_envs] ≤ 0.

    Expected:
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
    asset_cfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
    """
    device = env.device

    # contact sensor
    try:
        cs = env.scene.sensors[sensor_cfg.name]
    except KeyError:
        raise RuntimeError(f"[feet_impact_vel] sensor '{sensor_cfg.name}' not found in scene.sensors.")

    # robot articulation
    robot = env.scene.articulations.get(asset_cfg.name, None)
    if robot is None:
        
        robot = env.scene.get("robot", None)
    if robot is None:
        raise RuntimeError(f"[feet_impact_vel] articulation '{asset_cfg.name}' not found in scene.articulations.")

    # --- indices of the links covered by the sensor ---
    # The contact sensor usually stores them in .body_ids (or in .cfg.body_ids)
    feet_ids = None
    if hasattr(cs, "body_ids") and cs.body_ids is not None and len(cs.body_ids) > 0:
        feet_ids = cs.body_ids
    elif hasattr(cs, "cfg") and getattr(cs.cfg, "body_ids", None):
        feet_ids = cs.cfg.body_ids

    if not feet_ids:
        
        # (N, F, 3) — число F ног
        if hasattr(cs.data, "net_forces_w"):
            F = cs.data.net_forces_w.shape[1]            
            feet_ids = list(range(F))
        else:
            raise RuntimeError("[feet_impact_vel] sensor provides neither body_ids nor force data to output F.")

    feet_idx = torch.as_tensor(feet_ids, device=device, dtype=torch.long)

    # --- vertical leg speed (world) ---
    # robot.data.body_lin_vel_w: (N, B, 3)
    vz = robot.data.body_lin_vel_w[:, feet_idx, 2]  # (N, F)

    # --- contact (by magnitude of force) ---
    thr = float(contact_force_threshold)
    if use_history and hasattr(cs.data, "net_forces_w_history"):
        # (N, H, F, 3) ->we take the maximum in history, then the norm
        f_hist = cs.data.net_forces_w_history[:, :, feet_idx, :]    # (N, H, F, 3)
        fmag   = f_hist.norm(dim=-1).amax(dim=1)                    # (N, F)
    else:
        f_now = cs.data.net_forces_w[:, feet_idx, :]                # (N, F, 3)
        fmag  = f_now.norm(dim=-1)                                  # (N, F)

    contact_now = fmag > thr                                        # (N, F) bool

    # --- touchdown front ---
    key = store_key or f"_feet_prev_contact__{sensor_cfg.name}"
    if not hasattr(env, key):
        setattr(env, key, torch.zeros_like(contact_now, dtype=torch.bool))
    contact_prev = getattr(env, key)                                 # (N, F) bool
    touchdown = contact_now & (~contact_prev)                        # (N, F) bool

    # ---impact speed: downwards and only at the moment of contact ---
    neg_vz = torch.clamp(-vz, min=0.0)                               # (N, F) ≥ 0
    impact = torch.where(touchdown, neg_vz, torch.zeros_like(neg_vz))
    impact = torch.clamp(impact, max=float(clip))                    # cut outliers

    # updating the memory of the previous contact
    setattr(env, key, contact_now)

    # ---total penalty for stops ---
    penalty = impact.sum(dim=1)                                     # (N,)
    return penalty
    
def pelvis_height_target_reward(env: MathManagerBasedRLEnv,
                                target: float =  0.795, # 0.74,
                                alpha: float = 0.2) -> torch.Tensor:
    """
    Exponential reward: r = exp(-alpha * |z - target|)

    Args:
    env: MathManagerBasedRLEnv environment.
    target: desired pelvic height in meters.
    alpha: bell curve slope.

    Returns:
    Tensor[num_envs] — reward in the range (0‥1).
    """
    # We take the Z-coordinate of the pelvis
    asset = env.scene["robot"]
    pelvis_z = asset.data.root_pos_w[:, 2]           # shape [N]
    # print(pelvis_z)

    error = torch.abs(pelvis_z - target)      # |z − 0.7|
    reward = torch.exp(-alpha * error)     # e^(−α·err)

    return reward    

def no_command_motion_penalty(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),

    # When |cmd_lin| < lin_deadband → the penalty is almost max; the smaller the deadband, the "harsher"
    lin_deadband: float = 0.05,   # m/s
    ang_deadband: float = 0.05,   # rad/s

    # normalization scales (approximately for “typical” maximum speeds)
    lin_scale: float = 1.0,       # m/s → affects the magnitude of the linear penalty
    ang_scale: float = 1.0,       # rad/s → affects the magnitude of the angular penalty
) -> torch.Tensor:
    """
    Penalty for moving when there is NO move command.
    penalty = gate_lin * (||v_xy||/lin_scale)^2 + gate_ang * (|w_z|/ang_scale)^2,
    where gate_* ≈ 1 for a small team and → 0 as the team grows.
    """
    asset = env.scene[asset_cfg.name]

    # command [vx, vidle_double_support_bonusy, wz] in the database
    cmd = env.command_manager.get_term(command_name).command  # (N,3)
    cmd_lin_mag = cmd[:, :2].norm(dim=1)                      # (N,)
    cmd_ang_mag = cmd[:, 2].abs()                             # (N,)

    # base speed
    v_xy = asset.data.root_lin_vel_b[:, :2]                   # (N,2)
    w_z  = asset.data.root_ang_vel_b[:, 2]                    # (N,)

    # Smooth "curtains" (1 at zero command → 0 near the deadband and beyond)
    # The exponent produces a smooth and differentiable shape
    gate_lin = torch.exp(- (cmd_lin_mag / max(lin_deadband, 1e-6))**2)  # (N,)
    gate_ang = torch.exp(- (cmd_ang_mag / max(ang_deadband, 1e-6))**2)  # (N,)

    lin_term = (v_xy.norm(dim=1) / max(lin_scale, 1e-6))**2
    ang_term = (w_z.abs() / max(ang_scale, 1e-6))**2

    penalty = gate_lin * lin_term + gate_ang * ang_term
    return penalty    
    
def lateral_slip_penalty(env, command_name="base_velocity"):
    robot = env.scene["robot"]
    cmd   = env.command_manager.get_term(command_name).command  # (N,3): vx, vy, wz in base
    v_b   = robot.data.root_lin_vel_b[:, :2]                    # (N,2)
    #If the team is almost zero, we don't fine it. 
    mag = cmd[:,:2].norm(dim=1, keepdim=True) + 1e-6
    dir = cmd[:,:2] / mag
    # transverse component
    lat = v_b - (v_b*dir).sum(dim=1, keepdim=True)*dir
    return lat.norm(dim=1)    # positive result
    
def heading_alignment_reward(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_cmd_threshold: float = 0.05,   # m/s
    beta: float = 4.0,                 # «sharpness»
) -> torch.Tensor:
    """
    Reward ∈[0..1] for aligning the longitudinal axis of the body with the velocity command direction.
    Only works when |v_cmd| > lin_cmd_threshold.
    """
    robot = env.scene[asset_cfg.name]
    cmd   = env.command_manager.get_term(command_name).command  # (N,3)
    v_xy  = cmd[:, :2]
    v_mag = v_xy.norm(dim=1)
    gate  = v_mag > lin_cmd_threshold
    if not gate.any():
        return torch.zeros(env.num_envs, device=env.device)

    # a single vector of "where to go" in the world
    v_dir = v_xy / v_mag.clamp_min(1e-6).unsqueeze(-1)

    # longitudinal axis of the hull in the world
    fwd_w = math_utils.quat_apply(
        robot.data.root_quat_w,
        torch.tensor([1.0, 0.0, 0.0], device=env.device).expand_as(robot.data.root_pos_w),
    )[:, :2]
    fwd_dir = fwd_w / fwd_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    cosang = (fwd_dir * v_dir).sum(dim=-1).clamp(-1.0, 1.0)
    # 1 when aligned → drops when misaligned
    r = torch.exp(-beta * (1.0 - cosang))
    return r * gate.float()       


def leg_pelvis_torso_coalignment_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # body link names (as in USD):
    pelvis_body: str = "pelvis",
    torso_body: str = "torso_link",
    left_thigh_body: str = "left_hip_pitch_link",
    left_shank_body: str = "left_knee_link",
    right_thigh_body: str = "right_hip_pitch_link",
    right_shank_body: str = "right_knee_link",
    # Forward direction in the LSC links:
    forward_local: tuple[float, float, float] = (1.0, 0.0, 0.0),
    # Component weights:
    w_yaw: float = 1.0,     # co-orientation of segments with the pelvis along the course (XY)
    w_chain: float = 0.7,   # thigh↔calf coordination (each leg)
    w_upright: float = 0.3, # horizontality of the longitudinal axis of the pelvis/torso (less "on the toe/heel")
) -> torch.Tensor:
    """
    Returns r ∈ [0..1]. Encourages consistency in the direction of the leg, pelvis, and torso links:
    (A) YAW-aligned links with the pelvis (via XY),
    (B) consistent thigh↔shin "chain,"
    (C) "horizontal" longitudinal axis of the pelvis/torso.
    """
    device = env.device
    robot  = env.scene[asset_cfg.name]

    # cache of body indexes
    if not hasattr(env, "_coalignment_body_ids"):
        names = list(robot.data.body_names)
        def _idx(n: str) -> int:
            try:
                return names.index(n)
            except ValueError as e:
                raise RuntimeError(f"[coalignment] body '{n}' not found in robot.data.body_names") from e
        ids = [
            _idx(pelvis_body),
            _idx(torso_body),
            _idx(left_thigh_body), _idx(left_shank_body),
            _idx(right_thigh_body), _idx(right_shank_body),
        ]
        env._coalignment_body_ids = torch.as_tensor(ids, device=device, dtype=torch.long)

    ids = env._coalignment_body_ids  # [pelvis, torso, Lth, Lsh, Rth, Rsh]

    # "forward" in the world for every segment
    quats = robot.data.body_quat_w[:, ids, :]  # (N,6,4)
    f_loc = torch.tensor(forward_local, device=device, dtype=torch.float32).view(1,1,3)\
            .expand(quats.shape[0], quats.shape[1], 3)
    fwd_w = math_utils.quat_apply(quats, f_loc)  # (N,6,3)

    # XY projection and normalization
    fwd_xy = fwd_w[..., :2]
    fwd_xy = fwd_xy / fwd_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    pelvis_xy = fwd_xy[:, 0, :]  # (N,2)

    # (A) yaw alignment with the pelvis: torso, Lth, Lsh, Rth, Rsh
    cos_to_pelvis = (fwd_xy[:, 1:, :] * pelvis_xy.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)  # (N,5)
    yaw_align = 0.5 * (1.0 + cos_to_pelvis).mean(dim=1)  # (N,) в [0..1]

    #(B) consistency of the thigh↔calf chain (each leg)
    def _cos(i: int, j: int):
        return (fwd_xy[:, i, :] * fwd_xy[:, j, :]).sum(dim=-1).clamp(-1.0, 1.0)
    chain_align = 0.5 * (1.0 + 0.5 * (_cos(2, 3) + _cos(4, 5)))  # (N,) в [0..1]

    # (C) "horizontal" longitudinal axis of the pelvis and torso (less than |z|)
    z_pelvis = fwd_w[:, 0, 2].abs()
    z_torso  = fwd_w[:, 1, 2].abs()
    upright  = (1.0 - 0.5 * (z_pelvis + z_torso)).clamp(0.0, 1.0)  # (N,)

    # final weighted sum (without masks)
    denom = float(w_yaw + w_chain + w_upright)
    r = (w_yaw * yaw_align + w_chain * chain_align + w_upright * upright) / max(denom, 1e-6)
    return r.clamp(0.0, 1.0)
   
