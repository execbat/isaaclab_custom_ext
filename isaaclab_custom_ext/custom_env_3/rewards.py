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
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

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

def idle_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),

    # --- linear component ---
    lin_speed_threshold: float = 0.03,  # m/s: actually "almost standing"
    lin_scale: float = 1.0,             # additional scale for linear penalty

    # --- corner component around z ---
    ang_speed_threshold: float = 0.03,  # rad/s: essentially "almost no rotation"
    ang_scale: float = 1.0,             # additional scale for corner penalty

    # --- deadbands for readability (duplicate the meanings of min_cmd_* above) ---
    lin_deadband: float = 0.03,         # m/s: "command ≈ 0"
    ang_deadband: float = 0.03,         # rad/s: "command ≈ 0"
) -> torch.Tensor:
    """
    Penalty (>=0) for "stuck in place": there is a noticeable command (linear and/or angular),
    but the base barely moves or rotates.

    Returns Tensor[num_envs] (non-negative). Set weight < 0 in the config.
    Rules:
    - if ||cmd_xy|| > lin_deadband and ||v_xy|| < lin_speed_threshold:
    lin_pen = lin_scale * (min_cmd_speed - ||v_xy||)_+
    - if |cmd_wz| > ang_deadband and |wz| < ang_speed_threshold:
    ang_pen = ang_scale * (min_cmd_ang_speed - |wz|)_+
    - result: penalty = lin_pen + ang_pen
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # actual base speeds (in LSC)
    v_xy = asset.data.root_lin_vel_b[:, :2]                  # [N,2]
    speed_xy = torch.linalg.norm(v_xy, dim=1)                # [N]
    wz = asset.data.root_ang_vel_b[:, 2].abs()               # [N] angular velocity along z

    # command (vx, vy, wz, ...)
    cmd = env.command_manager.get_command(command_name)       # [N, >=3]
    cmd_xy = cmd[:, :2]
    cmd_speed = torch.linalg.norm(cmd_xy, dim=1)              # [N]
    cmd_wz = cmd[:, 2].abs()                                  # [N]

    #masks "there is a command, but no fact"
    idle_lin = (cmd_speed > float(lin_deadband)) & (speed_xy < float(lin_speed_threshold))
    idle_ang = (cmd_wz    > float(ang_deadband)) & (wz       < float(ang_speed_threshold))

    # shortfall to the minimum expected speed when there is a command
    lin_deficit = (float(lin_deadband) - speed_xy).clamp(min=0.0)
    ang_deficit = (float(ang_deadband) - wz      ).clamp(min=0.0)

    # penalties
    penalty = torch.zeros_like(speed_xy)
    if idle_lin.any():
        penalty[idle_lin] += float(lin_scale) * lin_deficit[idle_lin]
    if idle_ang.any():
        penalty[idle_ang] += float(ang_scale) * ang_deficit[idle_ang]

    return penalty 
    
def track_lin_vel_xy_exp_custom(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), DEBUG = False
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    r =  1.5 *torch.exp(-lin_vel_error / std**2)  -0.5
    if DEBUG:
        print(f'Linear CMD {env.command_manager.get_command(command_name)[:, :2]}')
        print(f'Linear reward {r}')
    return r
    
def track_ang_vel_z_exp_custom(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), DEBUG = False
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    r = 1.5 * torch.exp(-ang_vel_error / std**2)  -0.5
    if DEBUG:
        print(f'Angular CMD {env.command_manager.get_command(command_name)[:, 2]}')
        print(f'Angular reward {r}')
    return r
    
def angvel_flat_l2_product(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
 
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    ang_r = r =  torch.exp(-ang_vel_error / std**2) 
    
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    lin_r =  torch.exp(-lin_vel_error / std**2) 
    

    return ang_r * lin_r
    
def alternating_airtime_reward(
    env,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg =  SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot",          body_names=".*_ankle_roll_link"),

    # --- command gating ---
    lin_deadband: float = 0.03,     # m/s: below -> linear command considered "near zero"
    ang_deadband: float = 0.03,     # rad/s: below -> angular command considered "near zero"

    # --- contacts ---
    contact_force_threshold: float = 5.0,  # N: contact if |F| > threshold
    use_history: bool = True,              # robust to noise (use max over history window)

    # --- target swing timing (per-step shaping while airborne) ---
    target_swing_time: float = 0.35,       # s: desired airborne duration per leg
    swing_sigma: float = 0.10,             # s: Gaussian width; smaller = stricter to target

    # --- swing time cap ---
    max_swing_time: float = 1.0,           # s: hard cap (excess is penalized every step)
    excess_penalty_scale: float = 1.0,     # penalty per 1s of excess per step

    # --- helpers ---
    same_lead_penalty: float = 0.4,        # penalty if the same leg "leads" twice in a row
    flight_penalty: float = 1.0,           # penalty when both feet are airborne while moving
    idle_double_support_bonus_val: float = 1.0,  # bonus for double support at rest
) -> torch.Tensor:
    """
    Reward encourages each leg's current airborne time to match a desired target *every step*.
    Also penalizes if the same leg touches down twice in a row (no alternation).

    Behavior:
      • REST (no significant command): reward double support.
      • MOVING:
          - While a leg is airborne, add exp(-(t_air - target)^2 / (2*sigma^2)) in [0,1].
          - Penalize 'flight' when both feet are airborne.
          - Penalize repeated leader (same leg touches down consecutively).
      • SWING CAP: per-step penalty for any airborne time beyond max_swing_time.
    """
    if sensor_cfg is None:
        raise ValueError("Provide sensor_cfg with the contact sensor and [LeftFoot, RightFoot] ids.")
    if asset_cfg is None:
        raise ValueError("Provide asset_cfg for the robot (only for consistency here).")

    device = env.device
    N = env.num_envs
    cs = env.scene.sensors[sensor_cfg.name]

    # timestep
    dt = env.sim.cfg.dt * env.cfg.decimation

    # --- command gating: rest vs moving ---
    cmd = env.command_manager.get_term(command_name).command  # (N,3): [vx, vy, wz]
    cmd = torch.as_tensor(cmd, device=device, dtype=torch.float32)
    lin_mag = cmd[:, :2].norm(dim=1)
    ang_mag = cmd[:, 2].abs() if cmd.shape[1] >= 3 else torch.zeros_like(lin_mag)

    near_zero = (lin_mag < lin_deadband) & (ang_mag < ang_deadband)
    moving    = ~near_zero

    # --- contacts ---
    if use_history:
        f_hist = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]   # (N,H,2,3)
        fmag   = f_hist.norm(dim=-1).amax(dim=1)                               # (N,2)
    else:
        f_now  = cs.data.net_forces_w[:, sensor_cfg.body_ids, :]              # (N,2,3)
        fmag   = f_now.norm(dim=-1)                                           # (N,2)

    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold
    both_down = Lc & Rc
    any_down  = Lc | Rc
    flight    = ~any_down

    # --- persistent state ---
    need_init = not all(hasattr(env, a) for a in [
        "_air_prev_Lc", "_air_prev_Rc",
        "_air_in_L", "_air_in_R",
        "_air_t_L", "_air_t_R",
        "_air_last_lead"  # 0=L, 1=R, 2=none
    ])
    if need_init:
        env._air_prev_Lc = torch.zeros(N, dtype=torch.bool,    device=device)
        env._air_prev_Rc = torch.zeros(N, dtype=torch.bool,    device=device)
        env._air_in_L    = torch.zeros(N, dtype=torch.bool,    device=device)
        env._air_in_R    = torch.zeros(N, dtype=torch.bool,    device=device)
        env._air_t_L     = torch.zeros(N, dtype=torch.float32, device=device)
        env._air_t_R     = torch.zeros(N, dtype=torch.float32, device=device)
        env._air_last_lead = torch.full((N,), 2, dtype=torch.long, device=device)  # 2 = no leader yet

    # reset on episode end
    resets = (env.termination_manager.terminated | env.termination_manager.time_outs)
    if resets.any():
        env._air_prev_Lc[resets] = Lc[resets]
        env._air_prev_Rc[resets] = Rc[resets]
        env._air_in_L[resets]    = ~Lc[resets]
        env._air_in_R[resets]    = ~Rc[resets]
        env._air_t_L[resets]     = 0.0
        env._air_t_R[resets]     = 0.0
        env._air_last_lead[resets] = 2

    # events
    liftoff_L   = (~Lc) & env._air_prev_Lc
    liftoff_R   = (~Rc) & env._air_prev_Rc
    touchdown_L = Lc & (~env._air_prev_Lc)
    touchdown_R = Rc & (~env._air_prev_Rc)

    # update airborne flags
    env._air_in_L = ~Lc
    env._air_in_R = ~Rc

    # advance timers while airborne
    env._air_t_L = torch.where(env._air_in_L, env._air_t_L + dt, env._air_t_L)
    env._air_t_R = torch.where(env._air_in_R, env._air_t_R + dt, env._air_t_R)

    # reset timer at liftoff (start of a new swing)
    env._air_t_L = torch.where(liftoff_L, torch.zeros_like(env._air_t_L), env._air_t_L)
    env._air_t_R = torch.where(liftoff_R, torch.zeros_like(env._air_t_R), env._air_t_R)

    # --- reward ---
    reward = torch.zeros(N, dtype=torch.float32, device=device)

    # REST: prefer double support
    reward = reward + idle_double_support_bonus_val * (near_zero & both_down).float()

    # MOVING: per-step shaping towards target swing time; flight penalty; alternation check
    if moving.any():
        # penalize flight (both legs airborne)
        reward = reward - flight_penalty * (moving & flight).float()

        # per-step Gaussian score for airborne legs
        eps = 1e-12
        inv2sig2 = 1.0 / (2.0 * (swing_sigma ** 2) + eps)

        score_L = torch.zeros(N, device=device)
        score_R = torch.zeros(N, device=device)

        if env._air_in_L.any():
            dL = env._air_t_L[env._air_in_L] - float(target_swing_time)
            score_L[env._air_in_L] = torch.exp(-(dL * dL) * inv2sig2)

        if env._air_in_R.any():
            dR = env._air_t_R[env._air_in_R] - float(target_swing_time)
            score_R[env._air_in_R] = torch.exp(-(dR * dR) * inv2sig2)

        # average over airborne legs (0 if none airborne)
        num_air = env._air_in_L.float() + env._air_in_R.float()
        leg_score = torch.zeros(N, device=device)
        has_air = num_air > 0
        leg_score[has_air] = (score_L[has_air] * env._air_in_L[has_air].float() +
                              score_R[has_air] * env._air_in_R[has_air].float()) / num_air[has_air]
        reward = reward + moving.float() * leg_score

        # ----- alternation helper: penalize same leader twice in a row -----
        # Determine new leader at touchdown (2 means "no touchdown this step")
        new_lead = torch.where(
            touchdown_R, torch.ones(N, device=device, dtype=torch.long),
            torch.where(touchdown_L, torch.zeros(N, device=device, dtype=torch.long),
                        torch.full((N,), 2, device=device, dtype=torch.long))
        )
        same_lead = (new_lead != 2) & (env._air_last_lead != 2) & (new_lead == env._air_last_lead)
        reward = reward - same_lead_penalty * same_lead.float()

        # Update last leader where a touchdown occurred
        env._air_last_lead = torch.where(
            touchdown_L, torch.zeros_like(env._air_last_lead),
            torch.where(touchdown_R, torch.ones_like(env._air_last_lead), env._air_last_lead)
        )

    # SWING CAP: per-step penalty for any excess beyond max_swing_time
    excess_L = torch.clamp(env._air_t_L - float(max_swing_time), min=0.0)
    excess_R = torch.clamp(env._air_t_R - float(max_swing_time), min=0.0)
    reward = reward - float(excess_penalty_scale) * (excess_L + excess_R)

    # update prev contacts
    env._air_prev_Lc = Lc
    env._air_prev_Rc = Rc

    return reward
    
def step_phase_reward(
    env,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg:  SceneEntityCfg = SceneEntityCfg("robot",          body_names=".*_ankle_roll_link"),

    # --- commands and gates ---
    lin_deadband: float = 0.03,      # m/s: "almost stopped" (linear)
    ang_deadband: float = 0.03,      # rad/s: "almost stopped" (angular)
    use_history: bool = True,        # take max over history for robustness

    # --- contact force ---
    contact_force_threshold: float = 0.0,  # H: optional threshold (0 - no threshold)
    amp_ref: float = 400.0,                # H: desired max force for normalization and reference A

    # --- phase generator ---
    freq_gain_hz_per_mps: float = 2.0,     # f = k_f * |v|; at |v|=0.5 => 1 Hz; at |v|=1.0 => 2 Hz
    clamp_freq: tuple = (0.0, 4.0),        

    # --- exponent from MAE (gaussian kernel) ---
    std_vel: float = 0.25,                 # controls sharpness of exp decay
):
    """
    Returns the [num_envs] tensor with a reward.
    Expected order: sensor_cfg.body_ids: [LeftFoot, RightFoot].
    Uses env.command_manager[command_name].command (N,3): [vx, vy, wz].
    Reward is gated on (|v_cmd_lin| >= lin_deadband) OR (|wz_cmd| >= ang_deadband).
    """
    device = env.device
    N = env.num_envs
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # dt симуляции
    dt = env.sim.cfg.dt * env.cfg.decimation

    # --- command and movement gate (linear OR angular) ---
    cmd = env.command_manager.get_term(command_name).command  # (N,3) -> [vx, vy, wz]
    cmd = torch.as_tensor(cmd, device=device, dtype=torch.float32)
    lin_mag = cmd[:, :2].norm(dim=1)                          # (N,)
    ang_mag = cmd[:, 2].abs()                                 # (N,)
    moving = (lin_mag >= lin_deadband) | (ang_mag >= ang_deadband)

    # --- contact forces between two feet (norms) ---
    if use_history:
        f_hist = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # (N,H,2,3)
        fmag = f_hist.norm(dim=-1).amax(dim=1)                                # (N,2)        
    else:
        f_now = cs.data.net_forces_w[:, sensor_cfg.body_ids, :]               # (N,2,3)        
        fmag = f_now.norm(dim=-1)                                             # (N,2)

    if contact_force_threshold > 0.0:
        fmag = torch.where(fmag > contact_force_threshold, fmag, torch.zeros_like(fmag))

    # --- per-env state: accumulate oscillator phase ---
    if not hasattr(env, "_g2_phase"):
        env._g2_phase = torch.zeros(N, dtype=torch.float32, device=device)

    # frequency by *linear* speed 
    f_hz = freq_gain_hz_per_mps * lin_mag
    if clamp_freq is not None:
        f_hz = torch.clamp(f_hz, clamp_freq[0], clamp_freq[1])

    # phase increment: dφ = 2π f dt
    dphi = (2.0 * torch.pi * f_hz * dt).to(device)
    env._g2_phase = (env._g2_phase + dphi) % (2.0 * torch.pi)

    # --- Reference signals for legs (antiphase) ---
    phi = env._g2_phase
    s_ref_R = amp_ref * torch.relu(torch.sin(phi))
    s_ref_L = amp_ref * torch.relu(torch.sin(phi + torch.pi))

    # --- normalization of forces to [0,1] by amp_ref and clipping ---
    eps = 1e-6
    act_R = torch.clamp(fmag[:, 1] / (amp_ref + eps), 0.0, 1.0)  # assume order [L, R]
    act_L = torch.clamp(fmag[:, 0] / (amp_ref + eps), 0.0, 1.0)
    ref_R = torch.clamp(s_ref_R / (amp_ref + eps), 0.0, 1.0)
    ref_L = torch.clamp(s_ref_L / (amp_ref + eps), 0.0, 1.0)

    # --- MAE per leg ---
    mae_R = torch.abs(act_R - ref_R)
    mae_L = torch.abs(act_L - ref_L)

    # --- Gaussian kernel on error (faster learning than exp(-|e|/σ²)) ---
    inv2sig2 = 1.0 / (2.0 * (std_vel**2) + eps)
    r_R = torch.exp(-(mae_R**2) * inv2sig2)
    r_L = torch.exp(-(mae_L**2) * inv2sig2)

    # --- average legs (more stable than product) ---
    reward = 0.5 * (r_R + r_L)

    # --- gate: reward only when there is linear OR angular command ---
    reward = reward * moving.float()
    return reward
    
    
def com_projection_reward(
    env,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg:  SceneEntityCfg = SceneEntityCfg("robot",          body_names=".*_ankle_roll_link"),

    # --- gates on command ---
    lin_deadband: float = 0.03,          # m/s: "almost there" → target CoM without bias

    # --- contacts ---
    contact_force_threshold: float = 5.0,# H: contact if |F| > threshold
    use_history: bool = True,            # more noise-resistant (we take the maximum from history)

    # --- desired displacement of CoM in the direction of movement ---
    com_offset_gain: float = 0.15,       # m per (m/s): at |v|=1 we shift by 0.15 m
    max_offset: float = 0.25,            # m: maximum displacement limit
    beta: float = 10.0,                  # r = exp(-beta * mse)

    # --- behavior without support ---
    no_support_penalty: float = 0.0,     # can be >0 to penalize "jump" (two legs in the air)
):
    """
    Returns a tensor [N] with a reward.
    Expectations:
    • sensor_cfg.body_ids = [LeftFoot, RightFoot].
    • The velocity command is available in env.command_manager[command_name].command (N,3): [vx, vy, wz].
    • CoM is taken from robot.data.com_pos_w, if available; otherwise, the proxy is root_pos_w.
    • Stop positions are taken from robot.data.body_state_w[:, body_ids, :3] (world coordinates).
    """
    device = env.device
    N = env.num_envs
    robot = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    
    # dt = env.sim.cfg.dt * env.cfg.decimation

    # --- cmd ---
    cmd = env.command_manager.get_term(command_name).command  # (N,3)
    cmd = torch.as_tensor(cmd, device=device, dtype=torch.float32)
    vxy = cmd[:, :2]                         # (N,2)
    speed = vxy.norm(dim=1)                  # (N,)
    moving = speed >= lin_deadband
    dir_xy = torch.where(
        (speed > 1e-6).unsqueeze(1),
        vxy / (speed.unsqueeze(1) + 1e-12),
        torch.zeros_like(vxy)
    )                                         # (N,2)

    # --- contact forces ---
    if use_history:
        f_hist = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # (N,H,2,3)
        fmag   = f_hist.norm(dim=-1).amax(dim=1)                              # (N,2)
    else:
        f_now  = cs.data.net_forces_w[:, sensor_cfg.body_ids, :]             # (N,2,3)
        fmag   = f_now.norm(dim=-1)                                          # (N,2)

    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold
    any_down  = Lc | Rc
    both_down = Lc & Rc

    # --- foot positions (world), we take the centers of the rigid bodies of the feet ---
    body_pos_w = robot.data.body_state_w[:, sensor_cfg.body_ids, :3]  # (N,2,3)
    L_xy = body_pos_w[:, 0, :2]                                       # (N,2)
    R_xy = body_pos_w[:, 1, :2]                                       # (N,2)

    # --- maintain the "last support" if both legs are in the air ---
    need_init = not hasattr(env, "_g3_support_xy")
    if need_init:
        # initialization: take the average between the stops
        env._g3_support_xy = 0.5 * (L_xy + R_xy)

    # calculate the current reference point
    # 1) both on the reference point → midpoint between those in contact (usually both)
    # 2) one on the reference point → its position
    # 3) none → take the previous one (memory)
    support_xy = env._g3_support_xy.clone()

    # both in contact
    both_mask = both_down
    if both_mask.any():
        support_xy[both_mask] = 0.5 * (L_xy[both_mask] + R_xy[both_mask])
    # only left
    onlyL = Lc & (~Rc)
    if onlyL.any():
        support_xy[onlyL] = L_xy[onlyL]
    # only right
    onlyR = Rc & (~Lc)
    if onlyR.any():
        support_xy[onlyR] = R_xy[onlyR]
    # none - we leave the same

    # we'll update the memory only when there's at least someone on the support
    has_support = any_down
    env._g3_support_xy = torch.where(
        has_support.unsqueeze(1),
        support_xy,
        env._g3_support_xy
    )

    # --- desired point CoM ---
    offset_mag = torch.clamp(com_offset_gain * speed, 0.0, max_offset)  # (N,)
    # at almost zero speed offset→0 automatically
    target_xy = env._g3_support_xy + dir_xy * offset_mag.unsqueeze(1)   # (N,2)

    # --- actual projection CoM (x,y) ---
    if hasattr(robot.data, "com_pos_w"):
        com_xy = robot.data.com_pos_w[:, :2]     # (N,2)
    else:
        # fallback: use root position as CoM proxy
        com_xy = robot.data.root_pos_w[:, :2]    # (N,2)

    # --- MSE and Reward ---
    diff = com_xy - target_xy                    # (N,2)
    mse  = (diff ** 2).sum(dim=1)                # (N,) — square error in XY
    #print(f"MSE {mse}")
    reward = torch.exp(-beta * mse)              # (N,)

    # if there is no support, an additional fine may be imposed (optional)
    if no_support_penalty > 0.0:
        reward = reward - no_support_penalty * (~has_support).float()
    #print(f" COM reward: {reward}")
    return reward

def step_width_penalty(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg:  SceneEntityCfg = SceneEntityCfg("robot",          body_names=".*_ankle_roll_link"),

    # — nominal value and sensitivity —
    nominal_width: float = 0.20,      # m: desired distance between feet (or maximum without penalty)
    #beta: float = 20.0,               # r = exp(-beta * excess^2)

    # — contacts and gates —
    contact_force_threshold: float = 5.0,
    use_history: bool = True,
    gate_by_support: bool = True,     # If True, we don't penalize when both legs are in the air.
):
    """
    Returns an [N] tensor with a reward for the "step width."
    • sensor_cfg.body_ids are expected in the order [LeftFoot, RightFoot].
    • Foot positions are taken from robot.data.body_state_w (world), and the XY distance is calculated.
    """
    device = env.device
    N = env.num_envs
    robot = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Stop positions (world), we take XY projections
    body_pos_w = robot.data.body_state_w[:, sensor_cfg.body_ids, :3]  # (N,2,3)
    L_xy = body_pos_w[:, 0, :2]                                       # (N,2)
    R_xy = body_pos_w[:, 1, :2]                                       # (N,2)

    # Distance between feet on the ground
    dist_xy = (L_xy - R_xy).norm(dim=1)                               # (N,)

    # Excess over par
    excess = torch.clamp(dist_xy - nominal_width, min=0.0)            # (N,)
    penalty = excess

                                      # (N,)

    # Optional: No penalty when both legs are in the air (no support)
    if gate_by_support:
        if use_history:
            f_hist = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # (N,H,2,3)
            fmag   = f_hist.norm(dim=-1).amax(dim=1)                              # (N,2)
        else:
            f_now  = cs.data.net_forces_w[:, sensor_cfg.body_ids, :]             # (N,2,3)
            fmag   = f_now.norm(dim=-1)                                          # (N,2)

        Lc = fmag[:, 0] > contact_force_threshold
        Rc = fmag[:, 1] > contact_force_threshold
        any_down = Lc | Rc

        # where there is no support, we set the penalty to 0 (neutral) to avoid penalties in flight
        penalty = torch.where(any_down, penalty, torch.zeros_like(penalty))
    
    #print(f"penalty: {penalty}")
    return penalty   
    
    


def quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Isaac Lab: root_quat_w is (w,x,y,z) in world frame.
    Convert quaternion (w,x,y,z) to rotation matrix R_world_from_pelvis. Shape: [N,3,3].
    """
    w, x, y, z = q.unbind(dim=-1)
    n = torch.clamp((w*w + x*x + y*y + z*z).sqrt(), min=1e-9)
    w, x, y, z = w/n, x/n, y/n, z/n

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = torch.stack([
        1 - 2*(yy + zz),  2*(xy - wz),      2*(xz + wy),
        2*(xy + wz),      1 - 2*(xx + zz),  2*(yz - wx),
        2*(xz - wy),      2*(yz + wx),      1 - 2*(xx + yy),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def foot_symmetry_step_reward_cmddir(
    env,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg:  SceneEntityCfg = SceneEntityCfg("robot",          body_names=".*_ankle_roll_link"),

    # --- command gating ---
    lin_deadband: float = 0.03,     # m/s: near-zero linear command in pelvis XY
    ang_deadband: float = 0.03,     # rad/s: near-zero yaw command

    # --- contacts ---
    contact_force_threshold: float = 5.0,  # N: contact if |F| > threshold
    use_history: bool = True,              # robust to noise (max over history)

    # --- touchdown kernel (direction-aware) ---
    sym_lambda: float = 0.06,       # m: y = exp(-|x_td + dir*x_lo| / sym_lambda)
    sign_margin: float = 0.0,       # m: tolerance around 0 for sign checks

    # --- penalty scale when sign is wrong ---
    wrong_sign_penalty_scale: float = 0.10,  # multiplies core and flips sign: reward -> -scale*reward

    # --- standing preference (nonnegative) ---
    stand_sigma: float = 0.08,      # m
    stand_bonus: float = 1.0,       # scale
) -> torch.Tensor:
    """
    Direction-aware step reward around pelvis X with sign-sensitive scoring.

    REST (near-zero linear & angular command):
      + stand_bonus * exp(-(xL^2 + xR^2)/(2*stand_sigma^2)) when both feet in contact.

    MOVING:
      At each touchdown:
        - dir = +1 (forward) if pelvis-frame vx_cmd >= 0 else -1 (backward).
        - Check signs:
            forward:  x_td > +margin  and  x_lo < -margin
            backward: x_td < -margin  and  x_lo > +margin
        - Compute core = exp(-|x_td + dir*x_lo| / sym_lambda).
        - If signs OK  -> add +core   (0..1].
          If signs BAD -> add -wrong_sign_penalty_scale * core  (<= 0, scaled).
    """
    if sensor_cfg is None or asset_cfg is None:
        raise ValueError("Provide sensor_cfg ([L,R] contact sensor) and asset_cfg ('robot').")

    device = env.device
    N = env.num_envs
    robot = env.scene[asset_cfg.name]
    cs    = env.scene.sensors[sensor_cfg.name]

    # --- pelvis pose & transforms ---
    pelvis_pos_w  = robot.data.root_pos_w          # (N,3)
    pelvis_quat_w = robot.data.root_quat_w         # (N,4) (w,x,y,z)
    R_wp = quat_wxyz_to_rotmat(pelvis_quat_w)      # (N,3,3): world_from_pelvis
    R_pw = R_wp.transpose(1, 2)                    # (N,3,3): pelvis_from_world

    # --- command → pelvis frame ---
    cmd = env.command_manager.get_term(command_name).command   # (N,3): [vx_w, vy_w, wz]
    cmd = torch.as_tensor(cmd, device=device, dtype=torch.float32)
    v_xy_world = cmd[:, :2]
    wz_cmd     = cmd[:, 2]

    v_xyz_world  = torch.cat([v_xy_world, torch.zeros_like(v_xy_world[:, :1])], dim=1)   # (N,3)
    v_xyz_pelvis = (R_pw @ v_xyz_world.unsqueeze(-1)).squeeze(-1)                        # (N,3)
    v_xy_pelvis  = v_xyz_pelvis[:, :2]
    vx_cmd_pelvis = v_xy_pelvis[:, 0]

    lin_mag = torch.linalg.norm(v_xy_pelvis, dim=1)
    ang_mag = wz_cmd.abs()
    near_zero = (lin_mag < lin_deadband) & (ang_mag < ang_deadband)
    moving    = ~near_zero

    # direction along pelvis X: +1 forward, -1 backward
    dir_sign = torch.where(vx_cmd_pelvis >= 0.0,
                           torch.ones_like(vx_cmd_pelvis),
                           -torch.ones_like(vx_cmd_pelvis))

    # --- contacts ---
    if use_history:
        f_hist = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # (N,H,2,3)
        fmag   = f_hist.norm(dim=-1).amax(dim=1)                              # (N,2)
    else:
        f_now  = cs.data.net_forces_w[:, sensor_cfg.body_ids, :]             # (N,2,3)
        fmag   = f_now.norm(dim=-1)                                          # (N,2)

    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold
    both_down = Lc & Rc

    # --- foot X positions in pelvis frame ---
    body_pos_w = robot.data.body_state_w[:, sensor_cfg.body_ids, :3]  # (N,2,3)
    dL_w = body_pos_w[:, 0, :] - pelvis_pos_w
    dR_w = body_pos_w[:, 1, :] - pelvis_pos_w
    dL_p = (R_pw @ dL_w.unsqueeze(-1)).squeeze(-1)
    dR_p = (R_pw @ dR_w.unsqueeze(-1)).squeeze(-1)
    xL   = dL_p[:, 0]
    xR   = dR_p[:, 0]

    # --- persistent state: prev contacts + last liftoff X ---
    if not hasattr(env, "_fs_prev_Lc"):
        env._fs_prev_Lc = torch.zeros(N, dtype=torch.bool,    device=device)
        env._fs_prev_Rc = torch.zeros(N, dtype=torch.bool,    device=device)
        env._fs_last_liftoff_x_L = torch.zeros(N, dtype=torch.float32, device=device)
        env._fs_last_liftoff_x_R = torch.zeros(N, dtype=torch.float32, device=device)

    liftoff_L   = (~Lc) & env._fs_prev_Lc
    liftoff_R   = (~Rc) & env._fs_prev_Rc
    touchdown_L = Lc & (~env._fs_prev_Lc)
    touchdown_R = Rc & (~env._fs_prev_Rc)

    # cache liftoff X
    env._fs_last_liftoff_x_L = torch.where(liftoff_L, xL, env._fs_last_liftoff_x_L)
    env._fs_last_liftoff_x_R = torch.where(liftoff_R, xR, env._fs_last_liftoff_x_R)

    # --- reward ---
    reward = torch.zeros(N, dtype=torch.float32, device=device)

    # REST: encourage both feet near pelvis X=0
    if near_zero.any():
        inv2_s = 1.0 / (2.0 * (stand_sigma**2) + 1e-12)
        stand_score = torch.exp(-(xL**2 + xR**2) * inv2_s)
        reward += stand_bonus * (near_zero & both_down).float() * stand_score

    # MOVING: score only on touchdown events
    if moving.any():

        def add_td(td_mask, x_td_all, x_lo_last_all, dir_all):
            if not td_mask.any():
                return
            x_td = x_td_all[td_mask]
            x_lo = x_lo_last_all[td_mask]
            dsgn = dir_all[td_mask]

            # expected signs
            td_ok = torch.where(dsgn > 0, x_td >  sign_margin, x_td < -sign_margin)
            lo_ok = torch.where(dsgn > 0, x_lo < -sign_margin, x_lo >  sign_margin)
            ok = td_ok & lo_ok

            core = torch.exp(-(x_td + dsgn * x_lo).abs() / (sym_lambda + 1e-12))  # (0,1]

            out = torch.empty_like(core)
            # correct sign → +core
            out[ ok]  =  core[ ok]
            # wrong sign → -scale * core
            out[~ok]  = -float(wrong_sign_penalty_scale) * core[~ok]

            reward[td_mask] += out

        add_td(touchdown_L, xL, env._fs_last_liftoff_x_L, dir_sign)
        add_td(touchdown_R, xR, env._fs_last_liftoff_x_R, dir_sign)

    # update prev contacts
    env._fs_prev_Lc = Lc
    env._fs_prev_Rc = Rc

    return reward
    
def target_distance_exp_reward(
    env: ManagerBasedRLEnv,
    alpha: float = 1.0,      # steepness of the curve: higher alpha → faster drops to 0
    use_xy: bool = True,     # calculate the distance only along XY (this is usually what you need)
    max_dist: float | None = None,  # can be limited from above so that distant ones are still ≈0
) -> torch.Tensor:
    """
    r = exp(-alpha * dist)

    dist = distance between robot and target.
    dist = 0 → r = 1
    dist → large → r → 0
    """
    robot: Articulation | RigidObject = env.scene["robot"]
    target: RigidObject = env.scene["target"]

    if use_xy:
        robot_pos = robot.data.root_pos_w[:, :2]    # (N,2)
        target_pos = target.data.root_pos_w[:, :2]  # (N,2)
    else:
        robot_pos = robot.data.root_pos_w[:, :3]    # (N,3)
        target_pos = target.data.root_pos_w[:, :3]  # (N,3)

    dist = (robot_pos - target_pos).norm(dim=-1)    # (N,)

    if max_dist is not None:
        dist = torch.clamp(dist, 0.0, float(max_dist))

    reward = torch.exp(-alpha * dist)               # (N,) в (0..1]

    return reward    
