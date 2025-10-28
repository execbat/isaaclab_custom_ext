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
    min_cmd_speed: float = 0.10,        #m/s - we believe that this is "there is a command to go"
    lin_speed_threshold: float = 0.05,  # m/s - we actually stand, if lower
    scale: float = 1.0,                 # additional scale
) -> torch.Tensor:
    """
    Penalty (>=0) for "stuck in place" when the movement command is noticeable,
    and the base's linear velocity in the XY plane is low.

    Returns: Tensor[num_envs] (non-negative). Set weight < 0 in the config.

    Rule:
    if ||cmd_xy|| > min_cmd_speed and ||v_xy|| < lin_speed_threshold,
    penalty = scale * (min_cmd_speed - ||v_xy||)_+ , otherwise 0.
    """
    #actual base speed in LSC (XY)
    asset: RigidObject = env.scene[asset_cfg.name]
    v_xy = asset.data.root_lin_vel_b[:, :2]                     # [N,2]
    speed_xy = torch.linalg.norm(v_xy, dim=1)                   # [N]

    # speed command (vx, vy, wz, ...)
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]  # [N,2]
    cmd_speed = torch.linalg.norm(cmd_xy, dim=1)                   # [N]

    # «есть команда ехать», но «почти стоим»
    idle_mask = (cmd_speed > float(min_cmd_speed)) & (speed_xy < float(lin_speed_threshold))

    # linear penalty for underspeeding
    deficit = (float(min_cmd_speed) - speed_xy).clamp(min=0.0)

    penalty = torch.zeros_like(speed_xy)
    penalty[idle_mask] = float(scale) * deficit[idle_mask]
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
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg:  SceneEntityCfg = SceneEntityCfg("robot",          body_names=".*_ankle_roll_link"),    

    # --- commands and gates ---
    lin_deadband: float = 0.03,                 # m/s: "command ≈ 0"
    ang_deadband: float = 0.03,                 # rad/s

    # --- contacts ---
    contact_force_threshold: float = 5.0,       # H: contact is considered if |F| > thr
    use_history: bool = True,                   # we take the max from the sensor history (more resistant to noise)

    # --- target by leg flight time ---
    target_swing_time: float = 0.35,            # sec – target time for “leg in the air”
    swing_sigma: float = 0.10,                  # sec — bell width for exp(−(t−T)^2/σ^2)

    # --- weights and fines ---
    idle_double_support_bonus_val: float = 1.0, # bonus at rest for two-legged
    touchdown_bonus: float = 1.0,               # bonus at the moment of touch, if swing≈target
    shaping_weight: float = 0.3,                # soft bonus during flight (every step)
    same_lead_penalty: float = 0.5,             # penalty if the same leg "leads" in a row
    flight_penalty: float = 1.0,                # penalty if both legs are in the air while moving
):
    """
    Returns a tensor [num_envs] with the reward.

    Expectations:
    • sensor_cfg.body_ids are ordered as [LeftFoot, RightFoot] (ankle/foot).
    • The velocity command is available in env.command_manager.get_term(command_name).command (N,3): [vx, vy, wz].
    """
    device = env.device
    N = env.num_envs
    robot = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # --- dt of environment (drop to 1/60 in the absence of explicit dt) ---
    dt = env.sim.cfg.dt * env.cfg.decimation 
    # print(f'dt {dt}')
    # --- movement command and gates ---
    cmd = env.command_manager.get_term(command_name).command  # (N,3)
    cmd = torch.as_tensor(cmd, device=device, dtype=torch.float32)
    lin_mag = cmd[:, :2].norm(dim=1)
    ang_mag = cmd[:, 2].abs() if cmd.shape[1] >= 3 else torch.zeros_like(lin_mag)

    near_zero = (lin_mag < lin_deadband) & (ang_mag < ang_deadband)   # no movement
    moving    = ~near_zero                                            # movement

    # --- contacts on two stops (resistant to noise via .amax according to history) ---
    if use_history:
        f_hist = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # (N,H,2,3)
        fmag   = f_hist.norm(dim=-1).amax(dim=1)                              # (N,2)
    else:
        f_now  = cs.data.net_forces_w[:, sensor_cfg.body_ids, :]             # (N,2,3)
        fmag   = f_now.norm(dim=-1)                                          # (N,2)

    Lc = fmag[:, 0] > contact_force_threshold   # (N,)
    Rc = fmag[:, 1] > contact_force_threshold   # (N,)
    both_down = Lc & Rc
    any_down  = Lc | Rc
    flight    = ~any_down

    # --- per-env initialization/state storage ---
    _need_init = not all(hasattr(env, a) for a in [
        "_g1_prev_Lc", "_g1_prev_Rc", "_g1_air_L", "_g1_air_R",
        "_g1_t_air_L", "_g1_t_air_R", "_g1_last_lead"
    ])
    if _need_init:
        env._g1_prev_Lc  = torch.zeros(N, dtype=torch.bool, device=device)
        env._g1_prev_Rc  = torch.zeros(N, dtype=torch.bool, device=device)
        env._g1_air_L    = torch.zeros(N, dtype=torch.bool, device=device)    # is in the air now?
        env._g1_air_R    = torch.zeros(N, dtype=torch.bool, device=device)
        env._g1_t_air_L  = torch.zeros(N, dtype=torch.float32, device=device) #  accumulated air time
        env._g1_t_air_R  = torch.zeros(N, dtype=torch.float32, device=device)
        # who was the last "leader" (0=L, 1=R, 2=no/has not been yet)
        env._g1_last_lead = torch.full((N,), 2, dtype=torch.long, device=device)

    # reset state at the end of episodes
    resets = (env.termination_manager.terminated | env.termination_manager.time_outs)
    if resets.any():
        env._g1_prev_Lc[resets]  = Lc[resets]
        env._g1_prev_Rc[resets]  = Rc[resets]
        env._g1_air_L[resets]    = ~Lc[resets]
        env._g1_air_R[resets]    = ~Rc[resets]
        env._g1_t_air_L[resets]  = 0.0
        env._g1_t_air_R[resets]  = 0.0
        env._g1_last_lead[resets]= 2

    # --- liftoff/touchdown events ---
    liftoff_L  = (~Lc) & env._g1_prev_Lc    # went into the air
    liftoff_R  = (~Rc) & env._g1_prev_Rc
    touchdown_L= Lc & (~env._g1_prev_Lc)    # landed
    touchdown_R= Rc & (~env._g1_prev_Rc)

    # --- update flags "in the air" ---
    env._g1_air_L = ~Lc
    env._g1_air_R = ~Rc

    # --- increment of time "in the air" ---
    env._g1_t_air_L = torch.where(env._g1_air_L, env._g1_t_air_L + dt, env._g1_t_air_L)
    env._g1_t_air_R = torch.where(env._g1_air_R, env._g1_t_air_R + dt, env._g1_t_air_R)

    # on liftoff, reset the timer for the corresponding leg
    env._g1_t_air_L = torch.where(liftoff_L, torch.zeros_like(env._g1_t_air_L), env._g1_t_air_L)
    env._g1_t_air_R = torch.where(liftoff_R, torch.zeros_like(env._g1_t_air_R), env._g1_t_air_R)

    # --- base reward ---
    reward = torch.zeros(N, dtype=torch.float32, device=device)

    #1) REST: bonus for double support
    reward = reward + idle_double_support_bonus_val * (near_zero & both_down).float()

    #2) MOVEMENT: Alternation + Target Swing
    if moving.any():
        # (a) penalty for "flying the body" in movement (both legs in the air)
        reward = reward - flight_penalty * (moving & flight).float()

        # (b) touchdown bonus: exp(-(t−T)^2/σ^2)
        def touchdown_score(t_air):
            # Gaussian without 0.5: peak = 1.0 at t==target
            return torch.exp(-((t_air - target_swing_time) ** 2) / (swing_sigma ** 2 + 1e-12))

        td_L = moving & touchdown_L
        td_R = moving & touchdown_R

        score_L = torch.zeros(N, device=device)
        score_R = torch.zeros(N, device=device)
        if td_L.any():
            score_L[td_L] = touchdown_score(env._g1_t_air_L[td_L])
        if td_R.any():
            score_R[td_R] = touchdown_score(env._g1_t_air_R[td_R])

        # alternation: if the same leader in a row - penalty
        # leader - the leg that just completed the swing (touchdown)
        new_lead = torch.where(td_R, torch.ones(N, device=device, dtype=torch.long),
                       torch.where(td_L, torch.zeros(N, device=device, dtype=torch.long),
                                   torch.full((N,), 2, device=device, dtype=torch.long)))  #2: "no event"

        same_lead = (new_lead != 2) & (env._g1_last_lead != 2) & (new_lead == env._g1_last_lead)
        alt_ok    = (new_lead != 2) & ((env._g1_last_lead == 2) | (new_lead != env._g1_last_lead))

        # apply bonus/penalty only to those envs where there was a touchdown
        td_any = td_L | td_R
        td_reward = touchdown_bonus * (score_L + score_R)
        td_reward = torch.where(same_lead, td_reward - same_lead_penalty, td_reward)
        reward = reward + torch.where(td_any, td_reward, torch.zeros_like(td_reward))

        # update last_lead where touchdown was
        env._g1_last_lead = torch.where(td_L, torch.zeros_like(env._g1_last_lead),
                                 torch.where(td_R, torch.ones_like(env._g1_last_lead),
                                             env._g1_last_lead))

        # (c) "shaping" during flight - bring the duration closer to the target
        # current active swing timer (take max from left/right, but only when exactly one is in the air)
        single_support_air = env._g1_air_L ^ env._g1_air_R
        cur_t = torch.where(env._g1_air_L, env._g1_t_air_L,
                     torch.where(env._g1_air_R, env._g1_t_air_R, torch.zeros(N, device=device)))
        shaping = torch.exp(-((cur_t - target_swing_time) ** 2) / (swing_sigma ** 2 + 1e-12))
        reward = reward + shaping_weight * (moving & single_support_air).float() * shaping

    # --- update "previous contacts" ---
    env._g1_prev_Lc = Lc
    env._g1_prev_Rc = Rc
    
    # print(f"alternating_airtime_reward: {reward}")
    return reward 
    
def step_phase_reward(
    env,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg:  SceneEntityCfg = SceneEntityCfg("robot",          body_names=".*_ankle_roll_link"),

    # --- commands and gates ---
    lin_deadband: float = 0.03,      # m/s: "almost stopped"
    use_history: bool = True,        # we take the maximum from the sensor history (more resistant to noise)

    # --- contact force ---
    contact_force_threshold: float = 0.0,  # H: optional threshold (0 - no threshold)
    amp_ref: float = 800.0,                # H: desired max force for normalization and reference A

    # --- phase generator ---
    freq_gain_hz_per_mps: float = 2.0,     # f = k_f * |v|; at |v|=0.5 => 1 Hz; at |v|=1.0 => 2 Hz
    clamp_freq: tuple = (0.0, 4.0),        

    # --- exponent from MSE ---
    mse_beta: float = 5.0,                 # r_leg = exp(-beta * MSE_leg) 

):
    """
    Returns the [num_envs] tensor with a reward.
    Expected order: sensor_cfg.body_ids: [LeftFoot, RightFoot].
    Takes the velocity command from env.command_manager[command_name].command (N,3): [vx, vy, wz].
    """
    device = env.device
    N = env.num_envs
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # dt симуляции
    dt = env.sim.cfg.dt * env.cfg.decimation

    # --- command and movement flag ---
    cmd = env.command_manager.get_term(command_name).command  # (N,3)
    cmd = torch.as_tensor(cmd, device=device, dtype=torch.float32)
    lin_mag = cmd[:, :2].norm(dim=1)                          # (N,)
    moving = lin_mag >= lin_deadband

    # --- contact forces between two feet (norms) ---
    if use_history:
        f_hist = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # (N,H,2,3)
        fmag = f_hist.norm(dim=-1).amax(dim=1)                                # (N,2)
    else:
        f_now = cs.data.net_forces_w[:, sensor_cfg.body_ids, :]              # (N,2,3)
        fmag = f_now.norm(dim=-1)                                            # (N,2)

    if contact_force_threshold > 0.0:
        fmag = torch.where(fmag > contact_force_threshold, fmag, torch.zeros_like(fmag))

    # --- per-env state: accumulate oscillator phase ---
    if not hasattr(env, "_g2_phase"):
        env._g2_phase = torch.zeros(N, dtype=torch.float32, device=device)

    # frequency by speed: f = k_f * |v| (Hz)
    f_hz = freq_gain_hz_per_mps * lin_mag
    if clamp_freq is not None:
        f_hz = torch.clamp(f_hz, clamp_freq[0], clamp_freq[1])

    # phase increment: dφ = 2π f dt
    dphi = (2.0 * torch.pi * f_hz * dt).to(device)
    env._g2_phase = (env._g2_phase + dphi) % (2.0 * torch.pi)

    # --- Reference signals for legs ---
    # Right leg: φ_R = φ
    # Left leg: φ_L = φ + π (antiphase)
    phi = env._g2_phase
    s_ref_R = amp_ref * torch.relu(torch.sin(phi))
    s_ref_L = amp_ref * torch.relu(torch.sin(phi + torch.pi))

    # --- normalization of forces to [0,1] by amp_ref and clipping ---
    eps = 1e-6
    act_R = torch.clamp(fmag[:, 1] / (amp_ref + eps), 0.0, 1.0)  # assume order [L, R]
    act_L = torch.clamp(fmag[:, 0] / (amp_ref + eps), 0.0, 1.0)
    ref_R = torch.clamp(s_ref_R / (amp_ref + eps), 0.0, 1.0)
    ref_L = torch.clamp(s_ref_L / (amp_ref + eps), 0.0, 1.0)

    # --- MSE for each leg (for this step) ---
    mse_R = (act_R - ref_R) ** 2
    mse_L = (act_L - ref_L) ** 2

    # --- exponent from MSE, multiplication of legs ---
    r_R = torch.exp(-mse_beta * mse_R)
    r_L = torch.exp(-mse_beta * mse_L)
    reward = r_R * r_L

    # --- gate on movement: at rest we do not affect the total reward ---
    reward = reward * moving.float()
    # print(f"step_phase_reward: {reward}")
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
    reward = torch.exp(-beta * mse)              # (N,)

    # if there is no support, an additional fine may be imposed (optional)
    if no_support_penalty > 0.0:
        reward = reward - no_support_penalty * (~has_support).float()
    # print(f"reward: {reward}")
    return reward
          
