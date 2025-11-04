import torch
from typing import Optional

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
import omni.log
import omni
from pxr import UsdGeom, Gf, Sdf


# =========================
# utils
# =========================

def _norm_env_ids(env: ManagerBasedRLEnv, env_ids: Optional[torch.Tensor]) -> torch.Tensor:
    """Casts env_ids to 1D LongTensor on the correct device."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long).view(-1)
    return env_ids


def _target_rng(env: ManagerBasedRLEnv) -> torch.Generator:
    """A random number generator that cannot be reset by the manager."""
    if not hasattr(env, "_target_rng"):
        g = torch.Generator(device=env.device)
        g.manual_seed(torch.seed())  
        env._target_rng = g
    return env._target_rng


# =========================
#  TARGET
# =========================

@torch.no_grad()
def respawn_reached_targets(
    env: ManagerBasedRLEnv,
    env_ids: Optional[torch.Tensor],
    reach_radius: float = 0.35,
    r_min: float = 2.0,
    r_max: float = 6.0,
    z: float = 0.05,
):
    """If the robot approaches the target closer than reach_radius, recreate the target further away.."""
    env_ids = _norm_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    robot: Articulation = env.scene["robot"]
    target: RigidObject = env.scene["target"]

    d = target.data.root_pos_w[env_ids, :2] - robot.data.root_pos_w[env_ids, :2]
    dist = torch.linalg.norm(d, dim=1)
    hit_local = torch.nonzero(dist < reach_radius, as_tuple=False).squeeze(-1)
    if hit_local.numel() > 0:
        hit_global = env_ids[hit_local]
        respawn_target(env, hit_global, r_min=r_min, r_max=r_max, z=z)


@torch.no_grad()
def respawn_target(
    env: ManagerBasedRLEnv,
    env_ids: Optional[torch.Tensor],
    r_min: float = 2.0,
    r_max: float = 6.0,
    z: float = 0.05,
):
    """Target respawn similar to robot reset: default_root_state -> write_pose/vel -> reset()."""
    env_ids = _norm_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    robot: Articulation = env.scene["robot"]
    target: RigidObject = env.scene["target"]
    g = _target_rng(env)

    # center - the current pose of the robot
    base_xy = robot.data.root_pos_w[env_ids, :2]

    # evenly over the ring area [r_min, r_max]
    u = torch.rand(env_ids.numel(), generator=g, device=env.device)
    r = torch.sqrt(u * (r_max**2 - r_min**2) + r_min**2)
    yaw = torch.rand(env_ids.numel(), generator=g, device=env.device) * (2.0 * torch.pi)
    goal_xy = base_xy + torch.stack((r * torch.cos(yaw), r * torch.sin(yaw)), dim=1)

    # full state 
    root_state = target.data.default_root_state[env_ids].clone()  # (N, 13)
    root_state[:, 0:2] = goal_xy
    root_state[:, 2] = z
    root_state[:, 7:13] = 0.0

    target.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    target.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    target.reset(env_ids=env_ids)


# =========================
#  OBSTACLES (COLLECTION)
# =========================

@torch.no_grad()
def spawn_obstacles_at_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    num_obstacles: tuple[int, int] = (6, 10),
    r_max: float = 6.5,
    keepout_robot: float = 0.8,
    keepout_goal: float = 0.8,
    min_obstacle_gap: float = 0.5,
    obstacle_z: float = 0.4,
    sensor_name: float = "lidar_top"
) -> None:
    """Lays out cylinder columns around the robot like a "real" reset:
    read buffers -> modify -> write_object_pose_to_sim(..., env_ids) -> reset(env_ids).
    Vertical columns: quaternion = identity.
    """
    env_ids = _norm_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    robot: Articulation = env.scene["robot"]
    target: RigidObject = env.scene["target"]
    coll = env.scene["obstacles"]  # RigidObjectCollection

    # current collection buffers
    pos_w = coll.data.object_pos_w.clone()    # (E, M, 3)
    quat_w = coll.data.object_quat_w.clone()  # (E, M, 4)

    # we normalize the order of the axes, if there was (M, E, 3)
    if pos_w.shape[0] != env.scene.num_envs and pos_w.shape[1] == env.scene.num_envs:
        pos_w = pos_w.permute(1, 0, 2).contiguous()
        quat_w = quat_w.permute(1, 0, 2).contiguous()

    _, M = pos_w.shape[0], pos_w.shape[1]
    if M == 0:
        return

    base_xy = robot.data.root_pos_w[env_ids, :2]   # (N,2)
    goal_xy = target.data.root_pos_w[env_ids, :2]  # (N,2) — keep your distance from the target

    n_low, n_high = num_obstacles
    k_active = torch.randint(
        low=n_low, high=n_high + 1, size=(env_ids.numel(),), device=env.device
    ).clamp_(max=M)

    quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    for i, e in enumerate(env_ids.tolist()):
        placed_xy = []
        attempts, need = 0, int(k_active[i].item())

        while len(placed_xy) < need and attempts < 300:
            attempts += 1
            # evenly spaced around the robot in a square; can be replaced with a ring
            x = (torch.rand((), device=env.device) - 0.5) * 2.0 * (r_max + 1.0)
            y = (torch.rand((), device=env.device) - 0.5) * 2.0 * (r_max + 1.0)
            p_xy = torch.stack((x, y)) + base_xy[i]

            ok_robot = (p_xy - base_xy[i]).norm() > keepout_robot
            ok_goal = (p_xy - goal_xy[i]).norm() > keepout_goal
            ok_gap = True if not placed_xy else torch.all(
                (torch.stack(placed_xy) - p_xy).norm(dim=1) > min_obstacle_gap
            )
            if ok_robot and ok_goal and ok_gap:
                placed_xy.append(p_xy)

        # We fill all M spaces: active ones -> on stage, the rest are hidden underground
        for j in range(M):
            if j < len(placed_xy):
                pos_w[e, j, 0:2] = placed_xy[j]
                pos_w[e, j, 2] = obstacle_z
                quat_w[e, j, :] = quat_identity  # strictly vertical
            else:
                pos_w[e, j, 0:2] = 0.0
                pos_w[e, j, 2] = -1.0  # below ground level
                quat_w[e, j, :] = quat_identity

    pose7 = torch.cat((pos_w, quat_w), dim=-1)  # (E, M, 7)

    # Required with env_ids and reset collections
    pose7_sel = pose7.index_select(0, env_ids)  # (len(env_ids), num_objects, 7)
    coll.write_object_pose_to_sim(pose7_sel, env_ids=env_ids)
    coll.reset(env_ids=env_ids)
    

def randomize_obstacles_physx_and_usd(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    num_obstacles: tuple[int, int] = (6, 10),
    r_max: float = 6.5,
    keepout_robot: float = 0.8,
    keepout_goal: float = 0.8,
    min_obstacle_gap: float = 0.5,
    obstacle_z: float = 0.4,
    sensor_name: str = "lidar_top",
) -> None:
    """Spreads cylindrical columns on reset:
    - vectorially generates new poses in PhysX (RigidObjectCollection),
    - synchronizes the same poses in USD so that RegexRayCaster sees them in new locations.
    """
    env_ids = _norm_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    device = env.device

    robot: Articulation = env.scene["robot"]
    target: RigidObject = env.scene["target"]
    coll: RigidObjectCollection = env.scene["obstacles"]  # RigidObjectCollection

    # current collection buffers
    pos_w = coll.data.object_pos_w.clone()    # (E, M, 3)
    quat_w = coll.data.object_quat_w.clone()  # (E, M, 4), format wxyz

    # normalize the axes if there were (M, E, 3)
    if pos_w.shape[0] != env.scene.num_envs and pos_w.shape[1] == env.scene.num_envs:
        pos_w = pos_w.permute(1, 0, 2).contiguous()
        quat_w = quat_w.permute(1, 0, 2).contiguous()

    E, M = pos_w.shape[0], pos_w.shape[1]
    if M == 0:
        return

    N = env_ids.numel()
    base_xy = robot.data.root_pos_w[env_ids, :2]   # (N,2)
    goal_xy = target.data.root_pos_w[env_ids, :2]  # (N,2)

    n_low, n_high = num_obstacles
    k_active = torch.randint(
        low=n_low, high=n_high + 1, size=(N,), device=device
    ).clamp_(max=M)   # (N,)

    # unit quaternion in wxyz format
    quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    # ---- 1. VECTORLY generate candidates for all env ----

    C = max(n_high * 2, M) 

    # random squared offsets
    offsets = (torch.rand(N, C, 2, device=device) - 0.5) * 2.0 * (r_max + 1.0)  # (N,C,2)
    cand_xy = offsets + base_xy[:, None, :]                                     # (N,C,2)

    # indent from the robot and target
    d_robot = torch.linalg.norm(cand_xy - base_xy[:, None, :], dim=-1)  # (N,C)
    d_goal  = torch.linalg.norm(cand_xy - goal_xy[:, None, :], dim=-1)  # (N,C)
    mask_keepout = (d_robot > keepout_robot) & (d_goal > keepout_goal)  # (N,C)

    # pairwise distances between candidates for min_obstacle_gap
    diff = cand_xy[:, :, None, :] - cand_xy[:, None, :, :]             # (N,C,C,2)
    dist_cc = torch.linalg.norm(diff, dim=-1)                           # (N,C,C)
    eye = torch.eye(C, device=device).unsqueeze(0)                      # (1,C,C)
    dist_cc = dist_cc + eye * 1e6                                       # mask the diagonal
    min_dist_cc, _ = dist_cc.min(dim=-1)                                # (N,C)
    mask_gap = min_dist_cc > min_obstacle_gap

    valid_mask = mask_keepout & mask_gap                                # (N,C)

    # select up to M valid candidates per-env via topk
    # give valid candidates a score of [1,2], and invalid candidates a score of 0
    rand_scores = torch.rand_like(d_robot)
    score = torch.where(valid_mask, 1.0 + rand_scores, torch.zeros_like(rand_scores))  # (N,C)

    top_vals, top_idx = score.topk(M, dim=1)                             # (N,M)
    # выбранные XY
    chosen_xy = cand_xy.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, 2))  # (N,M,2)
    is_real = top_vals > 0.5                                                # (N,M)

    # mask "at index j < k_active"
    idx_m = torch.arange(M, device=device).unsqueeze(0).expand(N, -1)       # (N,M)
    active_mask = idx_m < k_active.unsqueeze(1)                              # (N,M)

    place_mask = active_mask & is_real                                      # (N,M)

    # ---- 2. Write new poses to pos_w / quat_w VECTORIALLY ---
    pos_local = pos_w[env_ids].clone()      # (N,M,3)
    quat_local = quat_w[env_ids].clone()    # (N,M,4)

    
    pos_local[..., 0:2] = 0.0
    pos_local[..., 2] = -1.0
    quat_local[:] = quat_identity

    # where place_mask=True, we put the selected positions and the desired height
    mask3 = place_mask.unsqueeze(-1)  # (N,M,1)

    pos_local[..., 0:2] = torch.where(mask3, chosen_xy, pos_local[..., 0:2])
    z_new = torch.full_like(pos_local[..., 2], obstacle_z)
    pos_local[..., 2] = torch.where(place_mask, z_new, pos_local[..., 2])

    # back to global buffers
    pos_w[env_ids] = pos_local
    quat_w[env_ids] = quat_local

    # ---- 3. Writing new poses in PhysX ----
    pose7_sel = torch.cat((pos_w[env_ids], quat_w[env_ids]), dim=-1)  # (N,M,7)
    coll.write_object_pose_to_sim(pose7_sel, env_ids=env_ids)
    coll.reset(env_ids=env_ids)

    # ---- 4. Synchronizing the same poses in USD for RayCaster ----
    stage = omni.usd.get_context().get_stage()
    pos_local_cpu = pos_local.detach().cpu().numpy()  # (N,M,3)

    for idx_e, e in enumerate(env_ids.tolist()):
        env_path = f"/World/envs/env_{e}"
        for j in range(M):
            prim_path = f"{env_path}/obst_{j:02d}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                continue

            xform = UsdGeom.Xformable(prim)
            x, y, z = pos_local_cpu[idx_e, j, :].tolist()

            ops = xform.GetOrderedXformOps()
            translate_op = None
            transform_op = None
            for op in ops:
                t = op.GetOpType()
                if t == UsdGeom.XformOp.TypeTranslate and translate_op is None:
                    translate_op = op
                elif t == UsdGeom.XformOp.TypeTransform and transform_op is None:
                    transform_op = op

            if translate_op is not None:
                translate_op.Set(Gf.Vec3d(x, y, z))
            elif transform_op is not None:
                m = Gf.Matrix4d(1.0)
                m.SetTranslateOnly(Gf.Vec3d(x, y, z))
                transform_op.Set(m)
            else:
                new_translate = xform.AddTranslateOp()
                new_translate.Set(Gf.Vec3d(x, y, z))
