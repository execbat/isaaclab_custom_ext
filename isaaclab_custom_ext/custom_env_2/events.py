import torch
from typing import Optional

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi


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
    """One random number generator per Wednesday (not reset by the manager)."""
    if not hasattr(env, "_target_rng"):
        g = torch.Generator(device=env.device)
        g.manual_seed(torch.seed())  
        env._target_rng = g
    return env._target_rng


# =========================
#  (TARGET)
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
    """If the robot approaches the target closer than reach_radius, we recreate the target further."""
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

    base_xy = robot.data.root_pos_w[env_ids, :2]

    u = torch.rand(env_ids.numel(), generator=g, device=env.device)
    r = torch.sqrt(u * (r_max**2 - r_min**2) + r_min**2)
    yaw = torch.rand(env_ids.numel(), generator=g, device=env.device) * (2.0 * torch.pi)
    goal_xy = base_xy + torch.stack((r * torch.cos(yaw), r * torch.sin(yaw)), dim=1)

    root_state = target.data.default_root_state[env_ids].clone()  # (N, 13)
    root_state[:, 0:2] = goal_xy
    root_state[:, 2] = z
    root_state[:, 7:13] = 0.0

    target.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    target.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
    target.reset(env_ids=env_ids)


# =========================
# obstacles (COLLECTION)
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
) -> None:
    """We lay out the cylinder columns around the robot, like a "real" reset:
    read the current collection buffers -> modifiable -> write_object_pose_to_sim(..., env_ids) -> reset(env_ids).
    """
    env_ids = _norm_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    robot: Articulation = env.scene["robot"]
    target: RigidObject = env.scene["target"]
    coll = env.scene["obstacles"]  # RigidObjectCollection


    pos_w = coll.data.object_pos_w.clone()    # (E, M, 3)
    quat_w = coll.data.object_quat_w.clone()  # (E, M, 4)

    # (M,E,3)
    if pos_w.shape[0] != env.scene.num_envs and pos_w.shape[1] == env.scene.num_envs:
        pos_w = pos_w.permute(1, 0, 2).contiguous()
        quat_w = quat_w.permute(1, 0, 2).contiguous()

    E, M = pos_w.shape[0], pos_w.shape[1]
    if M == 0:
        return

    base_xy = robot.data.root_pos_w[env_ids, :2]   # (N,2)
    goal_xy = target.data.root_pos_w[env_ids, :2]  # (N,2) 

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


        for j in range(M):
            if j < len(placed_xy):
                pos_w[e, j, 0:2] = placed_xy[j]
                pos_w[e, j, 2] = obstacle_z
                quat_w[e, j, :]  = quat_identity
            else:
                pos_w[e, j, 0:2] = 0.0
                pos_w[e, j, 2] = -1.0  # below the hround level
                quat_w[e, j, :]  = quat_identity

    pose7 = torch.cat((pos_w, quat_w), dim=-1)  # (E, M, 7)


    coll.write_object_pose_to_sim(pose7, env_ids=env_ids)
    coll.reset(env_ids=env_ids)


# =========================
# useful for control
# =========================

def _yaw_from_quat_xyzw(q_xyzw: torch.Tensor) -> torch.Tensor:
    """Returns yaw (rad) from quaternion(s) XYZW."""
    _, _, yaw = euler_xyz_from_quat(q_xyzw, wrap_to_2pi=False)
    return wrap_to_pi(yaw)


@torch.no_grad()
def commands_towards_target(
    env: ManagerBasedRLEnv,
    env_ids: Optional[torch.Tensor] = None,
    robot_entity: SceneEntityCfg = SceneEntityCfg("robot"),
    target_entity: SceneEntityCfg = SceneEntityCfg("target"),
    max_speed: float = 1.0,
    k_lin: float = 1.0,
    stop_radius: float = 0.15,
    allow_strafe: bool = False,
):
    """Simple command - go to target: linear speed ~ distance, heading = atan2."""
    robot = env.scene[robot_entity.name]
    target = env.scene[target_entity.name]

    base_state_w = robot.data.root_state_w   # (N, 13)
    target_state_w = target.data.root_state_w

    base_pos_w = base_state_w[:, 0:3]
    base_quat_w = base_state_w[:, 3:7]
    target_pos_w = target_state_w[:, 0:3]

    delta_w = target_pos_w - base_pos_w
    delta_xy = delta_w[:, :2]
    dist_xy = torch.linalg.norm(delta_xy, dim=1)

    heading_goal = torch.atan2(delta_w[:, 1], delta_w[:, 0])

    if not allow_strafe:
        v_forward = torch.clamp(k_lin * dist_xy, 0.0, max_speed)
        v_forward = torch.where(dist_xy < stop_radius, torch.zeros_like(v_forward), v_forward)
        vx_cmd = v_forward
        vy_cmd = torch.zeros_like(v_forward)
    else:
        delta_in_base_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(base_quat_w), delta_w)
        vx = torch.clamp(k_lin * delta_in_base_yaw[:, 0], -max_speed, max_speed)
        vy = torch.clamp(k_lin * delta_in_base_yaw[:, 1], -max_speed, max_speed)
        mask_stop = dist_xy < stop_radius
        vx_cmd = torch.where(mask_stop, torch.zeros_like(vx), vx)
        vy_cmd = torch.where(mask_stop, torch.zeros_like(vy), vy)

    cmd_term = env.command_manager.get_term("base_velocity")  # TargetChase/UniformVelocityCommand
    cmd = cmd_term.command  # (N, 3): lin_x, lin_y, heading
    cmd[:, 0] = vx_cmd
    cmd[:, 1] = vy_cmd
    cmd[:, 2] = heading_goal

