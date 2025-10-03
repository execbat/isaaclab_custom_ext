import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from isaaclab.managers import SceneEntityCfg

# ---------- TARGET ----------

@torch.no_grad()
def respawn_reached_targets(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None,
                            reach_radius: float = 0.35, r_min: float = 2.0, r_max: float = 6.0, z: float = 0.05):
    robot  = env.scene["robot"]
    target = env.scene["target"]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long).view(-1)

    d = target.data.root_pos_w[env_ids, :2] - robot.data.root_pos_w[env_ids, :2]
    dist = torch.linalg.norm(d, dim=1)
    hit_local = torch.nonzero(dist < reach_radius, as_tuple=False).squeeze(-1)
    if hit_local.numel() > 0:
        hit_global = env_ids[hit_local]
        respawn_target(env, hit_global, r_min=r_min, r_max=r_max, z=z)


def respawn_target(env: ManagerBasedRLEnv, env_ids: torch.Tensor, r_min=2.0, r_max=6.0, z=0.05):
    device = env.device
    env_ids = env_ids.to(device=device, dtype=torch.long).view(-1)
    N = env_ids.numel()
    if N == 0:
        return

    robot  = env.scene["robot"]
    target = env.scene["target"]

    base_xy = robot.data.root_pos_w[env_ids, :2]  # (N,2)

    yaw = torch.rand(N, device=device) * (2 * torch.pi)
    rad = r_min + torch.rand(N, device=device) * (r_max - r_min)
    goal_xy = base_xy + torch.stack((rad * torch.cos(yaw), rad * torch.sin(yaw)), dim=1)

    pos  = target.data.root_pos_w   # (E,3)
    quat = target.data.root_quat_w  # (E,4)

    pos[env_ids, 0:2] = goal_xy
    pos[env_ids, 2]   = z

    pose7_env = torch.cat((pos[env_ids], quat[env_ids]), dim=1)  # (N,7)
    target.write_root_pose_to_sim(pose7_env, env_ids=env_ids)

# ---------- OBSTACLES (RESET ONLY) ----------

def _collect_obstacle_objects(env: ManagerBasedRLEnv):
    """Собирает список объектов препятствий obst_XX. Fallback: scene['obstacles'] (один объект)."""
    obst_objs = []
    if hasattr(env.scene, "rigid_objects") and isinstance(env.scene.rigid_objects, dict):
        names = [k for k in env.scene.rigid_objects.keys() if k.startswith("obst_")]
        obst_objs = [env.scene[k] for k in names]
    else:
        names = [n for n in dir(env.scene) if n.startswith("obst_")]
        obst_objs = [getattr(env.scene, n) for n in names]

    if len(obst_objs) == 0 and "obstacles" in getattr(env.scene, "__getitem__", lambda k: {}) and isinstance(env.scene["obstacles"], object):
        return [env.scene["obstacles"]]
    return obst_objs


@torch.no_grad()
def spawn_obstacles_at_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    num_obstacles: tuple[int, int] = (6, 10),
    r_max: float = 6.5,
    keepout_robot: float = 0.8,
    keepout_goal: float = 0.8,
    min_obstacle_gap: float = 0.5,
    obstacle_z: float = 0.3,
) -> None:
    device = env.device
    N = env_ids.numel()

    robot  = env.scene["robot"]
    target = env.scene["target"]

    base_xy = robot.data.root_pos_w[env_ids, :2]   # (N,2)
    goal_xy = target.data.root_pos_w[env_ids, :2]  # (N,2) 

    coll = env.scene["obstacles"]  # RigidObjectCollection

    pos  = coll.data.object_pos_w   # (E, M, 3)
    quat = coll.data.object_quat_w  # (E, M, 4)

    if pos.shape[0] != env.scene.num_envs and pos.shape[1] == env.scene.num_envs:
        pos  = pos.permute(1, 0, 2)   # view
        quat = quat.permute(1, 0, 2)  # view

    E, M = pos.shape[0], pos.shape[1]
    if M == 0:
        return

    n_low, n_high = num_obstacles
    k_active = torch.randint(low=n_low, high=n_high + 1, size=(N,), device=device).clamp_(max=M)

    for i, e in enumerate(env_ids.tolist()):
        placed_xy = []
        attempts, need = 0, int(k_active[i].item())

        while len(placed_xy) < need and attempts < 300:
            attempts += 1
            x = (torch.rand((), device=device) - 0.5) * 2.0 * (r_max + 1.0)
            y = (torch.rand((), device=device) - 0.5) * 2.0 * (r_max + 1.0)
            p_xy = torch.stack((x, y)) + base_xy[i]

            ok_robot = (p_xy - base_xy[i]).norm() > keepout_robot
            ok_goal  = (p_xy - goal_xy[i]).norm()  > keepout_goal
            ok_gap   = True if not placed_xy else torch.all(
                (torch.stack(placed_xy) - p_xy).norm(dim=1) > min_obstacle_gap
            )
            if ok_robot and ok_goal and ok_gap:
                placed_xy.append(p_xy)

        for j in range(M):
            if j < len(placed_xy):
                pos[e, j, 0:2] = placed_xy[j]
                pos[e, j, 2]   = obstacle_z
            else:
                pos[e, j, 2]   = -1.0

    pose7 = torch.cat((pos, quat), dim=-1)
    coll.write_object_pose_to_sim(pose7)
    
def _yaw_from_quat_xyzw(q_xyzw: torch.Tensor) -> torch.Tensor:
    """Возвращает yaw (рад) из кватерниона(ов) XYZW."""
    _, _, yaw = euler_xyz_from_quat(q_xyzw, wrap_to_2pi=False)
    return wrap_to_pi(yaw)

@torch.no_grad()
def commands_towards_target(
    env,
    env_ids=None,
    robot_entity: SceneEntityCfg = SceneEntityCfg("robot"),
    target_entity: SceneEntityCfg = SceneEntityCfg("target"),
    max_speed: float = 1.0,
    k_lin: float = 1.0,
    stop_radius: float = 0.15,
    allow_strafe: bool = False,
):

    # 1) read states (world frame)
    robot = env.scene[robot_entity.name]
    target = env.scene[target_entity.name]

    base_state_w   = robot.data.root_state_w   # (N, 13): pos(3), quat(4=wxyz), lin vel(3), ang vel(3)
    target_state_w = target.data.root_state_w  # (N, 13)

    base_pos_w  = base_state_w[:, 0:3]
    base_quat_w = base_state_w[:, 3:7]
    target_pos_w = target_state_w[:, 0:3]

    # 2) vector to the target in Worls CS and 2D-distance
    delta_w = target_pos_w - base_pos_w
    delta_xy = delta_w[:, :2]
    dist_xy = torch.linalg.norm(delta_xy, dim=1)

    # 3) absolute heading in World atan2(dy, dx)
    heading_goal = torch.atan2(delta_w[:, 1], delta_w[:, 0])

    # 4) linear speeds
    if not allow_strafe:
        v_forward = torch.clamp(k_lin * dist_xy, 0.0, max_speed)
        # гslow down speed rear of the target
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


    cmd_term = env.command_manager.get_term("base_velocity")  # UniformVelocityCommand
    cmd = cmd_term.command  # (N, 3): lin_x, lin_y, heading (heading_command=True)
    cmd[:, 0] = vx_cmd
    cmd[:, 1] = vy_cmd
    cmd[:, 2] = heading_goal
