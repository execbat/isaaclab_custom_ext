# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

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
"""
General.
"""


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


class is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        return (reset_buf * (~env.termination_manager.time_outs)).float()


"""
Root penalties.
"""


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)


"""
Joint penalties.
"""


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


"""
Action penalties.
"""


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


"""
Contact sensor.
"""


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def desired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize if none of the desired contacts are present."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    zero_contact = (~contacts).all(dim=1)
    return 1.0 * zero_contact


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1)


"""
Velocity-tracking rewards.
"""


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    print(f'LIN CMD {env.command_manager.get_command(command_name)[:, :2]}')
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    print(f'ANG CMD {env.command_manager.get_command(command_name)[:, 2]}')
    return torch.exp(-ang_vel_error / std**2)

##################################    
# AnimalMath bonus for repeatability of external target
    
def miander_tracking_reward_exp(
    env,
    std: float = 0.25,                 # ширина колокола в НОРМИРОВАННЫХ [-1,1] единицах
    mask_name: str = "dof_mask",
    target_name: str = "target_joint_pose",
) -> torch.Tensor:
    """
    Экспоненциальный трекинг активных DOF к таргету:
        reward = exp( - MSE_active / std^2 )

    • mask=1 → DOF участвуют в ошибке; mask=0 → игнор
    • позиции нормируются в [-1,1] по soft-лимитам
    • если активных DOF нет → возвращаем 0
    """
    asset = env.scene["robot"]
    q      = asset.data.joint_pos
    device, dtype = q.device, q.dtype

    # команды
    mask_cmd   = env.command_manager.get_term(mask_name).command
    target_cmd = env.command_manager.get_term(target_name).command
    mask   = (mask_cmd > 0.5).to(device=device)                             # (N,J) bool
    target = torch.as_tensor(target_cmd, dtype=dtype, device=device)        # (N,J)

    # нормированные текущие углы [-1,1]
    qmin   = asset.data.soft_joint_pos_limits[..., 0].to(device=device, dtype=dtype)
    qmax   = asset.data.soft_joint_pos_limits[..., 1].to(device=device, dtype=dtype)
    offset = 0.5 * (qmin + qmax)
    norm   = 2.0 * (q - offset) / (qmax - qmin + 1e-6)                      # (N,J)

    # MSE по активным DOF
    se        = (norm - target).pow(2)                                      # (N,J)
    active_n  = mask.sum(dim=1)                                             # (N,)
    mse       = (se * mask).sum(dim=1) / active_n.clamp_min(1)              # (N,)

    # экспоненциальная награда
    r = torch.exp(-mse / (std ** 2))
    r = torch.where(active_n > 0, r, torch.zeros_like(r))
    return r



    
def miander_untracking_reward_exp(
    env,
    std: float = 0.25,                           # ширина «колокола» в НОРМИРОВАННЫХ [-1,1] ед.
    command_name: str = "base_velocity",
    lin_cmd_threshold: float = 0.05,
    leg_bits: tuple[int, ...] = (0,1,3,4,7,8,11,12,15,16,19,20),
    mask_name: str = "dof_mask",
    init_attr: str = "JOINT_INIT_POS_NORM",
) -> torch.Tensor:
    """
    Экспоненциальная награда за «удержание инита» на НЕмаскированных DOF:
        r = exp( - MSE_inactive / std^2 )

    • маска=0 → DOF участвуют (к иниту), маска=1 → игнор
    • при наличии команды на движение — вклад суставов ног (leg_bits) обнуляется
    • позы нормируются в [-1,1] по soft-лимитам
    """
    asset = env.scene["robot"]
    q = asset.data.joint_pos
    device, dtype = q.device, q.dtype

    # --- маска (inactive = !mask) ---
    mask_cmd = env.command_manager.get_term(mask_name).command
    inverse_mask = (mask_cmd <= 0.5).to(device=device)                 # (N,J) bool

    # --- нормированные текущие углы [-1,1] ---
    qmin = asset.data.soft_joint_pos_limits[..., 0].to(device=device, dtype=dtype)
    qmax = asset.data.soft_joint_pos_limits[..., 1].to(device=device, dtype=dtype)
    offset = 0.5 * (qmin + qmax)
    qn = 2.0 * (q - offset) / (qmax - qmin + 1e-6)                     # (N,J)

    # --- нормированный инит (1,J) или (N,J), на нужном девайсе ---
    init_qn = getattr(env, init_attr, None)
    if init_qn is None:
        # если нет кэша — попробуем default_joint_pos; иначе возьмём qn как инит
        init_q = getattr(asset.data, "default_joint_pos", None)
        if init_q is not None:
            init_q = torch.as_tensor(init_q, dtype=dtype, device=device)     # (J,)
            init_qn = 2.0 * (init_q - offset) / (qmax - qmin + 1e-6)         # (J,)
            init_qn = init_qn.unsqueeze(0)                                    # (1,J)
        else:
            init_qn = qn.detach().mean(dim=0, keepdim=True)                   # (1,J)
        setattr(env, init_attr, init_qn)
    else:
        init_qn = torch.as_tensor(init_qn, dtype=dtype, device=device)
        if init_qn.dim() == 1:
            init_qn = init_qn.unsqueeze(0)                                    # (1,J)

    # --- исключаем ноги при движении ---
    cmd = env.command_manager.get_term(command_name).command
    cmd = torch.as_tensor(cmd, dtype=dtype, device=device)                    # (N, C)
    move_gate = cmd[:, :2].norm(dim=1) > lin_cmd_threshold                    # (N,)

    W = inverse_mask.float()                                                  # (N,J)
    if move_gate.any() and leg_bits:
        if not hasattr(env, "_leg_bits_tensor"):
            env._leg_bits_tensor = torch.as_tensor(leg_bits, device=device, dtype=torch.long)
        W[move_gate][:, env._leg_bits_tensor] = 0.0

    # --- MSE по немаскированным (после обнуления ног при движении) ---
    se = (qn - init_qn).pow(2)                                                # (N,J)
    denom = W.sum(dim=1)                                                      # (N,)
    mse = (se * W).sum(dim=1) / denom.clamp_min(1.0)                          # (N,)

    # --- экспоненциальная награда ---
    r = torch.exp(-mse / (std ** 2))
    r = torch.where(denom > 0, r, torch.zeros_like(r))                        # если нечего оценивать → 0
    return r




def pelvis_height_target_reward(env: MathManagerBasedRLEnv,
                                target: float =  0.795, # 0.74,
                                alpha: float = 0.2) -> torch.Tensor:
    """
    Экспоненциальная награда: r = exp(-alpha * |z - target|)

    Args:
        env:   среда MathManagerBasedRLEnv.
        target: желаемая высота таза в метрах.
        alpha:  крутизна колокола 

    Returns:
        Tensor[num_envs] — reward в диапазоне (0‥1].
    """
    # Берём Z‑координату pelvis
    asset = env.scene["robot"]
    pelvis_z = asset.data.root_pos_w[:, 2]           # shape [N]
    # print(pelvis_z)

    error = torch.abs(pelvis_z - target)      # |z − 0.7|
    reward = torch.exp(-alpha * error)     # e^(−α·err)

    return reward
    


def feet_separation_and_alignment_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",

    # (0) прямой штраф по суставам ankle_pitch → к нейтрали
    ankle_pitch_joint_names=("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    w_ankle_neutral: float = 2.0,

    # (1) «ступня параллельно полу» через нормаль подошвы
    w_tilt: float = 1.0,
    foot_normal_local=(0.0, 0.0, 1.0),

    # (1b) «против носка/пятки»: продольная ось должна быть горизонтальна
    w_pitch_flat: float = 2.0,
    foot_forward_local=(1.0, 0.0, 0.0),

    # (2) выравнивание стоп с направлением таза (только при двухопорной фазе)
    w_align: float = 0.5,

    # (3) продольная длина шага (только при двухопорной)
    w_stride: float = 0.7,
    step_gain: float = 0.40,
    beta_stride: float = 3.0,
    v_near_zero: float = 0.05,
    w_opposite_at_zero: float = 0.5,

    # (4) перехлёст (только при двухопорной)
    w_cross: float = 0.7,
    beta_cross: float = 6.0,

    # боковая ширина (только при двухопорной)
    shoulder_width: float = 0.35,
    beta_sep: float = 2.0,
    w_sep: float = 1.0,

    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Суммарный penalty ≥ 0 за «некорректную» постановку стоп при контактах.
    Усиливает: (а) плоскость касания (без носка/пятки), (б) нейтраль по ankle_pitch,
    (в) корректную геометрию шага и отсутствие перехлёста/узкой стойки.
    """
    device = env.device
    robot: Articulation | RigidObject = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # --- helpers: локальные оси для обеих стоп (1,2,3)
    def _per_foot(vec):
        v = torch.as_tensor(vec, dtype=torch.float32, device=device)
        return v.view(1, 1, 3).expand(1, 2, 3)

    nrm_loc = _per_foot(foot_normal_local)     # нормаль подошвы в ЛСК стопы
    fwd_loc = _per_foot(foot_forward_local)    # продольная ось стопы в ЛСК
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3)

    # --- контакты (устойчиво к шуму за счёт .amax по истории)
    f_hist = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]    # (N,H,2,3)
    fmag = f_hist.norm(dim=-1).amax(dim=1)                                  # (N,2)
    contacts = fmag > contact_force_threshold                               # (N,2) [L,R]
    both_down = contacts.all(dim=1)                                         # (N,)
    n_down = contacts.float().sum(dim=1).clamp(min=1.0)                     # (N,)

    # --- позы/ориентации
    feet_pos_w  = robot.data.body_pos_w[:, sensor_cfg.body_ids, :]          # (N,2,3)
    feet_quat_w = robot.data.body_quat_w[:, sensor_cfg.body_ids, :]         # (N,2,4)
    root_pos_w  = robot.data.root_pos_w                                     # (N,3)
    root_quat_w = robot.data.root_quat_w                                    # (N,4)

    base_fwd_w = math_utils.quat_apply(
        root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(root_pos_w)
    )
    base_lat_w = math_utils.quat_apply(
        root_quat_w, torch.tensor([0.0, 1.0, 0.0], device=device).expand_as(root_pos_w)
    )
    f_xy = base_fwd_w[:, :2]
    f_xy = f_xy / f_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    l_xy = base_lat_w[:, :2]
    l_xy = l_xy / l_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # ---------- (0) ankle_pitch → к нейтрали (только для контактирующих ног)
    if not hasattr(env, "_ankle_pitch_ids"):
        jnames = list(robot.data.joint_names)
        try:
            env._ankle_pitch_ids = torch.tensor(
                [jnames.index(ankle_pitch_joint_names[0]),
                 jnames.index(ankle_pitch_joint_names[1])],
                device=device, dtype=torch.long
            )
        except ValueError as e:
            raise RuntimeError(f"Не найден ankle_pitch сустав: {e}")

        init_norm = env.JOINT_INIT_POS_NORM.to(device)
        env._ankle_pitch_init_norm = init_norm[env._ankle_pitch_ids]  # (2,)

    jidx = env._ankle_pitch_ids
    q    = robot.data.joint_pos
    qmin = robot.data.soft_joint_pos_limits[..., 0]
    qmax = robot.data.soft_joint_pos_limits[..., 1]
    qmid = 0.5 * (qmin + qmax)
    qhal = 0.5 * (qmax - qmin)
    qn   = ((q - qmid) / (qhal + 1e-6)).clamp(-1.0, 1.0)

    qn_ank    = qn[:, jidx]                                # (N,2)
    qn_target = env._ankle_pitch_init_norm.view(1, 2)      # (1,2)
    ankle_err = (qn_ank - qn_target).abs()                 # (N,2)
    ankle_neutral_pen = (ankle_err * contacts.float()).sum(dim=1) / n_down

    # ---------- (1) «ступни параллельно полу» через нормаль подошвы
    nrm_w  = math_utils.quat_apply(feet_quat_w, nrm_loc.expand_as(feet_pos_w))  # (N,2,3)
    cos_up = (nrm_w * world_up).sum(dim=-1).abs().clamp(0.0, 1.0)               # (N,2)
    tilt_each = 1.0 - cos_up
    tilt_pen  = (tilt_each * contacts.float()).sum(dim=1) / n_down

    # ---------- (1b) НОВОЕ: «против носка/пятки» — продольная ось горизонтальна
    foot_fwd_w = math_utils.quat_apply(feet_quat_w, fwd_loc.expand_as(feet_pos_w))  # (N,2,3)
    pitch_vert_each = foot_fwd_w[:, :, 2].abs()                                     # |z-компонента «вперёд»|
    pitch_flat_pen  = (pitch_vert_each * contacts.float()).sum(dim=1) / n_down

    # ---------- (2) выравнивание стоп по направлению таза (только при двухопорной)
    foot_dir_xy = foot_fwd_w[:, :, :2]
    foot_dir_xy = foot_dir_xy / foot_dir_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos_to_pelvis = (foot_dir_xy * f_xy.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)  # (N,2)
    align_term = 0.5 * (1.0 - cos_to_pelvis).mean(dim=1)
    align_pen  = torch.where(both_down, align_term, torch.zeros_like(align_term))

    # ---------- (3) продольная длина шага / «противоположность» при почти нулевой V
    rel_xy = feet_pos_w[:, :, :2] - root_pos_w[:, :2].unsqueeze(1)  # (N,2,2)
    sL = (rel_xy[:, 0, :] * f_xy).sum(dim=-1)
    sR = (rel_xy[:, 1, :] * f_xy).sum(dim=-1)
    d_long = (sL - sR).abs()

    v_cmd = env.command_manager.get_term(command_name).command[:, :2]
    v_mag = v_cmd.norm(dim=-1)
    d_des = step_gain * v_mag
    stride_term = 1.0 - torch.exp(-beta_stride * (d_long - d_des).abs())
    stride_pen  = torch.where(both_down, stride_term, torch.zeros_like(stride_term))

    opposite_bool  = (sL * sR) < 0
    low_speed_gate = (v_near_zero - v_mag).clamp(min=0.0) / max(v_near_zero, 1e-6)
    opposite_pen   = w_opposite_at_zero * (opposite_bool.float() * low_speed_gate)

    # ---------- (4) перехлёст (только при двухопорной)
    yL = (rel_xy[:, 0, :] * l_xy).sum(dim=-1)
    yR = (rel_xy[:, 1, :] * l_xy).sum(dim=-1)
    cross_depth = torch.relu(-yL) + torch.relu(+yR)
    cross_term  = 1.0 - torch.exp(-beta_cross * cross_depth)
    cross_pen   = torch.where(both_down, cross_term, torch.zeros_like(cross_term))

    # ---------- боковая ширина (только при двухопорной)
    dist_xy = (feet_pos_w[:, 0, :2] - feet_pos_w[:, 1, :2]).norm(dim=-1)
    sep_term = 1.0 - torch.exp(-beta_sep * (dist_xy - shoulder_width).abs())
    sep_pen  = torch.where(both_down, sep_term, torch.zeros_like(sep_term))

    # ---------- суммарный penalty
    penalty = (
        w_ankle_neutral * ankle_neutral_pen
      + w_tilt          * tilt_pen
      + w_pitch_flat    * pitch_flat_pen
      + w_align         * align_pen
      + w_stride        * stride_pen
      +                   opposite_pen
      + w_cross         * cross_pen
      + w_sep           * sep_pen
    )
    return penalty.clamp_min(0.0)






# Additional
def masked_progress_reward(env, eps=0.0, neg_scale: float = 0.1, mask_name: str = "dof_mask"):
    robot = env.scene["robot"]
    q, qd = robot.data.joint_pos, robot.data.joint_vel

    qmin = robot.data.soft_joint_pos_limits[..., 0]
    qmax = robot.data.soft_joint_pos_limits[..., 1]
    mid  = 0.5*(qmin+qmax)
    scl  = 2.0/(qmax - qmin + 1e-6)

    qn, qdn = (q - mid) * scl, qd * scl
    tgt = env.command_manager.get_term("target_joint_pose").command
    msk = env.command_manager.get_term(mask_name).command > 0.5

    err  = qn - tgt
    prog = -(err * qdn)                      # >0 если |err| уменьшается

    # опц. мёртвая зона против дрожи
    if eps > 0.0:
        prog = torch.sign(prog) * torch.clamp(prog.abs() - eps, min=0.0)

    # ослабляем отрицательную часть в 10 раз (neg_scale=0.1)
    prog = torch.where(prog >= 0.0, prog, neg_scale * prog)

    prog = torch.where(msk, prog, torch.zeros_like(prog))
    cnt  = msk.sum(dim=1).clamp_min(1)
    r    = prog.sum(dim=1) / cnt
    return r.clamp(-1.0, 1.0)

def unmasked_progress_to_init_reward(
    env,
    eps: float = 0.0,
    neg_scale: float = 0.1,
    mask_name: str = "dof_mask",
    # исключаем ноги при движении:
    command_name: str = "base_velocity",
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    leg_bits = (0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20),
    # где хранится нормированный инит:
    init_attr: str = "JOINT_INIT_POS_NORM",
):
    robot = env.scene["robot"]
    q, qd = robot.data.joint_pos, robot.data.joint_vel
    device, dtype = q.device, q.dtype

    # нормировка поз/скоростей в [-1, 1]
    qmin = robot.data.soft_joint_pos_limits[..., 0].to(device=device, dtype=dtype)
    qmax = robot.data.soft_joint_pos_limits[..., 1].to(device=device, dtype=dtype)
    mid  = 0.5 * (qmin + qmax)
    scl  = 2.0 / (qmax - qmin + 1e-6)
    qn, qdn = (q - mid) * scl, qd * scl

    # маска: берём НЕмаскированные DOF
    msk = env.command_manager.get_term(mask_name).command > 0.5
    msk = msk.to(device=device)
    inv = ~msk  # то, что оцениваем

    # если есть команда на движение — исключаем DOF ног из расчёта
    try:
        cmd = env.command_manager.get_term(command_name).command
        cmd = torch.as_tensor(cmd, dtype=dtype, device=device)   # (N, C)
        lin_mag = torch.zeros(cmd.shape[0], device=device, dtype=dtype)
        if cmd.shape[1] >= 2:
            lin_mag = (cmd[:, 0]**2 + cmd[:, 1]**2).sqrt()
        ang_mag = torch.zeros_like(lin_mag)
        if cmd.shape[1] >= 3:
            ang_mag = cmd[:, 2].abs()
        moving = (lin_mag > lin_deadband) | (ang_mag > ang_deadband)  # (N,)
    except Exception:
        moving = torch.zeros(q.shape[0], dtype=torch.bool, device=device)

    if leg_bits:
        leg_mask_1d = torch.zeros(q.shape[1], dtype=torch.bool, device=device)
        leg_mask_1d[list(leg_bits)] = True
        drop = moving.unsqueeze(1) & leg_mask_1d.unsqueeze(0)  # (N, D)
        inv = inv & ~drop

    # нормированный инит (кэшируется на нужном девайсе)
    init_qn = getattr(env, init_attr, None)
    if init_qn is None:
        init_q = getattr(robot.data, "default_joint_pos", None)
        if init_q is not None:
            init_q = torch.as_tensor(init_q, dtype=dtype, device=device)  # (D,)
            init_qn = (init_q - mid) * scl                                # (D,)
            if init_qn.dim() == 1:
                init_qn = init_qn.unsqueeze(0).expand(qn.shape[0], -1)    # (N, D)
        else:
            init_qn = qn.detach().clone()
        setattr(env, init_attr, init_qn)
    else:
        init_qn = torch.as_tensor(init_qn, dtype=dtype, device=device)
        if init_qn.dim() == 1:
            init_qn = init_qn.unsqueeze(0).expand(qn.shape[0], -1)

    # прогресс к инициальной позе: >0, если |err| уменьшается
    err  = qn - init_qn
    prog = -(err * qdn)

    # мёртвая зона против дрожи
    if eps > 0.0:
        prog = torch.sign(prog) * torch.clamp(prog.abs() - eps, min=0.0)

    # ослабляем «регресс»
    prog = torch.where(prog >= 0.0, prog, neg_scale * prog)

    # учитываем только выбранные DOF, остальным — 0 вклад
    prog = torch.where(inv, prog, torch.zeros_like(prog))

    cnt = inv.sum(dim=1).clamp_min(1)
    r   = prog.sum(dim=1) / cnt
    return r.clamp(-1.0, 1.0)    
    
def unmasked_stillness_penalty(
    env,
    w_vel: float = 1.0,
    w_tau: float = 0.0,   # включи >0, когда захочешь учитывать усилия
):
    """
    Позитивный cost для НЕмаскированных DOF (mask==0):
      • |joint_vel| всегда
      • |applied_torque| если доступен
    """
    robot = env.scene["robot"]

    # неактивные = (1 - mask)  — маску берём из команды dof_mask
    inactive_mask = (1.0 - env.command_manager.get_term("dof_mask").command).float()  # [N,J]
    denom = inactive_mask.sum(dim=1).clamp_min(1.0)

    # скорость суставов
    qd = robot.data.joint_vel  # [N,J]
    vel_cost = (qd.abs() * inactive_mask).sum(dim=1) / denom

    # усилие (в твоей сборке есть applied_torque; при отсутствии — пробуем computed_torque)
    tau = getattr(robot.data, "applied_torque", None)
    if tau is None:
        tau = getattr(robot.data, "computed_torque", None)

    if tau is not None:
        tau_cost = (tau.abs() * inactive_mask).sum(dim=1) / denom
    else:
        tau_cost = torch.zeros_like(vel_cost)

    return w_vel * vel_cost + w_tau * tau_cost
 
    
    
def masked_success_bonus(env, eps=0.03, bonus=1.0):
    robot = env.scene["robot"]
    q     = robot.data.joint_pos
    qmin  = robot.data.soft_joint_pos_limits[...,0]
    qmax  = robot.data.soft_joint_pos_limits[...,1]
    mid   = 0.5*(qmin+qmax)
    scl   = 2.0/(qmax-qmin+1e-6)

    qn  = (q - mid) * scl
    tgt = env.command_manager.get_term("target_joint_pose").command
    msk = env.command_manager.get_term("dof_mask").command > 0.5
   
    ok = ((qn - tgt).abs() <= eps) | (~msk)      # немаскированные считаем «ок»
    all_ok = ok.all(dim=1)
    return bonus * all_ok.float()
    
def lateral_slip_penalty(env, command_name="base_velocity"):
    robot = env.scene["robot"]
    cmd   = env.command_manager.get_term(command_name).command  # (N,3): vx, vy, wz in base
    v_b   = robot.data.root_lin_vel_b[:, :2]                    # (N,2)
    # если команда почти нулевая — не штрафуем
    mag = cmd[:,:2].norm(dim=1, keepdim=True) + 1e-6
    dir = cmd[:,:2] / mag
    # поперечная составляющая
    lat = v_b - (v_b*dir).sum(dim=1, keepdim=True)*dir
    return lat.norm(dim=1)    # положительное число   
    
def com_over_support_reward_fast(
    env,
    sensor_cfg: SceneEntityCfg,                 # ContactSensor c телами-опорами (обычно стопы)
    asset_cfg : SceneEntityCfg = SceneEntityCfg("robot"),
    contact_force_threshold: float = 5.0,       # Н: фильтр шума
    sigma: float = 0.06,                        # м: радиус точности
    weighted: bool = True,                      # True → средняя точка взвешена силой
    require_both_when_idle: bool = True,        # НОВОЕ: в покое требуем двухопорие
    mask_name: str = "dof_mask",
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    leg_bits: tuple[int, ...] = (0,1,3,4,7,8,11,12,15,16,19,20),
) -> torch.Tensor:
    """
    R ∈ [0,1]. Максимум, когда проекция CoM в XY совпадает с опорной точкой.
    При require_both_when_idle=True: если |cmd|≈0 и ноги выключены по маске — ревард только при двухопории.
    """
    device = env.device
    robot  = env.scene[asset_cfg.name]
    cs     = env.scene.sensors[sensor_cfg.name]

    # --- контакты (устойчиво к шуму по истории) ---
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)  # (N,K)
    contacts   = fmag > contact_force_threshold                                                  # (N,K)
    any_contact  = contacts.any(dim=1)                                                           # (N,)
    both_down    = contacts.all(dim=1)                                                           # (N,)

    # --- опорная точка XY ---
    feet_xy = robot.data.body_pos_w[:, sensor_cfg.body_ids, :2]                                  # (N,K,2)
    if weighted:
        w = (fmag * contacts.float())                                                            # (N,K)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        support_xy = (feet_xy * w.unsqueeze(-1)).sum(dim=1)                                      # (N,2)
    else:
        m = contacts.float()
        support_xy = (feet_xy * m.unsqueeze(-1)).sum(dim=1) / m.sum(dim=1, keepdim=True).clamp_min(1e-6)

    # --- центр масс XY ---
    if hasattr(robot.data, "com_pos_w"):
        com_xy = robot.data.com_pos_w[:, :2]
    else:
        pos_w = robot.data.body_pos_w[:, :, :2]
        masses = None
        for attr in ("body_masses", "link_masses", "masses"):
            if hasattr(robot.data, attr):
                masses = getattr(robot.data, attr)
                break
        if masses is None:
            com_xy = pos_w.mean(dim=1)
        else:
            m = masses.unsqueeze(-1)
            com_xy = (pos_w * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)

    # --- базовый ревард ---
    d2 = (com_xy - support_xy).square().sum(dim=1)
    inv_two_sigma2 = 0.5 / (sigma * sigma)
    reward = torch.exp(-d2 * inv_two_sigma2)                                                     # (N,)

    # --- «покой + ноги выключены по маске» → разрешаем ревард только при двухопории
    if require_both_when_idle:
        cmd   = env.command_manager.get_term("base_velocity").command  # (N,3)
        near0 = (cmd[:, :2].norm(dim=1) < lin_deadband) & (cmd[:, 2].abs() < ang_deadband)

        dof_mask = env.command_manager.get_term(mask_name).command > 0.5
        if not hasattr(env, "_leg_bits_tensor"):
            env._leg_bits_tensor = torch.as_tensor(leg_bits, device=device, dtype=torch.long)
        legs_inactive = (dof_mask[:, env._leg_bits_tensor].sum(dim=1) == 0)

        gate_idle = near0 & legs_inactive
        allow = torch.where(gate_idle, both_down, any_contact)
        return torch.where(allow, reward, torch.zeros_like(reward))
    else:
        return torch.where(any_contact, reward, torch.zeros_like(reward))
    
def no_command_motion_penalty(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),

    # когда |cmd_lin| < lin_deadband → штраф почти макс; чем меньше deadband, тем «жёстче»
    lin_deadband: float = 0.05,   # м/с
    ang_deadband: float = 0.05,   # рад/с

    # нормировочные масштабы (примерно под «типичные» макс-скорости)
    lin_scale: float = 1.0,       # м/с  → влияет на величину линейного штрафа
    ang_scale: float = 1.0,       # рад/с → влияет на величину углового штрафа
) -> torch.Tensor:
    """
    Штраф за движение, когда НЕТ команды на движение.
    penalty = gate_lin * (||v_xy||/lin_scale)^2 + gate_ang * (|w_z|/ang_scale)^2,
    где gate_* ≈ 1 при маленькой команде и → 0 при росте команды.
    """
    asset = env.scene[asset_cfg.name]

    # команда [vx, vy, wz] в базе
    cmd = env.command_manager.get_term(command_name).command  # (N,3)
    cmd_lin_mag = cmd[:, :2].norm(dim=1)                      # (N,)
    cmd_ang_mag = cmd[:, 2].abs()                             # (N,)

    # скорости базы
    v_xy = asset.data.root_lin_vel_b[:, :2]                   # (N,2)
    w_z  = asset.data.root_ang_vel_b[:, 2]                    # (N,)

    # плавные «шторки» (1 при нулевой команде → 0 около deadband и дальше)
    # экспонента даёт гладкую и дифференцируемую форму
    gate_lin = torch.exp(- (cmd_lin_mag / max(lin_deadband, 1e-6))**2)  # (N,)
    gate_ang = torch.exp(- (cmd_ang_mag / max(ang_deadband, 1e-6))**2)  # (N,)

    lin_term = (v_xy.norm(dim=1) / max(lin_scale, 1e-6))**2
    ang_term = (w_z.abs() / max(ang_scale, 1e-6))**2

    penalty = gate_lin * lin_term + gate_ang * ang_term
    return penalty

def joint_limit_saturation_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.10,     # «запретная зона» у границы в норм. шкале: 0.10 = 10% от края
    beta: float = 8.0,        # крутизна роста штрафа при заходе в margin
    use_mask: bool = False,   # если True — учитываем только активные DOF по dof_mask
    mask_name: str = "dof_mask",
) -> torch.Tensor:
    """
    ПЕНАЛЬТИ за «прилипание к лимитам» ещё ДО их пересечения.

    • Нормируем углы в [-1,1] по soft limits.
    • Считаем slack = 1 - |q_n|  (насколько далеко от края).
    • Если slack < margin → есть нарушение. violation = (margin - slack)/margin ∈ [0,1].
    • per_joint = 1 - exp(-beta * violation)  — гладкий рост к 1 у самой границы.
    • Возврат: средний штраф по (активным|всем) DOF: shape (N,), ≥0.
    """
    asset = env.scene[asset_cfg.name]
    q     = asset.data.joint_pos
    qmin  = asset.data.soft_joint_pos_limits[..., 0]
    qmax  = asset.data.soft_joint_pos_limits[..., 1]

    mid   = 0.5 * (qmin + qmax)
    half  = 0.5 * (qmax - qmin)
    qn    = ((q - mid) / (half + 1e-6)).clamp(-1.0, 1.0)   # норм. позы ∈ [-1,1]

    # расстояние до края (в норм. шкале)
    slack = 1.0 - qn.abs()                   # ∈ [0,1]
    # нарушение «зашли в margin-зону»
    violation = (margin - slack).clamp_min(0.0) / max(margin, 1e-6)   # ∈ [0,1]
    per_joint = 1.0 - torch.exp(-beta * violation)                    # гладко к 1

    if use_mask:
        mask = env.command_manager.get_term(mask_name).command > 0.5   # (N,J) bool
        num = mask.sum(dim=1).clamp_min(1)
        pen = (per_joint * mask.float()).sum(dim=1) / num
    else:
        pen = per_joint.mean(dim=1)

    return pen
    
def single_foot_stationary_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_force_threshold: float = 3.0,   # стало мягче (было 5.0)
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    command_name: str = "base_velocity",
    use_mask: bool = True,
    mask_name: str = "dof_mask",
    leg_bits: list[int] = (0,1,3,4,7,8,11,12,15,16,19,20),
    scale_by_asymmetry: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Штраф за одноопорную стойку ТОЛЬКО когда:
      • |cmd| ≈ 0  И  все суставы ног выключены по маске.
    """
    device = env.device
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # near-zero команды
    cmd   = env.command_manager.get_term(command_name).command
    near0 = (cmd[:, :2].norm(dim=1) < lin_deadband) & (cmd[:, 2].abs() < ang_deadband)

    # ноги выключены по маске?
    if use_mask and hasattr(env.command_manager, "get_term"):
        dof_mask = env.command_manager.get_term(mask_name).command  # (N,J)
        if not hasattr(env, "_leg_bits_tensor2"):
            env._leg_bits_tensor2 = torch.as_tensor(leg_bits, device=device, dtype=torch.long)
        legs_inactive = (dof_mask[:, env._leg_bits_tensor2].sum(dim=1) <= 0.0)
    else:
        legs_inactive = torch.zeros_like(near0, dtype=torch.bool)

    # строгий гейт
    gate = near0 & legs_inactive
    if not gate.any():
        return torch.zeros(env.num_envs, device=device)

    # контакты
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)  # (N,2)
    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold
    single_support = Lc ^ Rc

    pen = (single_support & gate).float()

    if scale_by_asymmetry:
        asym = (fmag[:, 0] - fmag[:, 1]).abs() / (fmag.sum(dim=1) + eps)
        pen = pen * (1.0 + asym)  # 1…~2

    return pen
    
def leg_symmetry_idle_reward_norm(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    dof_mask_name: str = "dof_mask",
    obs_key_norm: str = "dof_pos_norm",   # принимаем для совместимости с конфигом
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    left_dofs: list[int]  = (0, 3, 7, 11, 15, 19),
    right_dofs: list[int] = (1, 4, 8, 12, 16, 20),
    w_sym: float = 0.6,
    w_init: float = 0.4,
    beta_sym: float = 6.0,
    beta_init: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Ревард > 0, когда:
      • |cmd| ≈ 0  И  ноги выключены по маске.
    Поощряет:
      (1) схожесть L/R, (2) близость обеих ног к init-позе (в норме [-1,1]).
    """
    device = env.device
    robot  = env.scene[asset_cfg.name]

    # --- врата: нулевые команды ---
    cmd   = env.command_manager.get_term(command_name).command  # (N,3)
    near0 = (cmd[:, :2].norm(dim=1) < lin_deadband) & (cmd[:, 2].abs() < ang_deadband)

    # --- ноги выключены по маске ---
    dof_mask = env.command_manager.get_term(dof_mask_name).command  # (N,J) in {0,1}
    legs_mask = dof_mask[:, list(left_dofs) + list(right_dofs)]
    legs_inactive = legs_mask.sum(dim=1) <= 0.0

    gate = near0 & legs_inactive
    if not gate.any():
        return torch.zeros(env.num_envs, device=device)

    # --- нормированные позы ∈[-1,1] (как в твоих термах) ---
    q     = robot.data.joint_pos
    qmin  = robot.data.soft_joint_pos_limits[..., 0]
    qmax  = robot.data.soft_joint_pos_limits[..., 1]
    mid   = 0.5 * (qmin + qmax)
    half  = 0.5 * (qmax - qmin)
    qn    = ((q - mid) / (half + eps)).clamp(-1.0, 1.0)

    # init-поза уже в норме
    q0n = env.JOINT_INIT_POS_NORM.to(device=device, dtype=qn.dtype)  # (J,)

    # разбиение на ноги
    L = torch.as_tensor(left_dofs,  device=device, dtype=torch.long)
    R = torch.as_tensor(right_dofs, device=device, dtype=torch.long)
    qnL, qnR = qn[:, L], qn[:, R]
    q0L, q0R = q0n[L], q0n[R]

    # --- ошибки ---
    sym_err  = (qnL - qnR).abs().mean(dim=1)  # (N,)
    init_err = 0.5 * ((qnL - q0L).abs().mean(dim=1) + (qnR - q0R).abs().mean(dim=1))

    # --- колокола ---
    r_sym  = torch.exp(-beta_sym  * sym_err)
    r_init = torch.exp(-beta_init * init_err)
    reward = (w_sym * r_sym + w_init * r_init) * gate.float()
    return reward    

def alternating_step_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",

    # врата
    lin_cmd_threshold: float = 0.05,
    use_mask: bool = True,
    mask_name: str = "dof_mask",
    leg_bits: tuple[int, ...] = (0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20),

    # контакты/устойчивость
    contact_force_threshold: float = 5.0,
    initial_lead: str = "right",   # первая двухопорная: кто спереди
    step_gain: float = 0.40,
    beta_stride: float = 3.0,
) -> torch.Tensor:
    device = env.device
    robot  = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    N = env.num_envs

    # --- инициализация внутреннего состояния ---
    _altstep_state_init_(env, N, device, initial_lead)

    # --- гейт: есть команда двигаться вперёд ---
    cmd = env.command_manager.get_term(command_name).command  # (N,3)
    v_mag = cmd[:, :2].norm(dim=1)
    move_gate = v_mag > lin_cmd_threshold

    # ноги «выключены» по маске?
    if use_mask:
        dof_mask = env.command_manager.get_term(mask_name).command
        legs_inactive = (dof_mask[:, list(leg_bits)].sum(dim=1) <= 0.0)
    else:
        legs_inactive = torch.zeros(N, dtype=torch.bool, device=device)

    gate = move_gate & legs_inactive
    if not gate.any():
        # поддерживаем prev_both актуальным, чтобы не терять синхронизацию
        _altstep_state_update_contacts_(env, cs, sensor_cfg, contact_force_threshold)
        return torch.zeros(N, device=device)

    # --- контакты (устойчиво по истории) ---
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)  # (N,2)
    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold
    both_down = Lc & Rc

    # --- кто впереди (по продольной оси базы) ---
    feet_pos_w  = robot.data.body_pos_w[:, sensor_cfg.body_ids, :2]   # (N,2,2)
    root_pos_w  = robot.data.root_pos_w[:, :2]
    root_quat_w = robot.data.root_quat_w

    base_fwd_w = math_utils.quat_apply(
        root_quat_w, torch.tensor([1.0,0.0,0.0], device=device).expand_as(robot.data.root_pos_w)
    )[:, :2]
    f_xy = base_fwd_w / base_fwd_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    rel_xy = feet_pos_w - root_pos_w.unsqueeze(1)
    sL = (rel_xy[:, 0, :] * f_xy).sum(dim=-1)
    sR = (rel_xy[:, 1, :] * f_xy).sum(dim=-1)
    lead_now = (sR > sL).long()  # 0=L, 1=R

    # --- ресеты эпизодов ---
    resets = env.termination_manager.terminated | env.termination_manager.time_outs
    if resets.any():
        _altstep_state_reset_(env, resets, both_down, lead_now, initial_lead)

    # --- liftoff-трекинг с прошлого DS ---
    env._alt_liftoff_L |= ~Lc
    env._alt_liftoff_R |= ~Rc
    liftoff_lead = torch.where(lead_now.bool(), env._alt_liftoff_R, env._alt_liftoff_L)

    # --- вход в двухопорную фазу (rising edge) ---
    ds_on = both_down & (~env._alt_prev_both)

    # --- ожидаемый лидер и оценка длины шага ---
    exp_lead = env._alt_expected_lead              # 0=L, 1=R
    correct_lead = (lead_now == exp_lead)

    d_long = (sL - sR).abs()
    d_des  = step_gain * v_mag
    stride_score = torch.exp(-beta_stride * (d_long - d_des).abs()).clamp(0.0, 1.0)  # 0..1

    reward = (gate & ds_on & correct_lead & liftoff_lead).float() * (0.5 + 0.5 * stride_score)

    # --- переключить ожидаемого лидера и сбросить liftoff-флаги для тех env, где был DS ---
    if ds_on.any():
        env._alt_expected_lead[ds_on] ^= 1
        env._alt_liftoff_L[ds_on] = False
        env._alt_liftoff_R[ds_on] = False

    env._alt_prev_both = both_down
    return reward


# --- helpers ---
def _altstep_state_init_(env, N, device, initial_lead: str):
    if not hasattr(env, "_alt_prev_both"):
        init_lead_bit = 1 if str(initial_lead).lower().startswith("r") else 0
        env._alt_prev_both     = torch.zeros(N, dtype=torch.bool, device=device)
        env._alt_expected_lead = torch.full((N,), init_lead_bit, dtype=torch.long, device=device)
        env._alt_liftoff_L     = torch.zeros(N, dtype=torch.bool, device=device)
        env._alt_liftoff_R     = torch.zeros(N, dtype=torch.bool, device=device)

def _altstep_state_reset_(env, mask, both_down, lead_now, initial_lead: str):
    init_lead_bit = 1 if str(initial_lead).lower().startswith("r") else 0
    env._alt_prev_both[mask]     = both_down[mask]
    env._alt_expected_lead[mask] = init_lead_bit
    env._alt_liftoff_L[mask]     = False
    env._alt_liftoff_R[mask]     = False

def _altstep_state_update_contacts_(env, cs, sensor_cfg, contact_force_threshold: float):
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)
    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold
    env._alt_prev_both = (Lc & Rc)
    
def alternating_same_lead_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",

    # врата
    lin_cmd_threshold: float = 0.05,   # «есть команда на движение»
    use_mask: bool = True,
    mask_name: str = "dof_mask",
    leg_bits: tuple[int, ...] = (0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20),

    # контакты/оценка шага
    contact_force_threshold: float = 5.0,
    step_gain: float = 0.40,   # желаемая длина шага ≈ step_gain * |v_cmd|
    beta_stride: float = 3.0,  # резкость оценки длины шага
) -> torch.Tensor:
    device = env.device
    robot  = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    N = env.num_envs

    # --- init внутреннего состояния один раз ---
    _asl_state_init_(env, N, device)

    # --- гейт: есть команда на движение и ноги выключены по маске ---
    cmd   = env.command_manager.get_term(command_name).command  # (N,3)
    v_mag = cmd[:, :2].norm(dim=1)
    move_gate = v_mag > lin_cmd_threshold

    if use_mask:
        dof_mask = env.command_manager.get_term(mask_name).command
        legs_inactive = (dof_mask[:, list(leg_bits)].sum(dim=1) <= 0.0)
    else:
        legs_inactive = torch.zeros(N, dtype=torch.bool, device=device)

    gate = move_gate & legs_inactive
    if not gate.any():
        _asl_update_prev_both(env, cs, sensor_cfg, contact_force_threshold)
        return torch.zeros(N, device=device)

    # --- контакты (устойчиво по истории) ---
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)  # (N,2)
    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold
    both_down = Lc & Rc

    # --- кто впереди вдоль продольной оси базы ---
    feet_pos_w  = robot.data.body_pos_w[:, sensor_cfg.body_ids, :2]  # (N,2,2)
    root_pos_w  = robot.data.root_pos_w[:, :2]
    root_quat_w = robot.data.root_quat_w

    base_fwd_w = math_utils.quat_apply(
        root_quat_w, torch.tensor([1.0,0.0,0.0], device=device).expand_as(robot.data.root_pos_w)
    )[:, :2]
    f_xy = base_fwd_w / base_fwd_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    rel_xy = feet_pos_w - root_pos_w.unsqueeze(1)
    sL = (rel_xy[:, 0, :] * f_xy).sum(dim=-1)
    sR = (rel_xy[:, 1, :] * f_xy).sum(dim=-1)
    lead_now = (sR > sL).long()   # 0=L, 1=R

    # --- ресеты эпизодов ---
    resets = env.termination_manager.terminated | env.termination_manager.time_outs
    if resets.any():
        _asl_state_reset_(env, resets, both_down, lead_now)

    # --- liftoff-трекинг между двухопорными ---
    env._asl_lift_L |= ~Lc
    env._asl_lift_R |= ~Rc
    had_any_liftoff = env._asl_lift_L | env._asl_lift_R    # хоть одна нога отрывалась?

    # --- вход в двухопорную фазу (rising edge) ---
    ds_on = both_down & (~env._asl_prev_both)

    # --- «тот же лидер?» и масштабирование по длине шага/лифтоффу ---
    same_lead = env._asl_have_last & (lead_now == env._asl_last_lead)
    d_long = (sL - sR).abs()
    d_des  = step_gain * v_mag
    stride_miss = 1.0 - torch.exp(-beta_stride * (d_long - d_des).abs())     # 0..1 (чем дальше от желаемого — тем ↑)

    # базовый пенальти: когда гейт активен, вход в DS и лидер не сменился
    pen = (gate & ds_on & same_lead).float()

    # усиление, если вообще не было liftoff'а (скорее всего «проскользили» корпусом)
    no_liftoff = ~had_any_liftoff
    scale = 1.0 + 0.5 * no_liftoff.float()            # 1.0 или 1.5
    pen = pen * scale * (0.5 + 0.5 * stride_miss)     # 0.5..1.5

    # --- обновление состояния после входа в DS ---
    if ds_on.any():
        env._asl_last_lead[ds_on] = lead_now[ds_on]
        env._asl_have_last[ds_on] = True
        env._asl_lift_L[ds_on] = False
        env._asl_lift_R[ds_on] = False

    env._asl_prev_both = both_down
    return pen


# --- helpers: per-env состояние ---
def _asl_state_init_(env, N, device):
    if not hasattr(env, "_asl_prev_both"):
        env._asl_prev_both  = torch.zeros(N, dtype=torch.bool, device=device)
        env._asl_last_lead  = torch.zeros(N, dtype=torch.long, device=device)  # 0=L,1=R
        env._asl_have_last  = torch.zeros(N, dtype=torch.bool, device=device)
        env._asl_lift_L     = torch.zeros(N, dtype=torch.bool, device=device)
        env._asl_lift_R     = torch.zeros(N, dtype=torch.bool, device=device)

def _asl_state_reset_(env, mask, both_down, lead_now):
    env._asl_prev_both[mask] = both_down[mask]
    env._asl_have_last[mask] = False
    env._asl_lift_L[mask]    = False
    env._asl_lift_R[mask]    = False
    # last_lead оставляем как есть; он будет установлен при первом DS после ресета

def _asl_update_prev_both(env, cs, sensor_cfg, thr):
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)
    Lc = fmag[:, 0] > thr
    Rc = fmag[:, 1] > thr
    env._asl_prev_both = (Lc & Rc)    
    
def heading_alignment_reward(
    env,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_cmd_threshold: float = 0.05,   # м/с
    beta: float = 4.0,                 # «резкость»
) -> torch.Tensor:
    """
    Награда ∈[0..1] за согласование продольной оси корпуса с направлением команды скорости.
    Работает только при |v_cmd| > lin_cmd_threshold.
    """
    robot = env.scene[asset_cfg.name]
    cmd   = env.command_manager.get_term(command_name).command  # (N,3)
    v_xy  = cmd[:, :2]
    v_mag = v_xy.norm(dim=1)
    gate  = v_mag > lin_cmd_threshold
    if not gate.any():
        return torch.zeros(env.num_envs, device=env.device)

    # единичный вектор «куда ехать» в мире
    v_dir = v_xy / v_mag.clamp_min(1e-6).unsqueeze(-1)

    # продольная ось корпуса в мире
    fwd_w = math_utils.quat_apply(
        robot.data.root_quat_w,
        torch.tensor([1.0, 0.0, 0.0], device=env.device).expand_as(robot.data.root_pos_w),
    )[:, :2]
    fwd_dir = fwd_w / fwd_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    cosang = (fwd_dir * v_dir).sum(dim=-1).clamp(-1.0, 1.0)
    # 1 при сонаправленности → падает при рассогласовании
    r = torch.exp(-beta * (1.0 - cosang))
    return r * gate.float()
    
def swing_foot_clearance_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    contact_force_threshold: float = 5.0,
    lin_cmd_threshold: float = 0.05,   # активируем только при команде на ходьбу
    h_des: float = 0.06,               # желаемый клиренс, м
    beta: float = 120.0,               # «узость» колокола
) -> torch.Tensor:
    """
    Награда за клиренс маховой стопы при одноопорной фазе.
    r = exp(-beta * (clearance - h_des)^2), где clearance = z_swing - z_stance.
    Только когда |v_cmd| > порога. Награда 0 в двухопоре/беспконтактной фазе.
    """
    device = env.device
    robot  = env.scene[asset_cfg.name]
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # врата по команде
    cmd   = env.command_manager.get_term(command_name).command
    v_mag = cmd[:, :2].norm(dim=1)
    move_gate = v_mag > lin_cmd_threshold
    if not move_gate.any():
        return torch.zeros(env.num_envs, device=device)

    # контакты (устойчиво по истории)
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)  # (N,2)
    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold

    # одноопора?
    single_L = Lc & (~Rc)   # левая — опора, правая — мах
    single_R = Rc & (~Lc)   # правая — опора, левая — мах
    single   = single_L | single_R
    if not single.any():
        return torch.zeros(env.num_envs, device=device)

    feet_z = robot.data.body_pos_w[:, sensor_cfg.body_ids, 2]  # (N,2)
    # clearance = z(swing) - z(stance)
    clr_L = (feet_z[:, 1] - feet_z[:, 0])   # если левая — опора → правая мах
    clr_R = (feet_z[:, 0] - feet_z[:, 1])   # если правая — опора → левая мах
    clearance = torch.zeros(env.num_envs, device=device)
    clearance[single_L] = clr_L[single_L]
    clearance[single_R] = clr_R[single_R]
    clearance = clearance.clamp_min(0.0)    # не поощряем «в землю»

    r = torch.exp(-beta * (clearance - h_des).pow(2))
    return r * (single & move_gate).float()    
    
    
def masked_action_rate_l2(env, mask_name: str = "dof_mask") -> torch.Tensor:
    """
    L2 на Δaction ТОЛЬКО по активным DOF (mask==1). Снижает дёрганье там,
    где мы реально что-то трекаем по UDP.
    """
    act  = env.action_manager.action
    prev = env.action_manager.prev_action
    d    = (act - prev).pow(2)
    m    = (env.command_manager.get_term(mask_name).command > 0.5).float()
    num  = m.sum(dim=1).clamp_min(1.0)
    return (d * m).sum(dim=1) / num
    
def masked_success_stable_bonus(
    env,
    eps: float = 0.03,      # допуск по позе в норме [-1..1]
    vel_eps: float = 0.03,  # допуск по норм. скорости
    bonus: float = 1.0,
    mask_name: str = "dof_mask",
) -> torch.Tensor:
    """
    Бонус, когда активные DOF (mask==1) и близки к целям, и «успокоились» по скорости.
    Хорош для удержания достигнутого таргета (без раскачки).
    """
    robot = env.scene["robot"]
    q     = robot.data.joint_pos
    qd    = robot.data.joint_vel
    qmin, qmax = robot.data.soft_joint_pos_limits[...,0], robot.data.soft_joint_pos_limits[...,1]
    mid, scl = 0.5*(qmin+qmax), 2.0/(qmax-qmin+1e-6)
    qn, qdn  = (q - mid)*scl, qd*scl

    tgt = env.command_manager.get_term("target_joint_pose").command
    msk = env.command_manager.get_term(mask_name).command > 0.5

    ok_pos = (qn - tgt).abs() <= eps
    ok_vel = qdn.abs() <= vel_eps
    ok = torch.where(msk, ok_pos & ok_vel, torch.ones_like(ok_pos, dtype=torch.bool))
    return bonus * ok.all(dim=1).float()  
    
def leg_pelvis_torso_coalignment_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # Имена тел как в твоём USD:
    pelvis_body: str = "pelvis",
    torso_body: str = "torso_link",
    left_thigh_body: str = "left_hip_pitch_link",
    left_shank_body: str = "left_knee_link",
    right_thigh_body: str = "right_hip_pitch_link",
    right_shank_body: str = "right_knee_link",

    # Что считаем «вперёд» в ЛСК звеньев:
    forward_local: tuple[float, float, float] = (1.0, 0.0, 0.0),

    # Веса компонентов внутри терма:
    w_yaw: float = 1.0,    # сонаправленность сегментов с тазом по курсу (XY)
    w_chain: float = 0.7,  # согласованность бедро↔голень (каждая нога)
    w_upright: float = 0.3,# горизонтальность продольной оси таза/торса (меньше «на носок/пятку»)

    # Маска DOF: если соответствующие DOF активны (mask==1), вклад компоненты отключаем
    mask_name: str = "dof_mask",
    left_dofs:  tuple[int, ...] = (0, 3, 7, 11, 15, 19),   # L hip pitch/roll/yaw, knee, ankle pitch/roll
    right_dofs: tuple[int, ...] = (1, 4, 8, 12, 16, 20),   # R hip pitch/roll/yaw, knee, ankle pitch/roll
    torso_bits: tuple[int, ...] = (),                      # при желании добавь waist_yaw и т.п.
) -> torch.Tensor:
    """
    r ∈ [0..1]. Поощряет согласованность направления звеньев ног, таза и торса:
      (A) yaw-сонаправленность звеньев с тазом (через XY),
      (B) согласованность «цепочки» бедро↔голень по yaw,
      (C) горизонтальность продольной оси таза/торса (меньше |z|).
    Если соответствующие DOF находятся под активной маской (mask==1), вклад
    соответствующей компоненты обнуляется, чтобы не конфликтовать с внешними таргетами.
    """
    device = env.device
    robot  = env.scene[asset_cfg.name]

    # --- кэш индексов тел ---
    if not hasattr(env, "_coalignment_body_ids"):
        names = list(robot.data.body_names)
        def _idx(n: str) -> int:
            try:
                return names.index(n)
            except ValueError as e:
                raise RuntimeError(f"[coalignment] не найдено тело '{n}' среди robot.data.body_names") from e
        ids = [
            _idx(pelvis_body),
            _idx(torso_body),
            _idx(left_thigh_body), _idx(left_shank_body),
            _idx(right_thigh_body), _idx(right_shank_body),
        ]
        env._coalignment_body_ids = torch.as_tensor(ids, device=device, dtype=torch.long)
    ids = env._coalignment_body_ids  # порядок: [pelvis, torso, Lth, Lsh, Rth, Rsh]

    # --- «вперёд» в мире для каждого сегмента ---
    quats = robot.data.body_quat_w[:, ids, :]  # (N,6,4)
    f_loc = torch.tensor(forward_local, device=device, dtype=torch.float32)\
                .view(1,1,3).expand(quats.shape[0], quats.shape[1], 3)
    fwd_w = math_utils.quat_apply(quats, f_loc)  # (N,6,3)

    # --- проекция на XY и нормализация ---
    fwd_xy = fwd_w[..., :2]
    fwd_xy = fwd_xy / fwd_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    pelvis_xy = fwd_xy[:, 0, :]  # (N,2)

    # (A) сонаправленность по yaw с тазом: torso, Lth, Lsh, Rth, Rsh
    cos_to_pelvis = (fwd_xy[:, 1:, :] * pelvis_xy.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)  # (N,5)
    yaw_align = 0.5 * (1.0 + cos_to_pelvis).mean(dim=1)                                       # (N,) → [0..1]

    # (B) согласованность «цепочки» бедро↔голень (каждая нога)
    def _cos(i: int, j: int):
        return (fwd_xy[:, i, :] * fwd_xy[:, j, :]).sum(dim=-1).clamp(-1.0, 1.0)
    chain_align = 0.5 * (1.0 + 0.5 * (_cos(2, 3) + _cos(4, 5)))                               # (N,) → [0..1]

    # (C) «горизонтальность» продольной оси таза и торса (подавляет носок/пятку на уровне корпуса)
    z_pelvis = fwd_w[:, 0, 2].abs()
    z_torso  = fwd_w[:, 1, 2].abs()
    upright  = (1.0 - 0.5 * (z_pelvis + z_torso)).clamp(0.0, 1.0)                              # (N,)

    # --- гейтинг по маске DOF: если DOF активны — компоненту выключаем ---
    m = env.command_manager.get_term(mask_name).command > 0.5  # (N,J) bool

    def any_active(bits: tuple[int, ...]) -> torch.Tensor:
        if len(bits) == 0:
            return torch.zeros(m.shape[0], dtype=torch.bool, device=device)
        return m[:, list(bits)].any(dim=1)

    active_L = any_active(left_dofs)
    active_R = any_active(right_dofs)
    active_T = any_active(torso_bits)

    gateA = ~(active_L | active_R | active_T)  # yaw_align зависит от всех сегментов
    gateB = ~(active_L | active_R)             # цепочки зависят от ног
    gateC = ~active_T                          # upright — про таз/торс

    wA = w_yaw    * gateA.float()
    wB = w_chain  * gateB.float()
    wC = w_upright* gateC.float()

    denom = (wA + wB + wC).clamp_min(1e-6)
    r = (wA * yaw_align + wB * chain_align + wC * upright) / denom
    return r.clamp(0.0, 1.0)
    
def idle_double_support_bonus(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    contact_force_threshold: float = 5.0,
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    bonus: float = 1.0,
    # биты ног (слева + справа)
    leg_bits: tuple[int, ...] = (0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20),
) -> torch.Tensor:
    """
    Бонус за «стоялку»: когда команды ≈ 0 И ноги выключены маской — поощряем двухопорие.
    """
    device = env.device
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # near-zero команды
    cmd   = env.command_manager.get_term(command_name).command  # (N,3)
    near0 = (cmd[:, :2].norm(dim=1) < lin_deadband) & (cmd[:, 2].abs() < ang_deadband)

    # ноги выключены по маске? (mask==1 значит DOF активен → нам нужно, чтобы ни один из leg_bits не был активным)
    dof_mask = (env.command_manager.get_term("dof_mask").command > 0.5)  # (N,J) bool
    if not hasattr(env, "_idle_leg_bits"):
        env._idle_leg_bits = torch.as_tensor(leg_bits, device=device, dtype=torch.long)
    legs_inactive = (dof_mask[:, env._idle_leg_bits].sum(dim=1) <= 0)     # (N,) bool

    # требуем ОДНОВРЕМЕННО: near-zero команды И ноги выключены
    gate = near0 & legs_inactive
    if not gate.any():
        return torch.zeros(env.num_envs, device=device)

    # двухопорие по контактам (устойчиво по истории)
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)  # (N,2)
    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold
    both_down = Lc & Rc

    return bonus * (both_down & gate).float()
    
def prolonged_single_support_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    contact_force_threshold: float = 3.0,
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    use_mask: bool = True,
    mask_name: str = "dof_mask",
    # индексы ножных DOF; проверь, что соответствуют порядку joint_names
    left_dofs:  tuple[int, ...] = (0, 3, 7, 11, 15, 19),
    right_dofs: tuple[int, ...] = (1, 4, 8, 12, 16, 20),
) -> torch.Tensor:
    """
    Штраф (0/1) по правилам:
      A) Если |cmd|≈0 и обе ноги ПО МАСКЕ выключены → ОБЯЗАТЕЛЬНО двухопорие (обе стопы в контакте).
      B) Если |cmd|>0 и обе ноги ПО МАСКЕ выключены → ОБЯЗАТЕЛЬНО ≥1 контакт (нет «полёта») и чередование ног
         (две подряд «посадки» одной и той же ноги — штраф).
      C) Если активны DOF ТОЛЬКО одной ноги (по маске) → Другая (неактивная) нога ОБЯЗАНА быть в контакте с землёй.
         Если активны обе ноги — не штрафуем по этому правилу (даём политике свободу).
    """
    device = env.device
    N = env.num_envs
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # --- команды движения ---
    cmd = env.command_manager.get_term(command_name).command  # (N,3)
    lin_mag = cmd[:, :2].norm(dim=1)
    ang_mag = cmd[:, 2].abs()
    move_gate = (lin_mag > lin_deadband) | (ang_mag > ang_deadband)
    idle_gate = ~move_gate

    # --- активность масок по ногам ---
    if use_mask:
        dof_mask = (env.command_manager.get_term(mask_name).command > 0.5)  # (N,J) bool
        Lidx = torch.as_tensor(left_dofs,  device=device, dtype=torch.long)
        Ridx = torch.as_tensor(right_dofs, device=device, dtype=torch.long)
        active_L = dof_mask[:, Lidx].any(dim=1)  # (N,)
        active_R = dof_mask[:, Ridx].any(dim=1)  # (N,)
    else:
        # если маски не используем — считаем обе ноги «неактивными» для правил A/B
        active_L = torch.zeros(N, dtype=torch.bool, device=device)
        active_R = torch.zeros(N, dtype=torch.bool, device=device)

    legs_inactive = ~(active_L | active_R)   # обе ноги выключены
    legs_one_side = active_L ^ active_R      # активна ровно одна нога

    # --- контакты стоп (устойчиво к шуму: максимум по истории) ---
    fmag = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)  # (N,2)
    Lc = fmag[:, 0] > contact_force_threshold
    Rc = fmag[:, 1] > contact_force_threshold
    both   = Lc & Rc
    anyc   = Lc | Rc
    single = Lc ^ Rc
    flight = ~anyc

    # ---------------------------
    # A) Покой + обе ноги выключены → требуем двухопорие
    pen_idle_two = (idle_gate & legs_inactive & ~both).float()

    # ---------------------------
    # B) Движение + обе ноги выключены → запрет полёта + чередование
    pen_move_flight = (move_gate & legs_inactive & flight).float()

    # Чередование: штраф, если два подряд касания одной и той же ноги
    if not hasattr(env, "_pl_prev_Lc"):
        env._pl_prev_Lc   = torch.zeros(N, dtype=torch.bool, device=device)
        env._pl_prev_Rc   = torch.zeros(N, dtype=torch.bool, device=device)
        env._pl_last_touch= torch.full((N,), -1, dtype=torch.long, device=device)  # -1=нет, 0=L, 1=R

    rise_L = Lc & ~env._pl_prev_Lc
    rise_R = Rc & ~env._pl_prev_Rc
    touch_now = torch.full((N,), -1, dtype=torch.long, device=device)
    only_L = rise_L & ~rise_R
    only_R = rise_R & ~rise_L
    touch_now[only_L] = 0
    touch_now[only_R] = 1

    same_touch = (touch_now == env._pl_last_touch) & (touch_now >= 0)
    pen_move_alt = (move_gate & legs_inactive & same_touch).float()

    upd = touch_now >= 0
    env._pl_last_touch[upd] = touch_now[upd]
    env._pl_prev_Lc = Lc
    env._pl_prev_Rc = Rc

    # ---------------------------
    # C) Активна ровно одна нога → другая (неактивная) ОБЯЗАНА стоять на земле
    # если активна только левая — правая должна быть в контакте; если только правая — левая.
    pen_one_leg_cmd = ((legs_one_side &  active_L & ~active_R & ~Rc) |
                       (legs_one_side & ~active_L &  active_R & ~Lc)).float()

    # --- ресеты на границах эпизодов ---
    resets = env.termination_manager.terminated | env.termination_manager.time_outs
    if resets.any():
        env._pl_prev_Lc[resets]    = False
        env._pl_prev_Rc[resets]    = False
        env._pl_last_touch[resets] = -1

    # --- суммарный штраф ---
    penalty = pen_idle_two + pen_move_flight + pen_move_alt + pen_one_leg_cmd
    return penalty.clamp(0.0, 1.0)
    
# ---------------------------
# REWARD: mask=1 → к ТАРГЕТУ
# ---------------------------
import torch

# ==== helpers ====

def _expand_to_NJ(t: torch.Tensor, N: int, J: int, device, dtype) -> torch.Tensor:
    """Привести вход к (N, J). Допускаются: (J,), (N,), (1,J), (N,1), (N,J)."""
    if t is None:
        raise ValueError("Expected tensor, got None")
    t = torch.as_tensor(t, device=device)
    if t.dtype != torch.bool:
        t = t.to(dtype=dtype)

    if t.dim() == 1:
        if t.numel() == J:
            return t.view(1, J).expand(N, J)
        if t.numel() == N:
            return t.view(N, 1).expand(N, J)
        raise ValueError(f"Cannot broadcast 1D len {t.numel()} to (N={N}, J={J})")

    if t.dim() == 2:
        if t.shape == (N, J):
            return t
        if t.shape == (1, J):
            return t.expand(N, J)
        if t.shape == (N, 1):
            return t.expand(N, J)
        # иногда маску/таргет дают как (J,); было бы странно сюда попасть, но на всякий случай:
        if t.shape[0] == J and t.shape[1] == 1:
            return t.view(1, J).expand(N, J)
        raise ValueError(f"Cannot broadcast 2D {tuple(t.shape)} to (N={N}, J={J})")

    raise ValueError(f"Unsupported dim {t.dim()} for broadcasting to (N, J)")

def _per_dof_axis_weights(
    rng: torch.Tensor, device, dtype,
    axis_weights: torch.Tensor | None, eps: float = 1e-12
) -> torch.Tensor:
    """Вернёт пер-осьевые веса формы (1, J). Поддерживает rng (J,) или (N,J).
       Если axis_weights не задан — вес ∝ (qmax-qmin), нормированный на среднее.
    """
    r = torch.as_tensor(rng, device=device, dtype=dtype)
    if r.dim() == 1:
        rJ = r.view(1, -1)           # (1,J)
    elif r.dim() == 2:
        rJ = r[:1, :]                # (1,J)
    else:
        raise ValueError(f"rng dim {r.dim()} unsupported")

    J = rJ.shape[1]

    if axis_weights is None:
        w = rJ.clone().clamp_min(eps)         # (1,J)
        w = w / w.mean().clamp_min(eps)
        return w
    else:
        w = torch.as_tensor(axis_weights, device=device, dtype=dtype)
        if w.dim() == 1:
            if w.numel() != J:
                raise ValueError(f"axis_weights len {w.numel()} != J={J}")
            w = w.view(1, J)
        elif w.dim() == 2:
            if w.shape[1] != J:
                raise ValueError(f"axis_weights shape {tuple(w.shape)} incompatible with J={J}")
            if w.shape[0] != 1:
                w = w.mean(dim=0, keepdim=True)  # усредним по env, оставим (1,J)
        else:
            raise ValueError(f"axis_weights dim {w.dim()} unsupported")
        w = w.clamp_min(eps)
        w = w / w.mean().clamp_min(eps)
        return w  # (1,J)



# ==== rewards ====

def masked_target_proximity_reward_exp_vel(
    env,
    mask_name: str = "dof_mask",
    std_pos: float = 0.25, std_vel: float = 0.25,   # не используется
    w_pos: float = 1.0,  w_vel: float = 1.0,        # не используется
    gate: float = 0.25,                             # не используется
    init_attr: str = "JOINT_INIT_POS_NORM",         # не используется здесь, оставлен для совместимости
    axis_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Экспоненциальная награда по цели для активных DOF (mask==1):
      r = 1.5 * exp( - w_pos * MAE_pos / std_pos^2 ) - 0.5
    Позиции нормируются в [-1,1], расстояния масштабируются пер-осьовыми весами.
    Если нет активных DOF → 0.
    """
    asset = env.scene["robot"]
    q     = asset.data.joint_pos                     # (N, J)
    device, dtype = q.device, q.dtype
    eps = 1e-12

    N, J = q.shape

    # soft limits → нормированные координаты
    qmin = asset.data.soft_joint_pos_limits[..., 0]  # (J,) или (N,J)
    qmax = asset.data.soft_joint_pos_limits[..., 1]  # (J,) или (N,J)
    qn, mid, rng = _normalize_pose(q, qmin, qmax)    # qn: (N,J), mid/rng: (1,J)

    # target → (N,J) (предполагаем уже в норме; если нужно — можно добавить _maybe_norm_target)
    tgt_raw = env.command_manager.get_term("target_joint_pose").command
    tgt = _expand_to_NJ(tgt_raw, N, J, device, dtype)  # (N,J)

    # mask активных DOF → (N,J)
    msk_raw = env.command_manager.get_term(mask_name).command
    msk = _expand_to_NJ(msk_raw, N, J, device, torch.bool) > 0.5  # (N, J) bool

    # веса по осям → (1,J)
    axis_w = _per_dof_axis_weights(rng, device, dtype, axis_weights, eps)  # (1,J)

    # проверки форм (чтоб падало с понятным сообщением, если снова что-то не так)
    assert qn.shape == (N, J), f"qn shape {qn.shape} != {(N,J)}"
    assert tgt.shape == (N, J), f"tgt shape {tgt.shape} != {(N,J)}"
    assert msk.shape == (N, J), f"mask shape {msk.shape} != {(N,J)}"
    assert axis_w.shape == (1, J), f"axis_w shape {axis_w.shape} != {(1,J)}"

    W = msk.float()                                         # (N, J)
    denom = W.sum(dim=-1, keepdim=True).clamp_min(1.0)      # (N, 1)

    dist = (qn - tgt).abs() * axis_w                        # (N, J)
    pos_mae = (dist * W).sum(dim=-1, keepdim=True) / denom  # (N, 1)

    r = torch.exp(- (w_pos * pos_mae) / (std_pos**2 + eps)) 
    r = r.squeeze(-1)                                       # (N,)

    has_active = (W.sum(dim=-1) > 0)                        # (N,)
    return torch.where(has_active, r, torch.zeros_like(r))


def unmasked_init_proximity_reward_exp_vel(
    env,
    mask_name: str = "dof_mask",
    exclude_legs_when_moving: bool = True,
    std_pos: float = 0.25, std_vel: float = 0.25,   # не используется
    w_pos: float = 1.0,  w_vel: float = 1.0,        # не используется
    gate: float = 0.25,                             # не используется
    init_attr: str = "JOINT_INIT_POS_NORM",
    command_name: str = "base_velocity",
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    leg_bits = (0,1,3,4,7,8,11,12,15,16,19,20),
    axis_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Экспоненциальная награда по близости к INIT для неактивных DOF (mask==0):
      r = 1.5 * exp( - w_pos * MAE_pos_to_init / std_pos^2 ) - 0.5
    • ноги (leg_bits) исключаются из оценки при наличии команды на движение;
    • расстояния по DOF масштабируются весами axis_w;
    • если подходящих DOF нет → 0.
    """
    asset = env.scene["robot"]
    q     = asset.data.joint_pos
    device, dtype = q.device, q.dtype
    eps = 1e-12

    N, J = q.shape

    # нормированная поза
    qmin = asset.data.soft_joint_pos_limits[..., 0].to(device=device, dtype=dtype)  # (J,)
    qmax = asset.data.soft_joint_pos_limits[..., 1].to(device=device, dtype=dtype)  # (J,)
    mid  = 0.5 * (qmin + qmax)
    rng  = (qmax - qmin).clamp_min(1e-6)                                            # (J,)
    qn   = 2.0 * (q - mid) / rng                                                    # (N, J)

    # инверт-маска (оцениваем mask==0)
    msk_raw = env.command_manager.get_term(mask_name).command
    msk = _expand_to_NJ(msk_raw, N, J, device, torch.bool) > 0.5                    # (N, J)
    inv = ~msk                                                                      # (N, J)

    # init-поза в норме (кэшируем при первом вызове)
    init_qn = getattr(env, init_attr, None)
    if init_qn is None:
        init_q = getattr(asset.data, "default_joint_pos", None)
        if init_q is None:
            init_qn = torch.zeros(J, device=device, dtype=dtype)
        else:
            init_q   = torch.as_tensor(init_q, device=device, dtype=dtype)          # (J,)
            init_qn  = 2.0 * (init_q - mid) / rng                                   # (J,)
        setattr(env, init_attr, init_qn.detach().clone())                           # (J,)
    else:
        init_qn = torch.as_tensor(init_qn, device=device, dtype=dtype)              # (J,)
    init_qn = init_qn.view(1, J).expand(N, J)                                       # (N, J)

    # исключаем ноги при движении (если нужно)
    if exclude_legs_when_moving and leg_bits:
        try:
            cmd = env.command_manager.get_term(command_name).command
            cmd = torch.as_tensor(cmd, dtype=dtype, device=device)                  # (N, >=3) или (>=3,)
            cmd = _expand_to_NJ(cmd, N, max(cmd.shape[-1], 3), device, dtype)       # (N, M>=3)
            lin_mag = torch.sqrt((cmd[:, :2]**2).sum(dim=-1)) if cmd.shape[1] >= 2 else torch.zeros(N, device=device, dtype=dtype)  # (N,)
            ang_mag = cmd[:, 2].abs() if cmd.shape[1] >= 3 else torch.zeros(N, device=device, dtype=dtype)                           # (N,)
            moving = (lin_mag > lin_deadband) | (ang_mag > ang_deadband)            # (N,)
        except Exception:
            moving = torch.zeros(N, dtype=torch.bool, device=device)

        if moving.any():
            leg_mask_1d = torch.zeros(J, dtype=torch.bool, device=device)
            leg_mask_1d[list(leg_bits)] = True
            inv = inv & ~(moving.view(N, 1) & leg_mask_1d.view(1, J))               # (N, J)

    W = inv.float()                                                                  # (N, J)
    denom = W.sum(dim=-1, keepdim=True).clamp_min(1.0)                              # (N, 1)

    # пер-осьовые веса
    axis_w = _per_dof_axis_weights(rng, device, dtype, axis_weights, eps)           # (1, J)

    # отклонение к INIT
    dist = (qn - init_qn).abs() * axis_w                                            # (N, J)
    pos_mae = (dist * W).sum(dim=-1, keepdim=True) / denom                           # (N, 1)

    r = torch.exp(- (w_pos * pos_mae) / (std_pos**2 + eps))             # (N, 1)
    r = r.squeeze(-1)                                                                # (N,)

    has_eval = (W.sum(dim=-1) > 0)
    return torch.where(has_eval, r, torch.zeros_like(r))


# --------------------------------------------
# PENALTY: mask=1 → близко к ТАРГЕТУ (зеркало)
# --------------------------------------------
def masked_near_target_penalty_linear(
    env,
    mask_name: str = "dof_mask",
) -> torch.Tensor:
    """
    ЛИНЕЙНЫЙ пенальти (mask=1):
      p = mean(|q_norm - target|)/2  ∈ [0, 1]
    Зеркален masked_target_proximity_reward_exp_vel.
    """
    asset = env.scene["robot"]
    q = asset.data.joint_pos
    device, dtype = q.device, q.dtype

    # q ∈ [-1,1]
    qmin = asset.data.soft_joint_pos_limits[..., 0].to(device=device, dtype=dtype)
    qmax = asset.data.soft_joint_pos_limits[..., 1].to(device=device, dtype=dtype)
    qn   = 2.0 * (q - 0.5 * (qmin + qmax)) / (qmax - qmin + 1e-6) # (N,J)

    mask = (env.command_manager.get_term(mask_name).command > 0.5).to(device=device)
    tgt  = torch.as_tensor(env.command_manager.get_term("target_joint_pose").command,
                           dtype=dtype, device=device)

    err = (qn - tgt).pow(2)
    W   = mask.float()
    cnt = W.sum(dim=1).clamp_min(1.0)
    mse = (err * W).sum(dim=1) / cnt
    p   = (mse * 0.25).clamp(0.0, 1.0)

    return torch.where(W.sum(dim=1) > 0, p, torch.zeros_like(p))


# --------------------------------------------------------
# PENALTY: mask=0 → близко к ИНИТУ (зеркало + фильтр ног)
# --------------------------------------------------------
def unmasked_near_init_penalty_linear(
    env,
    mask_name: str = "dof_mask",
    exclude_legs_when_moving: bool = True,
    command_name: str = "base_velocity",
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    leg_bits = (0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20),
    init_attr: str = "JOINT_INIT_POS_NORM",
) -> torch.Tensor:
    """
    ЛИНЕЙНЫЙ пенальти (mask=0):
      p = mean(|q_norm - init_norm|)/2  ∈ [0, 1]
    Зеркален unmasked_init_proximity_reward_exp_vel (с той же фильтрацией ног).
    """
    asset = env.scene["robot"]
    q = asset.data.joint_pos
    device, dtype = q.device, q.dtype

    # q ∈ [-1,1]
    qmin = asset.data.soft_joint_pos_limits[..., 0].to(device=device, dtype=dtype)
    qmax = asset.data.soft_joint_pos_limits[..., 1].to(device=device, dtype=dtype)
    mid  = 0.5 * (qmin + qmax)
    rng  = (qmax - qmin + 1e-6)
    qn   = 2.0 * (q - mid) / rng                                   # (N,J)

    # inv mask
    mask = (env.command_manager.get_term(mask_name).command > 0.5).to(device=device)
    inv  = ~mask

    # init_norm
    init_qn = getattr(env, init_attr, None)
    if init_qn is not None:
        init_qn = torch.as_tensor(init_qn, dtype=dtype, device=device)
        if init_qn.dim() == 1:
            init_qn = init_qn.unsqueeze(0).expand(qn.shape[0], -1) # (N,J)
        else:
            init_qn = init_qn.expand(qn.shape[0], -1)
    else:
        init_q = getattr(asset.data, "default_joint_pos", None)
        if init_q is not None:
            init_q  = torch.as_tensor(init_q, dtype=dtype, device=device).unsqueeze(0)
            init_qn = 2.0 * (init_q - mid) / rng                   # (N,J)
        else:
            init_qn = torch.zeros_like(qn)
        setattr(env, init_attr, init_qn[0].detach().clone())

    # исключаем ноги при движении (если нужно)
    if exclude_legs_when_moving:
        try:
            cmd = env.command_manager.get_term(command_name).command
            cmd = torch.as_tensor(cmd, dtype=dtype, device=device)
            lin_mag = (cmd[:, :2].pow(2).sum(dim=1)).sqrt() if cmd.shape[1] >= 2 else torch.zeros(q.shape[0], device=device)
            ang_mag = cmd[:, 2].abs() if cmd.shape[1] >= 3 else torch.zeros_like(lin_mag)
            moving = (lin_mag > lin_deadband) | (ang_mag > ang_deadband)
        except Exception:
            moving = torch.zeros(q.shape[0], dtype=torch.bool, device=device)

        if leg_bits and moving.any():
            leg_mask_1d = torch.zeros(q.shape[1], dtype=torch.bool, device=device)
            leg_mask_1d[list(leg_bits)] = True
            inv = inv & ~(moving.unsqueeze(1) & leg_mask_1d.unsqueeze(0))

    err = (qn - init_qn).pow(2)
    W   = inv.float()
    cnt = W.sum(dim=1).clamp_min(1.0)
    mse = (err * W).sum(dim=1) / cnt
    p   = (mse * 0.25).clamp(0.0, 1.0)

    return torch.where(W.sum(dim=1) > 0, p, torch.zeros_like(p))
    


import torch

# --- helpers ---
def _normalize_pose(q: torch.Tensor, qmin: torch.Tensor, qmax: torch.Tensor):
    """Нормирует позы в [-1, 1] по мягким лимитам.
       q: (N,J). qmin/qmax: (J,) или (N,J) → приводим к (1,J).
       Возвращает: qn (N,J), mid (1,J), rng (1,J).
    """
    device, dtype = q.device, q.dtype
    N, J = q.shape

    qmin = torch.as_tensor(qmin, device=device, dtype=dtype)
    qmax = torch.as_tensor(qmax, device=device, dtype=dtype)

    # привести к (1,J)
    if qmin.dim() == 1:
        assert qmin.numel() == J, f"qmin len {qmin.numel()} != J={J}"
        qmin = qmin.view(1, J)
    elif qmin.dim() == 2:
        assert qmin.shape[1] == J, f"qmin shape {tuple(qmin.shape)} != (*, {J})"
        qmin = qmin[:1, :]  # берём первую строку (лимиты одинаковые по env)
    else:
        raise ValueError(f"qmin dim {qmin.dim()} unsupported")

    if qmax.dim() == 1:
        assert qmax.numel() == J, f"qmax len {qmax.numel()} != J={J}"
        qmax = qmax.view(1, J)
    elif qmax.dim() == 2:
        assert qmax.shape[1] == J, f"qmax shape {tuple(qmax.shape)} != (*, {J})"
        qmax = qmax[:1, :]
    else:
        raise ValueError(f"qmax dim {qmax.dim()} unsupported")

    mid = 0.5 * (qmin + qmax)                  # (1,J)
    rng = (qmax - qmin).clamp_min(1e-6)        # (1,J)
    qn  = 2.0 * (q - mid) / rng                # (N,J) - (1,J) / (1,J)

    return qn, mid, rng


def _normalize_vel(qdot: torch.Tensor, vel_limits, fallback: float = 1.0):
    """
    Нормирует скорости qdot в условный [-1,1].
    Поддерживаемые формы vel_limits:
      - None → fallback (скаляр или (J,))
      - (J,) или (1,J) → симметричные пределы по DOF
      - (N,) или (N,1) → симметричные пределы по env
      - (N,J) → симметричные пределы по env и DOF
      - (J,2) или (1,J,2) → [min,max] по DOF
      - (N,J,2) → [min,max] по env и DOF
    Возвращает:
      qdn : (N,J) — нормированные скорости
      vmax: (N,J) — использованные пределы (всегда >= 1e-6)
    """
    device, dtype = qdot.device, qdot.dtype
    N, J = qdot.shape

    def _expand_to_NJ_from_1J(t):  # t: (1,J) -> (N,J)
        return t.expand(N, J)

    if vel_limits is None:
        # fallback: скаляр или (J,)
        vmax = torch.as_tensor(fallback, device=device, dtype=dtype)
        if vmax.dim() == 0:
            vmax = vmax.expand(J).view(1, J)
        elif vmax.dim() == 1 and vmax.numel() == J:
            vmax = vmax.view(1, J)
        else:
            raise ValueError(f"fallback vel_limits must be scalar or (J,), got shape {tuple(vmax.shape)}")
        vmax = _expand_to_NJ_from_1J(vmax)

    else:
        v = torch.as_tensor(vel_limits, device=device, dtype=dtype)
        if v.dim() == 1:
            # (J,) или (N,)
            if v.numel() == J:
                vmax = v.abs().view(1, J)
                vmax = _expand_to_NJ_from_1J(vmax)
            elif v.numel() == N:
                vmax = v.abs().view(N, 1).expand(N, J)
            else:
                raise ValueError(f"vel_limits len {v.numel()} incompatible with N={N}, J={J}")

        elif v.dim() == 2:
            # (N,J) или (1,J) или (N,1) или (J,2)
            if v.shape == (N, J):
                vmax = v.abs()
            elif v.shape == (1, J):
                vmax = _expand_to_NJ_from_1J(v)
            elif v.shape == (N, 1):
                vmax = v.abs().expand(N, J)
            elif v.shape == (J, 2):
                vmaxJ = torch.max(v[:, 0].abs(), v[:, 1].abs()).view(1, J)  # (1,J)
                vmax = _expand_to_NJ_from_1J(vmaxJ)                          # (N,J)
            else:
                raise ValueError(f"Unsupported vel_limits shape {tuple(v.shape)}")

        elif v.dim() == 3:
            # (N,J,2) или (1,J,2)
            if v.shape[-1] != 2 or v.shape[1] != J:
                raise ValueError(f"Unsupported vel_limits shape {tuple(v.shape)}")
            vmaxNJ = torch.max(v[..., 0].abs(), v[..., 1].abs())  # (..., J)
            if v.shape[0] == 1:
                vmax = _expand_to_NJ_from_1J(vmaxNJ.view(1, J))
            elif v.shape[0] == N:
                vmax = vmaxNJ.view(N, J)
            else:
                raise ValueError(f"vel_limits first dim {v.shape[0]} incompatible with N={N}")
        else:
            raise ValueError(f"Unsupported vel_limits dim {v.dim()}")

    vmax = vmax.clamp_min(1e-6)        # защита от деления на ноль
    qdn = qdot / vmax                  # (N,J) / (N,J)
    return qdn, vmax



def _maybe_norm_target(tgt: torch.Tensor, mid: torch.Tensor, rng: torch.Tensor):
    """Если target уже в [-1,1] — возвращаем как есть; если в абсолютных — нормируем как позу."""
    device, dtype = tgt.device, tgt.dtype
    t = tgt
    # эвристика: если значения выходят далеко за [-1.5, 1.5], считаем абсолютными
    if t.abs().max() > 1.5:
        t = 2.0 * (t - mid.view(1, -1)) / rng.view(1, -1)
    return t


def masked_target_proximity_reward_exp_pos_and_zero_vel(
    env,
    mask_name: str = "dof_mask",
    std_pos: float = 0.25, std_vel: float = 0.25,
    w_pos: float = 1.0,  w_vel: float = 1.0,
    gate: float = 0.25,
    init_attr: str = "JOINT_INIT_POS_NORM",
    d_ref_norm: float = 1.0,
    vmax_norm: float = 1.0,
    deadband_norm: float = 0.01,
    vel_fallback: float = 1.0,
) -> torch.Tensor:
    """Экспоненциальная награда по MSE |v|-v_des относительно TARGET.
    Движение к цели → положит., от цели → отрицат. (по маске mask==1).
    """
    asset = env.scene["robot"]
    q    = asset.data.joint_pos
    qdot = asset.data.joint_vel
    device, dtype = q.device, q.dtype
    eps = 1e-12

    # --- позы -> [-1,1]
    qmin = asset.data.soft_joint_pos_limits[..., 0].to(device=device, dtype=dtype)
    qmax = asset.data.soft_joint_pos_limits[..., 1].to(device=device, dtype=dtype)
    qn, mid, rng = _normalize_pose(q, qmin, qmax)

    # --- скорости -> нормированные по лимитам
    vel_limits = getattr(asset.data, "joint_vel_limits", None)
    if vel_limits is None:
        vel_limits = getattr(asset.data, "joint_velocity_limits", None)
    qdn, _ = _normalize_vel(qdot, vel_limits, fallback=vel_fallback)

    # --- target и маска
    tgt = torch.as_tensor(env.command_manager.get_term("target_joint_pose").command, dtype=dtype, device=device)
    tgt = _maybe_norm_target(tgt, mid, rng)
    msk = (env.command_manager.get_term(mask_name).command > 0.5).to(device=device, dtype=torch.bool)

    # --- профиль желаемой скорости (модуль)
    dist = (qn - tgt).abs()
    x = dist / (d_ref_norm + eps)
    beta = 0.75
    v_des = vmax_norm * torch.pow(x / (1.0 + x), beta)
    if deadband_norm > 0.0:
        v_des = torch.where(dist < deadband_norm, torch.zeros_like(v_des), v_des)

    # --- проекция скоростей на направление к цели
    dirn = torch.sign(tgt - qn)          # +1 если qn<tgt, -1 если qn>tgt
    proj = qdn * dirn                    # >0 — движемся к цели; <0 — от цели

    towards = proj >= 0.0
    away    = ~towards

    # --- веса по группам
    W_all = msk
    W_t   = (W_all & towards).float()
    W_a   = (W_all & away).float()

    # счётчики активных DOF
    cnt_all = W_all.sum(dim=1)
    has_any = (cnt_all > 0)
    cnt_all = cnt_all.clamp_min(1.0)

    # --- MSE по модулю (|proj| ~ v_des)
    err = proj.abs() - v_des
    se  = err * err

    denom_t = W_t.sum(dim=1).clamp_min(1.0)
    denom_a = W_a.sum(dim=1).clamp_min(1.0)

    mse_t = (se * W_t).sum(dim=1) / denom_t      # к цели
    mse_a = (se * W_a).sum(dim=1) / denom_a      # от цели

    # --- экспоненциальные «оценки соответствия»
    score_t = torch.exp(- (w_vel * mse_t) / (std_vel * std_vel + eps))   # [0..1]
    score_a = torch.exp(- (w_vel * mse_a) / (std_vel * std_vel + eps))   # [0..1]

    # нормировочные веса по долям DOF в группах
    wt = W_t.sum(dim=1) / cnt_all
    wa = W_a.sum(dim=1) / cnt_all

    # итог: положит. вклад «к цели» минус отрицат. вклад «от цели»
    r = wt * score_t - wa * score_a
    return torch.where(has_any, r, torch.zeros_like(r))



def unmasked_init_proximity_reward_exp_pos_and_zero_vel(
    env,
    mask_name: str = "dof_mask",
    exclude_legs_when_moving: bool = True,
    std_pos: float = 0.25, std_vel: float = 0.25,
    w_pos: float = 1.0,  w_vel: float = 1.0,
    gate: float = 0.25,
    init_attr: str = "JOINT_INIT_POS_NORM",
    command_name: str = "base_velocity",
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    leg_bits = (0,1,3,4,7,8,11,12,15,16,19,20),
    d_ref_norm: float = 1.0,
    vmax_norm: float = 1.0,
    deadband_norm: float = 0.01,
    vel_fallback: float = 1.0,
) -> torch.Tensor:
    """Экспоненциальная награда по MSE |v|-v_des относительно INIT (для mask==0).
    К init → положит., от init → отрицат. Ноги можно исключать при движении.
    """
    asset = env.scene["robot"]
    q    = asset.data.joint_pos
    qdot = asset.data.joint_vel
    device, dtype = q.device, q.dtype
    eps = 1e-12

    # --- позы/скорости -> нормированные
    qmin = asset.data.soft_joint_pos_limits[..., 0].to(device=device, dtype=dtype)
    qmax = asset.data.soft_joint_pos_limits[..., 1].to(device=device, dtype=dtype)
    qn, mid, rng = _normalize_pose(q, qmin, qmax)

    vel_limits = getattr(asset.data, "joint_vel_limits", None)
    if vel_limits is None:
        vel_limits = getattr(asset.data, "joint_velocity_limits", None)
    qdn, _ = _normalize_vel(qdot, vel_limits, fallback=vel_fallback)

    # --- инверт-маска (mask==0)
    msk = (env.command_manager.get_term(mask_name).command > 0.5).to(device=device, dtype=torch.bool)
    inv = ~msk

    # --- INIT-поза (в норм.)
    init_qn = getattr(env, init_attr, None)
    if init_qn is not None:
        init_qn = torch.as_tensor(init_qn, dtype=dtype, device=device)
        init_qn = init_qn.unsqueeze(0).expand(qn.shape[0], -1) if init_qn.dim() == 1 else init_qn.expand(qn.shape[0], -1)
    else:
        init_q = getattr(asset.data, "default_joint_pos", None)
        if init_q is not None:
            init_q  = torch.as_tensor(init_q, dtype=dtype, device=device).unsqueeze(0)
            init_qn = 2.0 * (init_q - mid) / rng
        else:
            init_qn = torch.zeros_like(qn)
        setattr(env, init_attr, init_qn[0].detach().clone())

    # --- исключить ноги при движении (по командам)
    if exclude_legs_when_moving:
        try:
            cmd = env.command_manager.get_term(command_name).command
            cmd = torch.as_tensor(cmd, dtype=dtype, device=device)
            lin_mag = (cmd[:, :2].pow(2).sum(dim=1)).sqrt() if cmd.shape[1] >= 2 else torch.zeros(q.shape[0], device=device)
            ang_mag = cmd[:, 2].abs() if cmd.shape[1] >= 3 else torch.zeros_like(lin_mag)
            moving = (lin_mag > lin_deadband) | (ang_mag > ang_deadband)
        except Exception:
            moving = torch.zeros(q.shape[0], dtype=torch.bool, device=device)

        if leg_bits and moving.any():
            leg_mask_1d = torch.zeros(q.shape[1], dtype=torch.bool, device=device)
            leg_mask_1d[list(leg_bits)] = True
            inv = inv & ~(moving.unsqueeze(1) & leg_mask_1d.unsqueeze(0))

    # --- профиль v_des к INIT
    dist = (qn - init_qn).abs()
    x = dist / (d_ref_norm + eps)
    beta = 0.75
    v_des = vmax_norm * torch.pow(x / (1.0 + x), beta)
    if deadband_norm > 0.0:
        v_des = torch.where(dist < deadband_norm, torch.zeros_like(v_des), v_des)

    # --- проекция скоростей на направление к INIT
    dirn = torch.sign(init_qn - qn)   # к иниту
    proj = qdn * dirn

    towards = proj >= 0.0
    away    = ~towards

    # --- веса по группам
    W_all = inv
    W_t   = (W_all & towards).float()
    W_a   = (W_all & away).float()

    cnt_all = W_all.sum(dim=1)
    has_any = (cnt_all > 0)
    cnt_all = cnt_all.clamp_min(1.0)

    # --- MSE по модулю (|proj| ~ v_des)
    err = proj.abs() - v_des
    se  = err * err

    denom_t = W_t.sum(dim=1).clamp_min(1.0)
    denom_a = W_a.sum(dim=1).clamp_min(1.0)

    mse_t = (se * W_t).sum(dim=1) / denom_t
    mse_a = (se * W_a).sum(dim=1) / denom_a

    # --- экспоненциальные «оценки соответствия»
    score_t = torch.exp(- (w_vel * mse_t) / (std_vel * std_vel + eps))
    score_a = torch.exp(- (w_vel * mse_a) / (std_vel * std_vel + eps))

    wt = W_t.sum(dim=1) / cnt_all
    wa = W_a.sum(dim=1) / cnt_all

    r = wt * score_t - wa * score_a
    return torch.where(has_any, r, torch.zeros_like(r))


# MSE penalties
def masked_target_speed_mse_penalty(
    env,
    mask_name: str = "dof_mask",
    d_ref_norm: float = 1.0,
    vmax_norm: float = 1.0,
    deadband_norm: float = 0.01,
    vel_fallback: float = 1.0,
) -> torch.Tensor:
    """
    PENALTY (>=0): MSE между |проекцией норм. скоростей суставов на направление к target|
    и эталонным профилем v_des(dist). Оцениваем только mask==1.
    """
    asset = env.scene["robot"]
    q, qdot = asset.data.joint_pos, asset.data.joint_vel
    device, dtype = q.device, q.dtype
    eps = 1e-12

    # нормированные позы/скорости
    qmin = asset.data.soft_joint_pos_limits[..., 0].to(device=device, dtype=dtype)
    qmax = asset.data.soft_joint_pos_limits[..., 1].to(device=device, dtype=dtype)
    qn, mid, rng = _normalize_pose(q, qmin, qmax)

    vel_limits = getattr(asset.data, "joint_vel_limits", None)
    if vel_limits is None:
        vel_limits = getattr(asset.data, "joint_velocity_limits", None)
    qdn, _ = _normalize_vel(qdot, vel_limits, fallback=vel_fallback)

    # target и маска
    tgt = torch.as_tensor(env.command_manager.get_term("target_joint_pose").command, dtype=dtype, device=device)
    tgt = _maybe_norm_target(tgt, mid, rng)
    W = (env.command_manager.get_term(mask_name).command > 0.5).float()

    # желаемая скорость (модуль) по расстоянию
    dist = (qn - tgt).abs()
    x = dist / (d_ref_norm + eps)
    beta = 0.75
    v_des = vmax_norm * torch.pow(x / (1.0 + x), beta)
    if deadband_norm > 0.0:
        v_des = torch.where(dist < deadband_norm, torch.zeros_like(v_des), v_des)

    # модуль компоненты скорости "в сторону таргета" (без гейта towards)
    dirn = torch.sign(tgt - qn)         # куда нужно
    proj_abs = (qdn * dirn).abs()       # текущая скорость вдоль нужного направления (по модулю)

    # MSE по активным DOF
    denom = W.sum(dim=1).clamp_min(1.0)
    err = proj_abs - v_des
    vel_mse = ((err * err) * W).sum(dim=1) / denom
    return vel_mse

def unmasked_init_speed_mse_penalty(
    env,
    mask_name: str = "dof_mask",
    exclude_legs_when_moving: bool = True,
    command_name: str = "base_velocity",
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    leg_bits = (0,1,3,4,7,8,11,12,15,16,19,20),
    init_attr: str = "JOINT_INIT_POS_NORM",
    d_ref_norm: float = 1.0,
    vmax_norm: float = 1.0,
    deadband_norm: float = 0.01,
    vel_fallback: float = 1.0,
) -> torch.Tensor:
    """
    PENALTY (>=0): MSE между |проекцией норм. скоростей суставов на направление к INIT|
    и эталонным профилем v_des(dist). Оцениваем только mask==0 (с опц. исключением ног при движении).
    """
    asset = env.scene["robot"]
    q, qdot = asset.data.joint_pos, asset.data.joint_vel
    device, dtype = q.device, q.dtype
    eps = 1e-12

    # нормированные позы/скорости
    qmin = asset.data.soft_joint_pos_limits[..., 0].to(device=device, dtype=dtype)
    qmax = asset.data.soft_joint_pos_limits[..., 1].to(device=device, dtype=dtype)
    qn, mid, rng = _normalize_pose(q, qmin, qmax)

    vel_limits = getattr(asset.data, "joint_vel_limits", None)
    if vel_limits is None:
        vel_limits = getattr(asset.data, "joint_velocity_limits", None)
    qdn, _ = _normalize_vel(qdot, vel_limits, fallback=vel_fallback)

    # инверт-маска (оцениваем mask==0)
    msk = (env.command_manager.get_term(mask_name).command > 0.5).to(device=device)
    inv = ~msk

    # при движении исключить суставы ног (как раньше)
    if exclude_legs_when_moving:
        try:
            cmd = env.command_manager.get_term(command_name).command
            cmd = torch.as_tensor(cmd, dtype=dtype, device=device)
            lin_mag = (cmd[:, :2].pow(2).sum(dim=1)).sqrt() if cmd.shape[1] >= 2 else torch.zeros(q.shape[0], device=device)
            ang_mag = cmd[:, 2].abs() if cmd.shape[1] >= 3 else torch.zeros_like(lin_mag)
            moving = (lin_mag > lin_deadband) | (ang_mag > ang_deadband)
        except Exception:
            moving = torch.zeros(q.shape[0], dtype=torch.bool, device=device)

        if leg_bits and moving.any():
            leg_mask_1d = torch.zeros(q.shape[1], dtype=torch.bool, device=device)
            leg_mask_1d[list(leg_bits)] = True
            inv = inv & ~(moving.unsqueeze(1) & leg_mask_1d.unsqueeze(0))

    W = inv.float()

    # init-норма
    init_qn = getattr(env, init_attr, None)
    if init_qn is not None:
        init_qn = torch.as_tensor(init_qn, dtype=dtype, device=device)
        init_qn = init_qn.unsqueeze(0).expand(qn.shape[0], -1) if init_qn.dim() == 1 else init_qn.expand(qn.shape[0], -1)
    else:
        init_q = getattr(asset.data, "default_joint_pos", None)
        if init_q is not None:
            init_q  = torch.as_tensor(init_q, dtype=dtype, device=device).unsqueeze(0)
            init_qn = 2.0 * (init_q - mid) / rng
        else:
            init_qn = torch.zeros_like(qn)
        setattr(env, init_attr, init_qn[0].detach().clone())

    # профиль v_des к init
    dist = (qn - init_qn).abs()
    x = dist / (d_ref_norm + eps)
    beta = 0.75
    v_des = vmax_norm * torch.pow(x / (1.0 + x), beta)
    if deadband_norm > 0.0:
        v_des = torch.where(dist < deadband_norm, torch.zeros_like(v_des), v_des)

    # модуль компоненты скорости "к init" (без гейта towards)
    dirn = torch.sign(init_qn - qn)
    proj_abs = (qdn * dirn).abs()

    # MSE по выбранным DOF
    denom = W.sum(dim=1).clamp_min(1.0)
    err = proj_abs - v_des
    vel_mse = ((err * err) * W).sum(dim=1) / denom
    return vel_mse


    
def masked_dwell_bonus(
    env,
    mask_name: str = "dof_mask",
    eps: float = 0.04,          # допуск по позе (в норме [-1,1])
    vel_eps: float = 0.03,      # допуск по скорости (в норме)
    hold_steps: int = 3,        # после скольки подряд «ОК» шагов начать платить бонус
    bonus: float = 0.5,         # базовый бонус, когда выдержали hold_steps
    growth: float = 0.1,        # сколько добавлять за каждый следующий «ОК» шаг
    max_bonus: float = 3.0,     # верхняя планка
) -> torch.Tensor:
    """
    Платит всё больше, чем дольше ВСЕ активные DOF (mask==1) удерживаются в допуске
    по позиции и скорости. Счётчик сбрасывается, как только вышли из окна «ОК».
    """
    robot = env.scene["robot"]
    q, qd = robot.data.joint_pos, robot.data.joint_vel
    device, dtype = q.device, q.dtype

    # нормировка поз/скоростей в [-1,1]
    qmin, qmax = robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
    mid = 0.5 * (qmin + qmax)
    rng = (qmax - qmin).clamp_min(1e-6)
    qn, qdn = 2.0 * (q - mid) / rng, 2.0 * qd / rng

    # цель и маска
    tgt = torch.as_tensor(env.command_manager.get_term("target_joint_pose").command, dtype=dtype, device=device)
    msk = (env.command_manager.get_term(mask_name).command > 0.5)

    # «ОК» одновременно по позе и скорости (по активным DOF)
    ok_pos = (qn - tgt).abs() <= eps
    ok_vel = qdn.abs() <= vel_eps
    ok_all_j = torch.where(msk, ok_pos & ok_vel, torch.ones_like(ok_pos, dtype=torch.bool))
    ok_all = ok_all_j.all(dim=1)  # (N,)

    # держим счётчик по env
    if not hasattr(env, "_dwell_cnt"):
        env._dwell_cnt = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._dwell_cnt = torch.where(ok_all, env._dwell_cnt + 1, torch.zeros_like(env._dwell_cnt))

    # обнуляем счётчики на ресетах
    resets = env.termination_manager.terminated | env.termination_manager.time_outs
    if resets.any():
        env._dwell_cnt[resets] = 0

    # бонус: начинается после hold_steps и растёт на growth каждый шаг, ограничен max_bonus
    extra = (env._dwell_cnt - hold_steps).clamp_min(0).to(q.dtype)
    r = torch.clamp(bonus + growth * extra, max=max_bonus)
    r = torch.where(env._dwell_cnt >= hold_steps, r, torch.zeros_like(r))
    return r


def unmasked_dwell_bonus(
    env,
    mask_name: str = "dof_mask",
    eps: float = 0.04,          # допуск по позе (норма [-1,1])
    vel_eps: float = 0.03,      # допуск по скорости (норма)
    hold_steps: int = 8,        # после скольких подряд «ОК» шагов начать платить
    bonus: float = 0.5,         # базовый бонус при достижении hold_steps
    growth: float = 0.1,        # прибавка за каждый следующий «ОК» шаг
    max_bonus: float = 3.0,     # верхняя планка
    # init & движение
    init_attr: str = "JOINT_INIT_POS_NORM",
    exclude_legs_when_moving: bool = True,
    command_name: str = "base_velocity",
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    leg_bits = (0,1,3,4,7,8,11,12,15,16,19,20),
) -> torch.Tensor:
    """
    Платит всё больше, чем дольше ВСЕ DOF из инверт-маски (mask==0) удерживаются в допуске
    по позиции (относительно init) и скорости (≈0). Сбрасывает счётчик при выходе из окна «ОК».
    """
    robot = env.scene["robot"]
    q, qd = robot.data.joint_pos, robot.data.joint_vel
    device, dtype = q.device, q.dtype

    # нормировка поз/скоростей (как у masked-для совместимости: по soft-лимитам)
    qmin = robot.data.soft_joint_pos_limits[..., 0].to(device=device, dtype=dtype)
    qmax = robot.data.soft_joint_pos_limits[..., 1].to(device=device, dtype=dtype)
    mid  = 0.5 * (qmin + qmax)
    rng  = (qmax - qmin).clamp_min(1e-6)
    qn   = 2.0 * (q - mid) / rng
    qdn  = 2.0 * qd / rng

    # инверт-маска: оцениваем DOF где mask==0
    msk_bool = (env.command_manager.get_term(mask_name).command > 0.5).to(device=device)
    inv = ~msk_bool  # bool (N,J)

    # init-поза (в норме)
    init_qn = getattr(env, init_attr, None)
    if init_qn is not None:
        init_qn = torch.as_tensor(init_qn, dtype=dtype, device=device)
        if init_qn.dim() == 1:
            init_qn = init_qn.unsqueeze(0).expand(qn.shape[0], -1)
        else:
            init_qn = init_qn.expand(qn.shape[0], -1)
    else:
        init_q = getattr(robot.data, "default_joint_pos", None)
        if init_q is not None:
            init_q  = torch.as_tensor(init_q, dtype=dtype, device=device).unsqueeze(0)
            init_qn = 2.0 * (init_q - mid) / rng
        else:
            init_qn = torch.zeros_like(qn)
        setattr(env, init_attr, init_qn[0].detach().clone())

    # опционально исключаем ноги при движении
    if exclude_legs_when_moving:
        try:
            cmd = env.command_manager.get_term(command_name).command
            cmd = torch.as_tensor(cmd, dtype=dtype, device=device)
            lin_mag = (cmd[:, :2].pow(2).sum(dim=1)).sqrt() if cmd.shape[1] >= 2 else torch.zeros(q.shape[0], device=device)
            ang_mag = cmd[:, 2].abs() if cmd.shape[1] >= 3 else torch.zeros_like(lin_mag)
            moving  = (lin_mag > lin_deadband) | (ang_mag > ang_deadband)  # (N,)
        except Exception:
            moving = torch.zeros(q.shape[0], dtype=torch.bool, device=device)

        if leg_bits and moving.any():
            leg_mask_1d = torch.zeros(q.shape[1], dtype=torch.bool, device=device)
            leg_mask_1d[list(leg_bits)] = True
            # при движении не требуем удержания ног
            inv = inv & ~(moving.unsqueeze(1) & leg_mask_1d.unsqueeze(0))

    # «ОК» одновременно по позе (к init) и скорости (≈0) на DOF из inv
    ok_pos = (qn - init_qn).abs() <= eps
    ok_vel = qdn.abs() <= vel_eps
    ok_all_j = torch.where(inv, ok_pos & ok_vel, torch.ones_like(ok_pos, dtype=torch.bool))
    ok_all = ok_all_j.all(dim=1)  # (N,)

    # отдельный счётчик для unmasked-бонуса
    if not hasattr(env, "_dwell_cnt_unmasked"):
        env._dwell_cnt_unmasked = torch.zeros(q.shape[0], dtype=torch.long, device=device)
    env._dwell_cnt_unmasked = torch.where(ok_all, env._dwell_cnt_unmasked + 1,
                                          torch.zeros_like(env._dwell_cnt_unmasked))

    # обнуление на ресетах
    try:
        resets = env.termination_manager.terminated | env.termination_manager.time_outs
        if resets.any():
            env._dwell_cnt_unmasked[resets] = 0
    except Exception:
        pass

    # бонус с нарастанием после hold_steps
    extra = (env._dwell_cnt_unmasked - int(hold_steps)).clamp_min(0).to(q.dtype)
    r = torch.clamp(bonus + growth * extra, max=max_bonus)
    r = torch.where(env._dwell_cnt_unmasked >= int(hold_steps), r, torch.zeros_like(r))
    return r

# --- 2) Штраф за выход из «окна успеха» после того, как уже были в нём ---
def leaving_target_penalty(
    env,
    mask_name: str = "dof_mask",
    eps: float = 0.05,     # окно «успеха» по позе
    vel_eps: float = 0.04, # и по скорости
    cooldown: int = 0,     # опционально: требовать N шагов подряд вне окна, прежде чем штрафовать
) -> torch.Tensor:
    """
    Если был «успех» (все активные DOF в допуске по позе+скорости), а потом вышли из окна —
    отдаём штраф (0/1). Можно добавить cooldown, чтобы не рыпаться из-за одного шага шума.
    """
    robot = env.scene["robot"]
    q, qd = robot.data.joint_pos, robot.data.joint_vel
    device, dtype = q.device, q.dtype

    qmin, qmax = robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
    mid = 0.5 * (qmin + qmax)
    rng = (qmax - qmin).clamp_min(1e-6)
    qn, qdn = 2.0 * (q - mid) / rng, 2.0 * qd / rng

    tgt = torch.as_tensor(env.command_manager.get_term("target_joint_pose").command, dtype=dtype, device=device)
    msk = (env.command_manager.get_term(mask_name).command > 0.5)

    ok_pos = (qn - tgt).abs() <= eps
    ok_vel = qdn.abs() <= vel_eps
    ok_all_j = torch.where(msk, ok_pos & ok_vel, torch.ones_like(ok_pos, dtype=torch.bool))
    ok_now = ok_all_j.all(dim=1)  # (N,)

    # состояние «были в успехе»
    if not hasattr(env, "_was_ok"):
        env._was_ok = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    # выход из окна (был OK → стал НЕ OK)
    left_now = (~ok_now) & env._was_ok

    # cooldown по количеству подряд НЕ OK шагов перед штрафом
    if cooldown > 0:
        if not hasattr(env, "_not_ok_streak"):
            env._not_ok_streak = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        env._not_ok_streak = torch.where(~ok_now, env._not_ok_streak + 1, torch.zeros_like(env._not_ok_streak))
        penalize = left_now & (env._not_ok_streak >= cooldown)
    else:
        penalize = left_now

    # обновить "были ОК"
    env._was_ok = ok_now

    # сбросы
    resets = env.termination_manager.terminated | env.termination_manager.time_outs
    if resets.any():
        env._was_ok[resets] = False
        if cooldown > 0:
            env._not_ok_streak[resets] = 0

    return penalize.float()  # весом сделаем отрицательным


# --- 3) «Сгладить у цели»: штраф за дёрганье действий ТОЛЬКО внутри окна успеха ---
def masked_action_rate_near_target(
    env,
    mask_name: str = "dof_mask",
    gate: float = 0.25,    # включаем только если RMSE позиции по активным DOF <= gate
) -> torch.Tensor:
    """
    L2 на Δaction по активным DOF, НО только когда мы достаточно близко к цели.
    Удерживает и сглаживает вблизи таргета (не «клацать» туда-сюда).
    """
    # ошибка по позе для гейта
    robot = env.scene["robot"]
    q = robot.data.joint_pos
    device, dtype = q.device, q.dtype

    qmin, qmax = robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
    mid = 0.5 * (qmin + qmax)
    rng = (qmax - qmin).clamp_min(1e-6)
    qn = 2.0 * (q - mid) / rng

    tgt = torch.as_tensor(env.command_manager.get_term("target_joint_pose").command, dtype=dtype, device=device)
    msk = (env.command_manager.get_term(mask_name).command > 0.5).float()
    cnt = msk.sum(dim=1).clamp_min(1.0)

    pos_mse = ((qn - tgt).pow(2) * msk).sum(dim=1) / cnt
    pos_rmse = torch.sqrt(pos_mse + 1e-12)
    near = (pos_rmse <= gate).float()  # (N,)

    # action rate по маске
    act, prev = env.action_manager.action, env.action_manager.prev_action
    d = (act - prev).pow(2)
    num = msk.sum(dim=1).clamp_min(1.0)
    masked_l2 = (d * msk).sum(dim=1) / num  # (N,)

    return masked_l2 * near 
    
def track_lin_vel_xy_mse(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Положительный штраф: || v_xy_cmd - v_xy_body ||^2."""
    asset: RigidObject = env.scene[asset_cfg.name]
    v_xy = asset.data.root_lin_vel_b[:, :2]                                # (N,2)
    v_xy_cmd = env.command_manager.get_command(command_name)[:, :2].to(v_xy)
    err = v_xy_cmd - v_xy
    return (err * err).sum(dim=1)                                          # (N,)

def track_ang_vel_z_mse(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Положительный штраф: (wz_cmd - wz_body)^2."""
    asset: RigidObject = env.scene[asset_cfg.name]
    wz = asset.data.root_ang_vel_b[:, 2]                                   # (N,)
    wz_cmd = env.command_manager.get_command(command_name)[:, 2].to(wz)
    err = wz_cmd - wz
    return err * err
    
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
    r =  torch.exp(-lin_vel_error / std**2)  
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
    r = torch.exp(-ang_vel_error / std**2)  
    if DEBUG:
        print(f'Angular CMD {env.command_manager.get_command(command_name)[:, 2]}')
        print(f'Angular reward {r}')
    return r




def masked_target_proximity_reward_exp_vel_abs(
    env,
    mask_name: str = "dof_mask",
    std_pos: float = 0.25, std_vel: float = 0.25,
    w_pos: float = 1.0,  w_vel: float = 1.0,
    gate: float = 0.25,
    init_attr: str = "JOINT_INIT_POS_NORM",
    d_ref_norm: float = 1.0,
    vmax_norm: float = 1.0,
    deadband_norm: float = 0.01,
    vel_fallback: float = 1.0,
    axis_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Экспонента соответствия |v| профилю v_des относительно TARGET (mask==1). Диапазон [0..1]."""
    asset = env.scene["robot"]
    q    = asset.data.joint_pos        # (N,J)
    qdot = asset.data.joint_vel        # (N,J)
    device, dtype = q.device, q.dtype
    eps = 1e-12

    N, J = q.shape

    # позы -> [-1,1]
    qmin = asset.data.soft_joint_pos_limits[..., 0]
    qmax = asset.data.soft_joint_pos_limits[..., 1]
    qn, mid, rng = _normalize_pose(q, qmin, qmax)  # qn: (N,J), rng: (J,)

    # скорости -> нормированные
    vel_limits = getattr(asset.data, "joint_vel_limits", None)
    if vel_limits is None:
        vel_limits = getattr(asset.data, "joint_velocity_limits", None)
    qdn, _ = _normalize_vel(qdot, vel_limits, fallback=vel_fallback)  # (N,J)

    # target и маска -> (N,J)
    tgt_raw = env.command_manager.get_term("target_joint_pose").command
    tgt = _expand_to_NJ(tgt_raw, N, J, device, dtype)
    tgt = _maybe_norm_target(tgt, mid, rng)

    msk_raw = env.command_manager.get_term(mask_name).command
    msk = _expand_to_NJ(msk_raw, N, J, device, torch.bool) > 0.5  # (N,J)
    W = msk.float()                                               # (N,J)

    # пер-осьовые веса
    axis_w = _per_dof_axis_weights(rng, device, dtype, axis_weights, eps)  # (1,J)

    # расстояние и профиль скорости
    dist = (qn - tgt).abs() * axis_w                          # (N,J)
    x = dist / (d_ref_norm + eps)
    beta = 0.75
    v_des = vmax_norm * torch.pow(x / (1.0 + x), beta)        # (N,J)

    if deadband_norm > 0.0:
        deadband_eff = deadband_norm * axis_w                 # (1,J) -> (N,J)
        v_des = torch.where(dist < deadband_eff, torch.zeros_like(v_des), v_des)

    # модуль проекции скорости на направление к цели
    dirn = torch.sign(tgt - qn)                               # (N,J)
    proj_abs = (qdn * dirn).abs()                             # (N,J)

    # MSE по маске
    err = proj_abs - v_des
    se  = err * err                                           # (N,J)

    cnt = W.sum(dim=-1, keepdim=True).clamp_min(1.0)          # (N,1)
    has_any = (cnt.squeeze(-1) > 0)                           # (N,)
    mse = (se * W).sum(dim=-1, keepdim=True) / cnt            # (N,1)

    score = torch.exp(- (w_vel * mse) / (std_vel * std_vel + eps))  # (N,1)
    score = score.squeeze(-1)                                  # (N,)
    return torch.where(has_any, score, torch.zeros_like(score))


def unmasked_init_proximity_reward_exp_vel_abs(
    env,
    mask_name: str = "dof_mask",
    exclude_legs_when_moving: bool = True,
    std_pos: float = 0.25, std_vel: float = 0.25,
    w_pos: float = 1.0,  w_vel: float = 1.0,
    gate: float = 0.25,
    init_attr: str = "JOINT_INIT_POS_NORM",
    command_name: str = "base_velocity",
    lin_deadband: float = 0.03,
    ang_deadband: float = 0.03,
    leg_bits = (0,1,3,4,7,8,11,12,15,16,19,20),
    d_ref_norm: float = 1.0,
    vmax_norm: float = 1.0,
    deadband_norm: float = 0.01,
    vel_fallback: float = 1.0,
    axis_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Экспонента соответствия |v| профилю v_des относительно INIT (mask==0). Диапазон [0..1]."""
    asset = env.scene["robot"]
    q    = asset.data.joint_pos        # (N,J)
    qdot = asset.data.joint_vel        # (N,J)
    device, dtype = q.device, q.dtype
    eps = 1e-12

    N, J = q.shape

    # позы/скорости нормированные
    qmin = asset.data.soft_joint_pos_limits[..., 0]
    qmax = asset.data.soft_joint_pos_limits[..., 1]
    qn, mid, rng = _normalize_pose(q, qmin, qmax)  # (N,J), (J,), (J,)

    vel_limits = getattr(asset.data, "joint_vel_limits", None)
    if vel_limits is None:
        vel_limits = getattr(asset.data, "joint_velocity_limits", None)
    qdn, _ = _normalize_vel(qdot, vel_limits, fallback=vel_fallback)  # (N,J)

    # маска и инверсия -> (N,J)
    msk_raw = env.command_manager.get_term(mask_name).command
    msk = _expand_to_NJ(msk_raw, N, J, device, torch.bool) > 0.5
    inv = ~msk  # (N,J)

    # INIT-поза в норме
    init_qn = getattr(env, init_attr, None)
    if init_qn is None:
        init_q = getattr(asset.data, "default_joint_pos", None)
        if init_q is None:
            init_qn = torch.zeros(J, device=device, dtype=dtype)
        else:
            init_q  = torch.as_tensor(init_q, device=device, dtype=dtype).view(1, J)  # (1,J)
            init_qn = 2.0 * (init_q - mid.view(1, J)) / rng.view(1, J)                 # (1,J)
        setattr(env, init_attr, init_qn.squeeze(0).detach().clone())                   # (J,)
    else:
        init_qn = torch.as_tensor(init_qn, device=device, dtype=dtype).view(1, J)     # (1,J)
    init_qn = init_qn.expand(N, J)                                                    # (N,J)

    # исключать ноги при движении
    if exclude_legs_when_moving and leg_bits:
        try:
            cmd_raw = env.command_manager.get_term(command_name).command
            # cmd может быть (N,M) или (M,)
            cmd = torch.as_tensor(cmd_raw, device=device, dtype=dtype)
            if cmd.dim() == 1:
                cmd = cmd.view(1, -1).expand(N, -1)
            elif cmd.dim() == 2 and cmd.shape[0] != N:
                cmd = cmd.expand(N, cmd.shape[1])
            # линейная и угловая «мощность»
            if cmd.shape[1] >= 2:
                lin_mag = torch.sqrt((cmd[:, :2] ** 2).sum(dim=-1))
            else:
                lin_mag = torch.zeros(N, device=device, dtype=dtype)
            ang_mag = cmd[:, 2].abs() if cmd.shape[1] >= 3 else torch.zeros(N, device=device, dtype=dtype)
            moving = (lin_mag > lin_deadband) | (ang_mag > ang_deadband)             # (N,)
        except Exception:
            moving = torch.zeros(N, dtype=torch.bool, device=device)

        if moving.any():
            leg_mask_1d = torch.zeros(J, dtype=torch.bool, device=device)
            leg_mask_1d[list(leg_bits)] = True
            inv = inv & ~(moving.view(N, 1) & leg_mask_1d.view(1, J))                # (N,J)

    W = inv.float()                                                                   # (N,J)

    # пер-осьовые веса
    axis_w = _per_dof_axis_weights(rng, device, dtype, axis_weights, eps)            # (1,J)

    # профиль к INIT
    dist = (qn - init_qn).abs() * axis_w                                             # (N,J)
    x = dist / (d_ref_norm + eps)
    beta = 0.75
    v_des = vmax_norm * torch.pow(x / (1.0 + x), beta)                               # (N,J)
    if deadband_norm > 0.0:
        deadband_eff = deadband_norm * axis_w
        v_des = torch.where(dist < deadband_eff, torch.zeros_like(v_des), v_des)

    # модуль проекции скорости на направление к INIT
    dirn = torch.sign(init_qn - qn)                                                  # (N,J)
    proj_abs = (qdn * dirn).abs()                                                    # (N,J)

    # MSE по inv-маске
    err = proj_abs - v_des
    se  = err * err                                                                  # (N,J)

    cnt = W.sum(dim=-1, keepdim=True).clamp_min(1.0)                                 # (N,1)
    has_any = (cnt.squeeze(-1) > 0)                                                  # (N,)
    mse = (se * W).sum(dim=-1, keepdim=True) / cnt                                   # (N,1)

    score = torch.exp(- (w_vel * mse) / (std_vel * std_vel + eps))       # (N,1)
    score = score.squeeze(-1)                                                         # (N,)
    return torch.where(has_any, score, torch.zeros_like(score))
    
### COMBINED

    
# --- Комбо: mask=1 → к TARGET, итог = r_pos * r_vel_abs
def masked_target_proximity_reward_product(
    env,
    pos: dict,       # параметры для masked_target_proximity_reward_exp_vel
    vel_abs: dict,   # параметры для masked_target_proximity_reward_exp_vel_abs
) -> torch.Tensor:
    r_pos    = masked_target_proximity_reward_exp_vel(env, **pos)
    r_velabs = masked_target_proximity_reward_exp_vel_abs(env, **vel_abs)
    return r_pos * r_velabs


# --- Комбо: mask=0 → к INIT, итог = r_pos * r_vel_abs
def unmasked_init_proximity_reward_product(
    env,
    pos: dict,       # параметры для unmasked_init_proximity_reward_exp_vel
    vel_abs: dict,   # параметры для unmasked_init_proximity_reward_exp_vel_abs
) -> torch.Tensor:
    r_pos    = unmasked_init_proximity_reward_exp_vel(env, **pos)
    r_velabs = unmasked_init_proximity_reward_exp_vel_abs(env, **vel_abs)
    return r_pos * r_velabs    
    
def track_lin_ang_vel_exp_product(
    env,
    command_name: str = "base_velocity",
    std: float = math.sqrt(0.25),
) -> torch.Tensor:
    """Произведение r_lin * r_ang.
    r_lin = track_lin_vel_xy_exp(...), r_ang = track_ang_vel_z_exp(...).
    Оба возвращают (N,), итог тоже (N,). Диапазон [0..1].
    """
    r_lin = track_lin_vel_xy_exp(env, command_name=command_name, std=std)
    r_ang = track_ang_vel_z_exp(env, command_name=command_name, std=std)
    res = r_lin * r_ang
    #print(f'Tracking reward {res}, linear {r_lin}, angular {r_ang}')
    return res

def gait_product_reward(
    env,
    idle_cfg: dict | None = None,
    coalign_cfg: dict | None = None,
    heading_cfg: dict | None = None,
    altstep_cfg: dict | None = None,
    legsym_cfg: dict | None = None,
    orient_std: float = math.sqrt(0.25),  # при RMS-наклоне ≈ orient_std вклад ~ e^-1
    clamp01: bool = True,                 # подрезать каждый множитель в [0,1]
    eps: float = 0.0,                     # если >0, нижняя граница для множителей (антизалипание в ноль)
) -> torch.Tensor:
    """Общий ревард походки = r_idle * r_coalign * r_heading * r_altstep * r_legsym * r_orient."""
    asset = env.scene["robot"]
    q = asset.data.joint_pos
    device, dtype = q.device, q.dtype
    N = q.shape[0]
    one = torch.ones(N, device=device, dtype=dtype)

    def squash(x: torch.Tensor) -> torch.Tensor:
        if clamp01:
            x = torch.clamp(x, 0.0, 1.0)
        if eps > 0.0:
            x = torch.clamp(x, min=eps)
        return x

    r = one
    if idle_cfg is not None:
        r = r * squash(idle_double_support_bonus(env, **idle_cfg))
    if coalign_cfg is not None:
        r = r * squash(leg_pelvis_torso_coalignment_reward(env, **coalign_cfg))
    if heading_cfg is not None:
        r = r * squash(heading_alignment_reward(env, **heading_cfg))
    if altstep_cfg is not None:
        r = r * squash(alternating_step_reward(env, **altstep_cfg))
    if legsym_cfg is not None:
        r = r * squash(leg_symmetry_idle_reward_norm(env, **legsym_cfg))

    # flat_orientation_l2 -> экспонента (∈[0,1]) для мультипликативной смеси
    if orient_std is not None and orient_std > 0:
        orient_l2 = flat_orientation_l2(env).to(device=device, dtype=dtype)   # (N,)
        r_orient  = torch.exp(- orient_l2 / (orient_std + 1e-12))             # линейный знаменатель: L2/std
        r = r * squash(r_orient)

    return r 
    


# dog env
def progress_towards_target(
    env: ManagerBasedRLEnv,
    scale: float = 1.0,        
    clamp: float = 0.5,        
    stop_radius: float = 0.35, 
    near_bonus: float = 1000.0,   
) -> torch.Tensor:
    """Положительный, если расстояние до цели уменьшилось за шаг.

    r = scale * clamp(prev_dist - curr_dist, [-clamp, +clamp])
    """
    device = env.device

    robot  = env.scene["robot"]
    target = env.scene["target"]

    diff_xy = target.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    dist = torch.linalg.norm(diff_xy, dim=1)

    if not hasattr(env, "_prev_goal_dist") or env._prev_goal_dist.shape[0] != env.scene.num_envs:
        env._prev_goal_dist = dist.clone()
    
    prev = env._prev_goal_dist
    delta = prev - dist  # >0, если приблизились

    near = dist < stop_radius
    if near_bonus != 0.0:
        delta = delta + near.to(delta.dtype) * (near_bonus / max(scale, 1e-6))

    delta = torch.clamp(delta, min=-clamp, max=clamp)

    env._prev_goal_dist = dist

    return scale * delta       
    
def heading_alignment_to_target(
    env: ManagerBasedRLEnv,
    stop_radius: float = 0.35,
) -> Tensor:
    robot  = env.scene["robot"]
    target = env.scene["target"]

    def _yaw_from_quat_wxyz(q):
        # q = (w, x, y, z)
        siny_cosp = 2.0 * (q[:,0]*q[:,3] + q[:,1]*q[:,2])
        cosy_cosp = 1.0 - 2.0 * (q[:,2]**2 + q[:,3]**2)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _wrap_pi(a):  # [-pi, pi]
        return (a + torch.pi) % (2*torch.pi) - torch.pi

    p_r = robot.data.root_pos_w[:, :2]
    p_t = target.data.root_pos_w[:, :2]
    yaw = _yaw_from_quat_wxyz(robot.data.root_quat_w)

    d    = p_t - p_r
    dist = torch.linalg.norm(d, dim=1) + 1e-6
    heading = torch.atan2(d[:,1], d[:,0])
    err = _wrap_pi(heading - yaw)

    align = 0.5 * (torch.cos(err) + 1.0)
    align = torch.where(dist <= stop_radius, torch.ones_like(align), align)
    return align    
    
def trunk_upright_alignment(
    env,
    body_up_axis: str = "z",   # ось корпуса, которая должна смотреть в +Z мира: "x"|"y"|"z"
    power: float = 1.0,        # шейпинг: >1 усиливает штраф за наклон, 1 — линейно по косинусу
) -> Tensor:
    """
    Возвращает награду r ∈ [-1, +1] для КАЖДОГО env:
      +1  — ось корпуса (спина) строго вверх (совпадает с +Z мира)
       0  — ось горизонтальна
      -1  — ось направлена строго вниз (перевёрнут)

    Реализация: r = dot( R * e_body_axis, world_up ), где world_up = (0,0,1).
    Здесь R — вращение из корпуса в мир (root_quat_w), e_body_axis — базисный вектор оси корпуса.
    """
    robot = env.scene["robot"]
    q = robot.data.root_quat_w  # (N,4), порядок WXYZ

    # batch-конвертация кватерниона (WXYZ) в матрицу вращения (N,3,3)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    # R — активная матрица: v_world = R @ v_body
    R = torch.empty(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2*(yy + zz)
    R[:, 0, 1] = 2*(xy - wz)
    R[:, 0, 2] = 2*(xz + wy)
    R[:, 1, 0] = 2*(xy + wz)
    R[:, 1, 1] = 1 - 2*(xx + zz)
    R[:, 1, 2] = 2*(yz - wx)
    R[:, 2, 0] = 2*(xz - wy)
    R[:, 2, 1] = 2*(yz + wx)
    R[:, 2, 2] = 1 - 2*(xx + yy)

    axis_map = {"x": 0, "y": 1, "z": 2}
    idx = axis_map.get(body_up_axis.lower(), 2)

    # Мировой вектор выбранной оси корпуса — это СТОЛБЕЦ матрицы R (если v_world = R @ v_body).
    # Точка скалярного произведения с (0,0,1) равна Z-компоненте этого столбца:
    # dot( R @ e_idx, [0,0,1] ) = (R[:, :, idx])[..., 2]
    upright = R[:, 2, idx]  # cos угла между осью корпуса и +Z мира, диапазон [-1,1]

    if power != 1.0:
        # симметричный шейпинг без смены знака
        upright = 0.5 * (torch.sign(upright) * torch.abs(upright).pow(power) + 1)

    return upright  # (N,), в [-1,1]    
    
def com_over_support_height_reward_fast(
    env: "ManagerBasedEnv",
    sensor_cfg: SceneEntityCfg,                 # ContactSensor: body_ids лап (как в feet_slide)
    asset_cfg : SceneEntityCfg = SceneEntityCfg("robot"),  # те же звенья для позиций
    contact_force_threshold: float = 1.0,       # Н: как в feet_slide; при шуме подними до 5–20
    # высота CoM над опорной плоскостью:
    target_height: float = 0.33,                # м
    height_tolerance: float = 0.12,             # σ м (можно стягивать каррику́лумом к 0.05)
    slope_aware: bool = True,                   # True: плоскость по опорным точкам; False: мировой Z
    weighted: bool = True,                      # веса по силе контакта при центрах/плоскости
    # «нахождение над опорой»:
    inside_margin: float = 0.10,                # м, мягкая полоса на границе полигона
    beta_inside: float = 4.0,                   # крутизна сигмоиды (позже 8.0)
    # веса частей:
    weight_height: float = 1.0,
    weight_inside: float = 1.0,
    # отладка:
    DEBUG: bool = False,
    DEBUG_MAX_ENVS: int = 1,
) -> torch.Tensor:
    """
    Ревард = inside_score * height_score, где:
      inside_score = sigmoid(beta_inside * signed_inside / inside_margin)
      height_score = exp(-0.5 * ((height_dist - target_height)/height_tolerance)^2)

    Контакты: по net_forces_w_history (как в feet_slide) и sensor_cfg.body_ids.
    Опорные точки: реальные контактные позиции сенсора (если есть), иначе (x,y) звена + z грунта/минимальная z.
    """
    EPS = 1e-9
    device = env.device

    # --- сцена / сенсор / робот ---
    robot = env.scene[asset_cfg.name]
    cs    = env.scene.sensors[sensor_cfg.name]

    # ---------- 1) Контакты (как в feet_slide) ----------
    # contacts = max_t ||net_forces_w_history|| > thr
    if hasattr(cs.data, "net_forces_w_history"):
        f_hist = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]   # (N,H,K,3)
        fmag   = f_hist.norm(dim=-1).amax(dim=1)                               # (N,K)
    else:
        # фолбэк без истории — текущее значение (может быть шумнее)
        f_now = cs.data.net_forces_w[:, sensor_cfg.body_ids, :]               # (N,K,3)
        fmag  = f_now.norm(dim=-1)                                            # (N,K)

    contacts    = fmag > contact_force_threshold                               # (N,K) bool
    any_contact = contacts.any(dim=1)                                          # (N,)
    N, K = contacts.shape

    # ---------- 2) Позиции стоп и CoM ----------
    # важно: используем asset_cfg.body_ids для позиций — индексы должны совпасть с sensor_cfg.body_ids
    feet_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]               # (N,K,3)

    if hasattr(robot.data, "com_pos_w"):
        com_w = robot.data.com_pos_w                                           # (N,3)
    else:
        pos_w = robot.data.body_pos_w                                          # (N,B,3)
        masses = None
        for attr in ("body_masses", "link_masses", "masses"):
            if hasattr(robot.data, attr):
                masses = getattr(robot.data, attr)
                break
        if masses is None:
            com_w = pos_w.mean(dim=1)
        else:
            m = masses.to(device).view(1, -1, 1)
            com_w = (pos_w * m).sum(dim=1) / m.sum(dim=1, keepdim=True).clamp_min(EPS)

    up = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3)

    # ---------- 2a) Сформируем опорные точки (контактные позиции, если есть) ----------
    support_pts_w = None  # (N,K,3)

    def _weighted_mean_pos(pos, w):
        # pos: (N,H,K,3) или (N,1,K,3); w: (N,H,K) -> (N,K,3)
        w = w.clamp_min(EPS)
        return (pos * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True)

    # Попробуем взять контактные позиции из сенсора (история предпочтительна)
    pos_candidates = [
        "net_contact_pos_w_history", "contact_pos_w_history",
        "net_contact_positions_w_history", "contact_positions_w_history",
        "net_contact_pos_w", "contact_pos_w",
        "net_contact_positions_w", "contact_positions_w",
    ]
    for name in pos_candidates:
        if hasattr(cs.data, name):
            arr = getattr(cs.data, name)  # ожидаем (N,H,B,3) или (N,B,3)
            # приведём к (N,H,K,3)
            if arr.dim() == 3:
                arr = arr.unsqueeze(1)  # (N,1,B,3)
            arr = arr[:, :, sensor_cfg.body_ids, :]  # отфильтровали K стоп
            # веса — модуль контактной силы по истории, если есть:
            if hasattr(cs.data, "net_contact_forces_w_history"):
                F = cs.data.net_contact_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)  # (N,H,K)
            elif hasattr(cs.data, "contact_forces_w_history"):
                F = cs.data.contact_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)
            else:
                # если нет истории сил — равные веса по времени
                F = torch.ones(arr.shape[:-1], device=device)
            support_pts_w = _weighted_mean_pos(arr, F)  # (N,K,3)
            break

    # если сенсор не даёт контактных позиций — берём xy звена, z из террейна/минимума
    if support_pts_w is None:
        feet_xy = feet_pos_w[..., :2]  # (N,K,2)
        ground_h = None

        # попытка спросить высоту террейна
        try:
            if hasattr(env, "terrain") and hasattr(env.terrain, "get_height"):
                gh = env.terrain.get_height(feet_xy.reshape(-1, 2))
                if gh is not None:
                    ground_h = gh.to(device).view(feet_xy.shape[0], feet_xy.shape[1])  # (N,K)
            elif hasattr(env, "scene") and hasattr(env.scene, "terrain") and hasattr(env.scene.terrain, "get_height"):
                gh = env.scene.terrain.get_height(feet_xy.reshape(-1, 2))
                if gh is not None:
                    ground_h = gh.to(device).view(feet_xy.shape[0], feet_xy.shape[1])  # (N,K)
        except Exception:
            ground_h = None

        if ground_h is None:
            # фолбэк: минимальная z среди контактирующих лап (по энву), иначе мин. по всем стопам
            z_contact = torch.where(
                contacts,                              # (N,K) bool
                feet_pos_w[..., 2],                    # (N,K)
                torch.full_like(feet_pos_w[..., 2], float("inf")),
            )                                          # (N,K)
            z_min = z_contact.min(dim=1, keepdim=True).values          # (N,1)
            z_min_fallback = feet_pos_w[..., 2].min(dim=1, keepdim=True).values  # (N,1)
            # если нет контактов (inf), берём fallback
            z_min = torch.where(torch.isfinite(z_min), z_min, z_min_fallback)    # (N,1)
            z = z_min.expand(-1, feet_xy.shape[1])                               # (N,K)
        else:
            z = ground_h.to(device=device, dtype=feet_xy.dtype)                  # (N,K)

        support_pts_w = torch.cat([feet_xy, z.unsqueeze(-1)], dim=-1)            # (N,K,3)

    # будем строить плоскость именно по support_pts_w (а не по высоте шарниров)
    pts_for_plane = torch.where(contacts.unsqueeze(-1), support_pts_w, support_pts_w)

    # ---------- 3) Плоскость опоры p0, n ----------
    # центр опоры p0 — взвешенное среднее по силам
    w = (fmag * contacts.float()) if weighted else contacts.float()          # (N,K)
    w_sum = w.sum(dim=1, keepdim=True).clamp_min(EPS)
    p0 = (pts_for_plane * w.unsqueeze(-1)).sum(dim=1) / w_sum                # (N,3)

    cnt = contacts.sum(dim=1)                                               # (N,)
    has3 = cnt >= 3
    has2 = cnt == 2

    n = up.expand(N, 3).clone()                                             # дефолт — мировой up
    # ≥3 опоры: батч-SVD
    if slope_aware and has3.any():
        A   = pts_for_plane[has3] - p0[has3].unsqueeze(1)                   # (N3,K,3)
        A_w = A * w[has3].sqrt().unsqueeze(-1)
        try:
            _, _, Vh = torch.linalg.svd(A_w, full_matrices=False)           # (N3,3,3)
            n3 = Vh[:, -1, :]
        except RuntimeError:
            n3 = up.expand(has3.sum().item(), 3)
        n3 = torch.nn.functional.normalize(n3, dim=1)
        flip3 = (n3 * up).sum(dim=1, keepdim=True) < 0
        n3 = torch.where(flip3, -n3, n3)
        n[has3] = n3

    # ==2 опоры: нормаль «вдоль ребра» (устойчиво на склонах)
    if slope_aware and has2.any():
        idx2 = has2.nonzero(as_tuple=False).flatten()
        c2   = contacts[idx2]                                              # (N2,K)
        fp2  = pts_for_plane[idx2]                                         # (N2,K,3)
        sel  = torch.topk(c2.int(), k=2, dim=1).indices
        a3   = torch.gather(fp2, 1, sel.unsqueeze(-1).expand(-1, -1, 3))   # (N2,2,3)
        a = a3[:, 0, :]; b = a3[:, 1, :]
        edge   = b - a
        edge_u = edge / (edge.norm(dim=1, keepdim=True).clamp_min(EPS))
        side   = torch.cross(up.expand_as(edge_u), edge_u, dim=1)
        side   = side / side.norm(dim=1, keepdim=True).clamp_min(EPS)
        n2     = torch.cross(edge_u, side, dim=1)
        n2     = n2 / n2.norm(dim=1, keepdim=True).clamp_min(EPS)
        flip2  = (n2 * up).sum(dim=1, keepdim=True) < 0
        n2     = torch.where(flip2, -n2, n2)
        n[idx2] = n2

    if not slope_aware:
        n = up.expand_as(n)

    # расстояние CoM до плоскости по нормали
    height_dist = ((com_w - p0) * n).sum(dim=1)                             # (N,)

    # ---------- 4) 2D-СК, проекции и signed-inside ----------
    dot_n_up = (n * up).sum(dim=1, keepdim=True)
    ref = torch.where(
        dot_n_up.abs() < 0.99, up.expand_as(n),
        torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 3).expand_as(n)
    )
    u = torch.cross(n, ref, dim=1); u = torch.nn.functional.normalize(u, dim=1)
    v = torch.cross(n, u, dim=1)

    PP = support_pts_w - p0.unsqueeze(1)                                    # (N,K,3)
    alpha  = (PP * n.unsqueeze(1)).sum(dim=-1, keepdim=True)
    P_proj = PP - alpha * n.unsqueeze(1)
    pts_2d_x = (P_proj * u.unsqueeze(1)).sum(dim=-1)
    pts_2d_y = (P_proj * v.unsqueeze(1)).sum(dim=-1)
    pts_2d   = torch.stack([pts_2d_x, pts_2d_y], dim=-1)                    # (N,K,2)

    PC = com_w - p0
    alpha_c = (PC * n).sum(dim=-1, keepdim=True)
    C_proj  = PC - alpha_c * n
    com2d   = torch.stack([(C_proj * u).sum(dim=-1), (C_proj * v).sum(dim=-1)], dim=-1)  # (N,2)

    cnt_clamped = cnt.clamp_min(1).view(N, 1)
    ctr = (pts_2d * contacts.unsqueeze(-1)).sum(dim=1) / cnt_clamped

    ang = torch.atan2(pts_2d[..., 1] - ctr[:, None, 1], pts_2d[..., 0] - ctr[:, None, 0])
    ang_masked = torch.where(contacts, ang, torch.full_like(ang, float("inf")))
    _, order   = torch.sort(ang_masked, dim=1)
    poly       = torch.gather(pts_2d, 1, order.unsqueeze(-1).expand(-1, -1, 2))
    mask_sorted = torch.gather(contacts, 1, order)

    poly_next  = torch.roll(poly, shifts=-1, dims=1)
    e          = poly_next - poly
    edge_valid = (mask_sorted & torch.roll(mask_sorted, -1, dims=1)) & (cnt.view(N,1) >= 3)

    n_in = torch.stack([-e[..., 1], e[..., 0]], dim=-1)
    n_in = n_in / (n_in.norm(dim=-1, keepdim=True).clamp_min(EPS))
    s    = ((com2d.unsqueeze(1) - poly) * n_in).sum(dim=-1)
    s    = torch.where(edge_valid, s, torch.full_like(s, float("inf")))
    signed_inside_poly = s.min(dim=1).values                                 # (N,)

    is_M2 = (cnt == 2); is_M1 = (cnt == 1)
    a2 = poly[:, 0, :]; b2 = poly[:, 1, :]
    ab = b2 - a2; ab2 = (ab * ab).sum(dim=-1, keepdim=True).clamp_min(EPS)
    t  = ((com2d - a2) * ab).sum(dim=-1, keepdim=True) / ab2
    t  = t.clamp(0.0, 1.0)
    proj = a2 + t * ab
    dist_seg = (com2d - proj).norm(dim=-1)
    signed_inside_seg = inside_margin - dist_seg

    a1 = poly[:, 0, :]
    dist_pt = (com2d - a1).norm(dim=-1)
    signed_inside_pt = inside_margin - dist_pt

    neg_margin = torch.full((N,), -inside_margin, device=device)
    signed_inside = torch.where(
        cnt >= 3, signed_inside_poly,
        torch.where(is_M2, signed_inside_seg, torch.where(is_M1, signed_inside_pt, neg_margin))
    )

    inside_score = torch.sigmoid(beta_inside * (signed_inside / (inside_margin + EPS)))
    inside_score = torch.where(any_contact, inside_score, torch.zeros_like(inside_score))

    # ---------- 5) Высота ----------
    height_err   = height_dist - target_height
    height_score = torch.exp(-0.5 * (height_err / (height_tolerance + EPS)) ** 2)

    # ---------- 6) Итог ----------
    reward = weight_inside * inside_score * weight_height * height_score
    reward = torch.where(any_contact, reward, torch.zeros_like(reward))

    # ---------- DEBUG ----------
    if DEBUG:
        torch.set_printoptions(precision=3, sci_mode=False)
        cnt_hist = torch.bincount(cnt.to('cpu'), minlength=(K+1)).tolist()
        print(f"[CoM dbg] any_contact rate = {any_contact.float().mean().item():.2f}")
        print(f"[CoM dbg] contacts histogram (0..{K}): {cnt_hist[:K+1]}")
        for i in range(min(N, DEBUG_MAX_ENVS)):
            print(f"\n[CoM dbg] ENV {i}: cnt={int(cnt[i])}, any={bool(any_contact[i])}")
            print("   |F| per foot     :", fmag[i].tolist())
            print("   contact mask     :", contacts[i].tolist())
            print("   support pts (contact-weighted):", support_pts_w[i, contacts[i]].tolist() if cnt[i] > 0 else "[]")
            print("   support p0       :", p0[i].tolist())
            print("   support n        :", n[i].tolist())
            print(f"   height_dist={height_dist[i].item(): .3f}  target={target_height: .3f}  height_score={height_score[i].item():.3e}")
            print(f"   signed_inside={signed_inside[i].item(): .3f}  inside_margin={inside_margin: .3f}  inside_score={inside_score[i].item():.3e}")
            print(f"   CoM reward={reward[i].item():.3e}")

    return reward
