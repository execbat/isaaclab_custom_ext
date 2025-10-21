import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
import torch

from .rtx_lidar_lazy_hook import obs_rtx_lidar_points
from .observations import depth_avgpool, compressed_image_features

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


# help function for LIDAR obs compression
def lidar_height_channels_all(
    env,
    sensor_cfg: SceneEntityCfg,
    normalize: bool = False,        # нормировка по max_distance -> [0,1]
    clip_to_unit: bool = False,     # клипнуть [0,1] после нормировки
    fill_no_hit: float | None = None, # чем заполнить отсутствие пересечения (None -> max_distance)
    flatten: bool = True,           # вернуть (N, C*A) вместо (N, C, A)
) -> torch.Tensor:
    """
    Возвращает ВСЕ дистанции лидара (без редукции по азимуту):
      - форма (N, C, A) или (N, C*A) если flatten=True.

    Делает:
      • Восстановление стартов лучей как в RayCaster (учёт ray_alignment + offset).
      • Защита от NaN/Inf, отрицательных значений.
      • (Опц.) нормировка по sensor.cfg.max_distance.
    """
    from isaaclab.utils.math import quat_apply, quat_apply_yaw

    sensor = env.scene.sensors[sensor_cfg.name]   # RayCaster
    hits_w  = sensor.data.ray_hits_w              # (N, R, 3)
    pos_w   = sensor.data.pos_w                   # (N, 3)
    quat_w  = sensor.data.quat_w                  # (N, 4)
    max_d   = float(sensor.cfg.max_distance)
    N       = hits_w.shape[0]

    # --- robust: приводим ray_starts к (N, R, 3) ---
    rs = sensor.ray_starts.to(dtype=hits_w.dtype, device=hits_w.device)
    assert rs.shape[-1] == 3, f"ray_starts last dim must be 3, got {rs.shape}"
    R = rs.shape[-2]
    rs = rs.reshape(-1, R, 3)            # (B, R, 3)
    if rs.shape[0] == 1 and N > 1:
        rs = rs.expand(N, R, 3)
    if rs.shape[0] != N:
        reps = (N + rs.shape[0] - 1) // rs.shape[0]
        rs = rs.repeat(reps, 1, 1)[:N, :, :]
    ray_starts_local = rs                 # (N, R, 3)

    # --- мировые старты с учётом ray_alignment ---
    align = getattr(sensor.cfg, "ray_alignment", "base")
    if align == "world":
        ray_starts_w = ray_starts_local.clone()
        if hasattr(sensor, "ray_cast_drift"):
            ray_starts_w[:, :, 0:2] += sensor.ray_cast_drift[:, 0:2].unsqueeze(1)
        ray_starts_w += pos_w.unsqueeze(1)
    elif align == "yaw":
        ray_starts_w = quat_apply_yaw(
            quat_w.repeat_interleave(R, dim=0),
            ray_starts_local.reshape(-1, 3)
        ).reshape(N, R, 3)
        if hasattr(sensor, "ray_cast_drift"):
            ray_starts_w[:, :, 0:2] += quat_apply_yaw(quat_w, sensor.ray_cast_drift)[:, 0:2].unsqueeze(1)
        ray_starts_w += pos_w.unsqueeze(1)
    elif align == "base":
        ray_starts_w = quat_apply(
            quat_w.repeat_interleave(R, dim=0),
            ray_starts_local.reshape(-1, 3)
        ).reshape(N, R, 3)
        if hasattr(sensor, "ray_cast_drift"):
            ray_starts_w[:, :, 0:2] += quat_apply(quat_w, sensor.ray_cast_drift)[:, 0:2].unsqueeze(1)
        ray_starts_w += pos_w.unsqueeze(1)
    else:
        raise RuntimeError(f"[lidar_height_channels_all] Unsupported ray_alignment: {align}")

    # --- расстояния до хитов (N, R) ---
    dists = torch.linalg.norm(hits_w - ray_starts_w, dim=-1)

    # защита от NaN/Inf/отрицательных
    fill = max_d if fill_no_hit is None else float(fill_no_hit)
    dists = torch.nan_to_num(dists, nan=fill, posinf=fill, neginf=fill)
    dists = torch.clamp(dists, min=0.0)

    # --- раскладка в (N, C, A) без редукции ---
    C = int(sensor.cfg.pattern_cfg.channels)
    assert R % C == 0, f"Rays ({R}) must be divisible by channels ({C})."
    A = R // C
    dists = dists.view(N, C, A)  # (N, C, A)

    # --- нормировка (опционально) ---
    if normalize:
        scale = max(max_d, 1e-6)
        dists = dists / scale
        if clip_to_unit:
            dists = torch.clamp(dists, 0.0, 1.0)

    # --- форма для ObsManager ---
    if flatten:
        dists = dists.reshape(N, C * A)  # (N, C*A)
    
    #print(f"LIDAR {dists}, type {type(dists)}")
    return dists.to(torch.float32)
    
    
    
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
              
        # CUSTOM ADDED OBSERVATIONS
        '''
        # front camera Intel RealSense D435i
        cam_rgb_feat = ObsTerm(
            func=compressed_image_features,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "model_name": "theia-tiny-patch16-224-cdiv",  # or cddsv
                #"model_device": env.device,                  
            },
        )
        # preprocessed depth map from the front camera
        cam_depth_vec = ObsTerm(
            func=depth_avgpool,
            params={"sensor_cfg": SceneEntityCfg("front_camera"), "pool": 4},
        )
        '''
        # lidar observations RayCaster
        lidar_scan_full = ObsTerm(
            func=lidar_height_channels_all,
            params={
                "sensor_cfg": SceneEntityCfg("lidar_top"),
                "normalize": True,
                "clip_to_unit": False,
                "fill_no_hit": None,   # -> max_distance
                "flatten": True,       # shape (N, C*A)
            },
        )
        
#       # RTX LIDAR
#        rtx_lidar_points = ObsTerm(
#            func=obs_rtx_lidar_points,
#            params={"debug" : False},            
#        )
        
        
        # imu data
        imu_projected_gravity = ObsTerm(func=mdp.imu_projected_gravity)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel)
        imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc)
        

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


