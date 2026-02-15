import torch.nn.functional as F
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg
import torch
from isaaclab.utils.math import quat_apply, quat_apply_yaw

def depth_avgpool(env, sensor_cfg: SceneEntityCfg, data_type="distance_to_image_plane", pool=4, normalize=True):
    img = mdp.image(env=env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=normalize)  # (B,H,W,1)
    img = img.permute(0,3,1,2)                       # -> (B,1,H,W)
    img = F.avg_pool2d(img, kernel_size=pool, stride=pool)
    img = F.avg_pool2d(img, kernel_size=pool, stride=pool)
    img = F.avg_pool2d(img, kernel_size=pool, stride=pool)
    return img.flatten(1)                             # -> (B, (H/p)*(W/p))
    
    
    
class compressed_image_features(mdp.image_features):
    def _prepare_theia_transformer_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the Theia transformer model for inference (compact outputs by default)."""
        from transformers import AutoModel
        import torch
        import torch.nn.functional as F

        def _load_model() -> torch.nn.Module:
            model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor, *,
                       pool: str = "mean",           # "mean" | "cls" | "adaptive"
                       out_hw: tuple[int, int] | None = None,
                       flatten: bool = True) -> torch.Tensor:
            """
            Args:
                pool: "mean" → (B,192); "cls" → (B,192);
                      "adaptive" → (B,192,h,w) or (B,192*h*w) with flatten=True.
                out_hw: Objective (h,w) for adapted pool by patch map (required for pool="adaptive").
                flatten: whether to flatten (B,192,h,w) into (B,192*h*w) for pool="adaptive".
            """
            # NHWC uint8 -> NCHW float; normalization as for ImageNet
            x = images.to(model_device).permute(0, 3, 1, 2).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            x = (x - mean) / std

            # ViT: getting tokens (B, 1+N, 192), где N = (H/16)*(W/16)
            out = model.backbone.model(pixel_values=x, interpolate_pos_encoding=True)
            tokens = out.last_hidden_state   # (B, 1+N, 192)
            cls = tokens[:, 0]               # (B,192)
            patches = tokens[:, 1:]          # (B, N,192)

            if pool == "cls":
                return cls
            elif pool == "mean":
                return patches.mean(dim=1)    # (B,192)
            elif pool == "adaptive":
                assert out_hw is not None and len(out_hw) == 2, "Specify out_hw=(h,w) for pool='adaptive'."
                B, N, D = patches.shape
                # Let's restore the patch grid: (B,192, Hp, Wp)
                HpWp = int(round((N) ** 0.5))
                # If the entrance is not square, we will calculate Hp, Wp from the original dimensions
                Hp = images.shape[1] // 16
                Wp = images.shape[2] // 16
                if Hp * Wp != N:  # fallback to square approximation
                    Hp, Wp = HpWp, HpWp
                fmap = patches.reshape(B, Hp, Wp, D).permute(0, 3, 1, 2)  # (B,192,Hp,Wp)
                fmap = F.adaptive_avg_pool2d(fmap, out_hw)                 # (B,192,h,w)
                return fmap.flatten(1) if flatten else fmap
            else:
                raise ValueError(f"Unknown pool mode: {pool}")

        return {"model": _load_model, "inference": _inference}
        
        
def regex_lidar_distance_channels_all(
    env,
    sensor_cfg,                    # SceneEntityCfg
    normalize: bool = False,       # normalization by max_distance -> [0,1]
    clip_to_unit: bool = False,    # clip [0,1] after normalization
    fill_no_hit: float | None = None,  # how to fill the missing intersection (None -> max_distance)
    flatten: bool = True,          # return (N, C*A) instead of (N, C, A)
) -> torch.Tensor:
    """
    Returns ALL lidar ray distances without azimuth reduction:
    output: (N, C, A) or (N, C*A) if flatten=True.

    Does:
    • Restores ray origins in the world, taking into account ray_alignment + offset + drift (as in RayCaster).
    • Distances are calculated from the RAY START to the hit point (not from the sensor position) — as expected.
    • Masks NaN/Inf/negative distances.
    • (Optional) normalization by sensor.cfg.max_distance (+ optional clip in [0,1]).
    """
    # --- 1) we get the sensor and data ---
    sensor = env.scene.sensors[sensor_cfg.name]   # RegexRayCaster (or a regular RayCaster)
    hits_w  = sensor.data.ray_hits_w              # (N, R, 3)
    pos_w   = sensor.data.pos_w                   # (N, 3)
    quat_w  = sensor.data.quat_w                  # (N, 4)
    max_d   = float(sensor.cfg.max_distance)
    N       = hits_w.shape[0]

    # --- 2) robust: convert ray_starts to the form (N, R, 3) ---
    rs = sensor.ray_starts.to(dtype=hits_w.dtype, device=hits_w.device)
    R  = rs.shape[-2]
    rs = rs.reshape(-1, R, 3)            # (B, R, 3)
    if rs.shape[0] == 1 and N > 1:
        rs = rs.expand(N, R, 3)
    if rs.shape[0] != N:
        reps = (N + rs.shape[0] - 1) // rs.shape[0]
        rs = rs.repeat(reps, 1, 1)[:N, :, :]
    ray_starts_local = rs                 # (N, R, 3)

    # --- 3) world starts taking into account ray_alignment + drifts (as in RayCaster) ---
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
        raise RuntimeError(f"[regex_lidar_distance_channels_all] Unsupported ray_alignment: {align}")

    # --- 4) distances (N, R) from the start of the beam to the hit ---
    dists = torch.linalg.norm(hits_w - ray_starts_w, dim=-1)

    # protection from NaN/Inf/negative + "no hit" padding
    fill = max_d if fill_no_hit is None else float(fill_no_hit)
    dists = torch.nan_to_num(dists, nan=fill, posinf=fill, neginf=fill)
    dists = torch.clamp(dists, min=0.0)

    # --- 5) layout in (N, C, A) without azimuth reduction ---
    C = int(sensor.cfg.pattern_cfg.channels)
    assert R % C == 0, f"Rays ({R}) must be divisible by channels ({C})."
    A = R // C
    dists = dists.view(N, C, A)  # (N, C, A)

    # --- 6) normalization/clip (optional) ---
    if normalize:
        scale = max(max_d, 1e-6)
        dists = dists / scale
        if clip_to_unit:
            dists = torch.clamp(dists, 0.0, 1.0)

    # --- 7) form for ObsManager ---
    if flatten:
        dists = dists.reshape(N, C * A)  # (N, C*A)
    
    #print(f"Raycaster LIDAR obs {dists}")
    return dists.to(torch.float32)
    
def height_scan(env, sensor_cfg, offset = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    res = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    res = torch.exp(- res)
    #print(f"height scan obs: {res}")
    return res    
