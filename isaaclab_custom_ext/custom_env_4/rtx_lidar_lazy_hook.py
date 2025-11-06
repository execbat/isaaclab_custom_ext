# ==== RTX LiDAR observation (vectorized per-env, with last-value hold) ========
import numpy as np
import torch
import omni

from typing import Optional
from pxr import UsdGeom, Sdf, Gf
import omni.usd
import omni.kit.commands

# -------------------- Observation format --------------------
_RTX_LIDAR_NUM_POINTS   = 64
_RTX_LIDAR_CH           = 3
_RTX_LIDAR_MAX_POINTS   = _RTX_LIDAR_NUM_POINTS * _RTX_LIDAR_CH  # 192
_RTX_LIDAR_CONFIG_NAME  = "Example_Rotary"
_RTX_LIDAR_SCAN_RATE_HZ = 5.0
# Poll less frequently to reduce load on the graph:
_RTX_LIDAR_READ_EVERY   = 3  # read every N env steps

# -------------------- Caches --------------------
_RTX_LIDAR_OBJ          = {}   # sensor_path -> isaacsim.sensors.rtx.LidarRtx
_RTX_LIDAR_RP           = {}   # sensor_path -> rep.RenderProduct
_RTX_LIDAR_ANN          = {}   # sensor_path -> annotator (per sensor)
_RTX_LIDAR_WARM         = {}   # sensor_path -> bool
_RTX_LIDAR_LOGGED_KEYS  = {}   # sensor_path -> bool (kept to avoid repeated key scans)
_RTX_LIDAR_STEPCOUNT    = 0    # global step counter for throttling
_RTX_LIDAR_LAST_ROW     = {}   # sensor_path -> np.ndarray (_RTX_LIDAR_MAX_POINTS,)

# -------------------- Utilities --------------------
def _rtx_device(env):
    return getattr(env, "device", torch.device("cpu"))

def _env_prim_ns(i: int) -> str:
    return f"/World/envs/env_{i}"

def _lidar_sensor_path_for_env(i: int) -> str:
    return f"{_env_prim_ns(i)}/Robot/torso_link/mid360_link/rtx_lidar_sensor"

def _prim_exists(path: str) -> bool:
    try:
        stage = omni.usd.get_context().get_stage()
        return stage.GetPrimAtPath(path).IsValid()
    except Exception:
        return False

def _timeline_is_playing() -> bool:
    tl = omni.timeline.get_timeline_interface()
    return bool(tl.is_playing())

def _set_lidar_scan_rate(sensor_path: str, hz: float) -> None:
    """Set omni:sensor:Core:scanRateBaseHz on the sensor prim."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(sensor_path)
    if not prim.IsValid():
        return
    from pxr import Sdf
    attr = prim.GetAttribute("omni:sensor:Core:scanRateBaseHz")
    if not attr.IsValid():
        attr = prim.CreateAttribute("omni:sensor:Core:scanRateBaseHz", Sdf.ValueTypeNames.Float)
    attr.Set(float(hz))

def _spawn_rtx_lidar_prim(sensor_path: str, parent_path: Optional[str] = None, debug: Optional[bool] = False) -> bool:
    """Create LidarRtx at the given path and keep a Python reference to it."""
    from isaacsim.sensors.rtx import LidarRtx

    stage = omni.usd.get_context().get_stage()
    if parent_path is None:
        parent_path = str(Sdf.Path(sensor_path).GetParentPath())

    prim = stage.GetPrimAtPath(sensor_path)
    if prim.IsValid():
        _set_lidar_scan_rate(sensor_path, _RTX_LIDAR_SCAN_RATE_HZ)
        return True

    if not stage.GetPrimAtPath(parent_path).IsValid():
        return False

    sensor = LidarRtx(
        prim_path=sensor_path,
        translation=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        config_file_name=_RTX_LIDAR_CONFIG_NAME,
        **{"omni:sensor:Core:scanRateBaseHz": float(_RTX_LIDAR_SCAN_RATE_HZ)},
    )
    # Test visualisation
    if debug:
        sensor.enable_visualization()
    
    _RTX_LIDAR_OBJ[sensor_path] = sensor
    _set_lidar_scan_rate(sensor_path, _RTX_LIDAR_SCAN_RATE_HZ)
    return True

def _ensure_lidars_for_all_envs(env):
    """Prepare per-env sensor paths and init flags once."""
    if getattr(env, "_rtx_lidar_initialized", None) is None:
        num_envs = int(getattr(env, "num_envs", 1))
        env._rtx_lidar_initialized = [False] * num_envs
        env._rtx_lidar_prim_paths = [_lidar_sensor_path_for_env(i) for i in range(num_envs)]

# -------- per-sensor RP + annotator (more robust than a shared RP) ----------
def _ensure_rp_and_annotator(sensor_path: str):
    import omni.replicator.core as rep

    if sensor_path not in _RTX_LIDAR_RP:
        rp = rep.create.render_product(sensor_path, resolution=(1, 1))
        _RTX_LIDAR_RP[sensor_path] = rp

    if sensor_path not in _RTX_LIDAR_ANN:
        for name in (
            "IsaacCreateRTXLidarScanBuffer",
            "RtxSensorCpuIsaacCreateRTXLidarScanBuffer",
            "RtxSensorCreateRTXLidarScanBuffer",
        ):
            try:
                ann = rep.AnnotatorRegistry.get_annotator(name)
                ann.attach([_RTX_LIDAR_RP[sensor_path].path])  # attach by path
                _RTX_LIDAR_ANN[sensor_path] = ann
                _RTX_LIDAR_WARM[sensor_path] = False
                _RTX_LIDAR_LOGGED_KEYS[sensor_path] = True  # skip repeated key logging
                break
            except Exception:
                pass
        if sensor_path not in _RTX_LIDAR_ANN:
            raise RuntimeError("[RTX-LIDAR] no suitable annotator found for this build")

    # Optionally attach on the sensor object as well (helps in some 5.0 builds)
    sensor = _RTX_LIDAR_OBJ.get(sensor_path)
    if sensor is not None:
        attached = sensor.get_annotators()  # dict: name -> annotator
        name = "IsaacExtractRTXSensorPointCloudNoAccumulator"
        if name not in attached:
            sensor.attach_annotator(name)

def _read_points_from_rtx(sensor_path: str):
    """Return np.ndarray (N,3) or None if no data this frame."""
    import omni.replicator.core as rep
    _ensure_rp_and_annotator(sensor_path)
    ann = _RTX_LIDAR_ANN[sensor_path]

#    if not _RTX_LIDAR_WARM[sensor_path] and _timeline_is_playing():
#        rep.orchestrator.step()
#        rep.orchestrator.step()
#        _RTX_LIDAR_WARM[sensor_path] = True

    data = ann.get_data()
    if not data and _timeline_is_playing():
        rep.orchestrator.step()
        data = ann.get_data()
        if not data:
            return None
    elif not data:
        return None

    for _, val in data.items():
        arr = np.asarray(val)
        if arr.ndim >= 2 and arr.shape[-1] >= 3 and arr.shape[0] > 0:
            return arr.reshape(-1, arr.shape[-1])[:, :3].astype(np.float32, copy=False)
    return None

# -------------------- Packing to fixed length --------------------
def _to_fixed_len_flat(pc_xyz: np.ndarray) -> np.ndarray:
    """Pad/trim to a fixed number of points and return flat (XYZ...)."""
    out = np.zeros((_RTX_LIDAR_NUM_POINTS, _RTX_LIDAR_CH), dtype=np.float32)
    if pc_xyz is not None and pc_xyz.ndim == 2 and pc_xyz.shape[0] > 0:
        n = min(_RTX_LIDAR_NUM_POINTS, pc_xyz.shape[0])
        out[:n, :] = pc_xyz[:n, :3]
    return out.reshape(-1)

# -------------------- Main observation term --------------------
def obs_rtx_lidar_points(env, term_cfg=None, debug = False):
    """
    Returns torch.float32 tensor of shape (num_envs, _RTX_LIDAR_MAX_POINTS) on env.device.
    Per-env lazy sensor creation. Throttled reads (see _RTX_LIDAR_READ_EVERY).
    If the current frame has no data, re-use the last valid row for that sensor; if none exists, use zeros.
    """
    global _RTX_LIDAR_STEPCOUNT
    from pxr import Sdf

    device   = _rtx_device(env)
    num_envs = int(getattr(env, "num_envs", 1))

    _ensure_lidars_for_all_envs(env)

    batch_np = np.zeros((num_envs, _RTX_LIDAR_MAX_POINTS), dtype=np.float32)

    # Lazy spawn
    for i in range(num_envs):
        sensor_path = env._rtx_lidar_prim_paths[i]
        parent_path = str(Sdf.Path(sensor_path).GetParentPath())
        if not env._rtx_lidar_initialized[i]:
            if not _prim_exists(sensor_path):
                ok = _spawn_rtx_lidar_prim(sensor_path, parent_path, debug)
                env._rtx_lidar_initialized[i] = bool(ok and _prim_exists(sensor_path))
            else:
                env._rtx_lidar_initialized[i] = True

    # If timeline not playing yet: return last/zeros
    if not _timeline_is_playing():
        for i in range(num_envs):
            spath = env._rtx_lidar_prim_paths[i]
            last = _RTX_LIDAR_LAST_ROW.get(spath, None)
            if last is not None:
                batch_np[i, :] = last
        return torch.as_tensor(batch_np, dtype=torch.float32, device=device)

    _RTX_LIDAR_STEPCOUNT += 1
    do_read = (_RTX_LIDAR_STEPCOUNT % _RTX_LIDAR_READ_EVERY == 0)

    for i in range(num_envs):
        spath = env._rtx_lidar_prim_paths[i]
        pc_np = None

        if env._rtx_lidar_initialized[i] and do_read:
            pc_np = _read_points_from_rtx(spath)

        if pc_np is not None:
            row_np = _to_fixed_len_flat(pc_np)
            _RTX_LIDAR_LAST_ROW[spath] = row_np.copy()
        else:
            # Use last valid row or zeros
            row_np = _RTX_LIDAR_LAST_ROW.get(spath, None)
            if row_np is None:
                row_np = np.zeros((_RTX_LIDAR_MAX_POINTS,), dtype=np.float32)

        batch_np[i, :] = row_np
    res = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
    #print(f"res {res}")
    return res

