# ==== RTX LiDAR observation (vectorized per-env, with last-value hold) ========
import numpy as np
import torch
import omni

from typing import Optional
from pxr import UsdGeom, Sdf, Gf
import omni.usd
import omni.kit.commands
from isaacsim.sensors.rtx import get_gmo_data
import omni.replicator.core as rep
from isaacsim.sensors.rtx import LidarRtx
from typing import Literal, Optional, Tuple
import carb
from isaacsim.core.simulation_manager import SimulationManager


# -------------------- Observation format --------------------
_RTX_LIDAR_NUM_POINTS   = 64
_RTX_LIDAR_CH           = 3
_RTX_LIDAR_MAX_POINTS   = _RTX_LIDAR_NUM_POINTS * _RTX_LIDAR_CH  # 192
_RTX_LIDAR_CONFIG_NAME  = "Example_Rotary"# "Example_Rotary"
_RTX_LIDAR_SCAN_RATE_HZ = 1000.0
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

class LidarRtxDist(LidarRtx):
    def __init__(
        self,
        prim_path: str,
        name: str = "lidar_rtx",
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = np.array([1.0, 0.0, 0.0, 0.0]),
        config_file_name: Optional[str] = None,
        **kwargs,
    ) -> None:

        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            config_file_name=config_file_name,
            **kwargs,
        )
        
        self._render_product = rep.create.render_product(prim_path, resolution=(1, 1))
        self._render_product_path = self._render_product.path
        
    def attach_annotator(
        self,
        annotator_name: Literal[
            "IsaacComputeRTXLidarFlatScan",
            "IsaacExtractRTXSensorPointCloudNoAccumulator",
            "IsaacCreateRTXLidarScanBuffer",
        ],
    ) -> None:
        """Attach an annotator to the Lidar sensor.

        Args:
            param annotator_name (Literal): Name of the annotator to attach. Must be one of:
                - "IsaacComputeRTXLidarFlatScan"
                - "IsaacExtractRTXSensorPointCloudNoAccumulator"
                - "IsaacCreateRTXLidarScanBuffer"
        """
        if annotator_name in self._annotators:
            carb.log_warn(f"Annotator {annotator_name} already attached to {self._render_product_path}")
            return

        annotator = rep.AnnotatorRegistry.get_annotator(annotator_name)
        annotator.initialize(outputDistance=True) # ADDED
        annotator.attach([self._render_product_path])
        self._annotators[annotator_name] = annotator
        self._current_frame[annotator_name] = {"distance" : None}
        return        

    def _data_acquisition_callback(self, event: carb.events.IEvent):
        """Handle data acquisition callback for the Lidar sensor.

        Args:
            param event (carb.events.IEvent): The event that triggered the callback.
        """

        
        annotator_name, annotator =  "IsaacExtractRTXSensorPointCloudNoAccumulator", self._annotators["IsaacExtractRTXSensorPointCloudNoAccumulator"]
        data = annotator.get_data()
       
        try:
            #i = data["distance"][2]
            self._current_frame[annotator_name] = data
            print(f'SHAPE {data["data"].shape}')
            print()

        except:
            pass



    def get_current_frame(self) -> dict:
        """Get the current frame data from the Lidar sensor.

        Returns:
            dict: Dictionary containing the current frame data including rendering time,
                frame number, and any attached annotator data.
        """
        return self._current_frame["IsaacExtractRTXSensorPointCloudNoAccumulator"]



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





def _spawn_rtx_lidar_prim(sensor_path: str, parent_path: Optional[str] = None, debug: Optional[bool] = False) -> bool:
    """Create LidarRtx at the given path and keep a Python reference to it."""
    

    stage = omni.usd.get_context().get_stage()
    if parent_path is None:
        parent_path = str(Sdf.Path(sensor_path).GetParentPath())

    prim = stage.GetPrimAtPath(sensor_path)

    if not stage.GetPrimAtPath(parent_path).IsValid():
        return False

    sensor = LidarRtxDist(
        prim_path=sensor_path,
        translation=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        config_file_name=_RTX_LIDAR_CONFIG_NAME,
        #valid_range = (0, 10), # length of detection
        #scan_type = "solidState",
        **{"omni:sensor:Core:scanRateBaseHz": float(_RTX_LIDAR_SCAN_RATE_HZ)},
    )
    sensor.initialize(SimulationManager.get_physics_sim_view())
    
    # Test visualisation
    if debug:
        sensor.enable_visualization()
    
    _RTX_LIDAR_OBJ[sensor_path] = sensor

    annotator_name = "IsaacExtractRTXSensorPointCloudNoAccumulator" # ["IsaacComputeRTXLidarFlatScan", "IsaacExtractRTXSensorPointCloudNoAccumulator", "IsaacCreateRTXLidarScanBuffer"]
    sensor.attach_annotator(annotator_name)
    

    
    
    
    return True

def _ensure_lidars_for_all_envs(env):
    """Prepare per-env sensor paths and init flags once."""
    if getattr(env, "_rtx_lidar_initialized", None) is None:
        num_envs = int(getattr(env, "num_envs", 1))
        env._rtx_lidar_initialized = [False] * num_envs
        env._rtx_lidar_prim_paths = [_lidar_sensor_path_for_env(i) for i in range(num_envs)]




# -------------------- Main observation term --------------------
def obs_rtx_lidar_points(env, term_cfg=None, debug = False):
    """
    Returns torch.float32 tensor of shape (num_envs, _RTX_LIDAR_MAX_POINTS) on env.device.
    Per-env lazy sensor creation. Throttled reads (see _RTX_LIDAR_READ_EVERY).
    If the current frame has no data, re-use the last valid row for that sensor; if none exists, use zeros.
    """
    global _RTX_LIDAR_STEPCOUNT
    global _PREV_DATA
    from pxr import Sdf

    device   = _rtx_device(env)
    num_envs = int(getattr(env, "num_envs", 1))
    #print(f"num_envs {num_envs}")
    _ensure_lidars_for_all_envs(env) # creates all the links env._rtx_lidar_prim_paths

    batch_np = np.zeros((num_envs, _RTX_LIDAR_MAX_POINTS), dtype=np.float32)
    
    if not all(env._rtx_lidar_initialized):
        # Lazy spawn
        for i in range(num_envs):
            print(f"SPAWN")
            sensor_path = env._rtx_lidar_prim_paths[i]
            parent_path = str(Sdf.Path(sensor_path).GetParentPath())
            print(f"sensor_path {sensor_path}")
            print(f"parent_path {parent_path}")           
            ok = _spawn_rtx_lidar_prim(sensor_path, parent_path, debug)
            env._rtx_lidar_initialized[i] = bool(ok and _prim_exists(sensor_path))



    _RTX_LIDAR_STEPCOUNT += 1
    #print(f"env._rtx_lidar_prim_paths : {env._rtx_lidar_prim_paths}")
    #print(f"_RTX_LIDAR_OBJ : {_RTX_LIDAR_OBJ}")

    sensor_list = [ *map(lambda x: _RTX_LIDAR_OBJ[x], env._rtx_lidar_prim_paths)]
    data = [ *map(lambda x: x.get_current_frame(), sensor_list)]

    print(f"data {data}")
    print()
    #print(f"gmo {gmo}")
    print()
    print()



    res = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
    #print(f"res {res}")
    return res 
    

