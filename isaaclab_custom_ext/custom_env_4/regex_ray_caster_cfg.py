# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.utils import configclass

from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from .regex_ray_caster import RegexRayCaster   


@configclass
class RegexRayCasterCfg(SensorBaseCfg):
    """Configuration for the regex-enabled ray-cast sensor.

    The behavior and fields are fully compatible with RayCasterCfg,
    but :attr:`class_type` points to RegexRayCaster.
    """

    @configclass
    class OffsetCfg:
        """The sensor frame offset relative to the parent."""
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    class_type: type = RegexRayCaster

    mesh_prim_paths: list[str] = MISSING
    """List of mesh paths/patterns for raycast.
    Example: ["{ENV_REGEX_NS}/obst_*", "/World/ground"].
    The {ENV_REGEX_NS} placeholder and glob '*' (in names/segments) are supported.
    """

    offset: OffsetCfg = OffsetCfg()

    attach_yaw_only: bool | None = None
    """DEPRECATED: Use :attr:`ray_alignment`."""

    ray_alignment: Literal["base", "yaw", "world"] = "base"
    """Ray projection frame: base|yaw|world."""

    pattern_cfg: PatternBaseCfg = MISSING
    """A pattern that defines local starts and directions of rays."""

    max_distance: float = 1e6
    """Max. raycast distance, m."""

    drift_range: tuple[float, float] = (0.0, 0.0)
    """Sensor position drift range in the world frame (xyz), m."""

    ray_cast_drift_range: dict[str, tuple[float, float]] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)
    }
    """The drift range of the projection result in the local frame (xyz), m."""

    visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_MARKER_CFG.replace(
        prim_path="/Visuals/RayCaster"
    )
    """Visualizer config (used when debug_vis=True)."""
