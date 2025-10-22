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

    Поведение и поля полностью совместимы с RayCasterCfg,
    но :attr:`class_type` указывает на RegexRayCaster.
    """

    @configclass
    class OffsetCfg:
        """Смещение кадра сенсора относительно родителя."""
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # самое важное отличие:
    class_type: type = RegexRayCaster

    mesh_prim_paths: list[str] = MISSING
    """Список путей/паттернов мешей для рейкаста.
    Пример: ["{ENV_REGEX_NS}/obst_*", "/World/ground"].
    Поддерживаются плейсхолдер {ENV_REGEX_NS} и glob '*' (в именах/сегментах).
    """

    offset: OffsetCfg = OffsetCfg()

    attach_yaw_only: bool | None = None
    """DEPRECATED: используйте :attr:`ray_alignment`."""

    ray_alignment: Literal["base", "yaw", "world"] = "base"
    """Кадр проекции лучей: base|yaw|world."""

    pattern_cfg: PatternBaseCfg = MISSING
    """Паттерн, задающий локальные старты и направления лучей."""

    max_distance: float = 1e6
    """Макс. дистанция рейкаста, м."""

    drift_range: tuple[float, float] = (0.0, 0.0)
    """Диапазон дрейфа позиции сенсора в мировом кадре (xyz), м."""

    ray_cast_drift_range: dict[str, tuple[float, float]] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)
    }
    """Диапазон дрейфа результата проекции в локальном кадре (xyz), м."""

    visualizer_cfg: VisualizationMarkersCfg = RAY_CASTER_MARKER_CFG.replace(
        prim_path="/Visuals/RayCaster"
    )
    """Конфиг визуализатора (исп-ся при debug_vis=True)."""

