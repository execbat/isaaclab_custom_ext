# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.usd
import omni.physics.tensors.impl.api as physx
import warp as wp
from pxr import Usd, UsdGeom, UsdPhysics

from isaacsim.core.prims import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.sensors.ray_caster.ray_caster_data import RayCasterData
import fnmatch

if TYPE_CHECKING:
    from .regex_ray_caster_cfg import RegexRayCasterCfg




class RegexRayCaster(SensorBase):
    """Ray-casting сенсор с поддержкой regex и множественных мешей.

    Ключевые отличия от базового RayCaster:
    - `mesh_prim_paths` может содержать несколько путей;
    - каждый путь в `mesh_prim_paths` трактуется как **регулярное выражение** (Python re, fullmatch);
    - все совпавшие примы типа Mesh или Plane конвертируются в warp-меши;
    - рейкаст идёт по всем warp-мешам одновременно, берётся ближайшее попадание.

    Ограничения/поведение сохранены:
    - `cfg.prim_path` (листовой сегмент) не может содержать regex (см. оригинальное предупреждение);
    - поддерживаются статические меши, как в исходном RayCaster.
    """

    cfg: RegexRayCasterCfg

    def __init__(self, cfg: RegexRayCasterCfg):
        # проверка листа пути сенсора (как в оригинале)
        sensor_leaf = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_leaf) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the ray-caster sensor: {cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )
        super().__init__(cfg)

        self._data = RayCasterData()
        # Сюда соберём все warp-меши, по которым будем кастить
        self.meshes: dict[str, wp.Mesh] = {}
        self._retry_mesh_discovery = False 

    def __str__(self) -> str:
        return (
            f"Regex Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    # ----------------
    # Properties
    # ----------------
    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> RayCasterData:
        self._update_outdated_buffers()
        return self._data

    # ----------------
    # Operations
    # ----------------
    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            num_envs_ids = self._view.count
        else:
            num_envs_ids = len(env_ids)
        # дрейфы, как в оригинале
        r = torch.empty(num_envs_ids, 3, device=self.device)
        self.drift[env_ids] = r.uniform_(*self.cfg.drift_range)

        range_list = [self.cfg.ray_cast_drift_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=self.device)
        self.ray_cast_drift[env_ids] = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (num_envs_ids, 3), device=self.device
        )

    # ----------------
    # Implementation
    # ----------------
    def _initialize_impl(self):
        super()._initialize_impl()
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # Создаём view по типу прима (как в исходном RayCaster)
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._view = self._physics_sim_view.create_articulation_view(self.cfg.prim_path.replace(".*", "*"))
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
        else:
            self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
            omni.log.warn(f"The prim at path {prim.GetPath().pathString} is not a physics prim! Using XFormPrim.")

        # Загружаем warp-меши (regex + множественные)
        self._initialize_warp_meshes_regex()
        # Инициализируем лучи по паттерну
        self._initialize_rays_impl()

    # ---- NEW: regex expansion for meshes ----
    def _initialize_warp_meshes_regex(self):
        """Находит все примы по шаблонам (regex/glob) и рекурсивно добавляет все Mesh/Plane под ними."""
        if not self.cfg.mesh_prim_paths:
            raise RuntimeError("No mesh_prim_paths provided for RegexRayCaster.")

        stage = omni.usd.get_context().get_stage()
        seen_paths: set[str] = set()

        def _add_plane(plane_prim):
            mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
            wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
            self.meshes[plane_prim.GetPath().pathString] = wp_mesh
            omni.log.info(f"[RegexRayCaster] Added Plane: {plane_prim.GetPath()}")

        def _add_mesh(mesh_prim):
            """Добавляет любой UsdGeom.Mesh как warp-меш:
            - разворачивает полигоны в треугольники (triangle fan),
            - применяет world-трансформ к вершинам,
            - грузит в warp.
            """
            mesh_geom = UsdGeom.Mesh(mesh_prim)

            # 1) вершины в локале
            points = np.asarray(mesh_geom.GetPointsAttr().Get(), dtype=np.float32)

            # 2) полигоны (counts + indices) -> треугольники
            counts  = np.asarray(mesh_geom.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
            indices = np.asarray(mesh_geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            tris = []
            cursor = 0
            for n in counts:
                if n >= 3:
                    face = indices[cursor:cursor + n]
                    # triangle fan: (v0,v1,v2), (v0,v2,v3), ...
                    for k in range(1, n - 1):
                        tris.append([face[0], face[k], face[k + 1]])
                cursor += n
            if not tris:
                omni.log.warn(f"[RegexRayCaster] Mesh {mesh_geom.GetPath()} has no triangulable faces. Skipping.")
                return
            tris = np.asarray(tris, dtype=np.int32)

            # 3) world-трансформ (ВАЖНО: берём у prim, не у geom)
            #    omni.usd.get_world_transform_matrix возвращает матрицу 4x4.
            xform = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
            points = (points @ xform[:3, :3].T) + xform[:3, 3]

            # 4) в warp
            wp_mesh = convert_to_warp_mesh(points, tris, device=self.device)
            self.meshes[mesh_geom.GetPath().pathString] = wp_mesh

            omni.log.info(
                f"[RegexRayCaster] Added Mesh: {mesh_geom.GetPath()} "
                f"(V={len(points)}, F={len(tris)})"
            )

        def _collect_meshes_under(root_prim):
            """Рекурсивно обойти всех потомков и добавить Mesh/Plane."""
            for prim in Usd.PrimRange(root_prim):
                tname = prim.GetTypeName()
                if tname == "Mesh":
                    if prim.GetPath().pathString not in self.meshes:
                        _add_mesh(prim)
                elif tname == "Plane":
                    if prim.GetPath().pathString not in self.meshes:
                        _add_plane(prim)

        compiled = [self._compile_path_pattern(p) for p in self.cfg.mesh_prim_paths]

        matched_roots = []
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            # ищем "подстрочно", чтобы паттерн вроде {ENV_REGEX_NS}/obst_.*
            # сработал на /World/envs/env_0/obst_01 И на детях глубже
            if any(rx.search(p) for rx in compiled):
                matched_roots.append(prim)

        for root in matched_roots:
            root_path = root.GetPath().pathString
            if root_path in seen_paths:
                continue
            seen_paths.add(root_path)

            tname = root.GetTypeName()
            if tname in ("Mesh", "Plane"):
                # прямое совпадение с геометрией
                if tname == "Mesh":
                    _add_mesh(root)
                else:
                    _add_plane(root)
            else:
                # Xform/Scope и т.п. — рекурсивно собираем геометрию под ним
                _collect_meshes_under(root)
        
        omni.log.info(f"[RegexRayCaster] matched_roots={len(matched_roots)} "
                  f"e.g. {[p.GetPath().pathString for p in matched_roots[:6]]}")
        
        if not self.meshes:
            omni.log.warn(
                "[RegexRayCaster] No meshes found at init. Will retry on first update. "
                f"Patterns: {self.cfg.mesh_prim_paths}"
            )
            self._retry_mesh_discovery = True
        else:
            self._retry_mesh_discovery = False
            omni.log.info(f"[RegexRayCaster] Loaded {len(self.meshes)} meshes: "
                          f"{list(self.meshes.keys())[:12]}")

    def _initialize_rays_impl(self):
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(self.ray_directions)

        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos

        self.ray_starts = self.ray_starts.repeat(self._view.count, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self._view.count, 1, 1)

        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        self.ray_cast_drift = torch.zeros(self._view.count, 3, device=self.device)

        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        if getattr(self, "_retry_mesh_discovery", False) or not self.meshes:
            self._initialize_warp_meshes_regex()
            if not self.meshes:
                # если всё ещё пусто — вернём "пустые" хиты и выйдем без ошибки
                n = self._view.count if isinstance(env_ids, slice) else len(env_ids)
                self._data.ray_hits_w[env_ids] = torch.full(
                    (n, self.num_rays, 3), float("inf"), device=self._device
                )
                return
    
        # Поза сенсора
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")

        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        pos_w += self.drift[env_ids]

        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # Совместимость с устаревшим флагом
        if self.cfg.attach_yaw_only is not None:
            msg = (
                "Raycaster attribute 'attach_yaw_only' will be deprecated. "
                "Use 'ray_alignment' instead."
            )
            if self.cfg.attach_yaw_only:
                self.cfg.ray_alignment = "yaw"
                msg += " Setting ray_alignment to 'yaw'."
            else:
                self.cfg.ray_alignment = "base"
                msg += " Setting ray_alignment to 'base'."
            omni.log.warn(msg)

        # Трансформа лучей в мир
        if self.cfg.ray_alignment == "world":
            pos_w[:, 0:2] += self.ray_cast_drift[env_ids, 0:2]
            ray_starts_w = self.ray_starts[env_ids] + pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "yaw":
            pos_w[:, 0:2] += quat_apply_yaw(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids]) + pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "base":
            pos_w[:, 0:2] += quat_apply(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids]) + pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        else:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}.")

        # --- Главное отличие: рейкаст по нескольким мешам и выбор ближайшего попадания ---
        # Собираем хиты и дистанции со всех мешей
        hits_all = []
        dists_all = []
        for _, mesh in self.meshes.items():
            hits = raycast_mesh(
                ray_starts_w, ray_directions_w, max_dist=self.cfg.max_distance, mesh=mesh
            )[0]  # совместимость с базовой сигнатурой
            # дистанции считаем по точкам (inf останется inf, что удобно для argmin)
            dists = torch.linalg.norm(hits - ray_starts_w, dim=-1)
            hits_all.append(hits)
            dists_all.append(dists)

        # Стек по измерению "меш"
        if len(hits_all) == 1:
            # быстрый путь: меш один — ничего не стекаем
            selected_hits = hits_all[0]  # [N, R, 3]
        else:
            hits_stack = torch.stack(hits_all, dim=-1)   # [N, R, 3, M]
            dists_stack = torch.stack(dists_all, dim=-1) # [N, R, M]

            # argmin по оси мешей
            min_idx = torch.argmin(dists_stack, dim=-1)  # [N, R]

            # индекс должен иметь те же 4 измерения, что и hits_stack, кроме последнего (M),
            # где размер берём 1, потом его уберём .squeeze(-1)
            idx_expanded = min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 1)  # [N, R, 3, 1]

            # собираем вдоль последней оси (M) и убираем её
            selected_hits = torch.gather(hits_stack, dim=-1, index=idx_expanded).squeeze(-1)  # [N, R, 3]

        self._data.ray_hits_w[env_ids] = selected_hits
        # верт. дрейф (как в оригинале)
        self._data.ray_hits_w[env_ids, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.ray_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # бывает, что колбэк прилетает ещё до инициализации буфера
        if not hasattr(self, "_data") or self._data is None:
            return
        if getattr(self._data, "ray_hits_w", None) is None:
            return
        viz_points = self._data.ray_hits_w.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        if viz_points.numel() == 0:
            return
        if hasattr(self, "ray_visualizer"):
            self.ray_visualizer.visualize(viz_points)

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        self._view = None

    def _expand_env_ns(self, pattern: str) -> str:
        # {ENV_REGEX_NS} -> /World/envs/env_<что угодно до следующего />
        return pattern.replace("{ENV_REGEX_NS}", r"/World/envs/env_[^/]+")
        
    def _compile_path_pattern(self, pattern: str) -> re.Pattern:
        orig = pattern  # анализируем, что написал пользователь
        pattern = self._expand_env_ns(pattern)

        # glob считаем ТОЛЬКО если в исходной строке были * ? []
        # И при этом нет явных regex-признаков (.* () | + {} ).
        is_glob = (any(ch in orig for ch in "*?[]")
                   and not re.search(r"[.\(\)\|\+\{\}]", orig))

        if is_glob:
            rx_str = fnmatch.translate(pattern)  # даёт ^...$
            if rx_str.startswith("^"):
                rx_str = rx_str[1:]
            if rx_str.endswith("$"):
                rx_str = rx_str[:-1]
            return re.compile(rx_str)

        # иначе трактуем как regex (подстрочно — через .search)
        return re.compile(pattern)
