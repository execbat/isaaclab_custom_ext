# Copyright (c) 2025.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, List, Tuple

import omni.log
import omni.usd
import omni.physics.tensors.impl.api as physx
import warp as wp
from isaacsim.core.prims import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd, UsdGeom, UsdPhysics, Gf

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.sensors.ray_caster.ray_caster_data import RayCasterData

if TYPE_CHECKING:
    from .ray_caster_cfg import RayCasterCfg


class RegexRayCaster(SensorBase):
    """
    Ray-caster с поддержкой регулярных выражений в mesh_prim_paths.
    Пример:
        RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            mesh_prim_paths=["{ENV_REGEX_NS}/obst_*", "/World/ground"],
            ...
        )

    Идея производительности:
    - На инициализации для КАЖДОГО env строим единый warp.Mesh, склеивая все подходящие меши (и/или плоскость).
    - На апдейте лучей делаем ровно ОДИН вызов raycast_mesh на env (батч всех лучей env).
    """

    cfg: RayCasterCfg

    def __init__(self, cfg: RayCasterCfg):
        # запрет regex в листовом сегменте самого сенсора оставим как в оригинале —
        # он относится к prim_path сенсора, а не к мешам
        sensor_leaf = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_leaf) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the ray-caster sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )
        super().__init__(cfg)
        self._data = RayCasterData()
        # по одному mesh на каждый env (индекс соответствует порядку инстансов self._view)
        self.meshes_per_env: List[wp.Mesh] = []
        # абсолютные пути неймспейсов окружений, совпадающие по индексу с инстансами
        self.env_roots: List[str] = []

    def __str__(self) -> str:
        return (
            f"Regex Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of env meshes : {len(self.meshes_per_env)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    # ----------
    # Properties
    # ----------
    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> RayCasterData:
        self._update_outdated_buffers()
        return self._data

    # ----------
    # Operations
    # ----------
    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            n = self._view.count
        else:
            n = len(env_ids)
        r = torch.empty(n, 3, device=self.device)
        self.drift[env_ids] = r.uniform_(*self.cfg.drift_range)

        range_list = [self.cfg.ray_cast_drift_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")]
        ranges = torch.tensor(range_list, device=self.device)
        self.ray_cast_drift[env_ids] = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 3), device=self.device)
        
        self._update_buffers_impl(slice(None) if env_ids is None else env_ids)

    # ---------------
    # Implementation
    # ---------------
    def _initialize_impl(self):
        super()._initialize_impl()
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # создать view сенсора (как в оригинале)
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

        # получить привязку env-инстансов к реальным путям env'ов
        self.env_roots = self._resolve_env_roots_for_instances()

        # подготовить warp-меши: один на env (объединяем все совпавшие примы)
        self._initialize_warp_meshes_regex()

        # лучи и буферы
        self._initialize_rays_impl()
        self._update_buffers_impl(slice(None))
    # -- helpers: env roots & mesh building --

    def _resolve_env_roots_for_instances(self) -> list[str]:
        """
        Ищем все реальные инстансы сенсора (по leaf-имени), и для каждого
        извлекаем env_root как '/World/envs/env_<k>' по строгой регулярке.
        Это устраняет ложное попадание сегмента '/Robot' в env_root.
        """
        stage = omni.usd.get_context().get_stage()
        pat = self.cfg.prim_path
    
        # находим leaf сенсора (последний сегмент пути)
        last_name = pat.split("/")[-1]  # напр. 'torso_link', 'base_link' и т.п.

        # соберём все кандидаты с таким именем
        sensor_paths: list[str] = []
        for prim in stage.TraverseAll():
            if prim.GetName() == last_name:
                sensor_paths.append(prim.GetPath().pathString)
        sensor_paths.sort()

        if not sensor_paths:
            raise RuntimeError(f"No instances found for sensor leaf '{last_name}' from pattern '{pat}'.")

        # извлекаем '/World/envs/env_<k>' строго по regex
        env_roots: list[str] = []
        for p in sensor_paths:
            env_root = self._extract_env_root_from_path(p)
            if env_root is None:
                omni.log.warn(f"[RegexRayCaster] Could not extract env root from '{p}'. Skipping.")
                continue
            env_roots.append(env_root)
    
        # привязываем к количеству инстансов view
        if not env_roots:
            raise RuntimeError("[RegexRayCaster] Failed to resolve any env roots from sensor instances.")

        if len(env_roots) != self._view.count:
            omni.log.warn(
                f"[RegexRayCaster] Resolved {len(env_roots)} env roots, but sensor view has {self._view.count} instances. "
                f"Proceeding with min count."
            )

        # диагностический лог
        omni.log.info(f"[RegexRayCaster] sensor_paths (sample): {sensor_paths[:min(4, len(sensor_paths))]}")
        omni.log.info(f"[RegexRayCaster] env_roots => {env_roots[:min(8, len(env_roots))]}")

        return env_roots[: self._view.count]
    
    def _extract_env_root_from_path(self, full_path: str) -> str | None:
        """
        Достаём '/World/envs/env_<k>' из любого полного USD-пути.
        Возвращает None, если префикс не найден.
        """
        m = re.match(r"^(/World/envs/env_\d+)\b", full_path)
        return m.group(1) if m else None


    def _initialize_warp_meshes_regex(self):
        """
        Для каждого env собираем объединённый warp.Mesh из всех паттернов cfg.mesh_prim_paths,
        где {ENV_REGEX_NS} подставляется конкретным env root (например '/World/envs/env_0').
        Поддерживаются:
          - '{ENV_REGEX_NS}' в путях,
          - glob '*' внутри сегментов пути (например '{ENV_REGEX_NS}/obst_*'),
          - глобальные пути без плейсхолдера (например '/World/ground') — добавляются во все env,
          - PhysX Plane под указанным узлом (добавляется бесконечная плоскость).
        """
        if len(self.cfg.mesh_prim_paths) < 1:
            raise RuntimeError("mesh_prim_paths must contain at least one entry.")

        stage = omni.usd.get_context().get_stage()
        device = self.device

        # Убедимся, что у нас есть env_roots (например ['/World/envs/env_0', '/World/envs/env_1', ...])
        if not hasattr(self, "env_roots") or not self.env_roots:
            self.env_roots = self._resolve_env_roots_for_instances()

        self.meshes_per_env = []

        for env_idx, env_root in enumerate(self.env_roots):
            all_points = []
            all_faces = []
            total_vertices = 0

            for mesh_pat in self.cfg.mesh_prim_paths:
                # Соберём все Mesh для данного env и паттерна
                matched = self._resolve_mesh_prims_for_env(stage, env_root, mesh_pat)

                if matched == "INFINITE_PLANE":
                    # бесконечная плоскость
                    plane = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                    pts = plane.vertices.astype(np.float32, copy=False)
                    fcs = plane.faces.astype(np.int32, copy=False)
                    all_points.append(pts)
                    all_faces.append(fcs + total_vertices)
                    total_vertices += pts.shape[0]
                    continue

                # обычные Mesh-примы
                for mesh_prim in matched:
                    if not mesh_prim or not mesh_prim.IsValid():
                        continue
                    if mesh_prim.GetTypeName() != "Mesh":
                        continue

                    usd_mesh = UsdGeom.Mesh(mesh_prim)
                    points_local = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=np.float32)

                    # world-трансформ
                    world_mtx = np.array(omni.usd.get_world_transform_matrix(usd_mesh)).T.astype(np.float32, copy=False)
                    pts = (points_local @ world_mtx[:3, :3].T) + world_mtx[:3, 3]
                    indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    
                    all_points.append(pts)
                    all_faces.append(indices + total_vertices)
                    total_vertices += pts.shape[0]

            if not all_points:
                omni.log.error(
                    f"[RegexRayCaster] No meshes for env {env_idx} ('{env_root}'). "
                    f"Tried patterns: {self.cfg.mesh_prim_paths}"
                )
                raise RuntimeError(
                    f"No meshes found for env {env_idx} ('{env_root}') using patterns: {self.cfg.mesh_prim_paths}"
                )

            # конкатенация
            pts_cat = np.concatenate(all_points, axis=0).astype(np.float32, copy=False)
            faces_cat = np.concatenate(all_faces, axis=0).astype(np.int32, copy=False)
            wp_mesh = convert_to_warp_mesh(pts_cat, faces_cat, device=device)

            omni.log.info(
                f"[RegexRayCaster] env {env_idx}: merged mesh -> {pts_cat.shape[0]} verts, {faces_cat.shape[0]} faces."
            )
            self.meshes_per_env.append(wp_mesh)

        if not self.meshes_per_env:
            raise RuntimeError("Failed to build any env meshes for RegexRayCaster.")

    def _resolve_mesh_prims_for_env(self, stage: Usd.Stage, env_root: str, mesh_pat: str):
        """
        Возвращает:
          - "INFINITE_PLANE", если под указанным узлом есть PhysX Plane;
          - список Prim'ов типа Mesh, если нашли совпадения.
        Поддержка:
          * '{ENV_REGEX_NS}' -> env_root
          * '*' внутри сегментов пути
          * если паттерн совпал с контейнером (Xform), собираем ВСЕ дочерние Mesh.
        """
        # подстановка плейсхолдера
        base = mesh_pat.replace("{ENV_REGEX_NS}", env_root) if "{ENV_REGEX_NS}" in mesh_pat else mesh_pat

        # детект бесконечной плоскости (если под base есть Plane)
        def has_plane_under(path_prefix: str) -> bool:
            prim = stage.GetPrimAtPath(path_prefix)
            if not prim or not prim.IsValid():
                return False
            for p in Usd.PrimRange(prim):
                if p.GetTypeName() == "Plane":
                    return True
            return False

        if has_plane_under(base):
            return "INFINITE_PLANE"

        # glob -> regex
        def glob_to_re(s: str) -> str:
            s = re.escape(s)
            s = s.replace(r"\*", "[^/]*")
            return s

        pat_re = re.compile(rf"^{glob_to_re(base)}$")

        # ограничим поиск поддеревом env, если base внутри него
        search_root = env_root if base.startswith(env_root) else "/"
        root_prim = stage.GetPrimAtPath(search_root)
        if not root_prim or not root_prim.IsValid():
            return []

        # сначала находим все примы, чьи пути совпадают с шаблоном (контейнеры/меши)
        matched_containers = []
        for p in Usd.PrimRange(root_prim):
            path = p.GetPath().pathString
            if pat_re.match(path):
                matched_containers.append(p)

        # из них достаём Mesh (сам прим может быть Mesh, либо ищем дочерние Mesh)
        mesh_prims = []
        for c in matched_containers:
            if c.GetTypeName() == "Mesh":
                mesh_prims.append(c)
            else:
                for dp in Usd.PrimRange(c):
                    if dp.GetTypeName() == "Mesh":
                        mesh_prims.append(dp)
    
#        # диагностический лог
#        if mesh_prims:
#            if omni.log.is_info_enabled():
#                sample = ", ".join(p.GetPath().pathString for p in mesh_prims[:6])
#                if len(mesh_prims) > 6:
#                    sample += ", ..."
#                omni.log.info(f"[RegexRayCaster] '{base}' -> {len(mesh_prims)} Mesh: {sample}")
#        else:
#            omni.log.warn(f"[RegexRayCaster] Pattern '{base}' matched no Mesh.")

        return mesh_prims


    def _initialize_rays_impl(self):
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(self.ray_directions)

        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)

        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos

        # по одному набору лучей на каждый сенсор (env инстанс)
        self.ray_starts = self.ray_starts.repeat(self._view.count, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self._view.count, 1, 1)

        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        self.ray_cast_drift = torch.zeros(self._view.count, 3, device=self.device)

        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

    # -------------------------
    # Update (vectorized per env)
    # -------------------------
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # позы сенсоров
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

        # дрейф базовой позиции
        pos_w += self.drift[env_ids]

        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # обратная совместимость
        if self.cfg.attach_yaw_only is not None:
            msg = (
                "Raycaster attribute 'attach_yaw_only' will be deprecated. "
                "Please use 'ray_alignment' instead."
            )
            if self.cfg.attach_yaw_only:
                self.cfg.ray_alignment = "yaw"
                msg += " Setting ray_alignment to 'yaw'."
            else:
                self.cfg.ray_alignment = "base"
                msg += " Setting ray_alignment to 'base'."
            omni.log.warn(msg)

        # обработка по env (векторизация внутри env)
        # это 1 вызов raycast_mesh на env (батч всех его лучей)
        # сохраняем порядок env_ids — он может быть списком, а не срезом
        if isinstance(env_ids, slice):
            env_iter = list(range(self._view.count))[env_ids]
        else:
            env_iter = list(env_ids)

        for i_local, env_id in enumerate(env_iter):
            pw = pos_w[i_local : i_local + 1]  # (1,3)
            qw = quat_w[i_local : i_local + 1]  # (1,4)

            if self.cfg.ray_alignment == "world":
                # горизонтальный дрейф x,y в мировых координатах
                pw[:, 0:2] += self.ray_cast_drift[env_id, 0:2]
                rs_w = self.ray_starts[env_id].clone()
                rs_w += pw
                rd_w = self.ray_directions[env_id]
            elif self.cfg.ray_alignment == "yaw":
                pw[:, 0:2] += quat_apply_yaw(qw, self.ray_cast_drift[env_id : env_id + 1])[0, 0:2]
                rs_w = quat_apply_yaw(qw.repeat(1, self.num_rays), self.ray_starts[env_id : env_id + 1])[0]
                rs_w += pw
                rd_w = self.ray_directions[env_id]
            elif self.cfg.ray_alignment == "base":
                pw[:, 0:2] += quat_apply(qw, self.ray_cast_drift[env_id : env_id + 1])[0, 0:2]
                rs_w = quat_apply(qw.repeat(1, self.num_rays), self.ray_starts[env_id : env_id + 1])[0]
                rs_w += pw
                rd_w = quat_apply(qw.repeat(1, self.num_rays), self.ray_directions[env_id : env_id + 1])[0]
            else:
                raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}")

            # === один вызов raycast на env (батч = 1) ===
            res = raycast_mesh(
                rs_w.unsqueeze(0),  # (1, num_rays, 3), 
                rd_w.unsqueeze(0),  # (1, num_rays, 3)
                max_dist=self.cfg.max_distance,
                mesh=self.meshes_per_env[env_id],
            )

            hits_b = res[0] if isinstance(res, (tuple, list)) else res      # (1, num_rays, 3) или (num_rays,3)
            hits = hits_b[0] if hits_b.dim() == 3 else hits_b               # (num_rays, 3)

            self._data.ray_hits_w[env_id] = hits

            self._data.ray_hits_w[env_id, :, 2] += self.ray_cast_drift[env_id, 2]

    # -------------------------
    # Debug visualization
    # -------------------------
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.ray_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        viz_points = self._data.ray_hits_w.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        self.ray_visualizer.visualize(viz_points)

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)  # keep parent behavior
        self._view = None

