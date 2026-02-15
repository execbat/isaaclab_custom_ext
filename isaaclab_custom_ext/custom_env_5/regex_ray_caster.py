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
    """
    A raycasting sensor with regex and multiple mesh support.

    Key differences from the basic RayCaster:
    - `mesh_prim_paths` can contain multiple paths;
    - each path in `mesh_prim_paths` is treated as a **regular expression** (Python re, fullmatch);
    - all matching Mesh or Plane prims are converted to warp meshes;
    - the raycast is sent to all warp meshes simultaneously, with the closest hit taken.

    Restrictions/behavior retained:
    - `cfg.prim_path` (leaf segment) cannot contain regex (see original warning);
    - Static meshes are supported, as in the original RayCaster.
    """

    cfg: RegexRayCasterCfg

    def __init__(self, cfg: RegexRayCasterCfg):
        sensor_leaf = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_leaf) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the ray-caster sensor: {cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )
        super().__init__(cfg)

        self._data = RayCasterData()
        
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

        r = torch.empty(num_envs_ids, 3, device=self.device)
        self.drift[env_ids] = r.uniform_(*self.cfg.drift_range)

        range_list = [self.cfg.ray_cast_drift_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=self.device)
        self.ray_cast_drift[env_ids] = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (num_envs_ids, 3), device=self.device
        )

        self.meshes.clear()
        self._retry_mesh_discovery = True

    # ----------------
    # Implementation
    # ----------------
    def _initialize_impl(self):
        super()._initialize_impl()
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # Create a view based on the prim type
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

        # Load warp meshes (regex + multiple)
        self._initialize_warp_meshes_regex()
        # Initialize the rays according to the pattern
        self._initialize_rays_impl()

    # ---- regex expansion for meshes ----
    def _initialize_warp_meshes_regex(self):
        """Finds prims by regex / glob and collects ALL Mesh/Plane under them (including instances)."""
        import isaaclab.sim as sim_utils
        stage = omni.usd.get_context().get_stage()

        # clean up to avoid accumulation of garbage
        self.meshes.clear()
        seen_paths: set[str] = set()

        def _add_plane(plane_prim):
            mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
            wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
            key = plane_prim.GetPath().pathString
            self.meshes[key] = wp_mesh
            omni.log.info(f"[RegexRayCaster] Added Plane: {key}")

        def _add_mesh(mesh_prim):
            mesh_geom = UsdGeom.Mesh(mesh_prim)

            # vertices in local
            points = np.asarray(mesh_geom.GetPointsAttr().Get(), dtype=np.float32)

            # indices from USD
            indices = np.asarray(mesh_geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            
            if indices.size % 3 != 0:
                omni.log.warn(f"[RegexRayCaster] Non-triangle mesh {mesh_geom.GetPath()}, indices % 3 != 0, skipping.")
                return
            indices = indices.reshape(-1, 3)

            # world-transform
            xform = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
            points = (points @ xform[:3, :3].T) + xform[:3, 3]

            wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
            key = mesh_geom.GetPath().pathString
            self.meshes[key] = wp_mesh

            omni.log.info(
                f"[RegexRayCaster] Added Mesh: {key} (V={len(points)}, F={len(indices)})"
            )

        # compile regex from user patterns
        compiled = [self._compile_path_pattern(p) for p in self.cfg.mesh_prim_paths]

        matched_roots: list[Usd.Prim] = []
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if any(rx.search(p) for rx in compiled):
                matched_roots.append(prim)

        for root in matched_roots:
            root_path = root.GetPath().pathString
            if root_path in seen_paths:
                continue
            seen_paths.add(root_path)

            # collect ALL Mesh/Plane under this root, including instances
            mesh_prims = sim_utils.get_all_matching_child_prims(
                root_path,
                lambda p: p.GetTypeName() == "Mesh",
                stage=stage,
                traverse_instance_prims=True,
            )
            plane_prims = sim_utils.get_all_matching_child_prims(
                root_path,
                lambda p: p.GetTypeName() == "Plane",
                stage=stage,
                traverse_instance_prims=True,
            )

            if not mesh_prims and not plane_prims:
                tname = root.GetTypeName()
                if tname == "Mesh":
                    mesh_prims = [root]
                elif tname == "Plane":
                    plane_prims = [root]

            for m in mesh_prims:
                key = m.GetPath().pathString
                if key not in self.meshes:
                    _add_mesh(m)
            for pl in plane_prims:
                key = pl.GetPath().pathString
                if key not in self.meshes:
                    _add_plane(pl)

        omni.log.info(
            f"[RegexRayCaster] matched_roots={len(matched_roots)}, loaded meshes={len(self.meshes)}"
        )

        if not self.meshes:
            omni.log.warn(
                "[RegexRayCaster] No meshes found at init. Will retry on first update. "
                f"Patterns: {self.cfg.mesh_prim_paths}"
            )
            self._retry_mesh_discovery = True
        else:
            self._retry_mesh_discovery = False
        
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

        # Transformation of rays into the world
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

        # Collect hits and distances from all meshes
        hits_all = []
        dists_all = []
        for _, mesh in self.meshes.items():
            hits = raycast_mesh(
                ray_starts_w, ray_directions_w, max_dist=self.cfg.max_distance, mesh=mesh
            )[0] 
            # distances are calculated by points (inf will remain inf, which is convenient for argmin)
            dists = torch.linalg.norm(hits - ray_starts_w, dim=-1)
            hits_all.append(hits)
            dists_all.append(dists)

        # Stack by "mesh" dimension
        if len(hits_all) == 1:
            # fast way: one mesh - nothing to drain
            selected_hits = hits_all[0]  # [N, R, 3]
        else:
            hits_stack = torch.stack(hits_all, dim=-1)   # [N, R, 3, M]
            dists_stack = torch.stack(dists_all, dim=-1) # [N, R, M]

            # argmin along the mesh axis
            min_idx = torch.argmin(dists_stack, dim=-1)  # [N, R]

            # The index must have the same 4 dimensions as hits_stack, except for the last one (M),
            # where we take the size as 1, then remove it. .squeeze(-1)
            idx_expanded = min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 1)  # [N, R, 3, 1]

            # we collect along the last axis (M) and remove it
            selected_hits = torch.gather(hits_stack, dim=-1, index=idx_expanded).squeeze(-1)  # [N, R, 3]

        self._data.ray_hits_w[env_ids] = selected_hits
        # vert. drift
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
        return pattern.replace("{ENV_REGEX_NS}", r"/World/envs/env_[^/]+")
        
    def _compile_path_pattern(self, pattern: str) -> re.Pattern:
        orig = pattern 
        pattern = self._expand_env_ns(pattern)

        is_glob = (any(ch in orig for ch in "*?[]")
                   and not re.search(r"[.\(\)\|\+\{\}]", orig))

        if is_glob:
            rx_str = fnmatch.translate(pattern)  
            if rx_str.startswith("^"):
                rx_str = rx_str[1:]
            if rx_str.endswith("$"):
                rx_str = rx_str[:-1]
            return re.compile(rx_str)
       
        return re.compile(pattern)
