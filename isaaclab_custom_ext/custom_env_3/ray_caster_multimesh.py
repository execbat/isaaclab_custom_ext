# isaaclab_custom_ext/custom_sensors/ray_caster_multimesh.py

from __future__ import annotations

import re
import numpy as np
import torch
import omni
import warp as wp
from typing import Sequence, Dict, Iterable, List, Optional

from pxr import Usd, UsdGeom, UsdPhysics
from omni.usd import get_context

from isaacsim.core.prims import XFormPrim as _XFormPrim
import omni.physics.tensors.impl.api as physx

from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
from isaaclab.sensors.ray_caster.ray_caster import RayCaster
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw

# Accept both /World/envs/... and /World/ground/envs/...
ABS_ENV_REGEX = r"/World(?:/ground)?/envs/env_.*"


class RayCasterMultiMesh(RayCaster):
    """
    RayCaster with multi-mesh support + dynamic PhysX poses.

    - Mesh geometry is stored in LOCAL mesh coordinates (no baked world xform).
    - For each mesh we find the closest ancestor that has RigidBodyAPI and cache a PhysX RigidBodyView.
    - Every update:
        * Re-scan mesh list (to pick up respawned objects).
        * For each mesh: get current PhysX pose (pos+quat) of its rigid parent, build 4x4 transform.
        * Transform rays world->local, raycast, transform hits back local->world.
        * Keep nearest hit per ray across all meshes.
    """

    # --- initialization -------------------------------------------------------

    def _initialize_warp_meshes(self):
        self.meshes: Dict[str, wp.Mesh] = {}             # mesh_path -> wp.Mesh (local)
        self._mesh_paths: List[str] = []                 # current set of mesh paths
        self._mesh_parent_rb_path: Dict[str, str] = {}   # mesh_path -> parent rigid body prim path
        self._rb_views: Dict[str, physx.RigidBodyView] = {}  # rb_path -> PhysX view (count==1)

        self._refresh_mesh_list(add_missing_to_cache=True)

        if not self._mesh_paths:
            raise RuntimeError(
                f"[RayCasterMultiMesh] No meshes found for ray-casting! "
                f"Check mesh_prim_paths: {self.cfg.mesh_prim_paths}"
            )

    # --- helpers -------------------------------------------------------------

    def _normalize_path(self, p: str) -> str:
        """Make the prim path absolute and expand {ENV_REGEX_NS}."""
        if "{ENV_REGEX_NS}" in p:
            p = p.replace("{ENV_REGEX_NS}", ABS_ENV_REGEX)
        if not p.startswith("/"):
            raise ValueError(f"[RayCasterMultiMesh] Prim path '{p}' must be absolute (start with '/').")
        return p

    def _iter_matching_mesh_prims(self, root_pattern: str) -> Iterable[UsdGeom.Mesh]:
        """Yield all UsdGeom.Mesh prims under subtrees matching the regex pattern."""
        regex_str = "^" + root_pattern.replace("*", ".*") + "$"
        regex = re.compile(regex_str)

        stage = get_context().get_stage()
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            path_str = prim.GetPath().pathString
            if regex.match(path_str):
                for child in Usd.PrimRange(prim):
                    if child.GetTypeName() == "Mesh":
                        yield UsdGeom.Mesh(child)

    def _collect_matching_mesh_paths(self) -> List[str]:
        """Collect USD paths for all meshes that match any user pattern."""
        paths: List[str] = []
        for mesh_root in self.cfg.mesh_prim_paths:
            pattern = self._normalize_path(mesh_root)
            for mesh_prim in self._iter_matching_mesh_prims(pattern):
                paths.append(mesh_prim.GetPath().pathString)
        return sorted(set(paths))

    def _find_parent_rigid_body_path(self, mesh_path: str) -> Optional[str]:
        """Walk up parents until a prim with RigidBodyAPI is found; return its path or None."""
        stage = get_context().get_stage()
        prim = stage.GetPrimAtPath(mesh_path)
        while prim and prim.IsValid():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return prim.GetPath().pathString
            prim = prim.GetParent()
        return None

    def _get_or_create_rb_view(self, rb_path: str) -> physx.RigidBodyView:
        """Create a PhysX RigidBodyView for a single rigid body path (no regex), cache and return it."""
        view = self._rb_views.get(rb_path)
        if view is not None:
            return view
        sim_view = self._physics_sim_view
        # Use exact path (no regex), so the view has count==1 and indexing is trivial.
        view = sim_view.create_rigid_body_view(rb_path)
        if view.count != 1:
            # In practice with an exact path we expect exactly one body.
            omni.log.warn(f"[RayCasterMultiMesh] Unexpected body count {view.count} for rb '{rb_path}'")
        self._rb_views[rb_path] = view
        return view

    def _refresh_mesh_list(self, add_missing_to_cache: bool):
        """
        Re-scan the stage each call:
          - Update the set of mesh paths `_mesh_paths`.
          - Optionally add new meshes to cache (wp.Mesh) and create PhysX views for their rigid parents.
        """
        # ensure we have a physics sim view (same as base class did in _initialize_impl)
        if not hasattr(self, "_physics_sim_view"):
            from isaacsim.core.simulation_manager import SimulationManager
            self._physics_sim_view = SimulationManager.get_physics_sim_view()

        stage = get_context().get_stage()
        found_paths = self._collect_matching_mesh_paths()
        self._mesh_paths = found_paths

        if not add_missing_to_cache:
            return

        for path in found_paths:
            # Cache wp.Mesh (local coords)
            if path not in self.meshes:
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid() or prim.GetTypeName() != "Mesh":
                    continue
                mesh = UsdGeom.Mesh(prim)
                points = np.asarray(mesh.GetPointsAttr().Get())                   # local vertices
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
                self.meshes[path] = wp_mesh
                omni.log.info(
                    f"[RayCasterMultiMesh] Cached mesh (local): {path} "
                    f"verts={len(points)} faces={len(indices)}"
                )
            # Map mesh -> parent rigid body and ensure we have a PhysX view
            if path not in self._mesh_parent_rb_path:
                rb_path = self._find_parent_rigid_body_path(path)
                if rb_path is not None:
                    self._mesh_parent_rb_path[path] = rb_path
                    self._get_or_create_rb_view(rb_path)  # ensure cached
                else:
                    # Mesh without a rigid body parent — will be treated as static (USD xform)
                    self._mesh_parent_rb_path[path] = ""  # sentinel

    # --- core update ---------------------------------------------------------

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Same as base, but raycasting is done per mesh with **current PhysX pose** of its rigid parent."""
        # Keep meshes in sync (handles respawns/moves).
        self._refresh_mesh_list(add_missing_to_cache=True)

        # === sensor pose (copy of base logic) ===
        if isinstance(self._view, _XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            if hasattr(self._view, "get_world_poses"):
                pos_w, quat_w = self._view.get_world_poses(env_ids)
            else:
                raise RuntimeError(f"[RayCasterMultiMesh] Unsupported view type: {type(self._view)}")

        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        pos_w += self.drift[env_ids]

        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # === ray alignment (as in base) ===
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

        if self.cfg.ray_alignment == "world":
            pos_w[:, 0:2] += self.ray_cast_drift[env_ids, 0:2]
            ray_starts_w = self.ray_starts[env_ids] + pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "yaw":
            pos_w[:, 0:2] += quat_apply_yaw(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "base":
            pos_w[:, 0:2] += quat_apply(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        else:
            raise RuntimeError(f"[RayCasterMultiMesh] Unsupported ray_alignment: {self.cfg.ray_alignment}")

        # === multi-mesh raycast with PhysX poses ===
        B, R, _ = ray_starts_w.shape
        best_hits_w = torch.full((B, R, 3), float("inf"), device=self.device, dtype=ray_starts_w.dtype)
        best_dist = torch.full((B, R), float("inf"), device=self.device, dtype=ray_starts_w.dtype)

        stage = get_context().get_stage()

        for mesh_path in self._mesh_paths:
            wp_mesh = self.meshes.get(mesh_path)
            if wp_mesh is None:
                continue

            rb_path = self._mesh_parent_rb_path.get(mesh_path, "")
            # Build world transform from PhysX if rb exists; otherwise fall back to static USD xform.
            if rb_path:
                rb_view = self._get_or_create_rb_view(rb_path)
                # exact view has count==1
                tr = rb_view.get_transforms()[0]  # (7,) as [x,y,z, qx,qy,qz,qw]
                pos = tr[0:3]
                quat = tr[3:7]
                quat = convert_quat(quat.unsqueeze(0), to="wxyz")[0]  # (4,) wxyz
                # Rotation matrix from unit quaternion
                w, x, y, z = quat[0], quat[1], quat[2], quat[3]
                R = torch.tensor(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ],
                    device=self.device,
                    dtype=ray_starts_w.dtype,
                )
                t = pos.to(device=self.device, dtype=ray_starts_w.dtype)
            else:
                # Static fallback (USD xform)
                prim = stage.GetPrimAtPath(mesh_path)
                if not prim.IsValid():
                    continue
                M = np.array(omni.usd.get_world_transform_matrix(prim)).T
                R = torch.as_tensor(M[:3, :3], device=self.device, dtype=ray_starts_w.dtype)
                t = torch.as_tensor(M[:3, 3], device=self.device, dtype=ray_starts_w.dtype)

            # Inverse (handles rotation; scale from PhysX is identity)
            R_inv = torch.inverse(R)

            # World -> local
            S_w = ray_starts_w - t.view(1, 1, 3)                       # (B,R,3)
            S_l = torch.einsum("ij,brj->bri", R_inv, S_w)              # (B,R,3)
            D_l = torch.einsum("ij,brj->bri", R_inv, ray_directions_w) # (B,R,3)

            # Raycast in local mesh space
            hits_l = raycast_mesh(S_l, D_l, max_dist=self.cfg.max_distance, mesh=wp_mesh)[0]  # (B,R,3)

            # Local -> world
            H_w = torch.einsum("ij,brj->bri", R, hits_l) + t.view(1, 1, 3)

            # Distances in world; no-hit stays inf
            dists = torch.linalg.norm(H_w - ray_starts_w, dim=-1)  # (B,R)

            # Keep nearest
            mask = dists < best_dist
            if mask.any():
                best_dist[mask] = dists[mask]
                best_hits_w[mask.unsqueeze(-1).expand_as(H_w)] = H_w[mask.unsqueeze(-1).expand_as(H_w)]

        # Save and apply vertical drift on Z (as in base)
        self._data.ray_hits_w[env_ids] = best_hits_w
        self._data.ray_hits_w[env_ids, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)

    # Robust debug vis (skip when empty)
    def _debug_vis_callback(self, event):
        viz_points = self._data.ray_hits_w.reshape(-1, 3)
        finite = torch.isfinite(viz_points).all(dim=1)
        viz_points = viz_points[finite]
        if viz_points.numel() == 0:
            return
        self.ray_visualizer.visualize(viz_points)

