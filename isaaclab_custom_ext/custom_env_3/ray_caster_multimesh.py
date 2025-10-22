# isaaclab_custom_ext/custom_sensors/ray_caster_multimesh.py
from __future__ import annotations
import os
import re
from typing import List, Optional, Sequence, Tuple

import omni
import torch
import numpy as np
from pxr import Usd

import omni.physics.tensors.impl.api as physx
from isaacsim.core.prims import XFormPrim as _XFormPrim

from isaaclab.sensors.ray_caster.ray_caster import RayCaster
from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw

# -----------------------------------------------------------------------------
# Triton (опционально).
# -----------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# -----------------------------------------------------------------------------
# Утилиты
# -----------------------------------------------------------------------------
def _torch_device(dev) -> torch.device:
    return dev if isinstance(dev, torch.device) else torch.device(dev)

def _collect_env0_obst_slots() -> List[str]:
    """Собрать пути препятствий вида .../envs/env_0/obst_XX."""
    stage = omni.usd.get_context().get_stage()
    pat = re.compile(r".*/envs/env_0/obst_\d+$")
    out: List[str] = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.IsValid():
            p = prim.GetPath().pathString
            if pat.match(p):
                out.append(p)
    out.sort()
    return out

def _quat_to_axes_wxyz(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Получить ортонормальные оси (u,v,w) из кватерниона (wxyz). Возврат формы (...,3)."""
    w, x, y, z = q.unbind(-1)
    ww = w*w; xx = x*x; yy = y*y; zz = z*z
    wx = w*x; wy = w*y; wz = w*z
    xy = x*y; xz = x*z; yz = y*z
    u = torch.stack([1 - 2*(yy + zz), 2*(xy + wz),       2*(xz - wy)], dim=-1)
    v = torch.stack([2*(xy - wz),     1 - 2*(xx + zz),   2*(yz + wx)], dim=-1)
    w_ = torch.stack([2*(xz + wy),    2*(yz - wx),       1 - 2*(xx + yy)], dim=-1)
    return u, v, w_


# -----------------------------------------------------------------------------
# Конфиг
# -----------------------------------------------------------------------------
@configclass
class RayCasterMultiMeshCfg(RayCasterCfg):
    # геометрия цилиндров
    cylinder_radius: float = 0.4
    cylinder_height: float = 1.2

    # вычислительный режим
    use_fp16: bool = True

    # сетка (широкая фаза)
    grid_cell: float = 0.9          # ~ 2*r
    grid_pad:  float = 2.0          # запас по краям в клетках
    regrid_trigger_cells: float = 1.0

    # DDA
    dda_max_cells: int = 48         # ограничение шагов DDA (в ячейках)
    kmax: int = 8                   # макс. кандидатов на луч

    # тайлинг по лучам
    block_R: int = 1024             # размер плитки по лучам


# -----------------------------------------------------------------------------
# Triton ядро для DDA (широкая фаза) — опционально
# -----------------------------------------------------------------------------
def _build_triton_dda_kernel():
    if not _HAS_TRITON:
        return None

    @triton.jit
    def _dda_candidates_kernel(
        # --- per-ray inputs (flattened N=B*R) ---
        Sx_ptr, Sy_ptr, Dx_ptr, Dy_ptr,        # (N,) float32
        origin_x, origin_y,                    # float32
        cell_size,                             # float32
        nx: tl.constexpr, ny: tl.constexpr,    # grid dims (compile-time ints)
        # --- CSR grid buffers (flattened by env) ---
        cell_start_ptr,                        # (E*n_cells,) int32
        cell_count_ptr,                        # (E*n_cells,) int32
        cell_items_ptr,                        # (E*M,)       int32
        # --- outputs ---
        out_ptr,                               # (N*KMAX,)    int32
        # --- per-ray env index ---
        env_id_ptr,                            # (N,)         int32
        # --- meta / sizes ---
        KMAX: tl.constexpr,                    # candidates per ray (compile-time)
        DDA_STEPS: tl.constexpr,               # max DDA steps (compile-time)
        N: tl.constexpr,                       # total rays
        N_CELLS: tl.constexpr,                 # nx*ny (compile-time)
        BLOCK: tl.constexpr                    # block size
    ):
        # lane ids
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        lane_mask = offs < N

        # load inputs
        Sx = tl.load(Sx_ptr + offs, mask=lane_mask, other=0.0)
        Sy = tl.load(Sy_ptr + offs, mask=lane_mask, other=0.0)
        Dx = tl.load(Dx_ptr + offs, mask=lane_mask, other=0.0)
        Dy = tl.load(Dy_ptr + offs, mask=lane_mask, other=0.0)
        env = tl.load(env_id_ptr + offs, mask=lane_mask, other=0).to(tl.int32)

        # normalize direction in XY
        invn = tl.rsqrt(Dx*Dx + Dy*Dy + 1e-12)
        Dx *= invn
        Dy *= invn

        # starting cell (clamped)
        fx = (Sx - origin_x) / cell_size
        fy = (Sy - origin_y) / cell_size
        ix = tl.cast(tl.math.floor(fx), tl.int32)
        iy = tl.cast(tl.math.floor(fy), tl.int32)
        ix = tl.minimum(tl.maximum(ix, 0), nx - 1)
        iy = tl.minimum(tl.maximum(iy, 0), ny - 1)

        # DDA stepping params
        step_x = tl.where(Dx >= 0.0, 1, -1)
        step_y = tl.where(Dy >= 0.0, 1, -1)

        cell_x = origin_x + (tl.cast(ix, tl.float32) + tl.where(Dx >= 0.0, 1.0, 0.0)) * cell_size
        cell_y = origin_y + (tl.cast(iy, tl.float32) + tl.where(Dy >= 0.0, 1.0, 0.0)) * cell_size

        t_max_x  = tl.where(Dx != 0.0, (cell_x - Sx) / Dx, 1e30)
        t_max_y  = tl.where(Dy != 0.0, (cell_y - Sy) / Dy, 1e30)
        t_delta_x = tl.where(Dx != 0.0, cell_size / tl.abs(Dx), 1e30)
        t_delta_y = tl.where(Dy != 0.0, cell_size / tl.abs(Dy), 1e30)

        # output row ptr + init -1
        out_row0 = out_ptr + offs * KMAX
        for k in tl.static_range(0, KMAX):
            tl.store(out_row0 + k, tl.full((), -1, tl.int32), mask=lane_mask)

        out_count = tl.zeros((BLOCK,), dtype=tl.int32)

        # bases for CSR per env
        cs_base = env * N_CELLS   # base offset in cell_start/count
        it_base = env * 0         # base in items is handled by env*M at access time (see below)

        # DDA loop
        for _ in tl.static_range(DDA_STEPS):
            # clamp cell indices
            ix = tl.minimum(tl.maximum(ix, 0), nx - 1)
            iy = tl.minimum(tl.maximum(iy, 0), ny - 1)
            cid = iy * nx + ix  # (BLOCK,)

            # load start/count for each lane's current cell
            cs = tl.load(cell_start_ptr + cs_base + cid, mask=lane_mask, other=0)
            cc = tl.load(cell_count_ptr + cs_base + cid, mask=lane_mask, other=0)

            # iterate items of this cell
            i = tl.zeros((BLOCK,), dtype=tl.int32)
            # repeat mask: lanes that still have items and space in output
            repeat = (i < cc) & (out_count < KMAX) & lane_mask

            # emulate `tl.any(repeat)` via sum>0
            while (tl.sum(repeat, axis=0) > 0):
                # payload is laid as (E, M) contiguous rows → absolute index = env*M + (cs + i) % M
                # but we constructed CSR so that 0 <= cs+i < M; direct addressing:
                itm = tl.load(cell_items_ptr + env * M + cs + i, mask=repeat, other=-1)
                ok  = (itm >= 0) & repeat
                tl.store(out_row0 + out_count, itm, mask=ok)
                out_count = tl.where(ok, out_count + 1, out_count)
                i += 1
                repeat = (i < cc) & (out_count < KMAX) & lane_mask

            # step to next cell along the fastest boundary
            choose_x = t_max_x <= t_max_y
            ix      += tl.where(choose_x, step_x, 0)
            iy      += tl.where(choose_x, 0,      step_y)
            t_max_x  = tl.where(choose_x, t_max_x + t_delta_x, t_max_x)
            t_max_y  = tl.where(choose_x, t_max_y,             t_max_y + t_delta_y)

            # early exit when all lanes filled KMAX or inactive
            still_active = tl.sum((out_count < KMAX) & lane_mask, axis=0)
            if still_active == 0:
                break

    return _dda_candidates_kernel


# -----------------------------------------------------------------------------
# ГЛАВНЫЙ КЛАСС
# -----------------------------------------------------------------------------
class RayCasterMultiMesh(RayCaster):
    """
    CUDA-оптимизированный многомешевой (наклонные цилиндры) рейкастер:

    • Одна PhysX-view на '/.../envs/env_*/obst_*', позы читаем разом → (E,M,7).
    • Широкая фаза: равномерная XY-решётка (CSR) + DDA. Опционально — Triton-ядро.
    • Узкая фаза: аналитика луч–наклонный цилиндр целиком в тензорах CUDA.
    • Минимум аллокаций и копий, плиточный проход по лучам (block_R).
    """

    # ---------------- init ----------------
    def _initialize_warp_meshes(self):
        log = omni.log

        slots = _collect_env0_obst_slots()
        if len(slots) == 0:
            raise RuntimeError(
                "[RayCasterMultiMesh] No '/envs/env_0/obst_XX' found. "
                "Ensure obstacles '.../envs/env_#/obst_##' exist."
            )
        self._M = len(slots)

        # glob для PhysX view
        prefix = slots[0].split("/envs/env_0")[0]
        self._rb_glob_pattern = f"{prefix}/envs/env_*/obst_*"

        # геометрия
        r = getattr(self.cfg, "cylinder_radius", None)
        h = getattr(self.cfg, "cylinder_height", None)
        if r is None or h is None:
            raise RuntimeError(
                "[RayCasterMultiMesh] Please set cfg.cylinder_radius and cfg.cylinder_height (e.g., 0.4 / 1.2)."
            )
        self._cyl_radius = float(r)
        self._cyl_height = float(h)

        # режимы / параметры
        self._use_fp16 = bool(getattr(self.cfg, "use_fp16", True))
        self._grid_cell = float(getattr(self.cfg, "grid_cell", 0.9))
        self._grid_pad  = float(getattr(self.cfg, "grid_pad", 2.0))
        self._regrid_cells = float(getattr(self.cfg, "regrid_trigger_cells", 1.0))
        self._kmax = int(getattr(self.cfg, "kmax", 8))
        self._dda_max = int(getattr(self.cfg, "dda_max_cells", 48))
        self._block_R = int(getattr(self.cfg, "block_R", 1024))

        # device
        self._torch_device: torch.device = _torch_device(self.device)

        # PhysX view
        self._rb_view_all: Optional[physx.RigidBodyView] = None
        self._E_total: Optional[int] = None

        # grid
        self._grid = None
        self._grid_dirty = True
        self._last_grid_origin = None

        # кеш препятствий (E,M,3)
        self._C_all = None
        self._U_all = None
        self._V_all = None
        self._W_all = None

        # PhysX sim view
        if not hasattr(self, "_physics_sim_view"):
            from isaacsim.core.simulation_manager import SimulationManager
            self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # CUDA флаги
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Triton ядро (ленивая компиляция)
        self._triton_dda = _build_triton_dda_kernel()

        log.info(
            f"[RayCasterMultiMesh] M={self._M} | r={self._cyl_radius:.3f}, h={self._cyl_height:.3f} "
            f"| grid_cell={self._grid_cell:.3f} | kmax={self._kmax} | dda_max={self._dda_max} "
            f"| Triton={'on' if (self._triton_dda is not None and torch.cuda.is_available()) else 'off'}"
        )

    def _ensure_rb_view(self):
        if self._rb_view_all is not None:
            return
        view = self._physics_sim_view.create_rigid_body_view(self._rb_glob_pattern)
        self._rb_view_all = view
        if view is None or view.count == 0:
            raise RuntimeError(
                f"[RayCasterMultiMesh] No rigid bodies matched '{self._rb_glob_pattern}'."
            )
        if view.count % self._M != 0:
            omni.log.warn(
                f"[RayCasterMultiMesh] RigidBodyView count ({view.count}) not divisible by M ({self._M})."
            )
        self._E_total = view.count // self._M

    def _ensure_grid(self):
        if self._grid is not None:
            return
        E, M = self._E_total, self._M
        dev = self._torch_device
        self._grid = dict(
            origin=torch.zeros(2, device=dev, dtype=torch.float32),
            cell=float(self._grid_cell),
            shape=(1, 1),
            cell_start=torch.empty((E, 1), device=dev, dtype=torch.int32),
            cell_count=torch.empty((E, 1), device=dev, dtype=torch.int32),
            cell_items=torch.full((E, M), -1, device=dev, dtype=torch.int32),  # payload (CSR)
        )

    # ---------------- сетка + препятствия ----------------
    def _update_obstacles_and_grid(self, force_regrid: bool = False):
        """Обновить позы препятствий, кэш осей и (по необходимости) пересобрать CSR-сетку."""
        self._ensure_rb_view()
        self._ensure_grid()
        dev = self._torch_device

        # позы препятствий
        tf_all = self._rb_view_all.get_transforms().to(device=dev, dtype=torch.float32)  # (E*M, 7)
        tf_all = tf_all.view(self._E_total, self._M, 7)
        C = tf_all[..., 0:3]                          # (E,M,3)
        q = convert_quat(tf_all[..., 3:7], to="wxyz") # (E,M,4)
        u, v, w_axis = _quat_to_axes_wxyz(q)          # (E,M,3) * 3

        # кэш (FP16 при AMP)
        want_half = self._use_fp16 and torch.cuda.is_available()
        dtype_cache = torch.float16 if want_half else torch.float32
        self._C_all = C.to(dtype_cache)
        self._U_all = u.to(dtype_cache)
        self._V_all = v.to(dtype_cache)
        self._W_all = w_axis.to(dtype_cache)

        # bbox по XY
        Cx = C[..., 0];  Cy = C[..., 1]
        cell = float(self._grid_cell)
        xmin = torch.min(Cx) - self._grid_pad * cell
        xmax = torch.max(Cx) + self._grid_pad * cell
        ymin = torch.min(Cy) - self._grid_pad * cell
        ymax = torch.max(Cy) + self._grid_pad * cell

        origin = torch.stack([xmin, ymin]).to(device=dev, dtype=torch.float32)
        nx = int(torch.clamp(((xmax - xmin) / cell).ceil(), min=1).item())
        ny = int(torch.clamp(((ymax - ymin) / cell).ceil(), min=1).item())
        n_cells = nx * ny

        # редкое перестроение сетки при малом сдвиге
        if (not force_regrid) and (self._last_grid_origin is not None):
            delta_cells = torch.max(torch.abs((origin - self._last_grid_origin) / cell)).item()
            if delta_cells < self.cfg.regrid_trigger_cells and self._grid["cell_start"].shape[1] == n_cells:
                return
        self._last_grid_origin = origin.clone()

        # раскладка препятствий по клеткам → CSR
        ix = ((Cx - origin[0]) / cell).floor().to(torch.int64).clamp(0, nx - 1)  # (E,M) long
        iy = ((Cy - origin[1]) / cell).floor().to(torch.int64).clamp(0, ny - 1)  # (E,M) long
        cid = (iy * nx + ix).to(torch.int64)                                     # (E,M) long  [0..n_cells)

        cell_count = torch.zeros((self._E_total, n_cells), device=dev, dtype=torch.int32)
        cell_count.scatter_add_(1, cid, torch.ones_like(cid, dtype=torch.int32))   # long-индексы ок

        cell_start = torch.zeros_like(cell_count)
        torch.cumsum(cell_count, dim=1, out=cell_start)
        cell_start -= cell_count                                                  # эксклюзивная префикс-сумма

        # payload: ровно M позиций на env
        cell_items = torch.full((self._E_total, self._M), -1, device=dev, dtype=torch.int32)
        write_pos = cell_start.clone()

        offs_i32 = torch.gather(write_pos, 1, cid)           # (E,M) int32
        offs = offs_i32.to(torch.int64)                      # long для scatter
        write_pos.scatter_add_(1, cid, torch.ones_like(cid, dtype=torch.int32))

        item_ids = torch.arange(self._M, device=dev, dtype=torch.int32).unsqueeze(0).expand_as(cid)
        cell_items.scatter_(1, offs.clamp_max(self._M - 1), item_ids)

        # commit
        self._grid["origin"] = origin
        self._grid["cell"] = cell
        self._grid["shape"] = (nx, ny)
        self._grid["cell_start"] = cell_start          # (E, n_cells) int32
        self._grid["cell_count"] = cell_count          # (E, n_cells) int32
        self._grid["cell_items"] = cell_items          # (E, M) int32

    # ---------------- Triton-/torch-сбор кандидатов ----------------
    def _gather_candidates(self, Sxy: torch.Tensor, Dxy: torch.Tensor) -> torch.Tensor:
        """
        Сбор кандидатов пер-луч (широкая фаза).
        Вход:  Sxy, Dxy: (B, Rb, 2) float32
        Выход: (B, Rb, K) long, значения в [0..M-1] или -1.
        """
        device = Sxy.device
        B, Rb, _ = Sxy.shape
        K = int(self._kmax)
        M = int(self._M)

        if self._grid is None or self._grid["cell_start"] is None:
            raise RuntimeError("[RayCasterMultiMesh] Grid is not built.")

        origin = self._grid["origin"]
        cell = float(self._grid["cell"])
        nx, ny = self._grid["shape"]
        n_cells = nx * ny

        cell_start = self._grid["cell_start"].reshape(-1).to(torch.int32)  # (E*n_cells,)
        cell_count = self._grid["cell_count"].reshape(-1).to(torch.int32)
        cell_items = self._grid["cell_items"].reshape(-1).to(torch.int32)

        # сопоставление батча env_ids
        env_ids = getattr(self, "_curr_env_ids_tensor", None)
        if env_ids is None or env_ids.numel() != B:
            raise RuntimeError("[RayCasterMultiMesh] Internal: _curr_env_ids_tensor not set or wrong shape.")
        env_map = env_ids.to(device=device, dtype=torch.int32).view(B, 1).expand(B, Rb).reshape(-1)   # (N,)

        # плоские входы
        Sx = Sxy[..., 0].reshape(-1).to(torch.float32).contiguous()
        Sy = Sxy[..., 1].reshape(-1).to(torch.float32).contiguous()
        Dx = Dxy[..., 0].reshape(-1).to(torch.float32).contiguous()
        Dy = Dxy[..., 1].reshape(-1).to(torch.float32).contiguous()
        N = Sx.numel()

        out = torch.full((N, K), -1, device=device, dtype=torch.int32)

        if self._triton_dda is not None and torch.cuda.is_available():
            # запуск Triton ядра
            BLOCK = 256
            grid = ( (N + BLOCK - 1) // BLOCK, )
            self._triton_dda[grid](
                Sx, Sy, Dx, Dy,
                env_map,
                cell_start, cell_count, cell_items,
                n_cells, M,
                float(origin[0].item()), float(origin[1].item()),
                float(cell),
                nx, ny,
                out.reshape(-1),
                N,
                KMAX=K,
                DDA_STEPS=self._dda_max,
                BLOCK=BLOCK,
            )
        else:
            # torch fallback: стартовая ячейка и первые K элементов (без прохода по трассе, но стабильно)
            invn = torch.rsqrt(torch.clamp(Dx*Dx + Dy*Dy, min=1e-12))
            Dx = Dx * invn; Dy = Dy * invn
            ix = ((Sx - origin[0]) / cell).floor().to(torch.int64).clamp(0, nx - 1)
            iy = ((Sy - origin[1]) / cell).floor().to(torch.int64).clamp(0, ny - 1)
            cid = (iy * nx + ix).to(torch.int64)  # (N,)

            # берем элементы текущей ячейки: [start, start+count)
            starts = torch.gather(
                self._grid["cell_start"].index_select(0, env_map.to(torch.long)),  # (N, n_cells)
                1, cid.view(-1, 1)
            ).squeeze(1).to(torch.int64)
            counts = torch.gather(
                self._grid["cell_count"].index_select(0, env_map.to(torch.long)),
                1, cid.view(-1, 1)
            ).squeeze(1).to(torch.int64)

            ar = torch.arange(K, device=device, dtype=torch.int64).view(1, K)
            take = (ar < counts.view(-1, 1))
            offs = (starts.view(-1, 1) + ar).clamp_max(self._M - 1)

            items = torch.gather(
                self._grid["cell_items"].index_select(0, env_map.to(torch.long)),
                1, offs
            ).to(torch.int32)
            out = torch.where(take, items, torch.full_like(items, -1))

        return out.view(B, Rb, K).to(torch.long).clamp_(-1, M - 1)

    # ---------------- основной апдейт ----------------
    @torch.no_grad()
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        dev = self._torch_device
        cuda_ok = (dev.type == "cuda" and torch.cuda.is_available())
        use_amp = bool(self._use_fp16 and cuda_ok)

        # поза сенсора (как в базовом)
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

        # ray alignment
        if self.cfg.attach_yaw_only is not None:
            self.cfg.ray_alignment = "yaw" if self.cfg.attach_yaw_only else "base"

        if self.cfg.ray_alignment == "world":
            pos_w[:, 0:2] += self.ray_cast_drift[env_ids, 0:2]
            ray_starts_w = self.ray_starts[env_ids] + pos_w.unsqueeze(1)
            ray_dirs_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "yaw":
            pos_w[:, 0:2] += quat_apply_yaw(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_dirs_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "base":
            pos_w[:, 0:2] += quat_apply(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_dirs_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        else:
            raise RuntimeError(f"[RayCasterMultiMesh] Unsupported ray_alignment: {self.cfg.ray_alignment}")

        B, R, _ = ray_starts_w.shape

        # препятствия + сетка
        self._update_obstacles_and_grid(force_regrid=False)

        # подготовка типов/констант
        out_dtype = self._data.ray_hits_w.dtype
        comp_dtype = torch.float16 if use_amp else torch.float32
        radius = torch.as_tensor(self._cyl_radius, device=dev, dtype=comp_dtype)
        half_h = torch.as_tensor(self._cyl_height * 0.5, device=dev, dtype=comp_dtype)
        max_d = torch.as_tensor(self.cfg.max_distance, device=dev, dtype=comp_dtype)
        eps = torch.as_tensor(1e-8, device=dev, dtype=comp_dtype)
        M = int(self._M)

        # кэш препятствий для текущих env
        env_ids_long = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
        C_env = self._C_all.index_select(0, env_ids_long).to(comp_dtype)  # (B,M,3)
        U_env = self._U_all.index_select(0, env_ids_long).to(comp_dtype)
        V_env = self._V_all.index_select(0, env_ids_long).to(comp_dtype)
        W_env = self._W_all.index_select(0, env_ids_long).to(comp_dtype)

        # итог
        best_hits = torch.full((B, R, 3), float("inf"), device=dev, dtype=out_dtype)

        # сохраним env_ids для сборщика кандидатов
        self._curr_env_ids_tensor = torch.as_tensor(env_ids, device=dev, dtype=torch.int32)

        # тайлинг по лучам
        block_R = self._block_R if (0 < self._block_R <= R) else min(R, 2048)
        ray_blocks: List[Tuple[int, int]] = [(i, min(i + block_R, R)) for i in range(0, R, block_R)]

        for r0, r1 in ray_blocks:
            Rb = r1 - r0
            S = ray_starts_w[:, r0:r1, :].to(device=dev, dtype=comp_dtype)  # (B,Rb,3)
            D = ray_dirs_w[:,   r0:r1, :].to(device=dev, dtype=comp_dtype)  # (B,Rb,3)

            # нормализация направлений
            D = D / torch.linalg.norm(D, dim=-1, keepdim=True).clamp_min_(eps)

            # широкая фаза (DDA)
            cand = self._gather_candidates(S[..., :2].float(), D[..., :2].float())  # (B,Rb,K) long | -1
            K = cand.shape[-1]
            mask_valid = cand >= 0
            cand = cand.clamp_(min=0, max=M - 1)

            # собрать C/U/V/W по кандидатам
            # (B,M,3) -> (B,Rb,M,3) -> take_along_dim по dim=2
            C_exp = C_env.unsqueeze(1).expand(B, Rb, M, 3)
            U_exp = U_env.unsqueeze(1).expand(B, Rb, M, 3)
            V_exp = V_env.unsqueeze(1).expand(B, Rb, M, 3)
            W_exp = W_env.unsqueeze(1).expand(B, Rb, M, 3)

            idx4 = cand.unsqueeze(-1).unsqueeze(-1).expand(B, Rb, K, 1, 3)

            Ck = torch.take_along_dim(C_exp, idx4, dim=2).squeeze(3)  # (B,Rb,K,3)
            Uk = torch.take_along_dim(U_exp, idx4, dim=2).squeeze(3)
            Vk = torch.take_along_dim(V_exp, idx4, dim=2).squeeze(3)
            Wk = torch.take_along_dim(W_exp, idx4, dim=2).squeeze(3)

            # узкая фаза (аналитика)
            Srel = S.unsqueeze(2) - Ck  # (B,Rb,K,3)

            Su = torch.sum(Srel * Uk, dim=-1)    # (B,Rb,K)
            Sv = torch.sum(Srel * Vk, dim=-1)
            Sw = torch.sum(Srel * Wk, dim=-1)

            Du = torch.sum(D.unsqueeze(2) * Uk, dim=-1)  # (B,Rb,K)
            Dv = torch.sum(D.unsqueeze(2) * Vk, dim=-1)
            Dw = torch.sum(D.unsqueeze(2) * Wk, dim=-1)

            a = Du*Du + Dv*Dv
            b = 2.0 * (Su*Du + Sv*Dv)
            c_q = Su*Su + Sv*Sv - (radius*radius)

            a = torch.where(a.abs() < eps, eps, a)
            disc = b*b - 4.0*a*c_q

            INF = torch.as_tensor(float("inf"), device=dev, dtype=comp_dtype)
            sqrt_disc = torch.zeros_like(disc)
            pos_disc = disc >= 0
            sqrt_disc[pos_disc] = torch.sqrt(disc[pos_disc])

            t1 = (-b - sqrt_disc) / (2.0*a)
            t2 = (-b + sqrt_disc) / (2.0*a)
            t1 = torch.where(t1 > 0.0, t1, INF)
            t2 = torch.where(t2 > 0.0, t2, INF)

            z1 = Sw + t1*Dw
            z2 = Sw + t2*Dw
            ok1 = (z1 >= -half_h) & (z1 <= half_h)
            ok2 = (z2 >= -half_h) & (z2 <= half_h)

            t_side = torch.minimum(torch.where(ok1, t1, INF),
                                   torch.where(ok2, t2, INF))

            Dw_safe = torch.where(Dw.abs() < eps, eps, Dw)
            t_top = ( half_h - Sw) / Dw_safe
            t_bot = (-half_h - Sw) / Dw_safe
            t_top = torch.where(t_top > 0.0, t_top, INF)
            t_bot = torch.where(t_bot > 0.0, t_bot, INF)
            ok_top = (Su + t_top*Du)**2 + (Sv + t_top*Dv)**2 <= (radius*radius)
            ok_bot = (Su + t_bot*Du)**2 + (Sv + t_bot*Dv)**2 <= (radius*radius)

            t_caps = torch.minimum(torch.where(ok_top, t_top, INF),
                                   torch.where(ok_bot, t_bot, INF))

            t_k = torch.minimum(t_side, t_caps)                  # (B,Rb,K)
            t_k = torch.where(mask_valid, t_k, INF)
            t_k = torch.where(t_k <= max_d, t_k, INF)

            best_t_rb, _ = torch.min(t_k, dim=2)                 # (B,Rb)

            # мировая точка: P = S + t*D
            P = (S + best_t_rb.unsqueeze(-1) * D).to(dtype=out_dtype)
            best_hits[:, r0:r1, :] = P

        # Z-дрифт, как в базовой реализации
        best_hits[:, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1).to(dtype=out_dtype)
        self._data.ray_hits_w[env_ids] = best_hits

    # ---------------- debug vis ----------------
    def _debug_vis_callback(self, event):
        pts = self._data.ray_hits_w.reshape(-1, 3)
        finite = torch.isfinite(pts).all(dim=1)
        pts = pts[finite]
        if pts.numel() == 0:
            return
        self.ray_visualizer.visualize(pts)

