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
import triton
import triton.language as tl
# ---------------------------------------------------------------------
# Optional Triton (CUDA kernels). If not present, we’ll fallback.
# ---------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _ensure_triton_imports():
    # чтобы не падать, если модуль ещё не импортирован
    global triton, tl
    import triton
    import triton.language as tl
    return triton, tl

def _triton_const_i32(x: int) -> int:
    # Triton constexpr передаются как Python-литералы / kwargs; просто возвращаем int
    return int(x)

def _as_torch_contig(x, dtype=None):
    if dtype is not None and x.dtype != dtype:
        x = x.to(dtype)
    return x.contiguous()

def _as_int32(x):
    return _as_torch_contig(x, dtype=torch.int32)

def _as_fp32(x):
    return _as_torch_contig(x, dtype=torch.float32)

def _as_fp16(x):
    return _as_torch_contig(x, dtype=torch.float16)


def _gather_candidates(self, Sxy: torch.Tensor, Dxy: torch.Tensor) -> torch.Tensor:
    """
    Triton-ускоренный сбор кандидатов для каждого луча плитки:
      вход: Sxy, Dxy формы (B, Rb, 2), dtype=float32/float16
      выход: (B, Rb, K) long, значения в [0..M-1] или -1 (если меньше кандидатов)
    Требует, чтобы self._grid был построен (см. _update_obstacles_and_grid).
    """
    # импорт ядра (лениво)
    kernel = getattr(self, "_dda_kernel", None)
    if kernel is None:
        self._dda_kernel = _build_dda_kernel()
        kernel = self._dda_kernel

    dev = Sxy.device
    B, Rb, _ = Sxy.shape
    N = B * Rb
    K = int(self.cfg.kmax)

    # Плоские буферы входа
    Sx = _as_fp32(Sxy[..., 0].reshape(-1))
    Sy = _as_fp32(Sxy[..., 1].reshape(-1))
    Dx = _as_fp32(Dxy[..., 0].reshape(-1))
    Dy = _as_fp32(Dxy[..., 1].reshape(-1))

    # Для каждой строки (env в батче) сопоставим env_id
    # Простой вариант: идём по порядку env-ов в текущем батче 0..B-1,
    # но нам нужен их глобальный индекс в [0..E_total-1].
    # В _update_buffers_impl мы вызываем для подмножества env_ids — они уже есть на CPU.
    # Поэтому, чтобы не протягивать env_ids сюда каждый раз, закешируем их в self._curr_env_ids.
    env_ids = getattr(self, "_curr_env_ids_tensor", None)
    if env_ids is None or env_ids.numel() != B:
        raise RuntimeError("[RayCasterMultiMesh] Internal: _curr_env_ids_tensor is not set or has wrong shape.")
    env_ids = env_ids.to(device=dev, dtype=torch.int32)          # (B,)
    env_id_flat = env_ids.view(B, 1).expand(B, Rb).reshape(-1)   # (N,) int32

    # Грид-структуры
    grid = self._grid
    if grid is None or grid["cell_start"] is None:
        raise RuntimeError("[RayCasterMultiMesh] Grid is not built.")

    cell_start = _as_int32(grid["cell_start"].reshape(-1))   # (E*n_cells,)
    cell_count = _as_int32(grid["cell_count"].reshape(-1))   # (E*n_cells,)
    cell_items = _as_int32(grid["cell_items"].reshape(-1))   # (E*M,)
    nx, ny = grid["shape"]
    n_cells = int(nx * ny)
    M = int(self._M)
    origin = grid["origin"].to(device=dev, dtype=torch.float32)
    cell_size = float(grid["cell"])

    # Выход
    out = torch.full((N, K), -1, device=dev, dtype=torch.int32)

    # Запуск Triton
    grid_1d = (N,)
    self._dda_kernel[grid_1d](
        Sx, Sy, Dx, Dy,
        env_id_flat,
        cell_start, cell_count,
        cell_items,
        n_cells, M,
        origin[0].item(), origin[1].item(),
        cell_size,
        nx, ny,
        out,
        KMAX=_triton_const_i32(K),
        DDA_STEPS=_triton_const_i32(int(self.cfg.dda_max_cells)),
    )

    # Приводим к (B,Rb,K) и к типу long; гарантируем диапазон [ -1 | 0..M-1 ]
    out = out.view(B, Rb, K).to(torch.long)
    out = torch.where(out >= 0, out.clamp_(0, M - 1), out)  # -1 оставляем как "пусто"
    return out


def _build_dda_kernel():
    triton, tl = _ensure_triton_imports()

    @triton.jit
    def _dda_candidates_kernel(
        Sx_ptr, Sy_ptr, Dx_ptr, Dy_ptr,          # (N,) float32 — старт/напр. лучей по XY
        env_id_ptr,                               # (N,) int32   — id окружения для каждого луча
        cell_start_ptr, cell_count_ptr,           # (E, n_cells) int32 → плоские буферы
        cell_items_ptr,                           # (E, M) int32 → плоский буфер
        n_cells, M,                               # int32        — количество ячеек и препятствий на env
        origin_x, origin_y,                       # float32      — начало сетки
        cell_size,                                # float32      — размер ячейки
        nx, ny,                                   # int32        — размер сетки по осям
        out_cand_ptr,                             # (N, KMAX) int32 — выход
        KMAX: tl.constexpr,                       # compile-time
        DDA_STEPS: tl.constexpr,                  # compile-time
    ):
        pid = tl.program_id(0)  # 0..N-1

        # вход
        Sx = tl.load(Sx_ptr + pid)
        Sy = tl.load(Sy_ptr + pid)
        Dx = tl.load(Dx_ptr + pid)
        Dy = tl.load(Dy_ptr + pid)
        e  = tl.load(env_id_ptr + pid).to(tl.int32)

        # нормировка направления
        inv_len = tl.rsqrt(Dx * Dx + Dy * Dy + 1e-12)
        Dx = Dx * inv_len
        Dy = Dy * inv_len

        # начальная ячейка
        fx = (Sx - origin_x) / cell_size
        fy = (Sy - origin_y) / cell_size
        ix = tl.astype(tl.math.floor(fx), tl.int32)
        iy = tl.astype(tl.math.floor(fy), tl.int32)
        ix = tl.minimum(tl.maximum(ix, 0), nx - 1)
        iy = tl.minimum(tl.maximum(iy, 0), ny - 1)

        # шаги/границы DDA
        step_x = tl.where(Dx >= 0.0, 1, -1)
        step_y = tl.where(Dy >= 0.0, 1, -1)

        cell_x = origin_x + (tl.astype(ix, tl.float32) + tl.where(Dx >= 0.0, 1.0, 0.0)) * cell_size
        cell_y = origin_y + (tl.astype(iy, tl.float32) + tl.where(Dy >= 0.0, 1.0, 0.0)) * cell_size

        t_max_x = tl.where(Dx != 0.0, (cell_x - Sx) / Dx, 1e30)
        t_max_y = tl.where(Dy != 0.0, (cell_y - Sy) / Dy, 1e30)
        t_delta_x = tl.where(Dx != 0.0, cell_size / tl.abs(Dx), 1e30)
        t_delta_y = tl.where(Dy != 0.0, cell_size / tl.abs(Dy), 1e30)

        # выходной буфер кандидатов
        out_row = out_cand_ptr + pid * KMAX
        # инициализация -1
        for k in range(KMAX):
            tl.store(out_row + k, tl.full((), -1, tl.int32))
        out_count = tl.zeros((), tl.int32)

        # базовые офсеты для CSR по окружению e
        cs_base = e * n_cells
        it_base = e * M

        # DDA
        for _ in range(DDA_STEPS):
            ix = tl.minimum(tl.maximum(ix, 0), nx - 1)
            iy = tl.minimum(tl.maximum(iy, 0), ny - 1)
            cid = iy * nx + ix

            cs = tl.load(cell_start_ptr + cs_base + cid)
            cc = tl.load(cell_count_ptr + cs_base + cid)

            i = tl.zeros((), tl.int32)
            while (i < cc) & (out_count < KMAX):
                itm = tl.load(cell_items_ptr + it_base + cs + i)
                if itm >= 0:
                    tl.store(out_row + out_count, itm)
                    out_count += 1
                i += 1

            do_x = t_max_x <= t_max_y
            ix += tl.where(do_x, step_x, 0)
            iy += tl.where(do_x, 0,      step_y)
            t_max_x = tl.where(do_x, t_max_x + t_delta_x, t_max_x)
            t_max_y = tl.where(do_x, t_max_y,              t_max_y + t_delta_y)

            if out_count >= KMAX:
                break

    return _dda_candidates_kernel


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@configclass
class RayCasterMultiMeshCfg(RayCasterCfg):
    # geometry (must)
    cylinder_radius: float = 0.4
    cylinder_height: float = 1.2

    # perf
    use_fp16: bool = True           # AMP on CUDA
    block_R: int = 1024             # rays per tile (narrow-phase kernel launch size)
    kmax: int = 8                   # candidates per ray after broad phase
    dda_max_cells: int = 48         # DDA steps cap

    # grid
    grid_cell: float = 0.9          # ~ 2*r (or larger)
    grid_pad: float  = 2.0          # padding (cells)

    # regrid sensitivity (in cells)
    regrid_trigger_cells: float = 1.0


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _torch_device(dev) -> torch.device:
    return dev if isinstance(dev, torch.device) else torch.device(dev)

def _collect_env0_obst_slots() -> List[str]:
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
    # returns column vectors u,v,w of rotation matrix from unit quat (wxyz)
    w, x, y, z = q.unbind(-1)
    ww = w*w; xx = x*x; yy = y*y; zz = z*z
    wx = w*x; wy = w*y; wz = w*z
    xy = x*y; xz = x*z; yz = y*z
    u = torch.stack([1 - 2*(yy + zz), 2*(xy + wz),       2*(xz - wy)], dim=-1)
    v = torch.stack([2*(xy - wz),     1 - 2*(xx + zz),   2*(yz + wx)], dim=-1)
    w_ = torch.stack([2*(xz + wy),    2*(yz - wx),       1 - 2*(xx + yy)], dim=-1)
    return u, v, w_


# ---------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------
if _HAS_TRITON:

    

    @triton.jit
    def _narrow_phase_kernel(
        # rays (B*Rb)
        S_ptr, D_ptr,                                  # float16/32, shape (N, 3)
        # candidates per ray
        cand_ptr,                                      # int32, shape (N, K)
        # obstacles per env (already per-ray env-aligned views!)
        C_ptr, U_ptr, V_ptr, W_ptr,                   # float16/32, shape (N, M, 3) logically but we gather by cands
        M: tl.constexpr, K: tl.constexpr,
        # cylinder geom
        radius, half_h, max_d,
        # outputs
        out_ptr,                                      # float32, shape (N, 3)
        # sizes
        N: tl.constexpr
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        # load S, D
        Sx = tl.load(S_ptr + pid*3 + 0)
        Sy = tl.load(S_ptr + pid*3 + 1)
        Sz = tl.load(S_ptr + pid*3 + 2)
        Dx = tl.load(D_ptr + pid*3 + 0)
        Dy = tl.load(D_ptr + pid*3 + 1)
        Dz = tl.load(D_ptr + pid*3 + 2)
        invn = tl.rsqrt(tl.maximum(Dx*Dx + Dy*Dy + Dz*Dz, 1e-16))
        Dx *= invn; Dy *= invn; Dz *= invn

        best_t = 1e30
        # iterate over K candidates
        for j in tl.static_range(0, K):
            idx = tl.load(cand_ptr + pid*K + j)
            if (idx < 0) | (idx >= M):
                continue
            # gather cylinder center/axes
            Cx = tl.load(C_ptr + pid*M*3 + idx*3 + 0)
            Cy = tl.load(C_ptr + pid*M*3 + idx*3 + 1)
            Cz = tl.load(C_ptr + pid*M*3 + idx*3 + 2)

            Ux = tl.load(U_ptr + pid*M*3 + idx*3 + 0)
            Uy = tl.load(U_ptr + pid*M*3 + idx*3 + 1)
            Uz = tl.load(U_ptr + pid*M*3 + idx*3 + 2)

            Vx = tl.load(V_ptr + pid*M*3 + idx*3 + 0)
            Vy = tl.load(V_ptr + pid*M*3 + idx*3 + 1)
            Vz = tl.load(V_ptr + pid*M*3 + idx*3 + 2)

            Wx = tl.load(W_ptr + pid*M*3 + idx*3 + 0)
            Wy = tl.load(W_ptr + pid*M*3 + idx*3 + 1)
            Wz = tl.load(W_ptr + pid*M*3 + idx*3 + 2)

            # S_rel = S - C
            Rx = Sx - Cx
            Ry = Sy - Cy
            Rz = Sz - Cz

            # projections (Su,Sv,Sw) = dot(S_rel, U/V/W)
            Su = Rx*Ux + Ry*Uy + Rz*Uz
            Sv = Rx*Vx + Ry*Vy + Rz*Vz
            Sw = Rx*Wx + Ry*Wy + Rz*Wz
            Du = Dx*Ux + Dy*Uy + Dz*Uz
            Dv = Dx*Vx + Dy*Vy + Dz*Vz
            Dw = Dx*Wx + Dy*Wy + Dz*Wz

            # side quad: (Su + t Du)^2 + (Sv + t Dv)^2 = r^2
            a = Du*Du + Dv*Dv
            b = 2.0*(Su*Du + Sv*Dv)
            c = Su*Su + Sv*Sv - radius*radius
            a = tl.where(tl.abs(a) < 1e-12, 1e-12, a)
            disc = b*b - 4.0*a*c

            t_side = 1e30
            if disc >= 0.0:
                sdisc = tl.sqrt(disc)
                t1 = (-b - sdisc) / (2.0*a)
                t2 = (-b + sdisc) / (2.0*a)
                t1 = tl.where(t1 > 0.0, t1, 1e30)
                t2 = tl.where(t2 > 0.0, t2, 1e30)
                z1 = Sw + t1*Dw
                z2 = Sw + t2*Dw
                in1 = (z1 >= -half_h) & (z1 <= half_h)
                in2 = (z2 >= -half_h) & (z2 <= half_h)
                t_side = tl.minimum(tl.where(in1, t1, 1e30), tl.where(in2, t2, 1e30))

            # caps
            Dw_safe = tl.where(tl.abs(Dw) < 1e-12, tl.copysign(1e-12, Dw), Dw)
            tt = ( half_h - Sw) / Dw_safe
            tb = (-half_h - Sw) / Dw_safe
            tt = tl.where(tt > 0.0, tt, 1e30)
            tb = tl.where(tb > 0.0, tb, 1e30)
            in_top = (Su + tt*Du)*(Su + tt*Du) + (Sv + tt*Dv)*(Sv + tt*Dv) <= radius*radius
            in_bot = (Su + tb*Du)*(Su + tb*Du) + (Sv + tb*Dv)*(Sv + tb*Dv) <= radius*radius
            t_caps = tl.minimum(tl.where(in_top, tt, 1e30), tl.where(in_bot, tb, 1e30))

            t = tl.minimum(t_side, t_caps)
            t = tl.where(t <= max_d, t, 1e30)
            best_t = tl.minimum(best_t, t)

        # write world hit = S + best_t * D  (float32)
        Px = (Sx + best_t*Dx).to(tl.float32)
        Py = (Sy + best_t*Dy).to(tl.float32)
        Pz = (Sz + best_t*Dz).to(tl.float32)
        tl.store(out_ptr + pid*3 + 0, Px)
        tl.store(out_ptr + pid*3 + 1, Py)
        tl.store(out_ptr + pid*3 + 2, Pz)


# ---------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------
class RayCasterMultiMesh(RayCaster):
    """
    CUDA (Triton) ray caster for many tilted cylinders per env:
      * single PhysX view for '/.../envs/env_*/obst_*'
      * GPU CSR grid + DDA broad-phase (kernel)
      * Narrow-phase analytic intersection (kernel)
      * Fully batched, tiled by rays; AMP/FP16 enabled
    """

    def _initialize_warp_meshes(self):
        log = omni.log

        slots = _collect_env0_obst_slots()
        if len(slots) == 0:
            raise RuntimeError(
                "[RayCasterMultiMesh] No '/envs/env_0/obst_XX' found. "
                "Ensure obstacles '.../envs/env_#/obst_##' exist."
            )
        self._M = len(slots)

        # glob pattern for PhysX view (no regex)
        prefix = slots[0].split("/envs/env_0")[0]
        self._rb_glob_pattern = f"{prefix}/envs/env_*/obst_*"

        # geometry
        r = getattr(self.cfg, "cylinder_radius", None)
        h = getattr(self.cfg, "cylinder_height", None)
        if r is None or h is None:
            raise RuntimeError(
                "[RayCasterMultiMesh] Please set cfg.cylinder_radius and cfg.cylinder_height (e.g., 0.4 / 1.2)."
            )
        self._cyl_radius = float(r)
        self._cyl_height = float(h)

        # perf flags
        self._use_fp16 = bool(getattr(self.cfg, "use_fp16", True))
        self._block_R = int(getattr(self.cfg, "block_R", 1024))
        self._kmax = int(getattr(self.cfg, "kmax", 8))
        self._dda_max = int(getattr(self.cfg, "dda_max_cells", 48))

        # grid params
        self._grid_cell = float(getattr(self.cfg, "grid_cell", 0.9))
        self._grid_pad = float(getattr(self.cfg, "grid_pad", 2.0))
        self._regrid_cells = float(getattr(self.cfg, "regrid_trigger_cells", 1.0))

        # device
        self._torch_device: torch.device = _torch_device(self.device)

        # lazy PhysX view
        self._rb_view_all: Optional[physx.RigidBodyView] = None
        self._E_total: Optional[int] = None

        # grid buffers
        self._grid = None   # dict with: origin(2), cell(float), shape(nx,ny), cell_start(E,nc), cell_count(E,nc), cell_items(E,M)
        self._grid_dirty = True
        self._last_grid_origin = None

        # cached obstacles (E,M,3) fp16/32
        self._C_all = None
        self._U_all = None
        self._V_all = None
        self._W_all = None

        # PhysX sim view
        if not hasattr(self, "_physics_sim_view"):
            from isaacsim.core.simulation_manager import SimulationManager
            self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # cuda knobs
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        engine = "triton" if _HAS_TRITON and torch.cuda.is_available() else "torch"
        log.info(
            f"[RayCasterMultiMesh] CUDA LIDAR ({engine}). "
            f"M={self._M}, r={self._cyl_radius:.3f}, h={self._cyl_height:.3f}, "
            f"kmax={self._kmax}, block_R={self._block_R}, dda_max={self._dda_max}."
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
            cell_items=torch.full((E, M), -1, device=dev, dtype=torch.int32),  # CSR payload space
        )

    # ---------------- grid + obstacles ----------------
    def _update_obstacles_and_grid(self, force_regrid: bool = False):
        """
        Читает позы всех препятствий (E,M,7), кэширует центры/оси.
        При необходимости перестраивает сетку для широкой фазы (CSR).
        """
        self._ensure_rb_view()
        self._ensure_grid()
        dev = self._torch_device

        # ---- прочитать трансформы препятствий ----
        tf_all = self._rb_view_all.get_transforms().to(device=dev, dtype=torch.float32)  # (E*M, 7)
        tf_all = tf_all.view(self._E_total, self._M, 7)
        C = tf_all[..., 0:3]                          # (E,M,3)
        q = convert_quat(tf_all[..., 3:7], to="wxyz") # (E,M,4)
        u, v, w_axis = _quat_to_axes_wxyz(q)          # (E,M,3) * 3

        # кэш (полупрецизионный при необходимости)
        want_half = self._use_fp16 and torch.cuda.is_available()
        dtype_cache = torch.float16 if want_half else torch.float32
        self._C_all = C.to(dtype_cache)
        self._U_all = u.to(dtype_cache)
        self._V_all = v.to(dtype_cache)
        self._W_all = w_axis.to(dtype_cache)

        # ---- bbox для решётки ----
        Cx = C[..., 0]
        Cy = C[..., 1]

        cell = float(self._grid_cell)
        xmin = torch.min(Cx) - self._grid_pad * cell
        xmax = torch.max(Cx) + self._grid_pad * cell
        ymin = torch.min(Cy) - self._grid_pad * cell
        ymax = torch.max(Cy) + self._grid_pad * cell
    
        origin = torch.stack([xmin, ymin]).to(device=dev, dtype=torch.float32)
        nx = int(torch.clamp(((xmax - xmin) / cell).ceil(), min=1).item())
        ny = int(torch.clamp(((ymax - ymin) / cell).ceil(), min=1).item())
        n_cells = nx * ny

        # редко перестраиваем сетку, если сдвиг небольшой
        if (not force_regrid) and (self._last_grid_origin is not None):
            delta_cells = torch.max(torch.abs((origin - self._last_grid_origin) / cell)).item()
            if delta_cells < self.cfg.regrid_trigger_cells and self._grid["cell_start"].shape[1] == n_cells:
                return

        self._last_grid_origin = origin.clone()

        # ---- рассадка препятствий по клеткам, построение CSR ----
        ix = ((Cx - origin[0]) / cell).floor().to(torch.int64).clamp(0, nx - 1)  # (E,M) long
        iy = ((Cy - origin[1]) / cell).floor().to(torch.int64).clamp(0, ny - 1)  # (E,M) long
        cid = (iy * nx + ix).to(torch.int64)                                     # (E,M) long  [0..n_cells)

        # counts per cell
        cell_count = torch.zeros((self._E_total, n_cells), device=dev, dtype=torch.int32)
        cell_count.scatter_add_(1, cid, torch.ones_like(cid, dtype=torch.int32))   # индексы long — ок

        # exclusive prefix-sum -> starts
        cell_start = torch.zeros_like(cell_count)
        torch.cumsum(cell_count, dim=1, out=cell_start)
        cell_start -= cell_count

        # payload: ровно M записей на окружение
        cell_items = torch.full((self._E_total, self._M), -1, device=dev, dtype=torch.int32)

        # позиции записи (будем инкрементировать)
        write_pos = cell_start.clone()  # (E, n_cells) int32

        # offsets для каждой записи (E,M) — ВАЖНО: индексы long
        offs_int32 = torch.gather(write_pos, 1, cid)               # (E,M) int32
        offs = offs_int32.to(torch.int64)                          # -> long для scatter_

        # инкремент позиций записи
        write_pos.scatter_add_(1, cid, torch.ones_like(cid, dtype=torch.int32))
    
        # сами id препятствий
        item_ids = torch.arange(self._M, device=dev, dtype=torch.int32).unsqueeze(0).expand_as(cid)

        # запись в payload; индексы должны быть long
        # (offs гарантированно < M, но на всякий случай ограничим)
        cell_items.scatter_(1, offs.clamp_max(self._M - 1), item_ids)

        # ---- commit в структуру гридов ----
        self._grid["origin"] = origin
        self._grid["cell"] = cell
        self._grid["shape"] = (nx, ny)
        self._grid["cell_start"] = cell_start          # (E, n_cells) int32
        self._grid["cell_count"] = cell_count          # (E, n_cells) int32
        self._grid["cell_items"] = cell_items          # (E, M) int32

    # ---------------- main update ----------------
    @torch.no_grad()
    def _update_buffers_impl(self, env_ids: Sequence[int]):

        dev = self._torch_device
        cuda_ok = (dev.type == "cuda" and torch.cuda.is_available())
        use_amp = bool(self._use_fp16 and cuda_ok)

        # --- sensor pose (same as base) ---
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

        # --- obstacles + grid ---
        self._update_obstacles_and_grid(force_regrid=False)

        # typed constants
        out_dtype = self._data.ray_hits_w.dtype
        comp_dtype = torch.float16 if use_amp else torch.float32

        # flatten per-ray views
        # env index per ray (B,R) -> (N)
        env_idx = torch.as_tensor(env_ids, device=dev, dtype=torch.int32)
        env_map = env_idx.view(-1, 1).repeat(1, R).contiguous().view(-1)  # (N=B*R)

        # ray buffers flattened
        S_all = ray_starts_w.reshape(-1, 3).to(device=dev, dtype=comp_dtype)  # (N,3)
        D_all = ray_dirs_w.reshape(-1, 3).to(device=dev, dtype=comp_dtype)    # (N,3)
        N_tot = S_all.shape[0]

        # candidate buffer (N, K) int32
        K = self._kmax
        cand = torch.full((N_tot, K), -1, device=dev, dtype=torch.int32)

        # --- DDA broad phase ---
        nx, ny = self._grid["shape"]
        n_cells = nx * ny
        origin = self._grid["origin"]
        cell = float(self._grid["cell"])
        cell_start = self._grid["cell_start"]
        cell_count = self._grid["cell_count"]
        cell_items = self._grid["cell_items"]

        if _HAS_TRITON and cuda_ok:
            # pack XY and launch kernel
            Sx = S_all[:, 0].contiguous()
            Sy = S_all[:, 1].contiguous()
            Dx = D_all[:, 0].contiguous()
            Dy = D_all[:, 1].contiguous()
            grid = lambda META: (triton.cdiv(N_tot, META['BLOCK']),)
            _dda_candidates_kernel[grid](  # type: ignore
                Sx, Sy, Dx, Dy,
                origin[0].item(), origin[1].item(), cell, nx, ny,
                cell_start.reshape(-1), cell_count.reshape(-1), cell_items.reshape(-1),
                cand.reshape(-1),
                env_map,  # per-ray env
                K, self._dda_max,
                N_tot, n_cells,
                BLOCK=256
            )
        else:
            # fallback: simple per-ray cell lookup (vectorized)
            # Compute starting cell and take first K items in that cell.
            Sxy = S_all[:, :2]
            Dxy = D_all[:, :2]
            invn = torch.rsqrt(torch.clamp((Dxy*Dxy).sum(-1, keepdim=True), min=1e-12))
            Dxy = Dxy * invn
            ix = ((Sxy[:, 0] - origin[0]) / cell).floor().to(torch.int64).clamp(0, nx-1)
            iy = ((Sxy[:, 1] - origin[1]) / cell).floor().to(torch.int64).clamp(0, ny-1)
            cid = (iy * nx + ix)  # (N)
            # gather [start,count] per ray/env
            start = torch.gather(cell_start, 1, cid.view(N_tot,1)).squeeze(1)
            count = torch.gather(cell_count, 1, cid.view(N_tot,1)).squeeze(1)
            # fetch up to K items
            offs = start.view(-1,1) + torch.arange(K, device=dev, dtype=torch.int64).view(1,-1)
            mask = (torch.arange(K, device=dev).view(1,-1) < count.view(-1,1))
            items_flat = torch.gather(cell_items, 1, offs.clamp_max(self._M-1))
            cand = torch.where(mask, items_flat, torch.full_like(items_flat, -1))

        # --- narrow phase ---
        # materialize per-ray per-candidate obstacle arrays via gather:
        # We want tensors shaped (N, M, 3) logically, but we only use K cands; we’ll build (N, M,3) “views” by scatter-gather trick:
        # Simplify: repeat per-env arrays for each ray in that env, then gather by cand.

        # build per-ray aligned (N, M, 3) by indexing env first (no data copy thanks to view+expand)
        # E,M,3 -> N,M,3
        E = self._E_total
        M = self._M
        # map rays -> env rows
        C_env = self._C_all  # (E,M,3)
        U_env = self._U_all
        V_env = self._V_all
        W_env = self._W_all

        # select rows
        env_long = env_map.to(torch.long)
        C_sel = C_env.index_select(0, env_long)   # (N, M, 3)
        U_sel = U_env.index_select(0, env_long)
        V_sel = V_env.index_select(0, env_long)
        W_sel = W_env.index_select(0, env_long)

        # flatten for kernel (row-major)
        C_flat = C_sel.reshape(-1, 3).contiguous()
        U_flat = U_sel.reshape(-1, 3).contiguous()
        V_flat = V_sel.reshape(-1, 3).contiguous()
        W_flat = W_sel.reshape(-1, 3).contiguous()

        # We will address C/U/V/W as (N, M, 3) flattened in kernel by pid*M*3 + idx*3

        hits = torch.empty((N_tot, 3), device=dev, dtype=torch.float32)

        radius = float(self._cyl_radius)
        half_h = float(self._cyl_height * 0.5)
        max_d = float(self.cfg.max_distance)

        if _HAS_TRITON and cuda_ok:
            grid = lambda META: (triton.cdiv(N_tot, META['BLOCK']),)
            _narrow_phase_kernel[grid](   # type: ignore
                S_all.reshape(-1), D_all.reshape(-1),
                cand.reshape(-1),
                C_flat.reshape(-1), U_flat.reshape(-1), V_flat.reshape(-1), W_flat.reshape(-1),
                M, K,
                radius, half_h, max_d,
                hits.reshape(-1),
                N_tot,
                BLOCK=256
            )
        else:
            # fallback torch vector (batched by K)
            # gather C/U/V/W by candidates
            cand_clamped = cand.clamp(min=0, max=M-1).to(torch.long)  # (N,K)
            idx3 = cand_clamped.unsqueeze(-1).expand(N_tot, K, 3)    # (N,K,3)
            Ck = torch.gather(C_sel, 1, idx3)
            Uk = torch.gather(U_sel, 1, idx3)
            Vk = torch.gather(V_sel, 1, idx3)
            Wk = torch.gather(W_sel, 1, idx3)

            S = S_all[:, None, :].expand(N_tot, K, 3)
            D = D_all[:, None, :].expand(N_tot, K, 3)
            D = D / torch.linalg.norm(D, dim=-1, keepdim=True).clamp_min_(1e-8)

            Srel = S - Ck
            Su = (Srel*Uk).sum(-1); Sv = (Srel*Vk).sum(-1); Sw = (Srel*Wk).sum(-1)
            Du = (D*Uk).sum(-1);    Dv = (D*Vk).sum(-1);    Dw = (D*Wk).sum(-1)

            a = Du*Du + Dv*Dv
            b = 2.0*(Su*Du + Sv*Dv)
            c_q = Su*Su + Sv*Sv - radius*radius
            a = torch.where(a.abs() < 1e-12, torch.full_like(a, 1e-12), a)
            disc = b*b - 4.0*a*c_q

            t_side = torch.full_like(disc, float("inf"))
            pos = disc >= 0
            sdisc = torch.zeros_like(disc); sdisc[pos] = torch.sqrt(disc[pos])
            t1 = (-b - sdisc)/(2.0*a); t2 = (-b + sdisc)/(2.0*a)
            t1 = torch.where(t1>0.0, t1, t_side); t2 = torch.where(t2>0.0, t2, t_side)
            z1 = Sw + t1*Dw; z2 = Sw + t2*Dw
            ok1 = (z1>=-half_h) & (z1<=half_h); ok2 = (z2>=-half_h) & (z2<=half_h)
            t_side = torch.minimum(torch.where(ok1, t1, t_side), torch.where(ok2, t2, t_side))

            Dw_safe = torch.where(Dw.abs()<1e-12, torch.sign(Dw)*1e-12, Dw)
            tt = ( half_h - Sw)/Dw_safe; tb = (-half_h - Sw)/Dw_safe
            tt = torch.where(tt>0.0, tt, t_side); tb = torch.where(tb>0.0, tb, t_side)
            in_top = (Su+tt*Du)**2 + (Sv+tt*Dv)**2 <= radius*radius
            in_bot = (Su+tb*Du)**2 + (Sv+tb*Dv)**2 <= radius*radius
            t_caps = torch.minimum(torch.where(in_top, tt, t_side), torch.where(in_bot, tb, t_side))

            t = torch.minimum(t_side, t_caps)
            # mask invalid candidates (-1)
            t = torch.where(cand>=0, t, torch.full_like(t, float("inf")))
            t = torch.where(t <= max_d, t, torch.full_like(t, float("inf")))
            tmin, _ = t.min(dim=1)  # (N)
            hits = S_all + tmin.unsqueeze(-1)*D_all
            hits = hits.to(torch.float32)

        # write back (B,R,3)
        best_hits = hits.view(B, R, 3)
        # add Z drift like base
        best_hits[:, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1).to(best_hits.dtype)
        self._data.ray_hits_w[env_ids] = best_hits.to(out_dtype)

    # ----------------------------------
    # debug vis
    # ----------------------------------
    def _debug_vis_callback(self, event):
        pts = self._data.ray_hits_w.reshape(-1, 3)
        finite = torch.isfinite(pts).all(dim=1)
        pts = pts[finite]
        if pts.numel() == 0:
            return
        self.ray_visualizer.visualize(pts)

