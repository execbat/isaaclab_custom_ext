# isaaclab_custom_ext/custom_sensors/ray_caster_multimesh.py
from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence, Tuple
from contextlib import nullcontext

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


# --------------------------------------------------------------------------------------
# utils
# --------------------------------------------------------------------------------------

def _autocast_ctx(enable: bool, dtype=torch.float16):
    """Вернёт корректный autocast-контекст под любую версию torch. На CPU -> nullcontext()."""
    if not enable or not torch.cuda.is_available():
        return nullcontext()
    # PyTorch >= 2.0
    try:
        return torch.amp.autocast("cuda", dtype=dtype)
    except Exception:
        # Старый API
        return torch.cuda.amp.autocast(dtype=dtype)


def _torch_device(dev) -> torch.device:
    return dev if isinstance(dev, torch.device) else torch.device(dev)


def _quat_to_axes_wxyz(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Быстро получить ортонормальные оси (u,v,w) из юнит-кватерниона (w,x,y,z).
    Возвращает тензоры формы (...,3) — столбцы матрицы вращения.
    """
    w, x, y, z = q.unbind(-1)
    ww = w*w; xx = x*x; yy = y*y; zz = z*z
    wx = w*x; wy = w*y; wz = w*z
    xy = x*y; xz = x*z; yz = y*z
    # u, v, w — столбцы R
    u = torch.stack([1 - 2*(yy + zz), 2*(xy + wz),       2*(xz - wy)], dim=-1)
    v = torch.stack([2*(xy - wz),     1 - 2*(xx + zz),   2*(yz + wx)], dim=-1)
    w_ = torch.stack([2*(xz + wy),    2*(yz - wx),       1 - 2*(xx + yy)], dim=-1)
    return u, v, w_


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------

@configclass
class RayCasterMultiMeshCfg(RayCasterCfg):
    # Геометрия цилиндров (обязательно)
    cylinder_radius: float = 0.4
    cylinder_height: float = 1.2

    # Производительность / память
    use_fp16: bool = True

    # Широкая фаза: равномерная сетка по XY + DDA
    grid_cell: float = 0.9     # ≈ 2*r + запас
    grid_pad: float  = 2.0     # запас по краю (в клетках ~ grid_pad*cell)
    dda_max_cells: int = 64    # макс. шагов DDA вдоль луча
    kmax: int = 16             # макс. кандидатов-препятствий на луч после широкофазы

    # Тайлинг
    block_R: int = 512         # лучей на плитку
    block_B: int = 64          # энвов на плитку (обычно можно не трогать)


# --------------------------------------------------------------------------------------
# main class
# --------------------------------------------------------------------------------------

class RayCasterMultiMesh(RayCaster):
    """
    Полностью векторный LIDAR по множеству **наклонных** цилиндров (obst_XX) на окружение.

    Архитектура:
    - Один PhysX RigidBodyView по '/.../envs/env_*/obst_*' → разом читаем (E,M,7).
    - Широкая фаза: 2D grid (XY) + DDA → на луч отбираем до K кандидатов, а не все M.
    - Узкая фаза: аналитическое пересечение луч–наклонный цилиндр (радиус/высота из cfg).
    - Всё — тензорно, без питон-циклов по M/R; только фиксированное число DDA-шагов и тайлинг по R.
    """

    # ----------------------------------
    # init
    # ----------------------------------

    def _initialize_warp_meshes(self):
        log = omni.log

        # Найдём конкретные слоты '/envs/env_0/obst_XX' — от них определим M и префикс
        self._slots_env0: List[str] = self._collect_env0_obst_slots()
        if len(self._slots_env0) == 0:
            raise RuntimeError(
                "[RayCasterMultiMesh] No '/envs/env_0/obst_XX' found. "
                "Ensure obstacles exist and are named '.../envs/env_#/obst_##'."
            )
        self._M: int = len(self._slots_env0)

        # Глоб-маска (без регэкспов!) для PhysX RigidBodyView: '/.../envs/env_*/obst_*'
        prefix = self._slots_env0[0].split("/envs/env_0")[0]
        self._rb_glob_pattern: str = f"{prefix}/envs/env_*/obst_*"

        # Геометрия цилиндров обязательна
        r = getattr(self.cfg, "cylinder_radius", None)
        h = getattr(self.cfg, "cylinder_height", None)
        if r is None or h is None:
            raise RuntimeError(
                "[RayCasterMultiMesh] Please set cfg.cylinder_radius and cfg.cylinder_height (e.g., 0.4 / 1.2)."
            )
        self._cyl_radius: float = float(r)
        self._cyl_height: float = float(h)

        # Флаги/тайлинг
        self._use_fp16: bool = bool(getattr(self.cfg, "use_fp16", True))
        self._block_R: int = int(getattr(self.cfg, "block_R", 512))
        self._block_B: int = int(getattr(self.cfg, "block_B", 64))

        # Device
        self._torch_device: torch.device = _torch_device(self.device)

        # Лениво создадим PhysX view
        self._rb_view_all: Optional[physx.RigidBodyView] = None
        self._E_total: Optional[int] = None

        # Буферы широкой фазы (создадим позже)
        self._grid = None  # словарь с полями: Cx, Cy, cell_start, cell_count, cell_items, shape, origin, cell

        # Кэш поз и осей для узкой фазы (float32)
        self._C_all = None
        self._u_all = None
        self._v_all = None
        self._w_all = None

        # Флаг: нужно ли перестраивать решётку (например, при первых шагах или больших сдвигах сцены)
        self._grid_dirty: bool = True

        # PhysX SimulationView
        if not hasattr(self, "_physics_sim_view"):
            from isaacsim.core.simulation_manager import SimulationManager
            self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # Включим TF32/alloc конфиги для стабильности и снижения фрагментации
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        log.info(
            f"[RayCasterMultiMesh] Tilted-cylinder analytic LIDAR. "
            f"M={self._M}, pattern='{self._rb_glob_pattern}', "
            f"r={self._cyl_radius:.3f}, h={self._cyl_height:.3f}, "
            f"fp16={self._use_fp16}, block_R={self._block_R}, block_B={self._block_B}."
        )

    def _collect_env0_obst_slots(self) -> List[str]:
        """Вернёт XForm-примы препятствий под /envs/env_0: .../envs/env_0/obst_\\d+$"""
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
                f"[RayCasterMultiMesh] RigidBodyView count ({view.count}) not divisible by M ({self._M}). "
                "Floor-dividing; misalignment possible if obstacles missing."
            )
        self._E_total = view.count // self._M

    def _ensure_grid_buffers(self):
        if self._grid is not None:
            return
        E, M = self._E_total, self._M
        dev = self._torch_device
        self._grid = {
            "Cx": torch.empty((E, M), device=dev, dtype=torch.float32),
            "Cy": torch.empty((E, M), device=dev, dtype=torch.float32),
            "cell_start": None,   # (E, n_cells) int32
            "cell_count": None,   # (E, n_cells) int32
            "cell_items": None,   # (E, nnz)     int32 (значения индексов препятствий)
            "shape": None,        # (nx, ny)
            "origin": None,       # (2,)
            "cell": None          # float
        }

    # ----------------------------------
    # grid build / update
    # ----------------------------------

    def _update_obstacles_and_grid(self, force_regrid: bool = False):
        """
        Читает позы всех препятствий (E,M,7), кэширует центры/оси.
        При необходимости перестраивает сетку для широкой фазы.
        """
        self._ensure_rb_view()
        self._ensure_grid_buffers()

        dev = self._torch_device

        tf_all = self._rb_view_all.get_transforms().to(device=dev, dtype=torch.float32)  # (E*M,7)
        tf_all = tf_all.view(self._E_total, self._M, 7)

        C = tf_all[..., :3]                                    # (E,M,3)
        q = convert_quat(tf_all[..., 3:7], to="wxyz")          # (E,M,4)
        u, v, w_axis = _quat_to_axes_wxyz(q)                   # (E,M,3) * 3

        # кэш узкой фазы (float32)
        self._C_all = C
        self._u_all = u
        self._v_all = v
        self._w_all = w_axis

        # Обновим XY центров для решётки
        self._grid["Cx"].copy_(C[..., 0])
        self._grid["Cy"].copy_(C[..., 1])

        if not (self._grid_dirty or force_regrid):
            return

        # === построение равномерной сетки по всем env (общий bbox) ===
        cell = float(self.cfg.grid_cell)
        Cx = self._grid["Cx"];  Cy = self._grid["Cy"]

        xmin = torch.min(Cx) - self.cfg.grid_pad * cell
        xmax = torch.max(Cx) + self.cfg.grid_pad * cell
        ymin = torch.min(Cy) - self.cfg.grid_pad * cell
        ymax = torch.max(Cy) + self.cfg.grid_pad * cell

        nx = torch.clamp(((xmax - xmin) / cell).ceil().to(torch.int32), min=1)
        ny = torch.clamp(((ymax - ymin) / cell).ceil().to(torch.int32), min=1)
        n_cells = int((nx * ny).item())

        origin = torch.stack([xmin, ymin]).to(device=dev, dtype=torch.float32)
        self._grid["origin"] = origin
        self._grid["shape"]  = (int(nx.item()), int(ny.item()))
        self._grid["cell"]   = cell

        # === Раскладываем препятствия по клеткам, строим CSR (тензорно) ===
        ix = ((Cx - origin[0]) / cell).floor().to(torch.int32).clamp(0, nx-1)   # (E,M)
        iy = ((Cy - origin[1]) / cell).floor().to(torch.int32).clamp(0, ny-1)   # (E,M)
        cid_int32 = (iy * nx + ix)                                              # (E,M) в [0..n_cells)
        cid_long  = cid_int32.to(torch.long)                                    # индексы для gather/scatter

        cell_count = torch.zeros((self._E_total, n_cells), device=dev, dtype=torch.int32)
        # scatter_add_: индексы должны быть long
        cell_count.scatter_add_(1, cid_long, torch.ones_like(cid_long, dtype=torch.int32))   # (E, n_cells)

        cell_start = torch.zeros_like(cell_count)
        torch.cumsum(cell_count, dim=1, out=cell_start)
        cell_start -= cell_count                                                # эксклюзивный префикс-сумма

        # у каждого env сумма count == M → nnz = M
        nnz = int(cell_count.sum(dim=1).max().item())
        cell_items = torch.full((self._E_total, nnz), -1, device=dev, dtype=torch.int32)

        # заполним cell_items без питон-циклов:
        ones = torch.ones_like(cid_long, dtype=torch.int32)
        write_pos = cell_start.clone()                                          # (E, n_cells) int32

        # ВАЖНО: gather требует long индексы
        offs_int32 = torch.gather(write_pos, 1, cid_long)                       # (E, M) int32
        offs = offs_int32.to(torch.long)                                        # индексы -> long

        # увеличиваем write_pos: scatter_add_ с long-индексами
        write_pos.scatter_add_(1, cid_long, ones)

        item_ids = torch.arange(self._M, device=dev, dtype=torch.int32).unsqueeze(0).expand_as(cid_int32)
        # scatter_: индексы long
        cell_items.scatter_(1, offs, item_ids)

        self._grid["cell_start"] = cell_start        # int32
        self._grid["cell_count"] = cell_count        # int32
        self._grid["cell_items"] = cell_items        # int32

        self._grid_dirty = False


    # ----------------------------------
    # DDA broad-phase
    # ----------------------------------

    def _gather_candidates(self, Sxy: torch.Tensor, Dxy: torch.Tensor) -> torch.Tensor:
        """
        DDA по XY-сетке: для каждого луча выдаёт индексы кандидатов цилиндров.
        Sxy: (B,Rb,2), Dxy: (B,Rb,2) — направления должны быть нормализованы.
        Возврат: (B,Rb,K) int32, -1 = пусто.
        """
        dev = Sxy.device
        B, Rb, _ = Sxy.shape
        K = int(self.cfg.kmax)

        nx, ny = self._grid["shape"]
        origin = self._grid["origin"]         # (2,)
        cell   = self._grid["cell"]
        start  = self._grid["cell_start"]     # (E, n_cells) int32
        count  = self._grid["cell_count"]     # (E, n_cells) int32
        items  = self._grid["cell_items"]     # (E, nnz)     int32

        # начальная ячейка и шаги (int32 для хранения, но индексы -> long)
        sx_i32 = ((Sxy[..., 0] - origin[0]) / cell).floor().to(torch.int32).clamp(0, nx-1)   # (B,Rb)
        sy_i32 = ((Sxy[..., 1] - origin[1]) / cell).floor().to(torch.int32).clamp(0, ny-1)

        step_x = torch.sign(Dxy[..., 0]).to(torch.int32).clamp(min=-1, max=1)
        step_y = torch.sign(Dxy[..., 1]).to(torch.int32).clamp(min=-1, max=1)

        # границы следующих клеток
        next_x = origin[0] + (sx_i32 + (step_x > 0).to(torch.int32)) * cell
        next_y = origin[1] + (sy_i32 + (step_y > 0).to(torch.int32)) * cell

        # расстояние до границы по параметру t
        inv_dx = torch.where(Dxy[..., 0].abs() > 1e-9, 1.0 / Dxy[..., 0], torch.full_like(Dxy[..., 0], 1e9))
        inv_dy = torch.where(Dxy[..., 1].abs() > 1e-9, 1.0 / Dxy[..., 1], torch.full_like(Dxy[..., 1], 1e9))

        t_max_x   = (next_x - Sxy[..., 0]) * inv_dx
        t_max_y   = (next_y - Sxy[..., 1]) * inv_dy
        t_delta_x = (cell * step_x.to(Sxy.dtype)) * inv_dx
        t_delta_y = (cell * step_y.to(Sxy.dtype)) * inv_dy

        t_max_x   = t_max_x.abs()
        t_max_y   = t_max_y.abs()
        t_delta_x = t_delta_x.abs()
        t_delta_y = t_delta_y.abs()

        # кандидаты (-1 = пусто)
        cand = torch.full((B, Rb, K), -1, device=dev, dtype=torch.int32)

        n_steps = int(self.cfg.dda_max_cells)
        cell_id_i32 = sy_i32 * nx + sx_i32             # (B,Rb) int32
        cell_id = cell_id_i32.to(torch.long)           # индексы -> long

        # соберём до K кандидатов, набирая по L с клетки
        Lmax = min(4, K)  # можно тюнинговать; 4 — хороший компромисс

        for _ in range(n_steps):
            # gather требует long индексы
            cstart = torch.gather(start, 1, cell_id)   # (B,Rb) int32
            ccount = torch.gather(count, 1, cell_id)   # (B,Rb) int32

            # сформируем оффсеты чтения из items
            offs_base = cstart.to(torch.long).unsqueeze(-1) + torch.arange(Lmax, device=dev).view(1, 1, Lmax)
            valid = (torch.arange(Lmax, device=dev, dtype=torch.int32).view(1, 1, Lmax)
                     < ccount.unsqueeze(-1)).to(torch.bool)  # (B,Rb,Lmax)

            # offs_base: (B, Rb, Lmax) индексы в плоском списке cell_items по строкам
            idx = offs_base.clamp_max(items.shape[1] - 1).to(torch.long)    # (B,Rb,Lmax), long
            items_exp = items.unsqueeze(1).expand(-1, idx.shape[1], -1)     # (B,Rb,N)
            cell_items = torch.take_along_dim(items_exp, idx, dim=2).to(torch.int32)  # (B,Rb,Lmax)
            cell_items = torch.where(valid, cell_items, torch.full_like(cell_items, -1))

            # найдём свободные позиции (-1) в cand по оси K
            free_mask = (cand < 0)
            if not torch.any(free_mask):
                break

            free_left = free_mask.sum(dim=2, keepdim=True)  # (B,Rb,1)

            # размещаем до Lmax кандидатов
            for l in range(Lmax):
                pick = cell_items[..., l]                               # (B,Rb) int32
                m = (pick >= 0) & (free_left.squeeze(-1) > 0)
                if not torch.any(m):
                    continue
                # индекс первой свободной позиции
                kpos = torch.argmax(free_mask.to(torch.int32), dim=2)   # (B,Rb) long/int64
                # write (нужно адреса (B,Rb,kpos))
                # создаём плоские индексы
                b_idx, r_idx = torch.nonzero(m, as_tuple=True)
                kp = kpos[b_idx, r_idx]
                cand[b_idx, r_idx, kp] = pick[b_idx, r_idx]
                # обновляем маски/счётчики
                free_mask[b_idx, r_idx, kp] = False
                free_left[b_idx, r_idx, 0] -= 1

            # шаг DDA
            step_x_mask = t_max_x <= t_max_y
            sx_i32 = torch.where(step_x_mask, (sx_i32 + step_x).clamp(0, nx - 1), sx_i32)
            t_max_x = torch.where(step_x_mask, t_max_x + t_delta_x, t_max_x)

            sy_i32 = torch.where(~step_x_mask, (sy_i32 + step_y).clamp(0, ny - 1), sy_i32)
            t_max_y = torch.where(~step_x_mask, t_max_y + t_delta_y, t_max_y)

            cell_id_i32 = sy_i32 * nx + sx_i32
            cell_id = cell_id_i32.to(torch.long)

            # если всё заполнено — можно остановиться
            if not torch.any(cand < 0):
                break

        return cand  # (B,Rb,K) int32

    # ----------------------------------
    # update (vector)
    # ----------------------------------

    @torch.no_grad()
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """
        Полностью тензорное обновление:
          1) читаем позы препятствий и (по необходимости) перестраиваем сетку;
          2) собираем кандидатов по DDA;
          3) узкая фаза: аналитика луч–наклонный цилиндр по кандидатам;
          4) пишем best hits в буфер сенсора.
        """
        device = self._torch_device
        cuda_ok = (device.type == "cuda" and torch.cuda.is_available())
        use_amp = bool(self._use_fp16 and cuda_ok)

        # === 1) позы сенсора (как в базовом RayCaster) ===
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

        # === 2) прочитать препятствия и обновить сетку при необходимости ===
        self._update_obstacles_and_grid(force_regrid=self._grid_dirty)

        # === 3) вычисления плитками по R (по памяти) ===
        out_dtype = self._data.ray_hits_w.dtype
        best_hits = torch.full((B, R, 3), float("inf"), device=device, dtype=out_dtype)

        # константы / dtype вычислений
        comp_dtype = torch.float16 if use_amp else torch.float32
        r = torch.as_tensor(self._cyl_radius, device=device, dtype=comp_dtype)
        h = torch.as_tensor(self._cyl_height, device=device, dtype=comp_dtype)
        half_h = 0.5 * h
        max_d = torch.as_tensor(self.cfg.max_distance, device=device, dtype=comp_dtype)
        eps = torch.as_tensor(1e-8, device=device, dtype=comp_dtype)
        INF = torch.as_tensor(float("inf"), device=device, dtype=comp_dtype)

        # кэшированные центры/оси (float32) → comp_dtype
        C_all = self._C_all.to(device=device, dtype=comp_dtype)  # (E,M,3)
        u_all = self._u_all.to(device=device, dtype=comp_dtype)
        v_all = self._v_all.to(device=device, dtype=comp_dtype)
        w_all = self._w_all.to(device=device, dtype=comp_dtype)

        block_R = self._block_R if (0 < self._block_R <= R) else min(R, 1024)
        ray_blocks: List[Tuple[int, int]] = [(i, min(i + block_R, R)) for i in range(0, R, block_R)]

        with _autocast_ctx(use_amp, dtype=torch.float16):
            for r0, r1 in ray_blocks:
                S = ray_starts_w[:, r0:r1, :].to(device=device, dtype=comp_dtype)   # (B,Rb,3)
                D = ray_dirs_w[:,   r0:r1, :].to(device=device, dtype=comp_dtype)   # (B,Rb,3)
                Rb = r1 - r0

                # нормализуем направления
                D = D / torch.linalg.norm(D, dim=-1, keepdim=True).clamp_min_(eps)

                # --- широкая фаза: DDA по XY ---
                cand = self._gather_candidates(S[..., :2], D[..., :2])  # (B,Rb,K), int32
                K = cand.shape[-1]
                mask_valid = (cand >= 0)
                cand_clamped = torch.clamp(cand, 0, self._M - 1).to(torch.long)  # индексы -> long

                # --- узкая фаза: аналитика по кандидатам ---
                Ck  = torch.gather(C_all, 1, cand_clamped.unsqueeze(-1).expand(B, Rb, K, 3))  # (B,Rb,K,3)
                uk  = torch.gather(u_all, 1, cand_clamped.unsqueeze(-1).expand(B, Rb, K, 3))
                vk  = torch.gather(v_all, 1, cand_clamped.unsqueeze(-1).expand(B, Rb, K, 3))
                wk  = torch.gather(w_all, 1, cand_clamped.unsqueeze(-1).expand(B, Rb, K, 3))

                Srel = S.unsqueeze(2) - Ck  # (B,Rb,K,3)

                Su = torch.sum(Srel * uk, dim=-1)         # (B,Rb,K)
                Sv = torch.sum(Srel * vk, dim=-1)
                Sw = torch.sum(Srel * wk, dim=-1)

                Du = torch.sum(D.unsqueeze(2) * uk, dim=-1)  # (B,Rb,K)
                Dv = torch.sum(D.unsqueeze(2) * vk, dim=-1)
                Dw = torch.sum(D.unsqueeze(2) * wk, dim=-1)

                a = Du*Du + Dv*Dv
                b = 2.0 * (Su*Du + Sv*Dv)
                c_q = Su*Su + Sv*Sv - (r*r)

                a = torch.where(a.abs() < eps, eps, a)
                disc = b*b - 4.0*a*c_q

                sqrt_disc = torch.zeros_like(disc)
                pos_disc = disc >= 0
                sqrt_disc[pos_disc] = torch.sqrt(disc[pos_disc])

                INFk = torch.full_like(disc, INF)

                t1 = (-b - sqrt_disc) / (2.0*a)
                t2 = (-b + sqrt_disc) / (2.0*a)
                t1 = torch.where(t1 > 0.0, t1, INFk)
                t2 = torch.where(t2 > 0.0, t2, INFk)

                z1 = Sw + t1*Dw
                z2 = Sw + t2*Dw
                ok1 = (z1 >= -half_h) & (z1 <= half_h)
                ok2 = (z2 >= -half_h) & (z2 <= half_h)

                t_side = torch.minimum(torch.where(ok1, t1, INFk),
                                       torch.where(ok2, t2, INFk))

                Dw_safe = torch.where(Dw.abs() < eps, eps, Dw)
                t_top = ( half_h - Sw) / Dw_safe
                t_bot = (-half_h - Sw) / Dw_safe
                t_top = torch.where(t_top > 0.0, t_top, INFk)
                t_bot = torch.where(t_bot > 0.0, t_bot, INFk)
                ok_top = (Su + t_top*Du)**2 + (Sv + t_top*Dv)**2 <= (r*r)
                ok_bot = (Su + t_bot*Du)**2 + (Sv + t_bot*Dv)**2 <= (r*r)

                t_caps = torch.minimum(torch.where(ok_top, t_top, INFk),
                                       torch.where(ok_bot, t_bot, INFk))

                t_k = torch.minimum(t_side, t_caps)                # (B,Rb,K)
                t_k = torch.where(mask_valid, t_k, INFk)
                t_k = torch.where(t_k <= max_d, t_k, INFk)

                best_t_rb, _ = torch.min(t_k, dim=2)               # (B,Rb)

                # мировая точка: P = S + t*D
                P = (S + best_t_rb.unsqueeze(-1) * D).to(dtype=out_dtype)
                best_hits[:, r0:r1, :] = P

        # Добавим Z-дрифт (как в оригинале)
        best_hits[:, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1).to(dtype=out_dtype)

        # Коммит в буфер сенсора
        self._data.ray_hits_w[env_ids] = best_hits

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

