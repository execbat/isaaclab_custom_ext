from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------
# Helpers for recursive interpolate
# -----------------------------
def _is_seq(x: Any) -> bool:
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes))


def _to_py_number(x: Any) -> float | int:
    """Convert tensor scalar / numpy scalar / python scalar to Python number."""
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError(f"Expected scalar tensor at leaf, got shape={tuple(x.shape)}")
        return x.item()

    if isinstance(x, numbers.Number):
        return x

    # numpy scalar-like fallback
    if hasattr(x, "item"):
        return x.item()

    raise TypeError(f"Expected numeric leaf, got type={type(x)} value={x!r}")


def _rebuild_same_seq_type(template_seq: Sequence, items: list[Any]) -> Any:
    """Rebuild list/tuple preserving the original type when possible."""
    if isinstance(template_seq, tuple):
        return tuple(items)
    if isinstance(template_seq, list):
        return items
    # fallback for uncommon Sequence subclasses
    return type(template_seq)(items)


def _recurse(iv_elem: Any, fv_elem: Any, data_elem: Any, frac: float) -> Any:
    """
    Recursively interpolate:
        out = iv + frac * (fv - iv)

    `data_elem` is used as a structural/type template:
      - sequences -> recurse by element
      - mappings  -> recurse by keys
      - numeric leaves -> interpolate and preserve int/float style
    """
    # dict / mapping branch
    if isinstance(data_elem, Mapping):
        if not isinstance(iv_elem, Mapping) or not isinstance(fv_elem, Mapping):
            raise TypeError(
                f"Structure mismatch: data is Mapping, but initial/final are "
                f"{type(iv_elem)} / {type(fv_elem)}"
            )

        data_keys = set(data_elem.keys())
        if set(iv_elem.keys()) != data_keys or set(fv_elem.keys()) != data_keys:
            raise ValueError(
                "Key mismatch in recursive interpolate.\n"
                f"data keys={sorted(data_keys)}\n"
                f"initial keys={sorted(iv_elem.keys())}\n"
                f"final keys={sorted(fv_elem.keys())}"
            )

        return {
            k: _recurse(iv_elem[k], fv_elem[k], data_elem[k], frac)
            for k in data_elem.keys()
        }

    # sequence branch
    if _is_seq(data_elem):
        if not _is_seq(iv_elem) or not _is_seq(fv_elem):
            raise TypeError(
                f"Structure mismatch: data is sequence ({type(data_elem)}), "
                f"but initial/final are {type(iv_elem)} / {type(fv_elem)}"
            )

        if not (len(iv_elem) == len(fv_elem) == len(data_elem)):
            raise ValueError(
                f"Length mismatch in recursive interpolate: "
                f"len(initial)={len(iv_elem)}, len(final)={len(fv_elem)}, len(data)={len(data_elem)}"
            )

        out_items = [
            _recurse(iv_e, fv_e, d_e, frac)
            for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem)
        ]
        return _rebuild_same_seq_type(data_elem, out_items)

    # leaf branch (numeric)
    iv = _to_py_number(iv_elem)
    fv = _to_py_number(fv_elem)
    new_val = iv + frac * (fv - iv)

    # preserve leaf type from current data
    if isinstance(data_elem, bool):
        return bool(round(new_val))
    if isinstance(data_elem, int) and not isinstance(data_elem, bool):
        return int(round(new_val))
    if isinstance(data_elem, float):
        return float(new_val)

    # if leaf in data is not numeric (unexpected), return interpolated float
    return float(new_val)


def initial_final_interpolate_fn(
    env: "ManagerBasedRLEnv",
    env_id,  # IsaacLab may pass env_id or env_ids depending on hook; kept for compatibility
    data,
    initial_value,
    final_value,
    difficulty_term_str,
):
    """
    Interpolate between `initial_value` and `final_value` with fraction from DifficultyScheduler.

    Supports nested:
      - list / tuple
      - dict
      - scalar leaves (int/float)
    """
    difficulty_term: DifficultyScheduler = getattr(env.curriculum_manager.cfg, difficulty_term_str).func
    frac = float(difficulty_term.difficulty_frac)

    # early no-op (keep your original optimization)
    if frac < 0.1:
        return mdp.modify_env_param.NO_CHANGE

    # Important: no torch.tensor(...).tolist() here — we preserve original structures/types.
    return _recurse(initial_value, final_value, data, frac)


# -----------------------------
# Difficulty scheduler
# -----------------------------
class DifficultyScheduler(ManagerTermBase):
    """Adaptive difficulty scheduler based on a reward term.

    Compatible with IsaacLab curriculum call signature.
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        # initial difficulty
        init_difficulty = float(self.cfg.params.get("init_difficulty", 0))
        self.current_adr_difficulties = torch.full(
            (env.num_envs,),
            fill_value=init_difficulty,
            device=env.device,
            dtype=torch.float32,
        )

        # tracked reward term
        self.term_name: str = self.cfg.params.get("term_name", "target_distance_exp")

        # thresholds / smoothing
        self.promote_threshold: float = float(self.cfg.params.get("promote_threshold", 0.7))
        self.demote_threshold: float = float(self.cfg.params.get("demote_threshold", 0.3))
        self.ema_alpha: float = float(self.cfg.params.get("ema_alpha", 0.05))
        self.warmup_steps: int = int(self.cfg.params.get("warmup_steps", 0))

        if self.ema_alpha <= 0.0:
            raise ValueError(f"ema_alpha must be > 0, got {self.ema_alpha}")

        # per-env EMA state
        self._term_ema = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        self._has_ema = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        # cached reward term index
        self._term_idx: int | None = None
        self._resolve_term_index()

        # scalar fraction [0..1]
        self.difficulty_frac: float = 0.0

    def _resolve_term_index(self):
        names = self._env.reward_manager.active_terms
        if self.term_name not in names:
            raise ValueError(
                f"Reward term '{self.term_name}' not found in reward_manager.active_terms.\n"
                f"Active terms: {names}"
            )
        self._term_idx = names.index(self.term_name)

    def get_state(self):
        """Return state for checkpointing."""
        return {
            "difficulties": self.current_adr_difficulties.clone(),
            "term_ema": self._term_ema.clone(),
            "has_ema": self._has_ema.clone(),
            "term_name": self.term_name,
        }

    def set_state(self, state):
        """Restore state (supports old tensor-only checkpoints)."""
        if isinstance(state, torch.Tensor):
            self.current_adr_difficulties = state.clone().to(self._env.device)
            return

        self.current_adr_difficulties = state["difficulties"].clone().to(self._env.device)
        self._term_ema = state.get("term_ema", torch.zeros_like(self.current_adr_difficulties)).clone().to(self._env.device)
        self._has_ema = state.get(
            "has_ema",
            torch.zeros_like(self.current_adr_difficulties, dtype=torch.bool),
        ).clone().to(self._env.device)

        if "term_name" in state and state["term_name"] != self.term_name:
            self.term_name = state["term_name"]
            self._resolve_term_index()

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: Sequence[int],
        # compatibility params (kept intentionally)
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        pos_tol: float = 0.1,
        rot_tol: float | None = None,
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 50,
        promotion_only: bool = False,
        # optional overrides
        term_name: str | None = None,
        promote_threshold: float | None = None,
        demote_threshold: float | None = None,
        ema_alpha: float | None = None,
        warmup_steps: int | None = None,
    ):
        # NOTE: asset_cfg/object_cfg/pos_tol/rot_tol are unused, kept for signature compatibility.

        # dynamic term switch
        if term_name is not None and term_name != self.term_name:
            self.term_name = term_name
            self._resolve_term_index()

        promote_th = self.promote_threshold if promote_threshold is None else float(promote_threshold)
        demote_th = self.demote_threshold if demote_threshold is None else float(demote_threshold)
        alpha = self.ema_alpha if ema_alpha is None else float(ema_alpha)
        warm = self.warmup_steps if warmup_steps is None else int(warmup_steps)

        if alpha <= 0.0:
            raise ValueError(f"ema_alpha must be > 0, got {alpha}")

        # warmup: don't modify difficulty yet
        if warm > 0 and int(getattr(env, "common_step_counter", 0)) < warm:
            self.difficulty_frac = (
                self.current_adr_difficulties.mean() / max(float(max_difficulty), 1.0)
            ).item()
            return self.difficulty_frac

        # reward_manager._step_reward shape: [num_envs, num_terms]
        term_vals = env.reward_manager._step_reward[:, self._term_idx]

        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        if ids.numel() == 0:
            return self.difficulty_frac

        v = term_vals.index_select(0, ids)

        # EMA
        if alpha >= 1.0:
            ema = v
        else:
            has = self._has_ema.index_select(0, ids)
            old = self._term_ema.index_select(0, ids)
            ema = torch.where(has, (1.0 - alpha) * old + alpha * v, v)

        self._term_ema[ids] = ema
        self._has_ema[ids] = True

        # update difficulty
        cur = self.current_adr_difficulties.index_select(0, ids)

        move_up = ema > promote_th
        move_down = ema < demote_th

        down_val = cur if promotion_only else (cur - 1.0)
        updated = torch.where(move_up, cur + 1.0, torch.where(move_down, down_val, cur))
        updated = updated.clamp(min=float(min_difficulty), max=float(max_difficulty))

        self.current_adr_difficulties[ids] = updated

        # global normalized fraction [0..1]
        self.difficulty_frac = (
            self.current_adr_difficulties.mean() / max(float(max_difficulty), 1.0)
        ).item()

        return self.difficulty_frac
