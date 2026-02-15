from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def initial_final_interpolate_fn(env: ManagerBasedRLEnv, env_id, data, initial_value, final_value, difficulty_term_str):
    """
    Interpolate between initial value iv and final value fv, for any arbitrarily
    nested structure of lists/tuples in 'data'. Scalars (int/float) are handled
    at the leaves.
    """
    # get the fraction scalar on the device
    difficulty_term: DifficultyScheduler = getattr(env.curriculum_manager.cfg, difficulty_term_str).func
    frac = difficulty_term.difficulty_frac
    if frac < 0.1:
        # no-op during start, since the difficulty fraction near 0 is wasting of resource.
        return mdp.modify_env_param.NO_CHANGE

    # convert iv/fv to tensors, but we'll peel them apart in recursion
    initial_value_tensor = torch.tensor(initial_value, device=env.device)
    final_value_tensor = torch.tensor(final_value, device=env.device)

    return _recurse(initial_value_tensor.tolist(), final_value_tensor.tolist(), data, frac)


class DifficultyScheduler(ManagerTermBase):
    """Adaptive difficulty scheduler based on a reward term (e.g. target_distance_exp).

    Keeps same call signature as the baseline DifficultyScheduler so existing curriculum configs work.
    The difficulty increases when the reward term exceeds a threshold (optionally using EMA smoothing),
    and decreases otherwise (unless promotion_only is set).
    """

    def __init__(self, cfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        # initial difficulty
        init_difficulty = self.cfg.params.get("init_difficulty", 0)
        self.current_adr_difficulties = torch.ones(env.num_envs, device=env.device) * float(init_difficulty)

        # which reward term to track
        self.term_name: str = self.cfg.params.get("term_name", "target_distance_exp")

        # thresholds & smoothing (can be overridden from params or __call__)
        self.promote_threshold: float = float(self.cfg.params.get("promote_threshold", 0.7))
        self.demote_threshold: float = float(self.cfg.params.get("demote_threshold", 0.3))
        self.ema_alpha: float = float(self.cfg.params.get("ema_alpha", 0.05))  # 0<alpha<=1 ; 1 = no smoothing
        self.warmup_steps: int = int(self.cfg.params.get("warmup_steps", 0))

        # per-env EMA state
        self._term_ema = torch.zeros(env.num_envs, device=env.device)
        self._has_ema = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # cached index of the reward term
        self._term_idx: int | None = None
        self._resolve_term_index()

        self.difficulty_frac = 0.0

    def _resolve_term_index(self):
        names = self._env.reward_manager.active_terms
        if self.term_name not in names:
            raise ValueError(
                f"Reward term '{self.term_name}' not found in reward_manager.active_terms.\n"
                f"Active terms: {names}"
            )
        self._term_idx = names.index(self.term_name)

    def get_state(self):
        # keep compatibility: you can return just difficulties, but saving EMA makes resume nicer
        return {
            "difficulties": self.current_adr_difficulties.clone(),
            "term_ema": self._term_ema.clone(),
            "has_ema": self._has_ema.clone(),
        }

    def set_state(self, state):
        # accept both old-style (tensor) and new-style (dict) states
        if isinstance(state, torch.Tensor):
            self.current_adr_difficulties = state.clone().to(self._env.device)
            return
        self.current_adr_difficulties = state["difficulties"].clone().to(self._env.device)
        self._term_ema = state["term_ema"].clone().to(self._env.device)
        self._has_ema = state["has_ema"].clone().to(self._env.device)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: Sequence[int],
        # ---- keep the SAME signature fields as baseline for config compatibility ----
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        pos_tol: float = 0.1,
        rot_tol: float | None = None,
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 50,
        promotion_only: bool = False,
        # ---- extra knobs (optional) ----
        term_name: str | None = None,
        promote_threshold: float | None = None,
        demote_threshold: float | None = None,
        ema_alpha: float | None = None,
        warmup_steps: int | None = None,
    ):
        # NOTE: asset_cfg/object_cfg/pos_tol/rot_tol kept only for compatibility; not used here.

        # allow switching term name dynamically
        if term_name is not None and term_name != self.term_name:
            self.term_name = term_name
            self._resolve_term_index()

        promote_th = self.promote_threshold if promote_threshold is None else float(promote_threshold)
        demote_th = self.demote_threshold if demote_threshold is None else float(demote_threshold)
        alpha = self.ema_alpha if ema_alpha is None else float(ema_alpha)
        warm = self.warmup_steps if warmup_steps is None else int(warmup_steps)

        # warmup: no updates early
        if warm > 0 and getattr(env, "common_step_counter", 0) < warm:
            self.difficulty_frac = (torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)).item()
            return self.difficulty_frac

        # take current per-step reward term values
        # reward_manager._step_reward is [num_envs, num_terms]
        term_vals = env.reward_manager._step_reward[:, self._term_idx]  # shape [num_envs]

        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        v = term_vals[ids]

        # EMA smoothing (alpha=1 => v)
        if alpha >= 1.0:
            ema = v
            self._term_ema[ids] = v
            self._has_ema[ids] = True
        else:
            has = self._has_ema[ids]
            old = self._term_ema[ids]
            ema = torch.where(has, (1.0 - alpha) * old + alpha * v, v)
            self._term_ema[ids] = ema
            self._has_ema[ids] = True

        # decision rule:
        # - promote if ema > promote_threshold
        # - demote if ema < demote_threshold
        # - else keep difficulty
        move_up = ema > promote_th
        move_down = ema < demote_th

        cur = self.current_adr_difficulties[ids]
        demot = cur if promotion_only else (cur - 1.0)

        updated = torch.where(move_up, cur + 1.0, torch.where(move_down, demot, cur))
        updated = updated.clamp(min=float(min_difficulty), max=float(max_difficulty))
        self.current_adr_difficulties[ids] = updated

        self.difficulty_frac = (torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)).item()
        return self.difficulty_frac
