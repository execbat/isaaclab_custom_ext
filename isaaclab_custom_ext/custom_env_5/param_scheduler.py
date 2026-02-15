from __future__ import annotations
from typing import Sequence, Any, Tuple
import math

def _log_curriculum_scalar(env, name: str, value: float):
    """We write the scalar to the log so that rsl_rl can see it."""
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    log = env.extras.setdefault("log", {})
    log[f"Curriculum/{name}"] = float(value)

def _as_tuple(x: Any) -> Tuple[float, ...]:
    if isinstance(x, (tuple, list)):
        return tuple(float(v) for v in x)
    return (float(x),)

def _progress(env, *, num_steps: float, start_after: int = 0) -> float:
    """
    Returns t∈[0,1] with a delay: while common_step_counter < start_after, t=0.
    Then linearly increases to 1 in num_steps.
    """
    step = float(getattr(env, "common_step_counter", 0))
    num_steps = max(1.0, float(num_steps))
    start_after = max(0.0, float(start_after))
    progressed = max(0.0, step - start_after)
    return max(0.0, min(1.0, progressed / num_steps))

def lerp_scalar(
    env,
    env_ids: Sequence[int],
    old_value,
    *,
    start=None,
    end=None,
    num_steps: int = 100_000,
    start_after: int = 0,
    delay_steps: int | None = None,
    log_name: str | None = None,      # <- name for Curriculum/... logging
):
    """
    Linear: start -> end in num_steps, starting after start_after steps.
    """
    if end is None:
        return old_value
    if start is None:
        start = float(old_value)
    if delay_steps is not None:
        start_after = int(delay_steps)

    t = _progress(env, num_steps=num_steps, start_after=start_after)
    new_value = float(start) + (float(end) - float(start)) * t

    if log_name is not None:
        _log_curriculum_scalar(env, log_name, new_value)

    return new_value

def lerp_tuple(
    env,
    env_ids: Sequence[int],
    old_value,
    *,
    start=None,
    end=None,
    num_steps: int = 100_000,
    start_after: int = 0,
    delay_steps: int | None = None,
    log_name: str | None = None,
):
    cur = _as_tuple(old_value)
    s   = _as_tuple(cur if start is None else start)
    e   = _as_tuple(cur if end   is None else end)

    n = max(len(cur), len(s), len(e))
    s = tuple(s[i] if i < len(s) else s[-1] for i in range(n))
    e = tuple(e[i] if i < len(e) else e[-1] for i in range(n))

    if delay_steps is not None:
        start_after = int(delay_steps)
    t = _progress(env, num_steps=num_steps, start_after=start_after)

    new_tuple = tuple(sv + (ev - sv) * t for sv, ev in zip(s, e))

    if log_name is not None:
        wide_range = abs(new_tuple[0]) + abs(new_tuple[1])
        _log_curriculum_scalar(env, log_name, wide_range)

    return new_tuple

def cosine_warmup(
    env, env_ids: Sequence[int], old_value,
    *, start=None, end=None, num_steps=100_000, start_after: int = 0, delay_steps: int | None = None
):
    """
    Delayed cosine warm-up: 0→1 by cosine, start after start_after.
    """
    if end is None:
        return old_value
    if start is None:
        start = float(old_value)
    if delay_steps is not None:
        start_after = int(delay_steps)

    t = _progress(env, num_steps=num_steps, start_after=start_after)
    alpha = 0.5 * (1.0 - math.cos(math.pi * t))
    return float(start) + (float(end) - float(start)) * alpha

