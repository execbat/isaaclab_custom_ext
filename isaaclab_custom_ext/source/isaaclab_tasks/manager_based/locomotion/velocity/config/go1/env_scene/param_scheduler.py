from __future__ import annotations
from typing import Sequence, Any, Tuple
import math

def _as_tuple(x: Any) -> Tuple[float, ...]:
    if isinstance(x, (tuple, list)):
        return tuple(float(v) for v in x)
    return (float(x),)

def _progress(env, *, num_steps: float, start_after: int = 0) -> float:
    """
    Возвращает t∈[0,1] с задержкой: пока common_step_counter < start_after, t=0.
    Затем линейный рост до 1 за num_steps.
    """
    step = float(getattr(env, "common_step_counter", 0))
    # поддержка альтернативного имени параметра через kwargs выше по стеку (если вдруг)
    num_steps = max(1.0, float(num_steps))
    start_after = max(0.0, float(start_after))
    # «сколько прошли» после старта
    progressed = max(0.0, step - start_after)
    return max(0.0, min(1.0, progressed / num_steps))

def lerp_scalar(
    env, env_ids: Sequence[int], old_value,
    *, start=None, end=None, num_steps=100_000, start_after: int = 0, delay_steps: int | None = None
):
    """
    Линейно: start -> end за num_steps, начиная после start_after шагов.
    Пока шагов меньше — возвращает start (или old_value, если start не задан).
    """
    if end is None:
        return old_value
    if start is None:
        start = float(old_value)
    if delay_steps is not None:
        start_after = int(delay_steps)

    t = _progress(env, num_steps=num_steps, start_after=start_after)
    return float(start) + (float(end) - float(start)) * t

def lerp_tuple(
    env, env_ids: Sequence[int], old_value,
    *, start=None, end=None, num_steps=100_000, start_after: int = 0, delay_steps: int | None = None
):
    """
    То же для кортежей/диапазонов.
    Длины выравниваются: недостающие элементы повторяют последний.
    """
    cur = _as_tuple(old_value)
    s   = _as_tuple(cur if start is None else start)
    e   = _as_tuple(cur if end   is None else end)

    n = max(len(cur), len(s), len(e))
    s = tuple(s[i] if i < len(s) else s[-1] for i in range(n))
    e = tuple(e[i] if i < len(e) else e[-1] for i in range(n))

    if delay_steps is not None:
        start_after = int(delay_steps)
    t = _progress(env, num_steps=num_steps, start_after=start_after)
    return tuple(sv + (ev - sv) * t for sv, ev in zip(s, e))

def cosine_warmup(
    env, env_ids: Sequence[int], old_value,
    *, start=None, end=None, num_steps=100_000, start_after: int = 0, delay_steps: int | None = None
):
    """
    Косинусный прогрев с задержкой: 0→1 по косинусу, старт после start_after.
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

