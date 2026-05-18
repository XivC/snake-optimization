from __future__ import annotations

import math
import random
from typing import cast

from player.policy import Theta

THETA_DIM = 4
DEFAULT_BOUNDS: tuple[tuple[float, float], ...] = (
    (-4.9, 4.9),
    (-4.9, 4.9),
    (-4.9, 4.9),
    (-4.9, 4.9),
)


def _as_theta(values: list[float]) -> Theta:
    if len(values) != THETA_DIM:
        msg = f"Ожидался вектор размерности {THETA_DIM}, получено {len(values)}"
        raise ValueError(msg)
    return cast(Theta, tuple(float(x) for x in values))


def _clip_to_bounds(theta: Theta, bounds: tuple[tuple[float, float], ...]) -> Theta:
    return cast(
        Theta,
        tuple(min(bounds[i][1], max(bounds[i][0], theta[i])) for i in range(THETA_DIM)),
    )


def lhs_box(
    n: int,
    *,
    rng: random.Random,
    bounds: tuple[tuple[float, float], ...] = DEFAULT_BOUNDS,
) -> list[Theta]:
    """Latin Hypercube Sampling в кубе bounds. Покрывает каждый страт по каждой оси ровно один раз."""
    n = max(1, int(n))
    out: list[list[float]] = [[0.0] * THETA_DIM for _ in range(n)]
    for d in range(THETA_DIM):
        lo, hi = bounds[d]
        step = (hi - lo) / n
        strata = list(range(n))
        rng.shuffle(strata)
        for i, s in enumerate(strata):
            out[i][d] = lo + step * (s + rng.random())
    return [_as_theta(v) for v in out]


def random_uniform_box(
    n: int,
    *,
    rng: random.Random,
    bounds: tuple[tuple[float, float], ...] = DEFAULT_BOUNDS,
) -> list[Theta]:
    out: list[Theta] = []
    for _ in range(max(1, int(n))):
        v = [rng.uniform(b[0], b[1]) for b in bounds]
        out.append(_as_theta(v))
    return out


def gaussian_perturb(
    center: Theta,
    *,
    sigmas: tuple[float, float, float, float],
    n: int,
    rng: random.Random,
    bounds: tuple[tuple[float, float], ...] = DEFAULT_BOUNDS,
) -> list[Theta]:
    out: list[Theta] = []
    for _ in range(max(1, int(n))):
        v = [center[d] + rng.gauss(0.0, sigmas[d]) for d in range(THETA_DIM)]
        out.append(_clip_to_bounds(_as_theta(v), bounds))
    return out


def uniform_box_perturb(
    center: Theta,
    *,
    radii: tuple[float, float, float, float],
    n: int,
    rng: random.Random,
    max_changed_coords: int = 2,
    bounds: tuple[tuple[float, float], ...] = DEFAULT_BOUNDS,
) -> list[Theta]:
    """L∞-возмущения с ограничением «не более `max_changed_coords` координат меняем»."""
    out: list[Theta] = []
    for _ in range(max(1, int(n))):
        coords = list(range(THETA_DIM))
        rng.shuffle(coords)
        k = max(1, min(max_changed_coords, THETA_DIM))
        active = set(coords[:k])
        v = list(center)
        for d in range(THETA_DIM):
            if d in active and radii[d] > 0:
                v[d] = center[d] + rng.uniform(-radii[d], radii[d])
        out.append(_clip_to_bounds(_as_theta(v), bounds))
    return out


def axis_grid(
    center: Theta,
    *,
    dimension: int,
    values: list[float],
) -> list[Theta]:
    """Sweep одной координаты по списку значений (остальные = center)."""
    if not 0 <= dimension < THETA_DIM:
        msg = f"dimension должен быть 0..{THETA_DIM - 1}"
        raise ValueError(msg)
    out: list[Theta] = []
    for v in values:
        c = list(center)
        c[dimension] = float(v)
        out.append(_as_theta(c))
    return out


def plane_grid(
    center: Theta,
    *,
    dims: tuple[int, int],
    radius: float,
    steps: int,
) -> tuple[list[Theta], list[float], list[float]]:
    """
    Регулярная 2D-сетка в плоскости (dims[0], dims[1]) вокруг center,
    каждая ось — [c−radius, c+radius], steps×steps узлов.
    Возвращает плоский список θ и опорные значения по двум осям.
    """
    d0, d1 = int(dims[0]), int(dims[1])
    if not (0 <= d0 < THETA_DIM and 0 <= d1 < THETA_DIM) or d0 == d1:
        msg = "dims должен быть парой разных индексов в [0..3]"
        raise ValueError(msg)
    s = max(2, int(steps))
    r = abs(float(radius))
    xs = [center[d0] - r + (2.0 * r) * (i / (s - 1)) for i in range(s)]
    ys = [center[d1] - r + (2.0 * r) * (j / (s - 1)) for j in range(s)]
    out: list[Theta] = []
    for yi in ys:
        for xi in xs:
            c = list(center)
            c[d0] = xi
            c[d1] = yi
            out.append(_as_theta(c))
    return out, xs, ys


def sign_flip_neighbours(
    center: Theta,
    *,
    coords: list[int] | None = None,
) -> list[Theta]:
    """Отражение знака по выбранным координатам (по одной и попарно)."""
    use = coords if coords else list(range(THETA_DIM))
    seen: set[tuple[float, ...]] = set()
    out: list[Theta] = []
    for d in use:
        v = list(center)
        v[d] = -center[d]
        t = _as_theta(v)
        k = tuple(round(x, 6) for x in t)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def sparse_axis_probes(
    *,
    amplitudes: tuple[float, ...] = (3.0, 4.5),
    include_negative: bool = True,
    include_two_axis: bool = True,
) -> list[Theta]:
    """
    «Разреженные» зонды: 1–2 активные координаты, остальные = 0.
    Без них bootstrap систематически промахивается мимо конфигураций (a, 0, a, 0) и т.п.,
    которые часто оказываются глобальным оптимумом для разреженных policy-фич.
    """
    out: list[Theta] = []
    signs = (1, -1) if include_negative else (1,)
    for a in amplitudes:
        for d in range(THETA_DIM):
            for s in signs:
                v = [0.0] * THETA_DIM
                v[d] = s * float(a)
                out.append(_as_theta(v))
    if include_two_axis:
        for a in amplitudes:
            for i in range(THETA_DIM):
                for j in range(i + 1, THETA_DIM):
                    for s_i, s_j in ((1, 1), (1, -1), (-1, 1)):
                        v = [0.0] * THETA_DIM
                        v[i] = s_i * float(a)
                        v[j] = s_j * float(a)
                        out.append(_as_theta(v))
    return out


def stratified_box_corners(
    rng: random.Random,
    *,
    bounds: tuple[tuple[float, float], ...] = DEFAULT_BOUNDS,
    fraction: float = 0.85,
) -> list[Theta]:
    """16 угловых вершин куба, ужатых к центру на (1−fraction) (по умолчанию 85% от границ)."""
    out: list[Theta] = []
    for mask in range(1 << THETA_DIM):
        v: list[float] = []
        for d in range(THETA_DIM):
            lo, hi = bounds[d]
            sign = 1 if (mask >> d) & 1 else -1
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * float(fraction)
            v.append(mid + sign * half)
        out.append(_as_theta(v))
    rng.shuffle(out)
    return out


def novelty_candidates(
    known: list[Theta],
    *,
    rng: random.Random,
    bounds: tuple[tuple[float, float], ...] = DEFAULT_BOUNDS,
    n_candidates: int = 5,
    n_pool: int = 256,
) -> list[Theta]:
    """
    Max-min дистанция: генерим пул LHS точек, выбираем n_candidates с максимальным расстоянием
    до ближайшей известной (или до уже выбранной из novelty-набора).
    """
    pool = lhs_box(n_pool, rng=rng, bounds=bounds)
    chosen: list[Theta] = []
    anchors: list[Theta] = list(known)
    while pool and len(chosen) < n_candidates:
        best: Theta | None = None
        best_d = -1.0
        for cand in pool:
            d = _min_dist(cand, anchors) if anchors else _min_dist(cand, chosen) if chosen else float("inf")
            if d > best_d:
                best_d = d
                best = cand
        if best is None:
            break
        chosen.append(best)
        anchors.append(best)
        pool.remove(best)
    return chosen


def _min_dist(theta: Theta, pool: list[Theta]) -> float:
    if not pool:
        return float("inf")
    best = float("inf")
    for p in pool:
        d = math.sqrt(sum((theta[i] - p[i]) ** 2 for i in range(THETA_DIM)))
        if d < best:
            best = d
    return best
