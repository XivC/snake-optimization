from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, cast

from player.policy import Theta

from llm_as_optimizer.evaluator import EvalSummary, Evaluator, euclid, theta_key
from llm_as_optimizer.memory import AntiArchive, BeliefStore, EliteArchive
from llm_as_optimizer.sampler import (
    DEFAULT_BOUNDS,
    THETA_DIM,
    axis_grid,
    gaussian_perturb,
    novelty_candidates,
    plane_grid,
    uniform_box_perturb,
)

ALLOWED_TOOLS = {
    "evaluate_theta",
    "local_sweep",
    "plane_scan",
    "stability_test",
    "novelty_search",
}

MAX_SWEEP_POINTS = 16
MAX_PLANE_STEPS = 9
MAX_STABILITY_SAMPLES = 32
MAX_NOVELTY_CANDIDATES = 12


class ToolValidationError(ValueError):
    """Ошибка валидации аргументов tool call (передаётся LLM в качестве feedback)."""


def _as_theta(arg: Any, name: str) -> Theta:
    if not isinstance(arg, list | tuple) or len(arg) != THETA_DIM:
        msg = f"{name} должен быть массивом из {THETA_DIM} чисел; получено {arg!r}"
        raise ToolValidationError(msg)
    try:
        return cast(Theta, tuple(float(x) for x in arg))
    except (TypeError, ValueError) as e:
        msg = f"{name}: не удалось привести к float — {e}"
        raise ToolValidationError(msg) from e


def _coerce_int(arg: Any, name: str, *, lo: int, hi: int, default: int | None = None) -> int:
    if arg is None and default is not None:
        return default
    try:
        v = int(arg)
    except (TypeError, ValueError) as e:
        msg = f"{name}: ожидалось целое число, получено {arg!r}"
        raise ToolValidationError(msg) from e
    return max(lo, min(hi, v))


def _coerce_float(arg: Any, name: str, *, lo: float, hi: float, default: float | None = None) -> float:
    if arg is None and default is not None:
        return default
    try:
        v = float(arg)
    except (TypeError, ValueError) as e:
        msg = f"{name}: ожидалось число, получено {arg!r}"
        raise ToolValidationError(msg) from e
    return max(lo, min(hi, v))


@dataclass
class ToolResult:
    tool: str
    args: dict[str, Any]
    ok: bool
    payload: dict[str, Any]
    simulations_spent: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "ok": self.ok,
            "simulations_spent": int(self.simulations_spent),
            "result": self.payload,
        }


@dataclass
class ToolContext:
    evaluator: Evaluator
    elite: EliteArchive
    anti: AntiArchive
    beliefs: BeliefStore
    rng: random.Random

    def record_summary(self, summary: EvalSummary) -> None:
        """Авто-обновление elite/anti по эвристике."""
        self.elite.add(summary)
        cls = self.anti.classify_from_summary(summary)
        if cls is not None:
            ftype, notes = cls
            self.anti.add(summary, failure_type=ftype, notes=notes)


def evaluate_theta(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Симуляция одной θ. Расход: 1 симуляция (или 0, если θ уже в кэше)."""
    theta = _as_theta(args.get("theta"), "theta")
    before = ctx.evaluator.total_simulations
    summary = ctx.evaluator.evaluate(theta)
    spent = ctx.evaluator.total_simulations - before
    ctx.record_summary(summary)
    cur_best = ctx.elite.best()
    payload = {
        "summary": summary.to_dict(),
        "improved_elite": cur_best is not None and cur_best.J == summary.J,
    }
    return ToolResult(
        tool="evaluate_theta",
        args={"theta": list(theta)},
        ok=True,
        payload=payload,
        simulations_spent=spent,
    )


def local_sweep(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """1-D scan одной координаты вокруг center по списку значений `values`
    (или равномерному отрезку, если задан `radius`+`steps`)."""
    center = _as_theta(args.get("center_theta"), "center_theta")
    dim = _coerce_int(args.get("dimension"), "dimension", lo=0, hi=THETA_DIM - 1)
    raw_values = args.get("values")
    if raw_values is None:
        radius = _coerce_float(args.get("radius"), "radius", lo=0.05, hi=4.0, default=0.5)
        steps = _coerce_int(args.get("steps"), "steps", lo=3, hi=MAX_SWEEP_POINTS, default=7)
        values = [center[dim] - radius + (2.0 * radius) * (i / (steps - 1)) for i in range(steps)]
    else:
        if not isinstance(raw_values, list) or not raw_values:
            msg = "values должен быть непустым массивом чисел"
            raise ToolValidationError(msg)
        try:
            values = [float(v) for v in raw_values][:MAX_SWEEP_POINTS]
        except (TypeError, ValueError) as e:
            msg = f"values: не удалось привести к float — {e}"
            raise ToolValidationError(msg) from e

    thetas = axis_grid(center, dimension=dim, values=values)
    before = ctx.evaluator.total_simulations
    summaries = ctx.evaluator.evaluate_many(thetas, progress_prefix="[local_sweep]")
    spent = ctx.evaluator.total_simulations - before
    samples = []
    for v, s in zip(values, summaries):
        ctx.record_summary(s)
        samples.append(
            {
                "value": round(float(v), 4),
                "J": round(s.J, 3),
                "died": bool(s.died),
                "apples": int(s.apples),
            }
        )
    js = [s["J"] for s in samples]
    best_idx = max(range(len(samples)), key=lambda i: samples[i]["J"])
    sensitivity = (max(js) - min(js)) if js else 0.0
    payload = {
        "dimension": dim,
        "center_theta": [round(x, 4) for x in center],
        "samples": samples,
        "argmax_value": samples[best_idx]["value"],
        "argmax_J": samples[best_idx]["J"],
        "sensitivity_range": round(float(sensitivity), 3),
    }
    return ToolResult(
        tool="local_sweep",
        args={
            "center_theta": list(center),
            "dimension": dim,
            "values": values,
        },
        ok=True,
        payload=payload,
        simulations_spent=spent,
    )


def _detect_local_maxima(grid: list[list[float]]) -> list[tuple[int, int, float]]:
    """Простая 4-связная локальная max-фильтрация."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    out: list[tuple[int, int, float]] = []
    for i in range(h):
        for j in range(w):
            v = grid[i][j]
            is_max = True
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] > v:
                    is_max = False
                    break
            if is_max:
                out.append((i, j, v))
    out.sort(key=lambda t: -t[2])
    return out[:5]


def _detect_ridge(grid: list[list[float]]) -> list[dict[str, int]]:
    """Грубый «ridge» по строкам: argmax по каждой строке (если разброс по строке заметный)."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    ridge: list[dict[str, int]] = []
    for i in range(h):
        row = grid[i]
        if max(row) - min(row) < 0.2:
            continue
        j = max(range(w), key=lambda jj: row[jj])
        ridge.append({"i_y": i, "j_x": int(j)})
    return ridge


def plane_scan(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """2-D сетка J(θ) в плоскости двух координат. Возвращает сетку, локальные максимумы и ridge."""
    center = _as_theta(args.get("center_theta"), "center_theta")
    raw_dims = args.get("dims")
    if not isinstance(raw_dims, list | tuple) or len(raw_dims) != 2:
        msg = "dims должен быть массивом ровно из двух разных индексов координат (0..3)"
        raise ToolValidationError(msg)
    try:
        d0 = int(raw_dims[0])
        d1 = int(raw_dims[1])
    except (TypeError, ValueError) as e:
        msg = f"dims: не int — {e}"
        raise ToolValidationError(msg) from e
    if d0 == d1 or not (0 <= d0 < THETA_DIM) or not (0 <= d1 < THETA_DIM):
        msg = "dims должен быть парой РАЗНЫХ индексов в [0..3]"
        raise ToolValidationError(msg)
    radius = _coerce_float(args.get("radius"), "radius", lo=0.05, hi=4.0, default=0.7)
    steps = _coerce_int(args.get("steps"), "steps", lo=3, hi=MAX_PLANE_STEPS, default=5)

    thetas, xs, ys = plane_grid(center, dims=(d0, d1), radius=radius, steps=steps)
    before = ctx.evaluator.total_simulations
    summaries = ctx.evaluator.evaluate_many(thetas, progress_prefix="[plane_scan]")
    spent = ctx.evaluator.total_simulations - before
    grid: list[list[float]] = [[0.0] * steps for _ in range(steps)]
    for idx, s in enumerate(summaries):
        ctx.record_summary(s)
        i = idx // steps
        j = idx % steps
        grid[i][j] = s.J
    flat = [v for row in grid for v in row]
    best_idx = max(range(len(summaries)), key=lambda i: summaries[i].J)
    bi = best_idx // steps
    bj = best_idx % steps
    local_max = _detect_local_maxima(grid)
    ridge = _detect_ridge(grid)
    payload = {
        "dims": [d0, d1],
        "center_theta": [round(x, 4) for x in center],
        "axis_values": {
            "x_dim": d0,
            "x": [round(x, 4) for x in xs],
            "y_dim": d1,
            "y": [round(y, 4) for y in ys],
        },
        "grid_J": [[round(v, 3) for v in row] for row in grid],
        "max": {
            "value": round(summaries[best_idx].J, 3),
            "theta": [round(x, 4) for x in summaries[best_idx].theta],
            "cell": [bi, bj],
        },
        "local_maxima": [
            {"cell": [i, j], "value": round(v, 3)} for i, j, v in local_max
        ],
        "ridge": ridge,
        "global_stats": {
            "min": round(min(flat), 3) if flat else 0.0,
            "max": round(max(flat), 3) if flat else 0.0,
            "mean": round(statistics.fmean(flat), 3) if flat else 0.0,
            "std": round(statistics.pstdev(flat), 3) if len(flat) >= 2 else 0.0,
        },
    }
    return ToolResult(
        tool="plane_scan",
        args={
            "center_theta": list(center),
            "dims": [d0, d1],
            "radius": radius,
            "steps": steps,
        },
        ok=True,
        payload=payload,
        simulations_spent=spent,
    )


def stability_test(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """
    Локальная устойчивость в детерминированном мире: оцениваем J(θ), затем для
    `samples` гауссовых возмущений (σ = ε/√3) считаем |J(θ+δ) − J(θ)|.

    Возвращает:
      - stability_score = 1 − mean|ΔJ| / max(1, |J(θ)|) ∈ [0, 1];
      - collapse_probability — доля возмущений, у которых J упал больше чем на 0.6·J(θ).
    """
    theta = _as_theta(args.get("theta"), "theta")
    epsilon = _coerce_float(args.get("epsilon"), "epsilon", lo=0.05, hi=2.0, default=0.25)
    samples = _coerce_int(
        args.get("samples"),
        "samples",
        lo=4,
        hi=MAX_STABILITY_SAMPLES,
        default=10,
    )

    before = ctx.evaluator.total_simulations
    base = ctx.evaluator.evaluate(theta)
    ctx.record_summary(base)
    sigma = epsilon / math.sqrt(3.0)
    sigmas = (sigma, sigma, sigma, sigma)
    perturbed = gaussian_perturb(
        theta,
        sigmas=sigmas,
        n=samples,
        rng=ctx.rng,
        bounds=DEFAULT_BOUNDS,
    )
    p_summaries = ctx.evaluator.evaluate_many(perturbed, progress_prefix="[stability_test]")
    spent = ctx.evaluator.total_simulations - before

    deltas: list[float] = []
    collapses = 0
    rows: list[dict[str, Any]] = []
    base_j = base.J
    collapse_threshold = max(0.5, 0.6 * base_j) if base_j > 0 else 1.0
    for ps in p_summaries:
        ctx.record_summary(ps)
        d = abs(ps.J - base_j)
        deltas.append(d)
        if base_j > 0 and (base_j - ps.J) > collapse_threshold:
            collapses += 1
        rows.append(
            {
                "theta": [round(x, 4) for x in ps.theta],
                "J": round(ps.J, 3),
                "delta_J": round(ps.J - base_j, 3),
            }
        )
    mean_abs = statistics.fmean(deltas) if deltas else 0.0
    scale = max(1.0, abs(base_j))
    stability = max(0.0, min(1.0, 1.0 - mean_abs / scale))
    payload = {
        "theta": [round(x, 4) for x in theta],
        "epsilon": round(epsilon, 4),
        "base_J": round(base_j, 3),
        "samples": rows,
        "mean_abs_delta_J": round(mean_abs, 3),
        "stability_score": round(stability, 3),
        "collapse_probability": round(collapses / max(1, len(p_summaries)), 3),
    }
    return ToolResult(
        tool="stability_test",
        args={
            "theta": list(theta),
            "epsilon": epsilon,
            "samples": samples,
        },
        ok=True,
        payload=payload,
        simulations_spent=spent,
    )


def _sign_pattern(theta: Theta, *, eps: float = 0.5) -> tuple[int, int, int, int]:
    return cast(
        tuple[int, int, int, int],
        tuple(0 if abs(x) <= eps else (1 if x > 0 else -1) for x in theta),
    )


def novelty_search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Находит «пустые» области в текущем кэше: max-min greedy в LHS-пуле + список
    неосвещённых знаковых паттернов. Если args.evaluate=True (по умолчанию) — сразу симулирует."""
    n_candidates = _coerce_int(
        args.get("n_candidates"),
        "n_candidates",
        lo=2,
        hi=MAX_NOVELTY_CANDIDATES,
        default=5,
    )
    do_evaluate = bool(args.get("evaluate", True))

    known: list[Theta] = [s.theta for s in ctx.evaluator.all_summaries()]
    cands = novelty_candidates(
        known,
        rng=ctx.rng,
        bounds=DEFAULT_BOUNDS,
        n_candidates=n_candidates,
    )

    explored_patterns: set[tuple[int, int, int, int]] = {_sign_pattern(t) for t in known}
    all_patterns = {
        (a, b, c, d)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        for c in (-1, 0, 1)
        for d in (-1, 0, 1)
    }
    unexplored = sorted(all_patterns - explored_patterns)

    samples: list[dict[str, Any]] = []
    spent = 0
    if do_evaluate and cands:
        before = ctx.evaluator.total_simulations
        sums = ctx.evaluator.evaluate_many(cands, progress_prefix="[novelty_search]")
        spent = ctx.evaluator.total_simulations - before
        for s in sums:
            ctx.record_summary(s)
            samples.append(
                {
                    "theta": [round(x, 4) for x in s.theta],
                    "J": round(s.J, 3),
                    "died": bool(s.died),
                }
            )
    else:
        for t in cands:
            samples.append({"theta": [round(x, 4) for x in t], "J": None, "died": None})

    payload = {
        "n_candidates": n_candidates,
        "evaluated": do_evaluate,
        "candidates": samples,
        "unexplored_sign_patterns": [list(p) for p in unexplored[:12]],
        "known_unique_theta": len(known),
    }
    return ToolResult(
        tool="novelty_search",
        args={"n_candidates": n_candidates, "evaluate": do_evaluate},
        ok=True,
        payload=payload,
        simulations_spent=spent,
    )


_DISPATCH: dict[str, Any] = {
    "evaluate_theta": evaluate_theta,
    "local_sweep": local_sweep,
    "plane_scan": plane_scan,
    "stability_test": stability_test,
    "novelty_search": novelty_search,
}


def dispatch(tool: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    if tool not in _DISPATCH:
        msg = f"Неизвестный tool: {tool!r}. Доступные: {sorted(ALLOWED_TOOLS)}"
        raise ToolValidationError(msg)
    if not isinstance(args, dict):
        msg = f"args для {tool} должен быть object/dict; получено {type(args).__name__}"
        raise ToolValidationError(msg)
    return _DISPATCH[tool](args, ctx)


def safe_dispatch(tool: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Обёртка с error capture: при ошибке возвращает ToolResult c ok=False."""
    try:
        return dispatch(tool, args, ctx)
    except ToolValidationError as e:
        return ToolResult(
            tool=str(tool),
            args=dict(args) if isinstance(args, dict) else {"_raw": str(args)},
            ok=False,
            payload={"error": str(e), "type": "ToolValidationError"},
            simulations_spent=0,
        )
    except Exception as e:  # noqa: BLE001
        return ToolResult(
            tool=str(tool),
            args=dict(args) if isinstance(args, dict) else {"_raw": str(args)},
            ok=False,
            payload={"error": f"{type(e).__name__}: {e}"},
            simulations_spent=0,
        )


_ = euclid  # сохраняем импорт для возможного использования инструментами
_ = theta_key
_ = uniform_box_perturb
