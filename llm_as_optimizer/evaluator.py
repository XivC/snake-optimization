from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from player.policy import Theta

from llm_as_optimizer.snake_rollout import RolloutMetrics, simulate_metrics_packed

ThetaKey = tuple[float, float, float, float]
_KEY_DECIMALS = 6


def theta_key(theta: Theta) -> ThetaKey:
    return tuple(round(float(x), _KEY_DECIMALS) for x in theta)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class EvalSummary:
    """Сводка по одной четвёрке θ. В детерминированном режиме это ровно одна симуляция."""

    theta: Theta
    J: float
    apples: int
    steps: int
    died: bool

    # Совместимость со старыми потребителями (memory, prompts, tools).
    @property
    def mean_J(self) -> float:
        return self.J

    @property
    def std_J(self) -> float:
        return 0.0

    @property
    def min_J(self) -> float:
        return self.J

    @property
    def max_J(self) -> float:
        return self.J

    @property
    def mean_steps(self) -> float:
        return float(self.steps)

    @property
    def mean_apples(self) -> float:
        return float(self.apples)

    @property
    def death_rate(self) -> float:
        return 1.0 if self.died else 0.0

    @property
    def rollouts(self) -> int:
        return 1

    def to_dict(self) -> dict[str, object]:
        return {
            "theta": [round(float(x), 4) for x in self.theta],
            "J": round(self.J, 3),
            "apples": int(self.apples),
            "steps": int(self.steps),
            "died": bool(self.died),
        }


def _summary_from_metrics(theta: Theta, m: RolloutMetrics) -> EvalSummary:
    return EvalSummary(
        theta=theta,
        J=float(m.J),
        apples=int(m.apples),
        steps=int(m.steps),
        died=bool(m.died),
    )


class Evaluator:
    """
    Параллельный симулятор в детерминированном режиме.

    Один глобальный `base_seed` фиксирует мир: для каждой θ симуляция запускается ровно
    один раз. Результаты кэшируются по theta_key — повторный запрос той же θ возвращает
    уже посчитанное значение без новой симуляции.
    """

    def __init__(
        self,
        *,
        base_seed: int,
        max_steps: int,
        field_size: tuple[int, int],
        workers: int,
    ) -> None:
        self.base_seed = int(base_seed)
        self.max_steps = int(max_steps)
        self.field_size = field_size
        self.workers = max(1, workers if workers > 0 else os.cpu_count() or 4)
        self._cache: dict[ThetaKey, EvalSummary] = {}
        self._total_simulations = 0

    @property
    def total_simulations(self) -> int:
        return self._total_simulations

    @property
    def num_unique_theta(self) -> int:
        return len(self._cache)

    def all_summaries(self) -> list[EvalSummary]:
        return list(self._cache.values())

    def best(self) -> EvalSummary | None:
        sums = self.all_summaries()
        return max(sums, key=lambda s: s.J) if sums else None

    def get_summary(self, theta: Theta) -> EvalSummary | None:
        return self._cache.get(theta_key(theta))

    def evaluate(self, theta: Theta) -> EvalSummary:
        return self.evaluate_many([theta])[0]

    def evaluate_many(
        self,
        thetas: list[Theta],
        *,
        progress_prefix: str | None = None,
        progress_threshold: int = 200,
    ) -> list[EvalSummary]:
        """Считает только те θ, которых нет в кэше; возвращает summaries в порядке `thetas`."""
        pending: list[Theta] = []
        for theta in thetas:
            k = theta_key(theta)
            if k not in self._cache:
                pending.append(theta)
                self._cache[k] = None  # type: ignore[assignment]

        if pending:
            tasks: list[tuple[Theta, int, tuple[int, int], int]] = [
                (theta, self.max_steps, self.field_size, self.base_seed) for theta in pending
            ]
            total = len(tasks)
            show_progress = progress_prefix is not None and total >= progress_threshold

            if self.workers == 1:
                results: list[RolloutMetrics] = []
                step = max(50, total // 6) if show_progress else 0
                for i, t in enumerate(tasks, start=1):
                    results.append(simulate_metrics_packed(t))
                    if show_progress and step and (i % step == 0 or i == total):
                        print(f"  {progress_prefix} симуляции: {i}/{total} θ", flush=True)
            else:
                results = []
                with ProcessPoolExecutor(max_workers=self.workers) as pool:
                    if show_progress:
                        batch_size = max(100, total // 6)
                        done = 0
                        for start in range(0, total, batch_size):
                            end = min(start + batch_size, total)
                            part = tasks[start:end]
                            part_cs = max(1, len(part) // (self.workers * 4))
                            results.extend(
                                pool.map(simulate_metrics_packed, part, chunksize=part_cs)
                            )
                            done = end
                            print(
                                f"  {progress_prefix} симуляции: {done}/{total} θ",
                                flush=True,
                            )
                    else:
                        chunksize = max(1, total // (self.workers * 8))
                        results = list(
                            pool.map(simulate_metrics_packed, tasks, chunksize=chunksize)
                        )

            for theta, metric in zip(pending, results):
                self._cache[theta_key(theta)] = _summary_from_metrics(theta, metric)
            self._total_simulations += total

        return [self._cache[theta_key(t)] for t in thetas]


def euclid(a: Theta, b: Theta) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
