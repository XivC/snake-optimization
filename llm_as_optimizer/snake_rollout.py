"""Локальная симуляция Snake с полной телеметрией (без зависимости от ``player.rollout``)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from game import SnakeGame
from game.models import GameStatus

from player.player import Player

type PackedRollout = tuple[tuple[float, float, float, float], int, tuple[int, int], int]

DEATH_PENALTY = 1.0


def _score_objective(apples: int, died: bool) -> float:
    return float(apples) - (DEATH_PENALTY if died else 0.0)


@dataclass(frozen=True, slots=True)
class RolloutMetrics:
    J: float
    apples: int
    steps: int
    died: bool


def simulate_metrics_packed(packed: PackedRollout) -> RolloutMetrics:
    """Одна партия для пула: (θ, max_steps, field_size, seed) — как в ``evaluator.Evaluator``."""
    theta, max_steps, field_size, seed = packed
    rng = random.Random(seed)
    game = SnakeGame(rng, field_size)
    Player(game, theta, rng, max_steps=max_steps).play()
    final = game.get_state()
    apples = final.score
    died = final.status is GameStatus.GAME_FAILED
    return RolloutMetrics(
        J=_score_objective(apples, died),
        apples=int(apples),
        steps=int(final.step),
        died=bool(died),
    )
