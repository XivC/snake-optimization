from __future__ import annotations

from typing import cast

from player.policy import Theta

# Per-coordinate (anisotropic) trust region. Шире по «сильным» осям, уже по «шумным».
DEFAULT_RADII: tuple[float, float, float, float] = (0.5, 0.1, 0.5, 0.05)
MIN_RADII: tuple[float, float, float, float] = (0.05, 0.02, 0.05, 0.02)
MAX_RADII: tuple[float, float, float, float] = (1.5, 0.8, 1.5, 0.8)

GROW_FACTOR = 1.4
SHRINK_FACTOR = 0.7


def clip_anisotropic(
    theta: Theta,
    center: Theta,
    radii: tuple[float, float, float, float],
) -> Theta:
    """L∞-проекция в anisotropic шар вокруг center с радиусами по координатам."""
    return cast(
        Theta,
        tuple(
            min(center[i] + radii[i], max(center[i] - radii[i], theta[i]))
            for i in range(4)
        ),
    )


def inside_trust_region(
    theta: Theta,
    center: Theta,
    radii: tuple[float, float, float, float],
    *,
    eps: float = 1e-9,
) -> bool:
    return all(abs(theta[i] - center[i]) <= radii[i] + eps for i in range(4))


def grow_radius(r: float, *, max_r: float, factor: float = GROW_FACTOR) -> float:
    return min(max_r, r * factor)


def shrink_radius(r: float, *, min_r: float, factor: float = SHRINK_FACTOR) -> float:
    return max(min_r, r * factor)


def adapt_radii(
    radii: tuple[float, float, float, float],
    *,
    improved: bool,
    stalled: bool,
    min_radii: tuple[float, float, float, float] = MIN_RADII,
    max_radii: tuple[float, float, float, float] = MAX_RADII,
) -> tuple[float, float, float, float]:
    if improved:
        return cast(
            tuple[float, float, float, float],
            tuple(shrink_radius(radii[i], min_r=min_radii[i]) for i in range(4)),
        )
    if stalled:
        return cast(
            tuple[float, float, float, float],
            tuple(grow_radius(radii[i], max_r=max_radii[i]) for i in range(4)),
        )
    return radii
