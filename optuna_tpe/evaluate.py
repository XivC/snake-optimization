"""
Финальная оценка θ на фиксированных seeds.

ФАЗА 2 — ФИНАЛЬНАЯ ОЦЕНКА:
Запускается ПОСЛЕ optimizer.py.
Параметры здесь НЕ меняются — только честная проверка.

Запуск:
    uv run python optuna_tpe/evaluate.py

Запуск с кастомными весами:
    uv run python optuna_tpe/evaluate.py 12.85 0.776 5.895 -0.796
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from player import simulate

# --- Фиксированные параметры ---
EVAL_SEEDS = [1351, 2162, 6284, 1642, 3227, 1888, 1045, 1234, 5555, 9999]
FIELD_SIZE = (10, 10)
MAX_STEPS = 1000


def evaluate(theta: tuple, label: str = "") -> list:
    print(f"\n{'='*50}")
    print(f"Оценка θ = {theta}" + (f"  [{label}]" if label else ""))
    print(f"{'='*50}")
    print(f"{'Seed':>8}  {'J(θ)':>8}")
    print("-" * 20)

    scores = []
    for s in EVAL_SEEDS:
        j = simulate(theta, max_steps=MAX_STEPS, seed=s, field_size=FIELD_SIZE)
        scores.append(j)
        print(f"{s:>8}  {j:>8.1f}")

    print("-" * 20)
    print(f"{'Среднее':>8}  {np.mean(scores):>8.3f}")
    print(f"{'Медиана':>8}  {np.median(scores):>8.3f}")
    print(f"{'Мин':>8}  {np.min(scores):>8.1f}")
    print(f"{'Макс':>8}  {np.max(scores):>8.1f}")
    return scores


def save_eval_results(
    results_dir: Path,
    theta: tuple,
    scores: list,
    baseline_scores: list,
) -> None:
    """Сохраняет финальные результаты оценки."""
    out = {
        "method": "Optuna TPE",
        "eval_seeds": EVAL_SEEDS,
        "max_steps": MAX_STEPS,
        "field_size": list(FIELD_SIZE),
        "best_theta": {
            "w_food":   theta[0],
            "w_danger": theta[1],
            "w_space":  theta[2],
            "w_wall":   theta[3],
        },
        "results_per_seed": [
            {"seed": s, "J": j}
            for s, j in zip(EVAL_SEEDS, scores)
        ],
        "summary": {
            "mean":   float(np.mean(scores)),
            "median": float(np.median(scores)),
            "min":    float(np.min(scores)),
            "max":    float(np.max(scores)),
        },
        "baseline_summary": {
            "mean":   float(np.mean(baseline_scores)),
            "median": float(np.median(baseline_scores)),
            "min":    float(np.min(baseline_scores)),
            "max":    float(np.max(baseline_scores)),
        },
        "improvement_vs_baseline": round(
            float(np.mean(scores)) - float(np.mean(baseline_scores)), 3
        ),
    }

    out_path = results_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nФинальные результаты сохранены в {out_path}")


def main():
    # Кастомные веса из аргументов
    if len(sys.argv) == 5:
        theta = tuple(float(x) for x in sys.argv[1:])
        evaluate(theta, label="из аргументов")
        return

    print("=== Финальная оценка на фиксированных seeds ===")
    print(f"Eval seeds: {EVAL_SEEDS}")
    print(f"MAX_STEPS: {MAX_STEPS}\n")

    # Baseline
    baseline = (6.1540, -0.0650, -6.1440, 0.0720)
    baseline_scores = evaluate(baseline, label="baseline (дефолт)")

    # Лучшие веса из optimizer.py
    results_path = Path(__file__).parent / "results" / "best_theta.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)

        best_theta = tuple(data["theta"])
        scores = evaluate(best_theta, label="Optuna TPE best")

        print(f"\nTrials завершено : {data.get('n_trials_completed', '?')}")
        print(f"Train seed range : {data.get('train_seed_range', '?')}")
        print(f"Train score      : {data.get('train_score', '?'):.3f}")
        print("\nТоп-3 найденных θ (по train score):")
        for i, t in enumerate(data.get("top10", [])[:3], 1):
            p = t["params"]
            print(
                f"  {i}. train J={t['value']:.3f}"
                f"  θ=({p['w1']:.4f}, {p['w2']:.4f},"
                f" {p['w3']:.4f}, {p['w4']:.4f})"
            )

        # Сохраняем финальные результаты
        results_dir = Path(__file__).parent / "results"
        save_eval_results(results_dir, best_theta, scores, baseline_scores)

    else:
        print("\nФайл results/best_theta.json не найден.")
        print("Сначала запустите: uv run python optuna_tpe/optimizer.py")


if __name__ == "__main__":
    main()