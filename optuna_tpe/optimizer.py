"""
Optuna TPE оптимизатор весов θ для линейной политики змейки.

Оптимизатор прогоняется на случайных seeds (1–1000).
Один случайный seed на каждый trial.
Когда найдены хорошие θ — переходим к evaluate.py.

Запуск:
    uv run python optuna_tpe/optimizer.py
"""

import csv
import json
import random
import sys
import threading
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).parent.parent))

from player import simulate

# --- Конфигурация ---

TRAIN_SEED_RANGE = (1, 1000)
FIELD_SIZE = (10, 10)
MAX_STEPS = 1000
N_TRIALS = 500
TIMEOUT_PER_SEED = 30.0
PENALTY = 0.0

BOUNDS = {
    "w1": (0.0,  20.0),
    "w2": (-5.0,  5.0),
    "w3": (-10.0, 15.0),
    "w4": (-5.0,  5.0),
}


def safe_simulate(theta: tuple, seed: int) -> float:
    result = [PENALTY]

    def _run():
        result[0] = simulate(
            theta, max_steps=MAX_STEPS, seed=seed, field_size=FIELD_SIZE
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=TIMEOUT_PER_SEED)
    return result[0] if not t.is_alive() else PENALTY


def objective(trial: optuna.Trial) -> float:
    """Один случайный seed на каждый trial из диапазона 1–1000."""
    theta = (
        trial.suggest_float("w1", *BOUNDS["w1"]),
        trial.suggest_float("w2", *BOUNDS["w2"]),
        trial.suggest_float("w3", *BOUNDS["w3"]),
        trial.suggest_float("w4", *BOUNDS["w4"]),
    )
    train_seed = random.randint(*TRAIN_SEED_RANGE)
    j = safe_simulate(theta, seed=train_seed)

    # Сохраняем seed в trial для логирования
    trial.set_user_attr("train_seed", train_seed)
    trial.set_user_attr("j", j)

    return j


def save_meta(results_dir: Path) -> None:
    """Сохраняет мета-информацию о запуске."""
    meta = {
        "version": 1,
        "field_height": FIELD_SIZE[0],
        "field_width": FIELD_SIZE[1],
        "tpe_hyperparams": {
            "train_seed_range": list(TRAIN_SEED_RANGE),
            "n_trials": N_TRIALS,
            "max_steps": MAX_STEPS,
            "timeout_per_seed": TIMEOUT_PER_SEED,
            "bounds": BOUNDS,
            "sampler_seed": 42,
            "warm_start_points": 3,
        },
    }
    out_path = results_dir / "meta.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)


def save_best(results_dir: Path, theta: tuple, j: float) -> None:
    """Сохраняет лучшие найденные веса."""
    best = {
        "w_food":   theta[0],
        "w_danger": theta[1],
        "w_space":  theta[2],
        "w_wall":   theta[3],
        "J": j,
        "aborted": False,
    }
    out_path = results_dir / "best.json"
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)


def save_csv(results_dir: Path, study: optuna.Study) -> None:
    """Сохраняет историю всех trials в results.csv."""
    out_path = results_dir / "results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "w_food", "w_danger", "w_space", "w_wall",
            "J", "steps", "train_seed", "trial_number",
        ])
        for t in study.trials:
            if t.value is None:
                continue
            p = t.params
            writer.writerow([
                p.get("w1", ""),
                p.get("w2", ""),
                p.get("w3", ""),
                p.get("w4", ""),
                t.value,
                MAX_STEPS,
                t.user_attrs.get("train_seed", ""),
                t.number,
            ])


def save_result(study: optuna.Study, results_dir: Path) -> tuple:
    best_params = study.best_params
    best_theta = (
        best_params["w1"],
        best_params["w2"],
        best_params["w3"],
        best_params["w4"],
    )
    best_score = study.best_value

    results_dir.mkdir(exist_ok=True)

    # best_theta.json — для evaluate.py
    out_path = results_dir / "best_theta.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "theta": best_theta,
                "train_score": best_score,
                "train_seed_range": list(TRAIN_SEED_RANGE),
                "field_size": list(FIELD_SIZE),
                "max_steps": MAX_STEPS,
                "n_trials_completed": len(study.trials),
                "top10": [
                    {"number": t.number, "value": t.value, "params": t.params}
                    for t in sorted(
                        study.trials, key=lambda t: -(t.value or -999)
                    )[:10]
                ],
            },
            f,
            indent=2,
        )

    # best.json
    save_best(results_dir, best_theta, best_score)

    # results.csv — история всех trials
    save_csv(results_dir, study)

    print(f"\nЛучшие веса θ    = {best_theta}")
    print(f"Train score      = {best_score:.4f}")
    print(f"Train seed range = {TRAIN_SEED_RANGE}")
    print(f"Trials завершено = {len(study.trials)}")
    print(f"Сохранено в      {results_dir}/")
    print(f"  best_theta.json  — для evaluate.py")
    print(f"  best.json        — лучшие веса")
    print(f"  meta.json        — параметры запуска")
    print(f"  results.csv      — история trials")
    print(f"\nТеперь запустите evaluate.py для проверки на фиксированных seeds.")
    return best_theta, best_score


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=== Optuna TPE — фаза разработки (случайные seeds 1–1000) ===")
    print(f"Поле: {FIELD_SIZE}, макс. шагов: {MAX_STEPS}")
    print(f"Каждый trial — новый случайный seed из диапазона {TRAIN_SEED_RANGE}")
    print(f"Trials: {N_TRIALS}  (Ctrl+C = сохранить и выйти)\n")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    save_meta(results_dir)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.enqueue_trial({"w1": 12.850, "w2":  0.776, "w3":  5.895, "w4": -0.796})
    study.enqueue_trial({"w1": 16.946, "w2":  1.116, "w3":  4.543, "w4": -0.843})
    study.enqueue_trial({"w1": 13.493, "w2":  0.629, "w3":  2.841, "w4": -0.456})

    best_so_far = -999.0

    def print_progress(study: optuna.Study, trial: optuna.Trial) -> None:
        nonlocal best_so_far
        if trial.value is not None and trial.value > best_so_far:
            best_so_far = trial.value

        if trial.number > 0 and trial.number % 50 == 0:
            save_result(study, results_dir)
            print(f"  [автосохранение на trial {trial.number}]\n")

        if trial.number % 5 == 0:
            print(
                f"  Trial {trial.number:>4}/{N_TRIALS}"
                f"  текущий={trial.value:6.2f}"
                f"  лучший={best_so_far:6.2f}"
                f"  θ={tuple(round(v, 3) for v in study.best_params.values())}"
            )

    try:
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[print_progress])
    except KeyboardInterrupt:
        print("\n\nОстановлено — сохраняем лучший результат...")

    save_result(study, results_dir)


if __name__ == "__main__":
    main()