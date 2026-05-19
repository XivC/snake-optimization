"""
Визуализация результатов Optuna TPE.

Запуск:
    uv run python optuna_tpe/plot.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import csv

results_dir = Path(__file__).parent / "results"

# ── 1. Загрузка данных ──────────────────────────────────────────────

# eval_results.json — результаты по фиксированным seeds
with open(results_dir / "eval_results.json") as f:
    eval_data = json.load(f)

seeds = [r["seed"] for r in eval_data["results_per_seed"]]
tpe_scores = [r["J"] for r in eval_data["results_per_seed"]]
baseline_scores = [11, 15, 14, 12, 29, 14, 10, 11, 19, 2]  # из evaluate.py

# results.csv — история trials
trial_numbers = []
trial_scores = []
with open(results_dir / "results.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        trial_numbers.append(int(row["trial_number"]))
        trial_scores.append(float(row["J"]))

# ── 2. График 1: Baseline vs TPE по seeds ───────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Optuna TPE — результаты оптимизации", fontsize=14, fontweight="bold")

ax1 = axes[0]
x = np.arange(len(seeds))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_scores, width, label="Baseline", color="#64748B", alpha=0.85)
bars2 = ax1.bar(x + width/2, tpe_scores, width, label="Optuna TPE", color="#0EA5E9", alpha=0.9)

ax1.set_title("J(θ) по фиксированным seeds", fontsize=12)
ax1.set_xlabel("Seed")
ax1.set_ylabel("J(θ) — яблок съедено")
ax1.set_xticks(x)
ax1.set_xticklabels([str(s) for s in seeds], rotation=45, ha="right")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)
ax1.set_ylim(0, max(max(tpe_scores), max(baseline_scores)) * 1.15)

# Подписи среднего
mean_baseline = np.mean(baseline_scores)
mean_tpe = np.mean(tpe_scores)
ax1.axhline(mean_baseline, color="#64748B", linestyle="--", linewidth=1, alpha=0.7)
ax1.axhline(mean_tpe, color="#0EA5E9", linestyle="--", linewidth=1, alpha=0.7)
ax1.text(len(seeds) - 0.5, mean_baseline + 0.5, f"avg {mean_baseline:.1f}", color="#64748B", fontsize=9)
ax1.text(len(seeds) - 0.5, mean_tpe + 0.5, f"avg {mean_tpe:.1f}", color="#0EA5E9", fontsize=9)

# ── 3. График 2: История trials (сходимость) ────────────────────────

ax2 = axes[1]

# Накопленный максимум — показывает как рос лучший результат
running_best = []
current_best = -999
for s in trial_scores:
    if s > current_best:
        current_best = s
    running_best.append(current_best)

ax2.scatter(trial_numbers, trial_scores, alpha=0.25, s=8, color="#94A3B8", label="J каждого trial")
ax2.plot(trial_numbers, running_best, color="#0EA5E9", linewidth=2, label="Лучший J (накопленный)")
ax2.set_title("Сходимость оптимизатора", fontsize=12)
ax2.set_xlabel("Trial №")
ax2.set_ylabel("J(θ)")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(0, max(trial_scores) * 1.15)

plt.tight_layout()

out_path = results_dir / "plots.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"График сохранён в {out_path}")
plt.show()