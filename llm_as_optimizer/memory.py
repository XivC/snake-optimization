from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Literal

from player.policy import Theta

from llm_as_optimizer.evaluator import EvalSummary, theta_key

ELITE_MAX = 12
ANTI_MAX = 12
BELIEFS_MAX = 16

BeliefAction = Literal["add", "reinforce", "weaken", "remove"]


@dataclass
class EliteEntry:
    summary: EvalSummary

    def to_dict(self) -> dict[str, object]:
        d = self.summary.to_dict()
        d["tag"] = "elite"
        return d


@dataclass
class EliteArchive:
    max_size: int = ELITE_MAX
    entries: dict[tuple[float, ...], EliteEntry] = field(default_factory=dict)

    def add(self, summary: EvalSummary) -> None:
        k = theta_key(summary.theta)
        prev = self.entries.get(k)
        if prev is None or summary.J > prev.summary.J:
            self.entries[k] = EliteEntry(summary=summary)
        self._trim()

    def add_many(self, summaries: list[EvalSummary]) -> None:
        for s in summaries:
            self.add(s)

    def _trim(self) -> None:
        if len(self.entries) <= self.max_size:
            return
        ordered = sorted(self.entries.items(), key=lambda kv: -kv[1].summary.J)
        self.entries = dict(ordered[: self.max_size])

    def best(self) -> EvalSummary | None:
        if not self.entries:
            return None
        return max((e.summary for e in self.entries.values()), key=lambda s: s.J)

    def top_k(self, k: int) -> list[EvalSummary]:
        if k <= 0:
            return []
        return sorted(
            (e.summary for e in self.entries.values()),
            key=lambda s: -s.J,
        )[:k]

    def to_payload(self, k: int = 5) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for i, s in enumerate(self.top_k(k), start=1):
            row = s.to_dict()
            row["rank"] = i
            out.append(row)
        return out


@dataclass
class AntiEntry:
    summary: EvalSummary
    failure_type: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        d = self.summary.to_dict()
        d["failure_type"] = self.failure_type
        d["notes"] = self.notes
        return d


@dataclass
class AntiArchive:
    max_size: int = ANTI_MAX
    entries: dict[tuple[float, ...], AntiEntry] = field(default_factory=dict)

    def add(self, summary: EvalSummary, *, failure_type: str, notes: str) -> None:
        k = theta_key(summary.theta)
        prev = self.entries.get(k)
        if prev is None or summary.J < prev.summary.J:
            self.entries[k] = AntiEntry(summary=summary, failure_type=failure_type, notes=notes)
        self._trim()

    def _trim(self) -> None:
        if len(self.entries) <= self.max_size:
            return
        ordered = sorted(self.entries.items(), key=lambda kv: kv[1].summary.J)
        self.entries = dict(ordered[: self.max_size])

    def worst_k(self, k: int) -> list[AntiEntry]:
        return sorted(self.entries.values(), key=lambda e: e.summary.J)[:k]

    def to_payload(self, k: int = 5) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for i, e in enumerate(self.worst_k(k), start=1):
            row = e.to_dict()
            row["rank"] = i
            out.append(row)
        return out

    def classify_from_summary(self, summary: EvalSummary) -> tuple[str, str] | None:
        """Эвристика: что считать «провалом»? В детерминированном режиме это один прогон."""
        if summary.died and summary.J <= 0.5:
            return (
                "catastrophic_death",
                f"died after {summary.steps} steps, apples={summary.apples}",
            )
        if summary.apples <= 0 and summary.died:
            return (
                "starvation",
                f"apples=0, died after {summary.steps} steps",
            )
        if summary.died and summary.apples <= 2:
            return (
                "early_death",
                f"died with only {summary.apples} apples after {summary.steps} steps",
            )
        return None


@dataclass
class Belief:
    statement: str
    confidence: float
    support: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "statement": self.statement,
            "confidence": round(float(self.confidence), 3),
            "support": int(self.support),
        }


@dataclass
class BeliefStore:
    max_size: int = BELIEFS_MAX
    items: dict[str, Belief] = field(default_factory=dict)

    def apply(self, *, statement: str, confidence: float | None, action: BeliefAction) -> None:
        key = statement.strip().lower()
        if not key:
            return
        if action == "remove":
            self.items.pop(key, None)
            return
        prev = self.items.get(key)
        if action == "add":
            c = float(confidence) if confidence is not None else 0.6
            if prev is None:
                self.items[key] = Belief(statement=statement.strip(), confidence=max(0.0, min(1.0, c)))
            else:
                prev.confidence = max(prev.confidence, max(0.0, min(1.0, c)))
                prev.support += 1
        elif action == "reinforce":
            if prev is None:
                c = float(confidence) if confidence is not None else 0.7
                self.items[key] = Belief(statement=statement.strip(), confidence=max(0.0, min(1.0, c)))
            else:
                prev.confidence = max(0.0, min(1.0, prev.confidence + 0.1))
                prev.support += 1
        elif action == "weaken":
            if prev is not None:
                prev.confidence = max(0.0, prev.confidence - 0.15)
                if prev.confidence < 0.1:
                    self.items.pop(key, None)
        self._trim()

    def _trim(self) -> None:
        if len(self.items) <= self.max_size:
            return
        ordered = sorted(
            self.items.items(),
            key=lambda kv: (-kv[1].confidence, -kv[1].support),
        )
        self.items = dict(ordered[: self.max_size])

    def to_payload(self, k: int = 8) -> list[dict[str, object]]:
        if k <= 0:
            return []
        ordered = sorted(
            self.items.values(),
            key=lambda b: (-b.confidence, -b.support),
        )[:k]
        return [b.to_dict() for b in ordered]


def sensitivity_summary(top_k: list[EvalSummary]) -> dict[str, object]:
    """Простая «важность координат»: дисперсия по каждой оси среди элиты + диапазоны."""
    if not top_k:
        return {"per_coord_std": [0.0] * 4, "per_coord_range": [[0.0, 0.0]] * 4, "n": 0}
    cols: list[list[float]] = [[s.theta[d] for s in top_k] for d in range(4)]
    stds = [statistics.pstdev(c) if len(c) >= 2 else 0.0 for c in cols]
    ranges = [[min(c), max(c)] for c in cols]
    return {
        "per_coord_std": [round(x, 3) for x in stds],
        "per_coord_range": [[round(a, 3), round(b, 3)] for a, b in ranges],
        "n": len(top_k),
    }


def coordinate_observations(all_summaries: list[EvalSummary], *, min_points: int = 12) -> str:
    """Компактный per-axis сигнал: 'θ0:+1.4, θ1:0 zero-cluster, θ2:-0.6, θ3:0 zero-cluster' (≈80 char).

    Берём разницу средних между верхней и нижней четвертью оценённых θ + флаг «high-J solutions
    cluster near 0», когда модуль элиты по оси << модуль низа.
    """
    if len(all_summaries) < min_points:
        return f"n={len(all_summaries)}<{min_points}: no axis signal yet"
    by_desc = sorted(all_summaries, key=lambda s: -s.J)
    q = max(2, len(by_desc) // 4)
    hi = by_desc[:q]
    lo = by_desc[-q:]
    parts: list[str] = []
    for d in range(4):
        mh = statistics.fmean(s.theta[d] for s in hi)
        ml = statistics.fmean(s.theta[d] for s in lo)
        diff = mh - ml
        abs_hi = statistics.fmean(abs(s.theta[d]) for s in hi)
        abs_lo = statistics.fmean(abs(s.theta[d]) for s in lo)
        sign = "+" if diff > 0 else ("-" if diff < 0 else "0")
        marker = f"θ{d}:{sign}{abs(diff):.1f}"
        if abs_hi <= 1.0 and abs_lo - abs_hi >= 0.8:
            marker += " zero-cluster"
        parts.append(marker)
    return ", ".join(parts)
