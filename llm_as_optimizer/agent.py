from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from llm_as_optimizer.evaluator import Evaluator
from llm_as_optimizer.llm_client import DEFAULT_LLM_TIMEOUT_SEC, ask_agent
from llm_as_optimizer.memory import (
    AntiArchive,
    BeliefStore,
    EliteArchive,
)
from llm_as_optimizer.prompts import build_user_payload, user_message_content
from llm_as_optimizer.sampler import (
    DEFAULT_BOUNDS,
    THETA_DIM,
    lhs_box,
    sparse_axis_probes,
    stratified_box_corners,
)
from llm_as_optimizer.tools import ALLOWED_TOOLS, ToolContext, ToolResult, safe_dispatch
from llm_as_optimizer.trust_region import DEFAULT_RADII, MAX_RADII, adapt_radii

DEFAULT_TURNS = 12
DEFAULT_BOOTSTRAP_LHS = 12
DEFAULT_PATIENCE = 5
GAME_STEPS = 1000
FIELD_SIZE = (10, 10)
DEFAULT_WORKERS = 0

# Глобально-исследовательские инструменты (для подсчёта exploration share)
GLOBAL_EXPLORATION_TOOLS = {"novelty_search"}
# plane_scan / local_sweep считаются локальными по умолчанию (центрированы); novelty_search — глобальный


class _TeeTextIO:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return self._streams[0].isatty()

    def fileno(self) -> int:
        return self._streams[0].fileno()


def _fmt_theta(t: tuple[float, ...], *, decimals: int = 4) -> str:
    return "[" + ", ".join(f"{x:.{decimals}f}" for x in t) + "]"


def bootstrap_landscape(
    evaluator: Evaluator,
    elite: EliteArchive,
    anti: AntiArchive,
    *,
    rng: random.Random,
    n_lhs: int,
) -> dict[str, Any]:
    """Бесплатный для LLM bootstrap: LHS + corners + sparse axis-probes. Одна симуляция на θ."""
    sparse = sparse_axis_probes(amplitudes=(4.0,))
    samples = (
        lhs_box(n_lhs, rng=rng, bounds=DEFAULT_BOUNDS)
        + stratified_box_corners(rng, bounds=DEFAULT_BOUNDS)
        + sparse
    )
    from llm_as_optimizer.evaluator import theta_key as _k

    seen: set[tuple[float, ...]] = set()
    uniq: list = []
    for t in samples:
        kk = _k(t)
        if kk in seen:
            continue
        seen.add(kk)
        uniq.append(t)
    samples = uniq
    print(
        f"[bootstrap] LHS {n_lhs} + corners {2 ** THETA_DIM} + sparse {len(sparse)} = "
        f"{len(samples)} уникальных θ (по 1 симуляции, workers={evaluator.workers})",
        flush=True,
    )
    before = evaluator.total_simulations
    sums = evaluator.evaluate_many(samples, progress_prefix="[bootstrap]")
    spent = evaluator.total_simulations - before
    for s in sums:
        elite.add(s)
        cls = anti.classify_from_summary(s)
        if cls is not None:
            ftype, notes = cls
            anti.add(s, failure_type=ftype, notes=notes)
    best = elite.best()
    return {
        "evaluated": len(sums),
        "simulations_spent": spent,
        "best_J": best.J if best else 0.0,
        "best_theta": list(best.theta) if best else None,
    }


def run_agent(
    *,
    seed: int,
    turns: int,
    max_steps: int,
    workers: int,
    bootstrap_n: int,
    sim_budget: int | None,
    llm_timeout_sec: float,
    patience: int = DEFAULT_PATIENCE,
) -> dict[str, Any]:
    rng = random.Random(seed)
    evaluator = Evaluator(
        base_seed=seed,
        max_steps=max_steps,
        field_size=FIELD_SIZE,
        workers=workers,
    )
    elite = EliteArchive()
    anti = AntiArchive()
    beliefs = BeliefStore()
    ctx = ToolContext(evaluator=evaluator, elite=elite, anti=anti, beliefs=beliefs, rng=rng)

    boot = bootstrap_landscape(
        evaluator,
        elite,
        anti,
        rng=rng,
        n_lhs=bootstrap_n,
    )
    print(
        f"[bootstrap] лучший после старта: J={boot['best_J']:.4f} θ={_fmt_theta(tuple(boot['best_theta'] or [0]*4))}",
        flush=True,
    )

    trust_radii: tuple[float, float, float, float] = DEFAULT_RADII
    last_results_log: list[dict[str, Any]] = []
    last_hypothesis: str | None = None
    n_global_calls = 0
    n_total_calls = 0
    prev_best_j = elite.best().J if elite.best() else float("-inf")
    plateau_streak = 0

    for turn in range(1, turns + 1):
        if sim_budget is not None and evaluator.total_simulations >= sim_budget:
            print(
                f"[stop] исчерпан бюджет симуляций: {evaluator.total_simulations}/{sim_budget}",
                flush=True,
            )
            break
        if patience > 0 and plateau_streak >= patience:
            print(
                f"[stop] плато: best не растёт {plateau_streak} ходов подряд "
                f"(patience={patience}). best={prev_best_j:.4f}",
                flush=True,
            )
            break

        explore_share = (n_global_calls / n_total_calls) if n_total_calls > 0 else 0.0
        payload = build_user_payload(
            turn_1based=turn,
            total_turns=turns,
            simulations_spent=evaluator.total_simulations,
            sim_budget=sim_budget,
            elite=elite,
            anti=anti,
            beliefs=beliefs,
            all_summaries=evaluator.all_summaries(),
            last_tool_results=last_results_log,
            last_hypothesis=last_hypothesis,
            trust_radii=trust_radii,
            explore_share=explore_share,
            plateau_streak=plateau_streak,
            patience=patience if patience > 0 else None,
        )
        user_msg = user_message_content(payload)

        print(f"\n[turn {turn}/{turns}] запрос к LLM (таймаут {llm_timeout_sec:.0f} с)...", flush=True)
        try:
            response = ask_agent(user_msg, timeout_sec=llm_timeout_sec)
        except (RuntimeError, ValueError, TypeError, json.JSONDecodeError) as e:
            print(f"[turn {turn}] сбой/невалидный ответ LLM: {e}; пропускаем ход", flush=True)
            continue
        except Exception as e:  # noqa: BLE001
            print(f"[turn {turn}] сбой LLM/API ({type(e).__name__}): {e}; пропускаем ход", flush=True)
            continue

        hypothesis = str(response.get("hypothesis", "")).strip()
        confidence = response.get("confidence")
        reasoning = str(response.get("reasoning_brief", "")).strip()
        last_hypothesis = hypothesis or last_hypothesis

        # Belief updates
        for upd in response.get("belief_updates", []) or []:
            if not isinstance(upd, dict):
                continue
            stmt = str(upd.get("statement", "")).strip()
            act = str(upd.get("action", "")).strip().lower()
            conf = upd.get("confidence")
            if not stmt or act not in {"add", "reinforce", "weaken", "remove"}:
                continue
            beliefs.apply(statement=stmt, confidence=conf if conf is not None else None, action=act)  # type: ignore[arg-type]

        # Tool dispatch
        tool_calls = response.get("tool_calls") or []
        executed: list[ToolResult] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tname = str(tc.get("tool", "")).strip()
            targs = tc.get("args") or {}
            if tname not in ALLOWED_TOOLS:
                print(f"            пропуск неизвестного tool: {tname!r}", flush=True)
                continue
            print(
                f"            tool: {tname}  args={json.dumps(targs, ensure_ascii=False)[:240]}",
                flush=True,
            )
            res = safe_dispatch(tname, targs, ctx)
            executed.append(res)
            n_total_calls += 1
            if tname in GLOBAL_EXPLORATION_TOOLS:
                n_global_calls += 1
            ok_tag = "ok" if res.ok else f"ERR: {res.payload.get('error', '?')}"
            print(
                f"               -> {ok_tag}  simulations_spent={res.simulations_spent}",
                flush=True,
            )
            if sim_budget is not None and evaluator.total_simulations >= sim_budget:
                print(
                    f"               остановка: достигнут sim_budget={sim_budget}",
                    flush=True,
                )
                break

        last_results_log.extend(r.to_dict() for r in executed)
        last_results_log = last_results_log[-4:]

        cur_best = elite.best()
        cur_best_j = cur_best.J if cur_best else float("-inf")
        improved = cur_best_j - prev_best_j >= 0.5
        stalled = (cur_best_j - prev_best_j) < 1e-3
        trust_radii = adapt_radii(trust_radii, improved=improved, stalled=stalled, max_radii=MAX_RADII)
        if improved:
            plateau_streak = 0
        else:
            plateau_streak += 1
        prev_best_j = cur_best_j

        explore_share = (n_global_calls / n_total_calls) if n_total_calls > 0 else 0.0
        plateau_tag = (
            f" | plateau={plateau_streak}/{patience}"
            if patience > 0
            else f" | plateau={plateau_streak}"
        )
        print(
            f"[turn {turn}/{turns}] best={cur_best_j:.4f} θ*={_fmt_theta(cur_best.theta) if cur_best else '?'} | "
            f"sims_total={evaluator.total_simulations}{'/'+str(sim_budget) if sim_budget else ''} | "
            f"explore_share={explore_share:.2f} | trust_radii={[round(x,3) for x in trust_radii]}"
            f"{plateau_tag}",
            flush=True,
        )
        if hypothesis:
            print(f"            hypothesis ({confidence}): {hypothesis}", flush=True)
        if reasoning:
            print(f"            reasoning: {reasoning[:300]}", flush=True)
        print(
            f"            elite_top5: "
            + ", ".join(
                f"J={s.J:.2f} θ={_fmt_theta(s.theta, decimals=3)}" for s in elite.top_k(5)
            ),
            flush=True,
        )

    best = elite.best()
    summary = {
        "best": best.to_dict() if best else None,
        "total_simulations": evaluator.total_simulations,
        "unique_theta": evaluator.num_unique_theta,
        "elite_top": [s.to_dict() for s in elite.top_k(10)],
        "anti_top": anti.to_payload(k=5),
        "beliefs": beliefs.to_payload(k=8),
    }
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM Research Agent для оптимизации весов Snake-политики (tool-calling)"
    )
    p.add_argument("--seed", type=int, required=True, help="RNG seed")
    p.add_argument("--turns", type=int, default=DEFAULT_TURNS, help="Сколько обращений к LLM (бюджет ходов)")
    p.add_argument("--steps", type=int, default=GAME_STEPS, help="Лимит шагов на одну партию Snake")
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Параллельных процессов (0 = число ядер, 1 = без параллелизма)",
    )
    p.add_argument(
        "--bootstrap-n",
        type=int,
        default=DEFAULT_BOOTSTRAP_LHS,
        help="Стартовый LHS-набор θ (плюс 16 corners + 26 sparse). 0 — пропустить bootstrap.",
    )
    p.add_argument(
        "--sim-budget",
        type=int,
        default=None,
        metavar="N",
        help="Опциональный максимум симуляций (= уникальных θ). Достигли — стоп.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        metavar="K",
        help=(
            "Early-stop: после K turn-ов без улучшения best J — выход. "
            "0 = выключить (как раньше). По умолчанию 5."
        ),
    )
    p.add_argument(
        "--llm-timeout",
        type=float,
        default=DEFAULT_LLM_TIMEOUT_SEC,
        metavar="SEC",
        help="Таймаут HTTP к API LLM (сек.)",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default="llm_as_optimizer/logs",
        metavar="DIR",
        help="Каталог логов (файл: agent_seed_<seed>_<timestamp>.log)",
    )
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"agent_seed_{args.seed}_{ts}.log"
    log_fp = log_path.open("w", encoding="utf-8")
    orig_out = sys.stdout
    orig_err = sys.stderr
    try:
        sys.stdout = _TeeTextIO(orig_out, log_fp)
        sys.stderr = _TeeTextIO(orig_err, log_fp)
        print(f"[log] весь вывод дублируется в файл: {log_path.resolve()}", flush=True)
        llm_to = float(os.environ.get("LLM_TIMEOUT_SEC", str(DEFAULT_LLM_TIMEOUT_SEC))) if args.llm_timeout <= 0 else args.llm_timeout
        result = run_agent(
            seed=args.seed,
            turns=max(1, args.turns),
            max_steps=args.steps,
            workers=args.workers,
            bootstrap_n=max(0, args.bootstrap_n),
            sim_budget=args.sim_budget,
            llm_timeout_sec=max(30.0, llm_to),
            patience=max(0, args.patience),
        )
        print("\n--- итог ---", flush=True)
        if result["best"]:
            print(
                f"Best: J={result['best']['J']} θ={result['best']['theta']} "
                f"(apples={result['best']['apples']}, steps={result['best']['steps']}, died={result['best']['died']})",
                flush=True,
            )
        print(
            f"Всего симуляций: {result['total_simulations']}; "
            f"уникальных θ в кэше: {result['unique_theta']}",
            flush=True,
        )
        print("Топ-10 elite:", flush=True)
        for row in result["elite_top"]:
            print(f"  J={row['J']}  θ={row['theta']}", flush=True)
        if result["beliefs"]:
            print("Гипотезы (beliefs):", flush=True)
            for b in result["beliefs"]:
                print(f"  ({b['confidence']:.2f}, support={b['support']}) {b['statement']}", flush=True)
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
        log_fp.close()


if __name__ == "__main__":
    main()
