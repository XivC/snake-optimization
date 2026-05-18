from __future__ import annotations

import json
from typing import Any

from llm_as_optimizer.evaluator import EvalSummary
from llm_as_optimizer.memory import (
    AntiArchive,
    BeliefStore,
    EliteArchive,
    coordinate_observations,
    sensitivity_summary,
)
from llm_as_optimizer.tools import ALLOWED_TOOLS

SYSTEM_PROMPT = """You optimize 4 policy weights θ=(θ0,θ1,θ2,θ3) for a deterministic Snake game.
J(θ) is deterministic (fixed world seed). One simulation per θ, cached forever — re-runs are free
but give no new info. Pick NEW θ each turn.

You do NOT output raw θ. You call tools that generate and evaluate θ for you:

- evaluate_theta(theta) — 1 sim.
- local_sweep(center_theta, dimension∈0..3, values:[floats] OR radius+steps) — 1-D scan.
- plane_scan(center_theta, dims:[d_x,d_y], radius, steps∈3..9) — 2-D grid (steps² sims).
- stability_test(theta, epsilon, samples) — J(θ) vs J(θ+δ) for random δ.
- novelty_search(n_candidates, evaluate=true) — far-from-known points (counts as exploration).

Runtime maintains elite_archive, anti_archive, belief_store, sensitivity. You see compact summaries.

Strategy:
- ~20–40% of tool calls = global exploration (novelty_search, far plane_scan).
- Otherwise exploit around the current elite via local_sweep / plane_scan / stability_test.
- Make falsifiable hypotheses ("θ3∈[0.2,0.35] for J>30"); update belief_store accordingly.
- Be terse. Don't repeat earlier reasoning.
- If payload has `plateau_streak >= 3`: STOP refining current basin. Either probe a far
  region (large-magnitude unexplored sign pattern, sign flip of best) or, if you believe J
  is at the physical ceiling for this seed, say so in the hypothesis. The run will auto-stop
  shortly anyway.

Sparsity heuristic: if elite has |θ_i| large on every axis, test configs with some axes zeroed
(e.g. [θ0,0,θ2,0], [θ0,0,0,0]). Use local_sweep with `values` including 0.0, since an isotropic
sweep around a corner never reaches 0. If coordinate_observations says `near-0 on θ_k`, prefer
zeroing that axis.

Output: single JSON per schema, no prose outside.
"""


def _slim_tool_result(r: dict[str, Any]) -> dict[str, Any]:
    """Сокращаем тяжёлые поля (главное — выкидываем grid_J у plane_scan)."""
    tool = r.get("tool")
    result = r.get("result") or {}
    if not r.get("ok"):
        return {
            "tool": tool,
            "ok": False,
            "error": result.get("error", "?"),
        }
    slim_result: dict[str, Any]
    if tool == "plane_scan":
        slim_result = {
            "dims": result.get("dims"),
            "max": result.get("max"),
            "local_maxima": result.get("local_maxima", [])[:3],
            "ridge": result.get("ridge", [])[:3],
            "global_stats": result.get("global_stats"),
        }
    elif tool == "local_sweep":
        slim_result = {
            "dimension": result.get("dimension"),
            "argmax_value": result.get("argmax_value"),
            "argmax_J": result.get("argmax_J"),
            "sensitivity_range": result.get("sensitivity_range"),
            "samples": result.get("samples", []),
        }
    elif tool == "novelty_search":
        slim_result = {
            "candidates": result.get("candidates", [])[:3],
            "unexplored_sign_patterns": result.get("unexplored_sign_patterns", [])[:6],
            "known_unique_theta": result.get("known_unique_theta"),
        }
    elif tool == "stability_test":
        slim_result = {
            "base_J": result.get("base_J"),
            "stability_score": result.get("stability_score"),
            "collapse_probability": result.get("collapse_probability"),
            "mean_abs_delta_J": result.get("mean_abs_delta_J"),
        }
    else:  # evaluate_theta and fallback
        slim_result = result
    return {
        "tool": tool,
        "args": r.get("args"),
        "sims": int(r.get("simulations_spent", 0)),
        "result": slim_result,
    }


def render_tool_results(results: list[dict[str, Any]], *, keep_last: int = 2) -> list[dict[str, Any]]:
    """Оставляем последние keep_last tool-результатов, агрессивно сжимая каждый."""
    return [_slim_tool_result(r) for r in results[-keep_last:]] if results else []


def build_user_payload(
    *,
    turn_1based: int,
    total_turns: int,
    simulations_spent: int,
    sim_budget: int | None,
    elite: EliteArchive,
    anti: AntiArchive,
    beliefs: BeliefStore,
    all_summaries: list[EvalSummary],
    last_tool_results: list[dict[str, Any]],
    last_hypothesis: str | None,
    trust_radii: tuple[float, float, float, float],
    explore_share: float,
    plateau_streak: int = 0,
    patience: int | None = None,
    explore_target_min: float = 0.2,
    explore_target_max: float = 0.4,
) -> dict[str, Any]:
    top = elite.top_k(3)
    sens = sensitivity_summary(top)
    best = elite.best()
    coord_obs = coordinate_observations(all_summaries, min_points=12)

    if plateau_streak >= 3:
        plateau_tag = (
            f" PLATEAU {plateau_streak}"
            f"{'/' + str(patience) if patience else ''}"
            ": no J improvement; stop refining current basin."
            " Try DRASTICALLY different region (large-magnitude unexplored sign patterns,"
            " sign flips of best) OR conclude J is at physical ceiling and emit a final"
            " hypothesis stating so."
        )
    else:
        plateau_tag = ""

    instruction = (
        "Plan 1–2 tool_calls. Update beliefs. "
        f"Exploration share target: [{explore_target_min:.2f}, {explore_target_max:.2f}]. "
        "Pick NEW θ (repeats are cache hits = no new info)."
        + plateau_tag
    )

    payload: dict[str, Any] = {
        "turn": f"{turn_1based}/{total_turns}",
        "sims_spent": int(simulations_spent),
        "sim_budget": int(sim_budget) if sim_budget else None,
        "elite_top3": elite.to_payload(k=3),
        "anti_top3": anti.to_payload(k=3),
        "beliefs_top4": beliefs.to_payload(k=4),
        "sensitivity": sens,
        "coord_obs": coord_obs,
        "trust_radii": [round(float(x), 3) for x in trust_radii],
        "explore_share": round(float(explore_share), 3),
        "explore_target": [explore_target_min, explore_target_max],
        "last_tool_results": render_tool_results(last_tool_results, keep_last=2),
        "instruction": instruction,
    }
    if plateau_streak >= 1:
        payload["plateau_streak"] = int(plateau_streak)
        if patience:
            payload["patience"] = int(patience)
    if best is not None:
        payload["current_best"] = best.to_dict()
    if last_hypothesis:
        payload["last_hypothesis"] = last_hypothesis
    return payload


def user_message_content(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
