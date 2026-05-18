from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from llm_as_optimizer.prompts import SYSTEM_PROMPT
from llm_as_optimizer.tools import ALLOWED_TOOLS

load_dotenv()

POLZA_BASE_URL = "https://polza.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3.6-35b-a3b"
DEFAULT_LLM_TIMEOUT_SEC = 240.0


def _response_format_agent_step() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "research_agent_step",
            "strict": False,
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning_brief": {
                        "type": "string",
                        "maxLength": 280,
                        "description": "1 short sentence: what you're testing and why.",
                    },
                    "hypothesis": {
                        "type": "string",
                        "minLength": 8,
                        "maxLength": 180,
                        "description": "Current falsifiable hypothesis (one sentence).",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "belief_updates": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "properties": {
                                "statement": {"type": "string", "minLength": 4, "maxLength": 160},
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "action": {
                                    "type": "string",
                                    "enum": ["add", "reinforce", "weaken", "remove"],
                                },
                            },
                            "required": ["statement", "action"],
                        },
                    },
                    "tool_calls": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {
                                    "type": "string",
                                    "enum": sorted(ALLOWED_TOOLS),
                                },
                                "args": {"type": "object"},
                            },
                            "required": ["tool", "args"],
                        },
                    },
                },
                "required": ["hypothesis", "tool_calls"],
            },
        },
    }


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def ask_agent(
    user_content: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    key = os.environ.get("POLZA_AI_API_KEY")
    if not key:
        msg = "Задайте POLZA_AI_API_KEY в окружении или в .env"
        raise RuntimeError(msg)

    to = timeout_sec if timeout_sec is not None else float(
        os.environ.get("LLM_TIMEOUT_SEC", str(DEFAULT_LLM_TIMEOUT_SEC))
    )
    client = OpenAI(base_url=POLZA_BASE_URL, api_key=key, timeout=to)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        response_format=_response_format_agent_step(),  # type: ignore[arg-type]
    )
    raw = response.choices[0].message.content
    if raw is None or not raw.strip():
        msg = "Пустой ответ LLM"
        raise RuntimeError(msg)
    cleaned = _strip_json_fence(raw)
    out: object = json.loads(cleaned)
    if not isinstance(out, dict):
        msg = f"Ожидался JSON-объект, получено: {type(out).__name__}"
        raise TypeError(msg)
    if "tool_calls" not in out or not isinstance(out["tool_calls"], list):
        msg = "В ответе модели нет массива tool_calls"
        raise ValueError(msg)
    return out
