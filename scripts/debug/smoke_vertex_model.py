"""phase-60.1 (AW-4): one-shot Vertex model availability smoke.

Proves a Gemini model id is CURRENTLY SERVED on the Vertex publisher
endpoint -- the immutable 60.1 criterion requires "availability proven by
a live smoke call exiting 0, not by docs alone". The away week's root
cause was a pin (`gemini-2.0-flash`) whose family was discontinued
server-side on 2026-06-01 while every config still referenced it.

Two legs per model, both must pass:
  1. plain text generation (serving exists)
  2. structured output via JSON-schema enforcement (the Layer-1 pipeline
     depends on Vertex controlled generation; a model that serves text
     but rejects response_schema is NOT a valid pin)

Usage:
  source .venv/bin/activate
  python scripts/debug/smoke_vertex_model.py gemini-2.5-flash
  python scripts/debug/smoke_vertex_model.py gemini-3.1-flash-lite gemini-2.5-flash

Exit 0 iff EVERY model given on argv passes BOTH legs. Cost: pennies
(two ~20-token calls per model).
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")  # run from repo root

from backend.agents._genai_client import get_genai_client  # noqa: E402


def smoke_one(client, model_id: str) -> bool:
    from google.genai import types  # local import; SDK proven present by client

    ok = True

    # Leg 1: plain generation
    try:
        resp = client.models.generate_content(
            model=model_id,
            contents="Reply with the single word OK.",
            config=types.GenerateContentConfig(max_output_tokens=2000),
        )
        text = (resp.text or "").strip()
        usage = getattr(resp, "usage_metadata", None)
        print(f"[{model_id}] leg1 text_generation: PASS text={text!r} "
              f"tokens={getattr(usage, 'total_token_count', '?')}")
    except Exception as exc:
        print(f"[{model_id}] leg1 text_generation: FAIL {type(exc).__name__}: {exc}")
        ok = False

    # Leg 2: structured output (JSON schema enforcement)
    try:
        schema = {
            "type": "OBJECT",
            "properties": {
                "status": {"type": "STRING"},
                "score": {"type": "NUMBER"},
            },
            "required": ["status", "score"],
        }
        resp = client.models.generate_content(
            model=model_id,
            contents="Return status='ok' and score=1.",
            config=types.GenerateContentConfig(
                max_output_tokens=2000,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        import json

        parsed = json.loads(resp.text or "{}")
        assert "status" in parsed and "score" in parsed, f"schema keys missing: {parsed}"
        print(f"[{model_id}] leg2 structured_output: PASS parsed={parsed}")
    except Exception as exc:
        print(f"[{model_id}] leg2 structured_output: FAIL {type(exc).__name__}: {exc}")
        ok = False

    return ok


def main() -> int:
    models = sys.argv[1:]
    if not models:
        print("usage: python scripts/debug/smoke_vertex_model.py <model-id> [<model-id> ...]")
        return 2

    client = get_genai_client()
    if client is None:
        print("FAIL: get_genai_client() returned None (SDK absent or bad creds)")
        return 1

    results = {m: smoke_one(client, m) for m in models}
    print("---")
    for m, passed in results.items():
        print(f"{m}: {'PASS' if passed else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
