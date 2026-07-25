---
name: project-nan-json-leak-80-1
description: phase-80.1 NaN/JSON research — Starlette allow_nan=False is unavoidable, FastAPI ignores pydantic ser_json_inf_nan, yfinance keepna cannot drop the forming-session row, and the signals API shares its two leak-site tools with the live trading pipeline
metadata:
  type: project
---

Researched 2026-07-25 for masterplan step 80.1 (`GET /api/signals/{ticker}` 500s on every ticker).

**Four non-obvious facts that cost real fetch/measure effort to establish:**

1. **`allow_nan=False` is hardcoded** in `starlette/responses.py:194-201` (starlette 1.0.0) and
   `fastapi.encoders.jsonable_encoder` does NOT sanitise — `np.float64` IS a Python `float`
   subclass, so it takes the encoder's scalar fast-path untouched. Measured.
2. **A pydantic `response_model` does NOT fix it.** Pydantic's `ser_json_inf_nan` default is
   already `'null'` (and `TypeAdapter(dict).dump_json({"a": nan})` → `b'{"a":null}'`), but
   FastAPI's pipeline is `serialize_response → jsonable_encoder → JSONResponse.render`, which
   never consults the model config — fastapi/fastapi discussion **#11821**. Don't propose it again.
3. **yfinance has no flag for the forming-session placeholder row.** `keepna=False` is already
   the default; its mask at `yfinance/scrapers/history.py:495-499` is
   `(isna | == 0).all(axis=1)` — a row survives if ANY data column is real, and the placeholder
   row has a real non-zero `Volume`. So post-hoc `dropna(subset=["Close"])` is the only
   mechanism, not a workaround for a flag someone forgot.
4. **`backend/tools/sector_analysis.py` and `backend/tools/quant_model.py` are SHARED** between
   the display API (`backend/api/signals.py:94,:98`) and the Layer-1 trading pipeline
   (`backend/agents/orchestrator.py:1261`, `:1271-1273`). A fix inside `backend/tools/` changes
   trading inputs; a fix inside `backend/api/` cannot. This is the 80.1-vs-80.27 boundary.

**Why:** the caller specifically asked whether a "just fix the 500" change could move the live
paper-trading book, and the answer turned entirely on (4). NaN also launders into a confident
`NEUTRAL` because every NaN comparison is False (`quant_model._classify_signal` falls through;
`sector_analysis:136-153` leaves both booleans False), and `prompts.py:1111` uses stdlib
`json.dumps` with default `allow_nan=True`, so the Gemini agent literally receives `NaN` tokens.

**How to apply:** for any future "API returns 500 on a float" step, check the fix LOCATION
against the shared-consumer table before recommending anything, and don't re-derive facts 1-3.
Full brief: `handoff/current/research_brief_80.1.md` (archived to `handoff/archive/phase-80.1/`).
Related: [[project_metric_source_paths]], [[feedback_measure_dont_assert_claims]].
